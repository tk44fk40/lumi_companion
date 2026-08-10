"""音声セグメントのインテリジェント分割処理モジュール。

Janome を用いた形態素解析による文節単位での分割と、
フォールバックとしての句読点・文字数ベースの分割を提供します。
"""

import logging
import re
from typing import Any

from lumi_companion.models.audio import SubtitleSegment

logger = logging.getLogger(__name__)


class SegmentSplitter:
    """文字起こしセグメントを自然な文節・文字数で分割するクラス。"""

    def __init__(self, max_segment_chars: int = 25) -> None:
        """SegmentSplitter を初期化します。

        Args:
            max_segment_chars (int): 1セグメントの許容最大文字数。デフォルト 25。
        """
        self.max_segment_chars = max_segment_chars
        self._tokenizer: Any = None

    def _get_tokenizer(self) -> Any:
        """Janome Tokenizer インスタンスを遅延ロードで取得します。"""
        if self._tokenizer is None:
            try:
                from janome.tokenizer import Tokenizer

                logger.info("Janome Tokenizer をロード中...")
                self._tokenizer = Tokenizer()
            except ImportError:
                logger.warning(
                    "janome がインストールされていません。意味分割がスキップされます。"
                )
                self._tokenizer = False
        return self._tokenizer

    def split_segment_intelligently(
        self, segment: SubtitleSegment
    ) -> list[SubtitleSegment]:
        """形態素解析(Janome)と文字数・句読点を用いた高度な分割を行います。

        タイムスタンプは文字数の比率によって按分計算します。

        Args:
            segment (SubtitleSegment): 分割対象の字幕セグメント。

        Returns:
            list[SubtitleSegment]: 分割セグメントリスト。
        """
        max_chars = self.max_segment_chars

        if len(segment.text) <= max_chars or max_chars <= 0:
            return [segment]

        tokenizer = self._get_tokenizer()
        if not tokenizer:
            return self.split_segment_fallback(segment)

        # 1. 形態素解析で絶対に切断しない「文節の塊 (チャンク)」を作る
        tokens = tokenizer.tokenize(segment.text)

        bunsetsu_chunks: list[list[str]] = []
        current_bunsetsu: list[str] = []
        current_bunsetsu_pos: list[str] = []

        for token in tokens:
            surface = token.surface
            pos_parts = token.part_of_speech.split(",")
            pos1 = pos_parts[0]
            pos2 = pos_parts[1] if len(pos_parts) > 1 else ""

            is_prev_prefix = (
                current_bunsetsu_pos and current_bunsetsu_pos[-1] == "接頭詞"
            )

            # 付属語や記号、話し言葉は結合対象とする。
            # 接頭詞自体は新しい文節を作るが、その次の単語は必ず結合する (is_prev_prefix)
            is_dependent = (
                pos1 in ("助詞", "助動詞", "記号", "フィラー")
                or pos2 in ("非自立", "接尾")
                or is_prev_prefix
                or surface
                in (
                    "なん",
                    "す",
                    "じゃん",
                    "って",
                    "ん",
                    "てる",
                    "とく",
                    "ちゃう",
                    "じゃう",
                )
            )

            if current_bunsetsu and not is_dependent:
                bunsetsu_chunks.append(current_bunsetsu)
                current_bunsetsu = [surface]
                current_bunsetsu_pos = [pos1]
            else:
                current_bunsetsu.append(surface)
                current_bunsetsu_pos.append(pos1)

        if current_bunsetsu:
            bunsetsu_chunks.append(current_bunsetsu)

        # 2. 文節単位で max_chars に収まるようにテキストを分割する (ソフトリミット導入)
        split_bunsetsu_chunks: list[list[str]] = []
        current_chunk: list[str] = []
        current_chars = 0

        soft_limit = int(max_chars * 0.7)

        for b_chunk in bunsetsu_chunks:
            b_text = "".join(b_chunk)
            b_len = len(b_text)

            has_punctuation = any(
                p in b_text for p in ("。", "、", "！", "？", "!", "?", " ", "　")
            )

            if current_chunk and current_chars + b_len > max_chars:
                split_bunsetsu_chunks.append(current_chunk)
                current_chunk = [b_text]
                current_chars = b_len
            else:
                current_chunk.append(b_text)
                current_chars += b_len
                # ソフトリミットを超過し、かつ句読点が含まれていればここで早期分割
                if current_chars >= soft_limit and has_punctuation:
                    split_bunsetsu_chunks.append(current_chunk)
                    current_chunk = []
                    current_chars = 0

        if current_chunk:
            split_bunsetsu_chunks.append(current_chunk)

        # 3. 分割された各テキストに対応するタイムスタンプを文字数比率で按分計算する
        results: list[SubtitleSegment] = []
        total_chars = sum(len("".join(c)) for c in split_bunsetsu_chunks)
        duration = max(segment.end - segment.start, 0.1)
        current_time = segment.start

        log_lines = [
            f"文字数上限({max_chars}文字)のため文節・句読点境界で分割しました:"
        ]

        for idx, bunsetsu_list in enumerate(split_bunsetsu_chunks):
            stext = "".join(bunsetsu_list)
            stext_with_bar = " | ".join(bunsetsu_list)

            char_ratio = len(stext) / total_chars if total_chars > 0 else 1.0
            seg_duration = duration * char_ratio

            seg_start = current_time
            seg_end = (
                current_time + seg_duration
                if idx < len(split_bunsetsu_chunks) - 1
                else segment.end
            )

            new_seg = SubtitleSegment(
                start=round(seg_start, 3),
                end=round(seg_end, 3),
                text=stext.strip(),
            )
            results.append(new_seg)
            log_lines.append(
                f"  [{idx + 1}] '{stext_with_bar}' ({new_seg.start:.2f} -> {new_seg.end:.2f})"
            )

            current_time = seg_end

        if len(results) > 1:
            logger.info("\n".join(log_lines))

        return results

    def split_segment_fallback(self, segment: SubtitleSegment) -> list[SubtitleSegment]:
        """(フォールバック用) 文字数と句読点ベースで分割します。"""
        text = segment.text
        max_chars = self.max_segment_chars

        if len(text) <= max_chars or max_chars <= 0:
            return [segment]

        # 句読点・記号（。、！？!?\s）の直後で分割を試みる
        raw_chunks = [c for c in re.split(r"(?<=[。、！？!?\s])", text) if c]

        sub_texts: list[str] = []
        current_chunk_str = ""

        for chunk in raw_chunks:
            if not current_chunk_str:
                current_chunk_str = chunk
            elif len(current_chunk_str) + len(chunk) <= max_chars:
                current_chunk_str += chunk
            else:
                sub_texts.append(current_chunk_str)
                current_chunk_str = chunk
        if current_chunk_str:
            sub_texts.append(current_chunk_str)

        # 万が一句読点なしで max_chars を超過しているチャンクを文字数でカット
        final_texts: list[str] = []
        for st in sub_texts:
            if len(st) <= max_chars:
                final_texts.append(st)
            else:
                for i in range(0, len(st), max_chars):
                    final_texts.append(st[i : i + max_chars])

        total_len = sum(len(t) for t in final_texts)
        if total_len == 0:
            return [segment]

        duration = max(segment.end - segment.start, 0.1)
        result: list[SubtitleSegment] = []
        current_time = segment.start

        for idx, t in enumerate(final_texts):
            char_ratio = len(t) / total_len
            chunk_duration = duration * char_ratio
            seg_start = current_time
            seg_end = (
                current_time + chunk_duration
                if idx < len(final_texts) - 1
                else segment.end
            )

            trimmed_text = t.strip()
            if trimmed_text:
                result.append(
                    SubtitleSegment(
                        start=round(seg_start, 3),
                        end=round(seg_end, 3),
                        text=trimmed_text,
                    )
                )
            current_time = seg_end

        return result if result else [segment]
