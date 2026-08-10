"""音声認識・発言抽出プロセッサモジュール。

本モジュールは、Faster-Whisper (WhisperModel) と Silero VAD フィルタを組み合わせて
動画・音声トラックから発言区間を自動検出し、高精度な文字起こしを行います。
設定モジュール (config) に直接依存せず、依存性注入 (DI) を前提とした疎結合設計となっています。
"""

import asyncio
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from lumi_companion.audio.post_processor import TextPostProcessor
from lumi_companion.models.audio import SubtitleSegment

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_PROMPT: str = (
    "日本語のゲーム実況・雑談配信です。話し言葉や感嘆詞を含めて正確に文字起こしします。"
)


class AudioProcessor:
    """動画・音声ファイルから発言抽出を行う処理クラス（設定に非依存）。"""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "ja",
        beam_size: int = 5,
        initial_prompt: str = DEFAULT_INITIAL_PROMPT,
        condition_on_previous_text: bool = False,
        vad_filter: bool = True,
        vad_threshold: float = 0.35,
        vad_min_silence_duration_ms: int = 500,
        no_speech_threshold: float = 0.6,
        max_chars_per_second: float = 12.0,
        max_segment_chars: int = 25,
        post_process_to_hankaku: bool = False,
        post_process_normalize_nums: bool = True,
        post_process_lower: bool = False,
        post_process_remove_punct: bool = False,
        custom_dictionary_path: Path | None = None,
        word_timestamps: bool = True,
    ) -> None:
        """AudioProcessor を初期化します。

        Args:
            model_size (str): Whisper モデルサイズ。
            device (str): 実行デバイス (cuda/cpu/auto)。
            compute_type (str): 計算精度 (float16/int8/default/float32)。
            language (str): 認識言語コード。
            beam_size (int): ビームサーチ幅。
            initial_prompt (str): 初期誘導プロンプト。
            condition_on_previous_text (bool): 直前文脈への依存フラグ。
            vad_filter (bool): VAD フィルタ有効化フラグ。
            vad_threshold (float): VAD 検出閾値。
            vad_min_silence_duration_ms (int): VAD 最小無音時間(ms)。
            no_speech_threshold (float): 無音判定閾値。
            max_chars_per_second (float): 物理的発話速度の許容上限（文字/秒）。
            max_segment_chars (int): 1セグメントの許容最大文字数。デフォルト 25。
            post_process_to_hankaku (bool): 全角半角統一等の正規化を行うか。
            post_process_normalize_nums (bool): 数字正規化を行うか。
            post_process_lower (bool): 小文字化を行うか (デフォルト: False)。
            post_process_remove_punct (bool): 句読点・記号の除去を行うか。
            custom_dictionary_path (Path | None): 後処理置換辞書ファイルのパス。
            word_timestamps (bool): 単語レベルタイムスタンプを有効化し発声開始位置を自動補正するか。
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.condition_on_previous_text = condition_on_previous_text
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold
        self.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        self.no_speech_threshold = no_speech_threshold
        self.max_chars_per_second = max_chars_per_second
        self.max_segment_chars = max_segment_chars
        self.post_process_to_hankaku = post_process_to_hankaku
        self.post_process_normalize_nums = post_process_normalize_nums
        self.post_process_lower = post_process_lower
        self.post_process_remove_punct = post_process_remove_punct
        self.custom_dictionary_path = custom_dictionary_path
        self.word_timestamps = word_timestamps
        self._model: WhisperModel | None = None
        self._tokenizer: Any = None

    def _get_model(self) -> WhisperModel:
        """WhisperModel インスタンスを遅延ロードで取得します。

        Returns:
            WhisperModel: ロード済みの WhisperModel オブジェクト。
        """
        if self._model is None:
            logger.info(
                "WhisperModel をロード中 (model=%s, device=%s, compute_type=%s)...",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

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

    @staticmethod
    def _get_word_time(word_obj: object, attr_name: str) -> float | None:
        """単語オブジェクト(dict または Word オブジェクト)から指定された時刻属性を取得します。"""
        if isinstance(word_obj, dict):
            val = word_obj.get(attr_name)
            if isinstance(val, int | float):
                return float(val)
        else:
            val = getattr(word_obj, attr_name, None)
            if isinstance(val, int | float):
                return float(val)
        return None

    def _sanitize_segments(
        self,
        segments: Iterable[object],
        total_duration: float = 0.0,
    ) -> list[SubtitleSegment]:
        """Whisper 認識結果からハルシネーション（無音捏造・異常発話速度・重複）を自動判定し除外・正規化します。

        また、単語レベルのタイムスタンプ(words)が存在する場合、文頭単語の発声開始位置へ補正します。

        Args:
            segments (Iterable[object]): Faster-Whisper から返された Segment オブジェクトのイテラブル。
            total_duration (float, optional): 音声の全再生時間（秒）。進捗出力用。

        Returns:
            list[SubtitleSegment]: フィルタリングおよび発声位置補正済みの字幕セグメントリスト。
        """
        results: list[SubtitleSegment] = []
        last_valid_text = ""

        for segment in segments:
            text = getattr(segment, "text", "").strip()
            if not text:
                continue

            no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
            compression_ratio = getattr(segment, "compression_ratio", 0.0)

            # セグメント内のリピート判定 (完全に2等分できるなら片方にする)
            # 人間の意図的な繰り返し(「もしもし？もしもし？」)との誤爆を防ぐため、
            # ハルシネーションの可能性が高い場合（無音確率や圧縮率が高い）に限定
            half_len = len(text) // 2
            if len(text) >= 4 and text[:half_len] == text[half_len:]:
                if no_speech_prob > 0.1 or compression_ratio > 2.0:
                    logger.info(
                        "ハルシネーションとみられるセグメント内リピートを検出・短縮: '%s' -> '%s' (no_speech_prob=%.2f, comp_ratio=%.2f)",
                        text,
                        text[:half_len],
                        no_speech_prob,
                        compression_ratio,
                    )
                    text = text[:half_len]

            start = getattr(segment, "start", 0.0)
            end = getattr(segment, "end", 0.0)
            words = getattr(segment, "words", None) or []

            # 単語レベルのタイムスタンプが存在する場合、文頭単語の開始時刻を発声開始位置として補正
            if words:
                first_word_start = self._get_word_time(words[0], "start")
                if first_word_start is not None:
                    start = first_word_start

            duration = max(end - start, 0.1)
            chars_per_sec = len(text) / duration

            # ループハルシネーション（セグメント間）の除外
            if last_valid_text and no_speech_prob > 0.1:
                # 完全に一致するか、包含関係にある場合
                if text == last_valid_text or text in last_valid_text:
                    logger.info(
                        "ループ重複を自動ドロップ: 直前='%s', ドロップ対象='%s' (no_speech_prob=%.2f)",
                        last_valid_text,
                        text,
                        no_speech_prob,
                    )
                    continue

            # 無音区間の捏造セグメント判定 (no_speech_prob チェック)
            if no_speech_prob > self.no_speech_threshold:
                logger.debug(
                    "無音捏造セグメントを自動ドロップ: %s (no_speech_prob=%.2f)",
                    text,
                    no_speech_prob,
                )
                continue

            # 人間の解剖学的限界を超える異常発話速度の捏造判定 (4文字超のみ対象)
            if chars_per_sec > self.max_chars_per_second and len(text) > 4:
                logger.debug(
                    "異常発話速度の捏造セグメントを自動ドロップ: %s (%.1f文字/秒)",
                    text,
                    chars_per_sec,
                )
                continue

            clean_seg = SubtitleSegment(
                start=round(start, 3),
                end=round(end, 3),
                text=text,
            )

            results.append(clean_seg)
            last_valid_text = text

            if total_duration > 0:
                progress = min(100.0, (clean_seg.end / total_duration) * 100)
                logger.info(
                    "発言検出 [%5.1fs / %5.1fs (%3.0f%%)] [%.2fs -> %.2fs]: %s",
                    clean_seg.end,
                    total_duration,
                    progress,
                    clean_seg.start,
                    clean_seg.end,
                    clean_seg.text,
                )
            else:
                logger.info(
                    "発言検出 [%.2fs -> %.2fs]: %s",
                    clean_seg.start,
                    clean_seg.end,
                    clean_seg.text,
                )

        return results

    def _split_segment_intelligently(
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
            return self._split_segment_fallback(segment)

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
            # words 属性は不要になるため引き継がない
            results.append(new_seg)
            log_lines.append(
                f"  [{idx + 1}] '{stext_with_bar}' ({new_seg.start:.2f} -> {new_seg.end:.2f})"
            )

            current_time = seg_end

        if len(results) > 1:
            logger.info("\n".join(log_lines))

        return results

    def _split_segment_fallback(
        self, segment: SubtitleSegment
    ) -> list[SubtitleSegment]:
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

    def process_sync(self, file_path: Path | str) -> list[SubtitleSegment]:
        """同期的にファイル全体の音声を解析・文字起こしを実行します。

        Args:
            file_path (Path | str): 解析対象のメディアファイルパス。

        Returns:
            list[SubtitleSegment]: 抽出された字幕セグメントのリスト。

        Raises:
            FileNotFoundError: 対象ファイルが存在しない場合。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"メディアファイルが見つかりません: {path}")

        model = self._get_model()
        logger.info("発言抽出 (Whisper transcription) を開始: %s", path)

        raw_segments, info = model.transcribe(
            str(path),
            beam_size=self.beam_size,
            language=self.language,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=self.condition_on_previous_text,
            vad_filter=self.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": self.vad_min_silence_duration_ms,
                "threshold": self.vad_threshold,
            },
            no_speech_threshold=self.no_speech_threshold,
            word_timestamps=self.word_timestamps,
        )

        total_duration = getattr(info, "duration", 0.0)
        detected_language = getattr(info, "language", self.language)
        logger.info(
            "音声トラック解析中... (検出言語=%s, 総再生時間=%.1fs)",
            detected_language,
            total_duration,
        )

        # 1. 無音除外 & 発声位置補正 & 重複除去
        sanitized_segments = self._sanitize_segments(
            raw_segments, total_duration=total_duration
        )

        # 2. 置換辞書・正規化後処理の適用 (分割前に綺麗な日本語にする)
        post_processor = TextPostProcessor(
            dictionary_path=self.custom_dictionary_path,
            to_hankaku=self.post_process_to_hankaku,
            normalize_nums=self.post_process_normalize_nums,
            lower=self.post_process_lower,
            remove_punct=self.post_process_remove_punct,
        )
        normalized_segments = post_processor.apply_to_segments(sanitized_segments)

        # 3. 文節境界でのインテリジェント分割
        final_segments: list[SubtitleSegment] = []
        for seg in normalized_segments:
            final_segments.extend(self._split_segment_intelligently(seg))

        logger.info("発言抽出完了 (%d 件のセグメントを検出)", len(final_segments))
        return final_segments

    async def process_async(self, file_path: Path | str) -> list[SubtitleSegment]:
        """イベントループをブロックせずに非同期に文字起こしを実行します。

        Args:
            file_path (Path | str): 解析対象のメディアファイルパス。

        Returns:
            list[SubtitleSegment]: 抽出された字幕セグメントのリスト。
        """
        return await asyncio.to_thread(self.process_sync, file_path)
