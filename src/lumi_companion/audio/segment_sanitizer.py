"""音声セグメントのハルシネーション検出およびサニタイズモジュール。

Whisper による無音捏造や異常な発話速度のセグメントを検出し、
除外・正規化を行います。
"""

import logging
from collections.abc import Iterable

from lumi_companion.models.audio import SubtitleSegment

logger = logging.getLogger(__name__)


class SegmentSanitizer:
    """Whisper 認識結果のハルシネーション検出およびフィルタリングを行うクラス。"""

    def __init__(
        self,
        no_speech_threshold: float = 0.6,
        max_chars_per_second: float = 12.0,
    ) -> None:
        """SegmentSanitizer を初期化します。

        Args:
            no_speech_threshold (float): 無音判定閾値。デフォルト 0.6。
            max_chars_per_second (float): 物理的発話速度の許容上限（文字/秒）。デフォルト 12.0。
        """
        self.no_speech_threshold = no_speech_threshold
        self.max_chars_per_second = max_chars_per_second

    @staticmethod
    def get_word_time(word_obj: object, attr_name: str) -> float | None:
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

    def sanitize_segments(
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
                first_word_start = self.get_word_time(words[0], "start")
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
