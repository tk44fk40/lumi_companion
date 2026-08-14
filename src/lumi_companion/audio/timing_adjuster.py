"""字幕セグメントの表示タイミング補正モジュール。

本モジュールは、Whisper が検出した物理的な発声終了時刻に対し、
視聴者の読みやすさを向上させるための余韻パディング、最小表示時間の確保、
および次発話セグメントとの重複防止（ギャップ制御）を行います。
"""

import logging
from typing import Protocol, runtime_checkable

from lumi_companion.models.audio import SubtitleSegment

logger = logging.getLogger(__name__)


@runtime_checkable
class TimingAdjusterProtocol(Protocol):
    """字幕タイミング補正処理の抽象インターフェース (Strategy)。"""

    def adjust_segments(
        self,
        segments: list[SubtitleSegment],
        total_duration: float | None = None,
    ) -> list[SubtitleSegment]:
        """字幕セグメントリストの表示タイミングを補正します。

        Args:
            segments (list[SubtitleSegment]): 補正対象の字幕セグメントリスト。
            total_duration (float | None, optional): メディアの総再生時間（秒）。

        Returns:
            list[SubtitleSegment]: タイミング補正済みの字幕セグメントリスト。
        """
        ...


class SubtitleTimingAdjuster:
    """字幕セグメントの表示タイミング（余韻・最小表示時間・重複防止）を補正するクラス。"""

    def __init__(
        self,
        end_padding: float,
        min_duration: float,
        min_gap: float,
    ) -> None:
        """SubtitleTimingAdjuster を初期化します。

        Args:
            end_padding (float): 発話終了後の余韻表示秒数。
            min_duration (float): 字幕の最小表示秒数。
            min_gap (float): 連続するセグメント間の最小隙間秒数。
        """
        self.end_padding = end_padding
        self.min_duration = min_duration
        self.min_gap = min_gap

    def adjust_segments(
        self,
        segments: list[SubtitleSegment],
        total_duration: float | None = None,
    ) -> list[SubtitleSegment]:
        """字幕セグメントリストの終了時刻を自然な表示時間へ補正します。

        各セグメントに対して以下の補正を順に適用します：
        1. 余韻パディングの追加: 発話終了直後に字幕が消えるのを防ぐため、終了時刻を延長します。
        2. 最小表示時間の確保: 短い発言でも視聴者が読めるよう、最低表示秒数を下限とします。
        3. 次セグメントとの重複防止: 余韻延長によって次セグメントの表示と被らないよう、
           次の開始時刻 - min_gap でクリップします。
        4. 総再生時間の制限: total_duration が指定されている場合、動画全体の長さを超えないよう制限します。

        Args:
            segments (list[SubtitleSegment]): 補正対象の字幕セグメントリスト。
            total_duration (float | None, optional): メディアの総再生時間（秒）。

        Returns:
            list[SubtitleSegment]: タイミング補正済みの新しい字幕セグメントリスト。
        """
        if not segments:
            return []

        adjusted_segments: list[SubtitleSegment] = []
        count = len(segments)

        for i, seg in enumerate(segments):
            # 目的: 自然な読書体験のために余韻を追加し、かつ最低表示時間を確保する
            padded_end = seg.end + self.end_padding
            min_required_end = seg.start + self.min_duration
            target_end = max(padded_end, min_required_end)

            # 目的: 次の発話開始時刻と重なることを防止する
            if i + 1 < count:
                next_start = segments[i + 1].start
                # 次の開始時刻から指定ギャップを引いた時刻を上限とする
                max_allowed_end = next_start - self.min_gap
                # 開始時刻より前にならないよう保護しつつクリップ
                if max_allowed_end > seg.start:
                    target_end = min(target_end, max_allowed_end)
                else:
                    # 次のセグメントとほぼ隙間がない場合は元の end を超えない範囲で維持
                    target_end = min(target_end, next_start)

            # 目的: 動画全体の長さを超えて表示され続けることを防ぐ
            if total_duration is not None and total_duration > 0:
                target_end = min(target_end, total_duration)

            # 開始時刻以上の終了時刻を保証
            final_end = max(seg.start, target_end)

            adjusted_segments.append(
                SubtitleSegment(
                    start=round(seg.start, 3),
                    end=round(final_end, 3),
                    text=seg.text,
                )
            )

        logger.debug(
            "字幕タイミング補正完了: %d 件のセグメントを処理 (padding=%.2fs, min_duration=%.2fs, gap=%.2fs)",
            len(adjusted_segments),
            self.end_padding,
            self.min_duration,
            self.min_gap,
        )

        return adjusted_segments
