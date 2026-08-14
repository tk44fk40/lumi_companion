"""SubtitleTimingAdjuster の単体テストモジュール。

AAA (Arrange-Act-Assert) パターンに基づき、余韻パディング・最小表示時間・
次セグメントとの重複防止クリップ・総再生時間クリップなどの各機能を網羅的に検証します。
"""

import pytest

from lumi_companion.audio.timing_adjuster import (
    SubtitleTimingAdjuster,
    TimingAdjusterProtocol,
)
from lumi_companion.models.audio import SubtitleSegment


class TestSubtitleTimingAdjuster:
    """SubtitleTimingAdjuster の振る舞いを検証するテストクラス。"""

    def test_implements_protocol(self) -> None:
        """SubtitleTimingAdjuster が TimingAdjusterProtocol を満たしていることを検証します。"""
        # Arrange & Act
        adjuster = SubtitleTimingAdjuster(
            end_padding=0.8,
            min_duration=1.2,
            min_gap=0.05,
        )

        # Assert
        assert isinstance(adjuster, TimingAdjusterProtocol)

    def test_single_segment_padding_applied(self) -> None:
        """単一セグメントに対して余韻パディング (+0.8s) が正常に付与されることを検証します。"""
        # Arrange
        adjuster = SubtitleTimingAdjuster(
            end_padding=0.8,
            min_duration=1.2,
            min_gap=0.05,
        )
        segments = [
            SubtitleSegment(start=1.0, end=3.0, text="こんにちは"),
        ]

        # Act
        results = adjuster.adjust_segments(segments)

        # Assert
        assert len(results) == 1
        assert results[0].start == 1.0
        # 3.0 + 0.8 = 3.8秒
        assert results[0].end == pytest.approx(3.8)
        assert results[0].text == "こんにちは"

    def test_short_segment_min_duration_ensured(self) -> None:
        """短い発話（0.2秒）に対して最小表示時間（1.2秒）が保証されることを検証します。"""
        # Arrange
        adjuster = SubtitleTimingAdjuster(
            end_padding=0.8,
            min_duration=1.2,
            min_gap=0.05,
        )
        # 0.2秒の発話 (余韻を足しても 0.2 + 0.8 = 1.0s < min_duration 1.2s)
        segments = [
            SubtitleSegment(start=1.0, end=1.2, text="はい"),
        ]

        # Act
        results = adjuster.adjust_segments(segments)

        # Assert
        assert len(results) == 1
        assert results[0].start == 1.0
        # start(1.0) + min_duration(1.2) = 2.2秒
        assert results[0].end == pytest.approx(2.2)

    def test_consecutive_segments_gap_prevented(self) -> None:
        """連続するセグメント間で、次セグメントの開始時刻を超えないようクリップされることを検証します。"""
        # Arrange
        adjuster = SubtitleTimingAdjuster(
            end_padding=0.8,
            min_duration=1.2,
            min_gap=0.05,
        )
        segments = [
            SubtitleSegment(start=1.0, end=2.0, text="最初の一言"),
            SubtitleSegment(start=2.5, end=4.0, text="次の一言"),
        ]

        # Act
        results = adjuster.adjust_segments(segments)

        # Assert
        assert len(results) == 2
        # 1つ目: 2.0 + 0.8 = 2.8 だが、次の開始(2.5) - gap(0.05) = 2.45 でクリップ
        assert results[0].start == 1.0
        assert results[0].end == pytest.approx(2.45)
        assert results[1].start == 2.5
        # 2つ目: 4.0 + 0.8 = 4.8
        assert results[1].end == pytest.approx(4.8)

    def test_dense_segments_no_inversion(self) -> None:
        """隙間が極小（0.02秒）の場合でも、開始時刻を追い越さず矛盾が生じないことを検証します。"""
        # Arrange
        adjuster = SubtitleTimingAdjuster(
            end_padding=0.8,
            min_duration=1.2,
            min_gap=0.05,
        )
        segments = [
            SubtitleSegment(start=1.0, end=1.98, text="詰まった発話1"),
            SubtitleSegment(start=2.0, end=3.0, text="詰まった発話2"),
        ]

        # Act
        results = adjuster.adjust_segments(segments)

        # Assert
        assert len(results) == 2
        # 1つ目: 次の開始 2.0 - 0.05 = 1.95 だが元の end 1.98 より小さくならず start 以上を維持
        assert results[0].start <= results[0].end <= 2.0

    def test_total_duration_clamping(self) -> None:
        """メディアの総再生時間 (total_duration) を超えないようにクリップされることを検証します。"""
        # Arrange
        adjuster = SubtitleTimingAdjuster(
            end_padding=0.8,
            min_duration=1.2,
            min_gap=0.05,
        )
        segments = [
            SubtitleSegment(start=8.0, end=9.5, text="最後の発言"),
        ]

        # Act
        results = adjuster.adjust_segments(segments, total_duration=10.0)

        # Assert
        assert len(results) == 1
        # 9.5 + 0.8 = 10.3 だが total_duration 10.0 でクリップ
        assert results[0].end == pytest.approx(10.0)

    def test_empty_segments(self) -> None:
        """セグメントリストが空の場合に空リストを返すことを検証します。"""
        # Arrange
        adjuster = SubtitleTimingAdjuster(
            end_padding=0.8,
            min_duration=1.2,
            min_gap=0.05,
        )

        # Act
        results = adjuster.adjust_segments([])

        # Assert
        assert results == []

    def test_gap_smaller_than_min_gap(self) -> None:
        """次の開始時刻との差が min_gap 以下の極小セグメントでも正常にクリップされることを検証します。"""
        # Arrange
        adjuster = SubtitleTimingAdjuster(
            end_padding=0.8,
            min_duration=1.2,
            min_gap=0.05,
        )
        segments = [
            SubtitleSegment(start=1.0, end=1.02, text="短い1"),
            SubtitleSegment(start=1.03, end=2.0, text="短い2"),
        ]

        # Act
        results = adjuster.adjust_segments(segments)

        # Assert
        assert len(results) == 2
        assert results[0].start == 1.0
        assert results[0].end == pytest.approx(1.03)
