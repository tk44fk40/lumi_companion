"""Step 5 タイムライン検証スクリプトのテストモジュール。"""

from lumi_companion.models.audio import SubtitleSegment
from scripts.debug_step5_timeline import (
    filter_subtitles_by_timestamp,
    generate_timestamps,
)


def test_filter_subtitles_by_timestamp():
    """タイムスタンプによる字幕フィルタリングが正しく機能することを確認します。"""
    segments = [
        SubtitleSegment(start=1.0, end=3.0, text="A"),
        SubtitleSegment(start=4.0, end=6.0, text="B"),
        SubtitleSegment(start=7.0, end=9.0, text="C"),
    ]

    # 4.0秒の時点では A と B が含まれる (start <= 4.0)
    filtered = filter_subtitles_by_timestamp(segments, 4.0)
    assert len(filtered) == 2
    assert filtered[0].text == "A"
    assert filtered[1].text == "B"

    # 0.0秒の時点では何も含まれない
    filtered_0 = filter_subtitles_by_timestamp(segments, 0.0)
    assert len(filtered_0) == 0

    # 10.0秒の時点ではすべて含まれる
    filtered_10 = filter_subtitles_by_timestamp(segments, 10.0)
    assert len(filtered_10) == 3


def test_generate_timestamps():
    """検証用のタイムスタンプリスト生成が正しく機能することを確認します。"""
    # 400秒の動画、180秒間隔
    timestamps = generate_timestamps(duration_seconds=400.0, interval_seconds=180.0)
    assert timestamps == [0.0, 180.0, 360.0, 400.0]

    # 180秒ちょうどの動画
    timestamps_exact = generate_timestamps(
        duration_seconds=180.0, interval_seconds=180.0
    )
    assert timestamps_exact == [0.0, 180.0]

    # 動画の長さが間隔より短い場合
    timestamps_short = generate_timestamps(
        duration_seconds=100.0, interval_seconds=180.0
    )
    assert timestamps_short == [0.0, 100.0]
