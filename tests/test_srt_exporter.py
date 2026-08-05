"""SRT 記述処理およびタイムスタンプ変換単体テストモジュール。

本モジュールは、SRTExporter によるタイムスタンプフォーマットおよび SRT 形式テキスト出力を検証します。
"""

from pathlib import Path

from lumi_companion.audio.srt_exporter import SRTExporter, SubtitleSegment


def test_srt_exporter_format_timestamp() -> None:
    """タイムスタンプ秒数が SRT 形式 (HH:MM:SS,mmm) に正しく変換されることを検証します。"""
    # Arrange & Act (準備・実行)
    formatted = SRTExporter.format_timestamp(3661.5)

    # Assert (検証)
    assert formatted == "01:01:01,500"


def test_srt_exporter_save_srt(tmp_path: Path) -> None:
    """字幕セグメントから正しく .srt ファイルが生成されることを検証します。"""
    # Arrange (準備)
    segments = [
        SubtitleSegment(start=1.0, end=3.5, text="こんにちは"),
        SubtitleSegment(start=4.0, end=6.2, text="さようなら"),
    ]
    out_file = tmp_path / "subtitles.srt"

    # Act (実行)
    SRTExporter.save_srt(segments, out_file)

    # Assert (検証)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:03,500" in content
    assert "こんにちは" in content
    assert "00:00:04,000 --> 00:00:06,200" in content
    assert "さようなら" in content
