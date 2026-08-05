"""字幕記述処理およびタイムスタンプ変換単体テストモジュール。

本モジュールは、SubtitleExporter (旧 SRTExporter) によるタイムスタンプフォーマットおよび SRT / WebVTT 形式テキスト出力を検証します。
"""

from pathlib import Path

import pytest

from lumi_companion.audio.srt_exporter import (
    SubtitleExporter,
    SubtitleSegment,
)


def test_srt_exporter_format_timestamp() -> None:
    """タイムスタンプ秒数が SRT 形式 (HH:MM:SS,mmm) および WebVTT 形式 (HH:MM:SS.mmm) に正しく変換されることを検証します。"""
    # Arrange & Act
    formatted_srt = SubtitleExporter.format_timestamp(3661.5)
    formatted_vtt = SubtitleExporter.format_vtt_timestamp(3661.5)

    # Assert
    assert formatted_srt == "01:01:01,500"
    assert formatted_vtt == "01:01:01.500"


def test_srt_exporter_save_srt(tmp_path: Path) -> None:
    """字幕セグメントから正しく .srt ファイルが生成されることを検証します。"""
    # Arrange
    segments = [
        SubtitleSegment(start=1.0, end=3.5, text="こんにちは"),
        SubtitleSegment(start=4.0, end=6.2, text="さようなら"),
    ]
    out_file = tmp_path / "subtitles.srt"

    # Act
    SubtitleExporter.save_srt(segments, out_file)

    # Assert
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:03,500" in content
    assert "こんにちは" in content
    assert "00:00:04,000 --> 00:00:06,200" in content
    assert "さようなら" in content


def test_subtitle_exporter_save_vtt(tmp_path: Path) -> None:
    """字幕セグメントから正しく .vtt (WEBVTT) ファイルが生成されることを検証します。"""
    # Arrange
    segments = [
        SubtitleSegment(start=1.0, end=3.5, text="こんにちは"),
        SubtitleSegment(start=4.0, end=6.2, text="さようなら"),
    ]
    out_file = tmp_path / "subtitles.vtt"

    # Act
    SubtitleExporter.save_vtt(segments, out_file)

    # Assert
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT\n")
    assert "00:00:01.000 --> 00:00:03.500" in content
    assert "こんにちは" in content
    assert "00:00:04.000 --> 00:00:06.200" in content
    assert "さようなら" in content


def test_subtitle_exporter_save_json(tmp_path: Path) -> None:
    """字幕セグメントから正しく .json ファイルが生成されることを検証します。"""
    segments = [SubtitleSegment(start=1.0, end=3.5, text="テスト")]
    out_file = tmp_path / "subtitles.json"

    SubtitleExporter.save_json(segments, out_file)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert '"text": "テスト"' in content
    assert '"start": 1.0' in content


def test_subtitle_exporter_save_subtitles_auto_detect(tmp_path: Path) -> None:
    """save_subtitles メソッドで拡張子（.vtt, .srt, .json, 大文字 .VTT）に応じた自動判定出力が正しく行われることを検証します。"""
    segments = [SubtitleSegment(start=1.0, end=2.0, text="テスト")]

    # Act 1: .vtt
    vtt_file = tmp_path / "output.vtt"
    SubtitleExporter.save_subtitles(segments, vtt_file)
    assert "WEBVTT" in vtt_file.read_text(encoding="utf-8")

    # Act 2: .srt
    srt_file = tmp_path / "output.srt"
    SubtitleExporter.save_subtitles(segments, srt_file)
    assert "00:00:01,000 --> 00:00:02,000" in srt_file.read_text(encoding="utf-8")

    # Act 3: .json
    json_file = tmp_path / "output.json"
    SubtitleExporter.save_subtitles(segments, json_file)
    assert '"text": "テスト"' in json_file.read_text(encoding="utf-8")

    # Act 4: 大文字拡張子 (.VTT)
    vtt_upper_file = tmp_path / "output.VTT"
    SubtitleExporter.save_subtitles(segments, vtt_upper_file)
    assert "WEBVTT" in vtt_upper_file.read_text(encoding="utf-8")

    # Act 5: Unsupported extension
    invalid_file = tmp_path / "output.txt"
    with pytest.raises(ValueError, match="未対応の拡張子です"):
        SubtitleExporter.save_subtitles(segments, invalid_file)


def test_subtitle_exporter_save_subtitles_explicit_fmt(tmp_path: Path) -> None:
    """save_subtitles メソッドで明示的に fmt (vtt, srt, json, 大文字 SRT) を指定した場合の動作を検証します。"""
    segments = [SubtitleSegment(start=1.0, end=2.0, text="テスト")]

    # Act 1: vtt
    vtt_file = tmp_path / "output.custom_vtt"
    SubtitleExporter.save_subtitles(segments, vtt_file, fmt="vtt")
    assert "WEBVTT" in vtt_file.read_text(encoding="utf-8")

    # Act 2: srt
    srt_file = tmp_path / "output.custom_srt"
    SubtitleExporter.save_subtitles(segments, srt_file, fmt="srt")
    assert "00:00:01,000 --> 00:00:02,000" in srt_file.read_text(encoding="utf-8")

    # Act 3: json
    json_file = tmp_path / "output.custom_json"
    SubtitleExporter.save_subtitles(segments, json_file, fmt="json")
    assert '"text": "テスト"' in json_file.read_text(encoding="utf-8")

    # Act 4: 大文字 fmt ("SRT")
    srt_upper_file = tmp_path / "output.custom_srt_upper"
    SubtitleExporter.save_subtitles(segments, srt_upper_file, fmt="SRT")
    assert "00:00:01,000 --> 00:00:02,000" in srt_upper_file.read_text(encoding="utf-8")

    # Act 5: 不正な fmt
    invalid_file = tmp_path / "output.custom"
    with pytest.raises(ValueError, match="未対応のフォーマットです"):
        SubtitleExporter.save_subtitles(segments, invalid_file, fmt="invalid")
