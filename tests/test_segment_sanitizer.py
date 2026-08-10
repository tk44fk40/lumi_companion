"""SegmentSanitizer の単体テストモジュール。"""

from types import SimpleNamespace
from typing import Any

import pytest

from lumi_companion.audio.segment_sanitizer import SegmentSanitizer
from lumi_companion.models.audio import SubtitleSegment


def test_segment_sanitizer_drops_no_speech_hallucination() -> None:
    """no_speech_prob が高数値（無音捏造）のセグメントがドロップされることを検証します。"""
    # Arrange
    sanitizer = SegmentSanitizer(no_speech_threshold=0.6)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=0.0, end=2.0, text="正常なセグメント", no_speech_prob=0.1
        ),
        SimpleNamespace(start=2.5, end=4.0, text="無音捏造字幕", no_speech_prob=0.8),
    ]

    # Act
    results: list[SubtitleSegment] = sanitizer.sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "正常なセグメント"


def test_segment_sanitizer_drops_excessive_speech_rate() -> None:
    """0.3秒で10文字など、物理的発話限界（12文字/秒）を超える捏造セグメントがドロップされることを検証します。"""
    # Arrange
    sanitizer = SegmentSanitizer(max_chars_per_second=12.0)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=0.0, end=2.0, text="こんにちは", no_speech_prob=0.1
        ),  # 2.0s 5文字 = 2.5文字/秒
        SimpleNamespace(
            start=2.0, end=2.3, text="はいはいはいはいはいはい", no_speech_prob=0.1
        ),  # 0.3s 10文字 = 33.3文字/秒 (捏造)
    ]

    # Act
    results: list[SubtitleSegment] = sanitizer.sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "こんにちは"


def test_segment_sanitizer_preserves_natural_repetition() -> None:
    """1.5秒で「はいはいはい」（6文字, 4文字/秒）などの自然な独り言の繰り返しが正常保持されることを検証します。"""
    # Arrange
    sanitizer = SegmentSanitizer(max_chars_per_second=12.0)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=1.0, end=2.5, text="はいはいはい", no_speech_prob=0.05
        ),  # 1.5s 6文字 = 4.0文字/秒 (自然)
    ]

    # Act
    results: list[SubtitleSegment] = sanitizer.sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "はいはいはい"


def test_segment_sanitizer_preserves_short_utterances() -> None:
    """0.1秒の「はい」（2文字）など、4文字以下の短文は発話速度チェック対象外として安全に保持されることを検証します。"""
    # Arrange
    sanitizer = SegmentSanitizer(max_chars_per_second=12.0)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=0.0, end=0.1, text="はい", no_speech_prob=0.1
        ),  # 0.1s 2文字（短文保護対象）
    ]

    # Act
    results: list[SubtitleSegment] = sanitizer.sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "はい"


def test_segment_sanitizer_ignores_empty_or_whitespace_text() -> None:
    """空文字や空白のみのセグメントが安全に無視されることを検証します。"""
    # Arrange
    sanitizer = SegmentSanitizer()
    fake_segments: list[Any] = [
        SimpleNamespace(start=0.0, end=1.0, text="", no_speech_prob=0.0),
        SimpleNamespace(start=1.0, end=2.0, text="   ", no_speech_prob=0.0),
        SimpleNamespace(start=2.0, end=3.0, text="正常発話", no_speech_prob=0.1),
    ]

    # Act
    results: list[SubtitleSegment] = sanitizer.sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "正常発話"


def test_segment_sanitizer_drops_inter_segment_repetition() -> None:
    """直前のセグメントと全く同じテキスト（または包含関係）が連続し、無音確率が高い場合にドロップされることを検証します。"""
    # Arrange
    sanitizer = SegmentSanitizer()
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=0.0,
            end=2.0,
            text="ご視聴ありがとうございました。",
            no_speech_prob=0.05,
        ),
        # 全く同じテキストで時間が進み、無音確率が > 0.1 の場合はループとみなす
        SimpleNamespace(
            start=3.0,
            end=5.0,
            text="ご視聴ありがとうございました。",
            no_speech_prob=0.15,
        ),
        # 包含されている場合もループとみなす
        SimpleNamespace(
            start=5.0, end=7.0, text="ありがとうございました。", no_speech_prob=0.2
        ),
        SimpleNamespace(start=7.0, end=9.0, text="正常な次の発話", no_speech_prob=0.01),
    ]

    # Act
    results: list[SubtitleSegment] = sanitizer.sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 2
    assert results[0].text == "ご視聴ありがとうございました。"
    assert results[1].text == "正常な次の発話"


def test_segment_sanitizer_get_word_time_dict() -> None:
    """get_word_time が辞書型の入力に対して正しく動作することを検証します。"""
    # dict 型かつ対象キーが存在し、数値の場合
    assert SegmentSanitizer.get_word_time({"start": 1.23}, "start") == 1.23
    assert SegmentSanitizer.get_word_time({"start": 5}, "start") == 5.0
    # dict 型だが対象キーが存在しない、または数値でない場合
    assert SegmentSanitizer.get_word_time({"end": 4.56}, "start") is None
    assert SegmentSanitizer.get_word_time({"start": "invalid"}, "start") is None
    # 辞書型以外（オブジェクト）の場合でキーが存在しない、または数値でない場合
    obj = SimpleNamespace(start="not_a_number")
    assert SegmentSanitizer.get_word_time(obj, "start") is None
    assert SegmentSanitizer.get_word_time(obj, "end") is None


def test_segment_sanitizer_drops_intra_segment_repetition() -> None:
    """1つのセグメント内で完全に同じフレーズが繰り返されている場合、重複部分が除去されることを検証します。"""
    # Arrange
    sanitizer = SegmentSanitizer()
    fake_segments: list[Any] = [
        # ちょうど半分で繰り返されているパターン (ハルシネーションを模して compression_ratio を高くする)
        SimpleNamespace(
            start=0.0,
            end=2.0,
            text="あいうえおあいうえお",
            no_speech_prob=0.05,
            compression_ratio=2.5,
        ),
        SimpleNamespace(
            start=2.0,
            end=4.0,
            text="正常な発話です。",
            no_speech_prob=0.05,
            compression_ratio=1.0,
        ),
    ]

    # Act
    results: list[SubtitleSegment] = sanitizer.sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 2
    assert results[0].text == "あいうえお"
    assert results[1].text == "正常な発話です。"


def test_segment_sanitizer_logs_with_total_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """total_duration が指定された場合に進行状況（%）を含むログが出力されることを検証します。"""
    # Arrange
    sanitizer = SegmentSanitizer()
    fake_segments: list[Any] = [
        SimpleNamespace(start=0.0, end=10.0, text="途中経過テスト", no_speech_prob=0.1),
    ]

    # Act
    with caplog.at_level("INFO"):
        results = sanitizer.sanitize_segments(fake_segments, total_duration=100.0)

    # Assert
    assert len(results) == 1
    assert (
        "発言検出 [ 10.0s / 100.0s ( 10%)] [0.00s -> 10.00s]: 途中経過テスト"
        in caplog.text
    )
