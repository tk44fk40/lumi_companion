"""AudioProcessor (audio/processor.py) の単体テストモジュール。"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lumi_companion.audio.processor import DEFAULT_INITIAL_PROMPT, AudioProcessor
from lumi_companion.models.audio import SubtitleSegment


def test_audio_processor_initialization_defaults() -> None:
    """AudioProcessor がデフォルト値で正しく初期化されることを検証します。"""
    # Arrange & Act
    processor = AudioProcessor()

    # Assert
    assert processor.model_size == "large-v3-turbo"
    assert processor.device == "auto"
    assert processor.compute_type == "default"
    assert processor.language == "ja"
    assert processor.beam_size == 5
    assert processor.initial_prompt == DEFAULT_INITIAL_PROMPT
    assert processor.condition_on_previous_text is False
    assert processor.vad_filter is True
    assert processor.vad_threshold == 0.35
    assert processor.vad_min_silence_duration_ms == 500
    assert processor.no_speech_threshold == 0.6
    assert processor.max_chars_per_second == 12.0


def test_audio_processor_sanitize_segments_drops_no_speech_hallucination() -> None:
    """no_speech_prob が高数値（無音捏造）のセグメントがドロップされることを検証します。"""
    # Arrange
    processor = AudioProcessor(no_speech_threshold=0.6)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=0.0, end=2.0, text="正常なセグメント", no_speech_prob=0.1
        ),
        SimpleNamespace(start=2.5, end=4.0, text="無音捏造字幕", no_speech_prob=0.8),
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "正常なセグメント"


def test_audio_processor_sanitize_segments_drops_excessive_speech_rate() -> None:
    """0.3秒で10文字など、物理的発話限界（12文字/秒）を超える捏造セグメントがドロップされることを検証します。"""
    # Arrange
    processor = AudioProcessor(max_chars_per_second=12.0)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=0.0, end=2.0, text="こんにちは", no_speech_prob=0.1
        ),  # 2.0s 5文字 = 2.5文字/秒
        SimpleNamespace(
            start=2.0, end=2.3, text="はいはいはいはいはいはい", no_speech_prob=0.1
        ),  # 0.3s 10文字 = 33.3文字/秒 (捏造)
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "こんにちは"


def test_audio_processor_sanitize_segments_preserves_natural_repetition() -> None:
    """1.5秒で「はいはいはい」（6文字, 4文字/秒）などの自然な独り言の繰り返しが正常保持されることを検証します。"""
    # Arrange
    processor = AudioProcessor(max_chars_per_second=12.0)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=1.0, end=2.5, text="はいはいはい", no_speech_prob=0.05
        ),  # 1.5s 6文字 = 4.0文字/秒 (自然)
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "はいはいはい"


def test_audio_processor_sanitize_segments_preserves_short_utterances() -> None:
    """0.1秒の「はい」（2文字）など、4文字以下の短文は発話速度チェック対象外として安全に保持されることを検証します。"""
    # Arrange
    processor = AudioProcessor(max_chars_per_second=12.0)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=0.0, end=0.1, text="はい", no_speech_prob=0.1
        ),  # 0.1s 2文字（短文保護対象）
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "はい"


def test_audio_processor_sanitize_segments_ignores_empty_or_whitespace_text() -> None:
    """空文字や空白のみのセグメントが安全に無視されることを検証します。"""
    # Arrange
    processor = AudioProcessor()
    fake_segments: list[Any] = [
        SimpleNamespace(start=0.0, end=1.0, text="", no_speech_prob=0.0),
        SimpleNamespace(start=1.0, end=2.0, text="   ", no_speech_prob=0.0),
        SimpleNamespace(start=2.0, end=3.0, text="正常発話", no_speech_prob=0.1),
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "正常発話"


@patch("lumi_companion.audio.processor.WhisperModel")
def test_audio_processor_process_sync_calls_transcribe_with_parameters(
    mock_whisper_model_class: MagicMock,
) -> None:
    """process_sync がコンストラクタ引数通りのパラメータで transcribe を呼び出すことを検証します。"""
    # Arrange
    mock_model_instance = MagicMock()
    mock_whisper_model_class.return_value = mock_model_instance
    mock_model_instance.transcribe.return_value = (
        [SimpleNamespace(start=0.0, end=1.5, text="テスト音声", no_speech_prob=0.1)],
        None,
    )

    processor = AudioProcessor(
        model_size="small",
        vad_threshold=0.3,
        initial_prompt="カスタムプロンプト",
    )

    with patch("pathlib.Path.exists", return_value=True):
        # Act
        results = processor.process_sync("dummy.mp4")

        # Assert
        assert len(results) == 1
        assert results[0].text == "テスト音声"
        mock_model_instance.transcribe.assert_called_once_with(
            "dummy.mp4",
            beam_size=5,
            language="ja",
            initial_prompt="カスタムプロンプト",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "threshold": 0.3,
            },
            no_speech_threshold=0.6,
        )


@pytest.mark.asyncio
@patch("lumi_companion.audio.processor.WhisperModel")
async def test_audio_processor_process_async(
    mock_whisper_model_class: MagicMock,
) -> None:
    """process_async が非同期に process_sync を実行して結果を返却することを検証します。"""
    # Arrange
    mock_model_instance = MagicMock()
    mock_whisper_model_class.return_value = mock_model_instance
    mock_model_instance.transcribe.return_value = (
        [
            SimpleNamespace(
                start=0.0, end=1.5, text="非同期テスト音声", no_speech_prob=0.1
            )
        ],
        None,
    )

    processor = AudioProcessor()

    with patch("pathlib.Path.exists", return_value=True):
        # Act
        results = await processor.process_async("dummy_async.mp4")

        # Assert
        assert len(results) == 1
        assert results[0].text == "非同期テスト音声"
