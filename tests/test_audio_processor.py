"""AudioProcessor (audio/processor.py) の単体テストモジュール。"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lumi_companion.audio.processor import DEFAULT_INITIAL_PROMPT, AudioProcessor


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
    assert processor.post_process_to_hankaku is False
    assert processor.post_process_normalize_nums is True
    assert processor.post_process_lower is False
    assert processor.post_process_remove_punct is False


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
            word_timestamps=True,
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


@patch("lumi_companion.audio.processor.WhisperModel")
def test_audio_processor_with_timing_adjuster(
    mock_whisper_model_class: MagicMock,
) -> None:
    """AudioProcessor に timing_adjuster が渡された場合に adjust_segments が呼び出されることを検証します。"""
    # Arrange
    mock_model_instance = MagicMock()
    mock_whisper_model_class.return_value = mock_model_instance
    mock_model_instance.transcribe.return_value = (
        [SimpleNamespace(start=0.0, end=1.0, text="テスト発話", no_speech_prob=0.1)],
        SimpleNamespace(duration=5.0, language="ja"),
    )

    mock_adjuster = MagicMock()
    mock_adjuster.adjust_segments.return_value = [
        SimpleNamespace(start=0.0, end=1.8, text="テスト発話")
    ]

    processor = AudioProcessor(timing_adjuster=mock_adjuster)

    with patch("pathlib.Path.exists", return_value=True):
        # Act
        results = processor.process_sync("dummy.mp4")

        # Assert
        assert len(results) == 1
        assert results[0].end == 1.8
        mock_adjuster.adjust_segments.assert_called_once()
