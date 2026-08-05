from unittest.mock import MagicMock

import pytest

from lumi_companion.config import settings
from lumi_companion.core.context import AppContext
from lumi_companion.models.audio import SubtitleSegment
from lumi_companion.services.audio_service import AudioProcessorService


def test_audio_processor_service_initialization(mock_context: AppContext) -> None:
    """AudioProcessorService が正しく初期化されることを検証します。"""
    # Arrange & Act (準備・実行)
    service = AudioProcessorService(mock_context)

    # Assert (検証)
    assert service.context == mock_context
    assert service.processor is not None


def test_audio_processor_service_injects_settings_into_processor(
    mock_context: AppContext,
) -> None:
    """AudioProcessorService が settings の設定値を AudioProcessor に注入 (DI) することを検証します。"""
    # Arrange & Act (準備・実行)
    service = AudioProcessorService(mock_context)

    # Assert (検証)
    assert service.processor.model_size == settings.whisper_model_size
    assert service.processor.device == settings.whisper_device
    assert service.processor.vad_threshold == settings.whisper_vad_threshold
    assert service.processor.initial_prompt == settings.whisper_initial_prompt


@pytest.mark.asyncio
async def test_audio_processor_service_process_async(mock_context: AppContext) -> None:
    """AudioProcessorService が AudioProcessor 経由で非同期に発言抽出結果を返却することを検証します。"""
    # Arrange (準備)
    mock_processor = MagicMock()
    mock_processor.process_sync.return_value = [
        SubtitleSegment(start=1.0, end=3.0, text="モック発言1"),
        SubtitleSegment(start=4.0, end=6.0, text="モック発言2"),
    ]
    service = AudioProcessorService(context=mock_context, processor=mock_processor)

    # Act (実行)
    result = await service.process_audio_async("dummy_video.mp4")

    # Assert (検証)
    assert len(result.segments) == 2
    assert result.segments[0].text == "モック発言1"
    assert result.duration_seconds == 6.0
    mock_processor.process_sync.assert_called_once_with("dummy_video.mp4")


def test_audio_processor_service_file_not_found(mock_context: AppContext) -> None:
    """存在しない動画ファイルを指定した場合に FileNotFoundError が発生することを検証します。"""
    # Arrange (準備)
    service = AudioProcessorService(context=mock_context)

    # Act & Assert (実行・検証)
    with pytest.raises(FileNotFoundError):
        service.process_audio_sync("non_existent_video_path_12345.mp4")
