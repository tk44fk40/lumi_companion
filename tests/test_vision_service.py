from unittest.mock import patch

import pytest
from PIL import Image

from lumi_companion.core.context import AppContext
from lumi_companion.services.vision_service import VisionExtractorService


def test_vision_extractor_service_initialization(mock_context: AppContext) -> None:
    """VisionExtractorService が正しく初期化されることを検証します。"""
    # Arrange & Act (準備・実行)
    service = VisionExtractorService(mock_context)

    # Assert (検証)
    assert service.context == mock_context


@pytest.mark.asyncio
async def test_vision_extractor_service_extract_frame_async(
    mock_context: AppContext,
) -> None:
    """VisionExtractorService が FrameExtractor 経由で非同期にフレーム抽出結果を返却することを検証します。"""
    # Arrange (準備)
    service = VisionExtractorService(mock_context)
    dummy_image = Image.new("RGB", (640, 360))

    with (
        patch(
            "lumi_companion.services.vision_service.FrameExtractor.extract_frame_pil",
            return_value=dummy_image,
        ) as mock_pil,
        patch(
            "lumi_companion.services.vision_service.FrameExtractor.extract_frame_base64",
            return_value="bW9ja19iYXNlNjQ=",
        ) as mock_b64,
    ):
        # Act (実行)
        result = await service.extract_frame_async("dummy_video.mp4", 10.0)

        # Assert (検証)
        assert result.timestamp_seconds == 10.0
        assert result.width == 640
        assert result.height == 360
        assert result.image_base64 == "bW9ja19iYXNlNjQ="
        mock_pil.assert_called_once_with(
            video_path="dummy_video.mp4",
            timestamp_seconds=10.0,
            max_width=mock_context.max_image_width_px,
        )
        mock_b64.assert_called_once_with(
            video_path="dummy_video.mp4",
            timestamp_seconds=10.0,
            max_width=mock_context.max_image_width_px,
            quality=mock_context.jpeg_quality,
        )


def test_vision_extractor_service_file_not_found(mock_context: AppContext) -> None:
    """存在しない動画ファイルを指定した場合に FileNotFoundError が発生することを検証します。"""
    # Arrange (準備)
    service = VisionExtractorService(mock_context)

    # Act & Assert (実行・検証)
    with pytest.raises(FileNotFoundError):
        service.extract_frame_sync("non_existent_video_path_12345.mp4", 10.0)
