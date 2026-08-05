"""VisionExtractorService 単体テストモジュール。

本モジュールは、VisionExtractorService の初期化およびコンテキスト連携動作を検証します。
"""

from lumi_companion.core.context import AppContext
from lumi_companion.services.vision_service import VisionExtractorService


def test_vision_extractor_service_initialization(mock_context: AppContext) -> None:
    """VisionExtractorService が正しく初期化されることを検証します。"""
    # Arrange & Act (準備・実行)
    service = VisionExtractorService(mock_context)

    # Assert (検証)
    assert service.context == mock_context
