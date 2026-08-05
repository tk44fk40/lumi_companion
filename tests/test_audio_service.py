"""AudioProcessorService 単体テストモジュール。

本モジュールは、AudioProcessorService の初期化およびコンテキスト連携動作を検証します。
"""

from lumi_companion.core.context import AppContext
from lumi_companion.services.audio_service import AudioProcessorService


def test_audio_processor_service_initialization(mock_context: AppContext) -> None:
    """AudioProcessorService が正しく初期化されることを検証します。"""
    # Arrange & Act (準備・実行)
    service = AudioProcessorService(mock_context)

    # Assert (検証)
    assert service.context == mock_context
    assert service._model is None
