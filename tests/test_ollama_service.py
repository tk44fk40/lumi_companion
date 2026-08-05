"""OllamaClientService 単体テストモジュール。

本モジュールは、OllamaClientService の初期化およびコンテキスト連携動作を検証します。
"""

from lumi_companion.core.context import AppContext
from lumi_companion.services.ollama_service import OllamaClientService


def test_ollama_client_service_initialization(mock_context: AppContext) -> None:
    """OllamaClientService が正しく初期化されることを検証します。"""
    # Arrange & Act (準備・実行)
    service = OllamaClientService(mock_context)

    # Assert (検証)
    assert service.context == mock_context
    assert service.host == mock_context.ollama_host.rstrip("/")
