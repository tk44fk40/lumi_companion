from unittest.mock import AsyncMock

import pytest

from lumi_companion.core.context import AppContext
from lumi_companion.models.prompt import ChatMessage, OllamaPayload
from lumi_companion.services.ollama_service import OllamaClientService


def test_ollama_client_service_initialization(mock_context: AppContext) -> None:
    """OllamaClientService が正しく初期化されることを検証します。"""
    # Arrange & Act (準備・実行)
    service = OllamaClientService(mock_context)

    # Assert (検証)
    assert service.context == mock_context
    assert service.client.host == mock_context.ollama_host.rstrip("/")


@pytest.mark.asyncio
async def test_ollama_client_service_chat_async(mock_context: AppContext) -> None:
    """OllamaClientService が OllamaClient 経由で推論結果を取得できることを検証します。"""
    # Arrange (準備)
    mock_client = AsyncMock()
    mock_client.chat.return_value = {
        "model": "moondream",
        "message": {"role": "assistant", "content": "テストレスポンス"},
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    service = OllamaClientService(context=mock_context, client=mock_client)
    payload = OllamaPayload(
        model="moondream",
        messages=[ChatMessage(role="user", content="こんにちは")],
    )

    # Act (実行)
    response = await service.chat_async(payload)

    # Assert (検証)
    assert response.model == "moondream"
    assert response.content == "テストレスポンス"
    assert response.done is True
    mock_client.chat.assert_called_once_with(payload.to_dict())


@pytest.mark.asyncio
async def test_ollama_client_service_connection_error(
    mock_context: AppContext,
) -> None:
    """Ollama サーバーへの接続に失敗した場合に ConnectionError が発生することを検証します。"""
    # Arrange (準備)
    mock_client = AsyncMock()
    mock_client.chat.side_effect = ConnectionError("Ollama サーバーに接続できません")
    service = OllamaClientService(context=mock_context, client=mock_client)
    payload = OllamaPayload(
        model="moondream",
        messages=[ChatMessage(role="user", content="こんにちは")],
    )

    # Act & Assert (実行・検証)
    with pytest.raises(ConnectionError):
        await service.chat_async(payload)
