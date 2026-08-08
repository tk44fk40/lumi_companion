"""OllamaClient 単体テストモジュール。

本モジュールは、OllamaClient によるモデルチェック、ストリーミングプル、
推論リクエストおよびエラーハンドリングの動作を検証します。
"""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from lumi_companion.llm.ollama import OllamaClient

_orig_AsyncClient = httpx.AsyncClient


class TestOllamaClient:
    """OllamaClient の単体テストクラス。"""

    def test_init_host_formatting(self) -> None:
        """ホストURLの末尾スラッシュ削除とデフォルト値設定を検証します。"""
        client1 = OllamaClient(host="http://localhost:11434/")
        assert client1.host == "http://localhost:11434"

        client2 = OllamaClient(host="http://custom:11434")
        assert client2.host == "http://custom:11434"

    @pytest.mark.asyncio
    async def test_check_model_installed_exact_match(self) -> None:
        """モデル名が完全一致・タグ一致する場合に True、不一致時に False を返すことを検証します。"""
        client = OllamaClient(host="http://test:11434")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url == "http://test:11434/api/tags"
            return httpx.Response(
                200,
                json={"models": [{"name": "moondream:latest"}, {"name": "llava:7b"}]},
            )

        transport = httpx.MockTransport(mock_handler)
        with patch(
            "httpx.AsyncClient",
            side_effect=lambda *a, **kw: _orig_AsyncClient(transport=transport),
        ):
            assert await client.check_model_installed("moondream:latest") is True
            assert await client.check_model_installed("moondream") is True
            assert await client.check_model_installed("llava:7b") is True
            assert await client.check_model_installed("nonexistent") is False

    @pytest.mark.asyncio
    async def test_check_model_installed_http_error(self) -> None:
        """HTTPエラー発生時に warning を記録し False を返すことを検証します。"""
        client = OllamaClient(host="http://test:11434")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        transport = httpx.MockTransport(mock_handler)
        with patch(
            "httpx.AsyncClient",
            side_effect=lambda *a, **kw: _orig_AsyncClient(transport=transport),
        ):
            result = await client.check_model_installed("moondream")
            assert result is False

    @pytest.mark.asyncio
    async def test_pull_model_success(self) -> None:
        """モデルのストリーミングダウンロード進捗と正常完了を検証します。"""
        client = OllamaClient(host="http://test:11434")

        lines = [
            b"",
            b"invalid json line",
            json.dumps({"status": "downloading", "completed": 20, "total": 100}).encode(
                "utf-8"
            ),
            json.dumps(
                {"status": "downloading", "completed": 100, "total": 100}
            ).encode("utf-8"),
            json.dumps({"status": "verifying"}).encode("utf-8"),
            json.dumps({"status": "success"}).encode("utf-8"),
        ]

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url == "http://test:11434/api/pull"
            return httpx.Response(200, content=b"\n".join(lines))

        transport = httpx.MockTransport(mock_handler)
        with patch(
            "httpx.AsyncClient",
            side_effect=lambda *a, **kw: _orig_AsyncClient(transport=transport),
        ):
            await client.pull_model("moondream")

    @pytest.mark.asyncio
    async def test_pull_model_stream_ends_without_success_status(self) -> None:
        """Success ステータスが送られずにストリームが終了した場合も正常終了することを検証します。"""
        client = OllamaClient(host="http://test:11434")

        lines = [
            json.dumps({"status": "downloading", "completed": 50, "total": 100}).encode(
                "utf-8"
            ),
        ]

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"\n".join(lines))

        transport = httpx.MockTransport(mock_handler)
        with patch(
            "httpx.AsyncClient",
            side_effect=lambda *a, **kw: _orig_AsyncClient(transport=transport),
        ):
            await client.pull_model("moondream")

    @pytest.mark.asyncio
    async def test_pull_model_api_error(self) -> None:
        """ストリーミング中に API エラーが返された場合に RuntimeError が発生することを検証します。"""
        client = OllamaClient(host="http://test:11434")

        lines = [
            json.dumps({"error": "model not found"}).encode("utf-8"),
        ]

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"\n".join(lines))

        transport = httpx.MockTransport(mock_handler)
        with patch(
            "httpx.AsyncClient",
            side_effect=lambda *a, **kw: _orig_AsyncClient(transport=transport),
        ):
            with pytest.raises(
                RuntimeError, match="Ollama モデル 'moondream' の取得失敗"
            ):
                await client.pull_model("moondream")

    @pytest.mark.asyncio
    async def test_pull_model_http_error(self) -> None:
        """HTTP通信エラー発生時に RuntimeError が発生することを検証します。"""
        client = OllamaClient(host="http://test:11434")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        transport = httpx.MockTransport(mock_handler)
        with patch(
            "httpx.AsyncClient",
            side_effect=lambda *a, **kw: _orig_AsyncClient(transport=transport),
        ):
            with pytest.raises(
                RuntimeError, match="Ollama モデル 'moondream' の取得に失敗しました"
            ):
                await client.pull_model("moondream")

    @pytest.mark.asyncio
    async def test_ensure_model_available_already_installed(self) -> None:
        """既にインストール済みの場合はプルを実行しないことを検証します。"""
        client = OllamaClient()
        with (
            patch.object(
                client, "check_model_installed", new_callable=AsyncMock
            ) as mock_check,
            patch.object(client, "pull_model", new_callable=AsyncMock) as mock_pull,
        ):
            mock_check.return_value = True
            await client.ensure_model_available("moondream")
            mock_check.assert_called_once_with("moondream")
            mock_pull.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_model_available_needs_pull(self) -> None:
        """未インストールの場合はプルを実行することを検証します。"""
        client = OllamaClient()
        with (
            patch.object(
                client, "check_model_installed", new_callable=AsyncMock
            ) as mock_check,
            patch.object(client, "pull_model", new_callable=AsyncMock) as mock_pull,
        ):
            mock_check.return_value = False
            await client.ensure_model_available("moondream")
            mock_check.assert_called_once_with("moondream")
            mock_pull.assert_called_once_with("moondream")

    @pytest.mark.asyncio
    async def test_chat_success(self) -> None:
        """推論リクエストが正常に送信され結果が返却されることを検証します。"""
        client = OllamaClient(host="http://test:11434")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url == "http://test:11434/api/chat"
            return httpx.Response(
                200,
                json={
                    "model": "moondream",
                    "message": {"role": "assistant", "content": "テスト応答"},
                    "done": True,
                },
            )

        transport = httpx.MockTransport(mock_handler)
        with (
            patch.object(
                client, "ensure_model_available", new_callable=AsyncMock
            ) as mock_ensure,
            patch(
                "httpx.AsyncClient",
                side_effect=lambda *a, **kw: _orig_AsyncClient(transport=transport),
            ),
        ):
            result = await client.chat({"model": "moondream", "messages": []})
            mock_ensure.assert_called_once_with("moondream")
            assert result["message"]["content"] == "テスト応答"

    @pytest.mark.asyncio
    async def test_chat_connect_error(self) -> None:
        """接続エラー発生時に ConnectionError が発生することを検証します。"""
        client = OllamaClient(host="http://test:11434")

        with (
            patch.object(client, "ensure_model_available", new_callable=AsyncMock),
            patch(
                "httpx.AsyncClient.post",
                side_effect=httpx.ConnectError("Connection refused"),
            ),
        ):
            with pytest.raises(
                ConnectionError, match="Ollama サーバーに接続できません"
            ):
                await client.chat({"model": "moondream", "messages": []})

    @pytest.mark.asyncio
    async def test_chat_http_status_error(self) -> None:
        """HTTPステータスエラー発生時に RuntimeError が発生することを検証します。"""
        client = OllamaClient(host="http://test:11434")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        transport = httpx.MockTransport(mock_handler)
        with (
            patch.object(client, "ensure_model_available", new_callable=AsyncMock),
            patch(
                "httpx.AsyncClient",
                side_effect=lambda *a, **kw: _orig_AsyncClient(transport=transport),
            ),
        ):
            with pytest.raises(RuntimeError, match="Ollama API エラー"):
                await client.chat({"model": "moondream", "messages": []})
