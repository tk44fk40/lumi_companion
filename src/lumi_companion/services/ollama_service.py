"""ローカル Ollama LLM 通信サービスモジュール。

本モジュールは、モデルの事前存在確認、ストリーミング自動取得、
および API 推論送信を行うサービスを提供します。
"""

import json

import httpx

from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger
from lumi_companion.models.llm import OllamaResponse
from lumi_companion.models.prompt import OllamaPayload

logger = LumiLogger.get_logger(__name__)


class OllamaClientService:
    """ローカル Ollama LLM 通信サービス。"""

    def __init__(self, context: AppContext | None = None) -> None:
        """サービスの初期化を行います。

        Args:
            context (AppContext | None, optional): 設定コンテキスト。
        """
        self.context = context or AppContext()
        self.host = self.context.ollama_host.rstrip("/")

    async def check_model_installed(self, model_name: str) -> bool:
        """指定されたモデルがローカル Ollama に存在するか確認します。

        Args:
            model_name (str): 対象のモデル名。

        Returns:
            bool: 存在する場合は True、未存在の場合は False。
        """
        url = f"{self.host}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                installed = [m.get("name", "") for m in data.get("models", [])]

                for name in installed:
                    if (
                        name == model_name
                        or name == f"{model_name}:latest"
                        or (
                            ":" in model_name
                            and name.startswith(model_name.split(":")[0])
                            and name.endswith(model_name.split(":")[1])
                        )
                    ):
                        return True
                return False
        except httpx.HTTPError as e:
            logger.warning("Ollama モデル一覧の確認に失敗しました: %s", e)
            return False

    async def pull_model(self, model_name: str) -> None:
        """Ollama API (/api/pull) を呼び出してモデルを自動ダウンロードします。

        Args:
            model_name (str): 対象のモデル名。

        Raises:
            RuntimeError: ダウンロードに失敗した場合。
        """
        url = f"{self.host}/api/pull"
        logger.info("モデル '%s' の自動取得を開始します...", model_name)

        try:
            timeout = httpx.Timeout(self.context.download_timeout_seconds, connect=10.0)
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream(
                    "POST", url, json={"name": model_name, "stream": True}
                ) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        status_data = json.loads(line)
                        if "error" in status_data:
                            raise RuntimeError(status_data["error"])
                        if status_data.get("status") == "success":
                            logger.info("モデル '%s' のプル完了", model_name)
                            return
                    except json.JSONDecodeError:
                        pass
        except (httpx.HTTPError, RuntimeError) as e:
            logger.error("モデル '%s' のプル失敗: %s", model_name, e)
            raise RuntimeError(f"モデル '{model_name}' の取得失敗: {e}") from e

    async def chat_async(self, payload: OllamaPayload) -> OllamaResponse:
        """Ollama API (/api/chat) へ推論リクエストを送信します。

        Args:
            payload (OllamaPayload): 送信用ペイロードモデル。

        Returns:
            OllamaResponse: 推論応答データモデル。

        Raises:
            ConnectionError: Ollama サーバーに接続できない場合。
            RuntimeError: API からエラーが返却された場合。
        """
        if not await self.check_model_installed(payload.model):
            await self.pull_model(payload.model)

        url = f"{self.host}/api/chat"
        logger.info("Ollama 推論リクエスト送信中 (%s)...", payload.model)

        try:
            timeout = httpx.Timeout(self.context.http_timeout_seconds, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload.to_dict())
                response.raise_for_status()
                return OllamaResponse.from_dict(response.json())
        except httpx.ConnectError as e:
            logger.error("Ollama サーバー (%s) に接続できません。", self.host)
            raise ConnectionError(f"Ollama 接続エラー: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error("Ollama API エラー (ステータス %d)", e.response.status_code)
            raise RuntimeError(f"Ollama API エラー: {e}") from e
