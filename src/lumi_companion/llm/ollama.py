import json
import logging
from typing import Any

import httpx

from lumi_companion.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """ローカル Ollama サーバーとの通信を管理するクライアントクラス"""

    def __init__(self, host: str | None = None) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")

    async def check_model_installed(self, model_name: str) -> bool:
        """指定されたモデルがローカル Ollama に存在するか確認"""
        url = f"{self.host}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                installed_models = [
                    m.get("name", "") for m in data.get("models", [])
                ]

                # タグの比較 (例: qwen2-vl:2b, qwen2-vl:2b-latest, qwen2-vl:latest)
                for name in installed_models:
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
        """Ollama API (/api/pull) を呼び出してモデルを自動ダウンロード (成功まで待機)"""
        url = f"{self.host}/api/pull"
        logger.info("Ollama からモデル '%s' の自動プルを開始します...", model_name)
        payload = {"name": model_name, "stream": True}

        try:
            # モデルダウンロード用の長時間タイムアウト (30分)
            timeout = httpx.Timeout(1800.0, connect=10.0)
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream("POST", url, json=payload) as response,
            ):
                response.raise_for_status()
                last_status = ""
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        status_data = json.loads(line)
                        status_msg = status_data.get("status", "")
                        completed = status_data.get("completed", 0)
                        total = status_data.get("total", 0)

                        if status_msg == "success":
                            logger.info("モデル '%s' のダウンロードおよび構築が正常に完了しました。", model_name)
                            return

                        if total > 0:
                            percent = (completed / total) * 100
                            current_status = f"{status_msg} ({percent:.1f}%)"
                            if current_status != last_status and (int(percent) % 20 == 0 or percent >= 99.0):
                                logger.info("[モデルダウンロード進捗] %s", current_status)
                                last_status = current_status
                        elif status_msg and status_msg != last_status:
                            logger.info("[モデルダウンロード状態] %s", status_msg)
                            last_status = status_msg
                    except json.JSONDecodeError:
                        pass
            logger.info("モデル '%s' のプル処理が終了しました。", model_name)
        except httpx.HTTPError as e:
            logger.error("モデル '%s' のプルに失敗しました: %s", model_name, e)
            raise RuntimeError(f"Ollama モデル '{model_name}' の取得に失敗しました: {e}") from e

    async def ensure_model_available(self, model_name: str) -> None:
        """モデルが存在するか確認し、無ければ自動でプルを実行"""
        is_installed = await self.check_model_installed(model_name)
        if is_installed:
            logger.info("Ollama モデル '%s' の存在を確認しました。", model_name)
            return

        logger.info("Ollama モデル '%s' が見つからないため、自動取得を開始します。", model_name)
        await self.pull_model(model_name)

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ollama /api/chat API へリクエストを送信し推論結果を取得"""
        model_name = payload.get("model", settings.ollama_model)
        await self.ensure_model_available(model_name)

        url = f"{self.host}/api/chat"
        logger.info("Ollama 推論リクエストを送信中 (モデル: %s)...", model_name)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                response_data = response.json()
                logger.info("Ollama から推論応答を正常に受信しました。")
                return response_data
        except httpx.ConnectError as e:
            logger.error(
                "Ollama サーバー (%s) に接続できません。'ollama serve' が起動しているか確認してください。",
                self.host,
            )
            raise ConnectionError(f"Ollama サーバーに接続できません: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error("Ollama API エラー (ステータス %d): %s", e.response.status_code, e.response.text)
            raise RuntimeError(f"Ollama API エラー: {e}") from e
