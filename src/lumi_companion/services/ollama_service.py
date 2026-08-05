"""ローカル Ollama LLM 通信サービスモジュール。

本モジュールは、モデルの事前存在確認、ストリーミング自動取得、
および API 推論送信を行うサービスを提供します。
"""

from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger
from lumi_companion.llm.ollama import OllamaClient
from lumi_companion.models.llm import OllamaResponse
from lumi_companion.models.prompt import OllamaPayload

logger = LumiLogger.get_logger(__name__)


class OllamaClientService:
    """ローカル Ollama LLM 通信サービス (Protocol適合)。"""

    def __init__(
        self,
        context: AppContext | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        """サービスの初期化を行います。

        Args:
            context (AppContext | None, optional): 設定コンテキスト。
            client (OllamaClient | None, optional): LLM クライアント。
        """
        self.context = context or AppContext()
        self.client = client or OllamaClient(host=self.context.ollama_host)

    async def check_model_installed(self, model_name: str) -> bool:
        """指定されたモデルがローカル Ollama に存在するか確認します。

        Args:
            model_name (str): 対象のモデル名。

        Returns:
            bool: 存在する場合は True、未存在の場合は False。
        """
        return await self.client.check_model_installed(model_name)

    async def pull_model(self, model_name: str) -> None:
        """Ollama API (/api/pull) を呼び出してモデルを自動ダウンロードします。

        Args:
            model_name (str): 対象のモデル名。

        Raises:
            RuntimeError: ダウンロードに失敗した場合。
        """
        await self.client.pull_model(model_name)

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
        raw_res = await self.client.chat(payload.to_dict())
        return OllamaResponse.from_dict(raw_res)
