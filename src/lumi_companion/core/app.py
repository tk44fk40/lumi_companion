"""アプリケーションオーケストレーターモジュール。

本モジュールは、パイプラインのライフサイクルおよび各サービスコンポーネントの
結合・依存性注入 (DI) を統合管理する LumiApp クラスを提供します。
"""

from pathlib import Path

from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger
from lumi_companion.models import (
    AudioProcessResult,
    FrameExtractResult,
    OllamaPayload,
    OllamaResponse,
)
from lumi_companion.services import (
    AudioProcessorService,
    AudioServiceProtocol,
    LLMServiceProtocol,
    OllamaClientService,
    PromptBuilderService,
    PromptServiceProtocol,
    VisionExtractorService,
    VisionServiceProtocol,
)

logger = LumiLogger.get_logger(__name__)


class LumiApp:
    """アプリケーション全体を統括するオーケストレータークラス (DI対応)。"""

    def __init__(
        self,
        context: AppContext | None = None,
        audio_service: AudioServiceProtocol | None = None,
        vision_service: VisionServiceProtocol | None = None,
        prompt_service: PromptServiceProtocol | None = None,
        llm_service: LLMServiceProtocol | None = None,
    ) -> None:
        """依存性注入 (DI) をサポートした LumiApp を初期化します。

        Args:
            context (AppContext | None, optional): アプリ設定コンテキスト。
            audio_service (AudioServiceProtocol | None, optional): 音声解析サービス。
            vision_service (VisionServiceProtocol | None, optional): フレーム抽出サービス。
            prompt_service (PromptServiceProtocol | None, optional): プロンプト構築サービス。
            llm_service (LLMServiceProtocol | None, optional): LLM 通信サービス。
        """
        self.context = context or AppContext()

        # 指定されていない場合はデフォルト本番サービスを注入
        self.audio_service = audio_service or AudioProcessorService(self.context)
        self.vision_service = vision_service or VisionExtractorService(self.context)
        self.prompt_service = prompt_service or PromptBuilderService(self.context)
        self.llm_service = llm_service or OllamaClientService(self.context)

    async def run_step1_audio(
        self, video_path: Path | str | None = None
    ) -> AudioProcessResult:
        """Step 1: 音声トラックからの発言抽出を実行します。

        Args:
            video_path (Path | str | None, optional): 入力動画パス。

        Returns:
            AudioProcessResult: 音声処理結果。
        """
        target_path = Path(video_path or self.context.default_video_path)
        logger.info("--- Step 1: 発言抽出処理開始 ---")
        return await self.audio_service.process_audio_async(target_path)

    async def run_step2_vision(
        self,
        video_path: Path | str | None = None,
        timestamp_seconds: float = 10.0,
    ) -> FrameExtractResult:
        """Step 2: フレーム抽出 ＆ 480p リサイズを実行します。

        Args:
            video_path (Path | str | None, optional): 入力動画パス。
            timestamp_seconds (float, optional): 抽出秒。デフォルト 10.0秒。

        Returns:
            FrameExtractResult: 画像抽出結果。
        """
        target_path = Path(video_path or self.context.default_video_path)
        logger.info("--- Step 2: フレーム抽出処理開始 ---")
        return await self.vision_service.extract_frame_async(
            target_path, timestamp_seconds
        )

    async def run_step3_prompt(
        self,
        audio_result: AudioProcessResult | None = None,
        frame_result: FrameExtractResult | None = None,
        model: str | None = None,
    ) -> OllamaPayload:
        """Step 3: Ollama 投入用プロンプト JSON を構築します。

        Args:
            audio_result (AudioProcessResult | None, optional): 音声処理結果。
            frame_result (FrameExtractResult | None, optional): 画像抽出結果。
            model (str | None, optional): 使用モデル名。

        Returns:
            OllamaPayload: 構築された Ollama ペイロード。
        """
        logger.info("--- Step 3: プロンプト構築処理開始 ---")
        subtitles = audio_result.segments if audio_result else None
        image_b64 = frame_result.image_base64 if frame_result else None

        return self.prompt_service.build_payload(
            subtitles=subtitles,
            image_base64=image_b64,
            model=model or self.context.ollama_model,
            num_ctx=self.context.ollama_num_ctx,
        )

    async def run_step4_llm(
        self, payload: OllamaPayload
    ) -> OllamaResponse:
        """Step 4: ローカル Ollama サーバーへの推論リクエストを実行します。

        Args:
            payload (OllamaPayload): 送信用ペイロード。

        Returns:
            OllamaResponse: 推論応答結果。
        """
        logger.info("--- Step 4: Ollama 推論リクエスト開始 ---")
        return await self.llm_service.chat_async(payload)

    async def run_full_pipeline(
        self,
        video_path: Path | str | None = None,
        timestamp_seconds: float = 10.0,
        model: str | None = None,
    ) -> OllamaResponse:
        """Step 1 〜 Step 4 までのパイプラインを一括実行します。

        Args:
            video_path (Path | str | None, optional): 入力動画パス。
            timestamp_seconds (float, optional): 抽出フレーム秒数。
            model (str | None, optional): 推論モデル名。

        Returns:
            OllamaResponse: 最終推論応答。
        """
        logger.info("=== LumiApp パイプライン一括処理開始 ===")
        path = Path(video_path or self.context.default_video_path)

        audio_res = await self.run_step1_audio(path)
        frame_res = await self.run_step2_vision(path, timestamp_seconds)
        payload = await self.run_step3_prompt(audio_res, frame_res, model)
        response = await self.run_step4_llm(payload)

        logger.info("=== LumiApp パイプライン一括処理完了 ===")
        return response
