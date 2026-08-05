"""プロンプト構築サービスモジュール。

本モジュールは、配信理解AI「るみぽん」のキャラクタープロンプト、
発言文脈、画像Base64を結合した Ollama ペイロード構築サービスを提供します。
"""

from collections.abc import Sequence

from lumi_companion.audio.srt_exporter import SRTExporter
from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger
from lumi_companion.models.audio import SubtitleSegment
from lumi_companion.models.prompt import ChatMessage, OllamaPayload

logger = LumiLogger.get_logger(__name__)

SYSTEM_PROMPT_RUMIPON = """あなたはライブ配信をリアルタイムで一緒に視聴しているAIパートナー「るみぽん！」(Lumi)です。

【あなたの役割・キャラクター】
- 配信画面の状況（ゲーム画面やカメラ映像）と配信者の発言を聞いて、自然な一視聴者としてリアクションやコメントを返します。
- 明るく親しみやすい口調（「〜だよ！」「〜だね！」「おおっ！」など）で応答します。
- 長文の解説ではなく、チャットに流れるような短文（1〜2文程度、50文字以内）でレスポンスしてください。
- 画面と発言の文脈に合わせたリアクションを心がけてください。
"""


class PromptBuilderService:
    """Ollama API ペイロード構築サービス。"""

    def __init__(self, context: AppContext | None = None) -> None:
        """サービスの初期化を行います。

        Args:
            context (AppContext | None, optional): 設定コンテキスト。
        """
        self.context = context or AppContext()

    def format_subtitles_text(
        self, subtitles: Sequence[SubtitleSegment]
    ) -> str:
        """字幕セグメントをタイムスタンプ付きの文脈テキストに変換します。

        Args:
            subtitles (Sequence[SubtitleSegment]): 発言字幕リスト。

        Returns:
            str: 整形された発言文章。
        """
        if not subtitles:
            return "(直近の発言はありません)"

        lines: list[str] = []
        for seg in subtitles:
            ts = SRTExporter.format_timestamp(seg.start)
            lines.append(f"[{ts}] 発言: {seg.text}")
        return "\n".join(lines)

    def build_payload(
        self,
        subtitles: Sequence[SubtitleSegment] | None = None,
        image_base64: str | None = None,
        user_prompt: str | None = None,
        model: str | None = None,
        num_ctx: int | None = None,
    ) -> OllamaPayload:
        """Ollama API 用のペイロードオブジェクトを構築します。

        Args:
            subtitles (Sequence[SubtitleSegment] | None, optional): 発言字幕。
            image_base64 (str | None, optional): 画像 Base64 データ。
            user_prompt (str | None, optional): カスタムユーザー指示。
            model (str | None, optional): 推論モデル名。
            num_ctx (int | None, optional): コンテキスト長。

        Returns:
            OllamaPayload: 構築された Ollama ペイロード。
        """
        target_model = model or self.context.ollama_model
        target_num_ctx = num_ctx or self.context.ollama_num_ctx

        subtitles_str = self.format_subtitles_text(subtitles or [])
        instruction = (
            user_prompt
            or "現在の配信画面と配信者の発言を踏まえて、チャットへ一言リアクションコメントを返してください。"
        )

        user_content = f"【直近の配信者発言】\n{subtitles_str}\n\n【指示】\n{instruction}"

        images_list = [image_base64] if image_base64 else None
        user_msg = ChatMessage(
            role="user", content=user_content, images=images_list
        )
        sys_msg = ChatMessage(role="system", content=SYSTEM_PROMPT_RUMIPON)

        payload = OllamaPayload(
            model=target_model,
            messages=[sys_msg, user_msg],
            options={"num_ctx": target_num_ctx, "temperature": 0.7},
            stream=False,
        )

        logger.info("Ollama ペイロード構築完了 (モデル: %s)", target_model)
        return payload
