"""プロンプト構築モジュール。

本モジュールは、AI パートナー「るみぽん！」用のシステムプロンプト定義、
音声字幕テキストのフォーマット、および Ollama API 互換ペイロード JSON の構築を行います。
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lumi_companion.audio.srt_exporter import SubtitleExporter
from lumi_companion.config import settings
from lumi_companion.models.audio import SubtitleSegment

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_RUMIPON = """あなたはライブ配信をリアルタイムで一緒に視聴しているAIパートナー「るみぽん！」(Lumi)です。

【あなたの役割・キャラクター】
- 配信画面の状況（ゲーム画面やカメラ映像）と配信者の発言を聞いて、自然な一視聴者としてリアクションやコメントを返します。
- 明るく親しみやすい口調（「〜だよ！」「〜だね！」「おおっ！」など）で応答します。
- 長文の解説ではなく、チャットに流れるような短文（1〜2文程度、50文字以内）でレスポンスしてください。
- 画面と発言の文脈に合わせたリアクションを心がけてください。
"""


class PromptBuilder:
    """Ollama Chat API 互換プロンプト JSON 構築クラス。"""

    @classmethod
    def format_subtitles_text(cls, segments: Sequence[SubtitleSegment]) -> str:
        """字幕セグメントリストをタイムスタンプ付きのテキスト文章に整形します。

        Args:
            segments (Sequence[SubtitleSegment]): 発言字幕セグメントのシーケンス。

        Returns:
            str: タイムスタンプ付きで整形された字幕テキスト。
        """
        if not segments:
            return "(直近の発言はありません)"

        lines: list[str] = []
        for seg in segments:
            ts = SubtitleExporter.format_timestamp(seg.start)
            lines.append(f"[{ts}] 発言: {seg.text}")

        return "\n".join(lines)

    @classmethod
    def build_payload(
        cls,
        image_base64: str | None = None,
        subtitles: Sequence[SubtitleSegment] | None = None,
        user_prompt: str | None = None,
        model: str | None = None,
        num_ctx: int | None = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Ollama API (/api/chat) 送信用の JSON ペイロード辞書を構築します。

        Args:
            image_base64 (str | None, optional): 添付画像の Base64 文字列。
            subtitles (Sequence[SubtitleSegment] | None, optional): 字幕セグメント。
            user_prompt (str | None, optional): カスタムユーザー指示テキスト。
            model (str | None, optional): 対象モデル名。
            num_ctx (int | None, optional): コンテキストウィンドウ長。
            temperature (float, optional): 推論サンプリング温度。デフォルト 0.7。
            stream (bool, optional): ストリーミングレスポンスフラグ。デフォルト False。

        Returns:
            dict[str, Any]: Ollama Chat API 仕様に適合した辞書データ。
        """
        target_model = model or settings.ollama_model
        target_num_ctx = num_ctx or settings.ollama_num_ctx

        subtitles_str = cls.format_subtitles_text(subtitles or [])
        prompt_text = (
            user_prompt
            or "現在の配信画面と配信者の発言を踏まえて、チャットへ一言リアクションコメントを返してください。"
        )

        user_content = (
            f"【直近の配信者発言】\n{subtitles_str}\n\n【指示】\n{prompt_text}"
        )

        message_content: dict[str, Any] = {
            "role": "user",
            "content": user_content,
        }

        if image_base64:
            message_content["images"] = [image_base64]

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_RUMIPON,
                },
                message_content,
            ],
            "options": {
                "num_ctx": target_num_ctx,
                "temperature": temperature,
            },
            "stream": stream,
        }

        logger.info(
            "Ollama ペイロードを構築完了 (モデル: %s, num_ctx: %d, 画像有無: %s)",
            target_model,
            target_num_ctx,
            bool(image_base64),
        )
        return payload

    @classmethod
    def save_payload_json(
        cls, payload: dict[str, Any], output_path: Path | str
    ) -> Path:
        """ペイロード辞書を JSON ファイルとして書き出し保存します。

        Args:
            payload (dict[str, Any]): 送信用ペイロード辞書。
            output_path (Path | str): 保存先 JSON パス。

        Returns:
            Path: 保存された JSON ファイルの絶対パス。
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("Ollama ペイロード JSON を保存しました: %s", out_path.resolve())
        return out_path
