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
- あなたは配信者（ストリーマー）ではなく、配信を見ているリスナー（一視聴者）です。
- 配信者のプレイや発言に対して、チャット欄から明るく親しみやすい口調（「〜だよ！」「〜だね！」「おおっ！」など）で共感・応援コメントを返します。

【禁止事項・厳格な出力制約】
- 配信者のような発言（「ご視聴ありがとうございました」「配信終了です」等）や、AIアシスタントのような案内（「何かあれば教えてください」等）は絶対にしないでください。
- ハッシュタグ（#〜）やSNS風のメタ情報は出力に含めないでください。
- 出力はチャットへのコメント本文のみ（30文字前後）とし、解説や前置きは含めないでください。
- 内部的な推論や思考プロセスは極力短く簡潔に行い、すぐに結論を出力してください。
"""


class PromptBuilder:
    """Ollama Chat API 互換プロンプト JSON 構築クラス。"""

    @classmethod
    def format_subtitles_text(
        cls,
        segments: Sequence[SubtitleSegment],
        max_chars: int | None = None,
    ) -> str:
        """字幕セグメントリストをタイムスタンプ付きのテキスト文章に整形します。

        Args:
            segments (Sequence[SubtitleSegment]): 発言字幕セグメントのシーケンス。
            max_chars (int | None, optional): プロンプトに含める最大文字数。未指定時は設定値を使用。

        Returns:
            str: タイムスタンプ付きで整形された字幕テキスト。
        """
        if not segments:
            return "(直近の発言はありません)"

        limit_chars = (
            max_chars if max_chars is not None else settings.prompt_max_subtitle_chars
        )

        lines: list[str] = []
        current_chars = 0

        for seg in reversed(segments):
            ts = SubtitleExporter.format_timestamp(seg.start)
            line = f"[{ts}] 発言: {seg.text}"

            # 改行文字分を加味（最初の要素以外は +1 文字）
            line_len = len(line) + (1 if lines else 0)

            if lines and current_chars + line_len > limit_chars:
                break

            lines.append(line)
            current_chars += line_len

        return "\n".join(reversed(lines))

    @classmethod
    def build_payload(
        cls,
        image_base64: str | None = None,
        subtitles: Sequence[SubtitleSegment] | None = None,
        user_prompt: str | None = None,
        model: str | None = None,
        num_ctx: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Ollama API (/api/chat) 送信用の JSON ペイロード辞書を構築します。

        Args:
            image_base64 (str | None, optional): 添付画像の Base64 文字列。
            subtitles (Sequence[SubtitleSegment] | None, optional): 字幕セグメント。
            user_prompt (str | None, optional): カスタムユーザー指示テキスト。
            model (str | None, optional): 対象モデル名。
            num_ctx (int | None, optional): コンテキストウィンドウ長。
            temperature (float | None, optional): 推論サンプリング温度。未指定時は設定値を使用。
            stream (bool, optional): ストリーミングレスポンスフラグ。デフォルト False。

        Returns:
            dict[str, Any]: Ollama Chat API 仕様に適合した辞書データ。
        """
        target_model = model or settings.ollama_model
        target_num_ctx = num_ctx or settings.ollama_num_ctx
        target_temp = (
            temperature if temperature is not None else settings.ollama_temperature
        )

        subtitles_str = cls.format_subtitles_text(subtitles or [])
        prompt_text = (
            user_prompt
            or "現在の配信画面と配信者の発言を踏まえて、チャットへ一言リアクションコメントを返してください。"
        )

        user_content = (
            f"【直近の配信者発言】\n{subtitles_str}\n\n【指示】\n{prompt_text}\n\n"
            "※注意：ハッシュタグ(#)や配信者風の挨拶は禁止。配信を観ているリスナーとして短くコメントしてください。"
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
                "num_predict": settings.ollama_num_predict,
                "temperature": target_temp,
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
