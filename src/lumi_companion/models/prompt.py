"""プロンプト・メッセージ構造データモデルモジュール。

本モジュールは、Ollama API 互換のチャットメッセージおよび
JSON ペイロードを表現する @dataclass データ構造を提供します。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """チャットメッセージモデル。"""

    role: str
    content: str
    images: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """データモデルを辞書形式へ変換します。

        Returns:
            dict[str, Any]: 変換後の辞書データ。
        """
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.images:
            result["images"] = self.images
        return result


@dataclass
class OllamaPayload:
    """Ollama API (/api/chat) 送信用ペイロードモデル。"""

    model: str
    messages: list[ChatMessage]
    options: dict[str, Any] = field(default_factory=dict)
    stream: bool = False

    def to_dict(self) -> dict[str, Any]:
        """データモデルを辞書形式へ変換します。

        Returns:
            dict[str, Any]: 変換後の辞書データ。
        """
        return {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
            "options": self.options,
            "stream": self.stream,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OllamaPayload":
        """辞書データからインスタンスを構築します。

        Args:
            data (dict[str, Any]): 変換元の辞書データ。

        Returns:
            OllamaPayload: 構築されたインスタンス。
        """
        messages_raw = data.get("messages", [])
        messages = [
            ChatMessage(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                images=m.get("images"),
            )
            for m in messages_raw
        ]
        return cls(
            model=str(data.get("model", "")),
            messages=messages,
            options=dict(data.get("options", {})),
            stream=bool(data.get("stream", False)),
        )
