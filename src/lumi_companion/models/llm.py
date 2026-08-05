"""LLM応答結果データモデルモジュール。

本モジュールは、Ollama から受信したレスポンス構造を保持する
@dataclass データ構造を提供します。
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OllamaResponse:
    """Ollama レスポンスデータモデル。"""

    model: str
    content: str
    done: bool = True
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """データモデルを辞書形式へ変換します。

        Returns:
            dict[str, Any]: 変換後の辞書データ。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OllamaResponse":
        """辞書データからインスタンスを構築します。

        Args:
            data (dict[str, Any]): 変換元の辞書データ。

        Returns:
            OllamaResponse: 構築されたインスタンス。
        """
        message = data.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""

        return cls(
            model=str(data.get("model", "")),
            content=content,
            done=bool(data.get("done", True)),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
            raw_response=data,
        )
