"""動画フレーム抽出データモデルモジュール。

本モジュールは、抽出・リサイズされた画像フレーム情報を保持する
@dataclass データ構造を提供します。
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FrameExtractResult:
    """フレーム抽出結果データモデル。"""

    timestamp_seconds: float
    width: int
    height: int
    image_base64: str

    def to_dict(self) -> dict[str, Any]:
        """データモデルを辞書形式へ変換します。

        Returns:
            dict[str, Any]: 変換後の辞書データ。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameExtractResult":
        """辞書データからインスタンスを構築します。

        Args:
            data (dict[str, Any]): 変換元の辞書データ。

        Returns:
            FrameExtractResult: 構築されたインスタンス。
        """
        return cls(
            timestamp_seconds=float(data.get("timestamp_seconds", 0.0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            image_base64=str(data.get("image_base64", "")),
        )
