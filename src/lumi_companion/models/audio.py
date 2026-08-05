"""音声認識・発言抽出データモデルモジュール。

本モジュールは、字幕セグメントおよび音声処理結果を保持する
@dataclass データ構造を提供します。
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SubtitleSegment:
    """タイムスタンプ付き発言字幕データモデル。"""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        """データモデルを辞書形式へ変換します。

        Returns:
            dict[str, Any]: 変換後の辞書データ。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtitleSegment":
        """辞書データからインスタンスを構築します。

        Args:
            data (dict[str, Any]): 変換元の辞書データ。

        Returns:
            SubtitleSegment: 構築されたインスタンス。
        """
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")),
        )


@dataclass
class AudioProcessResult:
    """音声処理全体の出力結果データモデル。"""

    segments: list[SubtitleSegment]
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """データモデルを辞書形式へ変換します。

        Returns:
            dict[str, Any]: 変換後の辞書データ。
        """
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "duration_seconds": self.duration_seconds,
        }
