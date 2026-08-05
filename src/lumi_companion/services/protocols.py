"""パイプラインサービス抽象インターフェースプロトコルモジュール。

本モジュールは、依存性注入 (DI) による単体テストおよびコンポーネント分離を
実現するための Protocol 型定義を提供します。
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from lumi_companion.models import (
    AudioProcessResult,
    FrameExtractResult,
    OllamaPayload,
    OllamaResponse,
    SubtitleSegment,
)


@runtime_checkable
class AudioServiceProtocol(Protocol):
    """発言抽出・音声処理サービスのインターフェースプロトコル。"""

    async def process_audio_async(self, video_path: Path | str) -> AudioProcessResult:
        """非同期で動画の音声を解析し発言字幕リストを取得します。

        Args:
            video_path (Path | str): 入力動画ファイルパス。

        Returns:
            AudioProcessResult: 音声処理結果データモデル。
        """
        ...


@runtime_checkable
class VisionServiceProtocol(Protocol):
    """動画フレーム抽出・リサイズサービスのインターフェースプロトコル。"""

    async def extract_frame_async(
        self, video_path: Path | str, timestamp_seconds: float
    ) -> FrameExtractResult:
        """非同期で指定秒の画像フレームを抽出し Base64 データを取得します。

        Args:
            video_path (Path | str): 入力動画ファイルパス。
            timestamp_seconds (float): 抽出位置 (秒)。

        Returns:
            FrameExtractResult: フレーム抽出結果データモデル。
        """
        ...


@runtime_checkable
class PromptServiceProtocol(Protocol):
    """Ollama API 投入用プロンプト構築サービスのインターフェースプロトコル。"""

    def build_payload(
        self,
        subtitles: Sequence[SubtitleSegment] | None = None,
        image_base64: str | None = None,
        model: str | None = None,
        num_ctx: int | None = None,
    ) -> OllamaPayload:
        """字幕データおよび画像Base64から Ollama 用 JSON ペイロードを構築します。

        Args:
            subtitles (Sequence[SubtitleSegment] | None): 発言字幕リスト。
            image_base64 (str | None): 画像 Base64 データ。
            model (str | None): 推論モデル名。
            num_ctx (int | None): コンテキスト長。

        Returns:
            OllamaPayload: 構築された Ollama ペイロードモデル。
        """
        ...


@runtime_checkable
class LLMServiceProtocol(Protocol):
    """ローカル Ollama LLM 通信サービスのインターフェースプロトコル。"""

    async def chat_async(self, payload: OllamaPayload) -> OllamaResponse:
        """非同期で Ollama サーバーにリクエストを送信し推論応答を取得します。

        Args:
            payload (OllamaPayload): 送信用 Ollama ペイロードモデル。

        Returns:
            OllamaResponse: 推論応答データモデル。
        """
        ...
