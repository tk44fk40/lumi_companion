"""音声解析・発言抽出サービスモジュール。

本モジュールは、Silero VAD および Faster-Whisper を用いた
音声認識処理サービスを提供します。
"""

import asyncio
from pathlib import Path

from faster_whisper import WhisperModel

from lumi_companion.config import settings
from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger
from lumi_companion.models.audio import AudioProcessResult, SubtitleSegment

logger = LumiLogger.get_logger(__name__)


class AudioProcessorService:
    """音声解析および発言字幕抽出サービス。"""

    def __init__(self, context: AppContext | None = None) -> None:
        """サービスの初期化を行います。

        Args:
            context (AppContext | None, optional): 設定コンテキスト。
        """
        self.context = context or AppContext()
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        """Faster-Whisper モデルを遅延読み込みで取得します。

        Returns:
            WhisperModel: ロードされた Whisper モデル。
        """
        if self._model is None:
            logger.info(
                "Whisperモデル (%s) を読み込み中...", settings.whisper_model_size
            )
            self._model = WhisperModel(
                model_size_or_path=settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
        return self._model

    def process_audio_sync(self, video_path: Path | str) -> AudioProcessResult:
        """同期処理で音声トラックから発言区間を抽出し文字起こしを行います。

        Args:
            video_path (Path | str): 入力動画ファイルパス。

        Returns:
            AudioProcessResult: 音声解析結果モデル。

        Raises:
            FileNotFoundError: 入力動画ファイルが存在しない場合。
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"動画ファイルが存在しません: {path}")

        model = self._get_model()
        logger.info("音声認識処理を開始します: %s", path)

        segments_raw, info = model.transcribe(
            str(path),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        segments: list[SubtitleSegment] = []
        for seg in segments_raw:
            cleaned_text = seg.text.strip()
            if cleaned_text:
                segments.append(
                    SubtitleSegment(
                        start=seg.start,
                        end=seg.end,
                        text=cleaned_text,
                    )
                )

        logger.info("発言抽出完了 (計 %d 件)", len(segments))
        return AudioProcessResult(
            segments=segments,
            duration_seconds=float(info.duration),
        )

    async def process_audio_async(
        self, video_path: Path | str
    ) -> AudioProcessResult:
        """非同期で音声トラックから発言区間を抽出します。

        Args:
            video_path (Path | str): 入力動画ファイルパス。

        Returns:
            AudioProcessResult: 音声解析結果モデル。
        """
        # 重い推論処理をバックグラウンドスレッドに委譲
        return await asyncio.to_thread(self.process_audio_sync, video_path)
