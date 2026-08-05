"""音声認識・発言抽出プロセッサモジュール。

本モジュールは、Faster-Whisper (WhisperModel) と Silero VAD フィルタを組み合わせて
動画・音声トラックから発言区間を自動検出し、高精度な文字起こしを行います。
"""

import asyncio
import logging
from pathlib import Path

from faster_whisper import WhisperModel

from lumi_companion.config import settings
from lumi_companion.models.audio import SubtitleSegment

logger = logging.getLogger(__name__)


class AudioProcessor:
    """動画・音声ファイルから発言抽出を行う処理クラス。"""

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        """AudioProcessor を初期化します。

        Args:
            model_size (str | None, optional): Whisper モデルサイズ。省略時は設定値。
            device (str | None, optional): 実行デバイス (cuda/cpu)。省略時は設定値。
            compute_type (str | None, optional): 計算精度 (float16/int8)。省略時は設定値。
        """
        self.model_size = model_size or settings.whisper_model_size
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        """WhisperModel インスタンスを遅延ロードで取得します。

        Returns:
            WhisperModel: ロード済みの WhisperModel オブジェクト。
        """
        if self._model is None:
            logger.info(
                "WhisperModel をロード中 (model=%s, device=%s)...",
                self.model_size,
                self.device,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def process_sync(self, file_path: Path | str) -> list[SubtitleSegment]:
        """同期的にファイル全体の音声を解析・文字起こしを実行します。

        Args:
            file_path (Path | str): 解析対象のメディアファイルパス。

        Returns:
            list[SubtitleSegment]: 抽出された字幕セグメントのリスト。

        Raises:
            FileNotFoundError: 対象ファイルが存在しない場合。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"メディアファイルが見つかりません: {path}")

        model = self._get_model()
        logger.info("発言抽出 (Whisper transcription) を開始: %s", path)

        # faster-whisper は内部で vad_filter=True により VAD フィルタリング可能
        segments, _info = model.transcribe(
            str(path),
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            language="ja",
        )

        results: list[SubtitleSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                results.append(
                    SubtitleSegment(
                        start=round(segment.start, 3),
                        end=round(segment.end, 3),
                        text=text,
                    )
                )

        logger.info("発言抽出完了 (%d 件のセグメントを検出)", len(results))
        return results

    async def process_async(self, file_path: Path | str) -> list[SubtitleSegment]:
        """イベントループをブロックせずに非同期に文字起こしを実行します。

        Args:
            file_path (Path | str): 解析対象のメディアファイルパス。

        Returns:
            list[SubtitleSegment]: 抽出された字幕セグメントのリスト。
        """
        return await asyncio.to_thread(self.process_sync, file_path)
