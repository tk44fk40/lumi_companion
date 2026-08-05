"""音声解析・発言抽出サービスモジュール。

本モジュールは、Silero VAD および Faster-Whisper を用いた
音声認識処理サービスを提供します。
"""

import asyncio
from pathlib import Path

from lumi_companion.audio.processor import AudioProcessor
from lumi_companion.config import settings
from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger
from lumi_companion.models.audio import AudioProcessResult

logger = LumiLogger.get_logger(__name__)


class AudioProcessorService:
    """音声解析および発言字幕抽出サービス。"""

    def __init__(
        self,
        context: AppContext | None = None,
        processor: AudioProcessor | None = None,
    ) -> None:
        """サービスの初期化を行います。

        Args:
            context (AppContext | None, optional): 設定コンテキスト。
            processor (AudioProcessor | None, optional): 音声認識プロセッサ。
        """
        self.context = context or AppContext()
        if processor is not None:
            self.processor = processor
        else:
            self.processor = AudioProcessor(
                model_size=settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                language=settings.whisper_language,
                beam_size=settings.whisper_beam_size,
                initial_prompt=settings.whisper_initial_prompt,
                condition_on_previous_text=settings.whisper_condition_on_previous_text,
                vad_filter=settings.whisper_vad_filter,
                vad_threshold=settings.whisper_vad_threshold,
                vad_min_silence_duration_ms=settings.whisper_vad_min_silence_duration_ms,
                no_speech_threshold=settings.whisper_no_speech_threshold,
                post_process_normalize=settings.whisper_post_process_normalize,
                post_process_normalize_nums=settings.whisper_post_process_normalize_nums,
                post_process_lower=settings.whisper_post_process_lower,
                post_process_remove_punct=settings.whisper_post_process_remove_punct,
                custom_dictionary_path=settings.custom_dictionary_path,
            )

    def process_audio_sync(self, video_path: Path | str) -> AudioProcessResult:
        """同期処理で音声トラックから発言区間を抽出し文字起こしを行います。

        Args:
            video_path (Path | str): 入力動画ファイルパス。

        Returns:
            AudioProcessResult: 音声解析結果モデル。

        Raises:
            FileNotFoundError: 入力動画ファイルが存在しない場合。
        """
        segments = self.processor.process_sync(video_path)
        # 最終セグメントの終了時刻または0.0を全体の概算秒数とする
        duration = segments[-1].end if segments else 0.0

        return AudioProcessResult(
            segments=segments,
            duration_seconds=duration,
        )

    async def process_audio_async(self, video_path: Path | str) -> AudioProcessResult:
        """非同期で音声トラックから発言区間を抽出します。

        Args:
            video_path (Path | str): 入力動画ファイルパス。

        Returns:
            AudioProcessResult: 音声解析結果モデル。
        """
        # 重い推論処理をバックグラウンドスレッドに委譲
        return await asyncio.to_thread(self.process_audio_sync, video_path)
