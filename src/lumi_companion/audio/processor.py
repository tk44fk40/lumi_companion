"""音声認識・発言抽出プロセッサモジュール。

本モジュールは、Faster-Whisper (WhisperModel) と Silero VAD フィルタを組み合わせて
動画・音声トラックから発言区間を自動検出し、高精度な文字起こしを行います。
設定モジュール (config) に直接依存せず、依存性注入 (DI) を前提とした疎結合設計となっています。
"""

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path

from faster_whisper import WhisperModel

from lumi_companion.audio.post_processor import TextPostProcessor
from lumi_companion.models.audio import SubtitleSegment

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_PROMPT: str = "えーっと、そうだな。今日は何をしようかな。とりあえずこれを試してみるか……よし、これでいこう。"


class AudioProcessor:
    """動画・音声ファイルから発言抽出を行う処理クラス（設定に非依存）。"""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "ja",
        beam_size: int = 5,
        initial_prompt: str = DEFAULT_INITIAL_PROMPT,
        condition_on_previous_text: bool = False,
        vad_filter: bool = True,
        vad_threshold: float = 0.35,
        vad_min_silence_duration_ms: int = 500,
        no_speech_threshold: float = 0.6,
        max_chars_per_second: float = 12.0,
        post_process_normalize: bool = True,
        post_process_normalize_nums: bool = True,
        post_process_lower: bool = True,
        post_process_remove_punct: bool = True,
        custom_dictionary_path: Path | None = None,
    ) -> None:
        """AudioProcessor を初期化します。

        Args:
            model_size (str): Whisper モデルサイズ。
            device (str): 実行デバイス (cuda/cpu/auto)。
            compute_type (str): 計算精度 (float16/int8/default/float32)。
            language (str): 認識言語コード。
            beam_size (int): ビームサーチ幅。
            initial_prompt (str): 初期誘導プロンプト。
            condition_on_previous_text (bool): 直前文脈への依存フラグ。
            vad_filter (bool): VAD フィルタ有効化フラグ。
            vad_threshold (float): VAD 検出閾値。
            vad_min_silence_duration_ms (int): VAD 最小無音時間(ms)。
            no_speech_threshold (float): 無音判定閾値。
            max_chars_per_second (float): 物理的発話速度の許容上限（文字/秒）。
            post_process_normalize (bool): 全角半角統一等の正規化を行うか。
            post_process_normalize_nums (bool): 数字正規化を行うか。
            post_process_lower (bool): 小文字化を行うか。
            post_process_remove_punct (bool): 句読点・記号の除去を行うか。
            custom_dictionary_path (Path | None): 後処理置換辞書ファイルのパス。
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.condition_on_previous_text = condition_on_previous_text
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold
        self.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        self.no_speech_threshold = no_speech_threshold
        self.max_chars_per_second = max_chars_per_second
        self.post_process_normalize = post_process_normalize
        self.post_process_normalize_nums = post_process_normalize_nums
        self.post_process_lower = post_process_lower
        self.post_process_remove_punct = post_process_remove_punct
        self.custom_dictionary_path = custom_dictionary_path
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        """WhisperModel インスタンスを遅延ロードで取得します。

        Returns:
            WhisperModel: ロード済みの WhisperModel オブジェクト。
        """
        if self._model is None:
            logger.info(
                "WhisperModel をロード中 (model=%s, device=%s, compute_type=%s)...",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def _sanitize_segments(
        self,
        segments: Iterable[object],
    ) -> list[SubtitleSegment]:
        """Whisper 認識結果からハルシネーション（無音捏造・異常発話速度）を自動判定し除外・正規化します。

        Args:
            segments (Iterable[object]): Faster-Whisper から返された Segment オブジェクトのイテラブル。

        Returns:
            list[SubtitleSegment]: フィルタリングおよびクリーン化済みの字幕セグメントリスト。
        """
        results: list[SubtitleSegment] = []
        for segment in segments:
            text = getattr(segment, "text", "").strip()
            if not text:
                continue

            start = getattr(segment, "start", 0.0)
            end = getattr(segment, "end", 0.0)
            duration = max(end - start, 0.1)
            chars_per_sec = len(text) / duration

            # 1. 無音区間の捏造セグメント判定 (no_speech_prob チェック)
            no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
            if no_speech_prob > self.no_speech_threshold:
                logger.debug(
                    "無音捏造セグメントを自動ドロップ: %s (no_speech_prob=%.2f)",
                    text,
                    no_speech_prob,
                )
                continue

            # 2. 人間の解剖学的限界を超える異常発話速度の捏造判定 (4文字超のみ対象)
            if chars_per_sec > self.max_chars_per_second and len(text) > 4:
                logger.debug(
                    "異常発話速度の捏造セグメントを自動ドロップ: %s (%.1f文字/秒)",
                    text,
                    chars_per_sec,
                )
                continue

            results.append(
                SubtitleSegment(
                    start=round(start, 3),
                    end=round(end, 3),
                    text=text,
                )
            )

        return results

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

        raw_segments, _info = model.transcribe(
            str(path),
            beam_size=self.beam_size,
            language=self.language,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=self.condition_on_previous_text,
            vad_filter=self.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": self.vad_min_silence_duration_ms,
                "threshold": self.vad_threshold,
            },
            no_speech_threshold=self.no_speech_threshold,
        )

        results = self._sanitize_segments(raw_segments)
        post_processor = TextPostProcessor(
            dictionary_path=self.custom_dictionary_path,
            normalize=self.post_process_normalize,
            normalize_nums=self.post_process_normalize_nums,
            lower=self.post_process_lower,
            remove_punct=self.post_process_remove_punct,
        )
        results = post_processor.apply_to_segments(results)

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
