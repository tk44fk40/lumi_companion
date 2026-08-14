"""音声認識・発言抽出プロセッサモジュール。

本モジュールは、Faster-Whisper (WhisperModel) と各種フィルタを組み合わせて
動画・音声トラックから発言区間を自動検出し、高精度な文字起こしを行います。
設定モジュール (config) に直接依存せず、依存性注入 (DI) を前提とした疎結合設計となっています。
"""

import asyncio
import logging
from pathlib import Path

from faster_whisper import WhisperModel

from lumi_companion.audio.post_processor import TextPostProcessor
from lumi_companion.audio.segment_sanitizer import SegmentSanitizer
from lumi_companion.audio.segment_splitter import SegmentSplitter
from lumi_companion.audio.timing_adjuster import TimingAdjusterProtocol
from lumi_companion.models.audio import SubtitleSegment

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_PROMPT: str = (
    "日本語のゲーム実況・雑談配信です。話し言葉や感嘆詞を含めて正確に文字起こしします。"
)


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
        max_segment_chars: int = 25,
        post_process_to_hankaku: bool = False,
        post_process_normalize_nums: bool = True,
        post_process_lower: bool = False,
        post_process_remove_punct: bool = False,
        custom_dictionary_path: Path | None = None,
        word_timestamps: bool = True,
        timing_adjuster: TimingAdjusterProtocol | None = None,
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
            max_segment_chars (int): 1セグメントの許容最大文字数。デフォルト 25。
            post_process_to_hankaku (bool): 全角半角統一等の正規化を行うか。
            post_process_normalize_nums (bool): 数字正規化を行うか。
            post_process_lower (bool): 小文字化を行うか (デフォルト: False)。
            post_process_remove_punct (bool): 句読点・記号の除去を行うか。
            custom_dictionary_path (Path | None): 後処理置換辞書ファイルのパス。
            word_timestamps (bool): 単語レベルタイムスタンプを有効化し発声開始位置を自動補正するか。
            timing_adjuster (TimingAdjusterProtocol | None): 字幕表示タイミング補正処理 (DI)。
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
        self.max_segment_chars = max_segment_chars
        self.post_process_to_hankaku = post_process_to_hankaku
        self.post_process_normalize_nums = post_process_normalize_nums
        self.post_process_lower = post_process_lower
        self.post_process_remove_punct = post_process_remove_punct
        self.custom_dictionary_path = custom_dictionary_path
        self.word_timestamps = word_timestamps
        self.timing_adjuster = timing_adjuster
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

        raw_segments, info = model.transcribe(
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
            word_timestamps=self.word_timestamps,
        )

        total_duration = getattr(info, "duration", 0.0)
        detected_language = getattr(info, "language", self.language)
        logger.info(
            "音声トラック解析中... (検出言語=%s, 総再生時間=%.1fs)",
            detected_language,
            total_duration,
        )

        # 1. 無音除外 & 発声位置補正 & 重複除去
        sanitizer = SegmentSanitizer(
            no_speech_threshold=self.no_speech_threshold,
            max_chars_per_second=self.max_chars_per_second,
        )
        sanitized_segments = sanitizer.sanitize_segments(
            raw_segments, total_duration=total_duration
        )

        # 2. 置換辞書・正規化後処理の適用 (分割前に綺麗な日本語にする)
        post_processor = TextPostProcessor(
            dictionary_path=self.custom_dictionary_path,
            to_hankaku=self.post_process_to_hankaku,
            normalize_nums=self.post_process_normalize_nums,
            lower=self.post_process_lower,
            remove_punct=self.post_process_remove_punct,
        )
        normalized_segments = post_processor.apply_to_segments(sanitized_segments)

        # 3. 文節境界でのインテリジェント分割
        splitter = SegmentSplitter(max_segment_chars=self.max_segment_chars)
        split_segments: list[SubtitleSegment] = []
        for seg in normalized_segments:
            split_segments.extend(splitter.split_segment_intelligently(seg))

        # 4. 字幕表示タイミングの補正 (余韻パディング・最小表示時間・重複防止)
        if self.timing_adjuster is not None:
            final_segments = self.timing_adjuster.adjust_segments(
                split_segments, total_duration=total_duration
            )
        else:
            final_segments = split_segments

        logger.info("発言抽出完了 (%d 件のセグメントを検出)", len(final_segments))
        return final_segments

    async def process_async(self, file_path: Path | str) -> list[SubtitleSegment]:
        """イベントループをブロックせずに非同期に文字起こしを実行します。

        Args:
            file_path (Path | str): 解析対象のメディアファイルパス。

        Returns:
            list[SubtitleSegment]: 抽出された字幕セグメントのリスト。
        """
        return await asyncio.to_thread(self.process_sync, file_path)
