"""AudioProcessor (audio/processor.py) の単体テストモジュール。"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lumi_companion.audio.processor import DEFAULT_INITIAL_PROMPT, AudioProcessor
from lumi_companion.models.audio import SubtitleSegment


def test_audio_processor_initialization_defaults() -> None:
    """AudioProcessor がデフォルト値で正しく初期化されることを検証します。"""
    # Arrange & Act
    processor = AudioProcessor()

    # Assert
    assert processor.model_size == "large-v3-turbo"
    assert processor.device == "auto"
    assert processor.compute_type == "default"
    assert processor.language == "ja"
    assert processor.beam_size == 5
    assert processor.initial_prompt == DEFAULT_INITIAL_PROMPT
    assert processor.condition_on_previous_text is False
    assert processor.vad_filter is True
    assert processor.vad_threshold == 0.35
    assert processor.vad_min_silence_duration_ms == 500
    assert processor.no_speech_threshold == 0.6
    assert processor.max_chars_per_second == 12.0
    assert processor.post_process_to_hankaku is False
    assert processor.post_process_normalize_nums is True
    assert processor.post_process_lower is False
    assert processor.post_process_remove_punct is False


def test_audio_processor_sanitize_segments_drops_no_speech_hallucination() -> None:
    """no_speech_prob が高数値（無音捏造）のセグメントがドロップされることを検証します。"""
    # Arrange
    processor = AudioProcessor(no_speech_threshold=0.6)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=0.0, end=2.0, text="正常なセグメント", no_speech_prob=0.1
        ),
        SimpleNamespace(start=2.5, end=4.0, text="無音捏造字幕", no_speech_prob=0.8),
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "正常なセグメント"


def test_audio_processor_sanitize_segments_drops_excessive_speech_rate() -> None:
    """0.3秒で10文字など、物理的発話限界（12文字/秒）を超える捏造セグメントがドロップされることを検証します。"""
    # Arrange
    processor = AudioProcessor(max_chars_per_second=12.0)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=0.0, end=2.0, text="こんにちは", no_speech_prob=0.1
        ),  # 2.0s 5文字 = 2.5文字/秒
        SimpleNamespace(
            start=2.0, end=2.3, text="はいはいはいはいはいはい", no_speech_prob=0.1
        ),  # 0.3s 10文字 = 33.3文字/秒 (捏造)
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "こんにちは"


def test_audio_processor_sanitize_segments_preserves_natural_repetition() -> None:
    """1.5秒で「はいはいはい」（6文字, 4文字/秒）などの自然な独り言の繰り返しが正常保持されることを検証します。"""
    # Arrange
    processor = AudioProcessor(max_chars_per_second=12.0)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=1.0, end=2.5, text="はいはいはい", no_speech_prob=0.05
        ),  # 1.5s 6文字 = 4.0文字/秒 (自然)
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "はいはいはい"


def test_audio_processor_sanitize_segments_preserves_short_utterances() -> None:
    """0.1秒の「はい」（2文字）など、4文字以下の短文は発話速度チェック対象外として安全に保持されることを検証します。"""
    # Arrange
    processor = AudioProcessor(max_chars_per_second=12.0)
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=0.0, end=0.1, text="はい", no_speech_prob=0.1
        ),  # 0.1s 2文字（短文保護対象）
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "はい"


def test_audio_processor_sanitize_segments_ignores_empty_or_whitespace_text() -> None:
    """空文字や空白のみのセグメントが安全に無視されることを検証します。"""
    # Arrange
    processor = AudioProcessor()
    fake_segments: list[Any] = [
        SimpleNamespace(start=0.0, end=1.0, text="", no_speech_prob=0.0),
        SimpleNamespace(start=1.0, end=2.0, text="   ", no_speech_prob=0.0),
        SimpleNamespace(start=2.0, end=3.0, text="正常発話", no_speech_prob=0.1),
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 1
    assert results[0].text == "正常発話"


def test_audio_processor_sanitize_segments_drops_inter_segment_repetition() -> None:
    """直前のセグメントと全く同じテキスト（または包含関係）が連続し、無音確率が高い場合にドロップされることを検証します。"""
    # Arrange
    processor = AudioProcessor()
    fake_segments: list[Any] = [
        SimpleNamespace(
            start=0.0,
            end=2.0,
            text="ご視聴ありがとうございました。",
            no_speech_prob=0.05,
        ),
        # 全く同じテキストで時間が進み、無音確率が > 0.1 の場合はループとみなす
        SimpleNamespace(
            start=3.0,
            end=5.0,
            text="ご視聴ありがとうございました。",
            no_speech_prob=0.15,
        ),
        # 包含されている場合もループとみなす
        SimpleNamespace(
            start=5.0, end=7.0, text="ありがとうございました。", no_speech_prob=0.2
        ),
        SimpleNamespace(start=7.0, end=9.0, text="正常な次の発話", no_speech_prob=0.01),
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 2
    assert results[0].text == "ご視聴ありがとうございました。"
    assert results[1].text == "正常な次の発話"


def test_audio_processor_get_word_time_dict() -> None:
    """_get_word_time が辞書型の入力に対して正しく動作することを検証します。"""
    # dict 型かつ対象キーが存在し、数値の場合
    assert AudioProcessor._get_word_time({"start": 1.23}, "start") == 1.23
    assert AudioProcessor._get_word_time({"start": 5}, "start") == 5.0
    # dict 型だが対象キーが存在しない、または数値でない場合
    assert AudioProcessor._get_word_time({"end": 4.56}, "start") is None
    assert AudioProcessor._get_word_time({"start": "invalid"}, "start") is None
    # 辞書型以外（オブジェクト）の場合でキーが存在しない、または数値でない場合
    obj = SimpleNamespace(start="not_a_number")
    assert AudioProcessor._get_word_time(obj, "start") is None
    assert AudioProcessor._get_word_time(obj, "end") is None


def test_audio_processor_sanitize_segments_drops_intra_segment_repetition() -> None:
    """1つのセグメント内で完全に同じフレーズが繰り返されている場合、重複部分が除去されることを検証します。"""
    # Arrange
    processor = AudioProcessor()
    fake_segments: list[Any] = [
        # ちょうど半分で繰り返されているパターン (ハルシネーションを模して compression_ratio を高くする)
        SimpleNamespace(
            start=0.0,
            end=2.0,
            text="あいうえおあいうえお",
            no_speech_prob=0.05,
            compression_ratio=2.5,
        ),
        SimpleNamespace(
            start=2.0,
            end=4.0,
            text="正常な発話です。",
            no_speech_prob=0.05,
            compression_ratio=1.0,
        ),
    ]

    # Act
    results: list[SubtitleSegment] = processor._sanitize_segments(fake_segments)

    # Assert
    assert len(results) == 2
    assert results[0].text == "あいうえお"
    assert results[1].text == "正常な発話です。"


@patch("lumi_companion.audio.processor.WhisperModel")
def test_audio_processor_process_sync_calls_transcribe_with_parameters(
    mock_whisper_model_class: MagicMock,
) -> None:
    """process_sync がコンストラクタ引数通りのパラメータで transcribe を呼び出すことを検証します。"""
    # Arrange
    mock_model_instance = MagicMock()
    mock_whisper_model_class.return_value = mock_model_instance
    mock_model_instance.transcribe.return_value = (
        [SimpleNamespace(start=0.0, end=1.5, text="テスト音声", no_speech_prob=0.1)],
        None,
    )

    processor = AudioProcessor(
        model_size="small",
        vad_threshold=0.3,
        initial_prompt="カスタムプロンプト",
    )

    with patch("pathlib.Path.exists", return_value=True):
        # Act
        results = processor.process_sync("dummy.mp4")

        # Assert
        assert len(results) == 1
        assert results[0].text == "テスト音声"
        mock_model_instance.transcribe.assert_called_once_with(
            "dummy.mp4",
            beam_size=5,
            language="ja",
            initial_prompt="カスタムプロンプト",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "threshold": 0.3,
            },
            no_speech_threshold=0.6,
            word_timestamps=True,
        )


@pytest.mark.asyncio
@patch("lumi_companion.audio.processor.WhisperModel")
async def test_audio_processor_process_async(
    mock_whisper_model_class: MagicMock,
) -> None:
    """process_async が非同期に process_sync を実行して結果を返却することを検証します。"""
    # Arrange
    mock_model_instance = MagicMock()
    mock_whisper_model_class.return_value = mock_model_instance
    mock_model_instance.transcribe.return_value = (
        [
            SimpleNamespace(
                start=0.0, end=1.5, text="非同期テスト音声", no_speech_prob=0.1
            )
        ],
        None,
    )

    processor = AudioProcessor()

    with patch("pathlib.Path.exists", return_value=True):
        # Act
        results = await processor.process_async("dummy_async.mp4")

        # Assert
        assert len(results) == 1
        assert results[0].text == "非同期テスト音声"


def test_audio_processor_sanitize_segments_logs_with_total_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """total_duration が指定された場合に進行状況（%）を含むログが出力されることを検証します。"""
    # Arrange
    processor = AudioProcessor()
    fake_segments: list[Any] = [
        SimpleNamespace(start=0.0, end=10.0, text="途中経過テスト", no_speech_prob=0.1),
    ]

    # Act
    with caplog.at_level("INFO"):
        results = processor._sanitize_segments(fake_segments, total_duration=100.0)

    # Assert
    assert len(results) == 1
    assert (
        "発言検出 [ 10.0s / 100.0s ( 10%)] [0.00s -> 10.00s]: 途中経過テスト"
        in caplog.text
    )


def test_audio_processor_split_segment_intelligently() -> None:
    """形態素解析(Janome)と文字数・句読点を用いて、指定文字数上限を超えない自然な範囲で分割・時間補間されることを検証します。"""
    # Arrange
    processor = AudioProcessor(max_segment_chars=10)
    # 19文字 (10秒)
    # 文字ごとの時間: 10 / 19 = 約0.526秒/文字
    text = "今日はいい天気です。お出かけしましょう。"
    seg = SubtitleSegment(start=0.0, end=10.0, text=text)

    # Act
    split_results = processor._split_segment_intelligently(seg)

    # Assert
    assert len(split_results) == 2

    # 1. "今日はいい天気です。" = 10文字
    # 2. "お出かけしましょう。" = 10文字
    # 合計: 20文字
    # 時間: 10.0 * (10 / 20) = 5.0
    assert split_results[0].text == "今日はいい天気です。"
    assert split_results[0].start == 0.0
    assert abs(split_results[0].end - 5.0) < 0.01

    # 2. "お出かけしましょう。" = 10文字
    assert split_results[1].text == "お出かけしましょう。"
    assert split_results[1].start == split_results[0].end
    assert split_results[1].end == 10.0


def test_audio_processor_split_segment_bunsetsu_soft_limit() -> None:
    """文節結合（接頭詞や話し言葉）、および句読点によるソフトリミット（早期分割）が機能することを検証します。"""
    # 20文字制限 -> ソフトリミット 14文字
    processor = AudioProcessor(max_segment_chars=20)

    text = "今日はとても天気が良いので、お散歩に出かけたい気分なんですよね。"
    seg = SubtitleSegment(start=0.0, end=33.0, text=text)

    split_results = processor._split_segment_intelligently(seg)

    # 期待される分割:
    # 1: "今日はとても天気が良いので、" (15文字) -> ソフトリミットによる早期分割
    # 2: "お散歩に出かけたい気分なんですよね。" (18文字) -> ハードリミット(20)未満なのでそのまま
    assert len(split_results) == 2
    assert split_results[0].text == "今日はとても天気が良いので、"
    assert split_results[1].text == "お散歩に出かけたい気分なんですよね。"

    # 按分の検証: 14文字 / 32文字 * 33.0 = 14.4375秒 -> 14.438秒
    assert abs(split_results[0].end - 14.438) < 0.01
    assert split_results[1].end == 33.0


def test_audio_processor_split_segment_fallback_call() -> None:
    """フォールバック処理（_split_segment_fallback）が直接呼ばれた場合に適切に分割されることを検証します。"""
    processor = AudioProcessor(max_segment_chars=15)

    # 通常の分割テスト
    long_text = "文字だけで構成された長いテキストです。分割されるはずです。"
    seg = SubtitleSegment(start=0.0, end=10.0, text=long_text)
    split_results = processor._split_segment_fallback(seg)
    assert len(split_results) > 1
    for s in split_results:
        assert len(s.text) <= 15
        assert s.start < s.end

    # 分割不要な短いテキストのテスト (max_chars 未満)
    short_text = "短いテキスト"
    short_seg = SubtitleSegment(start=0.0, end=2.0, text=short_text)
    assert processor._split_segment_fallback(short_seg) == [short_seg]

    # 文字列が空 (total_len == 0 になるケース)
    empty_seg = SubtitleSegment(start=0.0, end=2.0, text="")
    assert processor._split_segment_fallback(empty_seg) == [empty_seg]

    # 句読点が存在しないため強制的に文字数でカットされるケース
    no_punct_text = "あいうえおかきくけこさしすせそたちつてと"  # 20文字
    no_punct_seg = SubtitleSegment(start=0.0, end=10.0, text=no_punct_text)
    results_no_punct = processor._split_segment_fallback(no_punct_seg)
    assert len(results_no_punct) == 2
    assert results_no_punct[0].text == "あいうえおかきくけこさしすせそ"  # 15文字
    assert results_no_punct[1].text == "たちつてと"  # 5文字


def test_audio_processor_split_segment_intelligently_fallback() -> None:
    """Janome の Tokenizer 取得に失敗した場合にフォールバック処理にフォワードされることを検証します。"""
    processor = AudioProcessor(max_segment_chars=5)
    seg = SubtitleSegment(start=0.0, end=5.0, text="テストテキストです。")

    with patch.object(processor, "_get_tokenizer", return_value=None):
        with patch.object(
            processor, "_split_segment_fallback", return_value=[seg]
        ) as mock_fallback:
            results = processor._split_segment_intelligently(seg)
            assert results == [seg]
            mock_fallback.assert_called_once_with(seg)
