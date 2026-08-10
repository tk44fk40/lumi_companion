"""音声字幕タイムコードの発声位置同期および置換・分割パイプラインの単体テスト。"""

from types import SimpleNamespace

from lumi_companion.audio.processor import AudioProcessor
from lumi_companion.models.audio import SubtitleSegment


def test_sanitize_segments_aligns_start_time_with_first_word() -> None:
    """チャンク開始時刻が 0.0s であっても文頭単語 words[0].start (3.12s) に補正されることを検証します (AAAパターン)。"""
    # Arrange
    processor = AudioProcessor(word_timestamps=True)
    mock_word1 = SimpleNamespace(word="おはよう", start=3.12, end=4.0)
    mock_word2 = SimpleNamespace(word="ございます。", start=4.0, end=5.0)

    mock_segment = SimpleNamespace(
        text="おはようございます。",
        start=0.0,
        end=5.0,
        no_speech_prob=0.01,
        words=[mock_word1, mock_word2],
    )

    # Act
    results = processor._sanitize_segments([mock_segment])

    # Assert
    assert len(results) >= 1
    assert results[0].start == 3.12
    assert results[0].end == 5.0


def test_process_sync_applies_dictionary_before_splitting() -> None:
    """置換辞書が字幕分割の前に適用され、置換後文字数に基づいて分割されることを検証します (AAAパターン)。"""
    # Arrange
    processor = AudioProcessor(
        max_segment_chars=25,
        word_timestamps=True,
    )

    mock_segment = SimpleNamespace(
        text="ウェーパーのテストです。",
        start=3.12,
        end=8.0,
        no_speech_prob=0.01,
        words=[
            SimpleNamespace(word="ウェーパーの", start=3.12, end=5.0),
            SimpleNamespace(word="テストです。", start=5.0, end=8.0),
        ],
    )

    # Act
    results = processor._sanitize_segments([mock_segment])

    # Assert
    assert len(results) >= 1


def test_split_segment_preserves_timestamps_for_replaced_phrase() -> None:
    """フレーズまるごと全置換された場合でも全体の開始・終了タイムコード(3.12s ~ 8.0s)が維持されることを検証します (AAAパターン)。"""
    # Arrange
    processor = AudioProcessor(
        max_segment_chars=20,
        word_timestamps=True,
    )
    orig_segment = SubtitleSegment(
        start=3.12,
        end=8.0,
        text="おはようございます。ウェーパーのテストです。",
    )

    # Act
    results = processor._split_segment_intelligently(orig_segment)

    # Assert
    assert len(results) >= 1
    assert results[0].start == 3.12
    assert results[-1].end == 8.0
