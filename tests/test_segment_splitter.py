"""SegmentSplitter の単体テストモジュール。"""

from unittest.mock import patch

from lumi_companion.audio.segment_splitter import SegmentSplitter
from lumi_companion.models.audio import SubtitleSegment


def test_segment_splitter_intelligently() -> None:
    """形態素解析(Janome)と文字数・句読点を用いて、指定文字数上限を超えない自然な範囲で分割・時間補間されることを検証します。"""
    # Arrange
    splitter = SegmentSplitter(max_segment_chars=10)
    # 19文字 (10秒)
    # 文字ごとの時間: 10 / 19 = 約0.526秒/文字
    text = "今日はいい天気です。お出かけしましょう。"
    seg = SubtitleSegment(start=0.0, end=10.0, text=text)

    # Act
    split_results = splitter.split_segment_intelligently(seg)

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


def test_segment_splitter_bunsetsu_soft_limit() -> None:
    """文節結合（接頭詞や話し言葉）、および句読点によるソフトリミット（早期分割）が機能することを検証します。"""
    # 20文字制限 -> ソフトリミット 14文字
    splitter = SegmentSplitter(max_segment_chars=20)

    text = "今日はとても天気が良いので、お散歩に出かけたい気分なんですよね。"
    seg = SubtitleSegment(start=0.0, end=33.0, text=text)

    split_results = splitter.split_segment_intelligently(seg)

    # 期待される分割:
    # 1: "今日はとても天気が良いので、" (15文字) -> ソフトリミットによる早期分割
    # 2: "お散歩に出かけたい気分なんですよね。" (18文字) -> ハードリミット(20)未満なのでそのまま
    assert len(split_results) == 2
    assert split_results[0].text == "今日はとても天気が良いので、"
    assert split_results[1].text == "お散歩に出かけたい気分なんですよね。"

    # 按分の検証: 14文字 / 32文字 * 33.0 = 14.4375秒 -> 14.438秒
    assert abs(split_results[0].end - 14.438) < 0.01
    assert split_results[1].end == 33.0


def test_segment_splitter_fallback_call() -> None:
    """フォールバック処理（split_segment_fallback）が直接呼ばれた場合に適切に分割されることを検証します。"""
    splitter = SegmentSplitter(max_segment_chars=15)

    # 通常の分割テスト
    long_text = "文字だけで構成された長いテキストです。分割されるはずです。"
    seg = SubtitleSegment(start=0.0, end=10.0, text=long_text)
    split_results = splitter.split_segment_fallback(seg)
    assert len(split_results) > 1
    for s in split_results:
        assert len(s.text) <= 15
        assert s.start < s.end

    # 分割不要な短いテキストのテスト (max_chars 未満)
    short_text = "短いテキスト"
    short_seg = SubtitleSegment(start=0.0, end=2.0, text=short_text)
    assert splitter.split_segment_fallback(short_seg) == [short_seg]

    # 文字列が空 (total_len == 0 になるケース)
    empty_seg = SubtitleSegment(start=0.0, end=2.0, text="")
    assert splitter.split_segment_fallback(empty_seg) == [empty_seg]

    # 句読点が存在しないため強制的に文字数でカットされるケース
    no_punct_text = "あいうえおかきくけこさしすせそたちつてと"  # 20文字
    no_punct_seg = SubtitleSegment(start=0.0, end=10.0, text=no_punct_text)
    results_no_punct = splitter.split_segment_fallback(no_punct_seg)
    assert len(results_no_punct) == 2
    assert results_no_punct[0].text == "あいうえおかきくけこさしすせそ"  # 15文字
    assert results_no_punct[1].text == "たちつてと"  # 5文字


def test_segment_splitter_intelligently_fallback() -> None:
    """Janome の Tokenizer 取得に失敗した場合にフォールバック処理にフォワードされることを検証します。"""
    splitter = SegmentSplitter(max_segment_chars=5)
    seg = SubtitleSegment(start=0.0, end=5.0, text="テストテキストです。")

    with patch.object(splitter, "_get_tokenizer", return_value=None):
        with patch.object(
            splitter, "split_segment_fallback", return_value=[seg]
        ) as mock_fallback:
            results = splitter.split_segment_intelligently(seg)
            assert results == [seg]
            mock_fallback.assert_called_once_with(seg)
