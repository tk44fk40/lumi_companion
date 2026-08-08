"""tests/test_post_processor.py

src/lumi_companion/audio/post_processor.py の単体テスト。
"""

from pathlib import Path

import pytest

from lumi_companion.audio.post_processor import TextPostProcessor
from lumi_companion.models import SubtitleSegment


class TestTextPostProcessor:
    """TextPostProcessor のテストクラス。"""

    def test_load_dictionary_yaml(self, tmp_path: Path) -> None:
        # Arrange
        yaml_path = tmp_path / "dict.yaml"
        yaml_path.write_text(
            "# サンプル辞書\nルミ: lumi_companion\n文字誤率: 文字誤り率\n",
            encoding="utf-8",
        )

        # Act
        dictionary = TextPostProcessor.load_dictionary(yaml_path)

        # Assert
        assert dictionary == {"ルミ": "lumi_companion", "文字誤率": "文字誤り率"}

    def test_load_dictionary_json(self, tmp_path: Path) -> None:
        # Arrange
        json_path = tmp_path / "dict.json"
        json_path.write_text(
            '{"ルミ": "lumi_companion", "文字誤率": "文字誤り率"}',
            encoding="utf-8",
        )

        # Act
        dictionary = TextPostProcessor.load_dictionary(json_path)

        # Assert
        assert dictionary == {"ルミ": "lumi_companion", "文字誤率": "文字誤り率"}

    def test_load_dictionary_not_found(self, tmp_path: Path) -> None:
        # Arrange
        non_existent = tmp_path / "non_existent.yaml"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            TextPostProcessor.load_dictionary(non_existent)

    def test_apply_to_text(self, tmp_path: Path) -> None:
        # Arrange
        yaml_path = tmp_path / "dict.yaml"
        # 長順優先置換の動作検証 ("ABC": "123", "AB": "99")
        yaml_path.write_text(
            "ルミ: lumi_companion\nAB: 99\nABC: 123\n",
            encoding="utf-8",
        )
        processor = TextPostProcessor(dictionary_path=yaml_path, remove_punct=False)

        # Act
        result = processor.apply_to_text("こんにちは、ルミ！ ABCとABのテスト。")

        # Assert
        # "ABC" が優先的に "123" に置換され、残りの "AB" が "99" に置換された上で数字全角化されること
        assert result == "こんにちは、lumi_companion！ １２３と９９のテスト。"

    def test_apply_to_segments(self, tmp_path: Path) -> None:
        # Arrange
        yaml_path = tmp_path / "dict.yaml"
        yaml_path.write_text("文字誤率: 文字誤り率\n", encoding="utf-8")
        processor = TextPostProcessor(dictionary_path=yaml_path, remove_punct=False)

        segments = [
            SubtitleSegment(start=0.0, end=2.0, text="文字誤率を計算します。"),
            SubtitleSegment(start=2.0, end=4.0, text="テスト継続中。"),
        ]

        # Act
        updated_segments = processor.apply_to_segments(segments)

        # Assert
        assert len(updated_segments) == 2
        assert updated_segments[0].text == "文字誤り率を計算します。"
        assert updated_segments[1].text == "テスト継続中。"

    def test_no_dictionary_path(self) -> None:
        # Arrange
        processor = TextPostProcessor(
            dictionary_path=None,
            to_hankaku=False,
            normalize_nums=False,
            lower=False,
            remove_punct=False,
        )

        # Act
        text = processor.apply_to_text("ルミのテスト")

        # Assert
        assert text == "ルミのテスト"

    def test_normalize_numbers_static(self) -> None:
        # Arrange
        raw = "第I章 十個のりんご ①番 １２３"

        # Act
        res = TextPostProcessor.normalize_numbers(raw)

        # Assert
        assert res == "第１章 １０個のりんご １番 １２３"

    def test_normalize_text_all_enabled(self) -> None:
        # Arrange
        processor = TextPostProcessor(
            dictionary_path=None,
            to_hankaku=True,
            normalize_nums=True,
            lower=True,
            remove_punct=True,
        )
        raw = "こんにちは、世界！ 第I章 100 ABC"

        # Act
        res = processor.apply_to_text(raw)

        # Assert
        # 句読点・記号・スペース除去、小文字化、数字正規化、半角化が適用されること
        assert res == "こんにちは世界第1章100abc"

    def test_load_dictionary_invalid_format(self, tmp_path: Path) -> None:
        # Arrange
        json_path = tmp_path / "invalid.json"
        json_path.write_text("[1, 2, 3]", encoding="utf-8")

        # Act & Assert
        with pytest.raises(ValueError, match="辞書ファイルの形式が正しくありません"):
            TextPostProcessor.load_dictionary(json_path)

    def test_apply_to_empty_text_and_segments(self) -> None:
        # Arrange
        processor = TextPostProcessor()

        # Act & Assert
        assert processor.apply_to_text("") == ""
        assert processor.apply_to_segments([]) == []

    def test_default_initialization(self) -> None:
        # Arrange & Act
        processor = TextPostProcessor()

        # Assert
        assert processor.remove_punct is False
        assert processor.to_hankaku is False
        assert processor.normalize_nums is True
        assert processor.lower is False

    def test_default_apply_to_text_preserves_punctuation(self) -> None:
        # Arrange
        processor = TextPostProcessor()
        raw = "こんにちは、世界！ テスト文ですね？ 第I章 100 ABC"

        # Act
        res = processor.apply_to_text(raw)

        # Assert
        # 句読点（、）、感嘆符（!）、疑問符（?）、空白が保持され、数字正規化が適用されること (小文字化・半角化はデフォルト無効)
        assert res == "こんにちは、世界！ テスト文ですね？ 第１章 １００ ABC"

    def test_normalize_text_keep_punct_with_newlines(self) -> None:
        # Arrange
        processor = TextPostProcessor(remove_punct=False)
        raw = "第一行\n\n第二行"

        # Act
        res = processor.normalize_text(raw)

        # Assert
        assert res == "第１行 第２行"

    def test_normalize_text_empty(self) -> None:
        # Arrange
        processor = TextPostProcessor()

        # Act & Assert
        assert processor.normalize_text("") == ""

    def test_normalize_text_normalize_only(self) -> None:
        # Arrange
        processor = TextPostProcessor(
            to_hankaku=True,
            normalize_nums=False,
            lower=False,
            remove_punct=False,
        )

        # Act & Assert
        assert processor.normalize_text("１２３ ＡＢＣ") == "123 ABC"

    def test_load_dictionary_other_extension_invalid(self, tmp_path: Path) -> None:
        # Arrange
        txt_path = tmp_path / "dict.txt"
        txt_path.write_text("- item1\n- item2", encoding="utf-8")

        # Act & Assert
        with pytest.raises(ValueError, match="辞書ファイルの形式が正しくありません"):
            TextPostProcessor.load_dictionary(txt_path)

    @pytest.mark.parametrize(
        "to_hankaku, normalize_nums, lower, remove_punct, expected",
        [
            # (to_hankaku, normalize_nums, lower, remove_punct)
            (False, False, False, False, "ＡＢＣ １２３ ① Ⅰ 十"),
            (False, False, False, True, "ＡＢＣ１２３①Ⅰ十"),
            (True, False, False, False, "ABC 123 1 I 十"),
            (True, False, False, True, "ABC1231I十"),
            (False, True, False, False, "ＡＢＣ １２３ １ １ １０"),
            (False, True, False, True, "ＡＢＣ１２３１１１０"),
            (False, False, True, False, "ａｂｃ １２３ ① ⅰ 十"),
            (False, False, True, True, "ａｂｃ１２３①ⅰ十"),
            (True, True, False, False, "ABC 123 1 1 10"),
            (True, True, False, True, "ABC1231110"),
            (True, False, True, False, "abc 123 1 i 十"),
            (True, False, True, True, "abc1231i十"),
            (False, True, True, False, "ａｂｃ １２３ １ １ １０"),
            (False, True, True, True, "ａｂｃ１２３１１１０"),
            (True, True, True, False, "abc 123 1 1 10"),
            (True, True, True, True, "abc1231110"),
        ],
    )
    def test_normalization_flags_independence(
        self,
        to_hankaku: bool,
        normalize_nums: bool,
        lower: bool,
        remove_punct: bool,
        expected: str,
    ) -> None:
        """各オプションが独立して機能し、意図しない副作用がないことを検証する。"""
        # Arrange
        processor = TextPostProcessor(
            dictionary_path=None,
            to_hankaku=to_hankaku,
            normalize_nums=normalize_nums,
            lower=lower,
            remove_punct=remove_punct,
        )
        raw_text = "ＡＢＣ １２３ ① Ⅰ 十"

        # Act
        result = processor.normalize_text(raw_text)

        # Assert
        assert result == expected
