"""tests/test_evaluate_cer.py

scripts/evaluate_cer.py の単体テスト。
"""

import json
from pathlib import Path

import pytest

from scripts.evaluate_cer import (
    CerResult,
    compute_cer,
    load_transcript,
    normalize_text,
)


class TestNormalizeText:
    """テキスト正規化関数のテスト。"""

    def test_normalize_basic(self) -> None:
        # Arrange
        raw_text = "こんにちは、世界！ 123 ＡＢＣ"

        # Act
        result = normalize_text(raw_text, remove_punct=True)

        # Assert
        # 全角英数が半角化され小文字化、句読点やスペースが除去されること
        assert result == "こんにちは世界123abc"

    def test_normalize_keep_punct(self) -> None:
        # Arrange
        raw_text = "こんにちは、世界！"

        # Act
        result = normalize_text(raw_text, remove_punct=False)

        # Assert
        assert result == "こんにちは、世界!"

    def test_normalize_numbers(self) -> None:
        # Arrange
        # 漢数字 (一, 二, 三, 十)、ローマ数字 (Ⅰ, II, III)、丸数字 (①)
        raw_text = "第1章 第I章 第Ⅰ章 一つの選択肢 十個のりんご ①番"

        # Act
        result = normalize_text(raw_text, remove_punct=False, normalize_nums=True)

        # Assert
        # すべて 1, 2, 3, 10 等の算用数字に統一されること
        assert "第1章 第1章 第1章 1つの選択肢 10個のりんご 1番" in result


class TestLoadTranscript:
    """字幕・テキストファイル読み込み機能のテスト。"""

    def test_load_txt(self, tmp_path: Path) -> None:
        # Arrange
        txt_path = tmp_path / "sample.txt"
        txt_path.write_text("本日は晴天なり。\nテスト文です。", encoding="utf-8")

        # Act
        content = load_transcript(txt_path)

        # Assert
        assert "本日は晴天なり。" in content
        assert "テスト文です。" in content

    def test_load_json(self, tmp_path: Path) -> None:
        # Arrange
        json_path = tmp_path / "subtitles.json"
        data = [
            {"start": 0.0, "end": 2.5, "text": "第一文です。"},
            {"start": 2.5, "end": 5.0, "text": "第二文です。"},
        ]
        json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        # Act
        content = load_transcript(json_path)

        # Assert
        assert content == "第一文です。 第二文です。"

    def test_load_srt(self, tmp_path: Path) -> None:
        # Arrange
        srt_path = tmp_path / "subtitles.srt"
        srt_content = (
            "1\n00:00:00,000 --> 00:00:02,500\n第一文です。\n\n"
            "2\n00:00:02,500 --> 00:00:05,000\n第二文です。\n"
        )
        srt_path.write_text(srt_content, encoding="utf-8")

        # Act
        content = load_transcript(srt_path)

        # Assert
        assert content == "第一文です。 第二文です。"

    def test_load_unsupported_format(self, tmp_path: Path) -> None:
        # Arrange
        dummy_path = tmp_path / "sample.invalid"
        dummy_path.write_text("test", encoding="utf-8")

        # Act & Assert
        with pytest.raises(ValueError, match="サポートされていないファイル形式"):
            load_transcript(dummy_path)


class TestComputeCer:
    """CER 計算ロジックのテスト。"""

    def test_compute_cer_exact_match(self) -> None:
        # Arrange
        ref = "本日は晴天なり。"
        hyp = "本日は晴天なり。"

        # Act
        res = compute_cer(ref, hyp, remove_punct=True)

        # Assert
        assert isinstance(res, CerResult)
        assert res.cer == 0.0
        assert res.substitutions == 0
        assert res.deletions == 0
        assert res.insertions == 0
        assert res.reference_length == 7

    def test_compute_cer_with_errors(self) -> None:
        # Arrange
        ref = "本日は晴天なり"  # N = 7
        hyp = "本日は青天ですなり"  # 「晴」->「青」(S=1)、「です」(I=2)

        # Act
        res = compute_cer(ref, hyp, remove_punct=True)

        # Assert
        assert res.substitutions == 1
        assert res.insertions == 2
        assert res.deletions == 0
        assert res.reference_length == 7
        assert res.cer == pytest.approx((1 + 2) / 7)
