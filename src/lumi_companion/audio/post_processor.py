"""音声認識結果のテキスト後処理・正規化モジュール。

本モジュールは、置換辞書 (.yaml / .json) による単語置換と、
Unicode (NFKC) 正規化、数字正規化 (漢数字・ローマ数字->算用数字)、
英小文字化、句読点・クリーン処理を一括で行うクラスを提供します。
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from lumi_companion.models import SubtitleSegment


class TextPostProcessor:
    """テキストおよび字幕セグメントに対する後処理・正規化クラス。"""

    def __init__(
        self,
        dictionary_path: Path | None = None,
        normalize: bool = True,
        normalize_nums: bool = True,
        lower: bool = True,
        remove_punct: bool = True,
    ) -> None:
        """TextPostProcessor を初期化します。

        Args:
            dictionary_path (Path | None): 置換辞書ファイル (.yaml / .json) のパス。
            normalize (bool): 全角半角統一 (NFKC) 等の正規化を行うか (デフォルト: True)。
            normalize_nums (bool): 数字正規化 (漢数字・ローマ数字->算用数字) を行うか (デフォルト: True)。
            lower (bool): 英小文字化を行うか (デフォルト: True)。
            remove_punct (bool): 句読点・記号・余白の除去を行うか (デフォルト: True)。
        """
        self.dictionary_path = dictionary_path
        self.normalize = normalize
        self.normalize_nums = normalize_nums
        self.lower = lower
        self.remove_punct = remove_punct

        self.dictionary: dict[str, str] = {}
        if dictionary_path and dictionary_path.exists():
            self.dictionary = self.load_dictionary(dictionary_path)

        # 競合防止のため文字数の長い順にソートしたキーワードリストを保持
        self._sorted_keys: list[str] = sorted(
            self.dictionary.keys(), key=len, reverse=True
        )

    @staticmethod
    def load_dictionary(file_path: Path) -> dict[str, str]:
        """置換辞書ファイル (.yaml / .yml / .json) を読み込みます。

        Args:
            file_path (Path): 辞書ファイルのパス。

        Returns:
            dict[str, str]: 置換マップ (置換前文字列 -> 置換後文字列)。

        Raises:
            FileNotFoundError: 指定されたパスにファイルが存在しない場合。
            ValueError: ファイル内容が文字列ペアの辞書形式でない場合。
        """
        if not file_path.exists():
            raise FileNotFoundError(f"辞書ファイルが存在しません: {file_path}")

        suffix = file_path.suffix.lower()
        content = file_path.read_text(encoding="utf-8")
        data: Any = None

        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content)
        elif suffix == ".json":
            data = json.loads(content)
        else:
            data = yaml.safe_load(content)

        if not isinstance(data, dict):
            raise ValueError(
                f"辞書ファイルの形式が正しくありません (dict 形式が必要です): {file_path}"
            )

        dictionary: dict[str, str] = {}
        for key, value in data.items():
            dictionary[str(key)] = str(value)

        return dictionary

    @staticmethod
    def normalize_numbers(text: str) -> str:
        """テキスト内の数字表現（漢数字、ローマ数字、全角数字等）を半角算用数字に統一正規化する。

        Args:
            text (str): 対象文字列。

        Returns:
            str: 正規化済みの文字列。
        """
        # 1. Unicode NFKC 正規化 (全角数字 -> 半角数字, 丸数字 ① -> 1 等)
        text = unicodedata.normalize("NFKC", text)

        # 2. ローマ数字 (アルファベット及び特殊記号) の長順変換
        roman_map = [
            ("VIII", "8"),
            ("VII", "7"),
            ("III", "3"),
            ("VI", "6"),
            ("IV", "4"),
            ("IX", "9"),
            ("II", "2"),
            ("V", "5"),
            ("X", "10"),
            ("I", "1"),
            ("Ⅷ", "8"),
            ("Ⅶ", "7"),
            ("Ⅲ", "3"),
            ("Ⅵ", "6"),
            ("Ⅳ", "4"),
            ("Ⅸ", "9"),
            ("Ⅱ", "2"),
            ("Ⅴ", "5"),
            ("Ⅹ", "10"),
            ("Ⅰ", "1"),
        ]
        for r_src, r_dst in roman_map:
            text = text.replace(r_src, r_dst)

        # 3. 漢数字のシンプル変換
        kanji_map = [
            ("十", "10"),
            ("九", "9"),
            ("八", "8"),
            ("七", "7"),
            ("六", "6"),
            ("五", "5"),
            ("四", "4"),
            ("三", "3"),
            ("二", "2"),
            ("一", "1"),
            ("〇", "0"),
            ("ゼロ", "0"),
        ]
        for k_src, k_dst in kanji_map:
            text = text.replace(k_src, k_dst)

        # 「101」など（十1 -> 11）の補正
        text = re.sub(r"10([1-9])", r"1\1", text)

        return text

    def normalize_text(self, text: str) -> str:
        """設定フラグに従ってテキストの正規化処理を適用します。

        Args:
            text (str): 対象文字列。

        Returns:
            str: 正規化済みの文字列。
        """
        if not text:
            return text

        result = text
        if self.normalize_nums:
            result = self.normalize_numbers(result)
        elif self.normalize:
            result = unicodedata.normalize("NFKC", result)

        if self.remove_punct:
            result = re.sub(r"[、、。！？!?\s\r\n]", "", result)
        else:
            result = re.sub(r"[\r\n]+", " ", result).strip()

        if self.lower:
            result = result.lower()

        return result

    def apply_to_text(self, text: str) -> str:
        """単一の文字列に対して正規化および置換辞書を適用します。

        Args:
            text (str): 処理対象の文字列。

        Returns:
            str: 後処理・正規化適用後の文字列。
        """
        if not text:
            return text

        result = text
        # 1. 単語置換辞書の適用 (文字数の長い順)
        if self.dictionary:
            for key in self._sorted_keys:
                val = self.dictionary[key]
                if key in result:
                    result = result.replace(key, val)

        # 2. テキスト正規化の適用
        if self.normalize or self.normalize_nums:
            result = self.normalize_text(result)

        return result

    def apply_to_segments(
        self, segments: list[SubtitleSegment]
    ) -> list[SubtitleSegment]:
        """字幕セグメントリストの各テキストに対して後処理・正規化を適用します。

        Args:
            segments (list[SubtitleSegment]): 元の字幕セグメントリスト。

        Returns:
            list[SubtitleSegment]: 後処理・正規化適用後の字幕セグメントリスト。
        """
        if not segments:
            return segments

        updated_segments: list[SubtitleSegment] = []
        for seg in segments:
            new_text = self.apply_to_text(seg.text)
            updated_segments.append(
                SubtitleSegment(start=seg.start, end=seg.end, text=new_text)
            )

        return updated_segments
