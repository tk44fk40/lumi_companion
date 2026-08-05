"""Step 1/ASR 精度評価スクリプト (CER: Character Error Rate).

本スクリプトは、正解字幕データ (Reference) と AI 認識結果 (Hypothesis) の
テキストを比較し、文字誤り率 (CER) および置換・削除・挿入数のエラー内訳を
算出する評価ツールです。
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jiwer
import srt

from lumi_companion.audio.post_processor import TextPostProcessor


@dataclass(frozen=True)
class CerResult:
    """CER 計算結果を格納するデータクラス。

    Attributes:
        cer: 文字誤り率 (0.0 ～ 1.0 以上)
        substitutions: 置換エラー数 (S)
        deletions: 削除エラー数 (D)
        insertions: 挿入エラー数 (I)
        hits: 正致一致文字数 (H)
        reference_length: 正解テキストの総文字数 (N)
        reference_normalized: 正規化後の正解テキスト
        hypothesis_normalized: 正規化後の認識結果テキスト
    """

    cer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_length: int
    reference_normalized: str
    hypothesis_normalized: str


def normalize_numbers(text: str) -> str:
    """テキスト内の数字表現（漢数字、ローマ数字、全角数字等）を半角算用数字に統一正規化する。"""
    return TextPostProcessor.normalize_numbers(text)


def normalize_text(
    text: str, remove_punct: bool = True, normalize_nums: bool = True
) -> str:
    """評価用にテキストを正規化する。"""
    processor = TextPostProcessor(
        dictionary_path=None,
        normalize=True,
        normalize_nums=normalize_nums,
        lower=True,
        remove_punct=remove_punct,
    )
    return processor.normalize_text(text)


def load_transcript(file_path: Path) -> str:
    """指定されたパスのファイルから字幕・テキストを抽出して結合する。

    Args:
        file_path: 入力ファイルのパス (.txt, .json, .srt)

    Returns:
        結合された発言テキスト

    Raises:
        ValueError: サポートされていない拡張子の場合
        FileNotFoundError: ファイルが存在しない場合
    """
    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが存在しません: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")

    if suffix == ".srt":
        content = file_path.read_text(encoding="utf-8")
        subtitles = list(srt.parse(content))
        return " ".join([sub.content.replace("\n", " ") for sub in subtitles])

    if suffix == ".json":
        raw = file_path.read_text(encoding="utf-8")
        data: object = json.loads(raw)
        texts: list[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "text" in item:
                    texts.append(str(item["text"]))
        elif isinstance(data, dict) and "segments" in data:
            segments = data["segments"]
            if isinstance(segments, list):
                for seg in segments:
                    if isinstance(seg, dict) and "text" in seg:
                        texts.append(str(seg["text"]))
        return " ".join(texts)

    raise ValueError(
        f"サポートされていないファイル形式です ({suffix})。.txt, .srt, .json を指定してください。"
    )


def compute_cer(
    reference_text: str,
    hypothesis_text: str,
    remove_punct: bool = True,
    normalize_nums: bool = True,
) -> CerResult:
    """正解テキストと認識結果テキストから CER を計算する。

    Args:
        reference_text: 正解テキスト
        hypothesis_text: 認識結果テキスト
        remove_punct: 句読点除去を行うか (デフォルト: True)
        normalize_nums: 数字正規化を行うか (デフォルト: True)

    Returns:
        CerResult オブジェクト
    """
    ref_norm = normalize_text(
        reference_text, remove_punct=remove_punct, normalize_nums=normalize_nums
    )
    hyp_norm = normalize_text(
        hypothesis_text, remove_punct=remove_punct, normalize_nums=normalize_nums
    )

    if not ref_norm:
        cer_val = 0.0 if not hyp_norm else float(len(hyp_norm))
        return CerResult(
            cer=cer_val,
            substitutions=0,
            deletions=0,
            insertions=len(hyp_norm),
            hits=0,
            reference_length=0,
            reference_normalized=ref_norm,
            hypothesis_normalized=hyp_norm,
        )

    cer_val = float(jiwer.cer(ref_norm, hyp_norm))
    measures = jiwer.process_characters(ref_norm, hyp_norm)

    return CerResult(
        cer=cer_val,
        substitutions=int(measures.substitutions),
        deletions=int(measures.deletions),
        insertions=int(measures.insertions),
        hits=int(measures.hits),
        reference_length=len(ref_norm),
        reference_normalized=ref_norm,
        hypothesis_normalized=hyp_norm,
    )


def main() -> None:
    """CLI エントリポイント。"""
    parser = argparse.ArgumentParser(
        description="正解データと AI 認識結果から CER (文字誤り率) を計測"
    )
    parser.add_argument(
        "--ref",
        type=Path,
        required=True,
        help="正解データ (.txt, .json, .srt)",
    )
    parser.add_argument(
        "--hyp",
        type=Path,
        required=True,
        help="認識結果データ (.txt, .json, .srt)",
    )
    parser.add_argument(
        "--keep-punct",
        action="store_true",
        help="句読点や記号を除去せずに評価に含める",
    )
    parser.add_argument(
        "--dict",
        type=Path,
        default=None,
        help="後処理置換辞書データ (.yaml, .json)",
    )
    parser.add_argument(
        "--no-normalize-nums",
        action="store_true",
        help="漢数字やローマ数字の算用数字への自動正規化を無効化する",
    )
    args = parser.parse_args()

    try:
        ref_raw = load_transcript(args.ref)
        hyp_raw = load_transcript(args.hyp)
    except Exception as e:
        print(f"[エラー] ファイル読み込み失敗: {e}", file=sys.stderr)
        sys.exit(1)

    from lumi_companion.audio.post_processor import TextPostProcessor

    dict_path: Path | None = args.dict
    if dict_path and dict_path.exists():
        processor = TextPostProcessor(dict_path)
        hyp_raw = processor.apply_to_text(hyp_raw)

    result = compute_cer(
        ref_raw,
        hyp_raw,
        remove_punct=not args.keep_punct,
        normalize_nums=not args.no_normalize_nums,
    )

    print("\n=== CER (文字誤り率) 評価結果 ===")
    print(f"正解ファイル:   {args.ref}")
    print(f"認識結果ファイル: {args.hyp}")
    if dict_path:
        print(f"適用置換辞書:   {dict_path}")
    print(f"句読点除去:     {'有効' if not args.keep_punct else '無効'}")
    print("-" * 50)
    print(f"CER (文字誤り率):  {result.cer:.2%} ({result.cer:.4f})")
    print(f"正解文字数 (N):    {result.reference_length}")
    print(f"一致文字数 (H):    {result.hits}")
    print(
        f"置換エラー (S):    {result.substitutions:<4} (主な要因: 専門用語や同音異義語の誤認識)"
    )
    print(
        f"削除エラー (D):    {result.deletions:<4} (主な要因: 早口や無音・BGMに埋もれた声の聞き逃し)"
    )
    print(
        f"挿入エラー (I):    {result.insertions:<4} (主な要因: ノイズやハルシネーション / 言い淀み・幻聴の発生)"
    )
    print("-" * 50)

    # 主なエラー傾向に基づくアドバイス
    errors = {
        "置換 (S)": (
            result.substitutions,
            "専門用語プロンプト設定や辞書登録を検討してください。",
        ),
        "削除 (D)": (
            result.deletions,
            "VADの感度調整や無音区間パラメータの見直しを検討してください。",
        ),
        "挿入 (I)": (
            result.insertions,
            "ノイズ除去やハルシネーション抑制のプロンプトを検討してください。",
        ),
    }
    max_err_name, (max_err_count, advice) = max(errors.items(), key=lambda x: x[1][0])
    if max_err_count > 0:
        print(f"💡 分析ヒント: 【{max_err_name}】が最も多く発生しています。")
        print(f"   👉 {advice}")
    else:
        print("💡 分析ヒント: エラーは検出されませんでした！素晴らしい認識精度です。")
    print("=" * 50)


if __name__ == "__main__":
    main()
