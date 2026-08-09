"""Step 5: タイムライン検証デバッグスクリプト。

本スクリプトは、動画全体の音声認識結果をもとに、一定間隔（例：3分ごと）および動画末尾の
タイムスタンプにおける「その時点までの発言文脈」と「その時点の画面フレーム」を抽出し、
LLMに連続して推論リクエストを送信することで、状況に応じた応答の変化を検証します。
各タイムスタンプごとの入力（音声・画像・プロンプト）と出力（応答）は debug_output/timeline/
に保存されます。
"""

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumi_companion.core.app import LumiApp
from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger
from lumi_companion.models.audio import SubtitleSegment
from lumi_companion.models.prompt import OllamaPayload
from lumi_companion.models.vision import FrameExtractResult


def filter_subtitles_by_timestamp(
    segments: list[SubtitleSegment], current_time: float
) -> list[SubtitleSegment]:
    """指定したタイムスタンプ以前に開始された字幕セグメントのみを抽出します。

    Args:
        segments (list[SubtitleSegment]): 元の字幕セグメントリスト。
        current_time (float): 現在のタイムスタンプ（秒）。

    Returns:
        list[SubtitleSegment]: 抽出された字幕セグメントリスト。
    """
    return [seg for seg in segments if seg.start <= current_time]


def generate_timestamps(
    duration_seconds: float, interval_seconds: float = 180.0
) -> list[float]:
    """検証用のタイムスタンプリストを生成します。

    0秒から始まり、指定間隔ごとにタイムスタンプを生成し、最後に動画末尾を追加します。
    （例: duration=400, interval=180 -> [0.0, 180.0, 360.0, 400.0]）

    Args:
        duration_seconds (float): 動画の総再生時間（秒）。
        interval_seconds (float, optional): 間隔（秒）。デフォルトは 180.0 (3分)。

    Returns:
        list[float]: タイムスタンプのリスト。
    """
    timestamps = []
    current = 0.0
    while current < duration_seconds:
        timestamps.append(current)
        current += interval_seconds
    if not timestamps or timestamps[-1] < duration_seconds:
        timestamps.append(duration_seconds)
    return timestamps


def save_timeline_debug_files(
    output_dir: Path,
    timestamp: float,
    subtitles: list[SubtitleSegment],
    frame_result: FrameExtractResult,
    payload: OllamaPayload,
    response: dict[str, Any],
) -> None:
    """タイムスタンプごとの検証用データ（字幕、画像、プロンプト、応答）をファイルに保存します。

    Args:
        output_dir (Path): 保存先ディレクトリ。
        timestamp (float): 現在のタイムスタンプ（秒）。
        subtitles (list[SubtitleSegment]): 抽出された字幕リスト。
        frame_result (FrameExtractResult): フレーム抽出結果。
        payload (OllamaPayload): LLMへのペイロード。
        response (dict[str, Any]): LLMからの応答生データ。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{int(timestamp):05d}s_"

    # 1. 抽出した発言
    subtitles_path = output_dir / f"{prefix}subtitles.json"
    with open(subtitles_path, "w", encoding="utf-8") as f:
        json.dump([seg.to_dict() for seg in subtitles], f, ensure_ascii=False, indent=2)

    # 2. 抽出した画像
    frame_path = output_dir / f"{prefix}frame.jpg"
    if frame_result.image_base64:
        image_bytes = base64.b64decode(frame_result.image_base64)
        with open(frame_path, "wb") as f:
            f.write(image_bytes)

    # 3. 構築されたプロンプト
    payload_path = output_dir / f"{prefix}payload.json"
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(
            payload.to_dict() if hasattr(payload, "to_dict") else vars(payload),
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 4. LLMの応答
    response_path = output_dir / f"{prefix}response.json"
    with open(response_path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)


async def main() -> None:
    """Step 5 タイムライン検証実行メイン関数。"""
    context = AppContext()
    logger = LumiLogger.get_logger("debug_step5_timeline", context)

    parser = argparse.ArgumentParser(
        description="Step 5: タイムラインに応じたLLM推論検証"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=context.default_video_path,
        help="対象の動画ファイルパス",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=context.ollama_model,
        help="Ollama モデル名",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=180.0,
        help="タイムスタンプ抽出間隔（秒）。デフォルトは 180秒（3分）",
    )
    args = parser.parse_args()

    app = LumiApp(context)

    # 1. 音声処理 (動画全体の文字起こし取得)
    audio_res = await app.run_step1_audio(args.video)
    duration = audio_res.duration_seconds
    logger.info("動画全体の長さ: %.1f秒", duration)

    # タイムスタンプ生成
    timestamps = generate_timestamps(duration, args.interval)
    logger.info("検証予定タイムスタンプ: %s", timestamps)

    timeline_dir = context.debug_output_dir / "timeline"
    logger.info("出力先ディレクトリ: %s", timeline_dir)

    for ts in timestamps:
        print("\n" + "=" * 50)
        logger.info(">>> タイムスタンプ %.1f秒 の検証を開始 <<<", ts)

        # 字幕のフィルタリング
        filtered_subtitles = filter_subtitles_by_timestamp(audio_res.segments, ts)
        logger.info("抽出された字幕数: %d件", len(filtered_subtitles))

        # 画像抽出
        frame_res = await app.run_step2_vision(args.video, ts)

        # プロンプト構築
        payload = app.prompt_service.build_payload(
            subtitles=filtered_subtitles,
            image_base64=frame_res.image_base64,
            model=args.model,
        )

        # LLM 推論
        response = await app.run_step4_llm(payload)

        # 応答表示
        print(f"🤖 【AI るみぽん！ の応答】 (Timestamp: {ts:.1f}s):")
        print("-" * 50)
        print(response.content.strip())
        print("=" * 50)

        # ファイル保存
        save_timeline_debug_files(
            output_dir=timeline_dir,
            timestamp=ts,
            subtitles=filtered_subtitles,
            frame_result=frame_res,
            payload=payload,
            response=response.raw_response,
        )
        logger.info("タイムスタンプ %.1f秒 のデバッグファイルを保存しました", ts)

    logger.info("Step 5 タイムライン検証処理が完了しました")


if __name__ == "__main__":
    asyncio.run(main())
