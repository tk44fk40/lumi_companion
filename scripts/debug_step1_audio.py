"""Step 1: 発言抽出デバッグスクリプト。

本スクリプトは、指定された動画ファイルから音声トラックを抽出し、
Silero VAD + Faster-Whisper による発言区間検出および文字起こし結果を
単体動作で確認するための非破壊デバッグツールです。
"""

import argparse
import asyncio
import sys
from pathlib import Path

# lumi_companion モジュールのパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumi_companion.audio.srt_exporter import SubtitleExporter
from lumi_companion.core.app import LumiApp
from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger


async def main() -> None:
    """Step 1 発言抽出デバッグ実行メイン関数。"""
    context = AppContext()
    logger = LumiLogger.get_logger("debug_step1_audio", context)

    parser = argparse.ArgumentParser(
        description="Step 1: 動画ファイルから音声を解析してタイムスタンプ付き字幕を抽出"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=context.default_video_path,
        help="対象の動画ファイルパス",
    )
    args = parser.parse_args()

    video_path: Path = args.video
    if not video_path.exists():
        logger.error("動画ファイルが存在しません: %s", video_path)
        sys.exit(1)

    app = LumiApp(context)
    result = await app.run_step1_audio(video_path)

    # 成果物の保存
    json_path = context.get_output_path("subtitles.json")
    srt_path = context.get_output_path("subtitles.srt")
    vtt_path = context.get_output_path("subtitles.vtt")

    SubtitleExporter.save_subtitles(result.segments, json_path)
    SubtitleExporter.save_subtitles(result.segments, srt_path)
    SubtitleExporter.save_subtitles(result.segments, vtt_path)

    logger.info("JSON 出力先: %s", json_path.resolve())
    logger.info("SRT 出力先:  %s", srt_path.resolve())
    logger.info("VTT 出力先:  %s", vtt_path.resolve())
    print(f"\n--- 抽出された発言字幕一覧 (全 {len(result.segments)} 件) ---")
    for seg in result.segments:
        ts = SubtitleExporter.format_timestamp(seg.start)
        print(f"[{ts}] {seg.text}")


if __name__ == "__main__":
    asyncio.run(main())
