"""Step 2: フレーム抽出デバッグスクリプト。"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumi_companion.core.app import LumiApp
from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger
from lumi_companion.vision.extractor import FrameExtractor


async def main() -> None:
    """Step 2 フレーム抽出デバッグ実行メイン関数。"""
    context = AppContext()
    logger = LumiLogger.get_logger("debug_step2_vision", context)

    parser = argparse.ArgumentParser(
        description="Step 2: 動画から指定位置のフレームを抽出・リサイズして保存"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=context.default_video_path,
        help="対象の動画ファイルパス",
    )
    parser.add_argument(
        "--timestamp",
        type=float,
        default=10.0,
        help="抽出位置 (秒)",
    )
    args = parser.parse_args()

    video_path: Path = args.video
    if not video_path.exists():
        logger.error("動画ファイルが存在しません: %s", video_path)
        sys.exit(1)

    app = LumiApp(context)
    result = await app.run_step2_vision(video_path, args.timestamp)

    # 画像ファイルの保存
    out_image_path = context.get_output_path("extracted_frame.jpg")
    FrameExtractor.save_extracted_frame(
        video_path, args.timestamp, out_image_path, context.max_image_width_px
    )

    logger.info("画像出力先: %s", out_image_path.resolve())
    logger.info("Base64 長さ: %d バイト", len(result.image_base64))
    print(f"\n[Base64 プレビュー]: {result.image_base64[:60]}...")


if __name__ == "__main__":
    asyncio.run(main())
