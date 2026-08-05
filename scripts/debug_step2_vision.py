import argparse
import asyncio
import logging
import sys
from pathlib import Path

# lumi_companion モジュールのパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumi_companion.config import settings
from lumi_companion.vision.extractor import FrameExtractor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger("debug_step2_vision")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: 動画から指定位置のフレームを抽出・リサイズして保存・画像プレビュー"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=settings.default_video_path,
        help="対象の動画ファイルパス",
    )
    parser.add_argument(
        "--timestamp",
        type=float,
        default=10.0,
        help="フレーム抽出位置 (秒)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.debug_output_dir,
        help="出力先ディレクトリルート",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=640,
        help="リサイズ後の最大横幅px (デフォルト: 640)",
    )
    args = parser.parse_args()

    video_path: Path = args.video
    output_dir: Path = args.output_dir
    timestamp: float = args.timestamp

    if not video_path.exists():
        logger.error(
            "指定されたファイルが見つかりません: %s", video_path.resolve()
        )
        logger.info(
            "提示: 'data/test_videos/' ディレクトリ内にテスト用動画 (sample.mp4 等) を配置してください。"
        )
        sys.exit(1)

    logger.info("=== Step 2: フレーム抽出 ＆ リサイズデバッグ処理開始 ===")
    logger.info("入力ファイル: %s", video_path)
    logger.info("抽出タイムスタンプ: %.2f 秒", timestamp)

    output_image_path = output_dir / "extracted_frame.jpg"

    # 非同期スレッドでフレーム画像抽出・保存を実行
    saved_path = await asyncio.to_thread(
        FrameExtractor.save_extracted_frame,
        video_path,
        timestamp,
        output_image_path,
        args.max_width,
    )

    # Ollama へ送信用 Base64 文字列を取得して確認
    b64_str = await FrameExtractor.extract_frame_base64_async(
        video_path, timestamp, args.max_width
    )

    logger.info("=== 出力完了 ===")
    logger.info("画像出力先: %s", saved_path.resolve())
    logger.info("Base64 文字列長: %d バイト", len(b64_str))
    print(f"\n[Base64データ プレビュー (先頭60文字)]: {b64_str[:60]}...")


if __name__ == "__main__":
    asyncio.run(main())
