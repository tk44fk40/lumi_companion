import argparse
import asyncio
import logging
import sys
from pathlib import Path

# lumi_companion のモジュールをインポートできるようにパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumi_companion.audio.processor import AudioProcessor
from lumi_companion.audio.srt_exporter import SRTExporter
from lumi_companion.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger("debug_step1_audio")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: 動画ストリームから発言を抽出して JSON および SRT 字幕を出力"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=settings.default_video_path,
        help="対象の動画・音声ファイルパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.debug_output_dir,
        help="出力先ディレクトリルート",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        help="Whisper モデルサイズ (tiny, base, small, medium, large-v3)",
    )
    args = parser.parse_args()

    video_path: Path = args.video
    output_dir: Path = args.output_dir

    if not video_path.exists():
        logger.error(
            "指定されたファイルが見つかりません: %s", video_path.resolve()
        )
        logger.info(
            "提示: 'data/test_videos/' ディレクトリ内にテスト用動画 (sample.mp4 等) を配置してください。"
        )
        sys.exit(1)

    logger.info("=== Step 1: 発言抽出デバッグ処理開始 ===")
    logger.info("入力ファイル: %s", video_path)
    logger.info("出力先フォルダ: %s", output_dir)

    processor = AudioProcessor(model_size=args.model_size)
    segments = await processor.process_async(video_path)

    json_path = output_dir / "subtitles.json"
    srt_path = output_dir / "subtitles.srt"

    SRTExporter.save_json(segments, json_path)
    SRTExporter.save_srt(segments, srt_path)

    logger.info("=== 出力完了 ===")
    logger.info("JSON 出力先: %s", json_path.resolve())
    logger.info("SRT 出力先:  %s", srt_path.resolve())

    print(f"\n--- 抽出された発言字幕一覧 (全 {len(segments)} 件) ---")
    for seg in segments:
        time_str = SRTExporter.format_timestamp(seg.start)
        print(f"[{time_str}] {seg.text}")


if __name__ == "__main__":
    asyncio.run(main())
