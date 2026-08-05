import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# lumi_companion モジュールのパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumi_companion.audio.srt_exporter import SubtitleSegment
from lumi_companion.config import settings
from lumi_companion.prompt.builder import PromptBuilder
from lumi_companion.vision.extractor import FrameExtractor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger("debug_step3_prompt")


def load_subtitles(path: Path) -> list[SubtitleSegment]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        return [SubtitleSegment(**item) for item in data]


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: 発言字幕と抽出画像フレームから Ollama 投入用 JSON ペイロードを構築・保存"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=settings.default_video_path,
        help="対象の動画ファイルパス (画像抽出用)",
    )
    parser.add_argument(
        "--subtitles-json",
        type=Path,
        default=settings.debug_output_dir / "subtitles.json",
        help="Step 1 で出力された字幕 JSON ファイルパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.debug_output_dir,
        help="出力先ディレクトリルート",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.ollama_model,
        help="Ollama モデル名 (デフォルト: qwen2-vl:2b)",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=settings.ollama_num_ctx,
        help="動的コンテキスト長 num_ctx (デフォルト: 4096)",
    )
    parser.add_argument(
        "--timestamp",
        type=float,
        default=10.0,
        help="画像フレーム抽出位置 (秒)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    subtitles_path: Path = args.subtitles_json
    video_path: Path = args.video

    logger.info("=== Step 3: Ollama 投入用プロンプト JSON 構築処理開始 ===")

    # 1. 字幕データのロード (ファイルがなければダミーサンプル生成)
    subtitles: list[SubtitleSegment] = []
    if subtitles_path.exists():
        logger.info("字幕 JSON をロード: %s", subtitles_path)
        subtitles = load_subtitles(subtitles_path)
    else:
        logger.warning(
            "字幕 JSON (%s) が無いため、ダミーのテスト字幕を生成します。",
            subtitles_path,
        )
        subtitles = [
            SubtitleSegment(
                start=1.0,
                end=3.5,
                text="こんにちは！今日も配信を始めていくよ！",
            ),
            SubtitleSegment(
                start=4.0,
                end=8.2,
                text="まずは最初のボス戦に挑戦してみようと思うんだけど、勝てるかな？",
            ),
        ]

    # 2. 画像フレームの Base64 エンコード取得 (動画があれば抽出、無ければダミー設定)
    image_base64: str | None = None
    if video_path.exists():
        logger.info(
            "動画 (%s) の %.2f秒位置からフレームを抽出して Base64 変換します",
            video_path,
            args.timestamp,
        )
        image_base64 = await FrameExtractor.extract_frame_base64_async(
            video_path, args.timestamp
        )
    else:
        logger.warning(
            "動画ファイル (%s) が無いため、画像なしテキストのみでペイロードを構築します。",
            video_path,
        )

    # 3. Ollama ペイロードの構築
    payload = PromptBuilder.build_payload(
        image_base64=image_base64,
        subtitles=subtitles,
        model=args.model,
        num_ctx=args.num_ctx,
    )

    # 4. debug_output/ollama_payload.json に保存
    payload_file = output_dir / "ollama_payload.json"
    PromptBuilder.save_payload_json(payload, payload_file)

    logger.info("=== 出力完了 ===")
    logger.info("JSON 出力先: %s", payload_file.resolve())
    logger.info("指定モデル: %s", payload["model"])
    logger.info("指定 num_ctx: %s", payload["options"]["num_ctx"])

    print("\n--- Ollama JSON ペイロードプレビュー (要約) ---")
    print(f"Model: {payload['model']}")
    print(f"Options: {payload['options']}")
    print(
        f"Messages Count: {len(payload['messages'])} (System prompt + User prompt)"
    )
    if image_base64:
        print(f"Attached Images: 1 (Base64 length: {len(image_base64)} chars)")


if __name__ == "__main__":
    asyncio.run(main())
