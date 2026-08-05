"""Step 3: プロンプト構築デバッグスクリプト。

本スクリプトは、抽出した音声字幕テキストおよび抽出フレーム画像から
Ollama Chat API 互換の送信ペイロード JSON を構築・出力し、
入力データの整合性を確認するための非破壊デバッグツールです。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumi_companion.core.app import LumiApp
from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger
from lumi_companion.prompt.builder import PromptBuilder


async def main() -> None:
    """Step 3 プロンプト構築デバッグ実行メイン関数。"""
    context = AppContext()
    logger = LumiLogger.get_logger("debug_step3_prompt", context)

    parser = argparse.ArgumentParser(
        description="Step 3: 字幕と画像から Ollama 投入用 JSON ペイロードを構築"
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
    args = parser.parse_args()

    app = LumiApp(context)
    audio_res = await app.run_step1_audio(args.video)
    frame_res = await app.run_step2_vision(args.video, 10.0)
    payload = await app.run_step3_prompt(audio_res, frame_res, args.model)

    out_json = context.get_output_path("ollama_payload.json")
    PromptBuilder.save_payload_json(payload.to_dict(), out_json)

    logger.info("JSON 出力先: %s", out_json.resolve())
    print("\n--- ペイロードプレビュー ---")
    print(json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)[:300] + "...")


if __name__ == "__main__":
    asyncio.run(main())
