"""Step 4: Ollama リクエストデバッグスクリプト。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumi_companion.core.app import LumiApp
from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger


def save_response_file(path: Path, data: dict[str, Any]) -> None:
    """レスポンス JSON データをファイルに書き込み保存します。

    Args:
        path (Path): 出力先パス。
        data (dict[str, Any]): レスポンス辞書データ。
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def main() -> None:
    """Step 4 Ollama 推論リクエストデバッグ実行メイン関数。"""
    context = AppContext()
    logger = LumiLogger.get_logger("debug_step4_ollama", context)

    parser = argparse.ArgumentParser(
        description="Step 4: Ollama に送信して推論応答を確認"
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
    response = await app.run_full_pipeline(args.video, model=args.model)

    out_json = context.get_output_path("ollama_response.json")
    save_response_file(out_json, response.raw_response)

    logger.info("応答 JSON 出力先: %s", out_json.resolve())
    print("\n" + "=" * 50)
    print(f"🤖 【AI るみぽん！ の応答コメント】 (モデル: {response.model}):")
    print("-" * 50)
    print(response.content.strip())
    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
