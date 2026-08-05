import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx

# lumi_companion モジュールのパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumi_companion.config import settings
from lumi_companion.llm.ollama import OllamaClient
from lumi_companion.prompt.builder import PromptBuilder

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger("debug_step4_ollama")


def load_payload(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_response(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: Ollama API にリクエストを投げて適切なAI応答が得られるか確認"
    )
    parser.add_argument(
        "--payload-json",
        type=Path,
        default=settings.debug_output_dir / "ollama_payload.json",
        help="Step 3 で出力された Ollama ペイロード JSON ファイルパス",
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
        default=None,
        help="上書き用 Ollama モデル名 (例: qwen2-vl:2b)",
    )
    args = parser.parse_args()

    payload_path: Path = args.payload_json
    output_dir: Path = args.output_dir

    logger.info("=== Step 4: ローカル Ollama リクエスト ＆ 応答確認処理開始 ===")

    # ペイロードの準備 (無ければデモ用に自前ビルド)
    payload: dict[str, Any]
    if payload_path.exists():
        logger.info("ペイロード JSON をロード: %s", payload_path)
        payload = load_payload(payload_path)
    else:
        logger.warning(
            "ペイロード JSON (%s) が無いため、その場で基本プロンプトを構築します。",
            payload_path,
        )
        payload = PromptBuilder.build_payload(
            user_prompt="配信を開始した配信者に対して、一言応援のコメントを返してください。"
        )

    # モデル名の上書き指定があれば適用
    if args.model:
        payload["model"] = args.model

    target_model = payload.get("model", settings.ollama_model)
    logger.info("使用予定モデル: %s", target_model)

    # Ollama クライアントで推論実行
    client = OllamaClient()
    try:
        response_data = await client.chat(payload)
    except (ConnectionError, RuntimeError, httpx.HTTPError) as e:
        logger.error("Ollama リクエスト処理中にエラーが発生しました: %s", e)
        sys.exit(1)

    # レスポンスの保存
    response_file = output_dir / "ollama_response.json"
    save_response(response_data, response_file)

    logger.info("=== 出力完了 ===")
    logger.info("応答 JSON 出力先: %s", response_file.resolve())

    # AI 生成テキストの抽出表示
    ai_message = response_data.get("message", {}).get("content", "(応答テキストなし)")

    print("\n" + "=" * 50)
    print(f"🤖 【AI るみぽん！ の応答コメント】 (モデル: {target_model}):")
    print("-" * 50)
    print(ai_message.strip())
    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
