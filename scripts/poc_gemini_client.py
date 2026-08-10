"""Gemini API / Vertex AI (Google GenAI SDK) 単体機能・レスポンス検証用 PoC スクリプト。

本スクリプトは、Google 公式の google-genai SDK を利用し、
Gemini 2.5 Flash (gemini-2.5-flash) に対して画像および字幕テキスト入力を送信し、
相槌・応答品質、およびレスポンス所要時間 (レイテンシ) を検証するためのコンポーネントです。
Vertex AI モード (ADC) および Gemini Developer API モード (API Key) の両方に対応します。
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# 環境変数の読み込み (OS環境変数を最優先、未設定の場合のみ .env をロード)
load_dotenv()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("poc_gemini_client")

MODEL_NAME = "gemini-2.5-flash"
DEBUG_OUTPUT_DIR = Path("debug_output")
POC_RESULT_FILE = DEBUG_OUTPUT_DIR / "poc_gemini_result.json"

SYSTEM_INSTRUCTION = """あなたはゲーム実況・動画配信者に寄り添う AI コンパニオン「ルミ」です。
配信者の言葉（字幕テキスト）と現在のゲーム画面画像を確認し、以下のルールに従って相槌・感想コメントを短く生成してください。

【ルール】
1. 配信者のテンポを崩さないよう、15〜40文字程度で簡潔に返答してください。
2. 親しみやすいタメ口（話し言葉）でリアクションしてください。
3. 画面に変化がある場合は画面の要素（敵、アイテム、風景等）に触れてください。
4. 思考プロセスや挨拶などの無駄な前置きは含めず、コメント本文のみを出力してください。
"""

SAMPLE_SUBTITLES = """
[00:00:10.500] 配信者: あ、こんなところに宝箱がある！開けてみよう。
[00:00:12.800] 配信者: うわ、罠だった！敵が出てきた！
"""


def create_dummy_image(
    width: int = 640, height: int = 360, color: tuple[int, int, int] = (40, 40, 80)
) -> Image.Image:
    """テスト・動作確認用のシンプルなダミー画像を生成します。"""
    return Image.new("RGB", (width, height), color=color)


async def run_gemini_poc(
    api_key: str | None = None,
    project_id: str | None = None,
    image_input: Image.Image | Path | str | None = None,
    subtitles_text: str = SAMPLE_SUBTITLES,
) -> dict[str, Any]:
    """Gemini API を公式 SDK 経由で呼び出し、単体性能を測定・検証します。

    Args:
        api_key: Developer API 用の API キー (指定時は API Key モードで動作)。
        project_id: GCP プロジェクト ID (指定時は Vertex AI モードで動作)。
        image_input: 入力画像 (PIL Image、Path、または画像ファイルパス)。
        subtitles_text: テキスト字幕履歴の文字列。

    Returns:
        実行結果、レスポンス文字列、処理時間等を含む辞書オブジェクト。
    """
    vertex_project = (
        project_id
        or os.environ.get("VERTEXAI_PROJECT")
        or os.environ.get("GCP_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    key = api_key or os.environ.get("GEMINI_API_KEY")

    # 画像の準備
    pil_image: Image.Image
    if isinstance(image_input, Image.Image):
        pil_image = image_input
    elif isinstance(image_input, str | Path) and Path(image_input).exists():
        pil_image = Image.open(image_input)
    else:
        logger.info("入力画像が指定されていないため、テスト用ダミー画像を生成します。")
        pil_image = create_dummy_image()

    prompt_content = f"【配信者の最新発言・字幕履歴】\n{subtitles_text.strip()}\n\nコメントをお願いします。"

    # クライアントの初期化 (Vertex AI モード優先、次いで API Key モード)
    if vertex_project:
        logger.info("--- Vertex AI モードで初期化 (Project: %s) ---", vertex_project)
        client = genai.Client(
            vertexai=True,
            project=vertex_project,
            location=os.environ.get("VERTEXAI_LOCATION", "us-central1"),
        )
    elif key:
        masked_key = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else "***"
        logger.info(
            "--- Gemini Developer API モードで初期化 (API Key: %s) ---", masked_key
        )
        client = genai.Client(api_key=key)
    else:
        logger.info("--- Vertex AI (ADC デフォルト) モードで初期化 ---")
        client = genai.Client(
            vertexai=True,
            location=os.environ.get("VERTEXAI_LOCATION", "us-central1"),
        )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.7,
        max_output_tokens=300,
    )

    logger.info(
        "--- Gemini API (google-genai SDK) リクエスト送信開始 (Model: %s) ---",
        MODEL_NAME,
    )
    start_time = time.perf_counter()

    response_text = ""
    error_message: str | None = None
    elapsed_time_sec = 0.0

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=[pil_image, prompt_content],
                config=config,
            ),
        )
        elapsed_time_sec = time.perf_counter() - start_time
        response_text = response.text or ""
        logger.info("--- Gemini API 推論成功 ---")

    except Exception as exc:
        elapsed_time_sec = time.perf_counter() - start_time
        error_message = str(exc)
        logger.error("Gemini API エラー: %s", exc)

    result_data: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "elapsed_time_seconds": round(elapsed_time_sec, 3),
        "prompt_text": prompt_content,
        "response_text": response_text,
        "error": error_message,
        "status": "success" if response_text and not error_message else "error_or_mock",
    }

    # 結果を debug_output に保存
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(POC_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    logger.info("PoC 検証結果を %s に保存しました。", POC_RESULT_FILE)
    return result_data


async def main() -> None:
    """PoC スクリプトのエントリーポイント。"""
    print("==================================================")
    print(f"  Gemini API ({MODEL_NAME}) 単体検証 PoC (Vertex AI / Developer API 対応)")
    print("==================================================")

    result = await run_gemini_poc()

    print("\n--- 測定・検証結果 ---")
    print(f"ステータス: {result['status']}")
    print(f"モデル名  : {result['model']}")
    print(f"処理時間  : {result['elapsed_time_seconds']} 秒")
    print(f"生成コメント: {result['response_text']}")
    if result["error"]:
        print(f"エラー詳細  : {result['error']}")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())
