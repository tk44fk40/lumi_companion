"""PromptBuilderService および PromptBuilder 単体テストモジュール。

本モジュールは、PromptBuilder による Ollama 用ペイロード構築・JSON 保存ロジックを検証します。
"""

import json
from pathlib import Path

from lumi_companion.core.context import AppContext
from lumi_companion.models import SubtitleSegment
from lumi_companion.prompt.builder import PromptBuilder
from lumi_companion.services.prompt_service import PromptBuilderService


def test_prompt_builder_service_build_payload(mock_context: AppContext) -> None:
    """PromptBuilderService のペイロード構築ロジックを検証します。"""
    # Arrange
    service = PromptBuilderService(mock_context)
    subtitles = [SubtitleSegment(start=1.0, end=2.0, text="テスト発言")]

    # Act
    payload = service.build_payload(
        subtitles=subtitles,
        image_base64="dGVzdA==",
        model="moondream",
    )

    # Assert
    assert payload.model == "moondream"
    assert len(payload.messages) == 2
    assert payload.messages[1].images == ["dGVzdA=="]
    assert "テスト発言" in payload.messages[1].content


def test_prompt_builder_format_subtitles_text_empty() -> None:
    """字幕セグメントが空の場合のテキスト変換を検証します。"""
    formatted = PromptBuilder.format_subtitles_text([])
    assert formatted == "(直近の発言はありません)"


def test_prompt_builder_build_payload_defaults() -> None:
    """PromptBuilder.build_payload でパラメータを省略した場合のデフォルト値を検証します。"""
    # Act
    payload = PromptBuilder.build_payload(
        image_base64=None,
        subtitles=None,
        user_prompt=None,
    )

    # Assert
    assert "images" not in payload["messages"][1]
    assert "(直近の発言はありません)" in payload["messages"][1]["content"]
    assert "現在の配信画面と配信者の発言を踏まえて" in payload["messages"][1]["content"]


def test_prompt_builder_build_payload_with_image() -> None:
    """PromptBuilder.build_payload で画像添付がある場合を検証します。"""
    payload = PromptBuilder.build_payload(
        image_base64="dGVzdA==",
    )

    assert payload["messages"][1]["images"] == ["dGVzdA=="]


def test_prompt_builder_build_payload_custom_prompt() -> None:
    """PromptBuilder.build_payload でカスタムプロンプトを指定した場合を検証します。"""
    # Act
    payload = PromptBuilder.build_payload(
        user_prompt="カスタム指示テキスト",
        model="custom-model",
        num_ctx=4096,
    )

    # Assert
    assert payload["model"] == "custom-model"
    assert payload["options"]["num_ctx"] == 4096
    assert "カスタム指示テキスト" in payload["messages"][1]["content"]


def test_prompt_builder_save_payload_json(tmp_path: Path) -> None:
    """PromptBuilder.save_payload_json による JSON 保存機能を検証します。"""
    payload = {"model": "test-model", "messages": []}
    out_file = tmp_path / "payload.json"

    saved_path = PromptBuilder.save_payload_json(payload, out_file)

    assert saved_path.exists()
    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_data["model"] == "test-model"
