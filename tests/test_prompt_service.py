"""PromptBuilderService 単体テストモジュール。

本モジュールは、PromptBuilderService による Ollama 用ペイロード構築ロジックを検証します。
"""

from lumi_companion.core.context import AppContext
from lumi_companion.models import SubtitleSegment
from lumi_companion.services.prompt_service import PromptBuilderService


def test_prompt_builder_service_build_payload(mock_context: AppContext) -> None:
    """PromptBuilderService のペイロード構築ロジックを検証します。"""
    # Arrange (準備)
    service = PromptBuilderService(mock_context)
    subtitles = [SubtitleSegment(start=1.0, end=2.0, text="テスト発言")]

    # Act (実行)
    payload = service.build_payload(
        subtitles=subtitles,
        image_base64="dGVzdA==",
        model="moondream",
    )

    # Assert (検証)
    assert payload.model == "moondream"
    assert len(payload.messages) == 2
    assert payload.messages[1].images == ["dGVzdA=="]
    assert "テスト発言" in payload.messages[1].content
