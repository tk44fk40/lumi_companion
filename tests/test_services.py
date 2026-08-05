"""PromptBuilderService などのサービス単体テストモジュール。"""

from lumi_companion.core.context import AppContext
from lumi_companion.models import SubtitleSegment
from lumi_companion.services.prompt_service import PromptBuilderService


def test_prompt_builder_service_build_payload(mock_context: AppContext) -> None:
    """PromptBuilderService のペイロード構築ロジックを検証します。"""
    service = PromptBuilderService(mock_context)
    subtitles = [SubtitleSegment(start=1.0, end=2.0, text="テスト発言")]

    payload = service.build_payload(
        subtitles=subtitles,
        image_base64="dGVzdA==",
        model="moondream",
    )

    assert payload.model == "moondream"
    assert len(payload.messages) == 2
    assert payload.messages[1].images == ["dGVzdA=="]
    assert "テスト発言" in payload.messages[1].content
