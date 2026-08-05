"""LumiApp オーケストレーター (DI対応) 統合テストモジュール。"""

from typing import Any

import pytest

from lumi_companion.core.app import LumiApp
from lumi_companion.core.context import AppContext


@pytest.mark.asyncio
async def test_lumi_app_with_dependency_injection(
    mock_context: AppContext,
    mock_audio_service: Any,
    mock_vision_service: Any,
    mock_prompt_service: Any,
    mock_llm_service: Any,
) -> None:
    """Mock サービスを DI 注入した LumiApp パイプライン実行を検証します。"""
    app = LumiApp(
        context=mock_context,
        audio_service=mock_audio_service,
        vision_service=mock_vision_service,
        prompt_service=mock_prompt_service,
        llm_service=mock_llm_service,
    )

    response = await app.run_full_pipeline(
        video_path="dummy_video.mp4",
        timestamp_seconds=10.0,
        model="moondream",
    )

    assert response.model == "moondream"
    assert "Mock の AI るみぽん！" in response.content
    assert response.done is True


@pytest.mark.asyncio
async def test_lumi_app_individual_steps(
    mock_context: AppContext,
    mock_audio_service: Any,
    mock_vision_service: Any,
) -> None:
    """個別の Step メソッド呼び出しを検証します。"""
    app = LumiApp(
        context=mock_context,
        audio_service=mock_audio_service,
        vision_service=mock_vision_service,
    )

    audio_res = await app.run_step1_audio("dummy.mp4")
    assert len(audio_res.segments) == 2
    assert audio_res.segments[0].text == "テスト発言1"

    frame_res = await app.run_step2_vision("dummy.mp4", 5.0)
    assert frame_res.height == 480
    assert frame_res.width == 854
    assert frame_res.timestamp_seconds == 5.0
