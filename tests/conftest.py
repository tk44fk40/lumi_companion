"""pytest 設定およびフィクスチャ定義モジュール。

本モジュールは、テストスイート全体で使用される Mock サービスおよび
コンテキストフィクスチャを提供します。
"""

import sys
from pathlib import Path

import pytest

# src ディレクトリを Python パスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumi_companion.core.context import AppContext
from lumi_companion.models import (
    AudioProcessResult,
    ChatMessage,
    FrameExtractResult,
    OllamaPayload,
    OllamaResponse,
    SubtitleSegment,
)


class MockAudioService:
    """テスト用 Mock 音声認識サービス。"""

    def __init__(self, context: AppContext | None = None) -> None:
        self.context = context or AppContext()

    async def process_audio_async(
        self, video_path: Path | str
    ) -> AudioProcessResult:
        """Mock の音声処理結果を返却します。"""
        return AudioProcessResult(
            segments=[
                SubtitleSegment(start=1.0, end=3.0, text="テスト発言1"),
                SubtitleSegment(start=4.0, end=6.0, text="テスト発言2"),
            ],
            duration_seconds=10.0,
        )


class MockVisionService:
    """テスト用 Mock フレーム抽出サービス。"""

    def __init__(self, context: AppContext | None = None) -> None:
        self.context = context or AppContext()

    async def extract_frame_async(
        self, video_path: Path | str, timestamp_seconds: float
    ) -> FrameExtractResult:
        """Mock のフレーム抽出結果を返却します。"""
        return FrameExtractResult(
            timestamp_seconds=timestamp_seconds,
            width=640,
            height=360,
            image_base64="dGVzdF9pbWFnZV9iYXNlNjQ=",
        )


class MockPromptService:
    """テスト用 Mock プロンプト構築サービス。"""

    def __init__(self, context: AppContext | None = None) -> None:
        self.context = context or AppContext()

    def build_payload(
        self,
        subtitles: list[SubtitleSegment] | None = None,
        image_base64: str | None = None,
        model: str | None = None,
        num_ctx: int | None = None,
    ) -> OllamaPayload:
        """Mock の Ollama ペイロードを構築します。"""
        return OllamaPayload(
            model=model or "moondream",
            messages=[
                ChatMessage(role="system", content="System Prompt"),
                ChatMessage(role="user", content="User Prompt"),
            ],
            options={"num_ctx": num_ctx or 4096},
        )


class MockLLMService:
    """テスト用 Mock LLM 通信サービス。"""

    def __init__(self, context: AppContext | None = None) -> None:
        self.context = context or AppContext()

    async def chat_async(self, payload: OllamaPayload) -> OllamaResponse:
        """Mock の LLM 応答を返却します。"""
        return OllamaResponse(
            model=payload.model,
            content="Mock の AI るみぽん！ レスポンスコメントです。",
            done=True,
            prompt_eval_count=100,
            eval_count=20,
        )


@pytest.fixture
def mock_context(tmp_path: Path) -> AppContext:
    """テスト用の独立した AppContext フィクスチャを提供します。"""
    return AppContext(
        debug_output_dir=tmp_path / "debug_output",
        log_file_path=tmp_path / "debug_output" / "app.log",
    )


@pytest.fixture
def mock_audio_service(mock_context: AppContext) -> MockAudioService:
    """MockAudioService フィクスチャを提供します。"""
    return MockAudioService(mock_context)


@pytest.fixture
def mock_vision_service(mock_context: AppContext) -> MockVisionService:
    """MockVisionService フィクスチャを提供します。"""
    return MockVisionService(mock_context)


@pytest.fixture
def mock_prompt_service(mock_context: AppContext) -> MockPromptService:
    """MockPromptService フィクスチャを提供します。"""
    return MockPromptService(mock_context)


@pytest.fixture
def mock_llm_service(mock_context: AppContext) -> MockLLMService:
    """MockLLMService フィクスチャを提供します。"""
    return MockLLMService(mock_context)
