"""@dataclass データモデル単体テストモジュール。"""

from lumi_companion.models import (
    ChatMessage,
    FrameExtractResult,
    OllamaPayload,
    OllamaResponse,
    SubtitleSegment,
)


def test_subtitle_segment_dict_conversion() -> None:
    """SubtitleSegment の辞書相互変換を検証します。"""
    seg = SubtitleSegment(start=1.5, end=4.0, text="こんにちは")
    d = seg.to_dict()
    assert d["start"] == 1.5
    assert d["text"] == "こんにちは"

    reconstructed = SubtitleSegment.from_dict(d)
    assert reconstructed.start == 1.5
    assert reconstructed.text == "こんにちは"


def test_frame_extract_result_dict_conversion() -> None:
    """FrameExtractResult の辞書相互変換を検証します。"""
    res = FrameExtractResult(
        timestamp_seconds=10.0, width=640, height=360, image_base64="abc=="
    )
    d = res.to_dict()
    assert d["width"] == 640

    reconstructed = FrameExtractResult.from_dict(d)
    assert reconstructed.height == 360
    assert reconstructed.image_base64 == "abc=="


def test_ollama_payload_dict_conversion() -> None:
    """OllamaPayload の辞書相互変換を検証します。"""
    payload = OllamaPayload(
        model="moondream",
        messages=[ChatMessage(role="user", content="Hello")],
        options={"num_ctx": 4096},
    )
    d = payload.to_dict()
    assert d["model"] == "moondream"
    assert len(d["messages"]) == 1

    reconstructed = OllamaPayload.from_dict(d)
    assert reconstructed.model == "moondream"
    assert reconstructed.messages[0].content == "Hello"


def test_ollama_response_dict_conversion() -> None:
    """OllamaResponse の辞書解析を検証します。"""
    raw_data = {
        "model": "llava",
        "message": {"role": "assistant", "content": "テスト応答"},
        "done": True,
        "prompt_eval_count": 50,
    }
    resp = OllamaResponse.from_dict(raw_data)
    assert resp.model == "llava"
    assert resp.content == "テスト応答"
    assert resp.prompt_eval_count == 50
