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
    """OllamaResponse の辞書解析および相互変換を検証します。"""
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

    d = resp.to_dict()
    assert d["model"] == "llava"
    assert d["content"] == "テスト応答"


def test_audio_process_result_dict_conversion() -> None:
    """AudioProcessResult の辞書変換を検証します。"""
    from lumi_companion.models import AudioProcessResult

    result = AudioProcessResult(
        segments=[SubtitleSegment(start=0.0, end=1.0, text="テスト")],
        duration_seconds=1.0,
    )
    d = result.to_dict()
    assert d["duration_seconds"] == 1.0
    assert len(d["segments"]) == 1
    assert d["segments"][0]["text"] == "テスト"


def test_chat_message_with_images() -> None:
    """ChatMessage の画像付き辞書変換を検証します。"""
    msg = ChatMessage(role="user", content="画像", images=["base64_data"])
    d = msg.to_dict()
    assert d["images"] == ["base64_data"]
