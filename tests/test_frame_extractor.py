"""FrameExtractor 単体テストモジュール。

本モジュールは、FrameExtractor による動画フレーム抽出、リサイズ計算、
Base64エンコード、JPEGバイト変換、保存および非同期処理を検証します。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from lumi_companion.vision.extractor import FrameExtractor


class TestFrameExtractor:
    """FrameExtractor の単体テストクラス。"""

    def test_extract_frame_pil_file_not_found(self) -> None:
        """存在しない動画ファイルを指定した場合に FileNotFoundError が発生することを検証します。"""
        with pytest.raises(FileNotFoundError, match="動画ファイルが見つかりません"):
            FrameExtractor.extract_frame_pil(
                "/nonexistent/video.mp4", timestamp_seconds=1.0
            )

    def test_extract_frame_pil_open_failed(self, tmp_path: Path) -> None:
        """動画ファイルを開けなかった場合に ValueError が発生することを検証します。"""
        dummy_video = tmp_path / "dummy.mp4"
        dummy_video.touch()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(ValueError, match="動画ファイルを開けませんでした"):
                FrameExtractor.extract_frame_pil(dummy_video, timestamp_seconds=1.0)

    def test_extract_frame_pil_read_failed(self, tmp_path: Path) -> None:
        """フレームの読み込みに失敗した場合に ValueError が発生することを検証します。"""
        dummy_video = tmp_path / "dummy.mp4"
        dummy_video.touch()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0.0  # fps <= 0 フォールバック検証
        mock_cap.read.return_value = (False, None)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(ValueError, match="読み込みに失敗しました"):
                FrameExtractor.extract_frame_pil(dummy_video, timestamp_seconds=1.0)
            mock_cap.release.assert_called_once()

    def test_extract_frame_pil_resize_and_aspect_ratio(self, tmp_path: Path) -> None:
        """1080p のフレームがアスペクト比を維持して 480p にリサイズされることを検証します。"""
        dummy_video = tmp_path / "dummy.mp4"
        dummy_video.touch()

        # 1920x1080 のダミーフレーム
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (True, dummy_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            image = FrameExtractor.extract_frame_pil(
                dummy_video, timestamp_seconds=2.0, max_height=480
            )
            assert image.height == 480
            assert image.width == int(1920 * (480 / 1080))
            mock_cap.set.assert_called_once_with(
                1, 60
            )  # CAP_PROP_POS_FRAMES = 1, 2.0 * 30 = 60
            mock_cap.release.assert_called_once()

    def test_extract_frame_pil_no_resize_when_small(self, tmp_path: Path) -> None:
        """高さが max_height 以下の場合はリサイズされないことを検証します。"""
        dummy_video = tmp_path / "dummy.mp4"
        dummy_video.touch()

        # 640x360 のダミーフレーム (高さ 360 <= max_height 480)
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (True, dummy_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            image = FrameExtractor.extract_frame_pil(
                dummy_video, timestamp_seconds=0.0, max_height=480
            )
            assert image.height == 360
            assert image.width == 640

    def test_extract_frame_bytes(self, tmp_path: Path) -> None:
        """抽出フレームが JPEG バイト列として返却されることを検証します。"""
        dummy_video = tmp_path / "dummy.mp4"
        dummy_video.touch()

        dummy_img = Image.new("RGB", (100, 100), color="blue")
        with patch.object(FrameExtractor, "extract_frame_pil", return_value=dummy_img):
            jpeg_bytes = FrameExtractor.extract_frame_bytes(
                dummy_video, timestamp_seconds=1.0
            )
            assert isinstance(jpeg_bytes, bytes)
            assert len(jpeg_bytes) > 0

    def test_extract_frame_base64(self, tmp_path: Path) -> None:
        """抽出フレームが Base64 文字列として返却されることを検証します。"""
        dummy_video = tmp_path / "dummy.mp4"
        dummy_video.touch()

        dummy_img = Image.new("RGB", (100, 100), color="red")
        with patch.object(FrameExtractor, "extract_frame_pil", return_value=dummy_img):
            b64_str = FrameExtractor.extract_frame_base64(
                dummy_video, timestamp_seconds=1.0
            )
            assert isinstance(b64_str, str)
            assert len(b64_str) > 0

    def test_save_extracted_frame(self, tmp_path: Path) -> None:
        """抽出フレームが指定パスへ JPEG 画像として保存されることを検証します。"""
        dummy_video = tmp_path / "dummy.mp4"
        dummy_video.touch()
        out_img_path = tmp_path / "frames" / "frame_1.jpg"

        dummy_img = Image.new("RGB", (200, 150), color="green")
        with patch.object(FrameExtractor, "extract_frame_pil", return_value=dummy_img):
            saved_path = FrameExtractor.save_extracted_frame(
                dummy_video, timestamp_seconds=1.0, output_path=out_img_path
            )
            assert saved_path == out_img_path
            assert out_img_path.exists()

    @pytest.mark.asyncio
    async def test_extract_frame_base64_async(self, tmp_path: Path) -> None:
        """非同期で Base64 フレームが取得できることを検証します。"""
        dummy_video = tmp_path / "dummy.mp4"
        dummy_video.touch()

        with patch.object(
            FrameExtractor, "extract_frame_base64", return_value="dummy_b64=="
        ) as mock_sync:
            result = await FrameExtractor.extract_frame_base64_async(
                dummy_video, timestamp_seconds=3.5
            )
            assert result == "dummy_b64=="
            mock_sync.assert_called_once_with(dummy_video, 3.5, 480)
