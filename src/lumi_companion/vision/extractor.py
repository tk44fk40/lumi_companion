"""動画フレーム抽出・画像処理モジュール。

本モジュールは、OpenCV および Pillow (PIL) を使用して
動画の指定タイムスタンプからフレーム画像を切り出し、
アスペクト比を保持したリサイズおよび Base64 変換を行います。
"""

import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class FrameExtractor:
    """動画から指定タイムスタンプのフレームを抽出・リサイズするクラス。"""

    @classmethod
    def extract_frame_pil(
        cls,
        video_path: Path | str,
        timestamp_seconds: float,
        max_width: int = 640,
    ) -> Image.Image:
        """指定秒位置のフレームを取得し、アスペクト比を維持してリサイズした PIL Image を返却します。

        Args:
            video_path (Path | str): 入力動画ファイルパス。
            timestamp_seconds (float): 抽出対象のタイムスタンプ (秒)。
            max_width (int, optional): 最大横幅 (px)。デフォルト 640。

        Returns:
            Image.Image: リサイズされた PIL Image オブジェクト。

        Raises:
            FileNotFoundError: 指定された動画ファイルが存在しない場合。
            ValueError: 動画ファイルが開けない、またはフレーム読み込みに失敗した場合。
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けませんでした: {path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0  # フォールバック値

            target_frame = int(timestamp_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(
                    f"タイムスタンプ {timestamp_seconds}秒 (フレーム {target_frame}) の読み込みに失敗しました。"
                )

            # BGR から RGB へ変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # アスペクト比を保持したままリサイズ
            width, height = image.size
            if width > max_width:
                new_height = int(height * (max_width / width))
                image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

            logger.info(
                "フレーム抽出完了: %s秒位置 (元のサイズ: %dx%d -> リサイズ後: %dx%d)",
                timestamp_seconds,
                width,
                height,
                image.width,
                image.height,
            )
            return image
        finally:
            cap.release()

    @classmethod
    def extract_frame_bytes(
        cls,
        video_path: Path | str,
        timestamp_seconds: float,
        max_width: int = 640,
        quality: int = 85,
    ) -> bytes:
        """抽出・リサイズしたフレームを JPEG バイト列として返却します。

        Args:
            video_path (Path | str): 入力動画ファイルパス。
            timestamp_seconds (float): 抽出対象のタイムスタンプ (秒)。
            max_width (int, optional): 最大横幅 (px)。デフォルト 640。
            quality (int, optional): JPEG 圧縮クオリティ。デフォルト 85。

        Returns:
            bytes: JPEG フォーマットのバイトデータ。
        """
        image = cls.extract_frame_pil(
            video_path, timestamp_seconds, max_width=max_width
        )
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()

    @classmethod
    def extract_frame_base64(
        cls,
        video_path: Path | str,
        timestamp_seconds: float,
        max_width: int = 640,
        quality: int = 85,
    ) -> str:
        """抽出・リサイズしたフレームを Base64 エンコード文字列として返却します。

        Args:
            video_path (Path | str): 入力動画ファイルパス。
            timestamp_seconds (float): 抽出対象のタイムスタンプ (秒)。
            max_width (int, optional): 最大横幅 (px)。デフォルト 640。
            quality (int, optional): JPEG 圧縮クオリティ。デフォルト 85。

        Returns:
            str: UTF-8 デコードされた Base64 文字列。
        """
        jpeg_bytes = cls.extract_frame_bytes(
            video_path, timestamp_seconds, max_width=max_width, quality=quality
        )
        return base64.b64encode(jpeg_bytes).decode("utf-8")

    @classmethod
    def save_extracted_frame(
        cls,
        video_path: Path | str,
        timestamp_seconds: float,
        output_path: Path | str,
        max_width: int = 640,
    ) -> Path:
        """抽出・リサイズしたフレームを指定パスへ JPEG 画像として保存します。

        Args:
            video_path (Path | str): 入力動画ファイルパス。
            timestamp_seconds (float): 抽出対象のタイムスタンプ (秒)。
            output_path (Path | str): 保存先画像パス。
            max_width (int, optional): 最大横幅 (px)。デフォルト 640。

        Returns:
            Path: 保存された画像ファイルの絶対パス。
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        image = cls.extract_frame_pil(
            video_path, timestamp_seconds, max_width=max_width
        )
        image.save(out_path, format="JPEG", quality=90)
        logger.info("フレーム画像を保存しました: %s", out_path.resolve())
        return out_path

    @classmethod
    async def extract_frame_base64_async(
        cls,
        video_path: Path | str,
        timestamp_seconds: float,
        max_width: int = 640,
    ) -> str:
        """非同期で Base64 フレーム文字列を取得します。

        Args:
            video_path (Path | str): 入力動画ファイルパス。
            timestamp_seconds (float): 抽出対象のタイムスタンプ (秒)。
            max_width (int, optional): 最大横幅 (px)。デフォルト 640。

        Returns:
            str: Base64 エンコード文字列。
        """
        return await asyncio.to_thread(
            cls.extract_frame_base64,
            video_path,
            timestamp_seconds,
            max_width,
        )
