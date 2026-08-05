"""動画フレーム抽出サービスモジュール。

本モジュールは、OpenCV および PIL を使用した動画フレーム抽出、
リサイズ、および Base64 エンコード処理を提供します。
"""

import asyncio
import base64
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image

from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger
from lumi_companion.models.vision import FrameExtractResult

logger = LumiLogger.get_logger(__name__)


class VisionExtractorService:
    """動画フレーム抽出およびリサイズサービス。"""

    def __init__(self, context: AppContext | None = None) -> None:
        """サービスの初期化を行います。

        Args:
            context (AppContext | None, optional): 設定コンテキスト。
        """
        self.context = context or AppContext()

    def extract_frame_sync(
        self, video_path: Path | str, timestamp_seconds: float
    ) -> FrameExtractResult:
        """同期処理で指定秒位置の画像フレームを取得し Base64 変換します。

        Args:
            video_path (Path | str): 入力動画ファイルパス。
            timestamp_seconds (float): 抽出位置 (秒)。

        Returns:
            FrameExtractResult: フレーム抽出結果モデル。

        Raises:
            FileNotFoundError: 入力動画ファイルが存在しない場合。
            ValueError: フレーム読み込みに失敗した場合。
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"動画ファイルが存在しません: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けませんでした: {path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            target_frame = int(timestamp_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(
                    f"タイムスタンプ {timestamp_seconds}秒の読み込みに失敗しました。"
                )

            # RGB 変換とリサイズ
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            max_width = self.context.max_image_width_px
            if image.width > max_width:
                new_height = int(image.height * (max_width / image.width))
                image = image.resize(
                    (max_width, new_height), Image.Resampling.LANCZOS
                )

            # JPEG エンコードおよび Base64 化
            buffer = BytesIO()
            image.save(
                buffer, format="JPEG", quality=self.context.jpeg_quality
            )
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            logger.info("フレーム抽出成功 (%dx%d)", image.width, image.height)
            return FrameExtractResult(
                timestamp_seconds=timestamp_seconds,
                width=image.width,
                height=image.height,
                image_base64=b64_str,
            )
        finally:
            cap.release()

    async def extract_frame_async(
        self, video_path: Path | str, timestamp_seconds: float
    ) -> FrameExtractResult:
        """非同期で指定秒位置の画像フレームを取得します。

        Args:
            video_path (Path | str): 入力動画ファイルパス。
            timestamp_seconds (float): 抽出位置 (秒)。

        Returns:
            FrameExtractResult: フレーム抽出結果モデル。
        """
        # 重い画像デコード・処理をバックグラウンドスレッドへ委譲
        return await asyncio.to_thread(
            self.extract_frame_sync, video_path, timestamp_seconds
        )
