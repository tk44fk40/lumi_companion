"""動画フレーム抽出サービスモジュール。

本モジュールは、OpenCV および PIL を使用した動画フレーム抽出、
リサイズ、および Base64 エンコード処理を提供します。
"""

import asyncio
from pathlib import Path

from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger
from lumi_companion.models.vision import FrameExtractResult
from lumi_companion.vision.extractor import FrameExtractor

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
        image = FrameExtractor.extract_frame_pil(
            video_path=video_path,
            timestamp_seconds=timestamp_seconds,
            max_width=self.context.max_image_width_px,
        )
        b64_str = FrameExtractor.extract_frame_base64(
            video_path=video_path,
            timestamp_seconds=timestamp_seconds,
            max_width=self.context.max_image_width_px,
            quality=self.context.jpeg_quality,
        )

        logger.info("フレーム抽出成功 (%dx%d)", image.width, image.height)
        return FrameExtractResult(
            timestamp_seconds=timestamp_seconds,
            width=image.width,
            height=image.height,
            image_base64=b64_str,
        )

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
