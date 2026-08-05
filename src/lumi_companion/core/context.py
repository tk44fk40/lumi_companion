"""アプリケーションコンテキストモジュール。

本モジュールは、システム全体で使用される定数、実行時フラグ、
および環境設定を管理するコンテキストクラスを提供します。
"""

from dataclasses import dataclass, field
from pathlib import Path

from lumi_companion.config import settings


@dataclass
class AppContext:
    """アプリケーション全体で共有されるコンテキスト・定数管理クラス。

    設定値や実行時パラメータ、パス情報を一元管理します。
    """

    # 外部サービス接続定数
    ollama_host: str = field(default_factory=lambda: settings.ollama_host)
    ollama_model: str = field(default_factory=lambda: settings.ollama_model)
    ollama_num_ctx: int = field(default_factory=lambda: settings.ollama_num_ctx)

    # 画像処理定数 (480p / 480px高さアスペクト比維持)
    max_image_height_px: int = 480
    jpeg_quality: int = 85

    # タイムアウト定数 (秒)
    http_timeout_seconds: float = 120.0
    download_timeout_seconds: float = 1800.0

    # パス定数
    default_video_path: Path = field(
        default_factory=lambda: settings.default_video_path
    )
    debug_output_dir: Path = field(default_factory=lambda: settings.debug_output_dir)

    # ログ出力パス
    log_file_path: Path = field(
        default_factory=lambda: settings.debug_output_dir / "app.log"
    )

    def get_output_path(self, filename: str) -> Path:
        """デバッグ出力ディレクトリ内のファイルパスを取得します。

        Args:
            filename (str): 出力ファイル名。

        Returns:
            Path: ディレクトリが確保された絶対パス。
        """
        # 出力先ディレクトリが存在しない場合は作成
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        return self.debug_output_dir / filename
