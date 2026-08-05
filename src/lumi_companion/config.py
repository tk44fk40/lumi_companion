"""アプリケーション環境設定モジュール。

本モジュールは、pydantic-settings を利用して .env や環境変数から
各種設定（Ollama 接続情報、Whisper 設定、デフォルトパス等）を読み込み・管理します。
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """lumi_companion アプリケーション全体の設定クラス。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama 設定 (Vision対応標準モデル: moondream, llava, llama3.2-vision 等)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "moondream"
    ollama_num_ctx: int = 4096

    # Whisper / 音声設定
    whisper_model_size: str = "base"  # small, medium, large-v3 等
    whisper_device: str = "auto"  # cuda / cpu
    whisper_compute_type: str = "default"  # float16, int8 等

    # デフォルトの入出力パス
    default_video_path: Path = Path("data/test_videos/sample.mp4")
    debug_output_dir: Path = Path("debug_output")


# グローバル設定インスタンス
settings = Settings()
