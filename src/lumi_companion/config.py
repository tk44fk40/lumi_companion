from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """lumi_companion アプリケーション全体の設定クラス"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama 設定
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2-vl:2b"
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
