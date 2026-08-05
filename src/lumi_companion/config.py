"""アプリケーション環境設定モジュール。

本モジュールは、pydantic-settings を利用して .env や環境変数から
各種設定（Ollama 接続情報、Whisper 設定、デフォルトパス等）を読み込み・管理します。
"""

import ctypes
import os
from pathlib import Path
from typing import Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def setup_cuda_libraries() -> bool:
    """Pip インストールされた nvidia-cublas-cu12 等の CUDA 共有ライブラリパスを自動設定・事前ロードします。

    Returns:
        bool: CUDA 共有ライブラリのセットアップおよびロードに成功した場合 True、
              パッケージ非存在や失敗した場合は False。
    """
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib

        dirs: list[str] = []
        for mod in (nvidia.cublas.lib, nvidia.cudnn.lib):
            if hasattr(mod, "__path__") and mod.__path__:
                dirs.append(str(list(mod.__path__)[0]))
            elif getattr(mod, "__file__", None):
                dirs.append(os.path.dirname(str(mod.__file__)))

        if not dirs:
            return False

        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        existing_paths = [p for p in current_ld_path.split(":") if p]

        updated_paths: list[str] = []
        for d in dirs:
            if os.path.exists(d) and d not in updated_paths:
                updated_paths.append(d)

        for p in existing_paths:
            if p not in updated_paths:
                updated_paths.append(p)

        os.environ["LD_LIBRARY_PATH"] = ":".join(updated_paths)

        for lib_dir in dirs:
            if not os.path.exists(lib_dir):
                continue
            for file in os.listdir(lib_dir):
                if file.endswith(".so") or ".so." in file:
                    try:
                        ctypes.CDLL(os.path.join(lib_dir, file))
                    except Exception:
                        pass
        return True
    except Exception:
        return False


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
    whisper_model_size: str = "large-v3-turbo"  # small, medium, large-v3-turbo 等
    whisper_device: str = "auto"  # cuda / cpu
    whisper_compute_type: str = "default"  # float16, int8 等
    whisper_language: str = "ja"
    whisper_beam_size: int = 5
    whisper_initial_prompt: str = "えーっと、そうだな。今日は何をしようかな。とりあえずこれを試してみるか……よし、これでいこう。"
    whisper_condition_on_previous_text: bool = False
    whisper_vad_filter: bool = True
    whisper_vad_threshold: float = 0.35
    whisper_vad_min_silence_duration_ms: int = 500
    whisper_no_speech_threshold: float = 0.6

    # デフォルトの入出力パス
    default_video_path: Path = Path("data/test_videos/sample.mp4")
    debug_output_dir: Path = Path("debug_output")
    custom_dictionary_path: Path = Path("data/custom_dictionary.yaml")

    # 後処理・テキスト正規化設定 (デフォルト全て True)
    whisper_post_process_normalize: bool = True
    whisper_post_process_normalize_nums: bool = True
    whisper_post_process_lower: bool = True
    whisper_post_process_remove_punct: bool = True

    @model_validator(mode="after")
    def _validate_whisper_device(self) -> Self:
        """whisper_device が 'auto' の際、CUDA ライブラリのセットアップを試み、
        成功時は 'cuda'、失敗時は 'cpu' へ自動フォールバックします。

        Returns:
            Self: バリデーション・環境セットアップ済みの Settings インスタンス。
        """
        if self.whisper_device == "auto":
            cuda_ok = setup_cuda_libraries()
            self.whisper_device = "cuda" if cuda_ok else "cpu"
        return self


# グローバル設定インスタンス
settings = Settings()
