"""設定モジュール (config.py) の単体テスト。"""

import os

from _pytest.monkeypatch import MonkeyPatch

from lumi_companion.config import Settings, setup_cuda_libraries


def test_setup_cuda_libraries_returns_bool() -> None:
    """setup_cuda_libraries が例外なく実行され bool 値を返すことを検証します。"""
    # Arrange & Act
    result = setup_cuda_libraries()

    # Assert
    assert isinstance(result, bool)


def test_setup_cuda_libraries_preserves_existing_ld_library_path(
    monkeypatch: MonkeyPatch,
) -> None:
    """既存の LD_LIBRARY_PATH が保持され、二重追加（重複）されないことを検証します。"""
    # Arrange
    dummy_path = "/usr/local/custom_cuda/lib"
    monkeypatch.setenv("LD_LIBRARY_PATH", dummy_path)

    # Act
    setup_cuda_libraries()
    ld_path_1 = os.environ.get("LD_LIBRARY_PATH", "")

    # 冪等性のテスト (2回目呼び出し)
    setup_cuda_libraries()
    ld_path_2 = os.environ.get("LD_LIBRARY_PATH", "")

    # Assert
    assert dummy_path in ld_path_1
    assert ld_path_1 == ld_path_2  # 2回呼んでもパスが重複・増殖しないこと


def test_whisper_device_fallback_on_cuda_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    """setup_cuda_libraries が失敗した場合、whisper_device が 'cpu' にフォールバックすることを検証します。"""

    # Arrange: setup_cuda_libraries をモックして False を返させる
    def mock_setup_failure() -> bool:
        return False

    monkeypatch.setattr(
        "lumi_companion.config.setup_cuda_libraries", mock_setup_failure
    )

    # Act
    settings = Settings(whisper_device="auto")

    # Assert
    assert settings.whisper_device == "cpu"


def test_whisper_device_explicit_setting() -> None:
    """whisper_device に明示的な値が設定された場合、その値が変更されないことを検証します。"""
    # Arrange & Act
    settings = Settings(whisper_device="cpu")

    # Assert
    assert settings.whisper_device == "cpu"
