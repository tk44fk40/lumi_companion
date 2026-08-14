"""LumiLogger 単体テストモジュール。

本モジュールは、LumiLogger によるコンソールおよびファイル出力ハンドラーの設定動作を検証します。
"""

import logging
from pathlib import Path

import pytest

from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import FlushFileHandler, FlushStreamHandler, LumiLogger


def test_logger_get_logger_no_context() -> None:
    """Context を指定しない場合のロガー初期化（デフォルト auto_flush=True）を検証します。"""
    logger = LumiLogger.get_logger("test_no_context")

    assert logger.name == "test_no_context"
    assert len(logger.handlers) >= 1
    assert any(isinstance(h, FlushStreamHandler) for h in logger.handlers)


def test_logger_get_logger_with_context(tmp_path: Path) -> None:
    """Context を指定した場合のファイルハンドラ設定（デフォルト auto_flush=True）を検証します。"""
    log_file = tmp_path / "test_app.log"
    context = AppContext(log_file_path=log_file)

    logger = LumiLogger.get_logger("test_with_context", context)

    assert any(isinstance(h, FlushFileHandler) for h in logger.handlers)
    logger.info("テストログメッセージ")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "テストログメッセージ" in content


def test_logger_get_logger_auto_flush_false(tmp_path: Path) -> None:
    """auto_flush=False を指定した場合に通常の StreamHandler / FileHandler が設定されることを検証します。"""
    log_file = tmp_path / "test_buffered.log"
    context = AppContext(log_file_path=log_file)

    logger = LumiLogger.get_logger("test_buffered", context, auto_flush=False)

    # FlushStreamHandler / FlushFileHandler ではなく、純粋な StreamHandler / FileHandler であることをチェック
    stream_handlers = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    file_handlers = [h for h in logger.handlers if type(h) is logging.FileHandler]
    assert len(stream_handlers) == 1
    assert len(file_handlers) == 1


def test_logger_get_logger_reuse_handlers() -> None:
    """既にハンドラーが存在する場合に既存ロガーを再利用することを検証します。"""
    logger1 = LumiLogger.get_logger("test_reuse")
    handler_count = len(logger1.handlers)

    logger2 = LumiLogger.get_logger("test_reuse")
    assert len(logger2.handlers) == handler_count


def test_logger_get_logger_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """ログファイル作成時に OSError が発生した場合のフォールバックを検証します。"""
    context = AppContext(log_file_path=Path("/invalid/path/test.log"))

    def mock_mkdir(*args: object, **kwargs: object) -> None:
        raise OSError("Permission denied")

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    logger = LumiLogger.get_logger("test_os_error", context)
    assert logger.name == "test_os_error"
    # ファイルハンドラーが追加されず StreamHandler のみで動作すること
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def test_logger_get_logger_package_child_no_duplicate_handlers(tmp_path: Path) -> None:
    """lumi_companion 配下の子ロガーを取得した際に親に集約され重複ハンドラーが付与されないことを検証します。"""
    log_file = tmp_path / "test_pkg.log"
    context = AppContext(log_file_path=log_file)

    child_logger = LumiLogger.get_logger("lumi_companion.test_child", context)
    pkg_logger = logging.getLogger("lumi_companion")

    # 親ロガー "lumi_companion" にハンドラーが設定されていること
    assert len(pkg_logger.handlers) >= 1
    # 子ロガー自身には重複してハンドラーが追加されないこと
    assert len(child_logger.handlers) == 0
