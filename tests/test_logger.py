"""LumiLogger 単体テストモジュール。

本モジュールは、LumiLogger によるコンソールおよびファイル出力ハンドラーの設定動作を検証します。
"""

import logging
from pathlib import Path

from lumi_companion.core.context import AppContext
from lumi_companion.core.logger import LumiLogger


def test_logger_get_logger_no_context() -> None:
    """context を指定しない場合のロガー初期化を検証します。"""
    logger = LumiLogger.get_logger("test_no_context")

    assert logger.name == "test_no_context"
    assert len(logger.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_logger_get_logger_with_context(tmp_path: Path) -> None:
    """context を指定した場合のファイルハンドラ設定を検証します。"""
    log_file = tmp_path / "test_app.log"
    context = AppContext(log_file_path=log_file)

    logger = LumiLogger.get_logger("test_with_context", context)

    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    logger.info("テストログメッセージ")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "テストログメッセージ" in content


def test_logger_get_logger_reuse_handlers() -> None:
    """既にハンドラーが存在する場合に既存ロガーを再利用することを検証します。"""
    logger1 = LumiLogger.get_logger("test_reuse")
    handler_count = len(logger1.handlers)

    logger2 = LumiLogger.get_logger("test_reuse")
    assert len(logger2.handlers) == handler_count
