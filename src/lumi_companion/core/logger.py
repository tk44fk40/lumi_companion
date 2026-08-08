"""カスタムロガーモジュール。

本モジュールは、コンソールおよびファイルへの統一されたロギングを提供する
ロギング管理クラスを提供します。
"""

import logging
import sys

from lumi_companion.core.context import AppContext


class FlushStreamHandler(logging.StreamHandler):
    """出力直後に自動的に flush() を呼び出す StreamHandler。"""

    def emit(self, record: logging.LogRecord) -> None:
        """ログレコードを出力した直後にストリームをフラッシュします。

        Args:
            record (logging.LogRecord): ログレコード。
        """
        super().emit(record)
        self.flush()


class FlushFileHandler(logging.FileHandler):
    """出力直後に自動的に flush() を呼び出す FileHandler。"""

    def emit(self, record: logging.LogRecord) -> None:
        """ログレコードを出力した直後にファイルストリームをフラッシュします。

        Args:
            record (logging.LogRecord): ログレコード。
        """
        super().emit(record)
        self.flush()


class LumiLogger:
    """アプリケーション統一カスタムロガー管理クラス。"""

    @classmethod
    def _configure_handlers(
        cls,
        target_logger: logging.Logger,
        context: AppContext | None = None,
        auto_flush: bool = True,
    ) -> None:
        """指定されたロガーにハンドラーを設定します。

        Args:
            target_logger (logging.Logger): 設定対象のロガー。
            context (AppContext | None, optional): ログ設定を保持するコンテキスト。
            auto_flush (bool, optional): ログ出力時に即時フラッシュを行うか。
        """
        if target_logger.handlers:
            return

        target_logger.setLevel(logging.INFO)

        # ログフォーマットの定義
        log_format = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
        )

        # コンソール出力ハンドラーの追加
        console_handler: logging.Handler
        if auto_flush:
            console_handler = FlushStreamHandler(sys.stdout)
        else:
            console_handler = logging.StreamHandler(sys.stdout)

        console_handler.setFormatter(log_format)
        target_logger.addHandler(console_handler)

        # コンテキスト指定がある場合はファイル出力ハンドラーも設定
        if context:
            try:
                log_file = context.log_file_path
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler: logging.Handler
                if auto_flush:
                    file_handler = FlushFileHandler(log_file, encoding="utf-8")
                else:
                    file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setFormatter(log_format)
                target_logger.addHandler(file_handler)
            except OSError as e:
                # ログファイルの作成に失敗した場合はコンソール出力のみで継続
                target_logger.warning("ログファイルの作成に失敗しました: %s", e)

    @classmethod
    def get_logger(
        cls,
        name: str,
        context: AppContext | None = None,
        auto_flush: bool = True,
    ) -> logging.Logger:
        """指定された名称の構造化ロガーを取得・初期化します。

        Args:
            name (str): ロガーの識別名。
            context (AppContext | None, optional): ログ設定を保持するコンテキスト。
            auto_flush (bool, optional): ログ出力時に即時フラッシュ（バッファリング無効化）を行うか。デフォルト True。

        Returns:
            logging.Logger: 設定済みの標準 Logger インスタンス。
        """
        # パッケージ共通ロガー "lumi_companion" も併せて構成（配下モジュールのログ伝播用）
        pkg_logger = logging.getLogger("lumi_companion")
        cls._configure_handlers(pkg_logger, context, auto_flush)

        logger = logging.getLogger(name)
        # lumi_companion 配下の子ロガーは親に伝播するため、二重出力を防ぐため個別ハンドラーは設定しない
        if not name.startswith("lumi_companion"):
            cls._configure_handlers(logger, context, auto_flush)

        return logger
