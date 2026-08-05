"""カスタムロガーモジュール。

本モジュールは、コンソールおよびファイルへの統一されたロギングを提供する
ロギング管理クラスを提供します。
"""

import logging
import sys

from lumi_companion.core.context import AppContext


class LumiLogger:
    """アプリケーション統一カスタムロガー管理クラス。"""

    @classmethod
    def get_logger(
        cls, name: str, context: AppContext | None = None
    ) -> logging.Logger:
        """指定された名称の構造化ロガーを取得・初期化します。

        Args:
            name (str): ロガーの識別名。
            context (AppContext | None, optional): ログ設定を保持するコンテキスト。

        Returns:
            logging.Logger: 設定済みの標準 Logger インスタンス。
        """
        logger = logging.getLogger(name)

        # 既にハンドラーが登録されている場合は再利用
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)

        # ログフォーマットの定義
        log_format = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
        )

        # コンソール出力ハンドラーの追加
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        # コンテキスト指定がある場合はファイル出力ハンドラーも設定
        if context:
            try:
                log_file = context.log_file_path
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(
                    log_file, encoding="utf-8"
                )
                file_handler.setFormatter(log_format)
                logger.addHandler(file_handler)
            except OSError as e:
                # ログファイルの作成に失敗した場合はコンソール出力のみで継続
                logger.warning("ログファイルの作成に失敗しました: %s", e)

        return logger
