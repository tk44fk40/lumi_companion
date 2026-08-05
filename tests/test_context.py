"""AppContext 単体テストモジュール。"""

from lumi_companion.core.context import AppContext


def test_app_context_defaults() -> None:
    """AppContext のデフォルト値設定を検証します。"""
    ctx = AppContext()
    assert ctx.max_image_height_px == 480
    assert ctx.jpeg_quality == 85
    assert ctx.http_timeout_seconds == 120.0


def test_app_context_get_output_path(mock_context: AppContext) -> None:
    """get_output_path でディレクトリ作成およびパス取得ができるか検証します。"""
    out_path = mock_context.get_output_path("test_file.txt")
    assert out_path.name == "test_file.txt"
    assert mock_context.debug_output_dir.exists()
