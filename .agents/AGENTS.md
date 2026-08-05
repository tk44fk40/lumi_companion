# lumi_companion 開発ルール

## 1. コーディング規約
- **スタイル**: PEP 8 / Google Python スタイルガイド準拠。
- **命名**: クラス `PascalCase`, 関数/変数/ファイル `snake_case`, 定数 `UPPER_SNAKE_CASE`
- **フォーマット**: 4スペースインデント, 1行最大88文字（Ruff標準）。日本語コメント/Docstringは句読点で適宜改行。

## 2. 計画書 / Issue / ブランチ / PR 規約
- **優先順位**: Git・ブランチ運用は本プロジェクト規約（Issue連動ブランチ）を最優先適用する。
- **Issue & ブランチ**: `gh issue create` ➔ ブランチ名 `issue-<Issue番号>/<機能名>`
- **PR & マージ**: `gh pr create` (本文に `Closes #<Issue番号>` 明記) ➔ スカッシュマージ (`gh pr merge --squash`)
- **計画書の作成**: 指示された場合は計画書を作成し、作業タスクリストで進捗を管理し、Issueへも反映する。
- **計画書メタデータ**: `implementation_plan.md` ヘッダーに関連 Issue, 作業ブランチ, PR 番号/URL を記載。

## 3. 規模制限 & 可読性 (KISS原則)
- 1 ファイル最大 300 行以下（推奨 100〜200 行）。超過時は単一責任原則に従いモジュール分割。
- ワンライナー詰め込みや難解・トリッキーなコードの禁止。

## 4. Docstring & コメント
- 全モジュール/クラス/関数に Google スタイル Docstring (日本語) を記述。
- **サマリー行**: 1 行目に簡潔なタイトル（複数行禁止）。空行を挟んで詳細記述。
- 関数 Docstring 必須項目: 概要, `Args`, `Returns`, `Raises`

## 5. 定数一元管理 (ハードコード禁止)
- 数値・文字列・パス・モデル名等の定数はコード内に直接ハードコードせず、`AppContext` または設定クラスに一元管理。

## 6. 例外処理
- `try-except` 内は最小限に留め、ログ記録 (`logger.error`) を中心とする。リカバリロジックは例外ブロック外に分離。

## 7. 設計パターン
1. **Strategy ＋ DI**: プロバイダー処理は `Protocol` で抽象化しコンストラクタ注入（モック化容易化）。
2. **Facade**: `LumiApp` 等のメインオーケストレーターはシンプルな単一インターフェースのみ公開。
3. **Context / State**: アプリ状態は `AppContext` にカプセル化。
4. **Producer-Consumer**: データ生成と推論消費の間に `asyncio.Queue` を挟み、イベントループを非ブロッキング化。

## 8. コード品質 & 自動化ツール
- Python 3.12 準拠、全関数・クラスに厳格な型注釈を付与。
- `uv run basedpyright` をエラー 0 件で通過すること（`Any` 排除、`Self` / 具体型を使用）。
- `uv run ruff check --fix .` および `uv run ruff format .` 準拠。

## 9. 非同期 (Asyncio) ブロッキング禁止
- 重い CPU/GPU 処理 (Whisper, VAD, OpenCV 等) は `asyncio.to_thread` や `loop.run_in_executor` でスレッドに委譲。

## 10. 非破壊デバッグ
- 中間出力 (字幕JSON/SRT, 画像等) は `debug_output/` へ出力。コンポーネント単体でデバッグ可能スクリプト化。

## 11. AIコード生成 & セルフチェック規範
コード作成・修正における自律ワークフロー（Issue作成、シグネチャ優先提示、TDDテスト先行作成、セルフチェック観点、エビデンスログ提示等）の具体的な手順については、[autonomous-code-workflow](file:///home/tk44/ghq/github.com/tk44fk40/lumi_companion/.agents/skills/autonomous-code-workflow/SKILL.md) スキルに完全準拠して実施すること。

## 12. テストコード (`tests/`) 規範
- **構造化**: モジュール単位で `tests/test_*.py` に分割。共通化は `conftest.py` を活用。
- **記述スタイル**: AAA (Arrange-Act-Assert) パターンを徹底。複雑な `if`/`for` をテスト内に書かない。
- **カバレッジナレッジ**: 未カバー許容指針は [docs/TEST_COVERAGE_KNOWLEDGE.md](file:///home/tk44/ghq/github.com/tk44fk40/lumi_companion/docs/TEST_COVERAGE_KNOWLEDGE.md) を参照・維持。
