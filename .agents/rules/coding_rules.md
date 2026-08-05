# lumi_companion 開発ルール (Development Rules)

本ドキュメントは「るみぽん！」(`lumi_companion`) の開発およびコード品質、非同期処理の指針を定めたものです。

---

## 1. ブランチ運用
- 個別の修正や日常的な作業・コミットは基本的にデフォルトブランチ (`master`) 上で行う。
- PR 用ブランチは提出が必要なタイミングで一括追従・スカッシュして作成する。

---

## 2. 型宣言 & コード品質の徹底
- Python 3.12 準拠。すべての関数・メソッド・クラスに明確な型注釈 (Type Hints) を付与する。
- 静的型チェックは `pyright src` を通過すること。
- コードフォーマットおよびリンターは `ruff` (`ruff check .`, `ruff format .`) を使用する。

---

## 3. 非同期 (Asyncio) イベントループのブロッキング禁止
- Whisper音声認識推論、VAD推論、OpenCVによる画像取得・リサイズなどの重いCPU/GPU処理は、メインの `asyncio` イベントループ上で直接実行しない。
- 必ず `asyncio.to_thread` や `loop.run_in_executor` を使用してバックグラウンドスレッドに委譲し、イベントループのブロッキングを防ぐ。

---

## 4. 非破壊デバッグ・成果物の透明性
- 各コンポーネント処理の中間出力（字幕JSON, 字幕SRT, 抽出画像, Ollamaペイロード等）は `debug_output/` ディレクトリへ非破壊的に出力可能とする。
- パイプライン全体を実行せずとも、単体コンポーネント（Step 1〜4等）の独立デバッグが実行できるようにスクリプト化する。

---

## 5. 設定管理の一元化
- Ollama のホストURL (`OLLAMA_HOST`)、デフォルトモデル (`OLLAMA_MODEL`)、コンテキスト長 (`OLLAMA_NUM_CTX`) 等は `pydantic-settings` (`src/lumi_companion/config.py`) 経由で一元管理する。
- ソースコード内へ直接ホストURLやモデル名をハードコードすることを禁止する。
