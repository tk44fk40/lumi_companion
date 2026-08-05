# Phase 1 開発計画 ＆ デバッグ手順書 (Phase 1 Development & Debugging Guide)

本ドキュメントは、「るみぽん！」(Lumi Companion) の Phase 1 初期構築における開発環境セットアップ、技術仕様、および Step 1 〜 Step 4 の段階的デバッグ実行手順を記載した開発者ガイドです。

---

## 1. 外部システム前提条件 (Prerequisites)

本システムの動作には、以下の外部ツールが事前導入・稼働している必要があります。

- **`FFmpeg`**: システムにインストール済みであること（`ffmpeg -version` が実行可能）。動画からの音声トラック分離・下処理に使用します。
- **`Ollama`**: ローカル環境で推論サーバーが稼働中であること（`http://localhost:11434` にて `ollama serve` または Ollama サービスが実行中）。

---

## 2. 開発環境のセットアップ (`uv`)

### 推奨 Python バージョン
- **Python 3.12**（`.python-version` により指定）
- `faster-whisper` (CTranslate2), `torch`, `onnxruntime`, `opencv-python` 等の C拡張パッケージのビルド済みホイールが最も安定して動作します。

### 事前インストール推奨デバッグツール (`uv tool install`)
プロジェクト仮想環境とは独立して、以下の開発用 CLI ツールを事前にインストールしておくことを推奨します。
```bash
uv tool install pyright
uv tool install ruff
uv tool install httpie
uv tool install rich-cli
uv tool install py-spy
```

### プロジェクト依存パッケージの同期
```bash
uv python pin 3.12
uv sync
# または requirements.txt からのインストール:
uv pip install -r requirements.txt
```

---

## 3. RTX 2070 (8GB VRAM) 環境でのテスト用推奨モデル

Phase 1 単体デバッグ・テスト用のデフォルトモデルとして **`qwen2-vl:2b`** を採用します。

- **VRAM消費量**: 約 2.0 ~ 2.5 GB（非常に軽量で高速）
- **特長**: 高速推論レスポンス、優れた日本語表現力・画面文字（OCR）認識能力。
- **自動モデルプル**: 対象モデルが Ollama 上に未ダウンロードの場合、プログラムが Ollama API (`POST /api/pull`) 経由で自動取得します。
- **コンテキスト長**: APIリクエスト時に `options.num_ctx: 4096` を動的指定します。

---

## 4. Step 1 〜 Step 4 段階的デバッグ実行手順

成果物はすべて `debug_output/` 配下に出力・保存されます。

### Step 1: 発言抽出（音声認識）＆ 字幕生成
- **実行コマンド**:
  ```bash
  uv run python scripts/debug_step1_audio.py --video sample.mp4
  ```
- **生成成果物**:
  - `debug_output/subtitles.json`: タイムスタンプ付き発言抽出データ
  - `debug_output/subtitles.srt`: 標準 SRT 字幕ファイル

### Step 2: フレーム抽出 ＆ リサイズ (480p)
- **実行コマンド**:
  ```bash
  uv run python scripts/debug_step2_vision.py --video sample.mp4 --timestamp 10
  ```
- **生成成果物**:
  - `debug_output/extracted_frame.jpg`: 指定秒（例: 10秒目）のリサイズ済み画像（480p相当）

### Step 3: Ollama投入用プロンプトJSONの生成
- **実行コマンド**:
  ```bash
  uv run python scripts/debug_step3_prompt.py --model qwen2-vl:2b --num-ctx 4096
  ```
- **生成成果物**:
  - `debug_output/ollama_payload.json`: 発言コンテキスト＋画像Base64＋プロンプト＋`options.num_ctx` を含むOllama API互換ペイロード

### Step 4: ローカルOllamaリクエスト ＆ 応答確認
- **実行コマンド**:
  ```bash
  uv run python scripts/debug_step4_ollama.py --model qwen2-vl:2b
  ```
- **生成成果物**:
  - `debug_output/ollama_response.json`: Ollama から返却されたAIコメント・リアクション応答JSON
