# Phase 1 開発計画 ＆ デバッグ手順書 (Phase 1 Development & Debugging Guide)

本ドキュメントは、「るみぽん！」(Lumi Companion) の Phase 1 初期構築における開発環境セットアップ、技術仕様、コンポーネント構成、シーケンス図、および Step 1 〜 Step 5 の段階的デバッグ実行手順を記載した開発者ガイドです。

---

## 1. システム構成図 ＆ シーケンス図 (Architecture Diagrams)

### 1.1 コンポーネント構成図 (Component Diagram)

```mermaid
graph TD
    subgraph InputMedia ["入力ソース (Input Media)"]
        VideoFile["ローカル動画ファイル<br>(data/test_videos/sample.mp4)"]
    end

    subgraph Phase1Core ["Phase 1 パイプラインモジュール (src/lumi_companion/)"]
        Config["[Config] config.py<br>(Pydantic Settings / 環境変数)"]
        
        subgraph AudioModule ["audio/ (発言抽出)"]
            AudioProc["[AudioProcessor] processor.py<br>(Silero VAD + Faster-Whisper)"]
            SRTExp["[SRTExporter] srt_exporter.py<br>(JSON / SRT 出力整形)"]
        end

        subgraph VisionModule ["vision/ (画像処理)"]
            FrameExt["[FrameExtractor] extractor.py<br>(OpenCV 480p リサイズ / Base64)"]
        end

        subgraph PromptModule ["prompt/ (プロンプト構築)"]
            PromptBld["[PromptBuilder] builder.py<br>(発言+画像Base64+num_ctx 結合)"]
        end

        subgraph LLMModule ["llm/ (LLMクライアント)"]
            OllamaCli["[OllamaClient] ollama.py<br>(自動モデルプル + /api/chat 推論)"]
        end
    end

    subgraph DebugOutputs ["デバッグ成果物 (debug_output/)"]
        SubJSON["subtitles.json"]
        SubSRT["subtitles.srt"]
        FrameJPG["extracted_frame.jpg"]
        PayloadJSON["ollama_payload.json"]
        ResponseJSON["ollama_response.json"]
    end

    subgraph ExternalServices ["外部サービス (External Services)"]
        OllamaServer["ローカル Ollama サーバー<br>(http://localhost:11434)"]
    end

    VideoFile ==> AudioProc & FrameExt
    Config -.-> AudioProc & FrameExt & PromptBld & OllamaCli

    AudioProc --> SRTExp
    SRTExp --> SubJSON & SubSRT
    FrameExt --> FrameJPG

    SRTExp & FrameExt --> PromptBld --> PayloadJSON
    PayloadJSON --> OllamaCli

    OllamaCli <-->|"1. GET /api/tags (存在チェック)<br>2. POST /api/pull (自動プル)<br>3. POST /api/chat (推論)"| OllamaServer
    OllamaCli --> ResponseJSON
```

---

### 1.2 処理シーケンス図 (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 開発者 / CLIスクリプト
    participant AP as [AudioProcessor]
    participant SE as [SRTExporter]
    participant FE as [FrameExtractor]
    participant PB as [PromptBuilder]
    participant OC as [OllamaClient]
    participant OS as Ollama Server (localhost:11434)
    participant Out as debug_output/

    Note over Dev, Out: --- Step 1: 発言抽出 & 字幕生成 ---
    Dev->>AP: process_async("sample.mp4")
    AP->>AP: Silero VAD + Faster-Whisper 認識
    AP-->>Dev: list[SubtitleSegment]
    Dev->>SE: save_json() & save_srt()
    SE->>Out: subtitles.json & subtitles.srt 出力

    Note over Dev, Out: --- Step 2: フレーム抽出 & リサイズ ---
    Dev->>FE: extract_frame_base64("sample.mp4", timestamp=10)
    FE->>FE: OpenCV フレーム読み込み -> 480p リサイズ -> JPEG/Base64
    FE-->>Dev: extracted_frame.jpg & Base64データ
    FE->>Out: extracted_frame.jpg 保存

    Note over Dev, Out: --- Step 3: Ollama 投入用プロンプト構築 ---
    Dev->>PB: build_payload(segments, image_base64, model="qwen3-vl:2b", num_ctx=4096)
    PB->>PB: システムプロンプト + 字幕文脈 + 画像Base64 結合
    PB-->>Dev: payload dict
    PB->>Out: ollama_payload.json 保存

    Note over Dev, Out: --- Step 4: ローカル Ollama 推論 ＆ 応答確認 ---
    Dev->>OC: generate_reaction(payload)
    OC->>OS: GET /api/tags (モデル存在チェック)
    alt モデルがローカルに存在しない場合
        OS-->>OC: モデル未存在
        OC->>OS: POST /api/pull {"name": "qwen3-vl:2b"}
        OS-->>OC: ストリーミングダウンロード完了
    else モデルが存在する場合
        OS-->>OC: モデル存在確認
    end
    OC->>OS: POST /api/chat (payload)
    OS-->>OC: 200 OK (AIリアクション応答)
    OC-->>Dev: Response dict
    OC->>Out: ollama_response.json 保存

    Note over Dev, Out: --- Step 5: タイムライン連動連続推論 ＆ デバッグ ---
    Dev->>AP: process_async("sample.mp4") 動画全体の文字起こし
    Dev->>Dev: 指定間隔（例: 3分）ごとのタイムスタンプ生成
    loop 各タイムスタンプ (ts) ごと
        Dev->>AP: ts 以前の発言字幕をフィルタ抽出
        Dev->>FE: extract_frame_base64("sample.mp4", timestamp=ts)
        Dev->>PB: build_payload(subtitles, frame_base64, model)
        Dev->>OC: generate_reaction(payload)
        OC-->>Dev: Response dict
        Dev->>Out: debug_output/timeline/{ts}s_subtitles, frame, payload, response 保存
    end
```

---

## 2. 外部システム前提条件 (Prerequisites)

本システムの動作には、以下の外部ツールが事前導入・稼働している必要があります。

- **`FFmpeg`**: システムにインストール済みであること（`ffmpeg -version` が実行可能）。動画からの音声トラック分離・下処理に使用します。
- **`Ollama`**: ローカル環境で推論サーバーが稼働中であること（`http://localhost:11434` にて `ollama serve` または Ollama サービスが実行中）。

---

## 3. 開発環境のセットアップ (`uv`)

### 推奨 Python バージョン
- **Python 3.12**（`.python-version` により指定）
- `faster-whisper` (CTranslate2), `torch`, `onnxruntime`, `opencv-python` 等の C拡張パッケージのビルド済みホイールが最も安定して動作します。

### 開発用依存パッケージ (`uv add --dev`)
型チェックツールやフォーマッター等の開発用ツールは、プロジェクトの開発依存関係 (`dev`) として管理します。
```bash
uv add --dev basedpyright ruff pytest-cov pre-commit
```

### プロジェクト依存パッケージの同期
```bash
uv python pin 3.12
uv sync
# または requirements.txt からのインストール:
uv pip install -r requirements.txt
```

---

## 4. RTX 2070 (8GB VRAM) 環境でのテスト用推奨モデル・プロバイダー

Phase 1 単体デバッグ・テストでは、ローカル Ollama モデルおよびクラウド Gemini API の両方をサポートし、柔軟に切替可能です。

### A. ローカル Ollama モデル (`qwen3-vl:2b`)
- **VRAM消費量**: 約 2.0 ~ 4.0 GB（コンテキスト長や処理画像による）
- **特長**: 完全ローカル動作、優れた日本語表現力・画面文字（OCR）認識能力。
- **自動モデルプル**: 対象モデルが Ollama 上に未ダウンロードの場合、プログラムが Ollama API (`POST /api/pull`) 経由で自動取得。
- **コンテキスト長**: APIリクエスト時に `options.num_ctx: 4096` を動的指定。

### B. クラウド Gemini API (`gemini-2.0-flash`)
- **VRAM消費量**: 0 GB（ローカル VRAM 非使用。`qwen3-vl:2b` の VRAM 負荷回避策として推奨）
- **特長**: 超高速・高精度の Vision / テキスト処理能力。
- **設定方法**: 環境変数 `GEMINI_API_KEY` を設定することで利用可能。

---

## 5. Step 1 〜 Step 5 段階的デバッグ実行手順

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

### Step 3: LLM投入用プロンプトJSONの生成
- **実行コマンド**:
  ```bash
  # Ollama (qwen3-vl:2b) 用
  uv run python scripts/debug_step3_prompt.py --model qwen3-vl:2b --num-ctx 4096

  # Gemini API (gemini-2.0-flash) 用
  uv run python scripts/debug_step3_prompt.py --provider gemini --model gemini-2.0-flash
  ```
- **生成成果物**:
  - `debug_output/ollama_payload.json` (または `llm_payload.json`): 発言コンテキスト＋画像Base64＋プロンプト＋設定パラメータを含むLLM API互換ペイロード

### Step 4: LLMリクエスト ＆ 応答確認
- **実行コマンド**:
  ```bash
  # Ollama (qwen3-vl:2b)
  uv run python scripts/debug_step4_ollama.py --model qwen3-vl:2b

  # Gemini API (gemini-2.0-flash)
  uv run python scripts/debug_step4_ollama.py --provider gemini --model gemini-2.0-flash
  ```
- **生成成果物**:
  - `debug_output/ollama_response.json`: LLM から返却されたAIコメント・リアクション応答JSON

### Step 5: タイムライン検証デバッグ（時系列連続推論シミュレーション）
動画全体の音声認識結果をもとに、一定間隔（デフォルト: 180秒＝3分ごと）および動画末尾のタイムスタンプにおける「その時点までの発言文脈」と「その時点の画面フレーム」を抽出し、連続で LLM に推論リクエストを送信することで、配信の流れに応じた応答の変化を総合検証します。

- **実行コマンド**:
  ```bash
  uv run python scripts/debug_step5_timeline.py --video sample.mp4 --interval 180
  ```
- **生成成果物** (`debug_output/timeline/`):
  - `debug_output/timeline/{timestamp}s_subtitles.json`: そのタイムスタンプまでに抽出された発言文脈リスト
  - `debug_output/timeline/{timestamp}s_frame.jpg`: そのタイムスタンプ時点のリサイズ済み画像
  - `debug_output/timeline/{timestamp}s_payload.json`: LLM への投入ペイロード JSON
  - `debug_output/timeline/{timestamp}s_response.json`: LLM からの AI リアクション応答生データ JSON
