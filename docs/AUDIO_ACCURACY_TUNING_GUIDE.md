# 音声認識精度チューニング・検証ガイド (Audio Accuracy Tuning Guide)

本ドキュメントは、`lumi_companion` における動画・音声の文字起こし（Faster-Whisper / VAD）精度向上、環境変数パラメータのチューニング手順、ハルシネーション（AIの捏造）自動補正メカニズム、および認識精度の妥当性検証手順をまとめたナレッジベースです。

---

## 1. 音声認識パラメータとチューニングガイド

優先順位: **OS環境変数 > `.env` ファイル > `config.py` デフォルト値**

| 設定キー | デフォルト推奨値 | 役割とチューニング指針 |
| :--- | :--- | :--- |
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | 独り言やボソボソ声の認識率が最も高いモデル。VRAM制限時は `medium` を選択。 |
| `WHISPER_DEVICE` | `auto` | `cuda` (GPU) または `cpu` を自動選択。 |
| `WHISPER_COMPUTE_TYPE` | `default` | GPU時は `float16`、メモリ削減時は `int8` を指定。 |
| `WHISPER_LANGUAGE` | `ja` | 認識言語コード（日本語固定）。 |
| `WHISPER_BEAM_SIZE` | `5` | 探索ビーム幅。精度重視時は `8` 等に引き上げ。 |
| `WHISPER_INITIAL_PROMPT` | `"えーっと、そうだな。..."` | 自然な日本語独り言実例文を設定し、モデルの出力文体を誘導。 |
| `WHISPER_CONDITION_ON_PREVIOUS_TEXT` | `False` | 直前文脈への依存を切り、小さな声での同一語句連続ループ（ハルシネーション）を防止。 |
| `WHISPER_VAD_FILTER` | `True` | Silero VAD による無音区間フィルタリング。 |
| `WHISPER_VAD_THRESHOLD` | `0.35` | 音声検出閾値（標準0.5）。値を下げる（0.30〜0.35）ことで小さな独り言の切り落としを防ぐ。 |
| `WHISPER_VAD_MIN_SILENCE_DURATION_MS` | `500` | 発話区間とみなす最小無音時間(ms)。 |
| `WHISPER_NO_SPEECH_THRESHOLD` | `0.6` | 無音判定閾値。 |

### ユースケース別チューニング例
- **ケース1: 独り言や声が小さく語頭・語尾が切れる場合**
  - `WHISPER_VAD_THRESHOLD=0.30` に下げて感度を上げ、`WHISPER_INITIAL_PROMPT` に口語表現を設定。
- **ケース2: ノイズが多く無音部分で幻覚文字が出力される場合**
  - `WHISPER_VAD_THRESHOLD=0.45` に上げ、`WHISPER_CONDITION_ON_PREVIOUS_TEXT=False` を徹底。

---

## 2. ハルシネーション（捏造）の自動判別と補正メカニズム

Whisper認識結果からAIの捏造（ハルシネーション）を判定し、後処理でドロップ（除去）補正してクリーンな字幕を出力します。

### 判別および自動補正仕様
1. **無音・ノイズ捏造の補正**:
   - 判定条件: `no_speech_prob > 0.6` (無音確率高)
   - 動作: 字幕リストから自動削除 (ドロップ)
2. **物理的発話速度異常の補正**:
   - 判定条件: `文字数 / 音声区間秒数 > 12.0文字/秒` かつ 4文字超
   - 動作: 人間の解剖学的発声限界を超える捏造セグメントとして自動削除 (ドロップ)
3. **自然な短文繰り返しの保護**:
   - 例: 1.5秒間で「はいはいはい」(6文字) は `発話速度 = 4文字/秒` であり正常範囲内のため、正常発声としてそのまま保持・採用。

---

## 3. 認識精度の妥当性検証・評価手順 (Accuracy Verification)

認識精度が期待通りか、および自動補正が正しく機能しているかを客観的に評価する手順。

### A. 自動テストによる検証
以下のテストを実行し、環境変数伝達および自動補正の正常性をアサート検証します。

```bash
uv run pytest tests/test_audio_processor.py tests/test_audio_service.py
```

### B. 文字誤り率 (CER: Character Error Rate) の計測方法
テスト用動画/音声に対する正解テキスト（Ground Truth）を用意し、以下の指標で認識精度を定量評価します。

$$\text{CER} = \frac{\text{挿入数} + \text{削除数} + \text{置換数}}{\text{正解テキストの総文字数}}$$

- **評価合格基準**:
  - 旧設定 (base / プロンプトなし): CER 30%〜50%
  - **新設定 (large-v3-turbo / 独り言プロンプト / VAD 0.35)**: **CER < 15%** を達成すること。
