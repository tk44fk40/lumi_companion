---
name: media-stream-analyzer
description: Analyzer skill for audio VAD transcription and video frame extraction
---

# Media Stream Analyzer Skill

動画の音声VAD・Whisper文字起こし品質および抽出フレーム画像(480p)の検証ガイド。
詳細なチューニング・検証ナレッジは [AUDIO_ACCURACY_TUNING_GUIDE.md](file:///home/tk44/ghq/github.com/tk44fk40/lumi_companion/docs/AUDIO_ACCURACY_TUNING_GUIDE.md) を参照。

## 役割
- 字幕フォーマット(`subtitles.json`, `subtitles.srt`)および品質検証
- 独り言・ボソボソ声向け音声認識パラメータ管理
- 抽出画像(`extracted_frame.jpg`)の解像度・アスペクト比チェック

## 音声認識最適化（独り言・ボソボソ声）推奨パラメータ
優先順位: OS環境変数 > `.env` > `config.py`

- `WHISPER_MODEL_SIZE`: `large-v3-turbo` (高精度・高速)
- `WHISPER_VAD_THRESHOLD`: `0.35` (小音量切り落とし防止)
- `WHISPER_INITIAL_PROMPT`: `"えーっと、そうだな。..."` (日本語独り言スタイル模倣)
- `WHISPER_CONDITION_ON_PREVIOUS_TEXT`: `False` (繰り返しハルシネーション防止)

## ハルシネーション自動判別 & 補正方針
Whisper認識結果からAIの捏造(ハルシネーション)を自動検知し、後処理でドロップ(除去)補正してクリーンな字幕のみを出力する。

### 判定 & 補正動作
- **無音捏造**: `no_speech_prob > 0.6` (無音確率高) ➔ **自動ドロップ (字幕から除去)**
- **発話速度異常**: `文字数 / 音声区間秒数 > 12文字/秒` かつ 4文字超 ➔ **自動ドロップ (字幕から除去)**
- **自然な繰り返し**: 1.5秒間で「はいはいはい」(6文字)等、速度・無音確率が正常 ➔ **正常保持 (そのまま採用)**

## 検証コマンド
```bash
uv run pytest tests/test_audio_processor.py tests/test_audio_service.py
```
- **パラメータ反映**: `WHISPER_VAD_THRESHOLD`, `WHISPER_INITIAL_PROMPT` 等の正確な伝達
- **捏造ドロップ**: 無音捏造の自動削除 & 自然な繰り返し（「はいはいはい」）の正当保持
- **言語品質**: 日本語テキストへの非日本語・不正記号の非混入
