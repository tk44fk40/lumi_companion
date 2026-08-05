---
name: media-stream-analyzer
description: Analyzer skill for audio VAD transcription and video frame extraction
---

# Media Stream Analyzer Skill

動画の音声VAD・Whisper文字起こし品質および抽出フレーム画像(480p)の検証スキル。
設定パラメータの詳細やハルシネーション（捏造）判定の厳密な仕様、CER計測手順等の全ナレッジは [AUDIO_ACCURACY_TUNING_GUIDE.md](file:///home/tk44/ghq/github.com/tk44fk40/lumi_companion/docs/AUDIO_ACCURACY_TUNING_GUIDE.md) を一意なソース (SSOT) として参照すること。

## 役割
- 字幕フォーマット (`subtitles.json`, `subtitles.srt`) および品質検証
- 音声認識パラメータ（独り言・ボソボソ声向け）の動作確認
- 抽出画像 (`extracted_frame.jpg`) の解像度・アスペクト比チェック

## 音声認識 & ハルシネーション自動補正の概要
* **パラメータ管理順位**: OS環境変数 > `.env` > `config.py`
* **自動補正動作**: 無音捏造 (`no_speech_prob > 0.6`) および物理的発話速度異常 (>12文字/秒) の自動ドロップ除去。
* 詳細仕様およびユースケース別チューニング例は [AUDIO_ACCURACY_TUNING_GUIDE.md](file:///home/tk44/ghq/github.com/tk44fk40/lumi_companion/docs/AUDIO_ACCURACY_TUNING_GUIDE.md) を参照。

## 検証コマンド
```bash
uv run pytest tests/test_audio_processor.py tests/test_audio_service.py
```
- **パラメータ反映**: `WHISPER_VAD_THRESHOLD`, `WHISPER_INITIAL_PROMPT` 等の正確な伝達検証
- **自動補正検証**: 無音捏造の自動削除 & 自然な短文繰り返しの正常保持アサート
- **言語品質**: 日本語テキストへの非日本語・不正記号の非混入確認
