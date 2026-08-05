---
name: media-stream-analyzer
description: Analyzer skill for audio VAD transcription and video frame extraction
---

# Media Stream Analyzer Skill

このスキルは、ローカル動画ファイルから抽出された音声区間 (Silero VAD + Faster-Whisper) の文字起こし精度やタイムスタンプの正常性、および抽出フレーム画像 (OpenCV) のアスペクト比・リサイズ品質 (480p) を分析・レポート表示するためのガイドです。

## 主な役割
- `subtitles.json` と `subtitles.srt` のフォーマット検証
- 抽出画像 `extracted_frame.jpg` の解像度、アスペクト比、ファイルサイズチェック
