---
name: ollama-inspector
description: Inspection and performance debugging tool for local Ollama service and models
---

# Ollama Inspector Skill

このスキルは、ローカル環境で稼働中の Ollama サーバー (`http://localhost:11434`) の状態検証、インストール済みモデル一覧の確認、未ダウンロードモデルの自動プル、および推論レスポンス性能 (tokens/sec) の計測を行うデバッグ用ガイド・スクリプト群です。

## 主な役割
- `http http://localhost:11434/api/tags` による起動モデルチェック
- Ollama API 経由での自動プル (`POST /api/pull`) の動作ログ・ストリーム解析
- リクエスト時の動的 `num_ctx` オプションの有効性確認
