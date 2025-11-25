# Streaming DVC + Gemini Streaming Captioning Prototype

> **注意**: このディレクトリはユーザーが作成したサンプル・プロトタイプです。本家の研究実装や公開されたチェックポイントとは独立しており、本番評価や論文上の結論をここに基づいて行わないでください。実データでの再現や正式な評価を行う場合は、公式の実装・チェックポイントを使用してください。

Google Compute EngineのT4 GPUインスタンス上でStreaming Dense Video Captioning（Streaming DVC）のメモリ抽出を行い、抽出した特徴をGeminiマルチモーダルAPIに渡して字幕を生成するハイブリッド構成のサンプルです。リポジトリにはGPU側のWebSocketサーバー（`server/`）と、動画をストリーミング送信するクライアントCLI（`client/`）を含みます。

## アーキテクチャ概要

```
┌──────────────┐      WebSocket      ┌──────────────────────────┐
│ Streaming    │<────────────────────│  クライアント (CLI)      │
│ DVC + Gemini │──── captions/json ─▶│  - 動画ファイル or カメラ │
│ GPU Server   │                    │  - フレームを JPEG 送信   │
└──────────────┘                    └──────────────────────────┘
       │
       │ (JAX/Scenic で Streaming DVC 特徴抽出)
       ▼
Gemini Captioner (google-generativeai) で字幕生成
```

- **プロトコル**: WebSocket (`/ws/stream`)
- **入力**: JPEG Base64形式のフレーム（`type="frame"`メッセージ）
- **出力**: Geminiが生成した字幕JSONを`type="caption"`メッセージとして返却
- **バッチング**: `frame_batch_size`枚たまるたびにStreaming DVC推論を実行（環境変数で調整可）

## ディレクトリ構成

```
server/
  app.py              # FastAPI エントリポイント（WebSocket サーバー）
  config.py           # 設定の Pydantic モデル
  dvc_pipeline.py     # Streaming DVC 特徴抽出ラッパー
  gemini_client.py    # Gemini API 呼び出し
  segment_utils.py    # トークン選択・フレームエンコード共通処理
  requirements.txt    # サーバー側依存関係
client/
  stream_client.py    # Websocket ストリーミング CLI
  requirements.txt    # クライアント依存関係
README.md
```

## 事前準備

1. **Compute Engine インスタンス**
         - GPU: NVIDIA T4（CUDA 12.xドライバー導入済み）
      - OS: Ubuntu 22.04 LTS推奨
   - メモリ: 32GB 以上（Streaming DVC 推論時の JAX 初期化に余裕を持たせるため）
   - `conda` or `mamba` が利用できる環境を整備（公式ドキュメント参照）

2. **Streaming DVC 周辺セットアップ**
   - `scenic`リポジトリと`t5x`, `dmvr`など論文ベースの依存関係をColab手順に準じて導入
   - `STREAMING_DVC_ROOT`を作成し、BERT vocabやローカルconfigを配置
   - `return_features_only=True`パッチを適用済みの`streaming_model.py`

3. **Gemini API キー**
   - [Google AI Studio](https://ai.google.dev/) で API キーを取得し、サーバーで `GEMINI_API_KEY` として設定

4. **Python 環境の用意**
   - サーバー/クライアントでそれぞれ仮想環境を作成

```bash
# 例: サーバー側
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt

# クライアント側
python -m venv .venv-client
source .venv-client/bin/activate
pip install -r client/requirements.txt
```

## サーバーの起動方法

1. 必要な環境変数を設定（`.env` から読み込む場合は `python-dotenv` を活用できます）。

```bash
export STREAMING_DVC_ROOT=/home/ubuntu/StreamingDVC
export STREAMING_DVC_CONFIG_MODULE=scenic.projects.streaming_dvc.configs.git_anet_streaming_input_output_local
export STREAMING_DVC_BATCH_SIZE=1
export STREAMING_DVC_FRAME_BATCH_SIZE=48
export GEMINI_API_KEY=xxxx
export GEMINI_MODEL=gemini-1.5-pro
# Vertex AI (optional: Phase 2-1)
export VERTEX_BERT_ENDPOINT_ID=projects/<proj>/locations/us-central1/endpoints/000000000000000000
export VERTEX_LOCATION=us-central1
export GOOGLE_CLOUD_PROJECT=<proj>
export VERTEX_BERT_BATCH_SIZE=4
```

2. `uvicorn` を使って FastAPI サーバーを起動。

```bash
cd server
uvicorn gce_streaming.server.app:app --host 0.0.0.0 --port 8000
```

- 起動時に Streaming DVC モデル初期化が行われるため、初回は数分かかります。
- `/healthz` でヘルスチェック可能。
- WebSocket エンドポイント: `ws://<server-ip>:8000/ws/stream`

### systemd サービス化例

`/etc/systemd/system/streaming-dvc.service`

```ini
[Unit]
Description=Streaming DVC Gemini Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/gce_streaming
EnvironmentFile=/home/ubuntu/gce_streaming/.env
ExecStart=/home/ubuntu/gce_streaming/.venv/bin/python -m uvicorn gce_streaming.server.app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

> `.env` に上記の環境変数を定義しておくと便利です。

## クライアントの使い方

1. WebSocket サーバーに接続可能であることを確認。
2. クライアント環境で以下を実行:

```bash
cd client
python stream_client.py --server ws://<server-ip>:8000/ws/stream --video sample.mp4 --fps 10
```

オプション:

- `--video camera`でローカルカメラを使用。
- `--fps` でフレーム送信レートを調整。
- `--loop` で動画を繰り返し送信。
- `--gemini` を付けると Gemini からの字幕応答のみをログ出力。

## 実運用へ向けた TODO

- [ ] Streaming DVC のチェックポイント読込処理を安定化（現在は失敗しても警告ログのみ）
- [ ] Gemini 応答の JSON バリデーション・再試行ロジック
- [ ] フレーム間タイムスタンプから正確な字幕区間を生成
- [ ] 認証/認可の導入（API キーまたはトークン）
- [ ] Dockerfile と Terraform/Ansible 等によるデプロイ自動化

## ライセンス・注意点

- Streaming DVC / Scenic のライセンスはオリジナルに従ってください。
- Gemini API 利用には Google の利用規約が適用されます。
- プロトタイプのため、本番運用前に十分な検証とセキュリティ対策を行ってください。
