# FastVLM リアルタイムカメラ説明生成

Apple/FastVLM-1.5Bモデルを使用して、カメラから取得したリアルタイム映像を自動的に説明するアプリケーションです。

## 概要

このアプリケーションは以下の機能を提供します：

- **リアルタイムカメラキャプチャ**：Webカメラからフレームをキャプチャ
- **自動画像説明生成**：FastVLM-1.5Bモデルを使用して画像の説明を自動生成
- **リアルタイム表示**：カメラフィードに説明テキストをオーバーレイ表示
- **スクリーンショット機能**：指定時点のフレームをキャプチャ

## システム要件

- Python 3.8以上
- CUDA対応GPU（推奨）またはCPU
- メモリ：最小8GB（推奨16GB以上）

## インストール

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

**主な依存ライブラリ：**
- `torch`：PyTorchディープラーニングフレームワーク
- `transformers`：Hugging Faceモデルライブラリ
- `opencv-python`：カメラキャプチャと画像処理
- `Pillow`：画像操作
- `pyyaml`：設定ファイル処理

### 2. 初回実行時の注意

初回実行時、HuggingFaceからモデル（約1.5GB）がダウンロードされます。インターネット接続が必要です。

## 使用方法

### 基本的な実行

```bash
python main.py
```

### キーボード操作

- `q`：アプリケーション終了
- `s`：現在のフレームをスクリーンショット保存

## 設定

`config.yaml`ファイルで以下のパラメータを調整可能：

### カメラ設定 (`camera`)

```yaml
camera:
  device_id: 0           # カメラデバイスID
  frame_width: 640       # フレーム幅（ピクセル）
  frame_height: 480      # フレーム高さ（ピクセル）
  fps: 30                # フレームレート
```

### モデル設定 (`model`)

```yaml
model:
  model_name: "apple/FastVLM-1.5B"  # HuggingFaceモデルID
  device: "cuda"                     # 実行デバイス ("cuda" or "cpu")
  max_length: 100                    # 生成テキストの最大長
  temperature: 0.7                   # サンプリング温度
```

### 処理設定 (`processing`)

```yaml
processing:
  inference_interval: 2              # N フレームごとに推論
  display_font_size: 0.6             # 表示フォントサイズ
  text_color: [0, 255, 0]           # テキスト色（BGR）
  background_color: [0, 0, 0]       # 背景色（BGR）
```

## パフォーマンス最適化

### GPUメモリが不足する場合

1. `config.yaml`の`model.device`を`"cpu"`に変更
2. `processing.inference_interval`を増加させて推論頻度を低下
3. `camera.frame_width`と`camera.frame_height`を減少

### 推論速度の改善

- `inference_interval`を増加（例：2→5）
- `model.max_length`を減少（例：100→50）

## プロジェクト構造

```
.
├── main.py              # メインアプリケーション
├── camera_utils.py      # カメラキャプチャユーティリティ
├── model_handler.py     # FastVLMモデルハンドラー
├── config.yaml          # 設定ファイル
├── requirements.txt     # Python依存ライブラリ
└── README.md           # このファイル
```

### ファイル説明

- **main.py**：メインアプリケーションクラス（`CameraDescriptionApp`）
  - カメラとモデルの初期化
  - メインループとフレーム処理
  - UI描画とキーボード操作処理

- **camera_utils.py**：カメラキャプチャクラス（`CameraCapture`）
  - OpenCVを使用したカメラ操作
  - フレーム取得とPIL Image変換
  - リソース管理

- **model_handler.py**：モデルハンドラークラス（`FastVLMHandler`）
  - Hugging Faceからのモデル読み込み
  - 画像説明生成推論
  - デバイス管理とリソース解放

## トラブルシューティング

### カメラが認識されない

```bash
# カメラデバイスの確認（macOS）
system_profiler SPCameraDataType

# config.yamlの device_id を調整
```

### メモリ不足エラー

```
RuntimeError: CUDA out of memory
```

**解決方法：**
- `device`を`"cpu"`に変更
- `inference_interval`を増加
- 画像解像度を低下

### モデル読み込みエラー

```
OSError: Can't load model
```

**確認事項：**
- インターネット接続
- Hugging Face モデルIDが正確
- ディスク容量が十分

## 注意事項

- 初回実行時は数分かかります（モデルダウンロード）
- GPU使用時は推論が高速ですがメモリ使用量が増加
- リアルタイム性能はハードウェアに依存

## ライセンス

このプロジェクトはMITライセンスで公開されています。

## 参考資料

- [Apple FastVLM - Hugging Face](https://huggingface.co/apple/FastVLM-1.5B)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
