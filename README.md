# FastVLM リアルタイムカメラ説明生成

Apple/FastVLM-1.5Bモデルを使用して、Webカメラから取得したリアルタイム映像を自動的に説明するアプリケーションです。ビジョン言語モデル（VLM）により、カメラの映像から画像の内容をリアルタイムで分析し、自然言語で説明を生成します。

## 📋 目次

- [概要](#概要)
- [特徴](#特徴)
- [システム要件](#システム要件)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [詳しい使用方法](#詳しい使用方法)
- [設定ガイド](#設定ガイド)
- [パフォーマンス最適化](#パフォーマンス最適化)
- [プロジェクト構造](#プロジェクト構造)
- [トラブルシューティング](#トラブルシューティング)
- [技術詳細](#技術詳細)

## 概要

このアプリケーションは以下の機能を提供します：

- **リアルタイムカメラキャプチャ**：Webカメラからフレームをキャプチャ
- **自動画像説明生成**：FastVLM-1.5Bモデルを使用して画像の説明を自動生成
- **リアルタイム表示**：カメラフィードに説明テキストをオーバーレイ表示
- **スクリーンショット機能**：指定時点のフレームをキャプチャして保存
- **柔軟な設定**：YAML設定ファイルで簡単にカスタマイズ可能

## 特徴

### FastVLM-1.5Bについて

FastVLM-1.5Bはアップルが開発した軽量で高速なビジョン言語モデルです：

- **モデルサイズ**：15億パラメータ（約1.5GB）
- **特徴**：FastViTHD視覚エンコーダーを採用し、高解像度画像でも高速に処理
- **利点**：
  - CPU/GPU両対応
  - メモリ効率が良い
  - 推論速度が高速
  - 日本語を含む多言語対応

## システム要件

### ハードウェア

- **CPU**：Intel Core i5以上 または Apple Silicon（M1/M2/M3）
- **メモリ**：
  - CPU実行時：最小4GB（推奨8GB以上）
  - GPU実行時：最小8GB（推奨16GB以上）
- **ディスク容量**：モデルダウンロード用に3GB以上の空き容量
- **Webカメラ**：USB接続またはビルトインカメラ

### ソフトウェア

- **Python**：3.8以上（推奨3.10以上）
- **OS**：macOS, Linux, Windows対応

## インストール

### ステップ1：リポジトリのクローン

```bash
git clone https://github.com/example/object-detection.git
cd object-detection
```

### ステップ2：Pythonの仮想環境を作成

```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### ステップ3：依存ライブラリのインストール

```bash
# 依存ライブラリを一括インストール
pip install -r requirements.txt
```

**インストール時間の目安**：
- インターネット速度が良い場合：5～10分
- 通常の環境：10～20分

**主な依存ライブラリ**：

| パッケージ | バージョン | 用途 |
|----------|-----------|------|
| torch | >=2.0.0 | PyTorchディープラーニングフレームワーク |
| transformers | >=4.40.0 | Hugging Faceモデルライブラリ |
| accelerate | >=0.20.0 | モデル高速実行 |
| opencv-python | >=4.8.0 | カメラキャプチャと画像処理 |
| pillow | >=10.1.0 | 画像操作 |
| numpy | >=1.24.0 | 数値計算 |
| pyyaml | >=6.0 | 設定ファイル処理 |
| timm | >=0.9.0 | FastVLMの視覚エンコーダー |

### ステップ4：初回実行時の注意

初回実行時、HuggingFaceからモデル（約1.5GB）がダウンロードされます：

1. **インターネット接続**が必須です
2. **ダウンロード時間の目安**：5～30分（インターネット速度に依存）
3. **ストレージ確認**：3GB以上の空き容量を確認してください

#### モデルキャッシュの位置

モデルは自動的に以下の位置に保存されます：

```
project_root/models/
├── hub/
│   ├── models--apple--FastVLM-1.5B/
│   │   ├── blobs/                 # モデルの実ファイル（~1.5GB）
│   │   │   ├── <hash値>.safetensors
│   │   │   └── ...その他のファイル
│   │   ├── refs/                  # リリースのリファレンス
│   │   └── snapshots/             # スナップショット
│   └── .locks/                    # ロックファイル
├── stored_tokens                  # キャッシュされたトークン
└── token                          # 認証トークン
```

> **重要**：`models/`フォルダはGitignoreに登録されているため、リポジトリには含まれません。初回実行時に自動ダウンロードされます。

## クイックスタート

### 最も簡単な実行方法

```bash
# 仮想環境を有効化
source venv/bin/activate

# アプリケーションを実行
python main.py
```

これだけで以下が自動的に実行されます：
1. ✅ カメラが初期化される
2. ✅ FastVLMモデルがロードされる（初回のみ時間がかかります）
3. ✅ カメラフィードが表示される
4. ✅ リアルタイムで画像説明が生成される

## 詳しい使用方法

### 基本的な実行フロー

#### 1. アプリケーション起動

```bash
python main.py
```

起動時のログ出力例：
```
2025-11-05 12:25:03,337 - __main__ - INFO - Configuration loaded from config.yaml
Camera initialized successfully
Resolution: 640x480 @ 30fps
2025-11-05 12:25:05,531 - model_handler - INFO - FastVLMHandler initialized for apple/FastVLM-1.5B on cpu
2025-11-05 12:25:05,531 - model_handler - INFO - Loading model: apple/FastVLM-1.5B
2025-11-05 12:25:05,531 - model_handler - INFO - Loading tokenizer...
...モデルロード中...
2025-11-05 12:26:00,497 - model_handler - INFO - Model loaded successfully
Application started. Press 'q' to quit, 's' to save screenshot
```

#### 2. カメラフィードの確認

起動後、以下の情報がカメラウィンドウに表示されます：

- **フレーム番号**：左上に `Frame: XXX` と表示
- **画像説明**：フレームごとに生成された説明テキスト
- **操作ガイド**：下部にキーボード操作ガイドが表示

#### 3. リアルタイム説明生成

設定ファイルの`processing.inference_interval`で指定したフレーム間隔（デフォルト2フレームごと）で以下が実行されます：

1. カメラからフレーム取得
2. FastVLMモデルで画像を分析
3. 画像の説明をテキスト生成
4. 画面に表示

### キーボード操作

| キー | 機能 | 説明 |
|-----|------|------|
| `q` | アプリケーション終了 | プログラムを終了します |
| `s` | スクリーンショット保存 | 現在のフレームを`screenshot_XXXX.png`として保存 |

### スクリーンショットの詳細

`s`キーを押すと：

```bash
# ファイルが保存されます
screenshot_120.png      # フレーム120の画像
screenshot_345.png      # フレーム345の画像
screenshot_789.png      # フレーム789の画像
```

保存場所：プロジェクトルートディレクトリ

### アプリケーション終了

以下の方法でアプリケーションを終了できます：

1. **キーボード操作**：`q`キーを押す
2. **ウィンドウ操作**：ウィンドウの閉じるボタンをクリック
3. **ターミナル操作**：`Ctrl+C`キーを押す

終了時のログ出力例：
```
2025-11-05 12:35:42,100 - __main__ - INFO - Quit requested
Camera released. Total frames captured: 450
Resources cleaned up
Application closed
```

## 設定ガイド

`config.yaml`ファイルを編集することで、アプリケーションの動作をカスタマイズできます。

### カメラ設定

```yaml
camera:
  device_id: 0           # カメラデバイスID（通常は0）
  frame_width: 640       # フレーム幅（ピクセル）
  frame_height: 480      # フレーム高さ（ピクセル）
  fps: 30                # フレームレート（フレーム/秒）
```

#### カメラ設定の詳細説明

**device_id**：
- 複数のカメラが接続されている場合、IDを変更します
- 値の例：0（内蔵カメラ）、1（USB接続カメラ1）、2（USB接続カメラ2）

**frame_width と frame_height**：
- 解像度が高いほど詳細な説明が生成されますが、処理が遅くなります
- 推奨値：
  - 高速処理重視：320x240, 480x360
  - バランス型：640x480（デフォルト）
  - 高精度重視：1280x720, 1920x1080

**fps**：
- フレームレート（1秒あたりのフレーム数）
- 値が大きいほどスムーズですが、CPUを多く使用します

### モデル設定

```yaml
model:
  model_name: "apple/FastVLM-1.5B"  # HuggingFaceモデルID
  device: "cuda"                     # 実行デバイス
  max_length: 100                    # 生成テキストの最大トークン数
  temperature: 0.7                   # サンプリング温度
```

#### モデル設定の詳細説明

**model_name**：
- デフォルト：`apple/FastVLM-1.5B`（推奨）
- 変更しない限り、このままでOKです

**device**：
- `"cuda"`：NVIDIA GPUを使用（高速）
- `"cpu"`：CPUを使用（遅いが、GPUがない場合に有効）
- 自動判定：CUDA対応GPUがない場合、自動的にCPUにフォールバック

**max_length**：
- 生成されるテキストの最大トークン数
- 値の目安：
  - 短い説明：50トークン
  - 標準説明：100トークン（デフォルト）
  - 詳しい説明：150～200トークン
- 値が大きいほど詳細な説明が生成されますが、推論時間が増加します

**temperature**：
- テキスト生成の多様性を制御（0.0～1.0）
- 値の目安：
  - 0.1～0.3：確定的で繰り返しが多い説明
  - 0.5～0.7：バランス型（デフォルト0.7）
  - 0.8～1.0：創造的でバリエーション豊かな説明

### 処理設定

```yaml
processing:
  inference_interval: 2              # N フレームごとに推論を実行
  display_font_size: 0.6             # 表示フォントサイズ
  text_color: [0, 255, 0]           # テキスト色（BGR形式）
  background_color: [0, 0, 0]       # 背景色（BGR形式）
```

#### 処理設定の詳細説明

**inference_interval**：
- N フレームごとに画像説明を生成
- 値の目安：
  - 1：毎フレーム推論（最も詳細だが遅い）
  - 2～5：バランス型（推奨）
  - 10以上：高速処理重視（説明更新が遅い）

**display_font_size**：
- テキスト表示のフォントサイズ
- 値の目安：0.3～1.0

**text_color と background_color**：
- OpenCVのBGR形式で色を指定
- 例：
  - 緑：`[0, 255, 0]`
  - 白：`[255, 255, 255]`
  - 赤：`[0, 0, 255]`
  - 青：`[255, 0, 0]`
  - 黒：`[0, 0, 0]`

### 設定ファイルの例

#### 例1：高速処理重視

```yaml
camera:
  frame_width: 480
  frame_height: 360
  fps: 15

model:
  device: "cpu"
  max_length: 50
  temperature: 0.5

processing:
  inference_interval: 5
```

#### 例2：高精度重視

```yaml
camera:
  frame_width: 1280
  frame_height: 720
  fps: 30

model:
  device: "cuda"
  max_length: 150
  temperature: 0.7

processing:
  inference_interval: 1
```

#### 例3：バランス型（デフォルト）

```yaml
camera:
  frame_width: 640
  frame_height: 480
  fps: 30

model:
  device: "cuda"
  max_length: 100
  temperature: 0.7

processing:
  inference_interval: 2
```

## パフォーマンス最適化

### CPUでの実行が遅い場合

1. **解像度を低下させる**
   ```yaml
   camera:
     frame_width: 320
     frame_height: 240
   ```

2. **推論頻度を低下させる**
   ```yaml
   processing:
     inference_interval: 5
   ```

3. **生成テキスト長を短縮**
   ```yaml
   model:
     max_length: 50
   ```

### GPUメモリが不足する場合

1. **デバイスをCPUに変更**
   ```yaml
   model:
     device: "cpu"
   ```

2. **精度を低下させる**
   ```yaml
   camera:
     frame_width: 480
     frame_height: 360
   ```

### メモリ使用量の削減

- **フレームバッファを削減**
- **モデルを軽量版に変更**（オプション）

## プロジェクト構造

```
object-detection/
├── main.py                    # メインアプリケーション
├── camera_utils.py            # カメラキャプチャユーティリティ
├── model_handler.py           # FastVLMモデルハンドラー
├── config.yaml                # 設定ファイル
├── requirements.txt           # Python依存ライブラリ
├── README.md                  # このファイル
├── .gitignore                 # Git除外設定
├── venv/                      # Python仮想環境（初回実行時作成）
├── models/                    # ダウンロードされたモデル（初回実行時作成）
└── screenshot_*.png           # スクリーンショット（s キー押下時作成）
```

### ファイル説明

#### main.py
メインアプリケーションの実装：
- `CameraDescriptionApp`クラス：アプリケーションのメイン処理
- 機能：
  - 設定ファイルの読み込み
  - カメラの初期化
  - モデルの読み込み
  - メインループの実行
  - UI描画とキーボード操作処理

#### camera_utils.py
カメラキャプチャの処理：
- `CameraCapture`クラス：カメラ操作を担当
- 機能：
  - OpenCVを使用したカメラの初期化と操作
  - フレームの取得
  - PIL Image への変換
  - リソース管理

#### model_handler.py
FastVLMモデルの処理：
- `FastVLMHandler`クラス：モデルの読み込みと推論を担当
- 機能：
  - Hugging Face からのモデル読み込み
  - 画像説明生成（推論）
  - デバイス（CPU/GPU）の管理
  - リソース解放

#### config.yaml
アプリケーション設定ファイル：
- カメラの設定
- モデルの設定
- UI処理の設定

## トラブルシューティング

### カメラが認識されない

**症状**：
```
Error: Unable to open camera device 0
```

**解決方法**：

1. **macOS の場合**
   ```bash
   # カメラデバイスの確認
   system_profiler SPCameraDataType
   ```

2. **Linux の場合**
   ```bash
   # カメラデバイスの確認
   ls /dev/video*

   # カメラの詳細情報
   v4l2-ctl --list-devices
   ```

3. **Windows の場合**
   - デバイスマネージャーで接続状況を確認

4. **config.yaml を修正**
   ```yaml
   camera:
     device_id: 1  # 別のカメラIDを試す
   ```

### メモリ不足エラー

**症状**：
```
RuntimeError: CUDA out of memory
MemoryError: Unable to allocate X.XX GiB for an array
```

**解決方法**：

1. **GPU使用時**
   ```yaml
   model:
     device: "cpu"  # CPUに切り替え
   ```

2. **画像解像度を低下**
   ```yaml
   camera:
     frame_width: 320
     frame_height: 240
   ```

3. **推論頻度を低下**
   ```yaml
   processing:
     inference_interval: 5
   ```

4. **他のアプリケーションを閉じる**

### モデル読み込みエラー

**症状**：
```
OSError: Can't load model
ValueError: Can't load 'apple/FastVLM-1.5B'
```

**確認事項**：
- ✅ インターネット接続が確立されているか
- ✅ ディスク容量が十分か（3GB以上）
- ✅ Hugging Face モデルIDが正確か
- ✅ `models/`フォルダの権限が正しいか

**解決方法**：

```bash
# モデルキャッシュをクリア
rm -rf models/

# 再度実行
python main.py
```

### 推論が遅い

**原因と解決方法**：

| 原因 | 解決方法 |
|------|---------|
| CPUで実行 | GPUの利用を検討 |
| 解像度が高すぎる | フレーム幅・高さを低下 |
| 推論間隔が短すぎる | inference_interval を増加 |
| 他のアプリが重い | メモリ使用量を削減 |

### カメラがブロックされている

**症状**（macOS）：
```
OpenCV: not authorized to capture video (status 0)
OpenCV: camera failed to properly initialize!
```

**解決方法**：

1. **設定 → セキュリティとプライバシー → カメラ** で、ターミナル（またはPython）へのアクセスを許可
2. Macを再起動
3. 再度実行

## 技術詳細

### FastVLMの仕組み

FastVLMは以下の構成で動作します：

1. **視覚エンコーダー（FastViTHD）**
   - 画像を視覚的な特徴に変換
   - 高解像度画像を効率的に処理

2. **言語モデル（Qwen2ベース）**
   - 視覚特徴をテキストに変換
   - 自然な説明文を生成

3. **マルチモーダル統合**
   - 画像と テキストプロンプトを組み合わせ
   - コンテキストに応じた説明を生成

### 推論フロー

```
カメラフレーム
    ↓
[画像前処理]
    ↓
[視覚エンコーダー]
    ↓
[言語モデル]
    ↓
[テキストデコード]
    ↓
[画面表示]
```

### パフォーマンス指標

| 環境 | 推論時間 | メモリ使用量 |
|-----|---------|-----------|
| Apple M1 (CPU) | 3～5秒 | 1～2GB |
| Apple M1 (GPU) | 1～2秒 | 2～3GB |
| NVIDIA RTX 3090 (GPU) | 0.5～1秒 | 4～6GB |

## ライセンス

このプロジェクトはMITライセンスで公開されています。

## 参考資料

- [Apple FastVLM - Hugging Face](https://huggingface.co/apple/FastVLM-1.5B)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)

## サポート

問題が発生した場合：

1. このREADMEのトラブルシューティングセクションを確認
2. `config.yaml`の設定を見直す
3. ログ出力を確認して、具体的なエラーメッセージを記録
4. GitHubのIssuesで報告（複数の人が同じ問題に直面している可能性があります）

## 今後の拡張予定

- [ ] 複数カメラ対応
- [ ] リアルタイム字幕生成
- [ ] 動画ファイルの処理対応
- [ ] WebUIでの設定変更
- [ ] モデルの選択肢拡大
