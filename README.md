# Face Detect Sample

このプロジェクトは、Go言語とOpenCVを使用してリアルタイムで顔検出とライブネス検出を行うサンプルアプリケーションです。WebSocketを介してブラウザに映像をストリーミングし、顔の検出とライブネスの結果を表示します。

## 機能

- リアルタイム顔検出
- ライブネス検出（瞬き、動き、テクスチャ、深度分析）
- WebSocketを使用したブラウザへの映像ストリーミング
- グレースフルシャットダウン

## 必要条件

- Go 1.23.5以上
- OpenCV 4.5以上
- `gocv`パッケージ
- `gorilla/mux`と`gorilla/websocket`パッケージ

## インストール

1. リポジトリをクローンします。

   ```bash
   git clone https://github.com/yourusername/face-detect-sample.git
   cd face-detect-sample
   ```

2. 必要なGoパッケージをインストールします。

   ```bash
   go mod tidy
   ```

3. OpenCVをインストールします。詳細な手順は[OpenCVの公式ドキュメント](https://opencv.org/)を参照してください。

4. `gocv`パッケージをインストールします。

   ```bash
   go get -u -d gocv.io/x/gocv
   ```

5. 必要なカスケード分類器ファイルをダウンロードして`data`ディレクトリに配置します。

   ```bash
   curl -o data/haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
   curl -o data/haarcascade_eye.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
   ```

## 使用方法

1. アプリケーションをビルドして実行します。

   ```bash
   air
   ```

2. ブラウザで`http://localhost:8080`にアクセスします。

3. リアルタイムで顔検出とライブネス検出の結果が表示されます。

## プロジェクト構成

face_detect_sample/
├── main.go
├── data/
│ ├── haarcascade_frontalface_default.xml
│ └── haarcascade_eye.xml
└── web/
  └── templates/
    └── index.html

## main.goの概要

`main.go`はプロジェクトのエントリーポイントであり、以下の機能を実装しています

- カメラの初期化と設定
- 顔検出とライブネス検出のための依存関係の初期化
- WebSocketを介したリアルタイム映像ストリーミング
- HTTPサーバーの設定と起動
- グレースフルシャットダウンの実装

## ライセンス

このプロジェクトはMITライセンスの下で提供されています。詳細は`LICENSE`ファイルを参照してください。

## 貢献

貢献を歓迎します！バグ報告、機能提案、プルリクエストをお待ちしています。

## 問い合わせ

質問やサポートが必要な場合は、[okamyuji@gmail.com](mailto:okamyuji@gmail.com)までご連絡ください。
