# rvla_app — OmniVLA-edge スマホアプリ

スマホがカメラ取得と **OmniVLA-edge のオンデバイス推論** を担い、Raspberry Pi
(raspicat) はモーター制御だけを行う構成のクライアント。既存 Path 3 の Jetson を
スマホへ置き換えたもの。設計は
[`../docs/design/mobile_port_spec.md`](../docs/design/mobile_port_spec.md)。

- 対象: Android / iOS (Flutter クロスプラットフォーム)
- 推論: ONNX Runtime Mobile (`onnxruntime` プラグイン)
- ゴール: text / pose / image
- Pi 送信: gRPC (`../proto/edge_action.proto`, coalesce+pace)

## 現状

Phase 1–4 実装済み (仕様書 §6)。Pi への gRPC 送信も動く: AppBar の Wi-Fi
アイコンから Pi の `IP[:ポート]` (既定 50061) を指定して接続する。Pi 側は

```bash
scripts/vla.sh run omnivla_edge_mobile --mode cmd_vel
```

で受信ノード + follower が立つ (`docs/USAGE.md` §5.7)。

ONNX モデルと CLIP 語彙が **未配置でも起動する** — その場合は推論がダミー軌道に
なり、ステータスに「ダミー」と表示される (Pi 側 ack も `ok-dummy` になる)。

配置すべきもの (詳細は各 README、生成は `scripts/export_*.py`):
- `assets/models/omnivla_edge.onnx`, `assets/models/clip_text.onnx` (Phase 1)
- `assets/clip/bpe_simple_vocab_16e6.txt.gz` (Phase 2)

gRPC の Dart スタブ (`lib/src/grpc/gen/`) はコミット済み。
`proto/edge_action.proto` を変えたら `../scripts/gen_proto.sh` で再生成する
(要 `dart pub global activate protoc_plugin`)。

## 開発環境 (このリポジトリで構築済み)

- Flutter 3.44.4 stable → `~/flutter` (fish の PATH に追加済み)
- JDK 21 → `/usr/lib/jvm/java-21-openjdk` (`JAVA_HOME`)
- Android SDK → `~/Android/Sdk` (platform 36 / build-tools 36 / platform-tools)

新しいシェルでは fish 設定が読み込まれ `flutter` が使える。bash で使う場合:

```bash
export PATH="$HOME/flutter/bin:$PATH"
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
export ANDROID_SDK_ROOT=$HOME/Android/Sdk ANDROID_HOME=$HOME/Android/Sdk
```

## コマンド

```bash
cd app
flutter pub get
flutter analyze
flutter test                 # 前処理コアの単体テスト
flutter build apk --debug    # Android
flutter run                  # 実機/エミュレータ接続時
```

iOS ビルドは macOS + Xcode が必要 (この WSL 環境では不可)。

## コード構成

`lib/src/` の各ファイルの役割は仕様書 §6「Phase 3 実装マップ」を参照。
