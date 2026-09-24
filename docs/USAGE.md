# USAGE

> **ROS 2 communication update (2026-09-24):** The edge and remote inference
> processes now exchange `rvln_msgs/Observation` and
> `rvln_msgs/ActionEmbedding` on `/rvln/observation` and
> `/rvln/remote_embedding`. Run `scripts/vla.sh run MODEL --mode remote
> --gpu` on the inference PC and `scripts/vla.sh run MODEL --mode edge` on the
> robot with the same `ROS_DOMAIN_ID`. `--host` and port 50051 instructions
> below describe the retired edge/remote gRPC transport. Mobile still uses
> `proto/edge_action.proto` and its own gRPC port. See the current command list
> with `scripts/vla.sh --help`.

ワークステーション、実機 Raspberry Pi Cat、または Gazebo シミュレーションで
`rvln` を実際に動かすための手順書。本ドキュメントは `README.md` の
続きという位置づけで、README がアーキテクチャと colcon ベースのビルドを扱う
のに対し、本ファイルは `scripts/vla.sh` を一次入口として具体的な運用シナリオを
追う。

サブコマンド・フラグの正確な一覧は `scripts/vla.sh --help` を参照。本書は
usage テキストに書けない「どの組み合わせをいつ使うか」を扱う。

なお `app/inference/` (スマートフォン) と `web/` (ブラウザ) の移植 — 端末上で
ONNX 推論して Pi に action をストリームする構成 — の送信側は本書の対象外
(`docs/design/mobile_port_spec.md` / `docs/design/web_port_spec.md` を参照)。
Pi 側の受け口の起動だけは本書 §5.7 で扱う。もう一方の `app/logger/`
(VLA 学習データロガー) は推論も本スタックも通さない完全に独立したアプリで、
本書の対象外 (`docs/design/logger_app_spec.md` を参照)。

## 1. 概要

システムは gRPC でつながる 2 ホストに分かれている:

```
         camera/goal                         action
           ----->                             <-----
   ┌──────────────────┐   gRPC StreamInfer   ┌──────────────────┐
   │  Edge (raspicat) │ ───────────────────▶ │  Remote (workstn) │
   │  ROS2 Humble     │ ◀─────────────────── │  VLA backbone     │
   └──────────────────┘                      └──────────────────┘
```

* **エッジ側**は ROS2 (`rvln_edge`) を実行し、カメラフレームを取得
  して JPEG エンコード、`Observation` メッセージとしてリモートへストリーム
  する。返ってきた embedding をアダプタが `nav_msgs/Path` に展開し、path
  follower が `cmd_vel` に変換する。
* **リモート側**は gRPC サーバ (`rvln_remote`) を立てる。バックエンド
  は `dummy` (CI/MVP)・`asyncvla` (Plan 2A)・`omnivla` (Plan 2B Path 1)・
  `omnivla_edge` (Plan 2B Path 3)・`movla` (自作 LFM2.5-VL Stage A ポリシー) から
  選ぶ。
* 例外が **Plan 2B Path 2** (`--mode edge-local`): OmniVLA-edge ポリシー全体を
  ロボット上で動かし、クラウドを一切使わない。
* すべて Docker イメージとして提供。`scripts/vla.sh` が必要なマウント・ネット
  ワーク・エントリコマンドを設定したうえで build/run する。基本形は
  `vla.sh run MODEL --mode MODE [OPTS]`。

`README.md` の非 Docker な colcon フローも開発用途として完全にサポートして
いる。§3.4 を参照。

## 2. 前提条件

ホスト要件:

* **ワークステーション (リモート側)** — Docker。`--mode remote --gpu` を使う
  場合は NVIDIA Container Toolkit も必要。`asyncvla`/`omnivla` イメージは
  大きなモデルを取得するため (AsyncVLA は約 15 GB)、ディスクと帯域に余裕を
  見ておくこと。
* **Jetson AGX Orin (ARM64)** — `aarch64` ホストでは `vla.sh` が自動的に
  `*-jetson` リモートイメージを選択し、`--gpus all` を `--runtime nvidia` に
  差し替える。JetPack との対応は Dockerfile ヘッダの `L4T_BASE`/
  `TORCH_VERSION` build args で合わせる。強制/無効化は
  `RVLN_JETSON=1`/`=0`。
* **ロボット (エッジ側)** — Pi (またはその他 ROS2 対応ホスト) 上の Docker。
  `real` イメージには rt-net の `raspicat_ros` パッケージが組み込まれている。
* **単一ホスト (loopback)** — 開発用。`--mode cmd_vel` で 1 コマンド、または
  `localhost` 経由でリモートとエッジを同一マシンで動作可能。

Docker フローならホスト側に ROS2 をインストールする必要はない (イメージに
ROS2 Humble が同梱)。ホスト側 ROS2 が必要になるのは §3.4 (colcon) のみ。

ネットワーク: エッジホストから所定の gRPC ポート (デフォルト `50051`) で
リモートへ到達できる必要がある。`vla.sh` の各起動は `--network host` を使う
ため、Linux ではポートフォワード設定は不要。DDS ディスカバリを隔離したい
場合は `ROS_DOMAIN_ID` をセットして起動する (全 ROS コンテナへ forward
される)。`sudo` 経由では `sudo ROS_DOMAIN_ID=N ./scripts/vla.sh …` のように
明示的に通すこと。

## 3. 初回セットアップ

クローン直後に一度だけ実行する作業。互いに独立しており順序は問わないが、
`run` サブコマンドはそれぞれ対応するイメージを必要とする。

### 3.1 Docker イメージの build

```bash
scripts/vla.sh build --all              # すべて
scripts/vla.sh build asyncvla           # リモート側 AsyncVLA
scripts/vla.sh build omnivla            # リモート側 OmniVLA (omnivla_edge の remote も兼用)
scripts/vla.sh build movla              # リモート側 movla (自作 Stage A, Jetson イメージなし)
scripts/vla.sh build real               # エッジ側フル (raspicat_ros 同梱)
scripts/vla.sh build sim                # エッジ側 + Gazebo
scripts/vla.sh build test               # CPU のみのテスト用イメージ
scripts/vla.sh build asyncvla-jetson    # ARM64 / Jetson AGX Orin 用
scripts/vla.sh build omnivla-jetson     # 同上
```

最低限便利な構成は `test` (pytest と fallback エッジが動く) に加えて、
リモート用に `asyncvla` か `omnivla` のいずれか。`real` と `sim` は実機
スタックや Gazebo が本当に必要になってから build すれば良い。

### 3.2 モデルチェックポイントのダウンロード

リモートのバックエンドは `./models/` から重みをロードする。ダウンロード
スクリプトは `huggingface_hub.snapshot_download` を使い、ホストの
`~/.cache/huggingface` を経由するため再実行は安価。

```bash
scripts/download_asyncvla_checkpoints.sh      # → models/AsyncVLA_release/  (~15 GB)
scripts/download_omnivla_checkpoints.sh       # → models/omnivla-original/  (Path 1)
scripts/download_omnivla_edge_checkpoints.sh  # → models/omnivla-edge/      (Path 2/3)
```

HuggingFace 上のリポジトリは公開設定なのでトークンは不要。実際に使う
バックエンドの分だけ落とせば良い。`dummy` バックエンドはチェックポイント
不要。

`movla` は例外で HF になく、学習ボックスから rsync で取得する:

```bash
scripts/download_movla_checkpoint.sh          # → models/movla/stage_a_v2/  (default run)
scripts/download_movla_checkpoint.sh RUN      # 別 run を指定
```

デフォルトは `nop@pve1ubuntu` の movla リポジトリ `runs/<run>/` を引く
(`MOVLA_TRAIN_HOST` / `MOVLA_TRAIN_REPO` で上書き)。`checkpoint.pt` と
`normalizer.json` が入る。

### 3.3 (任意) gRPC スタブの再生成

`proto/*.proto` を編集したとき、および `omnivla_edge_mobile` モード (§5.7) を
初めて使うときに実行する:

```bash
scripts/gen_proto.sh
```

`src/rvln_proto/rvln_proto/{rvln,edge_action}_pb2*.py`
が再生成される (gitignore 対象)。`protoc-gen-dart` が入っていれば
`app/inference/lib/src/grpc/gen/` の Dart スタブも再生成される — こちらは
コミットする。

### 3.4 (任意) ネイティブ colcon ビルド

Docker を使わず開発したい場合は `README.md` の Build 節に従う:

```bash
source /opt/ros/humble/setup.bash
vcs import src < raspicat.repos
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

manifest 変更時は `vcs import src < raspicat.repos` を再実行する。Docker
イメージは内部で同等のビルドを実行するので、`vla.sh` フローではこの作業は
必要ない。

## 4. モデルとモード

### 4.1 MODEL (= リモートバックエンド)

`vla.sh run MODEL …` の MODEL は
`asyncvla | omnivla | omnivla_edge | omnivla_edge_mobile | movla`:

| MODEL          | 用途                                    | 重み                                   | resume step            | remote イメージ                 |
|----------------|-----------------------------------------|----------------------------------------|------------------------|---------------------------------|
| `asyncvla`     | AsyncVLA 推論 (Plan 2A)                 | `models/AsyncVLA_release/`             | `750000`               | `rvln-asyncvla`         |
| `omnivla`      | OmniVLA-original 推論 (Path 1)          | `models/omnivla-original/`             | `120000`               | `rvln-omnivla`          |
| `omnivla_edge` | OmniVLA-edge (Path 2 local / Path 3 remote) | `models/omnivla-edge/omnivla-edge.pth` | なし (素の state_dict) | `rvln-omnivla` (Path 3) |
| `movla`        | 自作 LFM2.5-VL Stage A 推論 (§5.8)      | `models/movla/stage_a_v2/`             | — (checkpoint 内蔵)    | `rvln-movla` (Jetson なし) |
| `omnivla_edge_mobile` | モバイル移植: スマホが推論、このホストは受信のみ (§5.7) | 不要 (モデルはスマホ側) | — | — (エッジ系イメージのみ) |

resume step とウェイトパスは `scripts/vla.sh` の `RESUME_STEP` /
`WEIGHTS_DIR` 連想配列に固定で書かれている。変更する場合はそこを書き換える。

`dummy` バックエンドは `vla.sh run` の MODEL には現れない — サーバの
`--backend dummy` として存在し、pytest と手動起動 (§5.1 の補足) 用。

### 4.2 実行モード (`--mode MODE`)

| モード       | 走る場所                        | イメージ                      | コンテナ内で動くもの                                                     |
|--------------|---------------------------------|-------------------------------|--------------------------------------------------------------------------|
| `remote`     | GPU ワークステーション / Jetson | `asyncvla`/`omnivla`          | ROS 2 推論ノード (`rvln_remote/inference.launch.py`)。`--cpu`/`--gpu` 必須    |
| `edge`       | ロボット (Pi)                   | `real`                        | `edge_only.launch.py` (エッジノード + follower)。`--host` 必須           |
| `cmd_vel`    | 単一ホスト                      | `asyncvla`/`omnivla` + `real` | **1 コマンドで 2 コンテナ**: 127.0.0.1 bind のリモート + エッジ。follower は非モータトピック `/cmd_vel_vla` に publish (`edge_only.launch.py cmd_vel_topic:=/cmd_vel_vla`)。`--cpu`/`--gpu` 必須 |
| `sim`        | X11 の動くホスト                | `sim`                         | `sim.launch.py` (Gazebo + エッジ + follower)。`--host` 必須              |
| `edge-local` | ロボット (CUDA 必須)            | `real`                        | `omnivla_edge_local.launch.py` — Path 2、クラウドなしのスタンドアロン。`omnivla_edge` 専用 |

`omnivla_edge_mobile` は例外的に `--mode cmd_vel` のみ対応で、内容も上表と
異なる (VLA サーバなし・1 コンテナ、§5.7)。

コンテナトポロジの正準定義は `docker/compose.yaml` — モード = compose
profile で、`vla.sh` は profile の選択と `VLA_*` 変数の設定、構造差分
overlay (`docker/compose.{gpu,jetson,camera-*,sim-display}.yaml`) の追加
だけを行う。「どのモードで何が立つか」はこのファイルを読めば分かる。

補足:

* **カメラ**: `edge`/`cmd_vel`/`edge-local` では `--camera edge|realsense|
  /dev/videoN` でコンテナ内にカメラノードを起動できる (`edge` = v4l2 の
  `/dev/video0` プリセット、`realsense` は privileged + `/dev` bind で起動)。
  省略時は外部からフレームを供給する (別カメラ、
  `tools/publish_fake_image.py`、sim)。
* **`--image-topic TOPIC`**: デバイスを in-process で掴まない構成
  (`edge`/`cmd_vel`/`edge-local`/`sim`) でエッジがフレームを読むトピックを
  変更する。`/compressed` で終わるトピックは `CompressedImage` (JPEG) として
  購読される — カメラが別ホストにある場合 (raw `Image` は WiFi を越えられない)
  は必ずこちらを使う (例: `--image-topic /camera/image_raw/compressed`)。
* **`cmd_vel` の安全弁**: デフォルトではモータの繋がっていない
  `/cmd_vel_vla` に出す。実モータを回すには `--drive-motors` を明示する。
* **フォールバック**: `edge`/`sim` は対応するフルイメージが未 build の場合、
  警告のうえ `test` イメージにフォールバックする (rt-net パッケージ・Gazebo・
  torch なし。動作確認程度)。`edge-local` はフォールバックせずエラーになる —
  `vla.sh build real` が必要。

## 5. 実行レシピ集

コマンドはすべてリポジトリルートから実行可能で、`--network host` を使う。
IP やポートは適宜置き換えて使うこと。

### 5.1 シングルホスト・パイプライン確認 (`--mode cmd_vel`)

観測 → gRPC → embedding → path → cmd_vel の全経路を、実機なしで最速で回す
フロー。1 コマンドでリモートとエッジの両コンテナが立つ:

```bash
scripts/vla.sh run omnivla --mode cmd_vel --gpu    # GPU がなければ --cpu

# 別シェルで: フレームとゴールを流し込み、出力を観測
python3 tools/publish_fake_image.py                # 実カメラの代わり
scripts/control.sh goal pose 2 0                   # ゴール投入 (§6 参照)
scripts/bash.sh ros2 topic echo /cmd_vel_vla       # follower の出力を確認
```

リモート (127.0.0.1 bind) とエッジは compose profile `cmd_vel` の 2 サービス
として起動し、両方のログがフォアグラウンドに流れる。`Ctrl+C` (または片方の
終了) で両方まとめて片付く。実カメラで回すなら `--camera edge` などを追加。
モータの実駆動まで確認するときだけ `--drive-motors` を付ける。

補足 — モデルロードなしの純粋な `dummy` サーバを手で立てたい場合は、
compose を直接使ってサーバだけを test イメージで起動する
(`VLA_BACKEND` はじめ全変数にデフォルトがあり、`dummy` がそのデフォルト):

```bash
VLA_REMOTE_IMAGE=rvln-test \
    docker compose -f docker/compose.yaml --profile remote up
```

`rvln_proto` と `rvln_remote` は ament_python レイアウト
(`setup.cfg` に `script_dir`) なので `pip install -e` は最新の setuptools で
失敗する。compose の `remote` サービスが PYTHONPATH 方式で起動するのは
このため。

### 5.2 リモート ワークステーション + Pi 実機エッジ

ワークステーション (`10.0.0.5`) で GPU ポリシーを動かし、Pi でエッジを実行
する構成。

```bash
# ワークステーション
scripts/vla.sh run asyncvla --mode remote --gpu --host 10.0.0.5
# 10.0.0.5:50051 に bind。CUDA がなければ --gpu の代わりに --cpu

# Pi (カメラもコンテナ内で起動する場合)
scripts/vla.sh run asyncvla --mode edge --host 10.0.0.5 --camera edge
# RealSense なら --camera realsense。ポートはデフォルト 50051
```

任意: ポート明示指定 (ファイアウォール、マルチテナント環境など):

```bash
# ワークステーション: 全 IF にバインドしつつポート 9000
scripts/vla.sh run asyncvla --mode remote --gpu --host :9000

# Pi
scripts/vla.sh run asyncvla --mode edge --host 10.0.0.5:9000
```

### 5.3 Sim (Gazebo) + リモート ワークステーション

```bash
# ワークステーション (リモート)
scripts/vla.sh run omnivla --mode remote --gpu --host 10.0.0.5

# Sim ホスト (X11 が動くマシンならどこでも)
scripts/vla.sh run omnivla --mode sim --host 10.0.0.5
```

Sim 起動側は `image_topic:=/camera/color/image_raw` (raspicat の RealSense
トピック) に remap し、`gzclient` がホストで描画できるよう `DISPLAY` を
forward する。あわせてホスト UID 用の `/etc/passwd` エントリを合成して
Gazebo のユーザ情報欠落警告を抑える — 利用側で設定するものはない。

### 5.4 Localhost loopback (単一マシンで Sim、強力なホスト想定)

GPU と Gazebo を同居させたワークステーション向け:

```bash
# T1
scripts/vla.sh run omnivla --mode remote --gpu --host localhost
# T2
scripts/vla.sh run omnivla --mode sim    --host localhost
```

両コンテナとも host network namespace 上で動くため、コンテナ間でも
`localhost` で疎通する。

### 5.5 OmniVLA-edge (Plan 2B Path 2 / Path 3)

**Path 2 — ロボット上でスタンドアロン** (クラウドなし、CUDA 必須):

```bash
scripts/vla.sh run omnivla_edge --mode edge-local [--camera edge]
```

**Path 3 — "Jetson が推論し、Pi が制御する" リモート分割**:

```bash
# Jetson (または GPU ワークステーション): OmniVLA-edge サーバ
scripts/vla.sh run omnivla_edge --mode remote --gpu --host 10.0.0.5

# Pi: 軽量な path-only アダプタ (torch 不要) で接続
scripts/vla.sh run omnivla_edge --mode edge --host 10.0.0.5 --camera edge
```

どちらも `models/omnivla-edge/omnivla-edge.pth`
(`scripts/download_omnivla_edge_checkpoints.sh`) が必要。

### 5.6 OmniVLA を CPU で動かす (調査用)

GPU 無しのワークステーションで実バックエンドの挙動を確認したいときに有用:

```bash
scripts/vla.sh run omnivla --mode remote --cpu --host 127.0.0.1
scripts/vla.sh run omnivla --mode edge   --host 127.0.0.1   # または --mode sim
```

実測値 (16 コア / 14 GB RAM / WSL2):

* バックエンド単体スモーク — 1 推論 ~55-115 秒、ピーク RSS ~7.6 GB
* `--mode sim` と同居 — gzclient と推論で CPU を奪い合うため、1 推論が数分〜
  20 分超まで悪化することがある。sim 起動後に手で
  `docker exec <sim> pkill -9 gzclient` すると CPU が remote 側に解放されて
  劇的に速くなる (gzclient の WSL2 描画はあまり当てにならない)。
* `embedding_max_age_sec` (デフォルト 6 秒) は CPU では必ず超過する。配線確認
  だけなら `edge_params.yaml` のキャッシュ閾値を緩めるか、`/rvln/
  embedding` を直接 subscribe して初回到着を待つのが手早い。

実用テストは GPU 推奨。CPU は「パイプラインが繋がっているか」の検証用途に
限る。

### 5.7 モバイル/Web 移植の Pi 側受け口 (`omnivla_edge_mobile`)

スマホ (`app/inference/`) やブラウザ (`web/`) が **カメラ取得と推論の両方** を
担い、このホストは action chunk を受けて追従するだけの構成。VLA サーバも
カメラも立てない (受信は `edge_action_grpc_node`、WebSocket 版は
`edge_action_ws_node`)。

**gRPC 版 (スマホアプリ用, `proto/edge_action.proto`)**:

```bash
scripts/gen_proto.sh                                    # edge_action スタブ生成 (初回のみ)
scripts/vla.sh run omnivla_edge_mobile --mode cmd_vel   # 受信 + follower の 1 コンテナ
```

`EdgeActionService` が `0.0.0.0:50061` (変更は `--host BIND[:PORT]` か
`EDGE_ACTION_PORT`) で待ち受けるので、アプリの AppBar の Wi-Fi アイコンから
このホストの IP を指定して接続する。follower の出力は他モードと同じく既定で
非モーターの `/cmd_vel_vla` — 実走行時のみ `--drive-motors`。chunk が
`chunk_max_age_sec` (1 秒) 途切れると空 Path で safe-stop する。

```bash
scripts/bash.sh ros2 topic echo /cmd_vel_vla   # 追従出力の確認
```

**WebSocket 版 (web/ 用)** は `vla.sh` 未対応で、launch を直接起動する
(コンテナ内 or ネイティブ colcon 環境):

```bash
ros2 launch rvln_bringup phone_ws.launch.py    # port:=8765
```

プロトコルや web 側の使い方は `docs/design/web_port_spec.md` を参照。

### 5.8 movla (自作 LFM2.5-VL Stage A ポリシー)

`external/movla` の自作ポリシーを配信する。ワイヤ形状は OmniVLA-edge Path 3 と
同じメートルスケールの waypoint なので、エッジ側は path-only の `omnivla`
アダプタが自動選択される (torch 不要)。分割トポロジも Path 3 と同一:

```bash
# 全ローカル (単一ホスト): CPU でも動く
scripts/vla.sh run movla --mode cmd_vel --cpu       # GPU があれば --gpu

# または GPU ワークステーションでサーバ、Pi でエッジ
scripts/vla.sh run movla --mode remote --gpu --host 10.0.0.5
scripts/vla.sh run movla --mode edge --host 10.0.0.5 --camera edge
```

重みは `models/movla/stage_a_v2/`
(`scripts/download_movla_checkpoint.sh`、§3.2)。Stage A は言語のみのポリシーで
学習済みインストラクションテンプレートしか知らないため、ゴールは
`scripts/control.sh goal text "go straight ahead"` (ほか `"turn left ahead"` /
`"turn right ahead"`) のようにテンプレート文言を使う。POSE/IMAGE ゴールや
自由文は学習分布外になる。

## 6. 稼働中スタックの操作

### 6.1 `scripts/control.sh` — ゴール投入とモータ制御

ホストから、稼働中のエッジコンテナ (real/sim/test イメージを自動検出、
`RVLN_CONTAINER` で上書き可) の中で `control.py` を実行する薄い
ラッパー。`edge`/`cmd_vel`/`sim`/`edge-local` のどのモードでも同じように
使える (素の `--mode remote` にはエッジノードが居ないので対象外)。

```bash
scripts/control.sh motor on                          # raspimouse モータ電源 ON
scripts/control.sh goal pose 2 0 [THETA] [FRAME]     # POSE ゴール (FRAME 省略時 odom)
scripts/control.sh goal text "go down the hallway"   # TEXT (言語) ゴール
scripts/control.sh goal image /path/in/container.jpg # IMAGE ゴール (コンテナ内パス)
scripts/control.sh status                            # cmd_vel / odom を 1 回表示
scripts/control.sh stop                              # motor off (惰性停止)
scripts/control.sh logs [-f] [server|edge]           # docker logs ショートカット
```

エッジはゴールが届くまでアイドル (cmd_vel ゼロ) で、raspimouse はモータ電源
で cmd_vel をゲートするため、初回の典型は `motor on` → `goal pose 2 0`。
`logs` はモデルロードの進捗や実行時出力を追うのに使う (ホスト側処理で、
コンテナ内には入らない)。

ゴールトピックは TRANSIENT_LOCAL (latched) — `control.py` が publish して
即終了しても、あとから立ち上がったエッジに届く。手で `ros2 topic pub` する
場合も durability を `transient_local` にしないとエッジ (TRANSIENT_LOCAL
購読) と QoS 不整合になり届かないので注意。

### 6.2 `scripts/bash.sh` — 軽量 ROS2 シェル

ワークスペースのビルドを待たずに `ros2 topic` / `ros2 node` などを叩ける、
素の `ros:humble-ros-base` イメージのシェル。リポジトリは `/workspace` に
bind mount され、colcon overlay があれば source される。

```bash
scripts/bash.sh                     # 対話 bash
scripts/bash.sh ros2 topic list     # 1 コマンド実行して終了
```

稼働中の `vla.sh` スタックのトピックを見るには (1) `ROS_DOMAIN_ID` を
スタック側と同じ値で export しておくこと、(2) コンテナ間は /dev/shm を共有
しないため、このシェルは UDP-only の FastDDS プロファイルで起動される
(無効化は `RVLN_UDP_ONLY=0`)。詳細はスクリプト冒頭のコメント参照。

## 7. 設定リファレンス

### 7.1 エッジ — `src/rvln_edge/config/edge_params.yaml`

| Key                          | デフォルト                      | 備考                                         |
|------------------------------|---------------------------------|----------------------------------------------|
| `remote_address`             | `localhost:50051`               | launch arg `remote_address:=…` で上書き      |
| `obs_publish_rate_hz`        | `2.0`                           | リモートへ送る fps                            |
| `action_rate_hz`             | `10.0`                          | follower への path 再発行レート               |
| `image_size`                 | `[224, 224]`                    | JPEG リサイズ後のサイズ                       |
| `jpeg_quality`               | `85`                            | 1〜100                                       |
| `embedding_max_age_sec`      | `6.0`                           | これを越えると status → `DEGRADED`           |
| `embedding_hard_timeout_sec` | `15.0`                          | これを越えると status → `STALE`、safe-stop   |
| `goal_tolerance_m`           | `0.3`                           | ゴール到達判定                                |
| `image_topic`                | `/camera/image_raw`             | Sim では `/camera/color/image_raw`            |
| `goal_topic`                 | `/rvln/goal`            | TRANSIENT_LOCAL 購読 (latched)               |
| `path_topic`                 | `/rvln/predicted_path`  | `path_follower_node` が subscribe            |
| `status_topic`               | `/rvln/status`          | `DiagnosticArray`                            |
| `embedding_debug_topic`      | `/rvln/embedding`       | `publish_embedding_debug: true` のときのみ   |
| `adapter_kind`               | `stub`                          | `stub` / `asyncvla` / `omnivla` / `omnivla_edge_local` |
| `asyncvla_weights_path`      | `/workspace/models/AsyncVLA_release` | AsyncVLA エッジアダプタのみ              |
| `asyncvla_resume_step`       | `750000`                        | AsyncVLA エッジアダプタのみ                  |
| `asyncvla_device`            | `cpu`                           | AsyncVLA エッジアダプタのみ                  |
| `omnivla_edge_weights_path`  | `/workspace/models/omnivla-edge/omnivla-edge.pth` | Path 2 (`omnivla_edge_local`) のみ |
| `omnivla_edge_clip_type`     | `ViT-B/32`                      | 同上                                         |
| `omnivla_edge_device`        | `cuda:0`                        | 同上                                         |

`edge_only.launch.py` は上書き頻度の高いキー (`remote_address`、
`adapter_kind`、`image_topic`、`camera_kind`、`camera_device`、
`with_follower`、`cmd_vel_topic`、AsyncVLA 関連 3 つ) を launch 引数として
公開する。それ以外は YAML のみ。

### 7.2 リモート — `src/rvln_remote/config/remote_params.yaml`

```yaml
server:
  host: 0.0.0.0
  port: 50051
  max_concurrent_streams: 4

dummy:
  num_tokens: 8
  embed_dim: 1024
  inference_ms: 50.0
  model_version: "dummy-v1"
```

リモート推論ノードは `ros2 launch rvln_remote inference.launch.py` で起動する。
`backend`、`vla_path`、`resume_step`、`device`、`observation_topic`、
`embedding_topic` を launch 引数として指定できる。Docker の `scripts/vla.sh` も
同じ launch ファイルを使う。

### 7.3 path follower

`path_follower_node` は `with_follower:=true` のとき `edge_only.launch.py`
から起動される。制御則は OmniVLA-edge の参照実装を踏襲した単一ウェイポイント
PD (`waypoint_pd.py`) — path の `waypoint_select` 番目 (デフォルト 4) の点へ
の heading で操舵し、`control_dt` (1/3 秒) で速度化、`max_v=0.4` /
`max_w=1.0` に旋回半径を保ったままクランプする。旧 Pure Pursuit は
長ホライズン path で操舵ゲインが消失するため置き換えられた。

実行は 20 Hz。推論間隔より速く回るため、新しい path が来ない tick では直近の
「動いている」コマンドを `hold_timeout_sec` (1.0 秒) まで latch し続け、その
後 safe-stop する。新しい path が届いた場合は hold を無視してその計算結果
(停止を含む) を即座に反映する。受信 path の `frame_id` が `expected_frame`
(`base_link`) と異なる場合は警告して `cmd_vel` をゼロにする。

### 7.4 環境変数による上書き

| 変数                      | 効果                                                                    |
|---------------------------|--------------------------------------------------------------------------|
| `GRPC_PORT`               | `--host` のデフォルトポート (省略時は `50051`)                           |
| `EDGE_ACTION_PORT`        | スマホ→Pi `EdgeActionService` の既定ポート (省略時は `50061`、§5.7)      |
| `HF_CACHE_DIR`            | コンテナにマウントする HF キャッシュ (デフォルト `~/.cache/huggingface`) |
| `ROS_DOMAIN_ID`           | 全 ROS コンテナへ forward — DDS ディスカバリの隔離用                    |
| `RVLN_JETSON`     | `1` = Jetson イメージ + nvidia runtime を強制、`0` = x86 を強制         |
| `RVLN_REBUILD`    | セットするとエッジ系コンテナで `colcon build` を強制実行                |
| `RVLN_CONTAINER`  | `control.sh` が対象にするコンテナを明示指定                             |
| `RVLN_UDP_ONLY`   | `0` で `bash.sh` の UDP-only FastDDS プロファイルを無効化               |
| `RVLN_BASE_IMAGE` | `bash.sh` が使うベースイメージ (デフォルト `ros:humble-ros-base`)       |
| `ASYNCVLA_E2E`            | AsyncVLA E2E pytest スモークを有効化 (未設定時は skip)                  |
| `OMNIVLA_E2E`             | OmniVLA E2E pytest スモークを有効化 (未設定時は skip)                   |

## 8. トピックとインタフェース

### 8.1 ROS2 トピック

エッジスタックが使うトピック一覧。すべて §7 の launch 引数で remap 可能。

| Topic                            | 方向                | 型                                  | 備考                                |
|----------------------------------|---------------------|-------------------------------------|-------------------------------------|
| `/camera/image_raw`              | edge ← camera       | `sensor_msgs/Image`                 | Sim は `…/color/image_raw` で発行    |
| `/rvln/goal`             | edge ← user         | `rvln_msgs/GoalSpec`        | `POSE`/`TEXT`/`IMAGE`。TRANSIENT_LOCAL (latched) |
| `/rvln/predicted_path`   | follower ← edge     | `nav_msgs/Path`                     | `base_link` フレーム                |
| `/rvln/status`           | obs ← edge          | `diagnostic_msgs/DiagnosticArray`   | `OK`/`DEGRADED`/`WAITING_REMOTE`/`STALE` |
| `/rvln/embedding`        | obs ← edge (debug)  | `rvln_msgs/ActionEmbedding` | `publish_embedding_debug` 時のみ    |
| `/cmd_vel`                       | robot ← follower    | `geometry_msgs/Twist`               | stale またはフレーム不一致時はゼロ |
| `/cmd_vel_vla`                   | user ← follower     | `geometry_msgs/Twist`               | `--mode cmd_vel` の非モータ確認用。`--drive-motors` で `/cmd_vel` に切替 |

### 8.2 ライフサイクル

`vla_edge_node` は `LifecycleNode`。各 launch ファイルはプロセス起動の数秒後
に `ros2 lifecycle set` を実行して `unconfigured → inactive → active` まで
自動遷移させる (launch イベント経由の遷移は落ちることがあるため、明示的な
CLI 呼び出しに統一されている)。手動で上下させる場合:

```bash
ros2 lifecycle set /vla_edge_node deactivate
ros2 lifecycle set /vla_edge_node activate
```

### 8.3 gRPC サービス

`proto/rvln.proto` に `rvln.v1.VLAService` を定義:

```
rpc StreamInfer(stream Observation) returns (stream ActionEmbedding);
rpc GetModelInfo(ModelInfoRequest) returns (ModelInfo);
```

`Observation` は JPEG・`GoalSpec` (pose/text/image goal)・任意の現在 pose を
持つ。`ActionEmbedding` は FP16 でパックされた embedding を返し、エッジ
アダプタがこれを `nav_msgs/Path` に展開する。

## 9. テスト

`scripts/vla.sh test` は `rvln-test` イメージ内で pytest を実行する。
未 build なら自動でビルドされる。

```bash
scripts/vla.sh test                            # フルスイート
scripts/vla.sh test -k checkpoint              # pytest -k フィルタ
scripts/vla.sh test src/rvln_edge/test # パス指定で部分実行
```

`-k`、`-x`、`--lf` のようなフラグのみ呼び出しでは、デフォルトのテストパス
リストを自動的に prepend する。これにより pytest が cwd discovery に流れて
`external/` を歩き、依存欠落でクラッシュするのを防ぐ。

AsyncVLA / OmniVLA の E2E スモークは環境変数でゲートされており、GPU 無しでも
clean に skip する。デフォルトのスイートには含まれない:

```bash
ASYNCVLA_E2E=1 scripts/vla.sh test -k asyncvla_e2e
OMNIVLA_E2E=1 scripts/vla.sh test -k omnivla_e2e
```

## 10. トラブルシューティング

**`vla.sh: image XYZ not built; falling back to rvln-test`**
`real` または `sim` のフルイメージが未 build。fallback ではエッジスタックは
動くが、rt-net パッケージ・Gazebo・torch は使えない。実機やシミュレーション
が本当に必要なら以下で正式イメージを build する:

```bash
scripts/vla.sh build real    # または: build sim
```

`--mode edge-local` は fallback せずエラーになる (`vla.sh build real` 必須)。

**`--mode remote requires --cpu or --gpu`**
`remote` (と `cmd_vel`) は明示的なデバイス指定を要求する。デフォルトは無く、
`--gpus all` か CPU のみかを意識的に選ばせる仕様。

**`--mode <mode> requires --host HOST[:PORT]`**
`edge` と `sim` はリモートの所在を必要とする。`remote` は不要 — ローカルに
バインドし、`--host` 省略時は `0.0.0.0` がデフォルト。`cmd_vel` と
`edge-local` も不要 (前者は 127.0.0.1 固定、後者はクラウドなし)。

**エッジが `WAITING_REMOTE` から進まない**
`ActionEmbedding` の応答がエッジに届いていない。順に確認: リモートが起動
していて期待ポートで listen しているか、ホスト間で疎通するか
(`nc -z HOST PORT`)、`/rvln/goal` にゴールが publish されているか
(エッジは「最新画像」と「ゴール」の両方が揃って初めて送信を開始する)。
ゴール投入は `scripts/control.sh goal …` が確実 — 手動 publish は
durability を `transient_local` にしないと届かない (§6.1)。

**ゴールは入っているのにロボットが動かない**
raspimouse はモータ電源で cmd_vel をゲートする。`scripts/control.sh motor on`
を忘れていないか、`scripts/control.sh status` で cmd_vel が出ているか確認。

**エッジが `OK` → `DEGRADED` → `STALE` → safe-stop を繰り返す**
リモートは応答しているが `embedding_max_age_sec` より遅い。GPU に移すか、
`edge_params.yaml` の閾値を緩めること。

**`bash.sh` から `ros2 topic list` が空 / `echo` が無反応**
前者は DDS ドメイン不一致 — スタック側と同じ `ROS_DOMAIN_ID` を export して
から `bash.sh` を起動する。後者はコンテナ間で /dev/shm を共有していないのが
原因で、`bash.sh` はデフォルトの UDP-only プロファイルで回避済み
(`RVLN_UDP_ONLY=0` で無効化した場合は再発する)。

**Gazebo が `Error getting username: no matching password record` を出す**
`vla.sh` はコンテナ内に UID 用の `passwd` エントリを合成し、
`VLA_SIM_PASSWD`/`VLA_SIM_GROUP` 経由で `sim` サービスにマウントしている。
`vla.sh` を経由せず compose や `docker run` で `sim` イメージを直接起動する
場合は同等の処理を自分で行う必要がある (compose 直接利用時のデフォルトは
ホストの `/etc/passwd` をそのままマウント — ユーザ名解決は動くが HOME は
合成版と異なる)。

**`Service /spawn_entity unavailable. Was Gazebo started with GazeboRosFactory?`**
rt-net の `spawn_raspicat.launch.py` が `spawn_entity.py` を built-in 30 秒
タイムアウトで呼んでいるが、WSL2 や CPU 競合下では gazebo_ros_factory
プラグインの service 登録が間に合わずに諦めることがある。`sim.launch.py`
は最初の spawn 試行後に `get_model_list` を見て raspicat が居なければ
再 spawn する fallback を仕込んであるので、世界に robot が居なくなる事故は
通常起きない。それでも spawn しない場合は手動で:
`ros2 run gazebo_ros spawn_entity.py -entity raspicat -topic /robot_description -x 0 -y 0 -z 0 --timeout 120`。

**コンテナ内 colcon ビルドが毎回走る**
`docker/ros_entrypoint.sh` は `install/` に全 `rvln_*` パッケージが
揃っていれば colcon ステップを skip する。ソース変更後に強制再ビルドしたいときは
`RVLN_REBUILD=1`。逆に再ビルドが走るべきときに走らない場合は
ホスト側の `install/` (bind mount されている) を削除する。

**ライフサイクルノードが `unconfigured` から進まない**
launch はプロセス起動の数秒後に一度だけ `ros2 lifecycle set … configure` →
`activate` を実行する。プロセスが再起動 (例: `Ctrl+C` 後に同シェルで再起動)
しても launch system がこれを再実行しないと configure されない。手動で:
`ros2 lifecycle set /vla_edge_node configure`。

**HuggingFace のダウンロードが固まる、または認証エラー**
リポジトリは公開設定でトークン不要。`snapshot_download` が 401 を返す場合は
HF トークンをクリア (`huggingface-cli logout`) してリトライ。期限切れ
トークンが残っていると公開リポジトリでも 401 になる。

## 11. 参考

* `scripts/vla.sh --help` — サブコマンド・フラグの正準リファレンス
* `docker/compose.yaml` — 全モードのコンテナトポロジ (モード = profile) と
  構造差分 overlay (`docker/compose.*.yaml`)
* `scripts/control.sh` / `scripts/bash.sh` — 稼働中スタックの操作 (§6)
* `proto/rvln.proto` — gRPC インタフェース定義 (edge↔remote)
* `proto/edge_action.proto` — スマホ/Web → Pi の action chunk インタフェース (§5.7)
* `src/rvln_edge/launch/edge_only.launch.py` — エッジの launch 引数
* `src/rvln_bringup/launch/` — モード別 launch 構成 (`local_stack`
  (単一ホスト all-in-one、`backend:=` で選択) / `sim` / `omnivla_edge_local`
  / `mobile_cmd_vel` ほか)
* `src/rvln_edge/config/edge_params.yaml` — エッジパラメータ全件
* `src/rvln_remote/launch/inference.launch.py` — リモート推論ノードの起動
* `scripts/download_*_checkpoints.sh` — HF モデル取得ヘルパ
* `scripts/download_movla_checkpoint.sh` — movla 重み取得 (rsync、HF 非経由)
* `raspicat.repos` — rt-net ソースバージョンのピン (vcstool マニフェスト)
* `docs/design/mobile_port_spec.md` — スマートフォン推論移植 (`app/inference/`) の仕様
* `docs/design/web_port_spec.md` — ブラウザ移植 (`web/`) の仕様
* `docs/design/logger_app_spec.md` — VLA 学習データロガー (`app/logger/`) の仕様
