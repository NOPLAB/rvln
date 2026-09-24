# OmniVLA スマートフォン移植 仕様書 (v0.3)

> 実装: `app/` (Flutter) + Pi 側受け口 `rvln_edge/edge_action_grpc_node.py`
> (`vla.sh run omnivla_edge_mobile --mode cmd_vel`)。Phase 1–4 実装済み、残るは
> Phase 5 (実機統合) — §6 参照。ONNX モデル・CLIP 語彙が未配置の環境でも
> ダミー軌道で end-to-end 動作する。ブラウザ版の姉妹実装は `web_port_spec.md`
> (同じデータ契約、トランスポートは WebSocket)。

## 0. 目的とスコープ

OmniVLA-edge ポリシーを **スマートフォン上でオンデバイス推論** させ、モーター指令だけを
Raspberry Pi (raspicat) が実行する構成を作る。既存の **Path 3（Jetson が推論 / Pi が制御）**
の Jetson を**スマホに置き換えた**トポロジーに相当する。

```
┌─────────────────── スマートフォン (Android / iOS) ───────────────────┐
│  カメラ ─▶ 前処理 ─▶ [OmniVLA-edge (ONNX)] ─▶ action chunk (8×4)      │
│  ゴールUI(text/pose/image) ─┘        ▲                                │
│  CLIP text encoder (ONNX, キャッシュ) ┘                              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ 無線 (gRPC over Wi-Fi/LTE)
                                │ action chunk を送信
┌───────────────────────────────▼──────────────────────────────────────┐
│  Raspberry Pi (raspicat)                                              │
│  gRPC受信 ─▶ Path化(既存 OmniVLAEdgeAdapter相当) ─▶ pure-pursuit       │
│           ─▶ /cmd_vel ─▶ モーター                                     │
└───────────────────────────────────────────────────────────────────────┘
```

### 確定事項（ユーザー合意済み）
- 推論場所: **オンデバイス**（スマホ上）
- プラットフォーム: **クロスプラットフォーム（Android / iOS 両対応）**
- 役割分担: **スマホ = カメラ取得 + 推論**、**Pi = モーター指令のみ**
- ゴール指定モード: **text / pose / image すべて対応**

---

## 1. 移植対象コンポーネント

| 元 (repo) | 役割 | 移植先 |
|-----------|------|--------|
| `OmniVLA_edge` モデル (`models/omnivla_edge_model.py`) | 本体ポリシー | **ONNX 化してスマホで実行** |
| CLIP ViT-B/32 text encoder | text ゴールの埋め込み | **ONNX 化してスマホで実行**（ゴール変更時のみ、キャッシュ） |
| `omnivla_edge_engine.py` の前後処理 | リサイズ/正規化/リングバッファ/ゴールtensor組み立て | **アプリ側ネイティブ実装で再現**（下記 §3 が正解定義） |
| `OmniVLAEdgeAdapter` (Pi 側, torch不要) | chunk → nav_msgs/Path | Pi 側にそのまま残す |
| pure-pursuit follower | Path → /cmd_vel | Pi 側にそのまま残す |

**モデル weight**: `models/omnivla-edge/omnivla-edge.pth`（bare state_dict）を ONNX へ変換。
CUDA 依存は `forward` 内の `tensor.get_device()` 1 箇所のみ。ONNX export 時にトレースで
除去 / パッチして CPU/NPU/GPU で動くようにする。

---

## 2. アーキテクチャ決定（推奨・要確認）

| 項目 | 決定 | 理由 / 代替案 |
|------|------|--------------|
| **推論ランタイム** | **ONNX Runtime Mobile**（確定） | Android(NNAPI/XNNPACK) + iOS(CoreML EP) を単一グラフでカバー。Flutter プラグイン `onnxruntime 1.4.1` 採用済み。代替: ExecuTorch, TFLite |
| **アプリFW** | **Flutter**（確定, 3.44.4） | camera / onnxruntime / grpc-dart が揃う。iOS/Android UI 共通化。 |
| **スマホ↔Pi 通信** | **gRPC（スマホ=client, Pi=server）** | 新規 `proto/edge_action.proto` の `EdgeActionService.StreamActions`。**coalesce+pace** をアプリ側 `CoalescingSender` で実装済。 |
| **ゴール入力** | **スマホ UI**（確定） | text 欄 / 数値で pose(x,y,θ) / 現フレーム取得で image。`GoalPanel` 実装済。 |
| **pose 座標系** | **スマホ UI で直接指定**（確定, 未決事項C解決） | ロボット相対メートルを UI で入力。VIO/odom 連携は将来拡張。 |
| **搭載形態** | **raspicat に固定搭載**（確定, 未決事項E解決） | カメラ=ロボット前方視点。`centerCropToAspect` で学習時 FOV に寄せる。 |
| **推論精度** | fp32（初期）→ 後に fp16/int8 量子化 | まず正解一致、その後最適化 |

---

## 3. データ契約（正解定義 — `omnivla_edge_engine.py` と一致必須）

### 3.1 モデル入力（ONNX グラフの 7 入力）
| 名前 | 形状 | 内容 |
|------|------|------|
| `obs_images` | `(1, 18, 96, 96)` | 直近 6 フレーム(=context_size 5 + 現在)を ImageNet正規化・CHW・古い順で連結。不足時は最古フレームで前詰め |
| `goal_pose` | `(1, 4)` | `(x_fwd/0.1, y_left/0.1, cos θ, sin θ)`。予測 waypoint と同じロボット座標系 (x=前方, y=左) を 0.1m 単位に正規化。半径 30m でクランプ。pose モード以外は 0 埋め |
| `map_images` | `(1, 9, 96, 96)` | `cat(黒96, 黒96, 現在フレーム)`。衛星地図はゼロ埋め |
| `goal_image` | `(1, 3, 96, 96)` | image モード時は目標画像を96化・正規化。他は黒画像 |
| `modality_id` | `(1,)` int | text=7 / pose=4 / image=6 |
| `feat_text` | `(1, 512)` | CLIP text encoder 出力。text モード以外は空文字 `''` の埋め込み |
| `cur_large` | `(1, 3, 224, 224)` | 現在フレームを 224 化・正規化（FiLM 変調用） |

### 3.2 前処理（アプリ側で厳密再現）
- リサイズ: `cv2.INTER_AREA` 相当（縮小）。96×96 と 224×224 の 2 系統。
- 正規化: `(rgb/255 - mean)/std`, mean=`[0.485,0.456,0.406]`, std=`[0.229,0.224,0.225]`。
- チャネル順: **RGB**、レイアウト CHW。
- リングバッファ: 最大 `context_size+1 = 6` フレーム、古い順、現在最後。ゴール切替時は reset。

### 3.3 モデル出力
- `action_pred`: `(1, 8, 4)` = 各 waypoint `(x, y, cos θ, sin θ)`、**単位は waypoint-spacing (0.1 m/unit)**。
- スマホ側で x,y に 0.1 を掛けてメートル化してから送る (`scaled_to_m=true`, §5-D で決着)。Pi 側は `scaled_to_m=false` (生 spacing 単位) も受けられる。
- `dist_pred`, `mask` は v1 では未使用（送らない）。

### 3.4 CLIP テキスト経路
- トークナイザ: CLIP BPE（`clip.tokenize`, max_len=77, truncate=True）。→ ネイティブ実装（Kotlin/Swift/Dart）またはトークナイズ済み配列を ONNX に渡す。
- text encoder 出力 512-dim を fp32 化。**プロンプト文字列でキャッシュ**（ゴール変更時のみ再計算）。

---

## 4. スマホ↔Pi インターフェース（gRPC — 実装済み）

**正定義は `proto/edge_action.proto`**（PC 間の ROS 2 通信とは独立の
サービス）。`EdgeActionService.StreamActions(stream ActionChunk) returns
(stream ControlAck)` の双方向 stream で、スマホ = client / Pi = server。
フィールドの意味は proto のコメントが正。v0.2 草案からの差分は
`from_model` (bool, field 8) の追加のみ — ONNX 未配置のダミー軌道を送って
いることを Pi 側 ack (`status: "ok-dummy"`) で可視化するためのフラグ。

### Pi 側の責務（`edge_action_grpc_node.py` が実装）
- gRPC server として action chunk を受信 → `trajectory_to_path` で
  `nav_msgs/Path` 化 → 既存 `path_follower_node` が追従 → `/cmd_vel(_vla)`。
- **ウォッチドッグ**: `chunk_max_age_sec` (既定 1.0s) 以内に新しい chunk が
  来なければ空 Path を発行してモーター停止（既存 embedding_max_age の思想を踏襲）。
- ゴール切替(`goal_id`変化)はログに記録（追従は常に最新 Path 基準）。

実装メモは §6「Phase 4 実装メモ」を参照。

---

## 5. 主要リスク / 未決事項

### リスク
1. **カメラ・ドメインギャップ**: OmniVLA-edge は raspicat カメラの FOV/画角で学習。スマホカメラは画角・レンズ歪みが異なり、そのままでは挙動が劣化する恐れ。→ スマホ映像を学習時 FOV に**クロップ/リサイズして合わせる**前処理が要。要実測。
2. **オンデバイス性能**: EfficientNet-B0 ×3 + Transformer。目標 2–5 Hz。NPU/GPU EP で未達なら量子化(int8) or 解像度検討。要ベンチ。
3. **`get_device()` の ONNX 化**: forward の GPU 依存箇所を export 時に確実に除去できるか要検証（最悪モデル側を小改造）。
4. **数値一致**: cv2.INTER_AREA / ImageNet 正規化 / CLIP BPE をネイティブ再現した際の PyTorch 参照との誤差。ゴールデンテスト必須。
5. **無線遅延・切断**: Wi-Fi/LTE 経由の chunk 送信。Pi ウォッチドッグで安全側に倒す。

### 決定事項（旧・未決 A–F）
- **A. ランタイム**: ONNX Runtime に確定。
- **B. アプリFW**: Flutter に確定。
- **C. pose 座標系**: スマホ UI で直接指定に確定。
- **E. 搭載形態**: raspicat に固定搭載に確定。
- **D. メートル化の責務**: 決着 — アプリ (`GrpcEdgeClient`) はメートル化して送る
  (`scaled_to_m=true`)。web 版は生 spacing 単位 (`scaled_to_m=false`) で送り
  Pi 側が ×0.1 する。Pi 側 `decode_action_chunk` は両対応のまま維持。
- **F. 正解一致の検証手段**: ONNX 側は解決 — `app/scripts/export_*.py` が export
  時に onnxruntime と PyTorch の出力を突き合わせる (Phase 1 完了)。アプリ/web の
  前処理・CLIP BPE の実フレームでの完全ゴールデン照合は Phase 5 の残タスク。

---

## 6. 実装フェーズと進捗

| Phase | 内容 | 状態 |
|-------|------|------|
| **Phase 3 — アプリ骨組み** | カメラ取得→前処理→(ONNX or ダミー)推論→chunk 可視化→送信。ゴール UI 3 モード。 | ✅ **実装済** (`app/`) |
| **Phase 1 — モデル変換** | `omnivla-edge.pth` + CLIP を ONNX 化。`assets/models/*.onnx` に配置。PyTorch 参照と出力一致を検証。 | ✅ **実装済** (`app/scripts/export_{omnivla_edge,clip_text}_onnx.py` — export 時に PyTorch と出力照合。`quantize_onnx.py` も有) |
| **Phase 2 — 前処理ゴールデン** | Dart の前処理/リングバッファ/CLIP BPE を PyTorch `infer_chunk` と数値突き合わせ。CLIP 語彙 `assets/clip/` 配置。 | 🟡 語彙配置済・単体テスト有。実フレームでの完全ゴールデンは Phase 5 と合流 |
| **Phase 4 — Pi 側 server** | `EdgeActionService` server + ウォッチドッグ + pure-pursuit + sim 走行。アプリ側 `GrpcEdgeClient` を差し込み。 | ✅ **実装済** (`edge_action_grpc_node.py` + `GrpcEdgeClient`; `vla.sh run omnivla_edge_mobile --mode cmd_vel` で起動、e2e 疎通確認済) |
| **Phase 5 — 実機統合** | スマホ↔Pi 無線、実機 raspicat で end-to-end、FOV/性能/安全チューニング。 | ⬜ 未着手 |

### Phase 3 実装マップ (`app/lib/src/`)

| ファイル | 役割 | §対応 |
|----------|------|-------|
| `config.dart` | 固定ハイパラ・正規化定数 | §3 |
| `preprocessing.dart` | resize/正規化/リングバッファ/pose ベクトル | §3.2 |
| `camera_image_utils.dart` | CameraImage→RGB, FOV クロップ | §5-1 |
| `clip_tokenizer.dart` | CLIP BPE (語彙アセット待ち) | §3.4 |
| `inference/ort_runner.dart` | ONNX Runtime ラッパー (未配置ならフォールバック) | §3.1 |
| `omnivla_engine.dart` | `infer_chunk` 移植・オーケストレーション | §3 |
| `action_chunk.dart` | (8,4) 出力・メートル換算 | §3.3 |
| `grpc/edge_action_client.dart` | Pi 送信 (coalesce+pace) | §4 |
| `ui/*` | カメラプレビュー・ゴール入力・パス可視化 | — |

### Phase 4 実装メモ

- stub 生成は `scripts/gen_proto.sh` に統合 (Python は gitignore、Dart は
  `app/lib/src/grpc/gen/` にコミット。要 `dart pub global activate protoc_plugin`)。
- Pi 側 server は `rvln_edge/edge_action_grpc_node.py`。Web 版ブリッジ
  `edge_action_ws_node.py` の gRPC 双子で、同じ設計を守る
  (受信スレッドはロック付きスロットへ置くだけ / publish は ROS タイマ /
  `chunk_max_age_sec` 超過で空 Path safe-stop)。既定ポート 50061
  (`vla.sh` の `EDGE_ACTION_PORT`)。
- アプリ側は `GrpcEdgeClient` (AppBar の Wi-Fi アイコンから Pi の IP[:PORT] を
  指定して接続)。x,y はスマホ側でメートル化して送る (`scaled_to_m=true`,
  未決事項 D の推奨解を採用)。切断時はストリームを張り直すだけで送信ループは
  止めない。
- 起動: `vla.sh run omnivla_edge_mobile --mode cmd_vel`
  (EdgeActionService + path_follower、既定は `/cmd_vel_vla` = 非モーター。
  `--drive-motors` で実モーター)。

### 次アクション
1. **Phase 5**: 実機 raspicat + スマホ Wi-Fi で end-to-end (`--drive-motors`)、
   FOV/性能/安全チューニング。前処理・CLIP BPE の実フレームゴールデン照合 (§5-F 残)
   もここで行う。
