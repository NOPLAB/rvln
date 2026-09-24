# OmniVLA Web ポート (WebGPU) 仕様書 (v1.0)

> 実装: `web/` (React + Next.js 静的エクスポート) と
> `src/rvla_edge/rvla_edge/edge_action_ws_node.py` (Pi 側受け口)。
> データ契約は mobile_port_spec.md §3 (7 ONNX 入力 / 出力 (1,8,4)) をそのまま共有する。

## 0. 目的とスコープ

`app/` (Flutter) と同じ OmniVLA-edge ONNX 資産を**ブラウザ内** (WebGPU、なければ
wasm) で推論し、action chunk を WebSocket で Pi へ送ってモーターを駆動する。
Pi 側受け口は `edge_action_ws_node.py` — mobile_port_spec Phase 4 の gRPC 版
(`edge_action_grpc_node.py`) の **WebSocket 双子**で、ブラウザは素の gRPC を
話せないため web 経路では WS が正式トランスポート。両ノードは同じ
スロット+ウォッチドッグ設計を保つ。

```
┌────────── ブラウザ (localhost 配信) ──────────┐
│ カメラ/動画 → 中央クロップ → INTER_AREA 相当   │
│ リサイズ + ImageNet 正規化 → RingBuffer(6)     │
│ → onnxruntime-web [WebGPU EP → wasm]           │
│ → (8,4) chunk → CoalescingSender (≥100ms)      │
└──────────────┬────────────────────────────────┘
               │ WebSocket (JSON, values は fp16+base64)
┌──────────────▼────────────────────────────────┐
│ Pi: edge_action_ws_server (rvla_edge)  │
│ 受信 → fp16 decode → trajectory_to_path        │
│ → /rvla/predicted_path                 │
│ ウォッチドッグ: chunk_max_age 超過 → 空 Path    │
│ → 既存 path_follower が pure-pursuit / safe-stop│
└────────────────────────────────────────────────┘
```

## 1. web/ の構成

- Next.js App Router + `output: 'export'` — `out/` は完全静的。推論は全てクライアント。
- `web/src/lib/*` は `app/lib/src/*` (Dart) の 1:1 移植。**`config.ts` は
  `rvla_core/omnivla_edge_engine.py` / `app/lib/src/config.dart` と定数一致必須**。
- リサイズは cv2.INTER_AREA 相当の面積平均を TS で自前実装
  (`preprocessing.ts`)。canvas `drawImage` はブラウザ依存で数値が揺れるため不使用。
- モデル (`public/models/*.onnx`, 計 ~590MB) と CLIP 語彙は git 管理外。
  ローカルは `pnpm sync-assets` で `app/assets/` からコピー、GitHub Pages は
  deploy-pages workflow が GitHub Release `models-v1` のアセットから配置する。
  初回取得後は Cache Storage に保存。
- ONNX 未配置なら Flutter 版と同じダミー弧軌道 (琥珀色) にフォールバック。
- EP: `navigator.gpu` があれば `webgpu`、なければ `wasm` (このとき
  `ort.env.wasm.proxy = true` で worker 実行)。int64 入力は `BigInt64Array`。

## 2. WS プロトコル (edge_action.proto と意味的に同一)

`proto/edge_action.proto` の `ActionChunk` / `ControlAck` を JSON にしたもの
(スマホ経路は同 proto の gRPC 実装が稼働済み — mobile_port_spec §4)。
フィールド名は proto と揃えてあり、grpc-web 等へ移行する場合もそのまま写せる。

```jsonc
// ブラウザ → Pi (テキストフレーム)
{
  "type": "action_chunk",
  "frame_id": 42,            // 送信側の連番
  "capture_time_ms": 0,      // カメラ取得時刻 (epoch ms)
  "num_tokens": 8,
  "embed_dim": 4,            // (x, y, cos, sin)
  "values_fp16_b64": "...",  // fp16 LE 64byte を base64。web: packFp16 /
                             // Pi: rvla_proto.conversions で往復
  "scaled_to_m": false,      // false なら Pi 側で ×waypoint_spacing (0.1)
  "goal_id": "text:go to the door"
}
// Pi → ブラウザ
{ "type": "ack", "frame_id": 42, "following": true, "status": "ok" }
```

- スケーリングは mobile_port_spec 未決事項 D の「両対応」: web は
  `scaled_to_m=false` で生 spacing 単位を送り、Pi 側 `trajectory_to_path`
  (既存コード) が ×0.1 する。
- malformed は `status: "error: ..."` の ack で返し、Path は発行しない。

## 3. Pi 側 `edge_action_ws_server`

- 通常の rclpy Node (LifecycleNode ではない — ブリッジは状態遷移を持たず、
  安全機構は follower 側に既にある)。`websockets` は遅延 import。
- **受信スレッドから publish しない**: ws スレッドは最新 chunk をロック付き
  スロットへ置き、ROS タイマ (20Hz) が publish する (coalesce の受け側)。
- **ウォッチドッグ**: `chunk_max_age_sec` (既定 1.0s) 新規 chunk が無ければ
  空 Path を 1 回発行 → follower が「empty path (edge safe-stop)」で停止。
- 起動: `ros2 launch rvla_bringup phone_ws.launch.py`
  (follower 出力は既定 `/cmd_vel_vla`。実機駆動は `cmd_vel_topic:=/cmd_vel` を明示)。

## 4. 動作環境の制約

- **web は localhost 配信で使う**。localhost は secure context なので http でも
  カメラ + WebGPU + `ws://` (Pi) が同時に成立する。https 配信は `ws://` が
  mixed content で塞がれ、http の LAN IP 配信はカメラ/WebGPU が使えない。
  スマホ実機で web 版を使う場合は https + wss 化が必要 (Phase 5 相当)。
- COOP/COEP を付けない静的配信のため wasm はシングルスレッド。主経路は WebGPU。

## 5. テスト

- web: `pnpm test` (vitest) — 面積平均リサイズ / 正規化 / リングバッファ /
  fp16 pack / coalesce+pace / CLIP BPE の構造検証 (実語彙使用)。
- Pi: `scripts/vla.sh test src/rvla_edge/test/test_edge_action_ws.py`
  — decode (spacing / scaled_to_m / malformed) とウォッチドッグ。gRPC 双子は
  `test_edge_action_grpc.py`。
- ONNX 出力・前処理の PyTorch 参照とのゴールデン照合は mobile_port_spec
  Phase 2/F と共通 (GPU 環境側)。

## 6. 残作業

- [ ] 実機 Pi + ブラウザの end-to-end (Phase 5 相当の統合)
- [ ] `vla.sh` に phone_ws.launch.py の docker 起動を追加
      (gRPC 版は `vla.sh run omnivla_edge_mobile --mode cmd_vel` として実装済み —
      WS 版だけ手動 `ros2 launch` のまま)
- [x] gRPC (`edge_action.proto`) 本実装 — スマホ経路に `edge_action_grpc_node.py`
      として実装済み。web 経路はブラウザ制約により WS を正式トランスポートとする
- [ ] https + wss 化 (スマホから web 版を使う場合)
