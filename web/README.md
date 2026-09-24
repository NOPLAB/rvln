# rvla-web

OmniVLA-edge を**ブラウザ内で推論**する Web ポート (React + Next.js 静的エクスポート)。
`app/` (Flutter) と同じ ONNX 資産・前処理・データ契約 (docs/design/mobile_port_spec.md §3)
を使い、推論は onnxruntime-web の **WebGPU EP** (なければ wasm) で動く。
action chunk (8×4) は WebSocket で Pi の `edge_action_ws_node` へ送れる
(docs/design/web_port_spec.md 参照)。

## セットアップ

```bash
cd web
pnpm install          # postinstall が onnxruntime-web の wasm/mjs を public/ort/ へコピー
pnpm sync-assets      # app/assets からモデル (~590MB) と CLIP 語彙を public/ へコピー
pnpm dev              # http://localhost:3000
```

モデル未配置でも動く (ダミー軌道・琥珀色表示)。実推論には
`app/assets/models/{omnivla_edge,clip_text}.onnx` が必要
(`app/scripts/export_*.py` で生成したもの)。

## ビルド (SSG)

```bash
pnpm build            # out/ に完全静的サイトを出力
pnpm preview          # out/ をローカル配信
```

サブパス配信 (GitHub Pages 等) は `NEXT_PUBLIC_BASE_PATH=/repo-name pnpm build`。

## GitHub Pages デプロイ

`.github/workflows/deploy-pages.yml` が main への `web/**` push で自動デプロイする。
モデルは git ではなく **GitHub Release `models-v1` のアセット**からビルド時に
`public/` へ配置される。モデルを更新したら:

```bash
gh release upload models-v1 app/assets/models/*.onnx app/assets/clip/*.gz --clobber
gh workflow run deploy-pages.yml
```

Pages は https 配信のため Pi への `ws://` 接続は塞がれる (ブラウザ内推論のデモ用)。
Pi と繋ぐ実運用は localhost 配信で行うこと (下記)。

## 動作要件と注意

- **WebGPU**: Chrome/Edge 113+ (chrome://gpu で確認)。無ければ wasm に自動フォールバック
  (シングルスレッドなので遅い)。
- **localhost 配信で使うこと**。localhost は secure context なので http でも
  カメラ・WebGPU・`ws://` Pi 接続がすべて成立する。https 配信だと `ws://` が
  mixed content で塞がれ、http の LAN IP 配信だとカメラ/WebGPU が使えない。
  スマホ実機で使う段階では https + wss 化が必要 (Phase 5)。
- モデルは初回取得後 Cache Storage に保存される (UI の「モデルキャッシュ削除」で破棄)。
- カメラが無い環境は「動画ファイル」ソースでループ再生して推論できる。

## テスト

```bash
pnpm lint             # Biome (lint + format チェック; 自動修正は pnpm lint:fix)
pnpm typecheck        # tsc --noEmit
pnpm test             # vitest (前処理 / fp16 / coalesce+pace / トークナイザ構造)
```

CI (`.github/workflows/ci-web.yml`) は lint → typecheck → test → build を実行する。

CLIP トークナイザの完全ゴールデン照合と ONNX 出力の PyTorch 参照一致は
Phase 2/F と同じく GPU 環境の参照実装側で行う。

## 実装対応表

`src/lib/*` は `app/lib/src/*` (Dart) の 1:1 移植。定数 (`src/lib/config.ts`) は
エンジン/Flutter と**一致必須** — 値を変えるときは 3 箇所同時に。
