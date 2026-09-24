/**
 * OmniVLA-edge の固定ハイパーパラメータと前処理定数。
 *
 * `src/rvla_core/rvla_core/omnivla_edge_engine.py` の
 * `_MODEL_PARAMS` / モジュール定数、および `app/lib/src/config.dart` と
 * **一致必須**。値を変えると omnivla-edge.pth との整合が崩れる。
 * docs/design/mobile_port_spec.md §3 が正解定義。
 */

const contextSize = 5;

export const OmniVlaConfig = {
  /** 観測履歴・ゴール画像・マップの一辺 (px)。 */
  obsSize: 96,
  /** FiLM 変調用の大きい現在フレームの一辺 (px)。 */
  largeSize: 224,

  /** context_size。履歴フレーム数は contextSize + 1 (=現在)。 */
  contextSize,
  /** リングバッファ長 = 直近フレーム数 (6)。 */
  historyLen: contextSize + 1,

  /** len_traj_pred。1推論あたりの waypoint 数。 */
  lenTrajPred: 8,
  /** 各 waypoint の次元 (x, y, cos, sin)。 */
  actionDim: 4,

  /** waypoint-spacing (m/unit)。モデル出力の x,y はこの単位。 */
  metricWaypointSpacing: 0.1,
  /** pose ゴールの距離クランプ (m)。thres_dist。 */
  goalDistThresholdM: 30.0,

  /** CLIP text encoder の出力次元。 */
  clipTextDim: 512,
  /** CLIP BPE のトークン列長 (context_length)。 */
  clipContextLength: 77,

  imagenetMean: [0.485, 0.456, 0.406],
  imagenetStd: [0.229, 0.224, 0.225],

  // モダリティ id (run_omnivla_edge.py と一致)
  modalityPose: 4,
  modalityImage: 6,
  modalityText: 7,
} as const;
