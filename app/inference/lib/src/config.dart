/// OmniVLA-edge の固定ハイパーパラメータと前処理定数。
///
/// これらは `src/rvla_core/rvla_core/omnivla_edge_engine.py`
/// の `_MODEL_PARAMS` / モジュール定数と **一致必須**。値を変えると
/// omnivla-edge.pth との整合が崩れる。docs/design/mobile_port_spec.md §3 が正解定義。
library;

class OmniVlaConfig {
  // --- 画像サイズ ---
  /// 観測履歴・ゴール画像・マップの一辺 (px)。
  static const int obsSize = 96;

  /// FiLM 変調用の大きい現在フレームの一辺 (px)。
  static const int largeSize = 224;

  // --- 履歴 ---
  /// context_size。履歴フレーム数は contextSize + 1 (=現在)。
  static const int contextSize = 5;

  /// リングバッファ長 = 直近フレーム数 (6)。
  static const int historyLen = contextSize + 1;

  // --- 出力 (action chunk) ---
  /// len_traj_pred。1推論あたりの waypoint 数。
  static const int lenTrajPred = 8;

  /// 各 waypoint の次元 (x, y, cos, sin)。
  static const int actionDim = 4;

  // --- スケール ---
  /// waypoint-spacing (m/unit)。モデル出力の x,y はこの単位。
  static const double metricWaypointSpacing = 0.1;

  /// pose ゴールの距離クランプ (m)。thres_dist。
  static const double goalDistThresholdM = 30.0;

  // --- CLIP ---
  /// text encoder の出力次元。
  static const int clipTextDim = 512;

  /// CLIP BPE のトークン列長 (context_length)。
  static const int clipContextLength = 77;

  // --- ImageNet 正規化 ---
  static const List<double> imagenetMean = [0.485, 0.456, 0.406];
  static const List<double> imagenetStd = [0.229, 0.224, 0.225];

  // --- モダリティ id (run_omnivla_edge.py と一致) ---
  static const int modalityPose = 4; // pose only
  static const int modalityImage = 6; // image only
  static const int modalityText = 7; // language only
  static const int modalityTextPose = 8; // 未使用(v1)
}
