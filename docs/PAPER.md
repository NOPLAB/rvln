# ナビゲーションモデルの論文・実装・重み

最終確認: 2026-09-24。VLNと、比較に役立つ視覚ナビゲーションモデルを掲載する。「重み」は確認できた公開チェックポイントへのリンク。**未確認**は公式の公開先を確認できなかった場合、**—**は単独の論文や重みがない場合を示す。公開重みがあっても、このリポジトリで推論できるとは限らない。

| モデル | ペーパー | プロジェクト・コード | 重み | 簡単な概要 |
| --- | --- | --- | --- | --- |
| OmniVLA | [論文](https://arxiv.org/abs/2509.19480) | [プロジェクト](https://omnivla-nav.github.io/) / [コード](https://github.com/NHirose/OmniVLA) | [cloud](https://huggingface.co/NHirose/omnivla-original) / [edge](https://huggingface.co/NHirose/omnivla-edge) | 言語・目標画像・2D目標姿勢を組み合わせて指定できる移動ロボット向けモデル。 |
| AsyncVLA | [論文](https://arxiv.org/abs/2602.13476) | [プロジェクト](https://asyncvla.github.io/) / [コード](https://github.com/NHirose/AsyncVLA) | [release](https://huggingface.co/NHirose/AsyncVLA_release) | 遠隔の大きなVLAと機体側の軽量な反応制御を非同期に組み合わせる。 |
| movla | —（公開論文なし） | [社内コード](https://github.com/NOPLAB/movla) | 非公開（学習ホストから取得） | このリポジトリで利用する独自の移動ロボット用ポリシー。 |
| NaVILA | [論文](https://arxiv.org/abs/2412.04453) | [プロジェクト](https://navila-bot.github.io/) / [コード](https://github.com/AnjieCheng/NaVILA) | [8B](https://huggingface.co/a8cheng/navila-llama3-8b-8f) | 画像履歴と言語から前進・旋回・停止の短い指示を生成する。 |
| NaVIDA | [論文](https://arxiv.org/abs/2601.18188) | [コード](https://github.com/waynechu1021/NAVIDA) | [NaVIDA](https://huggingface.co/waynechu/NaVIDA) | 行動による視覚変化の学習と行動チャンクで、RGBからのVLNを改善する。 |
| InternVLA-N1 / DualVLN | [論文](https://arxiv.org/abs/2512.08186) | [InternNav](https://github.com/InternRobotics/InternNav) | [RGB DualVLN](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN) | 高位の画像上目標推論と高速な局所軌道生成を分けた二系統のVLN。 |
| StreamVLN | [論文](https://arxiv.org/abs/2507.05240) | [プロジェクト](https://streamvln.github.io/) / [コード](https://github.com/InternRobotics/StreamVLN) | [評価用](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3) / [実機向け](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world) | 動画を逐次処理し、短期対話履歴と圧縮した長期記憶を使う。 |
| OmniNav | [論文](https://arxiv.org/abs/2509.25687) | [コード](https://github.com/amap-cvlab/OmniNav) | [Flow Matching](https://www.modelscope.ai/models/chongchongjj/OmniNav_Flow/) / [Action Former](https://www.modelscope.ai/models/chongchongjj/OmniNav/) | 探索とVLNを統合し、R2R・RxR・OVON向けの行動を生成する。 |
| Uni-NaVid | [論文](https://arxiv.org/abs/2412.06224) | [コード](https://github.com/jzhzhang/Uni-NaVid) | [7B](https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/uninavid-7b-full-224-video-fps-1-grid-2) | 動画を入力として、指示追従・物体探索・追跡を一つのモデルで扱う。 |
| Qwen-RobotNav | [論文](https://arxiv.org/abs/2606.18112) | [公式リポジトリ](https://github.com/QwenLM/Qwen-RobotNav) | 未公開（公式に公開予定なし） | タスクモードと観測設定を切り替え、複数の移動課題を共通モデルで扱う。 |
| NavFoM | [論文](https://arxiv.org/abs/2509.12129) | [プロジェクト](https://pku-epic.github.io/NavFoM-Web/) | 未確認 | 脚・車輪・ドローン・車両をまたぎ、VLN・探索・追跡などを学習する。 |
| ABot-N1 | [論文](https://arxiv.org/abs/2607.10383) | [プロジェクト](https://amap-cvlab.github.io/ABot-Navigation/ABot-N1/) / [ベンチマークコード](https://github.com/amap-cvlab/ABot-Navigation) | 未確認 | 言語推論と画像上の目標点から連続ウェイポイントを作る。 |
| OmTrackVLA | —（独立論文なし。参考: [TrackVLA](https://arxiv.org/abs/2505.23189)） | [コード](https://github.com/om-ai-lab/OmTrackVLA) | [0.6B](https://huggingface.co/omlab/OmTrackVLA-0.6B) | 単眼動画と言語指示から追跡・追従用の短いウェイポイントを生成する。 |
| ViNT | [論文](https://arxiv.org/abs/2306.14846) | [プロジェクト](https://visualnav-transformer.github.io/) / [コード](https://github.com/robodhruv/visualnav-transformer) | [公式チェックポイント集](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing) | 目標画像などを条件に進む、複数ロボットのデータで学習した視覚ナビゲーション基盤モデル。 |
| NoMaD | [論文](https://arxiv.org/abs/2310.07896) | [プロジェクト](https://general-navigation-models.github.io/nomad/) / [コード](https://github.com/robodhruv/visualnav-transformer) | [公式チェックポイント集](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing) | 目標画像への移動と未知環境の探索を一つの拡散ポリシーで扱う。 |
| LiveVLN | [論文](https://arxiv.org/abs/2604.19536) | [コード](https://github.com/NIneeeeeem/LiveVLN) | —（既存VLNモデルの重みを利用） | 推論中も実行可能な行動を維持するランタイム手法。単独の学習済みモデルではない。 |

## このリポジトリでの扱い

OmniVLA、AsyncVLA、movla、NaVILA、NaVIDAには推論バックエンドがある。これらの重みは `scripts/download_checkpoints.sh` から取得できる。StreamVLNの実機向け重みとInternVLA-N1 DualVLNは同スクリプトに取得先だけ登録済みで、ROS推論にはまだ接続していない。入力・行動表現と検証上の注意は [VLNモデル調査](memo/vln-model-survey-2026.md) を参照。
