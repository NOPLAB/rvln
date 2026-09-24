# VLN model survey (2026-09-24)

This workspace predicts short robot-frame paths from a camera and a goal. The relevant
task is vision-and-language navigation (VLN), especially continuous execution (VLN-CE).
Manipulation scores on LIBERO/RoboTwin do not rank navigation ability. Reported SR/SPL
below are authors' results on the named split, not a common Raspicat benchmark.

## Shortlist

| Model | Public inference/weights | Task and output | Decision for this workspace |
| --- | --- | --- | --- |
| [Qwen-RobotNav](https://github.com/QwenLM/Qwen-RobotNav) | Report only; authors explicitly say weights will not be released | Instruction, object, point, tracking; eight `(x,y,theta)` waypoints. Reported R2R val-unseen SR 72.1, SPL 66.6 (8B). | Strong research reference; cannot add a reproducible backend. |
| [InternVLA-N1 / InternNav](https://github.com/InternRobotics/InternNav) | Code and System 2/dual-system weights | VLN-CE and physical navigation; dual fast/slow policy. RGB DualVLN reports R2R SR 64.3, SPL 58.5; RxR SR 61.4, SPL 51.8. | Highest-priority next integration, but the dual policy and temporal state need a separate runtime adapter. |
| [NavFoM](https://pku-epic.github.io/NavFoM-Web/) | Paper and project page; no code/checkpoint link on the official project page | Cross-embodiment VLN, search, tracking and driving; trajectory head. | Architecturally close to the ROS path contract, but reproducible integration depends on an official release. |
| [OmniNav](https://github.com/amap-cvlab/OmniNav) | Code and 2026-09 Flow Matching weights on ModelScope | R2R/RxR/OVON; continuous waypoints with optional slow-fast planning. | Strong second path-producing candidate. Its released scripts depend on task-specific Habitat versions; separate model inference from simulator before connecting ROS. |
| [StreamVLN](https://github.com/InternRobotics/StreamVLN) | Code plus benchmark and real-world checkpoints | Online video and language to discrete actions; authors report R2R SR 56.4, SPL 50.2, RxR SR 54.4, SPL 45.4 on their stated VLN-CE versions. | Valuable for persistent streaming context. Real-world Go2 deployment guide exists. License is CC BY-NC-SA 4.0. |
| [NaVILA](https://github.com/AnjieCheng/NaVILA) | Official code and [8B checkpoint](https://huggingface.co/a8cheng/navila-llama3-8b-8f) | Eight video frames and instruction to `move forward N cm`, `turn left/right N degree`, or `stop`. | **Added first**: explicit metric commands can be converted to existing Path contract. Low-level collision avoidance is outside the released language policy. |
| [NaVIDA](https://github.com/waynechu1021/NAVIDA) | Official code and [checkpoint](https://huggingface.co/waynechu/NaVIDA) (authors call it 3B; HF metadata says 4B) | RGB history/current frame and language to short action chunks. Authors report R2R SR 61.4/SPL 54.7 and RxR SR 57.4/SPL 49.6 on val-unseen. | **Added**: first action maps to Path. Chunk scheduling and real-world results remain unvalidated. |
| [Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid) | Training/evaluation code and [official 7B checkpoint](https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/uninavid-7b-full-224-video-fps-1-grid-2) | Video policy across navigation tasks. | Candidate for a future runtime adapter; upstream checkpoint is published, but ROS inference is not yet connected. |
| [OmTrackVLA](https://github.com/om-ai-lab/OmTrackVLA) | Code and [0.6B checkpoint](https://huggingface.co/omlab/OmTrackVLA-0.6B) | Text-conditioned visual tracking and following, short waypoints. | Good person-following specialty model, not a general route-instruction replacement. |
| [ABot-N1](https://github.com/amap-cvlab/ABot-Navigation) | Benchmark code and datasets; no model checkpoint link found in official repository | Five navigation tasks; strong reported point/POI/object goal scores. | Watch for checkpoint release. |
| [ViNT / NoMaD](https://github.com/robodhruv/visualnav-transformer) | Code and checkpoints, TurtleBot/LoCoBot deployment | Image/topological goal or exploration; metric waypoints. | Useful camera-navigation baseline, but not a direct free-form language model. |
| [LiveVLN](https://github.com/NIneeeeeem/LiveVLN) | Runtime framework; uses another model's weights | Overlaps sensing, inference and execution for StreamVLN/NaVIDA. Authors report 77.7% less waiting time for StreamVLN in their setting. | Evaluate as a latency layer after establishing a correct base policy and stop behavior. |

## Integration order

1. **NaVILA**: a small, explicit action space and published evaluation call make the
   ROS contract checkable. Use `backend:=navila`, `adapter_kind:=omnivla` (path-only).
2. **NaVIDA**: run the RGB policy with `backend:=navida`; compare its first-action
   policy against NaVILA on identical camera episodes.
3. **InternVLA-N1**: evaluate its RGB DualVLN variant and System 2 versus fast system
   separately. Its reported benchmark is more directly relevant than manipulation SOTA.
4. **OmniNav**: extract pure checkpoint inference from the Habitat evaluation wrapper;
   document coordinate frame, metres per waypoint, history and reset semantics.
5. **StreamVLN**: compare in a common offline replay, particularly instruction fidelity,
   latency and stop behavior. Its real-world checkpoint is distinct from the benchmark one.

Selection must use the same image stream, goal wording, route episodes, path follower,
latency measurement, and collision/stop definitions. Compare unseen indoor routes,
turn-in-place commands, image quality shifts, and goal-completion false positives.
Published R2R/RxR SR and SPL cannot establish Raspicat performance or safe operation.

## NaVILA native setup

Install the [official NaVILA repository](https://github.com/AnjieCheng/NaVILA) and its
documented dependencies in the remote GPU host's ROS-compatible Python environment.
Download its [official checkpoint](https://huggingface.co/a8cheng/navila-llama3-8b-8f)
outside Git, and make upstream `llava` importable. The backend follows the authors'
[evaluation inference code](https://github.com/AnjieCheng/NaVILA/blob/main/evaluation/vlnce_baselines/navila_trainer.py):
eight sampled/padded frames, Llama 3 conversation template, deterministic generation,
and a short action sentence. Example after sourcing ROS 2 and the built workspace:

```bash
scripts/download_checkpoints.sh navila

PYTHONPATH=/path/to/NaVILA:$PYTHONPATH \
ros2 launch rvln_remote inference.launch.py \
  backend:=navila vla_path:=/workspace/models/navila-llama3-8b-8f device:=cuda:0

ros2 launch rvln_edge edge_only.launch.py \
  adapter_kind:=omnivla with_follower:=true cmd_vel_topic:=/cmd_vel_vln_test
```

Use a text goal. The backend rejects pose/image goals and unexpected model text. It
limits one prediction to 75 cm forward or 45 degrees turn, and maps `stop` to a zero
waypoint. It resets video history when the instruction string changes. The ROS
Observation interface currently lacks an episode/reset identifier, so repeating exactly
the same instruction for a new episode needs a node restart or a future goal-ID field.
No checkpoint inference, simulator evaluation, or robot motion is validated by the
CPU contract tests. The Hugging Face checkpoint currently has no model card; review
weight terms separately from the source repository's Apache-2.0 license.

## NaVIDA native setup

Install `torch`, `transformers`, `qwen-vl-utils`, `flash-attn`, and Pillow in the
remote GPU host's ROS-compatible Python environment. The backend follows the
[official evaluation implementation](https://github.com/waynechu1021/NAVIDA/blob/main/src/eval/eval.py)
for Qwen2.5-VL loading, the image prompt, and action decoding. It omits Habitat,
top-down-map rendering, and the second predicted action. Its one-action output
lets the existing edge Path follower decide when to request the next observation.

```bash
scripts/download_checkpoints.sh navida
ros2 launch rvln_remote inference.launch.py \
  backend:=navida vla_path:=/workspace/models/NaVIDA device:=cuda:0
```

NaVIDA is text-goal only here. The backend defaults an unnumbered `forward` to
25 cm and an unnumbered turn to 15 degrees, as upstream does, and rejects moves
over 75 cm or turns over 45 degrees. It resets history on instruction changes;
reusing the exact instruction for a new episode still needs a node restart.
Only conversion contract tests have run; checkpoint inference and robot motion
have not been validated. The checkpoint is about 8.15 GB.

## Checkpoint staging

`scripts/download_checkpoints.sh --help` lists all named targets. Its NaVILA and
NaVIDA targets have ROS backends. The `streamvln_real` and `internvla_dualvln`
targets stage official research checkpoints only. Both models require temporal
policy adapters before this workspace can execute them. InternVLA-N1's RGB
DualVLN is a paired fast/slow policy, while StreamVLN's released server has a
hard-coded demo instruction; neither is interchangeable with a one-shot Path
predictor. The script never downloads these large files without an explicit
model name.
