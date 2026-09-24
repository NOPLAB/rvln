# VLN benchmark pilot: Gazebo camera replay and live Edge check

Date: 2026-09-24. This is an inference and integration pilot, **not a navigation
success ranking**. The paired replay input is a stationary robot's color camera,
20 frames at nominal 2 Hz in each of three Gazebo Classic worlds. The robot and
Edge ran locally; every backend inference ran on `pve1ubuntu` with a Slurm
reservation for one NVIDIA GeForce RTX 5090 (32 GB). Slurm jobs 2481–2498
completed successfully, one model process per scene.

## Paired replay result

The table uses 57 post-first-frame samples per model (19 frames × 3 scenes) for
steady inference time. `load` is process model-load time with warm filesystem
caches, not a cold boot measurement. `first` is the median first-inference time
from the three processes. GPU memory is peak PyTorch allocated memory, not total
GPU utilization. No warm-up calls were run. All 60 outputs per model passed the
finite 2-D shape check. The raw result files are under ignored `bench/runs/` on
this workstation and on the remote workspace; the paired summary and full
checkpoint file SHA-256 manifest are in `bench/results/`.

| Backend | Output shape | Load median ms | First frame median ms | Steady p50 / p95 ms | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| AsyncVLA step 750000 | 8 × 1024 | 12,341 | 633 | 65.9 / 67.1 | 15.2 |
| OmniVLA-original step 120000 | 8 × 4 | 11,251 | 637 | 66.0 / 67.1 | 15.0 |
| OmniVLA-edge remote | 8 × 4 | 4,071 | 449 | 12.3 / 13.8 | 1.4 |
| movla Stage A v9b | 8 × 4 | 7,398 | 623 | 65.5 / 68.7 | 3.1 |
| NaVILA | 1 × 4 | 8,753 | 917 | 367.9 / 373.4 | 17.5 |
| NaVIDA (SDPA) | 1 × 4 | 4,675 | 928 | 324.6 / 328.9 | 7.3 |

This is an inference-speed comparison on the same recorded images. It does not
compare action quality. The three worlds are `corridor`, `junction`, and `weave`;
their world descriptions and 12 proposed route episodes are versioned under
`bench/`. The camera at `/camera/color/image_raw` delivered RGB 640 × 480
images in each scene, and `/model_states` confirmed the robot and intended
obstacles. Frame records show at most 2.4 mm robot drift within a capture.
The junction landmarks are outside the initial camera view behind the crosswall,
which is relevant when interpreting its stationary replay.

## Live Remote → local Edge → Gazebo check

Slurm job 2499 loaded OmniVLA-edge, served its real GPU predictions over the
Tailnet, and was cancelled after the check to release the GPU. The local ROS
bridge paired 61 observations with 61 returned embeddings and passed them to
the `omnivla` Edge adapter. Median / p95 observation-to-embedding round trip
was 46.5 / 48.2 ms; the first request took 573 ms. The local follower issued
motion commands, and Gazebo odometry advanced from approximately 0.0 m to
2.50 m in the corridor. `/motor_power` had to be enabled explicitly; with it
off, `/cmd_vel` was present but the robot did not move. The motor was switched
off after the check. This run used a `go straight ahead` command and a safety
marker, not a scored destination episode. No intentional model stop, contact
log, or shortest-path oracle was recorded, so it contributes no SR or SPL.

## Provenance and limits

- Workspace base revision: `2c0148904dfae08692c6d415740dc7f0541e44c7` plus the
  uncommitted benchmark and Gazebo camera changes in this workspace. The exact
  working-tree state must be frozen before a publishable comparison.
- AsyncVLA source `b593ac5143287a62893159a42c88a471bf444f10`; OmniVLA
  submodule pointer `5182600cb4a9ee07684e17cdd2a6cbafc56b8a68`; movla
  source `0708ce756947759778bc8ac5151983e147326001`; NaVILA upstream
  `76b98f233dd0fff05dfcd69435eec6740febff9d`; scaling_on_scales
  `9c008a37540e761f53574b488979db6e49a64312`. The remote OmniVLA source
  directory has no `.git`, so its byte identity relative to the pointer has
  not been independently verified.
- Remote GPU driver 610.57.04; base environment PyTorch 2.12.1+cu130,
  CUDA 13.0, Transformers 5.12.1, PEFT 0.19.1. The isolated OpenVLA/NaVILA
  import overlay uses Transformers 4.40.1, PEFT 0.11.1, timm 0.9.10.
- AsyncVLA required the inference-only import patch in `bench/patches/` to
  avoid loading training datasets/strategies. NaVILA's unused InternVision
  FlashAttention import was made optional for its SigLIP checkpoint. These
  patches are applied to the remote upstream checkouts and are part of the
  tested configuration.
- NaVIDA uses SDPA in the absence of FlashAttention; its Qwen image content
  and generation call were repaired in the owned backend. The backend action
  conversion test passed (9 cases). The fixed-replay scorer tests passed (4
  cases). Python benchmark sources compiled successfully.
- `bench/results/checkpoints_2026-09-24.json` contains file-level SHA-256
  digests and tree-manifest digests for all six checkpoint paths. Hugging Face
  cache metadata and movla training logs are excluded; movla's `checkpoint.pt`
  and `normalizer.json` are included.

## Remaining navigation benchmark gates

The 12 route pilot and 60 paired held-out episodes in
`docs/VLN_BENCHMARK_PLAN.md` have **not** been executed. Before SR or SPL can be
reported, the episode runner must reset Gazebo and model history per route,
send each route's instruction, log model versus safety stops and contacts,
and validate collision-free shortest paths. The current image replay leaves
the robot stationary; the single live run was a control-path check. Hardware
performance and real-world navigation remain untested.
