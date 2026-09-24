# Repository Guidelines

## Project Structure & Module Organization

This is a ROS 2 Humble colcon workspace. `src/rvln_*` contains the owned ROS messages, inference core, remote ROS 2 node, edge nodes, and launch files. `proto/edge_action.proto` defines only the mobile-to-Pi contract; regenerate its stubs with `scripts/gen_proto.sh` after editing it. Edge and remote use `rvln_msgs` topics. `app/inference/` and `app/logger/` are separate Flutter apps, while `web/` is a Next.js browser client. `docker/` holds runtime images and Compose profiles; `docs/` holds operating and design notes. Python tests live in each package's `test/`, Dart tests in each app's `test/`, and web tests in `web/test/`. Downloaded weights in `models/` and runtime assets are not committed.

## Build, Test, and Development Commands

- On a ROS 2 Humble host, run `vcs import src < raspicat.repos`, `rosdep install --from-paths src --ignore-src -r -y`, then `colcon build --symlink-install`.
- Run `scripts/vla.sh build test` to build the CPU test image; `scripts/vla.sh test` runs the Python suite in it. Use `scripts/vla.sh --help` for supported run modes.
- In `web/`, run `pnpm install`, `pnpm dev`, and `pnpm build` for local development and a static export. Run `pnpm sync-assets` when ONNX models and CLIP assets are available in `app/inference/assets/`.
- In either `app/inference/` or `app/logger/`, run `flutter pub get`, `flutter analyze`, and `flutter test`.

## Coding Style & Naming Conventions

Follow the existing Python style and the root `.flake8` and `.pydocstyle` rules; Python lines are limited to 99 characters. Name Python tests `test_*.py`. Format handwritten Dart with `dart format`; generated gRPC Dart files are excluded from formatting checks. In `web/`, Biome uses two-space indentation and single quotes: run `pnpm lint` and `pnpm typecheck` before submitting changes. Keep generated protocol code synchronized with its `.proto` source.

## Testing Guidelines

Add focused tests beside the affected package or app. Run `scripts/vla.sh test -k <name>` for a Python subset, `flutter test` for a Flutter app, or `pnpm test` for web Vitest tests (`*.test.ts`). GPU and checkpoint integration tests require their documented assets and opt-in environment variables; a normal CPU run can skip them. No repository-wide coverage threshold is configured.

## Commit & Pull Request Guidelines

Recent commits use `type(scope): summary`, such as `fix(scripts): ...` or `feat(logger): ...`; use a concise scope when practical. In pull requests, describe the behavior changed, list the checks run, link a relevant issue when one exists, and include screenshots for visible UI changes. Call out required model assets or hardware when validation depends on them.
