# Development

## Python lint and formatting

Run from the repository root with [uv](https://docs.astral.sh/uv/) installed:

```sh
uvx ruff==0.16.9 check .
uvx ruff==0.16.9 format --check .
```

Ruff is pinned to 0.16.9 in CI and `pyproject.toml`. It targets Python 3.10
(ROS 2 Humble), uses a 99-character formatting width, and checks E4/E7/E9/F.
Both commands include owned ROS packages, scripts, benchmark code and tooling.
Submodules, generated protocol bindings, the verbatim upstream OmniVLA model,
and environment/runtime output directories are excluded, not benchmark sources.
The new Python CI complements the existing ament linters; `.flake8` and
`.pydocstyle` still govern those package checks.

## Tests

On a ROS 2 Humble host, build with `colcon build --symlink-install` and run
`colcon test`. For the CPU test image, run `scripts/vla.sh build test`, then
`scripts/vla.sh test`; pass `-k <name>` to select a subset. Hardware, GPU and
checkpoint tests require their documented dependencies/assets and opt-in flags.
Ruff itself does not require ROS or model dependencies.
