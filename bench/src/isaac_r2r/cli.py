"""R2R-CE transfer command registration; execution is loaded on demand."""

from __future__ import annotations

import argparse
import subprocess as subprocess  # compatibility: callers patch the historical worker launch location
from pathlib import Path


def run(args) -> dict:
    from isaac_r2r.evaluation import run as execute

    return execute(args)


def summarize_results(args) -> dict:
    from isaac_r2r.evaluation import summarize_results as execute

    return execute(args)


def run_full_split(args) -> dict:
    from isaac_r2r.evaluation import run_full_split as execute

    return execute(args)


def inventory_scenes(args) -> dict:
    from isaac_r2r.evaluation import inventory_scenes as execute

    return execute(args)


def audit_cma_inputs(args) -> dict:
    from isaac_r2r.evaluation import audit_cma_inputs as execute

    return execute(args)


def convert_scan_asset(args) -> dict:
    from isaac_r2r.evaluation import convert_scan_asset as execute

    return execute(args)


def calibrate_scan(args) -> dict:
    from isaac_r2r.evaluation import calibrate_scan as execute

    return execute(args)


def register(commands: argparse._SubParsersAction) -> None:
    r2r = commands.add_parser("r2r", help="run an R2R-CE transfer episode")
    r2r.add_argument("--split", type=Path, required=True)
    r2r.add_argument("--scenes", type=Path, required=True)
    r2r.add_argument("--episode", required=True)
    r2r.add_argument("--policy-url", default="http://127.0.0.1:8765")
    r2r.add_argument("--policy-id", help="require the server to attest this model identity")
    r2r.add_argument(
        "--policy-split",
        type=Path,
        help="aligned preprocessed R2R split for pretrained CMA tokens",
    )
    r2r.add_argument("--policy-tokens-json", help=argparse.SUPPRESS)
    r2r.add_argument("--policy-split-sha256", help=argparse.SUPPRESS)
    r2r.add_argument("--out", type=Path)
    r2r.add_argument("--max-steps", type=int, default=500)
    r2r.add_argument("--headless", action="store_true")
    r2r.add_argument("--check", action="store_true")
    r2r.set_defaults(handler=run)

    summary = commands.add_parser("summarize-r2r", help="summarize transfer results")
    summary.add_argument("results", type=Path, nargs="+")
    summary.add_argument("--split", type=Path, help="require exact episode coverage of this split")
    summary.add_argument("--out", type=Path, required=True)
    summary.set_defaults(handler=summarize_results)

    batch = commands.add_parser("r2r-batch", help="run and audit every episode in an R2R split")
    batch.add_argument("--split", type=Path, required=True)
    batch.add_argument("--scenes", type=Path, required=True)
    batch.add_argument("--policy-url", default="http://127.0.0.1:8765")
    batch.add_argument(
        "--policy-id", required=True, help="fixed model/checkpoint identity for the report"
    )
    batch.add_argument(
        "--policy-split",
        type=Path,
        help="aligned preprocessed R2R split for pretrained CMA tokens",
    )
    batch.add_argument("--out-dir", type=Path, required=True)
    batch.add_argument("--max-steps", type=int, default=500)
    batch.add_argument("--episode-timeout", type=int, default=3600)
    batch.add_argument(
        "--startup-retries",
        type=int,
        default=2,
        help="retry incomplete native Kit exits before an episode finishes",
    )
    batch.add_argument("--headless", action="store_true")
    batch.add_argument("--resume", action="store_true")
    batch.set_defaults(handler=run_full_split)

    audit = commands.add_parser(
        "inventory-r2r", help="inventory split scenes and local scan assets"
    )
    audit.add_argument("--split", type=Path, required=True)
    audit.add_argument("--scans-root", type=Path, required=True)
    audit.add_argument("--scenes", type=Path, help="optional USD scene registry")
    audit.add_argument("--out", type=Path)
    audit.set_defaults(handler=inventory_scenes)

    cma = commands.add_parser(
        "audit-cma-split", help="verify pretrained CMA token IDs and episode alignment"
    )
    cma.add_argument("--split", type=Path, required=True)
    cma.add_argument("--policy-split", type=Path, required=True)
    cma.add_argument("--out", type=Path)
    cma.set_defaults(handler=audit_cma_inputs)

    scan = commands.add_parser("convert-scan", help="import an MP3D GLB as a collidable Isaac USD")
    scan.add_argument("--glb", type=Path, required=True)
    scan.add_argument("--out", type=Path, required=True)
    scan.set_defaults(handler=convert_scan_asset)

    calibration = commands.add_parser(
        "calibrate-r2r", help="fit a Habitat-to-Isaac transform from landmarks"
    )
    calibration.add_argument("--landmarks", type=Path, required=True)
    calibration.add_argument("--scene", required=True)
    calibration.add_argument("--usd", type=Path, required=True)
    calibration.add_argument("--out", type=Path, required=True)
    calibration.add_argument("--max-error-m", type=float, default=0.05)
    calibration.add_argument(
        "--dome-intensity",
        type=float,
        default=0.0,
        help="verified dome-light intensity for this scene",
    )
    calibration.set_defaults(handler=calibrate_scan)
