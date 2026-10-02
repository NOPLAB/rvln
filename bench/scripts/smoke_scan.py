"""Exercise Isaac Sim's GLB importer with a generated triangle and inspect USD."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import uuid
from pathlib import Path

from bench.artifacts import sha256


def make_triangle_glb(path: Path) -> None:
    """Write a minimal valid glTF 2.0 binary mesh without external assets."""
    positions = struct.pack("<9f", 0, 0, 0, 2, 0, 0, 0, 0, 2)
    indices = struct.pack("<3H", 0, 1, 2)
    binary = positions + indices + b"\0\0"
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
            {
                "buffer": 0,
                "byteOffset": len(positions),
                "byteLength": len(indices),
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [2, 0, 2],
            },
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
    }
    data = json.dumps(document, separators=(",", ":")).encode("utf-8")
    data += b" " * (-len(data) % 4)
    total = 12 + 8 + len(data) + 8 + len(binary)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total)
        + struct.pack("<I4s", len(data), b"JSON")
        + data
        + struct.pack("<I4s", len(binary), b"BIN\0")
        + binary
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/validation/scan-import"))
    args = parser.parse_args()
    run_dir = args.out_dir / uuid.uuid4().hex
    run_dir.mkdir(parents=True)
    source = run_dir / "triangle.glb"
    output = run_dir / "triangle.usd"
    make_triangle_glb(source)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.cli",
            "convert-scan",
            "--glb",
            str(source),
            "--out",
            str(output),
        ],
        check=True,
        timeout=600,
    )
    metadata_path = output.with_suffix(".json")
    if not metadata_path.is_file():
        raise RuntimeError("Isaac conversion did not write metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["source_sha256"] != sha256(source):
        raise RuntimeError("converted scene source hash does not match the GLB")

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(output))
    if stage is None or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError("imported USD is missing or not Z-up")
    collisions = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not collisions or len(collisions) != metadata["meshes"]:
        raise RuntimeError("imported triangle is missing static collision")
    print(
        json.dumps(
            {
                "status": "completed",
                "run_dir": str(run_dir.resolve()),
                "source_up_axis": metadata["source_up_axis"],
                "meshes": metadata["meshes"],
                "colliders": len(collisions),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
