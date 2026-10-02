"""Isolate Isaac Kit's process-exit behavior from the benchmark CLI."""

import argparse
from pathlib import Path

from isaac_r2r.scans import convert_scan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    convert_scan(args.glb, args.out)


if __name__ == "__main__":
    main()
