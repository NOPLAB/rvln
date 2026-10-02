"""Simulator-independent benchmark command host."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import entry_points
from typing import Callable, Iterable


Registrar = Callable[[argparse._SubParsersAction], None]


def build_parser(registrars: Iterable[Registrar] | None = None) -> argparse.ArgumentParser:
    """Load adapters through package metadata; the core never imports them."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    if registrars is None:
        registrars = (
            entry.load()
            for entry in sorted(
                entry_points(group="rvln_bench.plugins"), key=lambda item: item.name
            )
        )
    for register in registrars:
        register(commands)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = args.handler(args)
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))
    if result is not None:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
