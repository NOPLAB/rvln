"""Hash every regular checkpoint file without loading weights into memory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    """Stream a file's content digest."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('model', nargs='+', help='backend=checkpoint path')
    args = parser.parse_args()
    records = {}
    for item in args.model:
        name, target = item.split('=', 1)
        root = Path(target)
        files = (
            sorted(p for p in root.rglob('*') if p.is_file() and not any(
                part in ('.cache', 'wandb') for part in p.relative_to(root).parts))
            if root.is_dir() else [root]
        )
        if not files:
            raise FileNotFoundError(root)
        entries = []
        for path in files:
            size = path.stat().st_size
            checksum = sha256(path)
            entries.append({'path': str(path.relative_to(root) if root.is_dir() else path.name),
                            'size_bytes': size, 'sha256': checksum})
            print(f'{name}: {entries[-1]["path"]} {size} {checksum}', flush=True)
        manifest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        records[name] = {'root': str(root.resolve()), 'files': entries,
                         'manifest_sha256': manifest}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'schema': 1, 'models': records}, indent=2) + '\n',
                        encoding='utf-8')


if __name__ == '__main__':
    main()
