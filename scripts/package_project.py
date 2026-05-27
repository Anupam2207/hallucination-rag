from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

EXCLUDE_PARTS = {
    '.venv', 'venv', 'env', '__pycache__', '.pytest_cache', '.mypy_cache',
    '.ruff_cache', '.ipynb_checkpoints', 'db/chroma', 'data/chunks',
    'data/processed', 'outputs', 'logs'
}
EXCLUDE_NAMES = {'.DS_Store', '.env'}
EXCLUDE_SUFFIXES = {'.pyc', '.pyo', '.log'}


def should_exclude(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if path.name in EXCLUDE_NAMES:
        return True
    if any(rel == part or rel.startswith(part + '/') for part in EXCLUDE_PARTS):
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    if rel.startswith('tests/.env'):
        return True
    return False


def package(root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob('*')):
            if path.is_dir() or should_exclude(path, root):
                continue
            zf.write(path, path.relative_to(root.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a clean project zip.')
    parser.add_argument('--output', default='hallucination-rag-clean.zip')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    package(root, output)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
