from pathlib import Path
from typing import Iterable


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def list_files(root: Path, suffixes: Iterable[str]) -> list[Path]:
    suffixes = {suffix.lower() for suffix in suffixes}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
