import hashlib
import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.logger import get_logger


logger = get_logger('loaders')
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.json', '.pdf'}


def _make_doc_id(relative_path: str) -> str:
    digest = hashlib.md5(relative_path.encode('utf-8')).hexdigest()[:12]
    stem = Path(relative_path).stem.lower().replace(' ', '_')
    return f'{stem}_{digest}'


def _flatten_json_value(value: Any) -> list[str]:
    parts: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = _flatten_json_value(item)
            if child:
                parts.append(f"{key}: {' '.join(child)}")
    elif isinstance(value, list):
        for item in value:
            parts.extend(_flatten_json_value(item))
    elif value is not None:
        parts.append(str(value))
    return parts


def load_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding='utf-8')


def load_json_file(file_path: Path) -> str:
    data = json.loads(file_path.read_text(encoding='utf-8'))
    flattened = _flatten_json_value(data)
    return '\n'.join(part for part in flattened if part.strip())


def load_pdf_file(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ''
        if page_text.strip():
            pages.append(page_text.strip())
    return '\n\n'.join(pages)


def load_documents(raw_data_dir: Path) -> list[dict]:
    documents: list[dict] = []
    if not raw_data_dir.exists():
        logger.warning('Raw data directory does not exist: %s', raw_data_dir)
        return documents

    for file_path in sorted(raw_data_dir.rglob('*')):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = file_path.relative_to(raw_data_dir.parent).as_posix()
        try:
            if file_path.suffix.lower() in {'.txt', '.md'}:
                text = load_text_file(file_path)
            elif file_path.suffix.lower() == '.json':
                text = load_json_file(file_path)
            else:
                text = load_pdf_file(file_path)
        except Exception as exc:
            logger.warning('Failed to load %s: %s', file_path, exc)
            continue

        if not text or not text.strip():
            logger.warning('Skipping empty document: %s', file_path)
            continue

        documents.append(
            {
                'doc_id': _make_doc_id(relative_path),
                'source_path': relative_path,
                'source_rel': relative_path,
                'file_name': file_path.name,
                'file_type': file_path.suffix.lower().lstrip('.'),
                'text': text,
            }
        )
    return documents
