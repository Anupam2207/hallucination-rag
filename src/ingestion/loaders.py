import hashlib
import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.config import get_config_value
from src.logger import get_logger

logger = get_logger("loaders")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


DEFAULT_CLEAN_PDF_KEYWORDS = {
    "fact_verification",
    "colbert_retrieval_model",
    "causes_of_hallucination",
    "vehicle_safety",
    "evolution_of_smartphones",
    "role_of_ai",
    "formula1_history",
    "ancient_indian_architecture",
}

DEFAULT_RESEARCH_PDF_KEYWORDS = {
    "hallucinationsurvey", "hallulens", "proofver", "ragtruth", "refind",
    "realm", "dpr", "lewis_rag", "truthfulqa", "colbert", "factverification",
}


def _config_list(key: str, default: set[str]) -> set[str]:
    values = get_config_value("settings", "knowledge_base", key, default=list(default))
    return {str(value).lower() for value in values or []}


def _should_skip_file(file_path: Path) -> tuple[bool, str]:
    """Return whether a file should be skipped by the active knowledge-base mode."""
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return True, "unsupported_extension"

    pdf_only = bool(get_config_value("settings", "knowledge_base", "pdf_only", default=True))
    if pdf_only and suffix != ".pdf":
        return True, "pdf_only_mode"
    if suffix != ".pdf":
        return False, ""

    mode = str(get_config_value("settings", "knowledge_base", "mode", default="demo")).lower()
    include_research = bool(get_config_value("settings", "knowledge_base", "include_research_pdfs", default=False))
    name = file_path.name.lower()
    clean_keywords = _config_list("clean_pdf_keywords", DEFAULT_CLEAN_PDF_KEYWORDS)
    research_keywords = _config_list("exclude_pdf_keywords", DEFAULT_RESEARCH_PDF_KEYWORDS)

    if any(keyword.lower() in name for keyword in clean_keywords):
        return False, ""
    if mode == "demo" and not include_research and any(keyword.lower() in name for keyword in research_keywords):
        return True, "demo_mode_research_pdf_filter"
    return False, ""


def _make_doc_id(relative_path: str) -> str:
    digest = hashlib.md5(relative_path.encode("utf-8")).hexdigest()[:12]
    stem = Path(relative_path).stem.lower().replace(" ", "_")
    return f"{stem}_{digest}"


def _infer_title(text: str, file_path: Path) -> str:
    """Infer a readable title while ignoring PDF page labels and boilerplate."""
    fallback = file_path.stem.replace("_", " ").title()
    ignored_prefixes = ("page ", "copyright", "permission to", "abstracting with credit")
    for line in text.splitlines():
        clean = line.strip().strip("# ").strip()
        if not clean:
            continue
        lower = clean.lower()
        if lower.startswith(ignored_prefixes):
            continue
        if lower in {"abstract", "introduction", "references", "bibliography"}:
            continue
        if len(clean) <= 180 and len(clean.split()) >= 2:
            return clean
    return fallback


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
    return file_path.read_text(encoding="utf-8", errors="ignore")


def load_json_file(file_path: Path) -> str:
    data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
    flattened = _flatten_json_value(data)
    return "\n".join(part for part in flattened if part.strip())


def load_pdf_pymupdf(file_path: Path) -> str:
    # Import lazily so the rest of the ingestion pipeline can still work if
    # PyMuPDF is unavailable. load_pdf_file will fall back to pypdf.
    import fitz  # type: ignore

    pages: list[str] = []
    doc = fitz.open(str(file_path))
    try:
        for page_number, page in enumerate(doc, start=1):
            try:
                text = page.get_text("text") or ""
                if text.strip():
                    pages.append(f"\nPage {page_number}\n{text.strip()}")
            except Exception as exc:
                logger.warning("PyMuPDF page extraction failed in %s page %s: %s", file_path, page_number, exc)
    finally:
        doc.close()
    return "\n\n".join(pages)


def load_pdf_pypdf(file_path: Path) -> str:
    pages: list[str] = []
    reader = PdfReader(str(file_path))
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"\nPage {page_number}\n{text.strip()}")
        except Exception as exc:
            logger.warning("PyPDF page extraction failed in %s page %s: %s", file_path, page_number, exc)
    return "\n\n".join(pages)


def load_pdf_file(file_path: Path) -> str:
    try:
        logger.info("Trying PyMuPDF: %s", file_path.name)
        text = load_pdf_pymupdf(file_path)
        if text.strip():
            return text
    except Exception as exc:
        logger.warning("PyMuPDF failed for %s: %s", file_path, exc)

    try:
        logger.info("Fallback PyPDF: %s", file_path.name)
        return load_pdf_pypdf(file_path)
    except Exception as exc:
        logger.warning("PDF failed entirely for %s: %s", file_path, exc)
        return ""


def load_documents(raw_data_dir: Path) -> list[dict]:
    documents: list[dict] = []
    loaded_count = 0
    skipped_count = 0

    if not raw_data_dir.exists():
        logger.warning("Raw data directory does not exist: %s", raw_data_dir)
        return documents

    skipped_reasons: dict[str, int] = {}
    for file_path in sorted(raw_data_dir.rglob("*")):
        if not file_path.is_file():
            continue
        skip, reason = _should_skip_file(file_path)
        if skip:
            if reason:
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        relative_path = file_path.relative_to(raw_data_dir.parent).as_posix()
        try:
            ext = file_path.suffix.lower()
            if ext in {".txt", ".md"}:
                text = load_text_file(file_path)
            elif ext == ".json":
                text = load_json_file(file_path)
            else:
                text = load_pdf_file(file_path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", file_path, exc)
            skipped_count += 1
            continue

        if not text.strip():
            logger.warning("Skipping empty document: %s", file_path)
            skipped_count += 1
            continue

        documents.append(
            {
                "doc_id": _make_doc_id(relative_path),
                "source_path": relative_path,
                "source_rel": relative_path,
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower().lstrip("."),
                "document_title": _infer_title(text, file_path),
                "text": text,
            }
        )
        loaded_count += 1

    logger.info("Loaded %s documents, skipped %s documents", loaded_count, skipped_count)
    for reason, count in sorted(skipped_reasons.items()):
        logger.info("Skipped %s files due to %s", count, reason)
    return documents
