import re
import unicodedata
from typing import Iterable

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_DOI_RE = re.compile(r"\bdoi\s*[:/]\s*\S+|https?://doi\.org/\S+", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\barxiv\s*:\s*\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_CITATION_RE = re.compile(r"\[(?:\d+|[A-Za-z]+\s+et\s+al\.?[, ]*\d{4})\]")
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?\d{1,4}$", re.IGNORECASE)
_EQUATION_HEAVY_RE = re.compile(r"^[\s\dA-Za-z]*[=∑∫√≈≤≥±×÷]{1,}.*$")
_REFERENCE_HEADING_RE = re.compile(r"^(references|bibliography|works cited)\s*$", re.IGNORECASE)
_SECTION_HEADING_RE = re.compile(
    r"^(abstract|introduction|background|related work|method|methodology|approach|experiments?|results?|discussion|conclusion|future work)\b.*",
    re.IGNORECASE,
)
_AFFILIATION_HINT_RE = re.compile(
    r"\b(university|institute|department|school of|college|laboratory|lab|faculty|centre|center)\b",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_common_noise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("•", "-").replace("\u00a0", " ")
    return text


def _is_probable_affiliation(line: str) -> bool:
    # Keep section titles and real factual text. Only remove short affiliation-like lines.
    if len(line.split()) > 18:
        return False
    if _EMAIL_RE.search(line):
        return True
    return bool(_AFFILIATION_HINT_RE.search(line) and not _SECTION_HEADING_RE.match(line))


def _is_isolated_equation(line: str) -> bool:
    if _SECTION_HEADING_RE.match(line):
        return False
    stripped = line.strip()
    if len(stripped.split()) > 14:
        return False
    math_chars = sum(1 for char in stripped if char in "=+-*/∑∫√≈≤≥±×÷{}_^")
    alpha_chars = sum(1 for char in stripped if char.isalpha())
    return bool(math_chars >= 2 and math_chars >= alpha_chars)


def clean_text(text: str) -> str:
    text = strip_common_noise(text)
    text = normalize_whitespace(text)

    cleaned_lines: list[str] = []
    in_references = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        if _REFERENCE_HEADING_RE.match(line):
            in_references = True
            continue
        if in_references:
            continue

        # Keep metadata in document metadata, not in chunk text.  This avoids
        # generic tokens such as "Domain:" polluting retrieval.  The real
        # phrase "Domain Name System" is preserved because it is not a
        # key-value metadata line.
        if re.match(r"^(domain|category|topic|keywords?|tags?)\s*:\s*", line, flags=re.IGNORECASE):
            continue

        if _PAGE_NUMBER_RE.match(line):
            continue
        if _URL_RE.search(line) or _DOI_RE.search(line) or _ARXIV_RE.search(line):
            line = _URL_RE.sub("", line)
            line = _DOI_RE.sub("", line)
            line = _ARXIV_RE.sub("", line).strip()
            if not line:
                continue
        if _is_probable_affiliation(line):
            continue
        if _is_isolated_equation(line) or _EQUATION_HEAVY_RE.match(line) and len(line.split()) <= 8:
            continue

        line = _CITATION_RE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return normalize_whitespace(cleaned)


def preprocess_text(text: str) -> str:
    return clean_text(text)


def preprocess_documents(documents: Iterable[dict]) -> list[dict]:
    processed: list[dict] = []
    for doc in documents:
        cleaned_text = preprocess_text(doc.get("text", ""))
        if not cleaned_text:
            continue
        processed.append({**doc, "text": cleaned_text, "text_length": len(cleaned_text)})
    return processed
