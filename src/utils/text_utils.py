import re

_WHITESPACE_RE = re.compile(r"[\t\x0b\x0c\r ]+")
_MULTILINE_RE = re.compile(r"\n{3,}")
_NON_PRINTABLE_RE = re.compile(r"[^\t\n\r\x20-\x7E\u00A0-\u024F]")


def strip_non_printable(text: str) -> str:
    return _NON_PRINTABLE_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = _MULTILINE_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    return normalize_whitespace(strip_non_printable(text))
