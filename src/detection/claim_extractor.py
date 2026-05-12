import re
from typing import List

_BULLET_RE = re.compile(r"^[\-\*•\d\.\)\s]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class ClaimExtractor:
    """Lightweight sentence-level claim extractor without external model downloads."""

    def extract_claims(self, answer: str) -> List[str]:
        if not answer.strip():
            return []

        parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(answer) if part.strip()]
        claims: List[str] = []
        for sentence in parts:
            cleaned = _BULLET_RE.sub("", sentence).strip()
            if len(cleaned) < 8:
                continue
            if not any(char.isalpha() for char in cleaned):
                continue
            claims.append(cleaned)
        return claims
