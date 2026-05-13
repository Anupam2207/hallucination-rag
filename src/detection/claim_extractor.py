import re

import spacy


class ClaimExtractor:
    def __init__(self) -> None:
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    @staticmethod
    def _clean_claim(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^\s*[-*•]\s*", "", text)
        text = re.sub(r"^\s*\d+[\).\s:-]+", "", text)
        return text.strip()

    @staticmethod
    def _is_meta_or_non_factual(text: str) -> bool:
        lower = text.lower().strip()

        meta_prefixes = (
            "note:",
            "disclaimer:",
            "source:",
            "sources:",
            "reference:",
            "references:",
            "based on the evidence",
            "based on retrieved evidence",
            "the original answer has been rewritten",
        )

        if lower.startswith(meta_prefixes):
            return True

        if "original answer has been rewritten" in lower:
            return True

        if "using only supported evidence" in lower:
            return True

        if re.fullmatch(r"\d+[\).]?", lower):
            return True

        if lower.endswith("consists of two main components:"):
            return True

        if lower.endswith("typically involves:"):
            return True

        if len(lower.split()) < 3:
            return True

        return False

    def extract_claims(self, answer: str) -> list[str]:
        if not answer or not answer.strip():
            return []

        # Convert numbered bullets into sentence-like lines before sentencizer.
        normalized = re.sub(r"\n\s*(\d+)[\).\s]+", r"\n", answer)
        normalized = re.sub(r"\n\s*[-*•]\s+", "\n", normalized)

        doc = self.nlp(normalized)
        claims: list[str] = []

        for sent in doc.sents:
            raw = sent.text.strip()
            if not raw:
                continue

            # Split remaining colon-style bullet fragments conservatively.
            pieces = [raw]
            if "\n" in raw:
                pieces = [part.strip() for part in raw.splitlines() if part.strip()]

            for piece in pieces:
                claim = self._clean_claim(piece)
                if not claim:
                    continue
                if self._is_meta_or_non_factual(claim):
                    continue
                claims.append(claim)

        return claims