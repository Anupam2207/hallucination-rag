import re

import spacy

from src.utils.text_cleaning import is_non_factual_assistant_phrase, normalize_for_detection


class ClaimExtractor:
    def __init__(self) -> None:
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    @staticmethod
    def _clean_claim(text: str) -> str:
        text = normalize_for_detection(text)
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
        if is_non_factual_assistant_phrase(text):
            return True
        if "original answer has been rewritten" in lower:
            return True
        if "using only supported evidence" in lower:
            return True
        if re.fullmatch(r"\d+[\).]?", lower):
            return True

        heading_suffixes = (
            "including:",
            "as follows:",
            "following:",
            "benefits:",
            "several benefits:",
            "two main stages:",
            "two main components:",
            "the process typically involves:",
            "typically consists of two main stages:",
            "typically consists of two main components:",
            "consists of two main components:",
            "consists of two main stages:",
            "typically involves:",
        )
        if lower.endswith(heading_suffixes):
            return True
        if len(lower.split()) < 3:
            return True
        return False

    def extract_claims(self, answer: str) -> list[str]:
        if not answer or not answer.strip():
            return []

        normalized = normalize_for_detection(answer)
        # Convert numbered bullets into sentence-like lines before sentencizer.
        normalized = re.sub(r"\n\s*(\d+)[\).\s]+", r"\n", normalized)
        normalized = re.sub(r"\n\s*[-*•]\s+", "\n", normalized)

        doc = self.nlp(normalized)
        claims: list[str] = []

        for sent in doc.sents:
            raw = sent.text.strip()
            if not raw:
                continue
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
