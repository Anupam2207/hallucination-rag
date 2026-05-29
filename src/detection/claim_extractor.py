from __future__ import annotations

import re

import spacy

from src.config import get_config_value
from src.utils.text_cleaning import (
    clean_malformed_lists,
    is_non_factual_assistant_phrase,
    normalize_for_detection,
    strip_evidence_citations,
)

FACT_RELATIONS = {
    "introduced",
    "introduces",
    "invented",
    "created",
    "creates",
    "developed",
    "develops",
    "published",
    "released",
    "proposed",
    "founded",
    "built",
    "won",
    "started",
    "launched",
    "uses",
    "use",
    "combines",
    "combine",
    "retrieves",
    "retrieve",
    "generates",
    "generate",
    "grounds",
    "ground",
    "stores",
    "store",
    "checks",
    "check",
    "verifies",
    "verify",
    "measures",
    "measure",
    "evaluates",
    "evaluate",
    "reduces",
    "reduce",
    "improves",
    "improve",
    "ranks",
    "rank",
    "contains",
    "contain",
    "requires",
    "require",
    "consists",
    "refers",
    "means",
}

_SPLIT_CONJUNCTION_RE = re.compile(r"\s+(?:and|but|while|whereas)\s+", re.IGNORECASE)
_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|uses?|combines?|retrieves?|generates?|grounds?|stores?|checks?|verifies?|"
    r"measures?|evaluates?|reduces?|improves?|ranks?|contains?|requires?|consists|refers|means|introduced|invented|"
    r"created|developed|published|released|proposed|founded|built|won|started|launched)\b",
    re.IGNORECASE,
)
_SUBJECT_VERB_RE = re.compile(
    r"^(?P<subject>.+?)\s+"
    r"(?P<verb>is|are|was|were|has|have|had|uses?|combines?|retrieves?|generates?|grounds?|stores?|checks?|"
    r"verifies?|measures?|evaluates?|reduces?|improves?|ranks?|contains?|requires?|consists|refers|means|introduced|"
    r"invented|created|developed|published|released|proposed|founded|built|won|started|launched)\b",
    re.IGNORECASE,
)
_EXPLICIT_SUBJECT_RE = re.compile(
    r"^(?:It|They|He|She|This|That|These|Those|it|they|he|she|this|that|these|those|[A-Z][A-Za-z0-9\-]*|[A-Z]{2,}|the\s+[a-zA-Z0-9\-]+|The\s+[A-Za-z0-9\-]+)\b"
)


class ClaimExtractor:
    """Fast rule-based claim extractor with optional atomic splitting.

    The default path remains sentence-based and CPU-light.  When atomic splitting
    is enabled, the extractor additionally splits list items, semicolon clauses,
    and coordinated factual clauses such as "X retrieves evidence and generates
    answers" into smaller checkable claims.
    """

    def __init__(self, atomic_splitting: bool | None = None) -> None:
        self.atomic_splitting = bool(
            get_config_value("settings", "detection", "atomic_claim_splitting_enabled", default=True)
            if atomic_splitting is None
            else atomic_splitting
        )
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    @staticmethod
    def _ensure_terminal_punctuation(text: str) -> str:
        text = text.strip()
        if text and not re.search(r"[.!?]$", text):
            text += "."
        return text

    @staticmethod
    def _clean_claim(text: str) -> str:
        text = normalize_for_detection(text)
        text = re.sub(r"^\s*[-*•]\s*", "", text)
        text = re.sub(r"^\s*\d+[\).\s:-]+", "", text)
        text = re.sub(r"^\s*(?:and|but|or)\s+", "", text, flags=re.IGNORECASE)
        # Remove dangling list markers created by LLM numbered lists, e.g.
        # "... can: 1." or "Improve accuracy 2."  Do not strip four-digit
        # factual years such as 2001 or 2020 from claims.
        text = re.sub(r"\s+\d{1,2}[\).]\s*$", "", text)
        return ClaimExtractor._ensure_terminal_punctuation(text.strip())

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
            "based on evidence",
            "the original answer has been rewritten",
        )
        if lower.startswith(meta_prefixes):
            return True
        if is_non_factual_assistant_phrase(text):
            return True
        heading_patterns = (
            "including:",
            "including:.",
            "include:",
            "include:.",
            "includes:",
            "includes:.",
            "as follows:",
            "as follows:.",
            "following:",
            "following:.",
            "benefits:",
            "benefits:.",
            "several benefits:",
            "several benefits:.",
            "two main stages:",
            "two main stages:.",
            "two main components:",
            "two main components:.",
            "the process typically involves:",
            "the process typically involves:.",
            "can:",
            "can:.",
        )
        if lower.endswith(heading_patterns) or " can:" in lower:
            return True
        if re.fullmatch(r"(?:common|main|key|important|typical|following|these|the)\s+[a-z0-9\-\s]{0,80}\b(?:include|includes|are|involve|involves)\s*:\.?", lower):
            return True
        if re.fullmatch(r"(?:common|main|key|important|typical)\s+[a-z0-9\-\s]{0,80}:\.?", lower):
            return True
        if re.search(r"\b(consists of|typically consists of|process typically)\b", lower) and lower.endswith(":"):
            return True
        if re.search(r"\b(main stages|main components)\b", lower) and re.search(r"[:\s]\d+[.]?$", lower):
            return True
        if re.fullmatch(r"\d+[\).]?", lower):
            return True
        return False

    @staticmethod
    def _is_claim(text: str) -> bool:
        clean = normalize_for_detection(text).strip()
        if len(clean) < 8:
            return False
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", clean)
        if len(tokens) < 2:
            return False
        years = re.findall(r"\b(?:19|20)\d{2}\b", clean)
        entities = re.findall(r"\b(?:[A-Z][a-zA-Z0-9]+|[A-Z]{2,}|[A-Za-z]+[A-Z][A-Za-z0-9]*)\b", clean)
        relation = any(word in clean.lower() for word in FACT_RELATIONS)
        definition = bool(re.search(r"\b(?:is|are|refers to|means|is defined as|stands for)\b", clean, flags=re.IGNORECASE))
        return bool(years) or bool(entities) or relation or definition

    @staticmethod
    def _preprocess_answer(answer: str) -> str:
        text = strip_evidence_citations(answer or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Preserve list structure before sentence normalization.
        text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[\).])\s+", "\n", text)
        text = re.sub(r"\s+(?=\d{1,2}[\).]\s+[A-Z])", "\n", text)
        text = re.sub(r"\s+(?=[-*•]\s+[A-Z])", "\n", text)
        text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[\).])\s+", "", text)
        text = clean_malformed_lists(text)
        return text

    @staticmethod
    def _split_semicolon_units(text: str) -> list[str]:
        if ";" not in text:
            return [text]
        parts = [part.strip(" ;") for part in re.split(r";\s*", text) if part.strip(" ;")]
        if len(parts) <= 1:
            return [text]
        return [ClaimExtractor._ensure_terminal_punctuation(part) for part in parts]

    @staticmethod
    def _infer_subject(text: str) -> str | None:
        match = _SUBJECT_VERB_RE.search(text.strip())
        if not match:
            return None
        subject = match.group("subject").strip(" ,")
        # Keep subjects short enough to avoid carrying whole clauses.
        if len(subject.split()) > 6:
            return None
        return subject

    @staticmethod
    def _has_explicit_subject(text: str) -> bool:
        return bool(_EXPLICIT_SUBJECT_RE.match(text.strip()))

    @staticmethod
    def _has_verblike_fact(text: str) -> bool:
        return bool(_VERB_RE.search(text or ""))

    def _split_compound_claims(self, text: str) -> list[str]:
        if not self.atomic_splitting:
            return [text]
        sentence = text.strip()
        if not _SPLIT_CONJUNCTION_RE.search(sentence):
            return [sentence]
        parts = [part.strip(" ,") for part in _SPLIT_CONJUNCTION_RE.split(sentence) if part.strip(" ,")]
        if len(parts) < 2 or len(parts) > 4:
            return [sentence]
        if not all(self._has_verblike_fact(part) for part in parts):
            # Try carrying the subject into verb-only tail clauses.
            subject = self._infer_subject(parts[0])
            if not subject:
                return [sentence]
            carried: list[str] = [parts[0]]
            for part in parts[1:]:
                if self._has_verblike_fact(part):
                    carried.append(part if self._has_explicit_subject(part) else f"{subject} {part}")
                else:
                    return [sentence]
            parts = carried
        else:
            subject = self._infer_subject(parts[0])
            carried = [parts[0]]
            for part in parts[1:]:
                if subject and not self._has_explicit_subject(part):
                    carried.append(f"{subject} {part}")
                else:
                    carried.append(part)
            parts = carried
        cleaned_parts = [self._ensure_terminal_punctuation(part) for part in parts]
        # Only split if at least two resulting clauses are factual claims.
        factual_count = sum(1 for part in cleaned_parts if self._is_claim(part))
        return cleaned_parts if factual_count >= 2 else [sentence]

    def _candidate_units(self, normalized: str) -> list[str]:
        units: list[str] = []
        for line in normalized.splitlines():
            line = line.strip()
            if not line:
                continue
            doc = self.nlp(line)
            for sent in doc.sents:
                sentence = sent.text.strip()
                if not sentence:
                    continue
                for semicolon_piece in self._split_semicolon_units(sentence):
                    units.extend(self._split_compound_claims(semicolon_piece))
        return units

    def extract_claims(self, answer: str) -> list[str]:
        if not answer:
            return []
        normalized = self._preprocess_answer(answer)
        claims: list[str] = []
        seen: set[str] = set()
        for piece in self._candidate_units(normalized):
            claim = self._clean_claim(piece)
            if not claim:
                continue
            if self._is_meta_or_non_factual(claim):
                continue
            if not self._is_claim(claim):
                continue
            normalized_claim = claim.lower()
            if normalized_claim in seen:
                continue
            seen.add(normalized_claim)
            claims.append(claim)
        return claims
