import re
import spacy

from src.utils.text_cleaning import (
    clean_malformed_lists,
    is_non_factual_assistant_phrase,
    normalize_for_detection
)

FACT_RELATIONS = {
    "introduced",
    "invented",
    "created",
    "developed",
    "published",
    "released",
    "proposed",
    "founded",
    "built",
    "won",
    "started",
    "launched",
}


class ClaimExtractor:

    def __init__(self):

        self.nlp = spacy.blank("en")

        self.nlp.add_pipe(
            "sentencizer"
        )


    @staticmethod
    def _clean_claim(text):

        text = normalize_for_detection(
            text
        )

        text = re.sub(
            r"^\s*[-*•]\s*",
            "",
            text
        )

        text = re.sub(
            r"^\s*\d+[\).\s:-]+",
            "",
            text
        )
        # Remove dangling list markers created by LLM numbered lists, e.g.
        # "... can: 1." or "Improve accuracy 2."  Do not strip four-digit
        # factual years such as 2001 or 2020 from claims.
        text = re.sub(r"\s+\d{1,2}[\).]\s*$", "", text)

        return text.strip()


    @staticmethod
    def _is_meta_or_non_factual(text):

        lower=text.lower().strip()

        meta_prefixes=(

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

        if is_non_factual_assistant_phrase(
            text
        ):

            return True

        heading_patterns = (
            "including:",
            "as follows:",
            "following:",
            "benefits:",
            "several benefits:",
            "two main stages:",
            "two main components:",
            "the process typically involves:",
            "can:",
        )
        if lower.endswith(heading_patterns) or " can:" in lower:
            return True
        if re.search(r"\b(consists of|typically consists of|process typically)\b", lower) and lower.endswith(":"):
            return True
        if re.search(r"\b(main stages|main components)\b", lower) and re.search(r"[:\s]\d+[.]?$", lower):
            return True

        return False


    @staticmethod
    def _is_claim(text):

        if len(text)<10:

            return False

        years=re.findall(

            r"\b(?:19|20)\d{2}\b",
            text

        )

        entities=re.findall(

            r"\b[A-Z][a-zA-Z]+\b",
            text

        )

        relation=any(

            word in text.lower()

            for word in FACT_RELATIONS
        )

        return (

            bool(years)
            or bool(entities)
            or relation

        )


    def extract_claims(
        self,
        answer
    ):

        if not answer:

            return []

        normalized=clean_malformed_lists(normalize_for_detection(
            answer
        ))

        # Put numbered list items on their own sentence-like lines and remove
        # numbering so claim extraction does not produce fragments such as
        # "can: 1." or "accuracy 2.".
        normalized=re.sub(

            r"\n\s*(\d+)[\).\s]+",
            "\n",
            normalized

        )
        normalized=re.sub(r"\s+(\d+)[\).]\s+", "\n", normalized)

        normalized=re.sub(

            r"\n\s*[-*•]\s+",
            "\n",
            normalized

        )

        doc=self.nlp(
            normalized
        )

        claims=[]

        seen=set()

        for sent in doc.sents:

            sentence=sent.text.strip()

            if not sentence:

                continue

            pieces=[sentence]

            if "\n" in sentence:

                pieces=[

                    p.strip()

                    for p in sentence.splitlines()

                    if p.strip()

                ]

            for piece in pieces:

                claim=self._clean_claim(
                    piece
                )

                if not claim:

                    continue

                if self._is_meta_or_non_factual(
                    claim
                ):

                    continue

                if not self._is_claim(
                    claim
                ):

                    continue

                normalized_claim=claim.lower()

                if normalized_claim in seen:

                    continue

                seen.add(
                    normalized_claim
                )

                claims.append(
                    claim
                )

        return claims