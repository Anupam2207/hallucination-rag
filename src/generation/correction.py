import re
from typing import Any

from src.generation.ollama_client import OllamaClient
from src.retrieval.evidence_intent import FACTUAL, annotate_evidence, is_credible_support_evidence
from src.retrieval.evidence_ranker import format_evidence_block
from src.retrieval.query_focus import extract_query_focus, has_topical_match
from src.utils.text_cleaning import clean_malformed_lists, remove_display_citations


INSUFFICIENT_EVIDENCE_RESPONSE = "Insufficient evidence available in the knowledge base."


class AnswerCorrector:
    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()

    @staticmethod
    def _remove_meta_lines(answer: str) -> str:
        blocked_prefixes = (
            "note:",
            "source:",
            "sources:",
            "reference:",
            "references:",
            "explanation:",
        )
        cleaned_lines: list[str] = []
        for line in answer.splitlines():
            stripped = line.strip()
            lower = stripped.lower()
            if not stripped:
                cleaned_lines.append(line)
                continue
            if lower.startswith(blocked_prefixes):
                continue
            if "original answer has been rewritten" in lower:
                continue
            if "using only supported evidence" in lower:
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _word_tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text or "")

    @classmethod
    def _is_malformed_answer(cls, answer: str) -> bool:
        cleaned = (answer or "").strip()
        if not cleaned:
            return True
        if cleaned == INSUFFICIENT_EVIDENCE_RESPONSE:
            return False
        words = cls._word_tokens(cleaned)
        if len(words) < 8:
            return True
        if not re.search(r"[.!?]\s*$", cleaned):
            return True
        if re.search(r"[,;:]\s*$", cleaned):
            return True
        trailing = words[-1].lower() if words else ""
        trailing_stopwords = {
            "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
            "with", "by", "as", "at", "from", "into", "about", "than", "while", "during",
            "between", "among", "through", "over", "under", "using", "via",
        }
        if trailing in trailing_stopwords:
            return True
        if cleaned.count("(") != cleaned.count(")") or cleaned.count("[") != cleaned.count("]"):
            return True
        return False

    @classmethod
    def _longest_common_word_run(cls, answer: str, evidence: str) -> int:
        answer_words = [word.lower() for word in cls._word_tokens(answer)]
        evidence_words = [word.lower() for word in cls._word_tokens(evidence)]
        if not answer_words or not evidence_words:
            return 0
        # Dynamic programming over word positions is cheap at correction lengths
        # and precisely enforces the consecutive-copy limit.
        previous = [0] * (len(evidence_words) + 1)
        longest = 0
        for answer_word in answer_words:
            current = [0] * (len(evidence_words) + 1)
            for index, evidence_word in enumerate(evidence_words, start=1):
                if answer_word == evidence_word:
                    current[index] = previous[index - 1] + 1
                    longest = max(longest, current[index])
            previous = current
        return longest

    @classmethod
    def _violates_copy_limit(cls, answer: str, evidence_block: str, max_consecutive_words: int = 12) -> bool:
        return cls._longest_common_word_run(answer, evidence_block) > max_consecutive_words

    def _paraphrase_if_extractive(self, answer: str, evidence_block: str, query: str) -> str:
        if not self._violates_copy_limit(answer, evidence_block):
            return answer
        paraphrase_prompt = f"""
Paraphrase the answer below using only the retrieved evidence.

Rules:
- Preserve the factual meaning.
- Synthesize across evidence where possible.
- Do not copy more than 12 consecutive words from the evidence.
- Return only the paraphrased answer.
- If the answer cannot be paraphrased safely, return: {INSUFFICIENT_EVIDENCE_RESPONSE}

User query:
{query}

Retrieved evidence:
{evidence_block}

Answer to paraphrase:
{answer}

Paraphrased answer:
""".strip()
        paraphrased = self.client.generate(paraphrase_prompt).strip()
        paraphrased = self._remove_meta_lines(paraphrased)
        paraphrased = self._sanitize_common_hallucinations(paraphrased, evidence_block, query)
        paraphrased = remove_display_citations(paraphrased)
        if self._is_malformed_answer(paraphrased) or self._violates_copy_limit(paraphrased, evidence_block):
            return INSUFFICIENT_EVIDENCE_RESPONSE
        return paraphrased


    @staticmethod
    def _relation_requested(query: str) -> bool:
        lower = (query or "").lower()
        relation_patterns = (
            "who introduced", "who invented", "who proposed", "who developed",
            "who created", "who founded", "introduced", "invented", "proposed",
            "developed", "created", "founded", "when",
        )
        return any(pattern in lower for pattern in relation_patterns)

    @staticmethod
    def _display_subject_from_query(query: str, evidence_text: str = "") -> str | None:
        lower = (query or "").lower()
        known = {
            "colbert": "ColBERT",
            "rag": "RAG",
            "truthfulqa": "TruthfulQA",
            "dpr": "DPR",
            "realm": "REALM",
            "bm25": "BM25",
            "llm": "LLM",
        }
        for token, display in known.items():
            if re.search(rf"\b{re.escape(token)}\b", lower):
                return display

        # Fall back to a prominent acronym or CamelCase term shared by query and evidence.
        query_terms = set(re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{2,}\b", query or ""))
        for term in query_terms:
            if re.search(rf"\b{re.escape(term)}\b", evidence_text or "", flags=re.IGNORECASE):
                if term.isupper() or any(ch.isupper() for ch in term[1:]):
                    return term
        match = re.search(r"\b[A-Z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*)+\b", evidence_text or "")
        return match.group(0) if match else None

    @staticmethod
    def _clean_people_span(people: str) -> str:
        people = re.split(r"\s+as\s+|\s+for\s+|\s+to\s+|\s+where\s+|\s+which\s+|\s+and\s+uses\s+", people.strip())[0]
        people = re.sub(r"\s+", " ", people)
        people = people.strip(" ,;:.()[]")
        # Remove organization/source fragments sometimes captured after the author names.
        people = re.sub(r"\b(?:in|during)\s+(?:19|20)\d{2}.*$", "", people).strip(" ,;:.")
        return people

    @classmethod
    def _extract_relation_fact_from_text(cls, query: str, text: str) -> str | None:
        subject = cls._display_subject_from_query(query, text)
        if not subject:
            return None
        clean_text = re.sub(r"\s+", " ", text or "").strip()
        if not clean_text:
            return None

        patterns = [
            # It was introduced by Omar Khattab and Matei Zaharia in 2020 ...
            r"\b(?:it|this\s+(?:model|method|system|approach|paper)|[A-Z][A-Za-z0-9\-]+)\s+was\s+(?P<relation>introduced|invented|proposed|developed|created|founded)\s+by\s+(?P<people>[^.]{3,180}?)\s+(?:in|during)\s+(?P<year>(?:19|20)\d{2})\b",
            # ColBERT was introduced by ... in 2020 ...
            r"\b[A-Z][A-Za-z0-9\-]*\s+(?:is|was)\s+(?P<relation>introduced|invented|proposed|developed|created|founded)\s+by\s+(?P<people>[^.]{3,180}?)\s+(?:in|during)\s+(?P<year>(?:19|20)\d{2})\b",
            # introduced by ... in 2020 (subject omitted because previous sentence/title gives it)
            r"\b(?P<relation>introduced|invented|proposed|developed|created|founded)\s+by\s+(?P<people>[^.]{3,180}?)\s+(?:in|during)\s+(?P<year>(?:19|20)\d{2})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, clean_text, flags=re.IGNORECASE)
            if not match:
                continue
            relation = match.group("relation").lower()
            people = cls._clean_people_span(match.group("people"))
            year = match.group("year")
            if not people or not year:
                continue
            # Avoid returning educational examples rather than factual evidence.
            window = clean_text[max(0, match.start() - 80): match.end() + 80].lower()
            if any(marker in window for marker in ("for example", "if a claim", "hypothetical", "factually contradictory", "claim says")):
                continue
            return f"{subject} was {relation} by {people} in {year}."
        return None

    @classmethod
    def _deterministic_relation_answer(cls, query: str, evidence_list: list[Any]) -> str | None:
        if not cls._relation_requested(query):
            return None
        focus = extract_query_focus(query)
        candidates: list[dict[str, Any]] = []
        for raw_item in evidence_list:
            if isinstance(raw_item, dict):
                item = dict(raw_item)
            else:
                item = {"text": str(raw_item), "metadata": {}}
            if "evidence_type" not in item or "evidence_credibility_score" not in item:
                item = annotate_evidence(item, query=query)
            if item.get("evidence_type") != FACTUAL:
                continue
            if not is_credible_support_evidence(item, min_credibility=0.20, strict_factual_mode=True):
                continue
            if not has_topical_match(str(item.get("text", "")), item.get("metadata", {}) or {}, focus):
                continue
            candidates.append(item)

        def rank(item: dict[str, Any]) -> tuple[float, float, float]:
            def as_float(key: str) -> float:
                try:
                    return float(item.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    return 0.0
            return (as_float("factual_assertion_score"), as_float("evidence_credibility_score"), as_float("final_score"))

        for item in sorted(candidates, key=rank, reverse=True):
            answer = cls._extract_relation_fact_from_text(query, str(item.get("text", "")))
            if answer and not cls._is_malformed_answer(answer):
                return answer
        return None

    @staticmethod
    def _sanitize_common_hallucinations(answer: str, evidence_block: str, query: str) -> str:
        evidence_lower = evidence_block.lower()
        query_lower = query.lower()
        cleaned = answer.strip()

        if "fine-tun" not in evidence_lower and "fine tun" not in evidence_lower:
            replacements = [
                (r"fine[- ]?tunes?\s+(?:a\s+)?generator\s+model\s+on\s+(?:these\s+|the\s+)?retrieved\s+(?:passages|texts?|documents?|information)", "uses the retrieved evidence as context"),
                (r"fine[- ]?tuning\s+(?:a\s+)?generator\s+model\s+on\s+(?:these\s+|the\s+)?retrieved\s+(?:passages|texts?|documents?|information)", "using the retrieved evidence as context"),
                (r"generator\s+model\s+is\s+fine[- ]?tuned\s+on\s+(?:these\s+|the\s+)?retrieved\s+(?:passages|texts?|documents?|information)", "generator uses the retrieved evidence as context"),
                (r"fine[- ]?tuned\s+on\s+(?:these\s+|the\s+)?retrieved\s+(?:passages|texts?|documents?|information)", "using the retrieved evidence as context"),
            ]
            for pattern, replacement in replacements:
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # Do not allow ColBERT answers to keep collaborative-filtering hallucinations
        # unless the evidence explicitly contains those ideas.
        unsupported_if_absent = [
            "collaborative filtering",
            "recommendation system",
            "recommendation systems",
            "recommendation accuracy",
            "e-commerce",
            "ecommerce",
            "product descriptions",
            "product reviews",
            "reviews",
            "collaborative bert",
            "open-source library",
            "open source library",
        ]
        for phrase in unsupported_if_absent:
            if phrase not in evidence_lower:
                cleaned = re.sub(rf",?\s*(?:and\s+)?{re.escape(phrase)}", "", cleaned, flags=re.IGNORECASE)

        unsupported_tasks = [
            "machine translation",
            "language translation",
            "sentiment analysis",
            "text classification",
            "conversational dialogue systems",
            "dialogue systems",
            "conversational ai",
        ]
        for phrase in unsupported_tasks:
            if phrase not in evidence_lower:
                cleaned = re.sub(rf",?\s*(?:and\s+)?{re.escape(phrase)}", "", cleaned, flags=re.IGNORECASE)

        # Generic RAG boilerplate should not appear for non-RAG questions.
        is_rag_query = any(term in query_lower for term in ("rag", "retrieval-augmented generation", "retrieval augmented generation"))
        if not is_rag_query:
            boilerplate_patterns = [
                r"(?:The\s+)?generator\s+uses\s+retrieved\s+passages\s+as\s+context\.?",
                r"(?:The\s+)?language\s+model\s+uses\s+retrieved\s+passages\s+as\s+context\.?",
            ]
            for pattern in boilerplate_patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
        return clean_malformed_lists(cleaned)

    def correct(
        self,
        query: str,
        original_answer: str,
        evidence_list: list[Any],
    ) -> str:
        evidence_block = format_evidence_block(evidence_list)
        if not evidence_block.strip():
            return INSUFFICIENT_EVIDENCE_RESPONSE

        deterministic_answer = self._deterministic_relation_answer(query, evidence_list)
        if deterministic_answer:
            return deterministic_answer

        prompt = f"""
You are a factual answer correction assistant.

Rewrite the original answer using only the retrieved evidence.

Rules:
- Return only the corrected answer.
- Do not include notes, explanations, sources, references, or meta-comments.
- Do not write citation markers such as [Evidence-1], [Evidence-2], [1], or [2]. The user interface displays evidence separately.
- Preserve supported facts and remove unsupported facts.
- Paraphrase the evidence; preserve factual meaning without copying long spans.
- Synthesize across multiple evidence chunks when possible.
- Do not copy more than 12 consecutive words from the retrieved evidence.
- Do not introduce facts that are not present in the retrieved evidence.
- Do not preserve any factual detail from the user query unless retrieved evidence explicitly supports it.
- Do not include years, dates, numbers, author names, paper names, benchmark names, task examples, or performance claims unless the evidence explicitly states them.
- Keep the answer concise, natural, and readable.
- Prefer short paragraphs. Avoid numbered lists unless the evidence clearly requires a list.
- Do not add generic RAG explanations unless the query is about RAG.
- Do not mention fine-tuning unless the retrieved evidence explicitly says fine-tuning or fine-tuned.
- Do not mention collaborative filtering, recommendation systems, e-commerce, product descriptions, or reviews unless the retrieved evidence explicitly states them.
- If the evidence is insufficient, say clearly that the knowledge base does not provide enough evidence.

User query:
{query}

Original answer:
{original_answer}

Retrieved evidence:
{evidence_block}

Corrected answer:
""".strip()

        answer = self.client.generate(prompt).strip()
        answer = self._remove_meta_lines(answer)
        answer = self._sanitize_common_hallucinations(answer, evidence_block, query)
        answer = remove_display_citations(answer)
        answer = self._paraphrase_if_extractive(answer, evidence_block, query)
        if self._is_malformed_answer(answer):
            return INSUFFICIENT_EVIDENCE_RESPONSE
        return answer
