import re
from typing import Any

from src.generation.ollama_client import OllamaClient
from src.retrieval.evidence_ranker import format_evidence_block
from src.utils.text_cleaning import clean_malformed_lists, remove_display_citations


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
            return "The retrieved evidence is insufficient to answer this question reliably."

        prompt = f"""
You are a factual answer correction assistant.

Rewrite the original answer using only the retrieved evidence.

Rules:
- Return only the corrected answer.
- Do not include notes, explanations, sources, references, or meta-comments.
- Do not write citation markers such as [Evidence-1], [Evidence-2], [1], or [2]. The user interface displays evidence separately.
- Preserve supported facts and remove unsupported facts.
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
        return answer
