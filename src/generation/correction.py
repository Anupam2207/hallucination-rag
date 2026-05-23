import re
from typing import Any

from src.generation.ollama_client import OllamaClient
from src.retrieval.evidence_ranker import format_evidence_block


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
        # Remove unwanted assistant metadata and explanatory comments from the output.
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
    def _sanitize_common_rag_hallucinations(answer: str, evidence_block: str) -> str:
        """Narrow post-processing guard for common RAG-specific hallucinations.

        This keeps the corrected answer aligned with the retrieved evidence. It is
        intentionally conservative: it only handles recurring unsupported wording
        observed in the demo outputs, especially per-query fine-tuning claims.
        """
        evidence_lower = evidence_block.lower()
        cleaned = answer.strip()

        if "fine-tun" not in evidence_lower and "fine tun" not in evidence_lower:
            replacements = [
                (
                    r"fine[- ]?tunes?\s+(?:a\s+)?generator\s+model\s+on\s+(?:these\s+|the\s+)?retrieved\s+(?:passages|texts?|documents?)",
                    "uses the retrieved passages as context",
                ),
                (
                    r"fine[- ]?tuning\s+(?:a\s+)?generator\s+model\s+on\s+(?:these\s+|the\s+)?retrieved\s+(?:passages|texts?|documents?)",
                    "using the retrieved passages as context",
                ),
                (
                    r"generator\s+model\s+is\s+fine[- ]?tuned\s+on\s+(?:these\s+|the\s+)?retrieved\s+(?:passages|texts?|documents?)",
                    "generator uses the retrieved passages as context",
                ),
                (
                    r"fine[- ]?tuned\s+on\s+(?:these\s+|the\s+)?retrieved\s+(?:passages|texts?|documents?|information)",
                    "using the retrieved evidence as context",
                ),
                (
                    r"fine[- ]?tunes?\s+(?:a\s+)?generator\s+model\s+on\s+(?:these\s+|the\s+)?retrieved\s+information",
                    "uses the retrieved evidence as context",
                ),
                (
                    r"generator\s+model\s+is\s+fine[- ]?tuned\s+on\s+(?:these\s+|the\s+)?retrieved\s+information",
                    "generator uses the retrieved evidence as context",
                ),
            ]
            for pattern, replacement in replacements:
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # Remove unsupported task examples if the evidence does not mention them.
        unsupported_tasks = [
            "machine translation",
            "language translation",
            "sentiment analysis",
            "text classification",
            "conversational dialogue systems",
            "dialogue systems",
        ]
        for phrase in unsupported_tasks:
            if phrase not in evidence_lower:
                cleaned = re.sub(rf",?\s*(?:and\s+)?{re.escape(phrase)}", "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
        return cleaned

    def correct(
        self,
        query: str,
        original_answer: str,
        evidence_list: list[Any],
    ) -> str:
        evidence_block = format_evidence_block(evidence_list)
        if not evidence_block.strip():
            return "The retrieved evidence is insufficient to answer this question reliably."

        # Compose a prompt that instructs Ollama to rewrite the answer using only the evidence.
        prompt = f"""
You are a factual answer correction assistant.

Rewrite the original answer so it is grounded only in the retrieved evidence.

Rules:
- Return only the corrected answer.
- Do not include notes, explanations, or meta-comments.
- Do not say "the original answer has been rewritten".
- Add citation markers such as [Evidence-1] or [Evidence-2] after claims that use evidence.
- Preserve supported claims and replace unsupported claims using retrieved evidence.
- Remove claims that are not directly supported by the retrieved evidence.
- Do not add new claims beyond the retrieved evidence.
- Do not preserve any factual detail from the user query unless retrieved evidence explicitly supports it.
- Do not include years, dates, numbers, author names, paper names, benchmark names, task examples, or performance claims unless the evidence explicitly states them.
- Do not say "introduced in 2021" or any other year unless the retrieved evidence explicitly contains that year.
- Keep the answer concise, natural, and grammatically correct.
- For standard RAG, say that the generator uses retrieved passages as context.
- Do not say the generator is fine-tuned unless the retrieved evidence explicitly says fine-tuning or fine-tuned.
- Do not claim RAG needs less training data unless the retrieved evidence explicitly says so.
- Do not mention explicit knowledge representation unless the retrieved evidence explicitly says so.
- Do not add unsupported task examples such as sentiment analysis, machine translation, text classification, summarization, conversational AI, or dialogue systems.
- If the evidence does not support the user query's specific factual claim, say so clearly.
- If the evidence is insufficient, say: "The retrieved evidence is insufficient to answer this question reliably."

User query:
{query}

Original answer:
{original_answer}

Retrieved evidence:
{evidence_block}

Corrected answer:
""".strip()

        answer = self.client.generate(prompt).strip()
        # Normalize the model output by removing any stray meta text.
        answer = self._remove_meta_lines(answer)
        answer = self._sanitize_common_rag_hallucinations(answer, evidence_block)
        return answer
