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

Your task is to rewrite the original answer so it is grounded only in the retrieved evidence.

Rules:
- Return only the corrected answer.
- Do not include notes, explanations, citations, references, or meta-comments.
- Do not say "the original answer has been rewritten".
- Do not mention evidence numbers like [1], [2], or [3].
- Remove claims that are not directly supported by the retrieved evidence.
- Do not add new claims beyond the retrieved evidence.
- Keep the answer concise.
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
        return self._remove_meta_lines(answer)
