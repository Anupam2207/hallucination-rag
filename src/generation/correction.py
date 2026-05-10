from src.generation.ollama_client import OllamaClient


class AnswerCorrector:

    def __init__(self):

        self.client = OllamaClient()

    def correct(
        self,
        query: str,
        evidence_list
    ):

        evidence_text = "\n\n".join(
            evidence_list
        )

        prompt = f"""
You are a factual verification assistant.

Using ONLY the evidence below, answer the question accurately.

If information is not supported by the evidence, do not include it.

QUESTION:
{query}

EVIDENCE:
{evidence_text}

CORRECTED ANSWER:
"""

        corrected = self.client.generate(
            prompt=prompt,
            temperature=0.1
        )

        return corrected.strip()