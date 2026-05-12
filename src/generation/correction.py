from src.config import load_all_configs
from src.generation.ollama_client import OllamaClient
from src.retrieval.evidence_ranker import format_evidence_block


class AnswerCorrector:
    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()
        configs = load_all_configs()
        self.prompt_template = configs["prompts"]["correction_prompt"]

    def correct(self, query: str, original_answer: str, evidence_list: list[dict]) -> str:
        if not evidence_list:
            return (
                "I could not verify the original answer because no supporting evidence "
                "was retrieved from the knowledge base."
            )

        evidence_text = format_evidence_block(evidence_list)
        prompt = self.prompt_template.format(
            query=query.strip(),
            original_answer=original_answer.strip(),
            evidence=evidence_text,
        )
        return self.client.generate(prompt)
