from src.config import load_all_configs
from src.generation.ollama_client import OllamaClient


class BaseAnswerGenerator:
    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()
        configs = load_all_configs()
        self.prompt_template = configs["prompts"]["base_answer_prompt"]

    def generate_answer(self, query: str) -> str:
        prompt = self.prompt_template.format(query=query.strip())
        return self.client.generate(prompt)
