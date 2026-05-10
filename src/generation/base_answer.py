from src.generation.ollama_client import OllamaClient


class BaseAnswerGenerator:

    def __init__(self):

        self.client = OllamaClient()

    def generate_answer(
        self,
        query: str
    ):

        prompt = f"""
Answer the following question clearly and factually.

Question:
{query}

Answer:
"""

        answer = self.client.generate(prompt)

        return answer.strip()