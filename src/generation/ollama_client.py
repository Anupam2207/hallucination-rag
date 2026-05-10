import requests


class OllamaClient:

    def __init__(
        self,
        host="http://localhost:11434",
        model="llama3.2:3b"
    ):
        self.host = host
        self.model = model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3
    ):

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        response = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        return response.json()["response"]