import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.ollama_client import OllamaClient


def main() -> None:
    client = OllamaClient()
    print(f'Connecting to Ollama at: {client.host}')
    print(f'Testing model: {client.primary_model}')
    response = client.generate('Answer in one sentence: What is retrieval-augmented generation?')
    print('\nModel response:')
    print(response)


if __name__ == '__main__':
    main()
