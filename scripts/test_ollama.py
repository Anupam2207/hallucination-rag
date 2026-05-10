import os
import sys
from pathlib import Path

import ollama
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    load_dotenv()
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name = "llama3.2:3b"

    print(f"Connecting to Ollama at: {host}")
    print(f"Testing model: {model_name}")

    try:
        client = ollama.Client(host=host)

        response = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Answer in one sentence: What is retrieval-augmented generation?"
                }
            ],
            options={"temperature": 0.2}
        )

        print("\nModel response:")
        print(response["message"]["content"])

    except Exception as e:
        print("\nOllama test failed.")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()