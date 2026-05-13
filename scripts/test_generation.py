import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.base_answer import BaseAnswerGenerator


def main() -> None:
    query = "What is retrieval-augmented generation?"
    generator = BaseAnswerGenerator()
    print(f"\nQuery:\n{query}\n")
    print("Raw LLM answer:\n")
    print(generator.generate_answer(query))


if __name__ == "__main__":
    main()
