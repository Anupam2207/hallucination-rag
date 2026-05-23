import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.nli_verifier import NLIVerifier


def main() -> None:
    parser = ArgumentParser(description="Smoke-test optional NLI verification.")
    parser.add_argument("--enable", action="store_true", help="Load the real NLI model. May download on first use.")
    args = parser.parse_args()

    verifier = NLIVerifier(enabled=args.enable)
    print(f"NLI enabled: {verifier.enabled}")
    if not args.enable:
        print("Model loading skipped. Run with --enable to test the real NLI model.")

    examples = [
        ("entailment", "RAG was introduced in 2020.", "RAG was introduced in 2020."),
        ("contradiction", "RAG was introduced in 2020.", "RAG was introduced in 2021."),
        ("neutral", "RAG combines retrieval with generation.", "Formula 1 cars use hybrid power units."),
    ]
    for expected, evidence, claim in examples:
        result = verifier.verify(claim, evidence)
        print("\nEvidence:", evidence)
        print("Claim:", claim)
        print("Expected:", expected)
        print("Result:", result)


if __name__ == "__main__":
    main()
