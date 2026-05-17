import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.nli_verifier import NLIVerifier


def main() -> None:
    parser = ArgumentParser(description="Smoke-test the optional NLI verifier.")
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Load the configured NLI model. This may download the model on first run.",
    )
    args = parser.parse_args()

    evidence = (
        "Retrieval-Augmented Generation combines information retrieval with language generation. "
        "A retriever fetches relevant passages from a knowledge base, and the generator uses "
        "those passages as context while answering the user."
    )
    supported_claim = "RAG uses retrieved passages as context while generating an answer."
    unsupported_claim = "RAG fine-tunes a generator model on retrieved passages for each query."

    verifier = NLIVerifier(enabled=args.enable)

    print(f"NLI enabled: {verifier.enabled}")
    if not args.enable:
        print("Model loading skipped. Run with --enable to test the real NLI model.")

    for name, claim in [
        ("supported", supported_claim),
        ("unsupported_fine_tuning", unsupported_claim),
    ]:
        result = verifier.verify(claim, evidence)
        print("\nCase:", name)
        print("Claim:", claim)
        print("Label:", result.get("label"))
        print("Score:", result.get("score"))
        print("Available:", result.get("available"))
        if result.get("error"):
            print("Error:", result.get("error"))
        if result.get("scores"):
            print("Scores:", result.get("scores"))


if __name__ == "__main__":
    main()
