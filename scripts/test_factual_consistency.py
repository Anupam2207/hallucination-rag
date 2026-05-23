import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.factual_consistency import run_factual_consistency_checks


def main() -> None:
    cases = [
        (
            "RAG was introduced in 2021.",
            "RAG was introduced in 2020.",
            "numeric_mismatch_with_evidence",
        ),
        (
            "The Formula 1 hybrid era began in 2014.",
            "The hybrid era officially began in Formula 1 in 2014.",
            None,
        ),
        (
            "Max Verstappen won the 2021 Formula 1 championship.",
            "Lewis Hamilton won the 2021 Formula 1 championship.",
            "entity_mismatch_with_evidence",
        ),
    ]
    for claim, evidence, expected_flag in cases:
        result = run_factual_consistency_checks(claim, evidence)
        print("\nClaim:", claim)
        print("Evidence:", evidence)
        print("Flags:", result["flags"])
        if expected_flag:
            assert expected_flag in result["flags"]
        else:
            assert result["flags"] == []
    print("\nFactual consistency smoke test passed.")


if __name__ == "__main__":
    main()
