from typing import Dict, Any


class NLIVerifier:
    """Placeholder for the stronger second-stage verifier.

    The NLI stage is intentionally deferred to a later phase so that the
    ingestion/indexing foundation stays lightweight and reproducible first.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def verify(self, claim: str, evidence: str) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "label": "not_run",
            "score": None,
            "claim": claim,
            "evidence": evidence,
        }
