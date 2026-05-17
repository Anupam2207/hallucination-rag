from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.config import get_config_value


@dataclass
class NLIResult:
    enabled: bool
    available: bool
    label: str
    score: float | None
    scores: Dict[str, float]
    error: str | None = None


class NLIVerifier:
    """Optional lightweight NLI verifier for claim-evidence entailment.

    This module is deliberately defensive. If the Hugging Face model cannot be
    loaded on a low-resource laptop, the app falls back to the similarity/rule
    detector instead of crashing.
    """

    def __init__(self, enabled: bool | None = None, model_name: str | None = None) -> None:
        self.enabled = bool(
            get_config_value("settings", "detection", "enable_nli", default=False)
            if enabled is None
            else enabled
        )
        self.model_name = model_name or str(
            get_config_value(
                "models",
                "verification",
                "nli_model_name",
                default="cross-encoder/nli-deberta-v3-small",
            )
        )
        self.model = None
        self.available = False
        self.error: str | None = None
        self.labels: list[str] = ["contradiction", "entailment", "neutral"]

        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            id2label = getattr(getattr(self.model, "model", None), "config", None)
            id2label = getattr(id2label, "id2label", None)
            if isinstance(id2label, dict) and id2label:
                self.labels = [str(id2label[i]).lower() for i in sorted(id2label)]
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on local model cache/network
            self.model = None
            self.available = False
            self.error = str(exc)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = values.astype(float)
        values = values - np.max(values)
        exp_values = np.exp(values)
        total = exp_values.sum()
        if total == 0:
            return np.zeros_like(values)
        return exp_values / total

    def verify(self, claim: str, evidence: str) -> Dict[str, Any]:
        if not self.enabled:
            return NLIResult(
                enabled=False,
                available=False,
                label="not_run",
                score=None,
                scores={},
                error=None,
            ).__dict__

        if not self.available or self.model is None:
            return NLIResult(
                enabled=True,
                available=False,
                label="unavailable",
                score=None,
                scores={},
                error=self.error or "NLI model is unavailable",
            ).__dict__

        if not claim.strip() or not evidence.strip():
            return NLIResult(
                enabled=True,
                available=True,
                label="not_enough_input",
                score=None,
                scores={},
                error=None,
            ).__dict__

        try:
            raw_scores = self.model.predict([(evidence, claim)], apply_softmax=False)
            raw_array = np.asarray(raw_scores)[0]
            if raw_array.ndim == 0:
                # Defensive fallback for unexpected single-score models.
                score = float(raw_array)
                return NLIResult(
                    enabled=True,
                    available=True,
                    label="entailment" if score >= 0.5 else "neutral",
                    score=round(score, 4),
                    scores={"entailment": round(score, 4)},
                    error=None,
                ).__dict__

            probs = self._softmax(raw_array)
            labels = self.labels[: len(probs)]
            if len(labels) != len(probs):
                labels = [f"label_{i}" for i in range(len(probs))]
            score_map = {label: round(float(prob), 4) for label, prob in zip(labels, probs)}
            best_index = int(np.argmax(probs))
            label = labels[best_index]
            score = round(float(probs[best_index]), 4)

            # Normalize common HF label variants.
            label_lower = label.lower()
            if "contrad" in label_lower:
                label = "contradiction"
            elif "entail" in label_lower:
                label = "entailment"
            elif "neutral" in label_lower:
                label = "neutral"

            return NLIResult(
                enabled=True,
                available=True,
                label=label,
                score=score,
                scores=score_map,
                error=None,
            ).__dict__
        except Exception as exc:  # pragma: no cover - environment/model dependent
            return NLIResult(
                enabled=True,
                available=False,
                label="unavailable",
                score=None,
                scores={},
                error=str(exc),
            ).__dict__
