from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

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
    """Optional CPU-friendly NLI verifier.

    Evidence is the premise and the generated claim is the hypothesis. The class
    is intentionally defensive: if the model cannot be loaded, the detector falls
    back to similarity + rule checks instead of crashing.
    """

    def __init__(
        self,
        enabled: bool | None = None,
        model_name: str | None = None,
        device: str | None = None,
        max_evidence_chars: int | None = None,
        cache_enabled: bool | None = None,
    ) -> None:
        self.enabled = bool(
            get_config_value("settings", "verification", "enable_nli", default=False)
            if enabled is None
            else enabled
        )
        # Backward compatibility with older config layout.
        self.model_name = model_name or str(
            get_config_value(
                "settings", "verification", "nli_model",
                default=get_config_value(
                    "models", "verification", "nli_model_name",
                    default="cross-encoder/nli-deberta-v3-small",
                ),
            )
        )
        self.device = device or str(
            get_config_value("settings", "verification", "nli_device", default="cpu")
        )
        self.max_evidence_chars = int(
            max_evidence_chars
            if max_evidence_chars is not None
            else get_config_value("settings", "verification", "nli_max_evidence_chars", default=900)
        )
        self.cache_enabled = bool(
            get_config_value("settings", "verification", "nli_cache_enabled", default=True)
            if cache_enabled is None
            else cache_enabled
        )
        self.model = None
        self.available = False
        self.error: str | None = None
        self.labels: list[str] = ["contradiction", "entailment", "neutral"]
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, device=self.device)
            config = getattr(getattr(self.model, "model", None), "config", None)
            id2label = getattr(config, "id2label", None)
            if isinstance(id2label, dict) and id2label:
                self.labels = [str(id2label[i]).lower() for i in sorted(id2label)]
            self.available = True
        except Exception as exc:  # pragma: no cover - model/cache/network dependent
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

    @staticmethod
    def _normalize_label(label: str) -> str:
        label_lower = label.lower()
        if "contrad" in label_lower:
            return "contradiction"
        if "entail" in label_lower:
            return "entailment"
        if "neutral" in label_lower:
            return "neutral"
        return label_lower

    def verify(self, claim: str, evidence: str) -> Dict[str, Any]:
        if not self.enabled:
            return NLIResult(False, False, "not_run", None, {}, None).__dict__
        if not self.available or self.model is None:
            return NLIResult(True, False, "unavailable", None, {}, self.error or "NLI model is unavailable").__dict__
        if not claim.strip() or not evidence.strip():
            return NLIResult(True, True, "not_enough_input", None, {}, None).__dict__

        evidence = evidence.strip()[: self.max_evidence_chars]
        claim = claim.strip()
        cache_key = (evidence, claim)
        if self.cache_enabled and cache_key in self._cache:
            return dict(self._cache[cache_key])

        try:
            raw_scores = self.model.predict([(evidence, claim)], apply_softmax=False)
            raw_array = np.asarray(raw_scores)[0]
            if raw_array.ndim == 0:
                score = float(raw_array)
                result = NLIResult(True, True, "entailment" if score >= 0.5 else "neutral", round(score, 4), {"entailment": round(score, 4)}, None).__dict__
            else:
                probs = self._softmax(raw_array)
                labels = self.labels[: len(probs)]
                if len(labels) != len(probs):
                    labels = [f"label_{i}" for i in range(len(probs))]
                normalized_labels = [self._normalize_label(label) for label in labels]
                score_map = {label: round(float(prob), 4) for label, prob in zip(normalized_labels, probs)}
                best_index = int(np.argmax(probs))
                result = NLIResult(
                    True,
                    True,
                    normalized_labels[best_index],
                    round(float(probs[best_index]), 4),
                    score_map,
                    None,
                ).__dict__
            if self.cache_enabled:
                self._cache[cache_key] = dict(result)
            return result
        except Exception as exc:  # pragma: no cover - environment/model dependent
            return NLIResult(True, False, "error", None, {}, str(exc)).__dict__
