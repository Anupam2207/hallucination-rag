from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import HallucinationDetector
from src.detection.nli_verifier import NLIVerifier
from src.detection.support_scorer import SupportScorer
from src.evaluation.classification_metrics import DEFAULT_LABELS, multiclass_classification_report
from src.evaluation.metrics import HallucinationMetrics
from src.generation.correction import INSUFFICIENT_EVIDENCE_RESPONSE
from src.retrieval.evidence_intent import annotate_evidence_list
from src.utils.text_cleaning import normalize_for_detection


class EvaluationHashEmbedder:
    """Deterministic, dependency-light embedder for reproducible local evaluation."""

    def __init__(self, n_features: int = 512) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
            lowercase=True,
        )

    def encode(self, texts: str | Iterable[str], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        matrix = self.vectorizer.transform(list(texts)).astype(np.float32).toarray()
        if normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix


class RuleAblationSupportScorer(SupportScorer):
    """Support scorer variant that disables factual rule caps for ablations."""

    def _apply_rule_caps(self, claim: str, all_evidence_texts: list[str], best_evidence: str, score: float):
        return score, [], {"flags": [], "details": {}}


class SimilarityOnlyDetector(HallucinationDetector):
    """Detector variant that ignores NLI and rule flags after scoring."""

    def __init__(self) -> None:
        super().__init__(
            support_scorer=RuleAblationSupportScorer(embedder=EvaluationHashEmbedder()),
            nli_verifier=NLIVerifier(enabled=False),
        )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def average(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_for_detection(text or "").lower()).strip()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,}", normalize_text(text)))


def similar_enough(a: str, b: str, threshold: float = 0.45) -> bool:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return False
    score = len(ta & tb) / max(1, len(ta | tb))
    return score >= threshold or ta.issubset(tb) or tb.issubset(ta)


def make_detector(enable_nli: bool = False, rules: bool = True) -> HallucinationDetector:
    scorer_cls = SupportScorer if rules else RuleAblationSupportScorer
    return HallucinationDetector(
        support_scorer=scorer_cls(embedder=EvaluationHashEmbedder()),
        nli_verifier=NLIVerifier(enabled=enable_nli),
    )


def record_evidence(record: Dict[str, Any], query: str | None = None) -> List[Dict[str, Any]]:
    evidence = record.get("evidence") or record.get("evidence_list") or []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(evidence, start=1):
        if isinstance(item, str):
            normalized.append({"text": item, "metadata": {"source_rel": f"manual_evidence_{index}.txt"}})
        elif isinstance(item, dict):
            copied = dict(item)
            copied.setdefault("metadata", {})
            normalized.append(copied)
    return annotate_evidence_list(normalized, query=query or str(record.get("query", "")))


def gold_claims(record: Dict[str, Any]) -> List[Dict[str, str]]:
    if isinstance(record.get("gold_claims"), list):
        claims = []
        for item in record["gold_claims"]:
            if isinstance(item, dict) and item.get("claim"):
                claims.append({"claim": str(item["claim"]), "label": str(item.get("label", "unsupported"))})
        if claims:
            return claims
    claims: list[dict[str, str]] = []
    for claim in record.get("gold_supported_claims", []) or []:
        claims.append({"claim": str(claim), "label": "supported"})
    for claim in record.get("gold_weak_claims", []) or []:
        claims.append({"claim": str(claim), "label": "weak_support"})
    for claim in record.get("gold_unsupported_claims", []) or []:
        claims.append({"claim": str(claim), "label": "unsupported"})
    return claims


def classify_claim(detector: HallucinationDetector, claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    detection = detector.detect(claim, evidence)
    if not detection.get("claims"):
        return {
            "claim": claim,
            "label": "unsupported",
            "support_score": 0.0,
            "rule_flags": ["claim_not_extracted"],
            "rule_categories": ["claim_not_extracted"],
        }
    result = dict(detection["claims"][0])
    result.setdefault("claim", claim)
    return result


def evaluate_detection_records(
    records: Sequence[Dict[str, Any]],
    enable_nli: bool = False,
    rules: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    detector = make_detector(enable_nli=enable_nli, rules=rules)
    rows: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    for record in records:
        evidence = record_evidence(record)
        for idx, gold in enumerate(gold_claims(record), start=1):
            predicted = classify_claim(detector, gold["claim"], evidence)
            true_label = gold["label"] if gold["label"] in DEFAULT_LABELS else "unsupported"
            pred_label = predicted.get("label", "unsupported")
            if pred_label not in DEFAULT_LABELS:
                pred_label = "unsupported"
            y_true.append(true_label)
            y_pred.append(pred_label)
            rows.append(
                {
                    "record_id": record.get("id", ""),
                    "category": record.get("category", ""),
                    "query": record.get("query", ""),
                    "claim_index": idx,
                    "claim": gold["claim"],
                    "gold_label": true_label,
                    "predicted_label": pred_label,
                    "support_score": predicted.get("support_score", 0.0),
                    "rule_flags": ";".join(predicted.get("rule_flags", []) or []),
                    "rule_categories": ";".join(predicted.get("rule_categories", []) or []),
                    "nli_label": predicted.get("nli_label", ""),
                    "nli_available": predicted.get("nli_available", False),
                }
            )
    summary = multiclass_classification_report(y_true, y_pred)
    summary.update({"records": len(records), "claims": len(rows), "nli_enabled": enable_nli, "rules_enabled": rules})
    return rows, summary


def is_malformed_output(answer: str) -> bool:
    clean = (answer or "").strip()
    if not clean:
        return True
    if clean == INSUFFICIENT_EVIDENCE_RESPONSE:
        return False
    if clean.count("(") != clean.count(")") or clean.count("[") != clean.count("]"):
        return True
    if not re.search(r"[.!?]$", clean):
        return True
    if len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", clean)) < 3:
        return True
    return False


def deterministic_correct_answer(detector: HallucinationDetector, answer: str, evidence: List[Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
    raw_detection = detector.detect(answer, evidence)
    if raw_detection.get("unsupported_count", 0) == 0:
        return answer.strip(), raw_detection
    kept: list[str] = []
    for claim in raw_detection.get("claims", []) or []:
        if claim.get("label") == "supported":
            kept.append(str(claim.get("claim", "")).strip())
    if not kept:
        return INSUFFICIENT_EVIDENCE_RESPONSE, raw_detection
    corrected = " ".join(claim for claim in kept if claim)
    if corrected and not re.search(r"[.!?]$", corrected):
        corrected += "."
    return corrected or INSUFFICIENT_EVIDENCE_RESPONSE, raw_detection


def evaluate_correction_records(records: Sequence[Dict[str, Any]], enable_nli: bool = False) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    detector = make_detector(enable_nli=enable_nli, rules=True)
    rows: list[dict[str, Any]] = []
    for record in records:
        answer = str(record.get("manual_answer") or " ".join(record.get("gold_supported_claims", []) + record.get("gold_unsupported_claims", []))).strip()
        evidence = record_evidence(record)
        corrected_answer, raw_detection = deterministic_correct_answer(detector, answer, evidence)
        corrected_detection = detector.detect(corrected_answer, evidence)
        metrics = HallucinationMetrics.summarize(raw_detection, corrected_detection)
        malformed = is_malformed_output(corrected_answer)
        no_change = normalize_text(corrected_answer) == normalize_text(answer)
        rows.append(
            {
                "id": record.get("id", ""),
                "category": record.get("category", ""),
                "query": record.get("query", ""),
                "raw_answer": answer,
                "corrected_answer": corrected_answer,
                "raw_unsupported_count": raw_detection.get("unsupported_count", 0),
                "corrected_unsupported_count": corrected_detection.get("unsupported_count", 0),
                "raw_critical_unsupported_count": metrics.get("raw_critical_unsupported_count", 0),
                "corrected_critical_unsupported_count": metrics.get("corrected_critical_unsupported_count", 0),
                "raw_support_ratio": metrics.get("raw_support_ratio", 0.0),
                "corrected_support_ratio": metrics.get("corrected_support_ratio", 0.0),
                "hallucination_reduction": metrics.get("hallucination_reduction", 0.0),
                "support_improvement": metrics.get("factual_improvement", 0.0),
                "correction_success": metrics.get("correction_success", False),
                "unsafe_correction": metrics.get("corrected_critical_unsupported_count", 0) > 0,
                "no_change": no_change,
                "malformed_correction": malformed,
                "correction_status": metrics.get("correction_status", ""),
            }
        )
    total = len(rows)
    summary = {
        "records": total,
        "correction_success_rate": average(1.0 if row["correction_success"] else 0.0 for row in rows),
        "hallucination_reduction": average(float(row["hallucination_reduction"] or 0.0) for row in rows),
        "support_improvement": average(float(row["support_improvement"] or 0.0) for row in rows),
        "unsafe_correction_rate": average(1.0 if row["unsafe_correction"] else 0.0 for row in rows),
        "corrected_unsupported_claim_count": sum(int(row["corrected_unsupported_count"] or 0) for row in rows),
        "corrected_critical_unsupported_claim_count": sum(int(row["corrected_critical_unsupported_count"] or 0) for row in rows),
        "no_change_rate": average(1.0 if row["no_change"] else 0.0 for row in rows),
        "malformed_correction_rate": average(1.0 if row["malformed_correction"] else 0.0 for row in rows),
    }
    return rows, summary


def write_confusion_matrix_png(matrix: Dict[str, Dict[str, int]], path: Path, labels: Sequence[str] = DEFAULT_LABELS) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.array([[matrix.get(t, {}).get(p, 0) for p in labels] for t in labels])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title("Claim-level confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(values[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
