from __future__ import annotations

from typing import Dict, Iterable, List


def precision_recall_f1(y_true: Iterable[str], y_pred: Iterable[str], positive_label: str = "unsupported") -> Dict[str, float]:
    true: List[str] = list(y_true)
    pred: List[str] = list(y_pred)
    tp = sum(t == positive_label and p == positive_label for t, p in zip(true, pred))
    fp = sum(t != positive_label and p == positive_label for t, p in zip(true, pred))
    fn = sum(t == positive_label and p != positive_label for t, p in zip(true, pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def faithfulness_score(detection_result: Dict) -> float:
    total = int(detection_result.get("claim_count", 0) or 0)
    if total == 0:
        return 0.0
    supported = int(detection_result.get("supported_count", 0) or 0)
    weak = int(detection_result.get("weak_count", 0) or 0)
    return round((supported + 0.5 * weak) / total, 4)


def completeness_score(raw_detection: Dict, corrected_detection: Dict) -> float:
    raw_supported = int(raw_detection.get("supported_count", 0) or 0) + int(raw_detection.get("weak_count", 0) or 0)
    corrected_supported = int(corrected_detection.get("supported_count", 0) or 0) + int(corrected_detection.get("weak_count", 0) or 0)
    if raw_supported == 0:
        return 0.0
    return round(min(1.0, corrected_supported / raw_supported), 4)


def citation_coverage(answer: str, claim_count: int) -> float:
    if claim_count <= 0:
        return 0.0
    citations = answer.count("[Evidence-")
    return round(min(1.0, citations / claim_count), 4)
