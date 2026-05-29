from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


DEFAULT_LABELS = ["supported", "weak_support", "unsupported"]


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


def confusion_matrix_counts(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> Dict[str, Dict[str, int]]:
    matrix = {true: {pred: 0 for pred in labels} for true in labels}
    for true, pred in zip(y_true, y_pred):
        true_label = true if true in labels else "unsupported"
        pred_label = pred if pred in labels else "unsupported"
        matrix[true_label][pred_label] += 1
    return matrix


def _binary_counts(y_true: List[str], y_pred: List[str], label: str) -> tuple[int, int, int, int]:
    tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
    fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
    fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
    tn = sum(t != label and p != label for t, p in zip(y_true, y_pred))
    return tp, fp, fn, tn


def multiclass_classification_report(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: Sequence[str] = DEFAULT_LABELS,
    positive_label: str = "unsupported",
) -> Dict[str, object]:
    true = list(y_true)
    pred = list(y_pred)
    total = len(true)
    accuracy = sum(t == p for t, p in zip(true, pred)) / total if total else 0.0
    per_label: Dict[str, Dict[str, float]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp, fp, fn, _tn = _binary_counts(true, pred, label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in true if t == label),
        }
        f1_values.append(f1)
    tp, fp, fn, tn = _binary_counts(true, pred, positive_label)
    positive = precision_recall_f1(true, pred, positive_label=positive_label)
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "count": total,
        "accuracy": round(accuracy, 4),
        "precision": positive["precision"],
        "recall": positive["recall"],
        "f1": positive["f1"],
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "unsupported_recall": positive["recall"],
        "false_positive_rate": round(false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
        "per_label": per_label,
        "confusion_matrix": confusion_matrix_counts(true, pred, labels),
    }


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
