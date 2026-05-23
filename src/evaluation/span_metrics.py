from __future__ import annotations

from typing import Dict, Iterable, List


def _span_set(spans: Iterable[Dict[str, int]]) -> set[int]:
    positions: set[int] = set()
    for span in spans:
        start = int(span.get("start", 0))
        end = int(span.get("end", 0))
        positions.update(range(start, max(start, end)))
    return positions


def span_iou(predicted_spans: List[Dict[str, int]], gold_spans: List[Dict[str, int]]) -> float:
    pred = _span_set(predicted_spans)
    gold = _span_set(gold_spans)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    return round(len(pred & gold) / len(pred | gold), 4)
