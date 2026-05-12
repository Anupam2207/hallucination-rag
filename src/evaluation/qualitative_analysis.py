from typing import Any, Dict, List


def select_most_improved(results: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    return sorted(
        results,
        key=lambda item: item.get("metrics", {}).get("factual_improvement", 0.0),
        reverse=True,
    )[:top_n]
