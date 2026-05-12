import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.result_formatter import flatten_pipeline_result
from src.utils.json_utils import load_jsonl


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python scripts/export_results_table.py <input_jsonl> <output_csv>")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    records = load_jsonl(input_path)
    flattened = [flatten_pipeline_result(record) for record in records]
    pd.DataFrame(flattened).to_csv(output_path, index=False)
    print(json.dumps({"rows": len(flattened), "output_csv": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
