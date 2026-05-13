"""Backward-compatible wrapper.

Use scripts/build_vector_index.py as the canonical vector-index builder.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_vector_index import main


if __name__ == "__main__":
    main()
