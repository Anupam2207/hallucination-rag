from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_TXT_DIR = RAW_DATA_DIR / "txt"
RAW_MD_DIR = RAW_DATA_DIR / "md"
RAW_JSON_DIR = RAW_DATA_DIR / "json"
RAW_PDF_DIR = RAW_DATA_DIR / "pdf"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
EVAL_DIR = DATA_DIR / "eval"
SAMPLES_DIR = DATA_DIR / "samples"

DB_DIR = PROJECT_ROOT / "db"
CHROMA_DIR = DB_DIR / "chroma"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"


def ensure_directories() -> None:
    """Create essential directories if they do not already exist."""
    required_dirs = [
        CONFIG_DIR,
        RAW_DATA_DIR,
        RAW_TXT_DIR,
        RAW_MD_DIR,
        RAW_JSON_DIR,
        RAW_PDF_DIR,
        PROCESSED_DATA_DIR,
        CHUNKS_DIR,
        EVAL_DIR,
        SAMPLES_DIR,
        CHROMA_DIR,
        OUTPUTS_DIR,
        RUNS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
