import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_all_configs
from src.generation.ollama_client import OllamaClient
from src.logger import get_logger
from src.paths import RAW_DATA_DIR, ensure_directories


def main() -> None:
    logger = get_logger("setup_check")

    logger.info("Creating required directories...")
    ensure_directories()

    logger.info("Loading configuration files...")
    configs = load_all_configs()

    logger.info("Project setup loaded successfully.")
    logger.info("Project name: %s", configs["settings"]["project"]["name"])
    logger.info("Generator model: %s", configs["models"]["generator"]["primary_model"])
    logger.info("Fallback model: %s", configs["models"]["generator"]["fallback_model"])
    logger.info("Active profile: %s", configs["settings"].get("runtime", {}).get("profile", "demo"))
    logger.info("Embedding model: %s", configs["models"]["embedding"]["model_name"])
    logger.info("Embedding backend: %s", configs["settings"].get("runtime", {}).get("embedding_backend", "auto"))
    logger.info("NLI enabled: %s", configs["settings"].get("verification", {}).get("enable_nli", False))
    logger.info(
        "NLI verification model: %s",
        configs["models"]["verification"]["nli_model_name"],
    )

    raw_files = sum(1 for path in RAW_DATA_DIR.rglob("*") if path.is_file())
    logger.info("Raw knowledge-base files found: %s", raw_files)

    ollama_on_path = shutil.which("ollama") is not None
    logger.info("Ollama CLI available on PATH: %s", ollama_on_path)

    client = OllamaClient()
    ok, message = client.health_check()
    if ok:
        logger.info("Ollama health check: %s", message)
    else:
        logger.warning("Ollama health check skipped/failed: %s", message)

    logger.info("All basic setup checks passed.")


if __name__ == "__main__":
    main()
