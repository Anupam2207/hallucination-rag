import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from src.config import load_all_configs
from src.logger import get_logger
from src.paths import ensure_directories


def main() -> None:
    load_dotenv()

    logger = get_logger("setup_check")
    logger.info("Creating required directories...")
    ensure_directories()

    logger.info("Loading configuration files...")
    configs = load_all_configs()

    logger.info("Project setup loaded successfully.")
    logger.info("Project name: %s", configs["settings"]["project"]["name"])
    logger.info("Generator model: %s", configs["models"]["generator"]["primary_model"])
    logger.info("Fallback model: %s", configs["models"]["generator"]["fallback_model"])
    logger.info("Embedding model: %s", configs["models"]["embedding"]["model_name"])
    logger.info(
        "NLI verification model: %s",
        configs["models"]["verification"]["nli_model_name"],
    )
    logger.info("OLLAMA_HOST: %s", os.getenv("OLLAMA_HOST", "Not set"))
    logger.info("All basic setup checks passed.")


if __name__ == "__main__":
    main()