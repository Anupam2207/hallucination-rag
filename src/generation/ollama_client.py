import os
from typing import Any

from dotenv import load_dotenv

from src.config import load_all_configs


load_dotenv()


class OllamaServiceError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


class OllamaClient:
    """Wrapper for the local Ollama Python client.

    This class handles model selection, fallback logic, and basic health checks.
    """
    def __init__(
        self,
        host: str | None = None,
        primary_model: str | None = None,
        fallback_model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        configs = load_all_configs()
        generation_cfg = configs["settings"]["generation"]
        models_cfg = configs["models"]["generator"]

        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.primary_model = primary_model or models_cfg["primary_model"]
        self.fallback_model = fallback_model or models_cfg.get("fallback_model")
        self.temperature = generation_cfg.get("temperature", 0.2) if temperature is None else temperature
        self.client = None
        self._import_error: str | None = None

        try:
            import ollama  # type: ignore

            self.client = ollama.Client(host=self.host)
        except Exception as exc:  # pragma: no cover - environment dependent
            self._import_error = str(exc)

    def _ensure_client(self) -> Any:
        if self.client is None:
            raise OllamaServiceError(
                "Ollama Python client is unavailable. Install the 'ollama' package and ensure the local "
                f"service is running. Import/init error: {self._import_error}"
            )
        return self.client

    def generate(self, prompt: str, model: str | None = None, temperature: float | None = None) -> str:
        client = self._ensure_client()
        target_model = model or self.primary_model
        options: dict[str, Any] = {"temperature": self.temperature if temperature is None else temperature}

        try:
            response = client.generate(model=target_model, prompt=prompt, options=options)
            return response["response"].strip()
        except Exception as exc:
            # If the primary model fails, attempt to fall back to a secondary model.
            if self.fallback_model and target_model != self.fallback_model:
                try:
                    response = client.generate(
                        model=self.fallback_model,
                        prompt=prompt,
                        options=options,
                    )
                    return response["response"].strip()
                except Exception:
                    pass
            raise OllamaServiceError(
                f"Failed to generate with Ollama at {self.host}. "
                f"Ensure Ollama is running and model '{target_model}' is installed. "
                f"Original error: {exc}"
            ) from exc

    def health_check(self) -> tuple[bool, str]:
        if self.client is None:
            return False, self._import_error or "Ollama Python client unavailable"
        try:
            self.client.list()
            return True, f"Ollama reachable at {self.host}. Installed models response received."
        except Exception as exc:  # pragma: no cover - environment dependent
            return False, str(exc)
