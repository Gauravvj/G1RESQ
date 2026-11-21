"""Ollama HTTP backend (CPU-friendly, works fully offline)."""
from typing import Optional

import requests

from ..config import Settings, get_settings
from .base import GenerationResult, LLMBackend, LLMBackendError


class OllamaBackend(LLMBackend):
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.name = self.settings.model_name

    def _post(self, fmt, messages) -> requests.Response:
        s = self.settings
        payload = {
            "model": s.model_name,
            "messages": messages,
            "stream": False,
            "format": fmt,                 # a JSON schema dict, or the string "json"
            "keep_alive": s.keep_alive,
            "options": {
                "temperature": s.temperature,
                "num_ctx": s.num_ctx,
                "num_predict": s.num_predict,
            },
        }
        try:
            return requests.post(
                f"{s.ollama_url}/api/chat", json=payload, timeout=s.request_timeout_sec
            )
        except requests.exceptions.ConnectionError as exc:
            raise LLMBackendError("Cannot reach Ollama. Is it running?") from exc
        except requests.exceptions.Timeout as exc:
            raise LLMBackendError(f"Ollama timed out after {s.request_timeout_sec}s") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMBackendError(str(exc)) from exc

    def generate(self, messages, json_schema=None) -> GenerationResult:
        resp = self._post(json_schema or "json", messages)
        if resp.status_code == 400 and json_schema is not None:
            # Schema rejected (older Ollama or unsupported keyword): plain JSON mode.
            resp = self._post("json", messages)
        if resp.status_code != 200:
            raise LLMBackendError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMBackendError("Ollama returned a non-JSON response") from exc
        return GenerationResult(
            text=data.get("message", {}).get("content", ""),
            output_tokens=data.get("eval_count", 0),
            total_sec=data.get("total_duration", 0) / 1e9,
            truncated=data.get("done_reason") == "length",
        )