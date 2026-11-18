"""Settings: model name, Ollama URL, context size, timeouts.

Status: placeholder. Implemented in M1.
"""
"""Runtime settings for the AI engine.

Values come from environment variables when set (names match .env.example),
otherwise from the defaults below. The defaults are tuned for an 8 GB CPU-only
laptop: small context window, short output, deterministic sampling.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://localhost:11434"
    model_name: str = "gemma3n:e2b"
    num_ctx: int = 2048            # context window; small = less RAM, faster
    num_predict: int = 200         # hard cap on output tokens
    temperature: float = 0.0       # deterministic output for triage
    request_timeout_sec: int = 300
    keep_alive: str = "10m"        # keep the model in RAM between requests
    max_report_chars: int = 2000   # longer reports are truncated


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        ollama_url=env("OLLAMA_URL", Settings.ollama_url),
        model_name=env("MODEL_NAME", Settings.model_name),
        num_ctx=int(env("NUM_CTX", Settings.num_ctx)),
        num_predict=int(env("NUM_PREDICT", Settings.num_predict)),
        request_timeout_sec=int(env("REQUEST_TIMEOUT_SEC", Settings.request_timeout_sec)),
    )
    
    