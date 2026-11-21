"""The one interface every LLM runtime must implement.

The assessment pipeline only knows this interface, so swapping Ollama for
llama.cpp or a test double never touches the assessment logic.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class LLMBackendError(RuntimeError):
    """The model could not be reached or returned an unusable HTTP response."""


@dataclass
class GenerationResult:
    text: str
    output_tokens: int = 0
    total_sec: float = 0.0
    truncated: bool = False   # True if the model hit the token limit mid-answer


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self, messages: list[dict], json_schema: Optional[dict] = None
    ) -> GenerationResult:
        """Send chat messages; optionally constrain output to a JSON schema."""