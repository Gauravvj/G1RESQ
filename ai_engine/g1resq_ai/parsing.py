"""Turn raw model text into a validated LLMAssessment, and verify its evidence."""
import json
import re
import unicodedata

from pydantic import ValidationError

from .schemas import LLMAssessment

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ParseError(ValueError):
    """Model output could not be parsed or failed schema validation.

    The message is written so it can be fed back to the model on retry.
    """


def extract_json(text: str) -> dict:
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ParseError("No JSON object found in the reply.")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ParseError(f"Invalid JSON: {exc.msg} at position {exc.pos}.") from exc
    if not isinstance(data, dict):
        raise ParseError("The JSON must be an object, not a list or value.")
    return data


def parse_assessment(text: str) -> LLMAssessment:
    data = extract_json(text)
    try:
        return LLMAssessment.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors()[:5]
        )
        raise ParseError(f"Schema validation failed: {problems}") from exc


def _normalize(s: str) -> str:
    """Casefold, drop punctuation, collapse whitespace (works for any script)."""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    return " ".join(s.split())


def check_evidence(report: str, evidence: list[str]) -> tuple[list[str], list[str]]:
    """Split evidence quotes into (found in the report, not found in the report)."""
    haystack = _normalize(report)
    supported, unsupported = [], []
    for quote in evidence:
        q = _normalize(quote)
        (supported if q and q in haystack else unsupported).append(quote)
    return supported, unsupported