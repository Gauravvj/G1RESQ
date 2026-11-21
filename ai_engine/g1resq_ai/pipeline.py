"""assess_report(): the single function the rest of G1RESQ will call.

Flow: safety rules (instant) -> LLM -> validate (+1 retry) -> evidence check
-> reconcile urgency = max(LLM, rule floor) -> compute missing info in code.
"""
import time
from typing import Optional

from .backends.base import LLMBackend, LLMBackendError
from .backends.ollama_backend import OllamaBackend
from .config import Settings, get_settings
from .parsing import ParseError, check_evidence, parse_assessment
from .prompts.triage_v3 import PROMPT_VERSION, build_messages, build_retry_message
from .safety.rules import check_safety
from .schemas import IncidentAssessment, LLMAssessment, llm_output_schema, max_urgency


def compute_missing_info(a: LLMAssessment) -> list[str]:
    """Missing information is derived from the extracted fields, not asked of the LLM."""
    missing: list[str] = []
    if not a.location:
        missing.append("location")
    if a.people_affected is None:
        missing.append("number_of_people")
    if a.incident_type in ("medical", "accident") and not a.injuries:
        missing.append("injury_details")
    return missing

def build_summary(a: LLMAssessment) -> str:
    """One-line summary built from validated fields. Deterministic, so it cannot hallucinate."""
    parts: list[str] = []
    if a.people_affected is not None:
        parts.append(f"{a.people_affected} {'person' if a.people_affected == 1 else 'people'} affected")
    if a.injuries:
        parts.append(a.injuries)
    if a.location:
        parts.append(f"at {a.location}")
    head = a.incident_type.capitalize()
    return f"{head}: " + ("; ".join(parts) if parts else "no details given") + "."


def assess_report(
    text: str,
    metadata: Optional[dict] = None,   # reserved: language, GPS, timestamp (later phases)
    backend: Optional[LLMBackend] = None,
    settings: Optional[Settings] = None,
) -> IncidentAssessment:
    settings = settings or get_settings()
    backend = backend or OllamaBackend(settings)

    clean = " ".join(text.split())
    if not clean:
        raise ValueError("Report text is empty.")
    report = clean[: settings.max_report_chars]      # what the LLM sees
    safety = check_safety(clean)                     # rules read the FULL text

    started = time.perf_counter()
    model_name = getattr(backend, "name", "")

    def fallback(reason: str) -> IncidentAssessment:
        fb = IncidentAssessment.fallback(
            report, reason, rule_floor=safety.floor, red_flags=safety.red_flags,
            llm_model=model_name, prompt_version=PROMPT_VERSION,
        )
        fb.latency_sec = round(time.perf_counter() - started, 2)
        return fb

    schema = llm_output_schema() if settings.use_json_schema else None
    messages = build_messages(report)
    notes: list[str] = []
    parsed: Optional[LLMAssessment] = None
    last_error = ""

    for attempt in (1, 2):
        try:
            result = backend.generate(messages, schema)
        except LLMBackendError as exc:
            return fallback(f"LLM backend error: {exc}")
        try:
            parsed = parse_assessment(result.text)
            if attempt == 2:
                notes.append("retry_used")
            break
        except ParseError as exc:
            last_error = str(exc) + (" The output was cut off." if result.truncated else "")
            if attempt == 1:
                messages = messages + [
                    {"role": "assistant", "content": result.text[:1000]},
                    build_retry_message(last_error),
                ]

    if parsed is None:
        return fallback(f"Invalid model output after retry: {last_error}")

    supported, unsupported = check_evidence(report, parsed.evidence)
    if unsupported:
        notes.append("evidence quote not found in report")
    if not supported:
        notes.append("no verifiable evidence")

    final_urgency = max_urgency(parsed.urgency, safety.floor)
    if final_urgency != parsed.urgency:
        notes.append(f"urgency raised by safety rules: {parsed.urgency} -> {final_urgency}")

    needs_review = (
        final_urgency in ("critical", "high")
        or bool(safety.red_flags)
        or bool(unsupported)
        or not supported
    )

    return IncidentAssessment(
        report_text=report,
        incident_type=parsed.incident_type,
        urgency=final_urgency,
        llm_urgency=parsed.urgency,
        rule_urgency_floor=safety.floor,
        red_flags=safety.red_flags,
        people_affected=parsed.people_affected,
        injuries=parsed.injuries,
        location=parsed.location,
        missing_info=compute_missing_info(parsed),
        summary=build_summary(parsed),
        evidence=supported,
        unsupported_evidence=unsupported,
        needs_human_review=needs_review,
        fallback_used=False,
        notes=notes,
        llm_model=model_name,
        prompt_version=PROMPT_VERSION,
        latency_sec=round(time.perf_counter() - started, 2),
    )