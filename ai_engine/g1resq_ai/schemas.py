
"""Data contracts for the G1RESQ AI engine.

Two models on purpose:
  * LLMAssessment      what we ask the LLM to produce (small and strict)
  * IncidentAssessment what the rest of the system receives (LLM output plus
                       safety-layer results and audit fields)

AI output is decision support only. A human makes every triage decision.
"""


from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

IncidentType = Literal["fire", "flood", "earthquake", "accident", "medical", "other"]
Urgency = Literal["critical", "high", "medium", "low"]
MissingInfo = Literal[
    "location", "number_of_people", "injury_details", "hazard_details", "contact_info"
]

URGENCY_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}

_EMPTY_WORDS = {"", "null", "none", "unknown", "n/a", "na"}


def max_urgency(*levels: Optional[str]) -> Optional[str]:
    """Return the most severe urgency among the given levels (None is ignored)."""
    present = [lvl for lvl in levels if lvl is not None]
    if not present:
        return None
    return max(present, key=URGENCY_RANK.__getitem__)


class LLMAssessment(BaseModel):
    """The compact JSON the LLM must return. Unknown extra keys are ignored."""

    model_config = ConfigDict(extra="ignore")

    incident_type: IncidentType
    urgency: Urgency
    people_affected: Optional[int] = Field(default=None, ge=0, le=100_000)
    injuries: Optional[str] = Field(default=None, max_length=200)
    location: Optional[str] = Field(default=None, max_length=200)
    missing_info: list[MissingInfo] = Field(default_factory=list)
    summary: str = Field(max_length=400)
    evidence: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("incident_type", "urgency", mode="before")
    @classmethod
    def _normalize_choice(cls, v):
        # Small models vary case/spacing: " Critical " -> "critical"
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("people_affected", "location", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        if isinstance(v, str) and v.strip().lower() in _EMPTY_WORDS:
            return None
        return v

    @field_validator("missing_info", "evidence", mode="before")
    @classmethod
    def _none_to_list(cls, v):
        return [] if v is None else v


class IncidentAssessment(BaseModel):
    """Final validated result returned by assess_report()."""

    report_text: str
    incident_type: IncidentType
    urgency: Urgency                          # final = max(LLM, rule floor)
    llm_urgency: Optional[Urgency] = None     # what the LLM said (None if it failed)
    rule_urgency_floor: Optional[Urgency] = None
    red_flags: list[str] = Field(default_factory=list)
    people_affected: Optional[int] = None
    injuries: Optional[str] = None
    location: Optional[str] = None
    missing_info: list[MissingInfo] = Field(default_factory=list)
    summary: str
    evidence: list[str] = Field(default_factory=list)
    unsupported_evidence: list[str] = Field(default_factory=list)
    needs_human_review: bool = True
    fallback_used: bool = False
    notes: list[str] = Field(default_factory=list)
    llm_model: str = ""
    prompt_version: str = ""
    latency_sec: Optional[float] = None

    @classmethod
    def fallback(
        cls,
        report_text: str,
        reason: str,
        rule_floor: Optional[str] = None,
        red_flags: Optional[list[str]] = None,
        llm_model: str = "",
        prompt_version: str = "",
    ) -> "IncidentAssessment":
        """Safe result when the LLM fails: never crash, never guess low."""
        return cls(
            report_text=report_text,
            incident_type="other",
            urgency=max_urgency("high", rule_floor),
            rule_urgency_floor=rule_floor,
            red_flags=red_flags or [],
            summary="Automatic assessment unavailable. Human review required.",
            needs_human_review=True,
            fallback_used=True,
            notes=[reason],
            llm_model=llm_model,
            prompt_version=prompt_version,
        )