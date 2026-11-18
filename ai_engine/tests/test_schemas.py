"""Schema validation tests.

Status: placeholder. Implemented in M1.
"""
import pytest
from pydantic import ValidationError

from g1resq_ai.schemas import IncidentAssessment, LLMAssessment, max_urgency


def test_valid_assessment_is_normalized():
    a = LLMAssessment.model_validate(
        {
            "incident_type": " Medical ",
            "urgency": "CRITICAL",
            "people_affected": "1",
            "location": "unknown",
            "missing_info": None,
            "summary": "Man not breathing.",
            "evidence": ["not breathing"],
            "extra_key_from_model": "ignored",
        }
    )
    assert a.incident_type == "medical"
    assert a.urgency == "critical"
    assert a.people_affected == 1
    assert a.location is None
    assert a.missing_info == []


def test_invalid_urgency_is_rejected():
    with pytest.raises(ValidationError):
        LLMAssessment.model_validate(
            {"incident_type": "fire", "urgency": "extreme", "summary": "x"}
        )


def test_invalid_missing_info_value_is_rejected():
    with pytest.raises(ValidationError):
        LLMAssessment.model_validate(
            {"incident_type": "fire", "urgency": "low", "summary": "x",
             "missing_info": ["favorite_color"]}
        )


def test_negative_people_rejected():
    with pytest.raises(ValidationError):
        LLMAssessment.model_validate(
            {"incident_type": "fire", "urgency": "low", "summary": "x",
             "people_affected": -3}
        )


def test_max_urgency():
    assert max_urgency("low", "critical") == "critical"
    assert max_urgency("high", None) == "high"
    assert max_urgency(None, None) is None


def test_fallback_is_safe():
    f = IncidentAssessment.fallback("some report", "LLM timeout")
    assert f.urgency == "high"          # unknown is never rated low
    assert f.needs_human_review is True
    assert f.fallback_used is True


def test_fallback_respects_rule_floor():
    f = IncidentAssessment.fallback("not breathing", "LLM timeout", rule_floor="critical")
    assert f.urgency == "critical"