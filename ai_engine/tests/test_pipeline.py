import json

import pytest

from g1resq_ai.backends.base import GenerationResult, LLMBackend, LLMBackendError
from g1resq_ai.config import Settings
from g1resq_ai.pipeline import assess_report
from g1resq_ai.schemas import llm_output_schema

REPORT = "My father is not breathing at 12 Park Street"

GOOD = {
    "incident_type": "medical",
    "urgency": "critical",
    "people_affected": 1,
    "injuries": "not breathing",
    "location": "12 Park Street",
    "summary": "Man not breathing.",
    "evidence": ["not breathing"],
}


class FakeBackend(LLMBackend):
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, messages, json_schema=None):
        self.calls.append((messages, json_schema))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return GenerationResult(text=reply)


def run(replies, text=REPORT):
    backend = FakeBackend(replies)
    return assess_report(text, backend=backend, settings=Settings()), backend


def test_valid_first_try():
    a, b = run([json.dumps(GOOD)])
    assert a.urgency == "critical" and not a.fallback_used
    assert a.needs_human_review is True
    assert len(b.calls) == 1 and "retry_used" not in a.notes


def test_schema_sent_to_model_hides_fields_computed_in_code():
    _, b = run([json.dumps(GOOD)])
    props = b.calls[0][1]["properties"]
    assert "missing_info" not in props and "summary" not in props


def test_retry_with_error_feedback_then_success():
    a, b = run(["not json at all", json.dumps(GOOD)])
    assert not a.fallback_used and "retry_used" in a.notes
    assert len(b.calls) == 2 and len(b.calls[1][0]) == 4


def test_two_failures_use_rule_floor_in_fallback():
    a, _ = run(["bad", "still bad"])          # REPORT contains "not breathing"
    assert a.fallback_used and a.urgency == "critical" and a.needs_human_review


def test_two_failures_without_red_flags_give_high():
    a, _ = run(["bad", "still bad"], text="Something happened near my house")
    assert a.fallback_used and a.urgency == "high"


def test_backend_error_gives_fallback_without_retry():
    a, b = run([LLMBackendError("down")])
    assert a.fallback_used and len(b.calls) == 1


def test_invented_evidence_is_flagged_even_for_low_urgency():
    reply = dict(GOOD, urgency="low", evidence=["he is unconscious"])
    a, _ = run([json.dumps(reply)])
    assert a.unsupported_evidence == ["he is unconscious"]
    assert a.needs_human_review is True


def test_rules_raise_an_llm_underestimate():
    reply = dict(GOOD, urgency="high", evidence=["unconscious"])
    a, _ = run([json.dumps(reply)], text="My brother is unconscious after falling from the roof")
    assert a.llm_urgency == "high" and a.urgency == "critical"
    assert a.red_flags and any("raised by safety rules" in n for n in a.notes)


def test_llm_can_go_above_the_floor():
    reply = dict(GOOD, urgency="critical", evidence=["bleeding"])
    a, _ = run([json.dumps(reply)], text="He is bleeding from the head")
    assert a.urgency == "critical" and a.rule_urgency_floor == "high"


def test_rules_read_text_beyond_the_llm_limit():
    long_text = ("The street is quiet. " * 200) + "Someone is not breathing."
    a, _ = run([json.dumps(dict(GOOD, urgency="low", evidence=[]))], text=long_text)
    assert a.urgency == "critical"


def test_missing_info_is_computed_in_code():
    reply = dict(GOOD, people_affected=None, location=None, injuries=None, missing_info=["contact_info"])
    a, _ = run([json.dumps(reply)])
    assert a.missing_info == ["location", "number_of_people", "injury_details"]


def test_missing_info_empty_when_everything_present():
    a, _ = run([json.dumps(GOOD)])
    assert a.missing_info == []


def test_empty_report_is_rejected():
    with pytest.raises(ValueError):
        assess_report("   ", backend=FakeBackend([]), settings=Settings())


def test_llm_output_schema_caps_evidence_at_two():
    assert llm_output_schema()["properties"]["evidence"]["maxItems"] == 2
    
def test_summary_is_built_in_code_not_by_the_llm():
    reply = dict(GOOD, summary="The model wrote this and it must be ignored.")
    a, _ = run([json.dumps(reply)])
    assert a.summary == "Medical: 1 person affected; not breathing; at 12 Park Street."


def test_summary_when_no_details_are_given():
    reply = dict(GOOD, incident_type="other", people_affected=None, injuries=None, location=None)
    a, _ = run([json.dumps(reply)])
    assert a.summary == "Other: no details given."


def test_reply_without_summary_field_is_valid():
    reply = {k: v for k, v in GOOD.items() if k != "summary"}
    a, _ = run([json.dumps(reply)])
    assert not a.fallback_used