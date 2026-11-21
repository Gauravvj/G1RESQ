import pytest

from g1resq_ai.parsing import ParseError, check_evidence, extract_json, parse_assessment


def test_extract_json_from_code_fence_and_chatter():
    assert extract_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_rejects_garbage():
    with pytest.raises(ParseError):
        extract_json("I cannot help with that.")


def test_extract_json_rejects_broken_json():
    with pytest.raises(ParseError):
        extract_json('{"a": 1,,}')


def test_parse_error_names_the_bad_field():
    with pytest.raises(ParseError) as exc:
        parse_assessment('{"incident_type":"fire","urgency":"extreme","summary":"x"}')
    assert "urgency" in str(exc.value)


def test_evidence_ignores_case_and_punctuation():
    report = "Bhai papa ko saans nahi aa rahi, jaldi ambulance bhejo!"
    ok, bad = check_evidence(report, ["Saans nahi aa rahi", "papa ko!"])
    assert len(ok) == 2 and bad == []


def test_evidence_flags_invented_quote():
    ok, bad = check_evidence("My father is not breathing.", ["he is unconscious"])
    assert ok == [] and bad == ["he is unconscious"]