"""Pydantic schemas for LLM output: coercion rules and the one-retry validation contract."""
import pytest
from pydantic import ValidationError

from leadscout import llm
from leadscout.models import ComplianceVerdict, ResearchFacts


@pytest.mark.parametrize("raw,expected", [
    (150, 150),
    (150.0, 150),
    ("200", 200),
    (0, None),
    (-5, None),
    (None, None),
    ("not a number", None),
])
def test_research_facts_coerces_employees(raw, expected):
    facts = ResearchFacts(estimated_employees=raw, confidence="low")
    assert facts.estimated_employees == expected


def test_research_facts_rejects_bad_confidence():
    with pytest.raises(ValidationError):
        ResearchFacts(confidence="very high")


def test_compliance_verdict_accepts_known_status():
    v = ComplianceVerdict(status="blocked", flagged=True, matches=[], reasoning="x")
    assert v.status == "blocked"


def test_compliance_verdict_rejects_unknown_status():
    with pytest.raises(ValidationError):
        ComplianceVerdict(status="maybe")


def test_ask_model_retries_once_on_validation_error(monkeypatch):
    calls = []

    def fake_ask_json(system, user, *, temperature=0.1, purpose="unknown"):
        calls.append(user)
        if len(calls) == 1:
            return {"confidence": "extremely high"}  # invalid literal -> ValidationError
        return {"summary": "ok", "confidence": "high"}

    monkeypatch.setattr(llm, "ask_json", fake_ask_json)
    result = llm.ask_model("sys", "user", ResearchFacts, purpose="research")
    assert result.confidence == "high"
    assert len(calls) == 2
    assert "failed validation" in calls[1]


def test_ask_model_raises_after_second_failure(monkeypatch):
    def fake_ask_json(system, user, *, temperature=0.1, purpose="unknown"):
        return {"confidence": "nope"}

    monkeypatch.setattr(llm, "ask_json", fake_ask_json)
    with pytest.raises(ValidationError):
        llm.ask_model("sys", "user", ResearchFacts, purpose="research")


def test_ask_model_succeeds_first_try_without_extra_call(monkeypatch):
    calls = []

    def fake_ask_json(system, user, *, temperature=0.1, purpose="unknown"):
        calls.append(user)
        return {"status": "clear", "flagged": False, "matches": [], "reasoning": "ok"}

    monkeypatch.setattr(llm, "ask_json", fake_ask_json)
    result = llm.ask_model("sys", "user", ComplianceVerdict, purpose="compliance")
    assert result.status == "clear"
    assert len(calls) == 1
