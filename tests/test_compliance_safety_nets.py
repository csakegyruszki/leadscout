"""Safety nets in compliance.screen: OpenSanctions hard evidence and an exact
abbreviation hit can never be silently cleared or under-reported by the LLM verdict.
Monkeypatches ask_model and screen_sanctions so this runs offline.
"""
import pytest

from leadscout import compliance
from leadscout.models import ComplianceVerdict, Lead, Research
from leadscout.sanctions import SanctionsScreen


def _clear_verdict(system, user, schema, *, purpose="compliance"):
    return ComplianceVerdict(status="clear", flagged=False, matches=[], reasoning="looks fine")


def test_blocked_evidence_forces_blocked_even_if_llm_says_clear(monkeypatch):
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(
        compliance, "screen_sanctions",
        lambda company, hq, run=None: SanctionsScreen(status="blocked_evidence",
                                             hits=[{"caption": "Rosneft", "score": 1.0, "topics": ["sanction"],
                                                    "match": True, "schema": "Company", "datasets": [],
                                                    "url": "https://x", "origin": "opensanctions:match"}]),
    )
    result = compliance.screen(
        Lead("a", "a@b.c", "Rosneft", "https://rosneft.ru"), Research(headquarters_country="Russia"))
    assert result.status == "blocked"
    assert result.sanctions_status == "blocked_evidence"
    assert result.sanctions_hits and result.sanctions_hits[0]["original_value"] == "Rosneft"


def test_review_evidence_upgrades_clear_to_review(monkeypatch):
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions",
                         lambda company, hq, run=None: SanctionsScreen(status="review_evidence", hits=[]))
    result = compliance.screen(Lead("a", "a@b.c", "X", "https://x.com"), Research())
    assert result.status == "review"


def test_review_evidence_does_not_downgrade_an_already_blocked_verdict(monkeypatch):
    def _blocked_verdict(system, user, schema, *, purpose="compliance"):
        return ComplianceVerdict(status="blocked", flagged=True, matches=[], reasoning="competitor match")

    monkeypatch.setattr(compliance, "ask_model", _blocked_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions",
                         lambda company, hq, run=None: SanctionsScreen(status="review_evidence", hits=[]))
    result = compliance.screen(Lead("a", "a@b.c", "CloudTrim Inc", "https://cloudtrim.com"), Research())
    assert result.status == "blocked"


def test_abbreviation_hit_upgrades_clear_to_review(monkeypatch):
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions", lambda company, hq, run=None: SanctionsScreen(status="none"))
    result = compliance.screen(Lead("a", "a@b.c", "SWC", "https://swc.example.com"), Research())
    assert result.status == "review"


def test_llm_matches_get_provenance_defaults(monkeypatch):
    def _verdict_with_match(system, user, schema, *, purpose="compliance"):
        return ComplianceVerdict(status="review", flagged=True,
                                  matches=[{"kind": "competitor", "term": "CloudTrim Inc", "score": 80,
                                            "reason": "shared token"}],
                                  reasoning="partial match")

    monkeypatch.setattr(compliance, "ask_model", _verdict_with_match)
    monkeypatch.setattr(compliance, "screen_sanctions", lambda company, hq, run=None: SanctionsScreen(status="none"))
    result = compliance.screen(Lead("a", "a@b.c", "Cloud Trim Solutions", "https://x.com"), Research())
    assert result.matches[0]["original_value"] == "Cloud Trim Solutions"
    assert result.matches[0]["origin"] == "llm:reasoning"


# --- Safety net 5: a screen that did not run cannot clear a lead ------------------

def _skipped_screen(note, reason):
    return lambda company, hq, run=None: SanctionsScreen(
        status="skipped", hits=[], note=note, reason=reason)


DEGRADED_REASONS = [
    "OpenSanctions request failed: ConnectTimeout: timed out",
    "OpenSanctions 503: upstream unavailable",
    "OpenSanctions 200 with an unusable body: '{bad'",
    "the cached screen from 2000-01-01 is 9756 days old (limit 30 days) and was not "
    "treated as current; no API key to revalidate it",
]


@pytest.mark.parametrize("note", DEGRADED_REASONS)
def test_a_configured_control_that_failed_cannot_clear_a_lead(note, monkeypatch):
    """Every one of these is a control that was configured and did not deliver, and all
    of them used to pass through untouched - an outage, a 200 nobody could parse and a
    decade-old cached answer all cleared the lead on the model's word alone. The
    provider's failure is not the provider's answer."""
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions", _skipped_screen(note, "degraded"))
    result = compliance.screen(
        Lead("a", "a@b.c", "Clean Co", "https://clean.example"),
        Research(headquarters_country="Germany", hq_source="gleif"))
    assert result.status == "review", "a failed control cleared the lead"
    assert any("did not complete" in r for r in [result.reasoning or ""] + list(result.matches and
               [m.get("reason", "") for m in result.matches] or [])), result.reasoning


def test_an_unconfigured_optional_control_is_recorded_rather_than_escalated(monkeypatch):
    """"Never installed" is not "tried and could not tell". Entity screening is an
    optional extra: the core do-not-engage rule is the restricted-jurisdiction
    check, which is deterministic and needs no API key. Sending every lead to review
    because an optional control is absent would fill a first run with rows
    nobody can act on - so the absence is recorded in the result, not inferred from
    silence and not escalated."""
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions",
                        _skipped_screen("OPENSANCTIONS_API_KEY is not set (see .env.example)",
                                        "not_configured"))
    result = compliance.screen(
        Lead("a", "a@b.c", "Clean Co", "https://clean.example"),
        Research(headquarters_country="Germany", hq_source="gleif"))
    assert result.status == "clear"
    assert "not configured" in (result.reasoning or "").lower(), result.reasoning


def test_an_unconfigured_control_still_cannot_override_a_restricted_jurisdiction(monkeypatch):
    """The control that the absent one is NOT standing in for: with no key at all, an
    established HQ in a restricted jurisdiction is still blocked deterministically."""
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions",
                        _skipped_screen("OPENSANCTIONS_API_KEY is not set", "not_configured"))
    result = compliance.screen(
        Lead("a", "a@b.c", "Tehran Systems", "https://example.ir"),
        Research(headquarters_country="Iran", hq_source="gleif"))
    assert result.status == "blocked"


def test_a_completed_clean_screen_still_clears(monkeypatch):
    """The control: safety net 5 must cost nothing when the screen actually ran."""
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions",
                        lambda company, hq, run=None: SanctionsScreen(status="none", hits=[]))
    result = compliance.screen(
        Lead("a", "a@b.c", "Clean Co", "https://clean.example"),
        Research(headquarters_country="Germany", hq_source="gleif"))
    assert result.status == "clear"
