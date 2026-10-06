"""Regressions from the pre-release audit (2026-09-20).

Three of them exist because running the audit's own repro contradicted the audit: a
sanctioned headquarters did not stay CLEAR as reported - it reached REVIEW - but the
sentence explaining it said "a near-exact name match to a do-not-engage competitor was
found", about a lead whose only hit was its jurisdiction. A false statement to the
person acting on it, and the reason invariant 9 exists.
"""
from unittest.mock import patch

import pytest

from leadscout import compliance
from leadscout.models import ComplianceVerdict, Lead, Research
from leadscout.sanctions import SanctionsScreen


@pytest.fixture(autouse=True)
def _offline():
    """The model always says "clear" here: these tests are about what the
    deterministic floors do when it is wrong."""
    def clear(system, user, schema, *, purpose="compliance"):
        return ComplianceVerdict(status="clear", flagged=False, matches=[],
                                 reasoning="different company, looks fine")
    with patch.object(compliance, "ask_model", clear), \
         patch.object(compliance, "screen_sanctions",
                      lambda c, h, run=None: SanctionsScreen(status="none", hits=[])):
        yield


def test_an_established_restricted_headquarters_blocks_without_the_model():
    r = compliance.screen(Lead("a", "a@s.example", "Snapp", "https://snapp.example.com"),
                          Research(headquarters_country="Iran", hq_source="gleif"))
    assert r.status == "blocked"
    assert "Iran" in r.reasoning and "restricted jurisdiction" in r.reasoning


def test_the_floor_names_the_surface_that_actually_matched():
    """A jurisdiction hit must not be explained as a competitor name match."""
    jurisdiction = compliance.screen(Lead("a", "a@s.example", "Snapp", "https://snapp.example.com"),
                                     Research(headquarters_country="Iran", hq_source="gleif"))
    assert "name match" not in jurisdiction.reasoning

    competitor = compliance.screen(Lead("a", "a@c.com", "CloudTrim Inc", "https://cloudtrim.com"),
                                   Research(headquarters_country="United States", hq_source="gleif"))
    assert "do-not-engage competitor" in competitor.reasoning


def test_an_exact_competitor_name_cannot_be_cleared_by_the_model():
    r = compliance.screen(Lead("a", "a@c.com", "CloudTrim Inc", "https://cloudtrim.com"),
                          Research(headquarters_country="United States", hq_source="gleif"))
    assert r.status == "review"  # a name match is a candidate, never a confirmed identity


def test_a_harmless_name_collision_is_not_escalated():
    """"Blue Cloud Bakery" shares a generic word with the list and nothing else."""
    r = compliance.screen(Lead("a", "a@b.example", "Blue Cloud Bakery", "https://bluecloud.example"),
                          Research(headquarters_country="Hungary", hq_source="gleif"))
    assert r.status == "clear"


def test_a_clearance_states_its_own_scope():
    r = compliance.screen(Lead("a", "a@b.example", "Blue Cloud Bakery", "https://bluecloud.example"),
                          Research(headquarters_country="Hungary", hq_source="gleif"))
    assert "not a KYC, AML or legal clearance" in r.reasoning


def test_a_deterministic_escalation_outranks_the_models_own_words():
    """The model's text may appear, but never as the reason for the outcome."""
    r = compliance.screen(Lead("a", "a@c.com", "CloudTrim Inc", "https://cloudtrim.com"),
                          Research(headquarters_country="United States", hq_source="gleif"))
    assert r.reasoning.startswith("A near-exact pre-screen match")
    assert "did not decide this outcome" in r.reasoning


_SEVERITY = {"clear": 0, "review": 1, "blocked": 2}


@pytest.mark.parametrize("model_says", ["clear", "review", "blocked"])
def test_a_stronger_identity_match_never_weakens_the_disposition(model_says):
    """Monotonicity. Raising ONLY the strength of a competitor identity match must not
    soften the outcome.

    This looked violated - an exact "CloudTrim Inc" returned REVIEW while the 94%
    "Cloud-Trim Ltd." was BLOCKED in the demo run - but those were two different
    experiments: the first held the model at "clear", the second is a live run where
    the model itself blocked. With the model held constant the ladder is monotone,
    which is what this asserts.
    """
    def verdict(system, user, schema, *, purpose="compliance"):
        return ComplianceVerdict(status=model_says, flagged=model_says != "clear",
                                 matches=[], reasoning="m")

    ladder = ["Bakery Kft", "Cloud-Trim Ltd.", "CloudTrim Inc"]  # weakest -> strongest
    with patch.object(compliance, "ask_model", verdict), \
         patch.object(compliance, "screen_sanctions",
                      lambda c, h, run=None: SanctionsScreen(status="none", hits=[])):
        severities = []
        for name in ladder:
            r = compliance.screen(Lead("a", "a@x.example", name, "https://x.example"),
                                  Research(headquarters_country="Hungary", hq_source="gleif"))
            severities.append(_SEVERITY[r.status])
    assert severities == sorted(severities), dict(zip(ladder, severities, strict=True))


def test_no_floor_can_downgrade_a_finding_already_made():
    """Aggregation is worst-case: once the model has blocked, nothing may soften it."""
    def blocked(system, user, schema, *, purpose="compliance"):
        return ComplianceVerdict(status="blocked", flagged=True, matches=[], reasoning="m")

    with patch.object(compliance, "ask_model", blocked), \
         patch.object(compliance, "screen_sanctions",
                      lambda c, h, run=None: SanctionsScreen(status="review_evidence", hits=[])):
        r = compliance.screen(
            Lead("a", "a@lidl.hu", "Lidl", "lidl.hu"),
            Research(headquarters_country="Germany", hq_source="llm",
                     domain_entity_name="Lidl Magyarország"))
    assert r.status == "blocked"


def test_the_floors_still_run_when_the_model_cannot_be_reached():
    """The deterministic layer is the safety layer, so it has to survive the model
    being down. Measured before the fix: `screen` raised LLMError before a single
    floor executed, so a lead with an Iran headquarters produced no decision at all -
    and in `cli.py batch` the exception aborted the remaining leads too."""
    def down(system, user, schema, *, purpose="compliance"):
        raise compliance.LLMError("all models in the fallback chain failed")

    with patch.object(compliance, "ask_model", down):
        blocked = compliance.screen(
            Lead("a", "a@s.example", "Snapp", "https://snapp.example.com"),
            Research(headquarters_country="Iran", hq_source="gleif"))
        assert blocked.status == "blocked"
        assert "Iran" in blocked.reasoning

        # Nothing to escalate, but an unassessed lead is never a cleared one.
        unassessed = compliance.screen(
            Lead("a", "a@b.example", "Blue Cloud Bakery", "https://bluecloud.example"),
            Research(headquarters_country="Hungary", hq_source="gleif"))
        assert unassessed.status == "review"
        assert "could not be reached" in unassessed.reasoning


def test_a_second_candidate_identity_is_not_screened_on_the_typed_leads_jurisdiction():
    """Entity-scope screening exists to stop one entity's facts deciding another's
    case. Passing `research.headquarters_country` into the candidate's own watchlist
    query reintroduced exactly that: a different legal entity filtered by the first
    one's country, which can suppress a real hit."""
    seen: list[tuple[str, object]] = []

    def record(name, country, run=None):
        seen.append((name, country))
        return SanctionsScreen(status="none", hits=[])

    with patch.object(compliance, "screen_sanctions", record):
        compliance.screen(
            Lead("a", "a@lidl.hu", "Lidl", "https://lidl.hu"),
            Research(headquarters_country="Germany", hq_source="llm",
                     domain_entity_name="Lidl Magyarorszag Bt"))

    assert ("Lidl", "Germany") in seen, seen          # the typed lead, on its own HQ
    alt = [c for n, c in seen if n != "Lidl"]
    assert alt == [None], f"candidate identity screened on an inherited jurisdiction: {alt}"
