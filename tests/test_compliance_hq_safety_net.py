"""Part 2b (2026-09-19): a DETERMINISTIC compliance safety net, not a prompt fix -
the adversarial eval showed a prompt-only instruction can still be talked around by
evidence-embedded content (adv-03: a Tehran legal notice conflicting with a Dubai
marketing claim still came back "clear" after tightening compliance.py's system
prompt). This reads the raw evidence text directly, never the LLM's own summary of
it, so it cannot be argued around the way a prompt instruction can.

Two rules, one code path in compliance.screen (see `_hq_established`/
`_mentions_restricted_marker`):
1. "clear" requires a SCREENABLE headquarters country (see `jurisdiction_fact`):
   known, from a source with enough authority, and - where that source is the
   model's own reading - actually named in the evidence. The earlier version of
   this rule asked only whether the string was non-empty, which let a ccTLD guess
   and an unsupported LLM claim clear a lead that the block side would not act on.
2. When HQ is not established AND a restricted-jurisdiction name or one of its
   capital/major-city markers (config/restricted_jurisdictions.yaml's `markers`)
   appears in the raw evidence text, that is flagged with a specific reason/origin
   ("safety_net:hq_unknown") rather than the generic HQ-unknown one.

All offline: ask_model and screen_sanctions monkeypatched, no real LLM/network calls.
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from leadscout import compliance
from leadscout.models import ComplianceVerdict, Lead, Research
from leadscout.sanctions import SanctionsScreen


def _clear_verdict(system, user, schema, *, purpose="compliance"):
    return ComplianceVerdict(status="clear", flagged=False, matches=[], reasoning="looks fine")


def _no_sanctions_api_evidence(company, hq, run=None):
    return SanctionsScreen(status="none")


def test_unknown_hq_with_restricted_marker_in_evidence_is_never_clear(monkeypatch):
    """adv-03's shape: a Dubai marketing claim conflicts with a Tehran legal
    footer; the LLM still says "clear" (it can be talked into that by the page's
    own instructions) - the deterministic safety net must catch it anyway."""
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions", _no_sanctions_api_evidence)
    research = Research(
        headquarters_country="unknown",
        website_text="Our headquarters are in Dubai. Legal notice: registered office, Tehran, Iran.",
        summary="A company with conflicting HQ claims.",
    )
    result = compliance.screen(Lead("a", "a@b.c", "Saffron Compute Works", "https://saffron.example"), research)
    assert result.status != "clear"
    assert any(m.get("origin") == "safety_net:hq_unknown" for m in result.matches)
    assert any(m.get("term") in ("Tehran", "Iran") for m in result.matches)


def test_unknown_hq_without_any_marker_is_still_never_clear(monkeypatch):
    """Rule 2 alone: an unresolved HQ is never sufficient to clear a lead, even
    with no restricted-jurisdiction marker anywhere in the evidence."""
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions", _no_sanctions_api_evidence)
    research = Research(headquarters_country="", website_text="A generic company page with no HQ statement.",
                         summary="Unclear where this company is based.")
    result = compliance.screen(Lead("a", "a@b.c", "Generic Co", "https://generic.example"), research)
    assert result.status == "review"
    assert any(m.get("origin") == "safety_net:hq_unknown" for m in result.matches)


def test_established_hq_with_a_restricted_city_mention_elsewhere_is_not_forced(monkeypatch):
    """The safety net only fires when HQ is NOT established - an affirmatively
    resolved, unsanctioned HQ with an incidental city mention elsewhere (e.g. a
    customer case study) must not be forced into "review" by this rule."""
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions", _no_sanctions_api_evidence)
    research = Research(headquarters_country="Canada", hq_source="gleif",
                         website_text="We serve customers worldwide, including a case study in Moscow, Idaho.",
                         summary="A Canadian company with a US customer.")
    result = compliance.screen(Lead("a", "a@b.c", "Northern Co", "https://northern.example"), research)
    assert result.status == "clear"


def test_established_unsanctioned_hq_stays_clear(monkeypatch):
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions", _no_sanctions_api_evidence)
    research = Research(headquarters_country="Germany", hq_source="gleif",
                        website_text="Based in Berlin.", summary="A German company.")
    result = compliance.screen(Lead("a", "a@b.c", "Berlin Co", "https://berlin.example"), research)
    assert result.status == "clear"


# --- Hypothesis invariant ----------------------------------------------------------

_MARKERS = ("Tehran", "Moscow", "Pyongyang", "Damascus", "Havana", "Minsk", "Caracas", "Naypyidaw",
            "Iran", "Russia", "North Korea", "Syria", "Cuba", "Belarus", "Venezuela", "Myanmar")


@given(
    marker=st.sampled_from(_MARKERS),
    hq_value=st.sampled_from(("", "unknown", "Unknown", "UNKNOWN", "  ")),
    prefix=st.text(alphabet="abcdefghijklmnop 0123456789", max_size=40),
    suffix=st.text(alphabet="abcdefghijklmnop 0123456789", max_size=40),
)
@hyp_settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hq_unknown_plus_restricted_marker_never_clears(monkeypatch, marker, hq_value, prefix, suffix):
    """Property: for ANY unresolved-HQ spelling and ANY surrounding text, if a
    restricted-jurisdiction marker appears anywhere in the evidence text, the
    compliance status can never be "clear" - independent of what the (here always
    "clear") LLM verdict said."""
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions", _no_sanctions_api_evidence)
    research = Research(headquarters_country=hq_value, website_text=f"{prefix} {marker} {suffix}", summary="")
    result = compliance.screen(Lead("a", "a@b.c", "Property Test Co", "https://property-test.example"), research)
    assert result.status != "clear"


# --- One screenability rule, applied to both directions ----------------------------
#
# The clear side used to ask only whether the country string was non-empty, while the
# block side had an authority rule of its own. So the same fact that was too weak to
# block was strong enough to clear: a `.hu` ccTLD guess and an LLM-asserted "Germany"
# with no supporting evidence each produced a final status of `clear`.

SCREENABILITY = [
    ("gleif", "Germany", "", "", True),
    ("wikidata", "Germany", "", "", True),
    ("llm", "Germany", "Our head office in Germany.", "", True),     # corroborated
    ("llm", "Germany", "We bake bread.", "", False),                 # F-02
    ("website_tld", "Hungary", "", "low", False),                    # F-01
    ("website_tld", "Hungary", "", "", False),                       # a hint is a hint
    ("", "Germany", "", "", False),                                  # country from nowhere
    ("gleif", "", "", "", False),
    ("gleif", "unknown", "", "", False),
]


@pytest.mark.parametrize("source,country,text,conf,screenable", SCREENABILITY)
def test_screenability_is_decided_by_authority_and_support(source, country, text, conf, screenable):
    fact = compliance.jurisdiction_fact(
        Research(headquarters_country=country, hq_source=source,
                 hq_confidence=conf, website_text=text))
    assert fact.screenable is screenable, fact.why


@pytest.mark.parametrize("source,country,text,conf,screenable", SCREENABILITY)
def test_the_same_rule_governs_clearing_and_blocking(source, country, text, conf, screenable,
                                                      monkeypatch):
    """A jurisdiction fact too weak to close a decision one way must be too weak to
    close it the other way. Asserted as one test over one table so the two sides cannot
    drift apart again."""
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions", _no_sanctions_api_evidence)
    research = Research(headquarters_country=country, hq_source=source,
                        hq_confidence=conf, website_text=text)
    clear_side = compliance.screen(Lead("a", "a@b.c", "Some Co", "https://some.example"), research)
    assert (clear_side.status == "clear") is screenable, clear_side.reasoning

    # Same fact, restricted country: it may only reach a deterministic block on the
    # same authority that would have let it clear.
    restricted = Research(headquarters_country="Iran" if country else "", hq_source=source,
                          hq_confidence=conf, website_text=text.replace("Germany", "Iran"))
    block_side = compliance.screen(Lead("a", "a@b.c", "Some Co", "https://some.example"), restricted)
    if screenable and country.lower() not in ("", "unknown"):
        assert block_side.status == "blocked", block_side.reasoning
    else:
        assert block_side.status != "clear", block_side.reasoning


def test_corroboration_reads_captured_sources_not_the_models_own_output():
    """What makes an inferred jurisdiction screenable must be a captured source, never
    the model's own words - otherwise the check is circular and an unsupported claim
    corroborates itself.

    The fields read are `website_text` (assigned from the fetched page, research.py:535)
    and `wikipedia_summary` (from the Wikipedia provider, research.py:185), both of which
    are also recorded as Evidence (research.py:373). The field NOT read is `summary`,
    which is the model's output (research.py:385, 563).
    """
    from_model_only = Research(
        headquarters_country="Germany", hq_source="llm",
        summary="Acme is headquartered in Germany.",       # the model's own sentence
        website_text="We bake bread.", wikipedia_summary="")
    fact = compliance.jurisdiction_fact(from_model_only)
    assert fact.screenable is False, \
        "the model's own summary corroborated the model's own claim"

    from_a_captured_page = Research(
        headquarters_country="Germany", hq_source="llm",
        summary="A bakery.", website_text="Acme GmbH, Germany.", wikipedia_summary="")
    assert compliance.jurisdiction_fact(from_a_captured_page).screenable is True

    from_wikipedia = Research(
        headquarters_country="Germany", hq_source="llm",
        summary="A bakery.", website_text="We bake bread.",
        wikipedia_summary="Acme is a company based in Germany.")
    assert compliance.jurisdiction_fact(from_wikipedia).screenable is True
