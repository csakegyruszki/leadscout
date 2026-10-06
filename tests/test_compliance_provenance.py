"""Structured competitors (auto-derived abbreviations), exact-only abbreviation
matching, and provenance (original_value/origin) on every prescreen hit.
"""
from leadscout.compliance import _COMPETITORS, prescreen
from leadscout.models import Lead, Research


def test_competitor_abbreviations_are_derived_correctly():
    by_name = {c["name"]: c["abbreviation"] for c in _COMPETITORS}
    assert by_name["CloudTrim Inc"] == "CT"
    assert by_name["SpendWise Cloud"] == "SWC"
    assert by_name["RightSize Cloud Co"] == "RSC"


def test_abbreviation_exact_match_is_a_review_candidate_hit():
    hits = prescreen(Lead("a", "a@b.c", "SWC", "https://swc.example.com"), Research())
    assert any(h["origin"] == "prescreen:abbreviation" and h["term"] == "SpendWise Cloud" for h in hits)


def test_abbreviation_near_miss_is_not_a_hit():
    hits = prescreen(Lead("a", "a@b.c", "SWCX", "https://swcx.example.com"), Research())
    assert not any(h["origin"] == "prescreen:abbreviation" for h in hits)


def test_every_prescreen_hit_carries_provenance():
    """`original_value` is the raw value THAT hit came from, which is the company name
    for a name/abbreviation hit and the registrable domain for a `prescreen:domain`
    one (added 2026-09-20). Asserting "always the company name" would have hidden
    which of the two surfaces actually matched - the one thing this field is for."""
    lead = Lead("a", "a@b.c", "Cloud-Trim Ltd.", "https://cloud-trim.io")
    hits = prescreen(lead, Research())
    assert hits
    expected = {"prescreen:domain": "cloud-trim.io"}
    for h in hits:
        assert h["origin"].startswith("prescreen:")
        assert h["original_value"] == expected.get(h["origin"], "Cloud-Trim Ltd.")
    assert any(h["origin"] == "prescreen:domain" for h in hits)


def test_sanctioned_hq_hit_has_hq_origin():
    """An HQ the evidence actually supports carries the origin the auto-block floor reads."""
    hits = prescreen(Lead("a", "a@b.c", "Snapp", "https://snapp.example.com"),
                     Research(headquarters_country="Iran", hq_source="gleif"))
    hq_hits = [h for h in hits if h["origin"] == "prescreen:hq"]
    assert hq_hits and hq_hits[0]["term"] == "Iran"


def test_an_unsupported_llm_hq_gets_its_own_origin_and_cannot_reach_the_block_floor():
    """The model may read a country out of the sources; the sources have to contain it.
    A country that appears nowhere but in the model's own answer is `unresolved`, and
    safety net 0 (which keys on `prescreen:hq`) must not see it - otherwise one
    hallucinated country name blocks a company with no human in the loop."""
    hits = prescreen(Lead("a", "a@b.c", "Snapp", "https://snapp.example.com"),
                     Research(headquarters_country="Iran", hq_source="llm"))
    assert [h for h in hits if h["origin"] == "prescreen:hq"] == []
    unsupported = [h for h in hits if h["origin"] == "prescreen:hq_unsupported"]
    assert unsupported and unsupported[0]["term"] == "Iran"
    assert "no accepted source names that country" in unsupported[0]["reason"]


def test_an_llm_hq_the_raw_evidence_names_does_reach_the_block_floor():
    """The other half of the same rule: corroborated extraction is established enough."""
    hits = prescreen(Lead("a", "a@b.c", "Snapp", "https://snapp.example.com"),
                     Research(headquarters_country="Iran", hq_source="llm",
                              website_text="About us: Snapp is headquartered in Tehran, Iran."))
    assert [h for h in hits if h["origin"] == "prescreen:hq"], hits


def test_sanctioned_tld_hit_has_tld_origin():
    hits = prescreen(Lead("a", "a@b.c", "Snapp", "https://snapp.ir"), Research())
    tld_hits = [h for h in hits if h["origin"] == "prescreen:tld"]
    assert tld_hits and tld_hits[0]["term"] == "Iran"
