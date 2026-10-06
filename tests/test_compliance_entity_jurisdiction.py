"""Identity uncertainty alone must not cause REVIEW.

`review` means a human has to decide something. An unresolved "which legal entity
applied" question is DATA uncertainty, and only becomes a decision when a resolved
candidate interpretation could change the competitor, sanctions-entity or jurisdiction
answer. So each resolved candidate identity is screened in its own right and the worst
outcome wins - blocked > review > clear - which makes new evidence able to make the
verdict stricter, never softer.

Lidl (2026-09-20) is the regression this encodes. The lead applied as "Lidl" from
`lidl.hu`; domain enrichment associates that site with "Lidl Magyarorszag"; company
evidence describes the German group. Neither candidate is a competitor or a sanctions
match, so whichever the lead meant, the answer is the same: CLEAR, with the ambiguity
recorded. The earlier version forced REVIEW on the mismatch alone and put a lead in
the human queue that a human could only have cleared.
"""
from leadscout import compliance
from leadscout.models import ComplianceVerdict, Lead, Research
from leadscout.sanctions import SanctionsScreen


def _clear_verdict(system, user, schema, *, purpose="compliance"):
    return ComplianceVerdict(status="clear", flagged=False, matches=[], reasoning="German retailer")


def _offline(monkeypatch, status: str = "none"):
    """No network: the LLM says clear and every sanctions screen returns `status`."""
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions",
                        lambda company, hq, run=None: SanctionsScreen(status=status, hits=[]))


def _lidl() -> tuple[Lead, Research]:
    return (Lead("Nemeth Zsofia", "nemeth.zsofia@lidl.hu", "Lidl", "lidl.hu"),
            Research(headquarters_country="Germany", hq_source="llm",
                     domain_entity_name="Lidl Magyarország",
                     # The evidence text has to name the country the model reported, or the
                     # jurisdiction is unresolved and THAT decides the outcome - these cases
                     # are about identity ambiguity, so the HQ has to be out of the way. The
                     # model's own summary deliberately does not count (it would be circular).
                     website_text="Lidl Stiftung & Co. KG is headquartered in Germany.",
                     summary="Lidl is a German international discount supermarket chain."))


def test_the_candidate_is_recorded_as_an_association_not_as_ownership():
    """Enrichment associates a name with a domain. It does not establish that the
    applicant IS that legal entity, and the wording may not claim it does."""
    hits = [h for h in compliance.prescreen(*_lidl())
            if h.get("origin") == "prescreen:entity_scope"]
    assert len(hits) == 1 and hits[0]["term"] == "Lidl Magyarország"
    assert "associates" in hits[0]["reason"]
    assert "belongs to" not in hits[0]["reason"]
    assert "not an established legal entity" in hits[0]["reason"]


def test_identity_ambiguity_alone_does_not_force_review(monkeypatch):
    _offline(monkeypatch)
    result = compliance.screen(*_lidl())
    assert result.status == "clear", result.reasoning
    assert "does not affect the outcome" in result.reasoning
    assert "Lidl Magyarország" in result.reasoning


def test_a_sanctions_record_on_the_second_identity_still_escalates(monkeypatch):
    """The ambiguity is screened, not waved through: if the candidate identity itself
    returns a watchlist record, the worst outcome wins."""
    _offline(monkeypatch, status="review_evidence")
    result = compliance.screen(*_lidl())
    assert result.status == "review"
    assert "second candidate identity" in result.reasoning


def test_a_blocking_record_on_the_second_identity_blocks(monkeypatch):
    _offline(monkeypatch, status="blocked_evidence")
    result = compliance.screen(*_lidl())
    assert result.status == "blocked"


def test_a_competitor_hiding_in_the_second_identity_is_caught(monkeypatch):
    """The typed name is unremarkable; the identity behind the domain is the
    competitor. Screening only what was typed would clear it."""
    _offline(monkeypatch)
    result = compliance.screen(
        Lead("A", "a@unimedia.tech", "Unimedia", "https://unimedia.tech"),
        Research(headquarters_country="Spain", hq_source="llm",
                 domain_entity_name="Unimedia CloudTrim"))
    assert result.status in ("review", "blocked")
    assert any(m.get("origin") == "prescreen:entity_scope_competitor" for m in result.matches)


def test_a_registry_established_headquarters_raises_no_candidate(monkeypatch):
    _offline(monkeypatch)
    result = compliance.screen(
        Lead("A", "a@masterplast.hu", "Masterplast", "https://masterplast.hu"),
        Research(headquarters_country="Hungary", hq_source="gleif",
                 domain_entity_name="Masterplast Nyrt"))
    assert result.status == "clear"
    assert "candidate identity" not in result.reasoning


def test_the_same_identity_spelled_differently_is_not_a_candidate():
    for typed, resolved in (("Zapier", "Zapier"), ("Zapier", "Zapier, Inc.")):
        hits = compliance.prescreen(
            Lead("A", "a@zapier.com", typed, "https://zapier.com"),
            Research(headquarters_country="United States", hq_source="llm",
                     domain_entity_name=resolved))
        assert not [h for h in hits if h.get("origin") == "prescreen:entity_scope"], resolved


def test_a_clearance_states_how_firmly_the_jurisdiction_is_known(monkeypatch):
    """With no floor firing, the model's own sentence is what the rep reads - and left
    alone it writes "the HQ is confirmed as Germany" about a country it inferred."""
    _offline(monkeypatch)
    inferred = compliance.screen(*_lidl())
    assert inferred.status == "clear"
    assert "inferred from research evidence, not established from a registry record" in inferred.reasoning

    from_registry = compliance.screen(
        Lead("A", "a@masterplast.hu", "Masterplast", "https://masterplast.hu"),
        Research(headquarters_country="Hungary", hq_source="gleif"))
    assert from_registry.status == "clear"
    assert "inferred from research evidence" not in from_registry.reasoning


def test_the_enrichment_identity_never_becomes_the_leads_canonical_identity(monkeypatch):
    """Entity contamination through another door: a candidate identity may inform the
    screen, and must never quietly rewrite who the lead IS. If "Lidl Magyarország"
    could become company/HQ/headcount, the Notion and Győr failures come straight
    back."""
    _offline(monkeypatch)
    lead, research = _lidl()
    research.estimated_employees = None
    before = (lead.company, research.headquarters_country, research.hq_source,
              research.estimated_employees, research.industry)

    compliance.screen(lead, research)

    assert (lead.company, research.headquarters_country, research.hq_source,
            research.estimated_employees, research.industry) == before
    assert lead.company == "Lidl"  # not "Lidl Magyarország"
    assert research.headquarters_country == "Germany"  # not inferred from the candidate


# --- Every material candidate is actually screened ---------------------------------
#
# Candidate creation used to require the typed name to be a SUBSTRING of the resolved
# one, so it only fired when the candidate was the more specific spelling of the same
# name. Measured: company "Acme" on a domain resolving to "Rosneft" produced no
# candidate at all, and the sanctions stub - which blocked Rosneft - was called exactly
# once, with "Acme". The materially different identity, which is the case that most
# needs screening, was the one case excluded. A second gate suppressed candidates
# whenever the TYPED entity had a registry HQ, which is a fact about a different
# question entirely.

def _recording_screen(monkeypatch, blocked=(), review=()):
    """Screening stub that records every name it was asked about."""
    seen: list[str] = []

    def screen(company, hq, run=None):
        seen.append(company)
        if company in blocked:
            return SanctionsScreen(status="blocked_evidence", hits=[{
                "caption": company, "score": 1.0, "topics": ["sanction"], "match": True,
                "schema": "Company", "datasets": [], "url": "https://x",
                "origin": "opensanctions:match"}])
        if company in review:
            return SanctionsScreen(status="review_evidence", hits=[])
        return SanctionsScreen(status="none", hits=[])

    monkeypatch.setattr(compliance, "ask_model",
                        lambda system, user, schema, purpose="compliance": ComplianceVerdict(
                            status="clear", flagged=False, matches=[], reasoning="looks fine"))
    monkeypatch.setattr(compliance, "screen_sanctions", screen)
    return seen


def _acme(entity: str) -> tuple[Lead, Research]:
    return (Lead("A", "a@acme.example", "Acme", "https://acme.example"),
            Research(headquarters_country="Germany", hq_source="gleif",
                     domain_entity_name=entity))


def test_a_materially_different_domain_identity_is_screened(monkeypatch):
    seen = _recording_screen(monkeypatch, blocked=("Rosneft",))
    result = compliance.screen(*_acme("Rosneft"))
    assert "Rosneft" in seen, f"the candidate identity was never screened: {seen}"
    assert result.status == "blocked", result.reasoning


def test_a_materially_different_competitor_identity_is_screened(monkeypatch):
    seen = _recording_screen(monkeypatch)
    result = compliance.screen(*_acme("CloudTrim Inc"))
    assert "CloudTrim Inc" in seen, f"the candidate identity was never screened: {seen}"
    assert result.status in ("review", "blocked"), result.reasoning


def test_a_clean_alternate_identity_is_screened_and_does_not_escalate(monkeypatch):
    """The control: screening every candidate must not turn a clean one into a flag,
    or the fix would trade a miss for noise."""
    seen = _recording_screen(monkeypatch)
    result = compliance.screen(*_acme("Acme Manufacturing GmbH"))
    assert "Acme Manufacturing GmbH" in seen, seen
    assert result.status == "clear", result.reasoning


def test_the_candidate_is_screened_on_its_own_jurisdiction_not_the_leads(monkeypatch):
    """The typed lead's country must not filter the candidate's watchlist query - the
    candidate is a different legal entity, and inheriting the first one's country can
    suppress a real hit."""
    asked: list[tuple[str, object]] = []

    def screen(company, hq, run=None):
        asked.append((company, hq))
        return SanctionsScreen(status="none", hits=[])

    monkeypatch.setattr(compliance, "ask_model",
                        lambda system, user, schema, purpose="compliance": ComplianceVerdict(
                            status="clear", flagged=False, matches=[], reasoning="fine"))
    monkeypatch.setattr(compliance, "screen_sanctions", screen)
    compliance.screen(*_acme("Rosneft"))
    assert ("Rosneft", None) in asked, asked
