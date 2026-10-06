"""A WEAK GLEIF candidate is an identity hint, never this lead's headquarters.

Adversarial battery 2026-09-20, case d: a lead submitted as company "Notion" with
website `notiontechnologies.com` (a Mumbai agency) produced a WEAK GLEIF candidate -
"Notion OU", Estonia - accepted on name similarity alone, because an unknown HQ plus a
generic gTLD leaves no country signal to confirm or conflict with. That is the
documented, intended acceptance behaviour. What was NOT intended: the evidence snippet
read exactly like a confirmed one, so the research LLM promoted the candidate record's
country to `headquarters_country=Estonia` and the summary asserted "Notion OU is a
company registered in Estonia" - an unrelated company, stated as fact.
"""
from leadscout.models import Evidence
from leadscout.providers import gleif


def _ev(snippet: str, strength: str) -> Evidence:
    return Evidence(id="ev-001", source_type="gleif", url="https://search.gleif.org/#/record/X",
                    observed_at="2026-09-20T00:00:00+00:00", content_sha256="0" * 64,
                    strength=strength, snippet=snippet, snapshot_path="", family="corporate_identity")


def test_weak_evidence_yields_no_headquarters_country():
    weak = _ev("possible name match only, NOT confirmed to be this lead: Notion OU (984500X), "
               "ACTIVE; that record's legal-address country is EE, its registered_jurisdiction=EE",
               "WEAK")
    assert gleif.legal_country_name_from_evidence(weak) is None


def test_a_corroborated_record_still_yields_its_country():
    strong = _ev("ZAPIER, INC. (549300X), US, ACTIVE, registered_jurisdiction=US-DE", "STRONG")
    assert gleif.legal_country_name_from_evidence(strong) == "United States"
    medium = _ev("MASTERPLAST NYRT (529900X), HU, ACTIVE, registered_jurisdiction=HU", "MEDIUM")
    assert gleif.legal_country_name_from_evidence(medium) == "Hungary"


def test_weak_snippet_says_the_country_belongs_to_the_candidate_not_the_lead(monkeypatch):
    """The wording is the fix: an LLM reading this block must not be able to take the
    country as the lead's. Asserted on the text, because the text is what it reads."""
    payload = {"data": [{"attributes": {
        "lei": "984500PD03ECHB99A61",
        "entity": {"legalName": {"name": "Notion OU"}, "status": "ACTIVE",
                   "legalAddress": {"country": "EE"}, "jurisdiction": "EE"}}}]}

    class _Resp:
        status_code = 200
        content = b'{"data": []}'

        @staticmethod
        def json():
            return payload

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(gleif.httpx, "Client", _Client)
    # Unknown HQ + generic gTLD: no country factor can be satisfied -> WEAK.
    result = gleif.run("Notion", None, "https://notiontechnologies.com")
    assert result.evidence and result.evidence[0].strength == "WEAK"
    snippet = result.evidence[0].snippet
    assert "NOT confirmed to be this lead" in snippet
    assert "that record's legal-address country is EE" in snippet
    # The old format - country as a bare field right after the LEI - is what misled the
    # model; it must not reappear.
    assert "), EE, ACTIVE" not in snippet
