import dataclasses
import json

import httpx

from leadscout.providers import headcount


class _Client:
    def __init__(self, payload, status=200):
        self.payload, self.status, self.calls = payload, status, []

    def get(self, url, params=None, headers=None):
        self.calls.append((url, params, headers))
        return httpx.Response(self.status, json=self.payload, request=httpx.Request("GET", url))


def _kg(homepage, score, n, name="Acme"):
    return {"data": [{"score": score, "entity": {"name": name, "homepageUri": homepage, "nbEmployees": n,
                                                 "nbEmployeesMin": 1000, "nbEmployeesMax": 5000,
                                                 "diffbotUri": "http://diffbot.com/entity/X1"}}]}


def test_diffbot_accepts_only_own_domain_and_sufficient_score(monkeypatch):
    monkeypatch.setattr(headcount, "settings", dataclasses.replace(headcount.settings, diffbot_token="tok"))
    cand, _ = headcount.diffbot("Acme", "acme.com", _Client(_kg("www.acme.com", 0.9, 1500)))
    assert cand is not None and cand.employees == 1500 and cand.source == "diffbot"
    wrong_domain, note = headcount.diffbot("Acme", "acme.com", _Client(_kg("acme.io", 0.99, 1500)))
    assert wrong_domain is None and "rejected" in note
    weak, _ = headcount.diffbot("Acme", "acme.com", _Client(_kg("acme.com", 0.5, 1500)))
    assert weak is None


def test_diffbot_token_never_reaches_the_recorded_snapshot(monkeypatch):
    monkeypatch.setattr(headcount, "settings", dataclasses.replace(headcount.settings, diffbot_token="SECRET-TOK-1"))
    payload = _kg("acme.com", 0.9, 1500)
    payload["request"] = {"echo": "token=SECRET-TOK-1"}
    recorded = {}

    class _Run:
        def next_evidence_id(self):
            return "ev-001"

        def record(self, ev, raw, stage=""):
            recorded["raw"] = raw

    monkeypatch.setattr(headcount, "fallback_evidence_id", lambda p, r: "ev-001")
    monkeypatch.setattr(headcount, "snapshot_path", lambda r, e: "x")
    headcount.diffbot("Acme", "acme.com", _Client(payload), run=_Run())
    assert b"SECRET-TOK-1" not in recorded["raw"]


def _brave(texts):
    return {"web": {"results": [{"url": f"https://ex{i}.com", "title": "Acme", "description": d}
                                for i, d in enumerate(texts)]}}


def test_web_search_requires_a_verbatim_quote(monkeypatch):
    monkeypatch.setattr(headcount, "settings", dataclasses.replace(headcount.settings, brave_api_key="k"))
    client = _Client(_brave(["Acme has about 2,300 employees worldwide."]))
    good = lambda *a, **k: {"employees": 2300, "quote": "about 2,300 employees", "result_index": 0}  # noqa: E731
    cand, _ = headcount.web_search("Acme", "acme.com", client, good)
    assert cand is not None and cand.employees == 2300
    invented = lambda *a, **k: {"employees": 9000, "quote": "9,000 employees", "result_index": 0}  # noqa: E731
    cand, note = headcount.web_search("Acme", "acme.com", client, invented)
    assert cand is None and "verbatim" in note


def test_only_the_identity_checked_figure_is_scored():
    kg = headcount.HeadcountCandidate("diffbot", 184, "ev-1", "kg")
    web = headcount.HeadcountCandidate("web_search", 50, "ev-2", "web")
    assert headcount.choose([kg, web]) == (kg, web)
    assert headcount.choose([web]) == (None, web)  # a web figure alone is a hint, never a score
    assert headcount.choose([]) == (None, None)


def test_no_keys_means_skipped_not_guessed(monkeypatch):
    monkeypatch.setattr(headcount, "settings", dataclasses.replace(headcount.settings, diffbot_token=""))
    monkeypatch.setattr(headcount, "settings", dataclasses.replace(headcount.settings, brave_api_key=""))
    result, cands = headcount.run("Acme", "acme.com", lambda *a, **k: json.loads("{}"))
    assert result.status == "skipped" and cands == []


def test_diffbot_match_is_robust_to_nan_score_and_subdomains(monkeypatch):
    monkeypatch.setattr(headcount, "settings", dataclasses.replace(headcount.settings, diffbot_token="t"))
    nan, note = headcount.diffbot("Acme", "acme.com", _Client(_kg("acme.com", "nan", 1500)))
    assert nan is None and "rejected" in note  # `score < X` would have accepted NaN
    sub, note = headcount.diffbot("Acme", "acme.com", _Client(_kg("eu.acme.com", 0.99, 1500)))
    assert sub is None  # a subdomain is not the company's own homepage
    dotted, _ = headcount.diffbot("Acme", "acme.com", _Client(_kg("https://WWW.acme.com.:443/", 0.9, 1500)))
    assert dotted is not None and dotted.employees == 1500


def test_a_token_echoed_in_the_entity_url_never_reaches_the_evidence(monkeypatch):
    monkeypatch.setattr(headcount, "settings", dataclasses.replace(headcount.settings, diffbot_token="SECRET-TOK-1"))
    payload = _kg("acme.com", 0.9, 1500)
    payload["data"][0]["entity"]["diffbotUri"] = "https://kg.example/e?token=SECRET-TOK-1"
    monkeypatch.setattr(headcount, "fallback_evidence_id", lambda p, r: "ev-001")
    monkeypatch.setattr(headcount, "snapshot_path", lambda r, e: "x")
    cand, _ = headcount.diffbot("Acme", "acme.com", _Client(payload))
    assert "SECRET-TOK-1" not in cand.evidence.url and "<redacted>" in cand.evidence.url


# --- Headcount authority ladder (2026-09-20 audit) --------------------------------
# applicant's own size band > identity-checked (exact-domain Diffbot) > model inference.
#
# The guard on the lookup used to be `not r.estimated_employees and not
# lead.company_size_band`, so a figure the model had inferred from the evidence text
# suppressed the identity-checked lookup completely - not outranked, never queried. The
# research system prompt explicitly invites that inference ("infer from ... office
# counts, or well-known scale"), so it fired routinely.

def _research_lead_with(monkeypatch, *, llm_employees, size_band, diffbot_employees):
    """Run research_lead offline with a stubbed model and a stubbed headcount provider."""
    from leadscout import research
    from leadscout.models import Lead, ProviderResult, ResearchFacts
    from leadscout.providers import headcount as hc
    from leadscout.providers import hunter as hunter_mod

    monkeypatch.setattr(research, "fetch_website", lambda url, max_chars=6000: ("about us", True, "<html/>"))
    monkeypatch.setattr(research, "fetch_wikipedia", lambda company: ("", ""))
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website", lambda company, website: (None, "NOT_FOUND"))
    monkeypatch.setattr(research.gleif_provider, "run",
                        lambda company, hq_country, website, run=None: ProviderResult(provider_name="gleif"))
    for mod in ("ats_provider", "github_provider", "footprint_provider",
                "trust_pages_provider", "vendor_provider"):
        provider = getattr(research, mod)
        monkeypatch.setattr(provider, "run", lambda *a, **k: ProviderResult(provider_name="stub", status="skipped"))
    _hunter_facts = hunter_mod.HunterFacts(contact_quality="unknown", contact_quality_note="")
    monkeypatch.setattr(research.hunter_provider, "run", lambda *a, **k: (
        ProviderResult(provider_name="hunter", status="skipped"), _hunter_facts))
    monkeypatch.setattr(research, "ask_model",
                        lambda system, user, schema, purpose="": ResearchFacts(
                            summary="s", industry="i", headquarters_country="Hungary",
                            estimated_employees=llm_employees))

    calls: list[str] = []

    def fake_hc_run(company, domain, ask_json, run=None):
        calls.append(company)
        cands = []
        if diffbot_employees is not None:
            cands.append(hc.HeadcountCandidate(source="diffbot", employees=diffbot_employees,
                                               evidence_id="ev-777", detail="Diffbot exact-domain match"))
        return ProviderResult(provider_name="headcount", status="ok"), cands

    monkeypatch.setattr(research.headcount_provider, "run", fake_hc_run)
    lead = Lead("a", "a@acme.com", "Acme", "https://acme.com", company_size_band=size_band or "")
    return research.research_lead(lead), calls


def test_an_identity_checked_figure_overrides_the_models_inference(monkeypatch):
    r, calls = _research_lead_with(monkeypatch, llm_employees=5000, size_band=None, diffbot_employees=1000)
    assert calls, "the identity-checked lookup was never queried"
    assert r.estimated_employees == 1000
    assert "Diffbot exact-domain match" in r.employees_source
    assert any("overrides the figure inferred" in u for u in r.uncertainties)


def test_the_models_inference_survives_when_no_identity_checked_figure_exists(monkeypatch):
    """Diffbot rejected on identity mismatch (no candidate): the inference is all there
    is, and it stays - labelled as the inference it is, not promoted."""
    r, calls = _research_lead_with(monkeypatch, llm_employees=5000, size_band=None, diffbot_employees=None)
    assert calls
    assert r.estimated_employees == 5000
    assert r.employees_source == "website/LLM research"


def test_the_applicants_own_size_band_short_circuits_the_lookup(monkeypatch):
    """First-party data about the entity that actually applied wins, and the third-party
    lookup is not even made - a group-vs-subsidiary mismatch is how it is usually wrong."""
    r, calls = _research_lead_with(monkeypatch, llm_employees=10000, size_band="201-1000",
                                   diffbot_employees=5000)
    assert calls == [], "the lookup ran even though the applicant stated its own size"
    assert r.estimated_employees == 10000  # the inference stays; the BAND drives the fit score


def test_the_outcome_does_not_depend_on_which_figure_arrived_first(monkeypatch):
    """Same three inputs, identity-checked figure resolved either before or after the
    model's: the result has to be the same both ways."""
    first, _ = _research_lead_with(monkeypatch, llm_employees=5000, size_band=None, diffbot_employees=1000)
    second, _ = _research_lead_with(monkeypatch, llm_employees=None, size_band=None, diffbot_employees=1000)
    assert first.estimated_employees == second.estimated_employees == 1000
    assert first.employees_source == second.employees_source
