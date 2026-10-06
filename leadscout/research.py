"""Company research: submitted website (+ a bounded same-origin crawl) + Wikidata
(website-matched) + Wikipedia (fallback text), then an LLM turns it into facts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
import trafilatura
from rapidfuzz import fuzz

from . import extract, infra
from .cloud import assess_cloud_usage
from .config import settings
from .llm import ask_json, ask_model
from .models import Evidence, Lead, ProviderResult, Research, ResearchFacts
from .profile_config import ResearchConfig, research_config
from .prompt_safety import INJECTION_WARNING, wrap_evidence
from .provenance import ProvenanceRun
from .providers import ats as ats_provider
from .providers import footprint as footprint_provider
from .providers import github as github_provider
from .providers import gleif as gleif_provider
from .providers import headcount as headcount_provider
from .providers import hunter as hunter_provider
from .providers import trust_pages as trust_pages_provider
from .providers import vendor as vendor_provider
from .providers import website as website_provider
from .providers import wikidata as wikidata_provider
from .util import USER_AGENT, UnsafeTargetError, registrable_domain, safe_get

# --- Design notes -----------------------------------------------------------------
# Why these sources (and not a paid enrichment API):
# - The website is the one source the lead itself vouched for; its own "about" text
#   is the best cheap signal of what the company does and how big it is. A few more
#   same-origin pages (careers/engineering especially) often say more about infra
#   intensity than the landing page does - see providers/website.py.
# - Wikidata's structured properties (HQ, employees, industry - see providers/
#   wikidata.py), accepted only when the entity's own official-website property
#   matches the domain the lead submitted, is a much harder identity claim to fake
#   by accident than fuzzy-matching a Wikipedia article title (which once returned
#   *Five Guys* for a bakery - see README "How I used AI tooling").
# - Wikipedia's plain-text summary stays as a free, key-less fallback text source.
# - All of it is free, reproducible, and every fact is snapshotted as `Evidence` and
#   logged to the provtrail ledger, so it can be traced back to the exact bytes.

logger = logging.getLogger("leadscout")

UA = USER_AGENT
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


def _normalise_url(url: str) -> str:
    """Add the scheme a form submission usually omits, and only that.

    A bare "zapier.com" has to become "https://zapier.com". Prepending unconditionally
    also turned "file:///C:/..." into "https://file:///C:/...", a nonsense host that
    then failed with ConnectError - so a non-http scheme was refused by accident rather
    than on purpose, and the report called it "unreachable" when it should never have
    been attempted. A URL that already carries a scheme is left alone, and
    `assert_public_url` refuses it if that scheme is not http(s).
    """
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url):
        url = "https://" + url
    return url


def fetch_website(url: str, max_chars: int = 6000) -> tuple[str, bool, str]:
    """Return (main text, ok, raw html). Never raises: an unreachable site is
    itself a signal. The raw HTML is kept (not just the extracted text) so the
    bounded crawl can find same-origin links on the landing page."""
    url = _normalise_url(url)
    try:
        with httpx.Client(timeout=settings.http_timeout, headers={"User-Agent": UA}) as c:
            r = safe_get(c, url)
        if r.status_code >= 400:
            return f"[HTTP {r.status_code}]", False, ""
        text = trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        rendered = website_provider.maybe_render_with_crawl4ai(url, r.text, text)
        if rendered:
            text = rendered
        return text[:max_chars], bool(text), r.text
    except UnsafeTargetError as e:
        # Refused, not unreachable - a distinction the rep-facing text has to keep. The
        # submitted URL resolved to a non-public address, so nothing was fetched and
        # nothing from it can reach the prompt, the tracker row or the notification.
        logger.warning("refusing to fetch submitted URL: %s", e)
        return "[refused: not a public address]", False, ""
    except Exception as e:  # noqa: BLE001 - we want the reason in the report, not a crash
        return f"[unreachable: {type(e).__name__}]", False, ""


def _apex_resolves(domain: str) -> bool:
    """True unless Cloudflare DoH answers the apex A-record query with Status 3
    (NXDOMAIN) - the domain genuinely does not exist. Fix #2 (v0.2 PART B):
    replaces the previous `website_text.startswith("[unreachable: ConnectError]")`
    string heuristic, which only ever inferred non-resolution from an *exception
    type name*, never actually checked whether the domain resolves. Any other DoH
    outcome (a real answer, a timeout, a non-200 response, or a request error) is
    left as "resolvable, or at least not provably not" - `unknown must not collapse
    into valid` cuts both ways: a network hiccup here must never be asserted as a
    live NXDOMAIN fact.
    """
    try:
        with httpx.Client(timeout=10, headers={"User-Agent": UA}) as c:
            r = c.get("https://cloudflare-dns.com/dns-query", params={"name": domain, "type": "A"},
                      headers={"accept": "application/dns-json"})
        if r.status_code != 200:
            return True
        return r.json().get("Status") != 3
    except Exception:  # noqa: BLE001 - a failed DoH check must never assert NXDOMAIN
        return True


def fetch_wikipedia(company: str) -> tuple[str, str]:
    """Best-effort fallback text source: search Wikipedia, return (summary, url)."""
    try:
        with httpx.Client(timeout=settings.http_timeout, headers={"User-Agent": UA}) as c:
            s = c.get(WIKI_SEARCH, params={
                "action": "query", "list": "search", "srsearch": company,
                "srlimit": 1, "format": "json",
            }).json()
            hits = s.get("query", {}).get("search", [])
            if not hits:
                return "", ""
            title = hits[0]["title"]
            # Relevance gate: Wikipedia search happily returns "Five Guys" for a bakery.
            # Only trust an article whose title actually resembles the company name.
            # (Identity for structured facts now comes from Wikidata's website match,
            # below; this stays a soft gate for the supplementary text only.)
            if fuzz.token_sort_ratio(title.lower(), company.lower()) < 75:
                return "", ""
            page = c.get(WIKI_SUMMARY.format(title=title.replace(" ", "_"))).json()
        return page.get("extract", ""), page.get("content_urls", {}).get("desktop", {}).get("page", "")
    except Exception:  # noqa: BLE001
        return "", ""


def _render_wikidata_text(facts: wikidata_provider.WikidataFacts) -> str:
    emp = f"{facts.employees} (as of {facts.employees_as_of})" if facts.employees else "unknown"
    return (
        f"Wikidata entity {facts.qid}, website-matched to {facts.website}\n"
        f"Country: {facts.country or 'unknown'}\n"
        f"Headquarters: {facts.headquarters or 'unknown'}\n"
        f"Employees: {emp}\n"
        f"Industry: {', '.join(facts.industries) or 'unknown'}\n"
        f"Parent organization: {facts.parent or 'none'}"
    )


def _build_system(_rc: ResearchConfig) -> str:
    """The research prompt; the tech_signals guidance is the profile's `research` section."""
    return f"""You are a B2B sales research assistant. You receive raw text about a company,
each source wrapped in an EVIDENCE block labelled with an evidence id like (ev-001), and
must return STRICT JSON with these keys:
summary (2-3 sentences, plain, useful to a sales rep),
industry (short label),
headquarters_country (country name in English, or "unknown"),
estimated_employees (integer or null; infer from headcount statements, size bands, office counts,
  Wikidata's Employees property, or well-known scale; null if no basis),
tech_signals (list of short strings, each ending in the evidence id it came from in parentheses,
  e.g. "{_rc.tech_signal_example} (ev-001)": {_rc.tech_signal_guidance}),
confidence ("high"|"medium"|"low" - how well the sources support the above),
evidence_ids (list of the evidence ids, e.g. ["ev-001","ev-002"], that support summary/industry/HQ),
uncertainties (list of short strings: what's still unclear or unsupported, e.g. "no headcount source").
Never invent facts not supported by the text. If the website text is missing or generic, say so in
summary, list it in uncertainties, and lower confidence.

{INJECTION_WARNING}"""


_SYSTEM = _build_system(research_config())


def research_lead(lead: Lead, run: ProvenanceRun | None = None) -> Research:
    r = Research()
    website_url = _normalise_url(lead.website)
    r.website_text, r.website_ok, raw_html = fetch_website(lead.website)
    r.sources.append(website_url)
    r.wikipedia_summary, r.wikipedia_url = fetch_wikipedia(lead.company)
    if r.wikipedia_url:
        r.sources.append(r.wikipedia_url)

    evidence_blocks: list[str] = []  # rendered <<<EVIDENCE ...>>> blocks for the prompt

    # the family enum has no general "company facts" or "reference text"
    # bucket, so these three non-provider source types (plain website/subpage text,
    # and Wikipedia's fallback summary) are filed under the closest fit: Wikipedia is
    # `encyclopedic`; the website's own text is general corporate-identity material,
    # same bucket as GLEIF/RDAP - see models.Evidence docstring for the full call.
    _FAMILY_BY_SOURCE_TYPE = {
        "website": "corporate_identity", "website_subpage": "corporate_identity",
        "wikipedia": "encyclopedic",
    }

    def _record(source_type: str, url: str, text: str, strength: str,
               *, raw: bytes | None = None, derived_text: str = "") -> str | None:
        """Record one evidence item (if `run` is given) and return its id, or None.

        Fix #3 (v0.2 PART B): when `raw` is given (the page's actual RAW response
        bytes - "website"/"website_subpage" callers below), `content_sha256` and
        the snapshot file hash/store THOSE bytes, not `text` - `text` (the
        trafilatura extraction) is kept separately in the new `Evidence.derived_text`
        field instead. Before this fix, the snapshot silently hashed the extracted
        text, so it never proved the byte-for-byte page content it claimed to.
        Every other caller (unchanged) still hashes/snapshots `text` itself, exactly
        as before - `content_sha256`'s meaning for those source_types is untouched.
        """
        if not text and raw is None:
            return None
        eid = run.next_evidence_id() if run is not None else f"ev-{len(r.evidence) + 1:03d}"
        raw_bytes = raw if raw is not None else text.encode("utf-8")
        ev = Evidence(
            id=eid, source_type=source_type, url=url,
            observed_at=datetime.now(UTC).isoformat(), content_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            strength=strength, snippet=(derived_text or text)[:200],
            snapshot_path=f"out/provenance/snapshots/{run.run_id if run else 'unrecorded'}/{eid}.txt",
            family=_FAMILY_BY_SOURCE_TYPE.get(source_type, "corporate_identity"),
            derived_text=derived_text,
        )
        r.evidence.append(ev)
        if run is not None:
            run.record(ev, raw_bytes, stage="research")
        return eid

    # 1. Website landing page. The snapshot is the raw fetched HTML (Fix #3); the
    # LLM prompt still sees the trafilatura-extracted text (`r.website_text`), never
    # the raw markup.
    website_ev_id = _record(
        "website", website_url, r.website_text if r.website_ok else "", "STRONG",
        raw=raw_html.encode("utf-8") if r.website_ok and raw_html else None, derived_text=r.website_text,
    )
    if website_ev_id:
        evidence_blocks.append(wrap_evidence(website_ev_id, "website", r.website_text))

    # 2. Bounded same-origin crawl (about/careers/engineering/... up to 4 pages).
    # Fix #4: each page's structured chunks (trafilatura main text + FAQPage
    # JSON-LD Q/A + <details>/<table> text) - used below to build every page/
    # domain-driven provider's input, not just the flat extracted text.
    subpages: list[dict] = []
    structured_pages: list[dict] = []  # [{"url", "text", "locator"}]
    if raw_html:
        structured_pages.extend(
            {"url": website_url, "text": c.text, "locator": c.locator}
            for c in extract.extract_structured(raw_html, r.website_text if r.website_ok else "")
        )
    if r.website_ok and raw_html:
        subpages = website_provider.crawl_subpages(website_url, raw_html)
        for page in subpages:
            sub_id = _record("website_subpage", page["url"], page["text"], page["strength"],
                             raw=page.get("html", "").encode("utf-8") or None, derived_text=page["text"])
            if sub_id:
                r.sources.append(page["url"])
                evidence_blocks.append(wrap_evidence(sub_id, f"website_subpage:{page['url']}", page["text"]))
            structured_pages.extend(
                {"url": page["url"], "text": c.text, "locator": c.locator}
                for c in extract.extract_structured(page.get("html", ""), page["text"])
            )

    # 2b. Evidence providers that read the fetched pages: ATS job-board links and a
    # linked GitHub org. Never guess a board slug or an org name - see providers/
    # ats.py and providers/github.py docstrings. A provider exception never reaches
    # here (each returns "degraded" with no evidence instead) - see models.ProviderResult.
    # `page_hrefs` is the actual <a href> targets across the landing page AND every
    # crawled subpage - a board/org link that lives only in a subpage's footer/nav
    # anchor (not in that subpage's trafilatura-extracted TEXT, and not on the
    # landing page at all) is otherwise invisible to ats.py/github.py's link
    # detection, which used to see extracted text only.
    page_texts = ([raw_html] if raw_html else []) + [p["text"] for p in subpages]
    page_hrefs = (website_provider.extract_hrefs(raw_html, website_url) if raw_html else []) + [
        href for p in subpages for href in p.get("hrefs", [])
    ]

    def _absorb(result) -> None:
        r.provider_results.append(result)
        r.evidence.extend(result.evidence)
        for ev in result.evidence:
            evidence_blocks.append(wrap_evidence(ev.id, ev.source_type, ev.snippet))

    # Registrable domain (leading "www." stripped): querying crt.sh with "www.<domain>"
    # instead of the bare domain returns a near-empty result that then caches forever
    # (see leadscout/util.py's registrable_domain docstring for the observed symptom).
    domain = registrable_domain(urlparse(website_url).netloc)

    # 2c-2e. The remaining page/domain-driven evidence providers, as an ordered
    # registry of callables (deferred, not called yet) rather than five repeated
    # `_absorb(x_provider.run(...))` call sites: ats/github read the fetched page
    # texts+hrefs for a board/org link (never guessed); footprint is the passive
    # crt.sh->DoH->IP-range network footprint, keyed on the domain alone so it runs
    # whether or not the website fetch succeeded; trust_pages reads the same page
    # texts plus its own status-page probes; vendor is Brave-search-gated case-study
    # discovery, skipped without BRAVE_API_KEY. Order doesn't change the outcome -
    # cloud.py's assessment dedupes by family across all of them regardless of
    # arrival order - so this is plain data, not a control-flow decision.
    page_providers = [
        lambda: ats_provider.run(lead.company, page_texts, hrefs=page_hrefs, run=run),
        lambda: github_provider.run(lead.company, page_texts, hrefs=page_hrefs, run=run),
        lambda: footprint_provider.run(domain, run=run),
        lambda: trust_pages_provider.run(domain, structured_pages, run=run),
        lambda: vendor_provider.run(lead.company, website_url, run=run),
    ]
    for make_result in page_providers:
        _absorb(make_result())

    # 3. Wikidata, accepted only on an official-website match. The ledger snapshot is
    # the raw entity JSON (full chain of custody); the prompt sees a short rendered
    # summary of it instead, to keep the prompt small.
    wikidata_facts, wikidata_outcome = wikidata_provider.lookup_by_website(lead.company, lead.website)
    # A lookup that did not happen is a missing channel, not a fact about the company.
    r.provider_results.append(ProviderResult(
        provider_name="wikidata",
        status="degraded" if wikidata_outcome == wikidata_provider.DEGRADED else "ok",
        note=f"wikidata lookup {wikidata_outcome.lower()}"))
    if wikidata_facts is not None:
        wikidata_url = f"https://www.wikidata.org/wiki/{wikidata_facts.qid}"
        wikidata_text = _render_wikidata_text(wikidata_facts)
        wd_id = run.next_evidence_id() if run is not None else f"ev-{len(r.evidence) + 1:03d}"
        raw_json = json.dumps(wikidata_facts.raw, ensure_ascii=False).encode("utf-8")
        ev = Evidence(
            id=wd_id, source_type="wikidata", url=wikidata_url,
            observed_at=datetime.now(UTC).isoformat(), content_sha256=hashlib.sha256(raw_json).hexdigest(),
            strength="STRONG", snippet=wikidata_text[:200],
            snapshot_path=f"out/provenance/snapshots/{run.run_id if run else 'unrecorded'}/{wd_id}.json",
            family="encyclopedic",
        )
        r.evidence.append(ev)
        if run is not None:
            run.record(ev, raw_json, stage="research")
        evidence_blocks.append(wrap_evidence(wd_id, "wikidata", wikidata_text))
        r.sources.append(wikidata_url)
        r.wikidata_industries = list(wikidata_facts.industries)
        r.wikidata_evidence_id = wd_id
    else:
        r.uncertainties.append("no website-matched Wikidata entity")

    # 3b. GLEIF legal-entity identity, using Wikidata's country as the known-HQ
    # factor when we have it (available before the LLM's own headquarters_country
    # extraction runs) - see providers/gleif.py docstring for the multi-factor
    # acceptance rule this depends on.
    hq_country = wikidata_facts.country if wikidata_facts is not None else None
    gleif_result = gleif_provider.run(lead.company, hq_country, lead.website, run=run)
    _absorb(gleif_result)
    # Only a country-corroborated record states THIS lead's registered jurisdiction: a
    # WEAK one is a name-only candidate never confirmed to be the lead at all (see the
    # WEAK branch in providers/gleif.py), so it must not fill this field either.
    if gleif_result.evidence and gleif_result.evidence[0].strength != "WEAK":
        r.registered_jurisdiction = (
            gleif_result.evidence[0].snippet.split("registered_jurisdiction=")[-1].strip())

    # 3c. Hunter.io contact-quality check (email-verifier + domain-search) -
    # informational only for the sales rep (models.Research.contact_quality/
    # contact_quality_note), never fed into `evidence_blocks`: its facts must not
    # reach the LLM's summary/industry/employees extraction, which IS what feeds
    # fit.py - see providers/hunter.py's design notes and the byte-identical-
    # outcomes test in tests/test_hunter.py. The one Evidence it can produce is
    # still appended to `r.evidence`/`r.provider_results` for the ledger and the
    # Technical sheet's cloud-families column, but its family (`corporate_identity`)
    # never contributes to cloud.py's assessment either way (see cloud.py's design
    # notes: "corporate_identity ... NEVER contribute to cloud").
    hunter_result, hunter_facts = hunter_provider.run(lead.email, domain, run=run)
    r.provider_results.append(hunter_result)
    r.evidence.extend(hunter_result.evidence)
    r.contact_quality = hunter_facts.contact_quality
    r.contact_quality_note = hunter_facts.contact_quality_note
    r.domain_entity_name = hunter_facts.domain_organisation

    # 4. Wikipedia summary, fallback text only.
    wiki_ev_id = _record("wikipedia", r.wikipedia_url, r.wikipedia_summary, "MEDIUM")
    if wiki_ev_id:
        evidence_blocks.append(wrap_evidence(wiki_ev_id, "wikipedia", r.wikipedia_summary))

    user = (
        f"Company name: {lead.company}\nWebsite: {lead.website} (domain {domain})\n"
        f"Contact: {lead.name}, {lead.job_title or 'title unknown'}\n"
        f"Self-reported size band: {lead.company_size_band or 'not given'}\n"
        f"Website fetch status: {'ok' if r.website_ok else 'FAILED'}\n\n"
        + ("\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence could be fetched)")
    )
    facts = ask_model(_SYSTEM, user, ResearchFacts, purpose="research")
    r.summary = facts.summary.strip()
    r.industry = facts.industry.strip()
    r.headquarters_country = facts.headquarters_country.strip()
    r.hq_source = "llm"
    r.estimated_employees = facts.estimated_employees
    r.tech_signals = [t for t in facts.tech_signals if t][:10]
    r.confidence = facts.confidence
    r.evidence_ids = list(facts.evidence_ids)
    r.uncertainties = r.uncertainties + list(facts.uncertainties)
    if r.estimated_employees:
        r.employees_source = "website/LLM research"

    # Headcount is the first fit criterion, so its authority ladder is:
    #   the applicant's own size band  >  an identity-checked third-party figure
    #   (providers/headcount.py's exact-domain Diffbot match)  >  the model's inference
    #   from the evidence text.
    #
    # The guard used to be `not r.estimated_employees and not lead.company_size_band`,
    # which made the model's own guess suppress the lookup entirely - the same inversion
    # as the HQ waterfall, and worse in one respect: the stronger source was not
    # outranked, it was never queried. The system prompt explicitly invites that guess
    # ("infer from ... office counts, or well-known scale"), so it fires often.
    #
    # The applicant's band still short-circuits the lookup: it is first-party data about
    # the entity that actually applied, and a group-vs-subsidiary mismatch is the typical
    # way a third-party figure is wrong. An identity-checked figure now overrides a
    # model-only one; an unverified web-search figure stays a hint, never scored.
    if not lead.company_size_band:
        llm_employees = r.estimated_employees
        hc_result, hc_candidates = headcount_provider.run(lead.company, domain, ask_json, run=run)
        r.provider_results.append(hc_result)
        r.evidence.extend(hc_result.evidence)
        scored, hint = headcount_provider.choose(hc_candidates)
        if scored is not None:
            if llm_employees and llm_employees != scored.employees:
                r.uncertainties.append(
                    f"identity-checked headcount ({scored.employees}) overrides the figure inferred "
                    f"from the evidence text ({llm_employees}) - the domain-matched record is the "
                    f"stronger source")
            r.estimated_employees = scored.employees
            r.employees_source = f"{scored.detail} ({scored.evidence_id})"
        if hint is not None:
            r.employees_hint = (f"web search suggests {hint.employees} - {hint.detail} ({hint.evidence_id}); "
                                "entity not verified, not used in the fit score")

    _resolve_headquarters(r, lead, gleif_result, wikidata_facts)

    # v0.2.2 infrastructure footprints (PART A, scope cut): built from
    # the footprint provider's own per-host observations (range match / RIPEstat
    # holder / cPanel trace) - see leadscout/infra.py. Wired in BEFORE cloud-usage
    # classification, per the infrastructure model's detection order.
    footprint_result = next((pr for pr in r.provider_results if pr.provider_name == "footprint"), None)
    if footprint_result is not None:
        r.infrastructure = infra.build_footprints(footprint_result)

    # Deterministic cloud-usage classification over every piece of
    # evidence collected above - never the LLM's own guess, see leadscout/cloud.py.
    # Fix #2: `domain_resolved` is now a real DoH Status check, not an exception-
    # string heuristic - but only spent when the website fetch already failed (a
    # successful fetch already proves the domain resolves, so there is nothing to
    # check, and no extra network call is made for the common case).
    domain_resolved = r.website_ok or _apex_resolves(domain)
    r.cloud_usage = assess_cloud_usage(r.evidence, provider_results=r.provider_results,
                                       domain_resolved=domain_resolved)
    return r


def _resolve_headquarters(
    r: Research, lead: Lead, gleif_result, wikidata_facts: wikidata_provider.WikidataFacts | None,
) -> None:
    """HQ waterfall, strongest source first: GLEIF -> Wikidata -> LLM extraction ->
    website ccTLD, only falling further when the stronger rung came back empty.

    That order is the correction. This function used to return immediately if the
    caller's LLM extraction had produced any non-"unknown" string, which made the LLM
    the TOP rung for the single highest-stakes field in the system - measured: with an
    accepted GLEIF record saying Germany and the LLM saying Iran, the run ended up on
    Iran/llm and GLEIF's answer was never read; with the same GLEIF record and the LLM
    saying "unknown", it correctly ended up on Germany/gleif. Its own docstring called
    that ordering "strongest first", and `compliance.py`'s authority ladder says the
    opposite ("registry ... > enrichment > LLM inference"). Two modules documented
    contradictory orders and the implementation followed the weaker one. It is also why
    every lead in the committed batch carried `hq_source="llm"`: not because the
    registries found nothing, but because they were never consulted.

    The rungs: (1) an ACCEPTED GLEIF legal-address country (structured,
    government-registry-backed - see providers/gleif.py's multi-factor acceptance
    rule), (2) a website-matched Wikidata entity's P17 country claim, (3) the LLM's own
    extraction from the evidence text - a reading of sources, not a source, (4) the
    submitted website's country-code TLD (weakest; a generic gTLD gives no country
    signal at all). GLEIF and Wikidata disagreeing leaves HQ unknown rather than
    silently picking one, and now also discards the LLM's value: a conflict between two
    registries is the worst moment to fall back on a guess. "unknown" must never
    collapse into a guessed "valid" value, and the HQ-unknown safety net then sends the
    lead to review, which is the honest outcome. The ccTLD rung exists because leaving
    HQ unknown for any non-.com company whose stronger rungs all came back empty (e.g.
    a local bakery on a .hu domain) used to trip that same "review" floor on an
    otherwise unremarkable business - a weak-but-structured signal beats staying
    unknown, it just cannot outrank a stronger one.
    """
    llm_country = (r.headquarters_country or "").strip()
    if llm_country.lower() == "unknown":
        llm_country = ""
    gleif_country = None
    if gleif_result.evidence:
        gleif_country = gleif_provider.legal_country_name_from_evidence(gleif_result.evidence[0])
    wikidata_country = wikidata_facts.country if wikidata_facts is not None else None

    # A stronger rung contradicting the LLM is worth recording even though the stronger
    # one simply wins: the rep should see that the sources disagreed, not just the answer.
    for stronger, label in ((gleif_country, "GLEIF legal-address"), (wikidata_country, "Wikidata")):
        if stronger and llm_country and stronger != llm_country:
            r.uncertainties.append(
                f"{label} country ({stronger}) overrides the country extracted from the "
                f"evidence text ({llm_country}) - the registry record is the stronger source")

    if gleif_country and wikidata_country and gleif_country != wikidata_country:
        r.headquarters_country = ""
        r.hq_source = ""
        r.uncertainties.append(
            f"GLEIF legal-address country ({gleif_country}) and Wikidata's country "
            f"({wikidata_country}) disagree - headquarters left unknown")
    elif gleif_country:
        r.headquarters_country = gleif_country
        r.hq_source = "gleif"
    elif wikidata_country:
        r.headquarters_country = wikidata_country
        r.hq_source = "wikidata"
    elif llm_country:
        r.headquarters_country = llm_country
        r.hq_source = "llm"
    else:
        tld_country = gleif_provider.country_from_website_tld(lead.website)
        if tld_country:
            r.headquarters_country = tld_country
            r.hq_source = "website_tld"
            r.hq_confidence = "low"


def research_from_text(lead: Lead, website_text: str, run: ProvenanceRun | None = None) -> Research:
    """Adversarial-eval entry point: feed `website_text` directly as
    the ONLY website evidence - no fetch, no crawl, no ATS/GitHub/footprint/
    trust_pages/vendor providers, no Wikipedia/Wikidata/GLEIF lookups. This is
    deliberately a much smaller subset of `research_lead`: the adversarial cases
    test whether the LLM prompt resists hostile content INSIDE the evidence text,
    not whether the provider fan-out behaves - and none of those providers make
    network calls we want in an eval run. `run` is accepted (matches `research_lead`'s
    signature) so a real `ProvenanceRun` can still record the one evidence item, but
    is optional and defaults to unrecorded (consistent with `research_lead`)."""
    r = Research()
    r.website_text = website_text
    r.website_ok = bool(website_text)
    website_url = _normalise_url(lead.website)
    r.sources.append(website_url)

    eid = run.next_evidence_id() if run is not None else "ev-001"
    raw = website_text.encode("utf-8")
    ev = Evidence(
        id=eid, source_type="website", url=website_url,
        observed_at=datetime.now(UTC).isoformat(), content_sha256=hashlib.sha256(raw).hexdigest(),
        strength="STRONG", snippet=website_text[:200],
        snapshot_path=f"out/provenance/snapshots/{run.run_id if run else 'unrecorded'}/{eid}.txt",
        family="corporate_identity",
    )
    r.evidence.append(ev)
    if run is not None:
        run.record(ev, raw, stage="research")
    evidence_blocks = [wrap_evidence(eid, "website", website_text)]

    domain = urlparse(website_url).netloc
    user = (
        f"Company name: {lead.company}\nWebsite: {lead.website} (domain {domain})\n"
        f"Contact: {lead.name}, {lead.job_title or 'title unknown'}\n"
        f"Self-reported size band: {lead.company_size_band or 'not given'}\n"
        f"Website fetch status: {'ok' if r.website_ok else 'FAILED'}\n\n"
        + ("\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence could be fetched)")
    )
    facts = ask_model(_SYSTEM, user, ResearchFacts, purpose="research")
    r.summary = facts.summary.strip()
    r.industry = facts.industry.strip()
    r.headquarters_country = facts.headquarters_country.strip()
    r.hq_source = "llm"
    r.estimated_employees = facts.estimated_employees
    r.tech_signals = [t for t in facts.tech_signals if t][:10]
    r.confidence = facts.confidence
    r.evidence_ids = list(facts.evidence_ids)
    r.uncertainties = list(facts.uncertainties)
    r.cloud_usage = assess_cloud_usage(r.evidence)
    return r
