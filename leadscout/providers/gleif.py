"""GLEIF (Global LEI Foundation) legal-entity identity check.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
from rapidfuzz import fuzz

from ..compliance import normalise_name
from ..config import settings
from ..models import Evidence, ProviderResult
from ._base import UA, fallback_evidence_id, snapshot_path

# --- Design notes -----------------------------------------------------------------
# Never accepts on name similarity alone: GLEIF's `lei-records?filter[entity.legalName]=`
# does a loose match, not an exact one - "Snapp" returns SNIPP SNAPP AS (Norway) and
# SNAPP PADDY PROCESSING (India), neither of which is the ride-hailing "Snapp" the lead
# actually is. A candidate is accepted only when ALL of: (1) name similarity >= 85 after
# stripping legal suffixes (`compliance.normalise_name`, same rule used for the
# do-not-engage prescreen), (2) the legal address country matches the company's known HQ
# country - or, when HQ is unknown, doesn't *conflict* with a country the website's own
# ccTLD implies (a generic .com/.io gives no country signal to conflict with, so it just
# caps confidence at MEDIUM instead of blocking), and (3) `entity.status == "ACTIVE"`.
# Measured 2026-09-19: Zapier (HQ US) -> ZAPIER, INC. US ACTIVE, accept STRONG; Snapp
# (HQ Iran) -> both Norway and India candidates rejected on country, zero evidence.
#
# `entity.jurisdiction` (e.g. "US-DE") is exposed as `registered_jurisdiction` in the
# Evidence snippet - a separate fact from the operational HQ country, which a shell
# company can register far from where it actually operates.

API = "https://api.gleif.org/api/v1/lei-records"
NAME_SIM_THRESHOLD = 85

# Country-name (as Wikidata/the LLM renders it, English) -> ISO 3166-1 alpha-2, for
# the countries this pipeline's fixtures and sample leads actually use. Not a full
# gazetteer - an unmapped name is treated as "can't compare", never as a match.
_COUNTRY_TO_ISO2 = {
    "united states of america": "US", "united states": "US", "usa": "US",
    "iran": "IR", "islamic republic of iran": "IR",
    "hungary": "HU", "italy": "IT", "india": "IN", "norway": "NO",
    "united kingdom": "GB", "germany": "DE", "france": "FR", "netherlands": "NL",
    "canada": "CA", "australia": "AU", "spain": "ES", "sweden": "SE",
}
# ccTLD -> ISO2, for the website-derived fallback when HQ is unknown. A generic gTLD
# (.com/.io/.net/...) is deliberately absent - it carries no country signal.
_TLD_TO_ISO2 = {
    "us": "US", "ir": "IR", "hu": "HU", "it": "IT", "in": "IN", "no": "NO",
    "uk": "GB", "de": "DE", "fr": "FR", "nl": "NL", "ca": "CA", "au": "AU",
}


# Canonical English rendering per ISO2, for research.py's HQ fallback (item 4):
# a shorter, canonical name per code drawn from the keys above (first alias wins).
_ISO2_TO_COUNTRY_NAME = {
    "US": "United States", "IR": "Iran", "HU": "Hungary", "IT": "Italy",
    "IN": "India", "NO": "Norway", "GB": "United Kingdom", "DE": "Germany",
    "FR": "France", "NL": "Netherlands", "CA": "Canada", "AU": "Australia",
    "ES": "Spain", "SE": "Sweden",
}


def _iso2_from_country_name(name: str | None) -> str | None:
    return _COUNTRY_TO_ISO2.get((name or "").strip().lower()) or None


def legal_country_name_from_evidence(ev: Evidence) -> str | None:
    """Render an accepted GLEIF Evidence's legal-address ISO2 (embedded in its
    snippet, "<name> (<lei>), <ISO2>, ACTIVE, registered_jurisdiction=...") back to
    an English country name, for research.py's structured HQ fallback (item 4:
    used only when the LLM's own headquarters_country came back empty/unknown).

    A WEAK record is a name-only candidate that no country factor corroborated, so it
    yields no HQ at all - the waterfall falls through to Wikidata/ccTLD instead of
    promoting an unconfirmed entity's address to this lead's headquarters.
    """
    if ev.strength == "WEAK":
        return None
    try:
        iso2 = ev.snippet.split("), ", 1)[1].split(",", 1)[0].strip()
    except IndexError:
        return None
    return _ISO2_TO_COUNTRY_NAME.get(iso2)


def _iso2_from_website(website: str) -> str | None:
    host = urlparse(website if "//" in website else f"//{website}").netloc.lower()
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    return _TLD_TO_ISO2.get(tld)


def country_from_website_tld(website: str) -> str | None:
    """The website's own country-code TLD, rendered as an English country name -
    the last, weakest rung of research.py's structured HQ waterfall (item 4:
    LLM -> GLEIF -> Wikidata -> website ccTLD), used only when none of the
    stronger sources resolved an HQ. Reuses the same ccTLD table `_candidate_strength`
    already uses for GLEIF's own country-conflict check, so both call sites treat
    "what does this ccTLD imply" identically. A generic gTLD (.com/.io/...) returns
    None - it carries no country signal at all, by design (see `_TLD_TO_ISO2`)."""
    iso2 = _iso2_from_website(website)
    return _ISO2_TO_COUNTRY_NAME.get(iso2) if iso2 else None


def _candidate_strength(legal_country: str, hq_country: str | None, website: str) -> str | None:
    """None = reject; else the strength to record. STRONG requires the known HQ
    country to match the legal address; MEDIUM requires the website's OWN ccTLD
    country to match instead (a real, if weaker, confirming signal); a generic gTLD
    with HQ unknown confirms nothing at all, so it's WEAK - identity-only, never
    MEDIUM (REVIEW-6bA-verified.md #5: MEDIUM used to be reachable with no country
    factor satisfied at all)."""
    hq_iso2 = _iso2_from_country_name(hq_country)
    if hq_iso2:
        return "STRONG" if legal_country == hq_iso2 else None
    web_iso2 = _iso2_from_website(website)
    if web_iso2:
        return "MEDIUM" if web_iso2 == legal_country else None
    return "WEAK"  # HQ unknown, generic gTLD - no country signal to confirm or conflict with


def run(company: str, hq_country: str | None, website: str, run=None) -> ProviderResult:
    try:
        with httpx.Client(timeout=settings.http_timeout, headers={"User-Agent": UA}) as client:
            r = client.get(API, params={"filter[entity.legalName]": company})
        # Every non-200 is `degraded`, not `ok`. A 429 or a 401 means the lookup did not
        # happen; returning `ok` made that byte-identical downstream to a lookup that ran
        # and found nothing, which is the one thing ProviderResult's own contract forbids
        # ("a failed provider is degraded ... never negative evidence"). cloud.py fills
        # `missing_channels` from `degraded` alone, so a rate-limited call used to let the
        # run report the confident NO_PUBLIC_CLOUD_EVIDENCE instead of INSUFFICIENT_EVIDENCE.
        if r.status_code != 200:
            return ProviderResult(provider_name="gleif", status="degraded",
                                  note=f"HTTP {r.status_code}", calls=1)

        # Captured before parsing (REVIEW-6bA-verified.md #1): the accepted
        # candidate's Evidence.content_sha256 traces back to the exact response
        # bytes GLEIF returned (the whole lei-records payload), not a re-dump of
        # only the one accepted record.
        raw = r.content
        records = r.json().get("data", [])
        company_norm = normalise_name(company)

        for rec in records:
            entity = rec.get("attributes", {}).get("entity", {})
            legal_name = entity.get("legalName", {}).get("name", "")
            similarity = fuzz.token_sort_ratio(company_norm, normalise_name(legal_name))
            if similarity < NAME_SIM_THRESHOLD:
                continue
            if entity.get("status") != "ACTIVE":
                continue
            legal_country = entity.get("legalAddress", {}).get("country", "")
            strength = _candidate_strength(legal_country, hq_country, website)
            if strength is None:
                continue

            jurisdiction = entity.get("jurisdiction", "unknown")
            lei = rec.get("attributes", {}).get("lei", "")
            if strength == "WEAK":
                # A WEAK candidate satisfied NO country factor: it is an active company
                # whose name resembles the lead's, not a company confirmed to BE the
                # lead. This snippet used to be formatted identically to a confirmed
                # one ("Notion OU (LEI), EE, ACTIVE, ..."), so the research LLM read the
                # candidate record's country as the lead's headquarters and the summary
                # asserted "Notion OU is a company registered in Estonia" - for a lead
                # whose own domain (notiontechnologies.com) belongs to an unrelated
                # Indian agency (adversarial battery 2026-09-20, case d). Country and
                # jurisdiction stay visible, but as properties OF THE CANDIDATE RECORD,
                # never of the lead: the module docstring's "identity-only" promise,
                # now actually carried by the text an LLM reads.
                snippet = (f"possible name match only, NOT confirmed to be this lead: "
                           f"{legal_name} ({lei}), ACTIVE; that record's legal-address "
                           f"country is {legal_country}, its "
                           f"registered_jurisdiction={jurisdiction}")
            else:
                snippet = (f"{legal_name} ({lei}), {legal_country}, ACTIVE, "
                           f"registered_jurisdiction={jurisdiction}")
            eid = fallback_evidence_id("gleif", run)
            ev = Evidence(
                id=eid, source_type="gleif", url=f"https://search.gleif.org/#/record/{lei}",
                observed_at=datetime.now(UTC).isoformat(),
                content_sha256=hashlib.sha256(raw).hexdigest(), strength=strength,
                snippet=snippet[:200],
                snapshot_path=snapshot_path(run, eid),
                family="corporate_identity", freshness="current", origin="provider:gleif",
            )
            if run is not None:
                run.record(ev, raw, stage="gleif")
            return ProviderResult(provider_name="gleif", status="ok", evidence=[ev],
                                   note=f"accepted {legal_name} ({jurisdiction})", calls=1)

        return ProviderResult(provider_name="gleif", status="ok",
                               note=f"{len(records)} candidate(s), none passed name+country+status", calls=1)
    except Exception as e:  # noqa: BLE001 - a malformed/unreachable response must degrade, never crash
        return ProviderResult(provider_name="gleif", status="degraded", note=f"{type(e).__name__}: {e}", calls=1)
