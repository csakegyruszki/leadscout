"""Wikidata company lookup, accepted only when the entity's official website (P856)
matches the lead's own domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from ..config import settings
from ..util import USER_AGENT

# --- Design notes -----------------------------------------------------------------
# Why: the old Wikipedia relevance gate (fuzzy-matching the article *title* to the
# company name, `token_sort_ratio >= 75`) is easy to fool - it once returned *Five
# Guys* for a bakery and *Rolls-Royce Silver Cloud* for "Cloud-Trim" (see README "How I
# used AI tooling"). A structured property match against the domain the lead itself
# submitted is a much harder identity claim to fake by accident: P856 has to literally
# be the same website. Wikipedia's plain-text summary stays as a fallback text source
# (research.py still fetches it) for companies with no Wikidata item or no P856 match.

# Wikimedia's API etiquette requires an identifying User-Agent with contact info;
# a generic one gets 403s (measured 2026-09-19).
UA = USER_AGENT
API_URL = "https://www.wikidata.org/w/api.php"
ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

P_OFFICIAL_WEBSITE = "P856"
P_COUNTRY = "P17"
P_HEADQUARTERS = "P159"
P_EMPLOYEES = "P1128"
P_INDUSTRY = "P452"
P_PARENT = "P749"


def _registrable_host(url_or_domain: str) -> str:
    """Best-effort "registrable host": lowercase, strip a leading www., strip port
    and path. Not a full public-suffix-list implementation - good enough to compare
    "zapier.com" to "https://zapier.com/" or "www.zapier.com"."""
    s = url_or_domain.strip().lower()
    if "//" not in s:
        s = "//" + s
    parsed = urlparse(s)
    host = (parsed.netloc or parsed.path).split(":")[0].rstrip("/")
    return host[4:] if host.startswith("www.") else host


def _claims(entity: dict, prop: str) -> list[dict]:
    return entity.get("claims", {}).get(prop, [])


def _best_string_value(claims: list[dict]) -> str | None:
    for c in claims:
        val = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(val, str):
            return val
    return None


def _item_ids(claims: list[dict]) -> list[str]:
    ids = []
    for c in claims:
        val = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(val, dict) and val.get("entity-type") == "item":
            ids.append(val["id"])
    return ids


def _latest_quantity(claims: list[dict]) -> tuple[str, str | None] | None:
    """Pick the claim with the latest P585 (point in time) qualifier. Returns
    (amount, date) for the winner, or None if there are no claims at all. A claim
    with no P585 qualifier sorts before any dated one (empty string < any ISO date)."""
    best: tuple[str, str | None] | None = None
    for c in claims:
        val = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if not isinstance(val, dict) or "amount" not in val:
            continue
        amount = str(val["amount"]).lstrip("+")
        date = None
        for q in c.get("qualifiers", {}).get("P585", []):
            date = q.get("datavalue", {}).get("value", {}).get("time")
        if best is None or (date or "") > (best[1] or ""):
            best = (amount, date)
    return best


def _labels_for(qids: list[str], client: httpx.Client) -> dict[str, str]:
    unique = sorted(set(qids))
    if not unique:
        return {}
    r = client.get(API_URL, params={
        "action": "wbgetentities", "ids": "|".join(unique),
        "props": "labels", "languages": "en", "format": "json",
    })
    r.raise_for_status()
    entities = r.json().get("entities", {})
    return {qid: e.get("labels", {}).get("en", {}).get("value", qid) for qid, e in entities.items()}


def _search_candidates(company: str, client: httpx.Client) -> list[str]:
    r = client.get(API_URL, params={
        "action": "wbsearchentities", "search": company, "language": "en",
        "limit": 5, "format": "json",
    })
    r.raise_for_status()
    return [hit["id"] for hit in r.json().get("search", [])]


def _entity_data(qid: str, client: httpx.Client) -> dict:
    r = client.get(ENTITY_DATA_URL.format(qid=qid))
    r.raise_for_status()
    return r.json()["entities"][qid]


@dataclass
class WikidataFacts:
    qid: str
    website: str
    country: str | None = None
    headquarters: str | None = None
    employees: int | None = None
    employees_as_of: str | None = None
    industries: list[str] = field(default_factory=list)
    parent: str | None = None
    raw: dict = field(default_factory=dict)  # the accepted entity's raw JSON, for the ledger


#: Outcome of a lookup, kept separate from its result. "Nothing matched" and "the
#: lookup did not happen" both used to be a bare None, which made a Wikidata outage
#: indistinguishable from a company that genuinely has no entity - and research.py
#: recorded "no website-matched Wikidata entity" as an uncertainty either way, so an
#: outage was reported as a finding about the company. FOUND/NOT_FOUND are completed
#: measurements; DEGRADED is the absence of one and must never be read as evidence.
FOUND, NOT_FOUND, DEGRADED = "FOUND", "NOT_FOUND", "DEGRADED"


def lookup_by_website(company: str, website: str) -> tuple[WikidataFacts | None, str]:
    """Search Wikidata for `company`; accept the first candidate whose P856
    (official website) has the same registrable host as `website`.

    Returns `(facts, outcome)` and never raises. `outcome` is FOUND, NOT_FOUND (the
    search ran and no candidate matched the domain) or DEGRADED (the API was
    unreachable, rate-limited or answered something unparseable).
    """
    target_host = _registrable_host(website)
    if not target_host:
        return None, NOT_FOUND
    try:
        with httpx.Client(timeout=settings.http_timeout, headers={"User-Agent": UA}) as client:
            candidates = _search_candidates(company, client)
            for qid in candidates:
                entity = _entity_data(qid, client)
                site = _best_string_value(_claims(entity, P_OFFICIAL_WEBSITE))
                if not site or _registrable_host(site) != target_host:
                    continue

                country_ids = _item_ids(_claims(entity, P_COUNTRY))
                hq_ids = _item_ids(_claims(entity, P_HEADQUARTERS))
                industry_ids = _item_ids(_claims(entity, P_INDUSTRY))
                parent_ids = _item_ids(_claims(entity, P_PARENT))
                labels = _labels_for(country_ids + hq_ids + industry_ids + parent_ids, client)

                facts = WikidataFacts(qid=qid, website=site, raw=entity)
                if country_ids:
                    facts.country = labels.get(country_ids[0])
                if hq_ids:
                    facts.headquarters = labels.get(hq_ids[0])
                facts.industries = [labels.get(i, i) for i in industry_ids][:5]
                if parent_ids:
                    facts.parent = labels.get(parent_ids[0])

                emp = _latest_quantity(_claims(entity, P_EMPLOYEES))
                if emp:
                    try:
                        facts.employees = int(float(emp[0]))
                    except (TypeError, ValueError):
                        facts.employees = None
                    facts.employees_as_of = emp[1]
                return facts, FOUND
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        return None, DEGRADED
    return None, NOT_FOUND
