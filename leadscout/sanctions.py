"""Real sanctions/PEP screening via the OpenSanctions match API.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from rapidfuzz import fuzz

from .config import settings
from .models import Evidence

# --- Design notes -----------------------------------------------------------------
# Why topics, not `match`: OpenSanctions' `match` flag says "the API is confident this
# result and the query name the same entity" - it says nothing about *why* that entity
# is on file. Measured 2026-09-19: "Masterplast" (HQ Hungary) matched at 0.88 (`match`
# true) against a public-company registry record (`topics: [corp.public]`) - a routine
# disclosure, not a sanctions hit. "Snapp" (HQ Iran) matched at 0.6 against a FINRA
# securities action against an unrelated *person* named Steven Allen Snapp (`schema:
# Person`). Screening on `match` alone would either block a routine manufacturer or
# clear an Iran-HQ lead on the strength of an irrelevant American's surname.
#
# That first cut (schema + topics) still wasn't enough. The same day, "CT Inc" scored
# 0.83 `match: true` against an unrelated debarred "CT TRANSPORTATION COMPANY, INC", and
# "Cloud Solutions Ltd" scored 0.73 `match: true` against an unrelated debarred "Cloud
# Solutions LLC" - both real hard-topic, match=true hits, both about someone else
# entirely. Short or generic company names are weak queries: OpenSanctions' own
# confidence in the *name* match degrades faster than its confidence in the *topic*
# does, so `classify()` below also requires a high score, a real caption-similarity
# check, and a "not a trivial/short name" guard before a hit can force `blocked`.

if TYPE_CHECKING:
    from .provenance import ProvenanceRun

MATCH_URL = "https://api.opensanctions.org/match/default"
ENTITY_URL = "https://www.opensanctions.org/entities/{id}/"
CACHE_DIR = settings.out_dir / "cache" / "opensanctions"

# Topics that are themselves the compliance concern - a hit tagged with one of these
# means the company (or whoever controls it) is sanctioned, embargoed, debarred, or
# named in a criminal proceeding of the kind that ends a sales relationship. Anything
# else (corp.public, corp.disqual, reg.action, fin.*, role.*, poi) is a registry
# fact, not a reason to block.
_HARD_TOPICS = {
    "sanction", "sanction.linked", "sanction.counter",
    "export.control", "export.risk", "debarment",
    "crime", "crime.fraud", "crime.terror", "crime.war",
}

# Business words generic enough that matching one proves nothing about identity.
_GENERIC_TOKENS = {
    "cloud", "solutions", "systems", "services", "group", "trading", "tech",
    "technologies", "holdings", "international", "global", "consulting",
    "partners", "capital", "company", "co", "inc", "ltd",
}

# Small ISO2 map for the ~40 jurisdictions this project cares about. Unknown countries
# are omitted from the query rather than guessed - a wrong country code would narrow
# the search and could hide a real hit.
_ISO2 = {
    "united states": "us", "us": "us", "usa": "us",
    "united kingdom": "gb", "uk": "gb",
    "germany": "de", "austria": "at", "poland": "pl",
    "czechia": "cz", "czech republic": "cz", "slovakia": "sk",
    "romania": "ro", "ukraine": "ua", "turkey": "tr",
    "china": "cn", "india": "in",
    "united arab emirates": "ae", "uae": "ae",
    "israel": "il", "france": "fr", "italy": "it", "spain": "es",
    "netherlands": "nl", "switzerland": "ch", "sweden": "se",
    "norway": "no", "denmark": "dk", "finland": "fi", "ireland": "ie",
    "canada": "ca", "australia": "au", "japan": "jp", "south korea": "kr",
    "brazil": "br", "mexico": "mx", "singapore": "sg",
    "hungary": "hu", "iran": "ir", "russia": "ru", "belarus": "by",
    "cuba": "cu", "syria": "sy", "north korea": "kp",
    "venezuela": "ve", "myanmar": "mm",
}


def _iso2(country: str) -> str | None:
    return _ISO2.get((country or "").strip().lower())


def _is_hard_topic(topic: str) -> bool:
    return topic in _HARD_TOPICS or topic.startswith("crime.")


def _normalise(name: str) -> str:
    """Lazy import: compliance.py imports screen_sanctions from here, so importing
    compliance at module load time would be circular. By call time both modules are
    fully loaded."""
    from .compliance import normalise_name
    return normalise_name(name)


def _is_weak_query(query_norm: str) -> bool:
    """A weak query name can't support a confident sanctions match on its own.

    Deliberately narrower than "one token is always weak": that reading would also
    mark distinctive single-word names like "rosneft" or "sovcombank" as weak, which
    contradicts the measured expectation that those should be able to reach
    `blocked_evidence`. What actually makes a name weak here is being *trivial* (only
    generic business words) or *short* (its longest token is initialism-length) - not
    merely being one word.
    """
    tokens = query_norm.split()
    if not tokens:
        return True
    if all(t in _GENERIC_TOKENS for t in tokens):
        return True
    if max(len(t) for t in tokens) <= 3:
        return True
    return False


def _name_similarity(query_norm: str, caption_norm: str) -> int:
    return int(max(
        fuzz.token_sort_ratio(query_norm, caption_norm),
        fuzz.partial_ratio(query_norm, caption_norm),
    ))


def _evaluate(raw: dict, query_norm: str) -> dict:
    """Per-hit evaluation shared by `classify()` and `_build_hits()`."""
    topics = (raw.get("properties") or {}).get("topics") or []
    hard = [t for t in topics if _is_hard_topic(t)]
    score = float(raw.get("score", 0.0))
    match = bool(raw.get("match"))
    caption_norm = _normalise(raw.get("caption", ""))
    name_sim = _name_similarity(query_norm, caption_norm)
    weak = _is_weak_query(query_norm)

    meets_blocked = bool(hard) and match and score >= 0.85 and name_sim >= 80 and not weak
    meets_review = bool(hard) and not meets_blocked and (score >= 0.5 or match)

    why = None
    if hard and not meets_blocked:
        if weak:
            why = "weak query name"
        elif score < 0.85:
            why = "score below 0.85"
        elif name_sim < 80:
            why = f"caption similarity {name_sim}"
        elif not match:
            why = "match=false"

    return {
        "topics": topics, "score": score, "match": match, "name_sim": name_sim,
        "weak": weak, "meets_blocked": meets_blocked, "meets_review": meets_review, "why": why,
    }


def classify(hits: list[dict], query_name: str) -> str:
    """Pure decision function: raw OpenSanctions `results` -> overall status.

    Testable directly against captured API payloads (see tests/test_sanctions.py),
    no network needed. Rules: a Person-schema hit is dropped (it's not evidence about
    the company); a hard-topic hit only forces "blocked_evidence" when the API's own
    `match` is true AND the score is >= 0.85 AND the caption is actually similar to
    the query name (>= 80) AND the query name itself isn't too weak/generic to trust;
    a hard-topic hit that falls short of that but still scores >= 0.5 or has
    `match=true` is "review_evidence"; anything else with hits is "info"; no hits at
    all is "none".
    """
    query_norm = _normalise(query_name)
    evaluated = [_evaluate(r, query_norm) for r in hits if r.get("schema") != "Person"]
    if any(e["meets_blocked"] for e in evaluated):
        return "blocked_evidence"
    if any(e["meets_review"] for e in evaluated):
        return "review_evidence"
    if evaluated:
        return "info"
    return "none"


def _build_hits(raw_results: list[dict], query_name: str) -> list[dict]:
    query_norm = _normalise(query_name)
    hits = []
    for r in raw_results:
        if r.get("schema") == "Person":
            continue
        ev = _evaluate(r, query_norm)
        hit = {
            "caption": r.get("caption", ""),
            "score": ev["score"],
            "match": ev["match"],
            "schema": r.get("schema", ""),
            "datasets": (r.get("datasets") or [])[:3],
            "topics": ev["topics"],
            "url": ENTITY_URL.format(id=r["id"]) if r.get("id") else "",
            "origin": "opensanctions:match",
        }
        if ev["why"]:
            hit["why"] = ev["why"]
        hits.append(hit)
    return hits


@dataclass
class SanctionsScreen:
    status: str  # "blocked_evidence" | "review_evidence" | "info" | "none" | "skipped"
    hits: list[dict] = field(default_factory=list)
    note: str = ""
    # Why a `skipped` screen was skipped. "skipped" on its own collapsed three states
    # that a compliance decision has to tell apart:
    #
    #   ""               the screen ran: `status` is its answer
    #   "not_configured" no API key and no cached answer - this control was never
    #                    claimed, so its silence says nothing about this lead
    #   "degraded"       the control was configured and did not deliver: an outage, a
    #                    non-200, a 200 nobody could parse, or a cached answer too old
    #                    to stand for the present and no key to revalidate it
    #
    # `degraded` is a control that failed and therefore cannot clear a lead.
    # `not_configured` is an absent optional capability - the core do-not-engage
    # rule is the restricted-jurisdiction check, which is deterministic and runs without
    # any API key, so a deployment starting without one is not screening on nothing.
    reason: str = ""


# --- Disk cache --------------------------------------------------------------------
# The OpenSanctions match API is metered and this key hit its monthly quota during
# testing (429 "exceeded its rate limit for the month"). Caching every response to
# disk, keyed by exactly what was queried, means a rerun - or a deployment without a
# key at all - still sees the real measured data, not a guess. The cache directory is
# kept on purpose: it holds the evidence behind past decisions, not a throwaway artifact.

def _cache_key(name: str, iso2: str | None) -> str:
    return hashlib.sha256(f"{name}|{iso2 or ''}".encode()).hexdigest()


def _cache_path(name: str, iso2: str | None) -> Path:
    return CACHE_DIR / f"{_cache_key(name, iso2)}.json"


# A screen is a statement about *now*. A cached answer stays usable for a month and is
# then revalidated, because "OpenSanctions said no match" ages: designations are added
# daily. Measured before this existed: a cache entry stamped 2000-01-01 with a score-1.0
# exact hit still returned `blocked_evidence` and decided the lead, and a stale empty
# result still cleared one. Neither is a current screen.
CACHE_MAX_AGE_DAYS = 30


def _cache_age_days(cached: dict) -> float | None:
    """Age of a cached response in days, or None if it cannot be established.

    An unparseable or missing `fetched_at` returns None and is treated by the caller as
    stale: an entry that cannot say when it was fetched has not shown that it is current.
    """
    raw = cached.get("fetched_at")
    if not isinstance(raw, str):
        return None
    try:
        fetched = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=UTC)
    return (datetime.now(UTC) - fetched).total_seconds() / 86400.0


def _read_cache(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _write_cache(path: Path, name: str, iso2: str | None, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": {"name": name, "country": iso2},
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "api",
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def list_cache() -> list[dict]:
    """Metadata for every cached response, for `python -m leadscout.cli cache-list`."""
    if not CACHE_DIR.exists():
        return []
    rows = []
    for p in sorted(CACHE_DIR.glob("*.json")):
        data = _read_cache(p)
        if data is None:
            continue
        q = data.get("query", {})
        rows.append({
            "name": q.get("name", "?"), "country": q.get("country") or "",
            "fetched_at": data.get("fetched_at", ""), "source": data.get("source", ""),
            "n_results": len(data.get("results", [])), "file": p.name,
        })
    return rows


def _record_evidence(run: ProvenanceRun, company: str, results: list[dict], source_note: str,
                     observed_at: str | None = None) -> None:
    """Snapshot the raw OpenSanctions results (cached or live) as one evidence item.

    `observed_at` is when the data was ACQUIRED, not when it was read: served from the
    cache, `now()` would date a 2026-09-19 answer to today and the evidence trail would
    read as a fresh observation of old data.
    """
    raw = json.dumps(results, ensure_ascii=False).encode("utf-8")
    eid = run.next_evidence_id()
    snippet = f"{len(results)} result(s), {source_note}"[:200]
    ev = Evidence(
        id=eid, source_type="opensanctions", url=MATCH_URL,
        observed_at=observed_at or datetime.now(UTC).isoformat(),
        content_sha256=hashlib.sha256(raw).hexdigest(),
        strength="STRONG" if results else "WEAK", snippet=snippet,
        snapshot_path=f"out/provenance/snapshots/{run.run_id}/{eid}.json",
        family="corporate_identity",
    )
    run.record(ev, raw, stage="sanctions")


def _body_note(r) -> str:
    """A short, safe description of a response body for a degradation note."""
    try:
        return f"{r.text[:120]!r}"
    except Exception:  # noqa: BLE001 - a note must never be the thing that raises
        return "<unreadable>"


def _extract_results(r) -> list[dict] | None:
    """The `results` list from a match response, or None if the body is not one.

    Returns None - not `[]` - for every unusable shape, because the difference between
    "the watchlist returned nothing" and "the answer could not be read" is the whole
    point: the first is negative evidence, the second is no evidence at all.
    """
    try:
        body = r.json()
    except ValueError:                      # includes json.JSONDecodeError
        return None
    if not isinstance(body, dict):
        return None
    responses = body.get("responses")
    if not isinstance(responses, dict):
        return None
    q = responses.get("q")
    if not isinstance(q, dict):
        return None
    results = q.get("results")
    if not isinstance(results, list):
        return None
    return [item for item in results if isinstance(item, dict)]


def screen_sanctions(company: str, hq_country: str, run: ProvenanceRun | None = None) -> SanctionsScreen:
    """Query (or read the cache for) the OpenSanctions match API.

    Never raises: a missing key, a non-200 response, a network error, a 200 whose body
    this code cannot read, and a cached answer too old to stand for the present all come
    back as `status="skipped"` with the reason in `note`, so a screening outage degrades
    the lead (less evidence) instead of crashing the pipeline or inventing a clean
    result. `skipped` is a real state downstream, not a quiet one: compliance will not
    clear a lead on it.

    A cache hit is checked *before* the key check, so a deployment with no key at all
    still sees real data for every query this project has already made - but only while
    that data is younger than `CACHE_MAX_AGE_DAYS`. When `run` is given, the raw results
    actually used (cached or live) are recorded as evidence, dated when they were
    fetched rather than when they were read.
    """
    iso2 = _iso2(hq_country)
    cache_path = _cache_path(company, iso2)
    cached = _read_cache(cache_path)
    stale_note = ""
    if cached is not None:
        age = _cache_age_days(cached)
        if age is not None and age <= CACHE_MAX_AGE_DAYS:
            results = cached.get("results", [])
            hits = _build_hits(results, company)
            for h in hits:
                h["origin"] = "opensanctions:match(cache)"
            if run is not None:
                _record_evidence(run, company, results, f"cache, source={cached.get('source', 'cache')}",
                                 observed_at=cached.get("fetched_at"))
            return SanctionsScreen(
                status=classify(results, company), hits=hits,
                note=f"cache {cached.get('fetched_at', '?')} ({cached.get('source', 'cache')})",
            )
        # Too old to stand for the present, so it is revalidated below if that is
        # possible. It is not discarded from the note: "we hold an old answer and could
        # not refresh it" is a different state from "we have never asked".
        when = cached.get("fetched_at", "unknown date")
        stale_note = (f"the cached screen from {when} is "
                      f"{'of unknown age' if age is None else f'{age:.0f} days old'} "
                      f"(limit {CACHE_MAX_AGE_DAYS} days) and was not treated as current")

    if not settings.opensanctions_api_key:
        if stale_note:
            return SanctionsScreen(status="skipped", reason="degraded",
                                   note=f"{stale_note}; no API key to revalidate it")
        return SanctionsScreen(status="skipped", reason="not_configured",
                               note="OPENSANCTIONS_API_KEY is not set (see .env.example)")

    properties: dict = {"name": [company]}
    if iso2:
        properties["country"] = [iso2]
    payload = {"queries": {"q": {"schema": "Company", "properties": properties}}}
    headers = {"Authorization": f"ApiKey {settings.opensanctions_api_key}"}

    try:
        with httpx.Client(timeout=settings.http_timeout) as client:
            r = client.post(MATCH_URL, json=payload, headers=headers, params={"threshold": 0.5, "limit": 5})
    except httpx.HTTPError as e:
        return SanctionsScreen(status="skipped", reason="degraded",
                               note=f"OpenSanctions request failed: {type(e).__name__}: {e}")

    if r.status_code != 200:
        return SanctionsScreen(status="skipped", reason="degraded",
                               note=f"OpenSanctions {r.status_code}: {r.text[:200]}")

    results = _extract_results(r)
    if results is None:
        # 200 with a body this code cannot read is a FAILED measurement, and a failed
        # measurement is not a clean screen. Before this, `{"unexpected": 1}` and a body
        # missing `results` both coalesced to `[]` and were returned - and cached - as
        # `status="none"`, indistinguishable from a real no-match; a body that was not
        # JSON at all, or was a JSON list, raised straight out of a function documented
        # as never raising.
        return SanctionsScreen(status="skipped", reason="degraded",
                               note=f"OpenSanctions 200 with an unusable body: {_body_note(r)}")
    _write_cache(cache_path, company, iso2, results)
    if run is not None:
        _record_evidence(run, company, results, "live API call")
    return SanctionsScreen(status=classify(results, company), hits=_build_hits(results, company))
