"""Passive network footprint: crt.sh subdomains -> Cloudflare DoH resolution ->
CNAME-suffix / official-IP-range / RIPEstat-holder mapping -> aggregation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import httpx

from ..config import settings
from ..models import Evidence, ProviderResult
from ._base import UA, fallback_evidence_id, snapshot_path
from ._cache import _cache_get, _cache_put

# --- Design notes -----------------------------------------------------------------
# Never claims a provider from a bare hostname alone: a CNAME suffix, an official
# range-file hit, or a RIPEstat holder name is the actual signal. Measured
# 2026-09-19 (raw/dns-rdap-ripe__*.json): zapier.com -> ASN 16509 AMAZON-02 (AWS);
# snapp.ir -> ArvanCloud (edge-only CDN, Iran); masterplast.hu -> Rackforest (no
# cloud match at all); a CDN-fronted site -> Cloudflare (edge-only).
#
# Cloud evidence is aggregated WITHIN the family: at most ONE `network_footprint`
# Evidence per real provider (never one per host - three S3-backed hostnames are one
# fact, "uses AWS for N services", not three). A provider whose first-party hosts are
# ALL edge-scope becomes `edge_delivery` WEAK instead, and never claims AWS/Azure/
# GCP/OCI as a *workload* user from edge alone.

# 15s measured too tight (a real run took ~10-20s wall time already just for crt.sh,
# see the design notes' footprint smoke-test note) - raised so a slow-but-live
# crt.sh query isn't mistaken for a dead one.
CRTSH_TIMEOUT = 25.0
CACHE_DIR = settings.out_dir / "cache"
CRTSH_CACHE = CACHE_DIR / "crtsh"
RANGES_CACHE = CACHE_DIR / "ranges"
RANGES_TTL_S = 24 * 3600
# A genuine empty crt.sh result (no certificates ever issued for the domain) is
# indistinguishable from a query keyed on the wrong host (e.g. "www.<domain>"
# instead of the bare registrable domain - see leadscout/util.py) or a transient
# rate-limit/timeout - never cache either of those as "no subdomains" forever.
# A real, non-empty result is cached with a 7-day TTL, not forever: certificate
# transparency logs keep gaining new entries.
CRTSH_CACHE_TTL_S = 7 * 24 * 3600
MAX_SUBDOMAINS = 30
_PREFERRED = ("api", "app", "www", "files", "assets", "cdn", "status", "auth", "docs", "data", "static")

_RANGE_URLS = {
    "AWS": "https://ip-ranges.amazonaws.com/ip-ranges.json",
    "GCP": "https://www.gstatic.com/ipranges/cloud.json",
    "OCI": "https://docs.oracle.com/en-us/iaas/tools/public_ip_ranges.json",
}
# (substring in the resolved CNAME, provider-or-None, service, scope). First match wins;
# specific patterns are listed before their generic catch-all.
_CNAME_RULES = (
    (".cloudfront.net", "AWS", "CloudFront", "edge"),
    (".s3.amazonaws.com", "AWS", "S3", "storage"),
    ("execute-api.", "AWS", "API Gateway", "workload"),
    (".elb.amazonaws.com", "AWS", "ELB", "workload"),
    (".elasticbeanstalk.com", "AWS", "Elastic Beanstalk", "workload"),
    (".amazonaws.com", "AWS", "AWS", "workload"),
    (".azurewebsites.net", "Azure", "App Service", "workload"),
    (".azure-api.net", "Azure", "API Management", "workload"),
    (".cloudapp.azure.com", "Azure", "Cloud Service", "workload"),
    (".azureedge.net", "Azure", "Front Door/CDN", "edge"),
    (".azurefd.net", "Azure", "Front Door", "edge"),
    (".blob.core.windows.net", "Azure", "Blob Storage", "storage"),
    (".run.app", "GCP", "Cloud Run", "workload"),
    (".appspot.com", "GCP", "App Engine", "workload"),
    (".storage.googleapis.com", "GCP", "Cloud Storage", "storage"),
    (".googleusercontent.com", "GCP", "Storage", "storage"),
    (".oraclecloud.com", "OCI", "OCI", "workload"),
    (".cloudflare.net", "Cloudflare", "CDN", "edge"),
    (".fastly.net", None, "Fastly", "edge"),
    (".akamaiedge.net", None, "Akamai", "edge"),
    (".vercel.app", None, "Vercel", "edge"),
    (".netlify.app", None, "Netlify", "edge"),
)
_RIPE_PROVIDER_HOLDERS = (("AMAZON", "AWS"), ("MICROSOFT", "Azure"), ("GOOGLE", "GCP"), ("ORACLE", "OCI"))
_RIPE_EDGE_HOLDERS = ("CLOUDFLARE", "FASTLY", "AKAMAI")


def crtsh_subdomains(domain: str, client: httpx.Client) -> list[str]:
    """Subdomains from crt.sh, deduped, no wildcards, capped at MAX_SUBDOMAINS with
    preferred infra-ish names first. `domain` should already be the bare
    registrable domain (leading "www." stripped, see leadscout/util.py) - crt.sh's
    own certificate-transparency index is keyed on the domain actually named in
    issued certificates, and a "www.<domain>" query returns a near-empty result
    for a normal site (whose certs cover the apex/wildcard, not "www." alone).

    Cached to disk with a CRTSH_CACHE_TTL_S (7-day) TTL - crt.sh is slow/rate
    limited, not volatile, but a 7-day-old answer can still miss newly issued
    certs. A genuinely EMPTY result is never cached (nor is a timeout/error, which
    never reaches this far) - an empty crt.sh response is much more often a
    transient rate-limit/glitch or a wrong query than a real "zero certificates
    ever issued", and caching it "forever" (the previous behaviour, no TTL at all)
    turned one bad query into a permanent false negative.
    """
    cache_path = CRTSH_CACHE / f"{domain}.json"
    cached = _cache_get(cache_path, ttl_s=CRTSH_CACHE_TTL_S)
    if cached is not None:
        names = cached
    else:
        r = client.get("https://crt.sh/", params={"q": f"%.{domain}", "output": "json"}, timeout=CRTSH_TIMEOUT)
        r.raise_for_status()
        rows = r.json() if r.text.strip() else []
        names = sorted({n for row in rows for n in str(row.get("name_value", "")).split("\n")})
        if names:
            _cache_put(cache_path, names)
    subs = {n.strip().lower() for n in names if n and not n.startswith("*.") and n.endswith(domain)}
    subs.discard(domain)
    preferred = [s for s in subs if any(s.split(".")[0] == p for p in _PREFERRED)]
    rest = sorted(subs - set(preferred))
    return (sorted(preferred) + rest)[:MAX_SUBDOMAINS]


class _Budget:
    """Per-run HTTP call ceiling + memoisation for DoH/RIPEstat/range-file lookups
    (REVIEW-6bB-verified.md #5). Without this, N first-party hosts sharing one CNAME
    target or resolving to the same IP would re-query DoH/RIPEstat/the range files N
    times for an identical answer, and a large subdomain list had no ceiling at all."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.calls = 0
        self.cname_memo: dict[str, dict] = {}
        self.a_memo: dict[str, dict] = {}
        self.ripe_memo: dict[str, tuple[str | None, bool]] = {}
        self.ranges_memo: dict[str, list] = {}
        # Fix #1 (v0.2 PART B): the raw RIPEstat prefix-overview response bytes for
        # each IP actually looked up - the snapshot content for the network_holder
        # Evidence `run()` builds below, so "address registered to X" cites the
        # exact bytes that observation came from, not just the parsed holder string.
        self.ripe_raw: dict[str, bytes] = {}

    def allow(self) -> bool:
        return self.calls < self.max_calls

    def spend(self, n: int = 1) -> None:
        self.calls += n


def _doh(name: str, rtype: str, client: httpx.Client, budget: _Budget | None = None) -> dict:
    memo = None
    if budget is not None:
        memo = budget.cname_memo if rtype == "CNAME" else budget.a_memo
        if name in memo:
            return memo[name]
        if not budget.allow():
            return {}
    r = client.get("https://cloudflare-dns.com/dns-query", params={"name": name, "type": rtype},
                    headers={"accept": "application/dns-json"}, timeout=10)
    if budget is not None:
        budget.spend()
    result = r.json() if r.status_code == 200 else {}
    if memo is not None:
        memo[name] = result
    return result


def _classify_cname(cname: str) -> tuple[str | None, str, str] | None:
    low = cname.lower()
    for pattern, provider, service, scope in _CNAME_RULES:
        if pattern in low:
            return provider, service, scope
    return None


def _range_index(client: httpx.Client, budget: _Budget | None = None):
    """Shared `ranges.RangeIndex` (AWS/GCP/OCI/Azure published feeds, most-specific-
    prefix-wins) - PART A item 3: footprint's own AWS/GCP/OCI range check now goes
    through the same authoritative matcher `network_observation.py`/`infra.py` use,
    instead of the smaller AWS/GCP/OCI-only prefix-list this module used to build by
    hand. Memoised on the budget (or module-level when there is none) so a run with
    many hosts builds the index once."""
    from .. import ranges as ranges_mod
    cache = budget.ranges_memo if budget is not None else _NO_BUDGET_RANGES_MEMO
    if "_index" not in cache:
        cache["_index"] = ranges_mod.RangeIndex(client)
    return cache["_index"]


_NO_BUDGET_RANGES_MEMO: dict = {}


def _ip_in_ranges(ip: str, provider: str, client: httpx.Client, budget: _Budget | None = None) -> bool:
    try:
        idx = _range_index(client, budget)
        match = idx.match(ip)
    except (httpx.HTTPError, ValueError):
        return False
    return bool(match and match.provider == provider)


def _ripe_holder_provider(
    ip: str, client: httpx.Client, budget: _Budget | None = None,
) -> tuple[str | None, bool, str | None]:
    """Returns (provider-or-None, is_edge_holder, raw_holder-or-None). Memoised by
    IP - several first-party hosts often share one IP/CDN edge and would otherwise
    re-query RIPEstat for an answer we already have (REVIEW-6bB-verified.md #5).

    The third element (added PART A item 4, scope cut) is the raw
    RIPEstat holder string even when it does NOT map to one of the four named
    clouds or a recognised edge vendor - `infra.py` uses it as an open-string
    network holder (e.g. "RACKFORCET-AS", "HETZNER-AS") for the "no authoritative
    range match" case (the infrastructure model: "an arbitrary network holder ... is
    recorded as network_holder only"). This module's own Evidence generation
    (`classify_host`/`_aggregate`) is unaffected - it still only ever names a
    provider from the closed AWS/Azure/GCP/OCI/Cloudflare set it always did."""
    if budget is not None and ip in budget.ripe_memo:
        return budget.ripe_memo[ip]
    if budget is not None and not budget.allow():
        return None, False, None
    try:
        r = client.get("https://stat.ripe.net/data/prefix-overview/data.json",
                        params={"resource": ip}, timeout=10)
        if budget is not None:
            budget.spend()
            budget.ripe_raw[ip] = r.content if r.status_code == 200 else b""
        asns = r.json().get("data", {}).get("asns", [{}]) if r.status_code == 200 else [{}]
        holder_raw = str(asns[0].get("holder", "")) or None
        holder = holder_raw.upper() if holder_raw else ""
    except (httpx.HTTPError, KeyError, IndexError):
        result = (None, False, None)
        if budget is not None:
            budget.ripe_memo[ip] = result
        return result
    result: tuple[str | None, bool, str | None] = (None, False, holder_raw)
    for token, provider in _RIPE_PROVIDER_HOLDERS:
        if token in holder:
            result = (provider, False, holder_raw)
            break
    else:
        if any(token in holder for token in _RIPE_EDGE_HOLDERS):
            result = (None, True, holder_raw)
    if budget is not None:
        budget.ripe_memo[ip] = result
    return result


def classify_host(host: str, client: httpx.Client, budget: _Budget | None = None) -> tuple[str | None, str, str] | None:
    """One host -> (provider-or-None, service, scope), or None if unclassifiable."""
    cname_resp = _doh(host, "CNAME", client, budget)
    for ans in cname_resp.get("Answer", []):
        hit = _classify_cname(str(ans.get("data", "")))
        if hit:
            return hit
    a_resp = _doh(host, "A", client, budget)
    ips = [a["data"] for a in a_resp.get("Answer", []) if a.get("type") == 1]
    for ip in ips:
        for provider in ("AWS", "GCP", "OCI"):
            if _ip_in_ranges(ip, provider, client, budget):
                return provider, provider, "workload"
        provider, is_edge, _raw_holder = _ripe_holder_provider(ip, client, budget)
        if provider:
            return provider, provider, "workload"
        if is_edge:
            return None, "CDN", "edge"
    return None


def _aggregate(observations: list[tuple[str, str | None, str, str]]) -> list[Evidence]:
    """observations: (host, provider, service, scope). One Evidence per provider
    group - network_footprint if any non-edge scope present (MEDIUM, STRONG at
    >=3 distinct hosts), else edge_delivery WEAK."""
    by_provider: dict[str, list[tuple[str, str, str]]] = {}
    for host, provider, service, scope in observations:
        key = provider or "other"
        by_provider.setdefault(key, []).append((host, service, scope))
    out: list[Evidence] = []
    for provider, obs in by_provider.items():
        hosts = sorted({h for h, _, _ in obs})
        services = sorted({s for _, s, _ in obs})
        non_edge = [o for o in obs if o[2] != "edge"]
        provider_field = provider if provider in ("AWS", "Azure", "GCP", "OCI", "Cloudflare") else "other"
        if non_edge:
            strength = "STRONG" if len(hosts) >= 3 else "MEDIUM"
            family = "network_footprint"
            # Deterministic mixed-scope precedence - never let an edge-only host's
            # scope, or iteration order, decide the aggregate's scope: workload >
            # storage > edge (edge already excluded from `non_edge`) (REVIEW-6bB-
            # verified.md #4).
            non_edge_scopes = {o[2] for o in non_edge}
            if "workload" in non_edge_scopes:
                scope_field = "workload"
            elif "storage" in non_edge_scopes:
                scope_field = "storage"
            else:
                scope_field = sorted(non_edge_scopes)[0]
        else:
            strength, family, scope_field = "WEAK", "edge_delivery", "edge"
        snippet = f"{provider_field}: {len(hosts)} host(s), services={services}, scope={scope_field}"
        ev = Evidence(
            id="", source_type="footprint", url=f"https://{hosts[0]}",
            observed_at=datetime.now(UTC).isoformat(),
            content_sha256="", strength=strength, snippet=snippet[:200], snapshot_path="",
            family=family, provider=provider_field, scope=scope_field, freshness="current",
            origin="provider:footprint",
        )
        # Full detail (all hosts/services/observations, not just the 200-char
        # snippet) goes into the snapshot JSON, not just the display snippet
        # (REVIEW-6bB-verified.md #4) - stashed here and consumed by `_finalise`.
        ev._detail = {  # type: ignore[attr-defined]
            "provider": provider_field, "hosts": hosts, "distinct_services": services,
            "scope": scope_field,
            "observations": [{"host": h, "service": s, "scope": sc} for h, s, sc in obs],
        }
        out.append(ev)
    return out


def _domain_age_evidence(domain: str, client: httpx.Client) -> Evidence | None:
    r = client.get(f"https://rdap.org/domain/{domain}", timeout=10)
    if r.status_code != 200:
        return None
    events = r.json().get("events", [])
    reg = next((e.get("eventDate") for e in events if e.get("eventAction") == "registration"), None)
    if not reg:
        return None
    try:
        age_days = (datetime.now(UTC) - datetime.fromisoformat(reg.replace("Z", "+00:00"))).days
    except ValueError:
        return None
    return Evidence(
        id="", source_type="footprint", url=f"https://rdap.org/domain/{domain}",
        observed_at=datetime.now(UTC).isoformat(), content_sha256="", strength="WEAK",
        snippet=f"domain registered {reg}, age {age_days} days"[:200], snapshot_path="",
        family="corporate_identity", freshness="current", origin="provider:footprint",
    )


def _finalise(ev: Evidence, run, raw: bytes | None = None) -> Evidence:
    """`raw` lets a caller hand over the actual observed bytes (e.g. RIPEstat's raw
    JSON response) as the snapshot content, instead of falling back to the
    `_detail`/snippet-derived bytes every other footprint Evidence uses (Fix #1)."""
    eid = fallback_evidence_id("footprint", run, id(ev))
    if raw is None:
        detail = getattr(ev, "_detail", None)
        raw = json.dumps(detail, sort_keys=True).encode("utf-8") if detail is not None else ev.snippet.encode("utf-8")
    ev.id = eid
    ev.content_sha256 = hashlib.sha256(raw).hexdigest()
    ev.snapshot_path = snapshot_path(run, eid, "txt")
    if run is not None:
        run.record(ev, raw, stage="footprint")
    return ev


def _holder_evidence(ip: str, holder: str, raw: bytes, run) -> Evidence:
    """Fix #1: the RIPEstat holder observation itself, recorded as `network_footprint`
    MEDIUM Evidence (registration fact only - infra.py still never promotes this to a
    `provider` claim, see its own docstring) - `infra.py`'s UNKNOWN/MANAGED_HOSTING
    footprints now cite this id instead of shipping with an empty `evidence_ids`."""
    ev = Evidence(
        id="", source_type="footprint",
        url=f"https://stat.ripe.net/data/prefix-overview/data.json?resource={ip}",
        observed_at=datetime.now(UTC).isoformat(), content_sha256="", strength="MEDIUM",
        snippet=f"address {ip} registered to {holder}"[:200], snapshot_path="",
        family="network_footprint", provider=None, scope="unknown", freshness="current",
        origin="provider:footprint",
    )
    return _finalise(ev, run, raw=raw or ev.snippet.encode("utf-8"))


def _cpanel_evidence(domain: str, subdomains: list[str], run) -> Evidence:
    """Fix #1: the crt.sh subdomain-name observation the cPanel trace is based on
    (cpanel./webdisk./cpcalendars. prefixes) - recorded so infra.py's MANAGED_HOSTING
    footprint has an evidence id to cite instead of an empty list."""
    hits = sorted(s for s in subdomains if s.split(".")[0] in ("cpanel", "webdisk", "cpcalendars"))
    ev = Evidence(
        id="", source_type="footprint", url=f"https://crt.sh/?q=%.{domain}&output=json",
        observed_at=datetime.now(UTC).isoformat(), content_sha256="", strength="MEDIUM",
        snippet=f"crt.sh subdomain name(s) show a cPanel trace: {hits}"[:200], snapshot_path="",
        family="network_footprint", provider=None, scope="unknown", freshness="current",
        origin="provider:footprint",
    )
    raw = json.dumps(hits, sort_keys=True).encode("utf-8")
    return _finalise(ev, run, raw=raw)


def run(domain: str, run=None, max_calls: int | None = None) -> ProviderResult:
    """`domain` is the bare registrable domain (netloc without scheme/path).

    `max_calls` overrides `settings.max_http_calls` (default 40) - a hard ceiling on
    the total HTTP requests this single provider run may make, on top of the
    DoH/RIPEstat/range-file memoisation in `_Budget` (REVIEW-6bB-verified.md #5).
    Once the ceiling is hit, remaining subdomains are left unclassified rather than
    keeping their evidence out of the run entirely - a partial, budget-limited view
    is still `status="ok"`, not `degraded`."""
    ceiling = max_calls if max_calls is not None else getattr(settings, "max_http_calls", 40)
    budget = _Budget(ceiling)
    truncated = False
    try:
        with httpx.Client(timeout=settings.http_timeout, headers={"User-Agent": UA},
                          follow_redirects=True) as client:
            subdomains = crtsh_subdomains(domain, client)
            budget.spend()
            observations: list[tuple[str, str | None, str, str]] = []
            checked = 0
            for host in subdomains:
                if not budget.allow():
                    truncated = True
                    break
                hit = classify_host(host, client, budget)
                checked += 1
                if hit:
                    provider, service, scope = hit
                    observations.append((host, provider, service, scope))
            evidence = [_finalise(e, run) for e in _aggregate(observations)] if observations else []
            age_ev = _domain_age_evidence(domain, client) if budget.allow() else None
            if age_ev:
                budget.spend()
                evidence.append(_finalise(age_ev, run))
    except httpx.TimeoutException:
        return ProviderResult(provider_name="footprint", status="degraded", note="crt.sh timeout",
                               calls=budget.calls)
    except Exception as e:  # noqa: BLE001 - a malformed/unreachable response must degrade, never crash
        return ProviderResult(provider_name="footprint", status="degraded",
                               note=f"{type(e).__name__}: {e}", calls=budget.calls)

    # Fix #1 (v0.2 PART B): a holder/cPanel observation is now Evidence, not just a
    # stashed dict/bool - infra.py's UNKNOWN/MANAGED_HOSTING footprints need a real
    # `evidence_ids` entry, not an empty list. One Evidence per DISTINCT holder
    # string (several IPs sharing one holder is one fact, not N).
    holder_records: list[dict] = []
    seen_holders: set[str] = set()
    for ip, (provider, is_edge, holder) in budget.ripe_memo.items():
        if provider or is_edge or not holder or holder in seen_holders:
            continue
        seen_holders.add(holder)
        ev = _holder_evidence(ip, holder, budget.ripe_raw.get(ip, b""), run)
        evidence.append(ev)
        holder_records.append({"ip": ip, "holder": holder, "evidence_id": ev.id})

    cpanel_trace = any(s.split(".")[0] in ("cpanel", "webdisk", "cpcalendars") for s in subdomains)
    cpanel_evidence_id = None
    if cpanel_trace:
        cpanel_ev = _cpanel_evidence(domain, subdomains, run)
        evidence.append(cpanel_ev)
        cpanel_evidence_id = cpanel_ev.id

    note = f"{checked}/{len(subdomains)} subdomain(s) checked, {len(evidence)} evidence item(s)"
    if truncated:
        note += f" (MAX_HTTP_CALLS={ceiling} reached, {len(subdomains) - checked} subdomain(s) skipped)"
    result = ProviderResult(provider_name="footprint", status="ok", evidence=evidence, note=note, calls=budget.calls)
    # PART A item 4 (scope cut): stashed for infra.py, same pattern as
    # github.py's `_repos` - no new network calls, just reads what classify_host's
    # own RIPEstat lookups already cached on the budget (`_ripe_holder_provider`'s
    # raw holder, third tuple element) plus a crt.sh-name cPanel trace. PART B: each
    # record/flag now also carries the Evidence id `run()` just built for it above.
    result._holder_records = holder_records  # type: ignore[attr-defined]
    result._cpanel_trace = cpanel_trace  # type: ignore[attr-defined]
    result._cpanel_evidence_id = cpanel_evidence_id  # type: ignore[attr-defined]
    return result
