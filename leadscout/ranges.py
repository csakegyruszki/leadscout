"""Authoritative cloud IP-range matcher (the infrastructure model "Detection order and
the range matcher", v0.2.2).

Published provider feeds (AWS, GCP, OCI - reusing `providers.footprint._RANGE_URLS`,
plus Azure Public Cloud Service Tags, resolved from the Microsoft download-details
page) are loaded into a local, most-specific-prefix-wins index; matching an IP is
then a pure local computation, never a per-host API call (the main invariant this
module exists to satisfy - see the module docstring in the infrastructure model).

A feed's generic superset entry (AWS "AMAZON") never overrides a more specific
service entry (AWS "S3", "CLOUDFRONT") for the same IP: the index keeps every
matching prefix and `RangeIndex.match()` picks the most specific (longest prefix
length) one, with a specific-service tiebreak over a same-length generic entry.

Published feeds are incomplete (AWS excludes some services and BYOIP addresses,
the infrastructure model), so `match()` returning None is never evidence of "no cloud" -
callers (`infra.py`) must not treat it that way.
"""
from __future__ import annotations

import ipaddress
import re

import httpx

from .config import settings
from .providers._cache import _cache_get, _cache_put

CACHE_DIR = settings.out_dir / "cache" / "ranges"
FEEDS_TTL_S = 24 * 3600

AWS_URL = "https://ip-ranges.amazonaws.com/ip-ranges.json"
GCP_URL = "https://www.gstatic.com/ipranges/cloud.json"
OCI_URL = "https://docs.oracle.com/en-us/iaas/tools/public_ip_ranges.json"
AZURE_DETAILS_URL = "https://www.microsoft.com/en-us/download/details.aspx?id=56519"
_AZURE_JSON_RE = re.compile(
    r"https://download\.microsoft\.com/download/[^\"'\s]+ServiceTags_Public_\d+\.json")

# AWS `service` values that name a CDN/edge/anycast-DNS product rather than a
# compute/workload range (the infrastructure model classification rule 1).
_AWS_EDGE_SERVICES = frozenset({"CLOUDFRONT", "GLOBALACCELERATOR", "ROUTE53", "ROUTE53_HEALTHCHECKS"})
_AWS_GENERIC_SERVICES = frozenset({"AMAZON", "EC2"})
_GCP_GENERIC_SERVICES = frozenset({"Google Cloud", "GOOGLE"})


class _Entry:
    __slots__ = ("network", "provider", "service", "region", "source_feed", "source_version", "network_role")

    def __init__(self, network, provider, service, region, source_feed, source_version, network_role):
        self.network = network
        self.provider = provider
        self.service = service
        self.region = region
        self.source_feed = source_feed
        self.source_version = source_version
        self.network_role = network_role

    @property
    def prefixlen(self) -> int:
        return self.network.prefixlen

    @property
    def is_generic(self) -> bool:
        if self.provider == "AWS":
            return self.service in _AWS_GENERIC_SERVICES
        if self.provider == "GCP":
            return self.service in _GCP_GENERIC_SERVICES
        if self.provider == "Azure":
            return "." not in self.service  # e.g. "AzureCloud" vs "AzureFrontDoor.Frontend"
        return False


def _fetch_json(client: httpx.Client, url: str, cache_path, ttl_s: float = FEEDS_TTL_S):
    cached = _cache_get(cache_path, ttl_s=ttl_s)
    if cached is not None:
        return cached
    r = client.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    _cache_put(cache_path, data)
    return data


def _resolve_azure_json_url(client: httpx.Client) -> str | None:
    cache_path = CACHE_DIR / "azure_json_url.json"
    cached = _cache_get(cache_path, ttl_s=FEEDS_TTL_S)
    if cached is not None:
        return cached.get("url")
    r = client.get(AZURE_DETAILS_URL, timeout=20)
    r.raise_for_status()
    m = _AZURE_JSON_RE.search(r.text)
    url = m.group(0) if m else None
    _cache_put(cache_path, {"url": url})
    return url


def _load_aws(client: httpx.Client) -> list[_Entry]:
    data = _fetch_json(client, AWS_URL, CACHE_DIR / "aws.json")
    version = data.get("createDate")
    out = []
    for p in data.get("prefixes", []):
        try:
            net = ipaddress.ip_network(p["ip_prefix"], strict=False)
        except (KeyError, ValueError):
            continue
        service = p.get("service", "AMAZON")
        role = "EDGE" if service in _AWS_EDGE_SERVICES else "PUBLIC_CLOUD_COMPUTE" \
            if service in _AWS_GENERIC_SERVICES else "PUBLIC_CLOUD_SERVICE"
        out.append(_Entry(net, "AWS", service, p.get("region"), "aws-ip-ranges", version, role))
    return out


def _load_gcp(client: httpx.Client) -> list[_Entry]:
    data = _fetch_json(client, GCP_URL, CACHE_DIR / "gcp.json")
    version = data.get("creationTime") or str(data.get("syncToken", ""))
    out = []
    for p in data.get("prefixes", []):
        prefix = p.get("ipv4Prefix") or p.get("ipv6Prefix")
        if not prefix:
            continue
        try:
            net = ipaddress.ip_network(prefix, strict=False)
        except ValueError:
            continue
        service = p.get("service", "Google Cloud")
        role = "PUBLIC_CLOUD_COMPUTE" if service in _GCP_GENERIC_SERVICES else "PUBLIC_CLOUD_SERVICE"
        out.append(_Entry(net, "GCP", service, p.get("scope"), "gcp-cloud-ranges", version, role))
    return out


def _load_oci(client: httpx.Client) -> list[_Entry]:
    data = _fetch_json(client, OCI_URL, CACHE_DIR / "oci.json")
    version = str(data.get("last_updated_timestamp", ""))
    out = []
    for region in data.get("regions", []):
        for c in region.get("cidrs", []):
            try:
                net = ipaddress.ip_network(c["cidr"], strict=False)
            except (KeyError, ValueError):
                continue
            tags = c.get("tags", []) or ["OCI"]
            out.append(_Entry(net, "OCI", tags[0], region.get("region"), "oci-ip-ranges", version,
                              "PUBLIC_CLOUD_SERVICE" if len(tags) > 1 or tags[0] != "OCI"
                              else "PUBLIC_CLOUD_COMPUTE"))
    return out


def _load_azure(client: httpx.Client) -> list[_Entry]:
    url = _resolve_azure_json_url(client)
    if not url:
        return []
    fname = url.rsplit("/", 1)[-1]
    data = _fetch_json(client, url, CACHE_DIR / fname)
    version = str(data.get("changeNumber", ""))
    out = []
    for v in data.get("values", []):
        name = v.get("name", "")
        props = v.get("properties", {})
        role = "EDGE" if name.startswith("AzureFrontDoor.") else \
            "PUBLIC_CLOUD_COMPUTE" if name in ("AzureCloud", "AzurePublicCloud") else "PUBLIC_CLOUD_SERVICE"
        for prefix in props.get("addressPrefixes", []):
            try:
                net = ipaddress.ip_network(prefix, strict=False)
            except ValueError:
                continue
            out.append(_Entry(net, "Azure", name, props.get("region") or None,
                              "azure-service-tags", version, role))
    return out


class RangeIndex:
    """Loaded once per run (memoised by the caller, e.g. a `_Budget`), matched
    locally per IP - no per-host network call."""

    def __init__(self, client: httpx.Client):
        self.entries: list[_Entry] = []
        for loader in (_load_aws, _load_gcp, _load_oci, _load_azure):
            try:
                self.entries += loader(client)
            except (httpx.HTTPError, ValueError, KeyError):
                continue  # a stale/unreachable feed degrades to "no match from this feed", never a crash

    def match(self, ip: str):
        """Returns a `models.RangeMatch` for the most specific matching entry, or
        None. Ties on prefix length prefer a specific service over a generic one
        (the "generic superset never overrides a specific entry" rule) -
        deterministic beyond that by (provider, service) so results are stable."""
        from .models import RangeMatch
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return None
        candidates = [e for e in self.entries if addr in e.network]
        if not candidates:
            return None
        candidates.sort(key=lambda e: (-e.prefixlen, e.is_generic, e.provider, e.service))
        best = candidates[0]
        return RangeMatch(
            provider=best.provider, service=best.service, region=best.region,
            source_feed=best.source_feed, source_version=best.source_version,
            network_role=best.network_role,
        )
