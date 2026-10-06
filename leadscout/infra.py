"""Deterministic infrastructure-footprint classifier (the infrastructure model, PART A
item 5 - scope cut). Builds `models.InfrastructureFootprint` items from
the footprint provider's own already-collected evidence and per-host observations
- makes no HTTP calls of its own, and is NOT a new provider module.

Scope-cut reduction from the original PART A draft: no asset-relationship
taxonomy, no control-model/hosting-model axes, no PRIVATE_CLOUD conjunction (see
the TODO below - deferred to a later part). Categories: PUBLIC_CLOUD, PRIVATE_CLOUD
(not produced yet), MANAGED_HOSTING, EDGE, UNKNOWN.
"""
from __future__ import annotations

from .models import Evidence, InfrastructureFootprint, ProviderResult

_CPANEL_REASON = "crt.sh subdomain names show a cPanel trace (cpanel./webdisk./cpcalendars.)"


def _strength_to_confidence(strength: str) -> str:
    if strength in ("STRONG", "DIRECT"):
        return "HIGH"
    if strength == "MEDIUM":
        return "MEDIUM"
    return "LOW"


def _from_network_footprint_evidence(evidence: list[Evidence]) -> list[InfrastructureFootprint]:
    """Range match / CNAME-suffix / RIPE-holder hits that already reached
    footprint.py's own (closed, big-4 + Cloudflare) provider naming -> PUBLIC_CLOUD
    (or EDGE for `edge_delivery`). "Range match -> PUBLIC_CLOUD (or EDGE)" (scope-cut
    instruction) - footprint.py's `_ip_in_ranges` already goes through
    `ranges.RangeIndex` (PART A item 3), so a `network_footprint` Evidence item
    naming AWS/GCP/OCI/Azure here IS the range-matcher's own signal, not a guess."""
    seen: set[tuple[str, str | None]] = set()
    out: list[InfrastructureFootprint] = []
    for ev in evidence:
        if ev.family == "network_footprint" and ev.provider:
            if (ev.scope or "") == "edge":
                # An edge-scoped observation (CloudFront-style service tag, CDN host) is edge delivery,
                # never workload hosting - even when the range match named a public-cloud provider.
                key = ("EDGE", ev.provider)
            else:
                key = ("PUBLIC_CLOUD", ev.provider)
            if key in seen:
                continue
            seen.add(key)
            out.append(InfrastructureFootprint(
                scope=ev.scope or "unknown", category=key[0], provider=ev.provider,
                confidence=_strength_to_confidence(ev.strength), evidence_ids=[ev.id],
                reasoning=[f"footprint network_footprint evidence ({ev.id}) names {ev.provider} "
                          "via CNAME-suffix / IP-range / RIPE-holder match"],
            ))
        elif ev.family == "edge_delivery":
            key = ("EDGE", ev.provider or "CDN")
            if key in seen:
                continue
            seen.add(key)
            out.append(InfrastructureFootprint(
                scope="edge", category="EDGE", provider=ev.provider,
                confidence="LOW", evidence_ids=[ev.id],
                reasoning=[f"footprint edge_delivery evidence ({ev.id}) - CDN/edge only, "
                          "the workload behind it is unestablished"],
            ))
    return out


def build_footprints(footprint_result: ProviderResult) -> list[InfrastructureFootprint]:
    """`footprint_result` is the `ProviderResult` `providers/footprint.py`'s `run()`
    returned (research.py passes it straight through). Never raises: a degraded
    footprint run (no `_holder_records`/`_cpanel_trace` attached) simply yields
    whatever `evidence` it did collect, which may be an empty list.

    PART B fix #1: `footprint.py`'s `run()` now records the cPanel trace and each
    distinct RIPEstat holder as its own Evidence item (never bare dicts/flags with
    nothing to cite) - every footprint built here carries at least one evidence id.
    A cPanel trace and the address holder it shares a host with are ONE fact (a
    managed-hosting provider's own reseller infrastructure, e.g. Artizan/WebSupport),
    not two separate footprints - merged into a single MANAGED_HOSTING item citing
    both evidence ids when both are present."""
    footprints = _from_network_footprint_evidence(footprint_result.evidence)

    cpanel_trace = getattr(footprint_result, "_cpanel_trace", False)
    cpanel_evidence_id = getattr(footprint_result, "_cpanel_evidence_id", None)
    holder_records = getattr(footprint_result, "_holder_records", [])
    seen_holders: set[str] = set()

    if cpanel_trace and holder_records:
        primary = holder_records[0]
        holder = primary.get("holder")
        seen_holders.add(holder)
        evidence_ids = [eid for eid in (cpanel_evidence_id, primary.get("evidence_id")) if eid]
        footprints.append(InfrastructureFootprint(
            scope="website", category="MANAGED_HOSTING", provider=None, network_holder=holder,
            confidence="MEDIUM", evidence_ids=evidence_ids,
            reasoning=[
                _CPANEL_REASON,
                f"address ({primary.get('ip')}) registered to {holder}; operator not established "
                "- no authoritative range match (the infrastructure model main invariant)",
            ],
        ))
    elif cpanel_trace:
        footprints.append(InfrastructureFootprint(
            scope="website", category="MANAGED_HOSTING", provider=None,
            confidence="MEDIUM", evidence_ids=[cpanel_evidence_id] if cpanel_evidence_id else [],
            reasoning=[_CPANEL_REASON],
        ))

    for rec in holder_records:
        holder = rec.get("holder")
        if not holder or holder in seen_holders:
            continue
        seen_holders.add(holder)
        # A RIPEstat holder is a REGISTRATION fact, never an operator/provider claim
        # (the infrastructure model main invariant: "network holder != operator") - the
        # semantic-constraint fix: `provider` stays None here, the raw
        # holder goes only into `network_holder`, and the reasoning text is always
        # "address registered to X; operator not established", never "hosted by"/
        # "runs on X" (which would silently assert an operator/provider claim this
        # signal alone can never support).
        footprints.append(InfrastructureFootprint(
            scope="unknown", category="UNKNOWN", provider=None, network_holder=holder,
            confidence="LOW", evidence_ids=[rec["evidence_id"]] if rec.get("evidence_id") else [],
            reasoning=[
                f"address ({rec.get('ip')}) registered to {holder}; operator not established "
                "- no authoritative range match (the infrastructure model main invariant)",
                # TODO / future work (deferred past PART A's scope cut): a PRIVATE_CLOUD
                # conjunction over first-party-linked engineering evidence (own cloud
                # naming, IaaS/PaaS platform tech, tenant/quota operators, multi-DC
                # context) - see the infrastructure model's classification rule 5.
            ],
        ))
    return footprints
