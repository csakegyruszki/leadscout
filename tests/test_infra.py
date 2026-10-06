"""infra.build_footprints: turns footprint.py's own evidence + per-host side
channels (`_holder_records`/`_cpanel_trace`) into `InfrastructureFootprint`s -
offline, no network calls (infra.py makes none of its own).
"""
from __future__ import annotations

from leadscout import infra
from leadscout.models import Evidence, ProviderResult


def _ev(**overrides) -> Evidence:
    base = dict(
        id="ev-001", source_type="footprint", url="https://example.com",
        observed_at="2026-09-19T00:00:00+00:00", content_sha256="a" * 64,
        strength="MEDIUM", snippet="s", snapshot_path="out/x.txt",
        family="network_footprint",
    )
    base.update(overrides)
    return Evidence(**base)


def test_range_matched_evidence_yields_public_cloud_footprint():
    result = ProviderResult(provider_name="footprint", evidence=[_ev(provider="AWS", strength="STRONG")])
    out = infra.build_footprints(result)
    assert len(out) == 1
    assert out[0].category == "PUBLIC_CLOUD"
    assert out[0].provider == "AWS"
    assert out[0].confidence == "HIGH"


def test_edge_delivery_evidence_yields_edge_footprint_not_public_cloud():
    result = ProviderResult(provider_name="footprint",
                            evidence=[_ev(id="ev-002", family="edge_delivery", provider=None, strength="WEAK")])
    out = infra.build_footprints(result)
    assert len(out) == 1
    assert out[0].category == "EDGE"
    assert out[0].provider is None


def test_cpanel_trace_yields_managed_hosting_medium_confidence():
    result = ProviderResult(provider_name="footprint", evidence=[])
    result._cpanel_trace = True
    out = infra.build_footprints(result)
    assert len(out) == 1
    assert out[0].category == "MANAGED_HOSTING"
    assert out[0].confidence == "MEDIUM"
    assert out[0].provider is None


def test_hetzner_style_holder_is_network_holder_never_provider_or_public_cloud():
    """A bare RIPEstat network holder
    (Hetzner/Rackforest-style, no authoritative range match) must never be stored
    or rendered as an operator/provider claim - `provider` stays None,
    `network_holder` carries the raw string, `category` stays UNKNOWN (not
    PUBLIC_CLOUD), and the reasoning text must never say "hosted by"/"runs on"."""
    for holder in ("HETZNER-AS", "RACKFORCET-AS"):
        result = ProviderResult(provider_name="footprint", evidence=[])
        result._holder_records = [{"ip": "203.0.113.9", "holder": holder}]
        out = infra.build_footprints(result)
        assert len(out) == 1
        fp = out[0]
        assert fp.provider is None
        assert fp.network_holder == holder
        assert fp.category != "PUBLIC_CLOUD"
        assert fp.category == "UNKNOWN"
        joined = " ".join(fp.reasoning).lower()
        assert "hosted by" not in joined
        assert "runs on" not in joined
        assert "registered to" in joined
        assert "operator not established" in joined


def test_duplicate_holders_across_ips_dedupe_to_one_footprint():
    result = ProviderResult(provider_name="footprint", evidence=[])
    result._holder_records = [
        {"ip": "203.0.113.9", "holder": "RACKFORCET-AS"},
        {"ip": "203.0.113.10", "holder": "RACKFORCET-AS"},
    ]
    out = infra.build_footprints(result)
    assert len(out) == 1


def test_degraded_footprint_result_with_no_side_channels_yields_no_footprints():
    result = ProviderResult(provider_name="footprint", status="degraded", note="crt.sh timeout")
    assert infra.build_footprints(result) == []


def test_holder_record_with_evidence_id_is_cited(monkeypatch=None):
    """Fix #1: `footprint.py`'s `run()` now attaches an `evidence_id` to every
    holder record - `build_footprints` must carry it into `evidence_ids`, not
    leave the footprint with nothing to cite."""
    result = ProviderResult(provider_name="footprint", evidence=[])
    result._holder_records = [{"ip": "203.0.113.9", "holder": "RACKFORCET-AS", "evidence_id": "ev-005"}]
    out = infra.build_footprints(result)
    assert len(out) == 1
    assert out[0].evidence_ids == ["ev-005"]


def test_artizan_style_cpanel_and_holder_merge_into_one_managed_hosting_footprint():
    """Fix #1 (Artizan): a cPanel trace and the address holder sharing that same
    host (WebSupport-style reseller hosting) are ONE fact, not two footprints -
    a single MANAGED_HOSTING item citing both evidence ids."""
    result = ProviderResult(provider_name="footprint", evidence=[])
    result._cpanel_trace = True
    result._cpanel_evidence_id = "ev-003"
    result._holder_records = [{"ip": "81.0.0.1", "holder": "WEBSUPPORT-AS", "evidence_id": "ev-004"}]
    out = infra.build_footprints(result)
    assert len(out) == 1
    fp = out[0]
    assert fp.category == "MANAGED_HOSTING"
    assert fp.network_holder == "WEBSUPPORT-AS"
    assert fp.provider is None
    assert sorted(fp.evidence_ids) == ["ev-003", "ev-004"]
