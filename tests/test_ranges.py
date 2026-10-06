"""ranges.RangeIndex: local, most-specific-prefix-wins matching over the loaded
AWS/GCP/OCI/Azure feeds - offline, entries built directly (no HTTP)."""
from __future__ import annotations

import ipaddress

from leadscout import ranges


def _entry(cidr, provider, service, role, region=None, feed="test-feed", version="1"):
    return ranges._Entry(ipaddress.ip_network(cidr), provider, service, region, feed, version, role)


def _index_with(entries):
    idx = ranges.RangeIndex.__new__(ranges.RangeIndex)
    idx.entries = entries
    return idx


def test_cloudfront_service_is_classified_edge():
    idx = _index_with([_entry("13.32.0.0/15", "AWS", "CLOUDFRONT", "EDGE")])
    match = idx.match("13.32.0.1")
    assert match is not None
    assert match.provider == "AWS"
    assert match.service == "CLOUDFRONT"
    assert match.network_role == "EDGE"


def test_globalaccelerator_and_route53_are_edge_too():
    for service in ("GLOBALACCELERATOR", "ROUTE53"):
        idx = _index_with([_entry("13.32.0.0/15", "AWS", service, "EDGE")])
        assert idx.match("13.32.0.1").network_role == "EDGE"


def test_azure_frontdoor_prefix_is_edge():
    idx = _index_with([_entry("13.107.213.0/24", "Azure", "AzureFrontDoor.Frontend", "EDGE")])
    match = idx.match("13.107.213.5")
    assert match.network_role == "EDGE"
    assert match.provider == "Azure"


def test_generic_amazon_never_overrides_a_more_specific_service_at_same_ip():
    """A /16 tagged generic AMAZON and a /24 tagged S3 (or CLOUDFRONT) inside it: the
    most-specific (longest prefix) entry wins, and on an exact tie the specific
    service beats the generic one - the generic superset entry never overrides a
    more specific match (the infrastructure model "Detection order and the range
    matcher")."""
    idx = _index_with([
        _entry("13.32.0.0/12", "AWS", "AMAZON", "PUBLIC_CLOUD_COMPUTE"),
        _entry("13.32.0.0/24", "AWS", "S3", "PUBLIC_CLOUD_SERVICE"),
    ])
    match = idx.match("13.32.0.5")
    assert match.service == "S3"
    assert match.network_role == "PUBLIC_CLOUD_SERVICE"


def test_generic_and_specific_same_prefix_length_specific_wins():
    idx = _index_with([
        _entry("13.32.0.0/24", "AWS", "AMAZON", "PUBLIC_CLOUD_COMPUTE"),
        _entry("13.32.0.0/24", "AWS", "CLOUDFRONT", "EDGE"),
    ])
    match = idx.match("13.32.0.5")
    assert match.service == "CLOUDFRONT"


def test_no_match_returns_none():
    idx = _index_with([_entry("13.32.0.0/24", "AWS", "AMAZON", "PUBLIC_CLOUD_COMPUTE")])
    assert idx.match("203.0.113.9") is None


def test_invalid_ip_returns_none():
    idx = _index_with([_entry("13.32.0.0/24", "AWS", "AMAZON", "PUBLIC_CLOUD_COMPUTE")])
    assert idx.match("not-an-ip") is None


def test_oci_generic_tag_is_compute_named_tag_is_service():
    generic = _entry("140.91.0.0/16", "OCI", "OCI", "PUBLIC_CLOUD_COMPUTE")
    named = _entry("140.92.0.0/16", "OCI", "OBJECT_STORAGE", "PUBLIC_CLOUD_SERVICE")
    idx = _index_with([generic, named])
    assert idx.match("140.91.0.1").network_role == "PUBLIC_CLOUD_COMPUTE"
    assert idx.match("140.92.0.1").network_role == "PUBLIC_CLOUD_SERVICE"
