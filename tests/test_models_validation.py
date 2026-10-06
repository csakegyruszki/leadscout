"""Evidence's five enum-shaped fields (strength/family/provider/scope/freshness) are
Literal[...] types, validated in __post_init__ (REVIEW-6bA-verified.md #3) - an
out-of-enum value must raise at construction, not drift silently into cloud.py's
classification. One rejection test per field, plus one construction test proving
every value actually named in the enum is accepted.
"""
import pytest

from leadscout.models import Evidence

_BASE = dict(
    id="ev-001", source_type="website", url="https://example.com",
    observed_at="2026-09-19T00:00:00+00:00", content_sha256="a" * 64,
    strength="STRONG", snippet="ok", snapshot_path="out/provenance/snapshots/run/ev-001.txt",
    family="corporate_identity",
)


def _make(**overrides) -> Evidence:
    return Evidence(**{**_BASE, **overrides})


def test_valid_construction_does_not_raise():
    _make()


@pytest.mark.parametrize("bad", ["strong", "unknown", "", "very_strong", None])
def test_rejects_out_of_enum_strength(bad):
    with pytest.raises(ValueError):
        _make(strength=bad)


@pytest.mark.parametrize("bad", ["unknown", "", "website", "cloud", None])
def test_rejects_out_of_enum_family(bad):
    with pytest.raises(ValueError):
        _make(family=bad)


@pytest.mark.parametrize("bad", ["", " aws", "aws ", " "])
def test_rejects_malformed_provider(bad):
    """v0.2.2 (PART A item 1): CloudProvider is an OPEN string now, not a closed
    enum - the infrastructure model's "Main invariant" (there is no provider list; a
    network/service fingerprint can name a provider a closed four-way enum never
    anticipated, e.g. "Hetzner", "OpenStack"). Only an empty or non-stripped
    string is rejected."""
    with pytest.raises(ValueError):
        _make(provider=bad)


@pytest.mark.parametrize("ok", ["AWS", "aws", "amazon", "unknown", "Hetzner-AS", "RACKFORCET-AS"])
def test_accepts_any_non_empty_stripped_provider(ok):
    _make(provider=ok)


def test_provider_none_is_accepted():
    _make(provider=None)


@pytest.mark.parametrize("bad", ["workload_", "", "Storage", None])
def test_rejects_out_of_enum_scope(bad):
    with pytest.raises(ValueError):
        _make(scope=bad)


@pytest.mark.parametrize("bad", ["Current", "", "past", None])
def test_rejects_out_of_enum_freshness(bad):
    with pytest.raises(ValueError):
        _make(freshness=bad)


@pytest.mark.parametrize("value", ["DIRECT", "STRONG", "MEDIUM", "WEAK"])
def test_every_strength_enum_value_is_accepted(value):
    _make(strength=value)


@pytest.mark.parametrize("value", [
    "first_party_statement", "vendor_case_study", "ats_hiring", "engineering_footprint",
    "network_footprint", "edge_delivery", "corporate_identity", "encyclopedic", "policy",
])
def test_every_family_enum_value_is_accepted(value):
    _make(family=value)


@pytest.mark.parametrize("value", ["AWS", "Azure", "GCP", "OCI", "Cloudflare", "other"])
def test_every_provider_enum_value_is_accepted(value):
    _make(provider=value)


@pytest.mark.parametrize("value", ["workload", "storage", "edge", "saas_dependency", "unknown"])
def test_every_scope_enum_value_is_accepted(value):
    _make(scope=value)


@pytest.mark.parametrize("value", ["current", "historical", "unknown"])
def test_every_freshness_enum_value_is_accepted(value):
    _make(freshness=value)
