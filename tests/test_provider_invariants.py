"""Cross-provider invariant (REVIEW-6bA-verified.md #6): `status == "degraded"` must
always imply `evidence == []` - a failed provider yields NO evidence, never a partial
result assembled before the failure (absence of a signal is not evidence the signal
doesn't exist elsewhere - see models.ProviderResult's docstring). One parametrised
test forces a hard network error in each provider (httpx.Client itself raises, so
every provider takes its outermost except-branch) and checks the invariant holds for
all six `ProviderResult`-shaped providers.
"""
import httpx
import pytest

from leadscout.providers import ats, footprint, github, gleif, trust_pages, vendor


class _FakeVendorSettingsWithKey:
    brave_api_key = "test-key"
    http_timeout = 5.0


def _boom(*args, **kwargs):
    raise httpx.ConnectTimeout("timed out")


_CASES = [
    pytest.param(ats, lambda: ats.run("Acme", ['<a href="https://jobs.lever.co/acme">Jobs</a>']), id="ats"),
    pytest.param(github, lambda: github.run("Acme", ['<a href="https://github.com/acme">GitHub</a>']), id="github"),
    pytest.param(gleif, lambda: gleif.run("Acme", "United States", "https://acme.example"), id="gleif"),
    pytest.param(footprint, lambda: footprint.run("acme.example"), id="footprint"),
    pytest.param(trust_pages, lambda: trust_pages.run("acme.example", ["some page text"]), id="trust_pages"),
    pytest.param(vendor, lambda: vendor.run("Acme"), id="vendor"),
]


@pytest.mark.parametrize("module, call", _CASES)
def test_degraded_implies_no_evidence(monkeypatch, module, call):
    if module is vendor:
        monkeypatch.setattr(vendor, "settings", _FakeVendorSettingsWithKey())
    monkeypatch.setattr(module.httpx, "Client", _boom)
    result = call()
    assert result.status == "degraded", f"{module.__name__} did not degrade on a hard network error"
    assert result.evidence == [], f"{module.__name__} leaked evidence on a degraded result"
