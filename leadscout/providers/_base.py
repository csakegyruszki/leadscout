"""Shared skeleton bits every provider in this package repeats: the User-Agent
string and the evidence-id/snapshot-path fallback used when a provider runs
without a `ProvenanceRun` (offline tests, ad-hoc calls).
"""

from __future__ import annotations

from ..util import USER_AGENT

# --- Design notes -----------------------------------------------------------------
# Deliberately does NOT wrap `httpx.Client` construction or the per-provider
# `except Exception -> degraded` block: provider tests patch each provider
# module's own `httpx.Client` attribute
# (`monkeypatch.setattr(<provider>.httpx, "Client", fake_client)`), and each
# provider's degraded-path `note`/`calls` accounting differs slightly (calls made
# so far, partial evidence already collected, etc.) - routing either through a
# shared helper here would either break test patching or force every provider
# into an identical shape it does not have.

UA = USER_AGENT


def fallback_evidence_id(provider: str, run: object | None, seq: int = 1) -> str:
    """The evidence id to use when there is no `ProvenanceRun` to assign one
    (`run is None` - offline/unit-test call sites). Mirrors each provider's
    previous inline `f"ev-<name>-N"` literal so existing fallback ids are
    unchanged."""
    if run is not None and hasattr(run, "next_evidence_id"):
        return run.next_evidence_id()  # type: ignore[no-any-return]
    return f"ev-{provider}-{seq}"


def snapshot_path(run: object | None, eid: str, ext: str = "json") -> str:
    """Same `out/provenance/snapshots/<run_id>/<eid>.<ext>` shape every provider
    built inline; `run_id` falls back to "unrecorded" with no run tracker. `ext`
    defaults to "json" (most providers); footprint/trust_pages/vendor pass their
    own previous extension (.txt/.html) unchanged."""
    run_id = getattr(run, "run_id", None) if run is not None else None
    return f"out/provenance/snapshots/{run_id or 'unrecorded'}/{eid}.{ext}"
