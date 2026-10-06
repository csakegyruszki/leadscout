"""ProvenanceRun: evidence snapshots + provtrail ledger records, offline.

Central invariants: the contact name/email never touch the ledger or a snapshot; any
failure inside record() degrades the run's status instead of raising into the
pipeline; finish() reflects both a failed verify and a failed record().
"""
import hashlib
import json

import pytest

from leadscout import provenance
from leadscout.models import Evidence


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "LEDGER_DIR", tmp_path / "provenance")
    monkeypatch.setattr(provenance, "LEDGER_PATH", tmp_path / "provenance" / "ledger.jsonl")
    monkeypatch.setattr(provenance, "SNAPSHOT_DIR", tmp_path / "provenance" / "snapshots")


def _website_evidence(eid: str = "ev-001") -> Evidence:
    return Evidence(
        id=eid, source_type="website", url="https://example.com",
        observed_at="2026-09-19T00:00:00+00:00", content_sha256="a" * 64,
        strength="STRONG", snippet="About us: we make widgets.",
        snapshot_path=f"out/provenance/snapshots/run/{eid}.txt",
        family="corporate_identity",
    )


def test_record_writes_snapshot_and_ledger_entry():
    run = provenance.ProvenanceRun.start("Acme")
    run.record(_website_evidence(), b"About us: we make widgets.", stage="research")
    status, head, count = run.finish()
    assert status == "ok"
    assert count == 1
    assert head != "none"
    assert provenance.LEDGER_PATH.exists()


def test_ledger_and_snapshots_never_contain_contact_name_or_email():
    run = provenance.ProvenanceRun.start("Acme")
    run.record(_website_evidence(), b"About us: we make widgets, not people.", stage="research")
    run.finish()

    ledger_text = provenance.LEDGER_PATH.read_text(encoding="utf-8")
    assert "Priya Nair" not in ledger_text
    assert "priya@cloud-trim.io" not in ledger_text

    for snap in provenance.SNAPSHOT_DIR.rglob("*"):
        if snap.is_file():
            content = snap.read_bytes()
            assert b"Priya Nair" not in content
            assert b"priya@cloud-trim.io" not in content


def test_ledger_records_company_name_only_in_extra():
    run = provenance.ProvenanceRun.start("Acme")
    run.record(_website_evidence(), b"content", stage="research")
    run.finish()
    line = provenance.LEDGER_PATH.read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(line)
    assert rec["extra"]["company"] == "Acme"
    assert rec["session_id"] == run.run_id


def test_failing_add_degrades_the_run_but_does_not_raise(monkeypatch):
    run = provenance.ProvenanceRun.start("Acme")

    def _boom(*args, **kwargs):
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(run._ledger, "add", _boom)
    run.record(_website_evidence(), b"content", stage="research")  # must not raise
    status, head, count = run.finish()
    assert status == "degraded"
    assert count == 0


def test_finish_is_degraded_when_verify_itself_fails(monkeypatch):
    run = provenance.ProvenanceRun.start("Acme")
    run.record(_website_evidence(), b"content", stage="research")

    def _boom(*args, **kwargs):
        raise RuntimeError("disk error")

    monkeypatch.setattr(run._ledger, "verify", _boom)
    status, head, count = run.finish()
    assert status == "degraded"
    assert head == "none"


def test_next_evidence_id_increments():
    run = provenance.ProvenanceRun.start("Acme")
    assert run.next_evidence_id() == "ev-001"
    assert run.next_evidence_id() == "ev-002"
    assert run.next_evidence_id() == "ev-003"


def test_record_policy_files_records_two_policy_lists_and_the_profile():
    # do-not-engage list + restricted jurisdictions + the active profile file itself.
    run = provenance.ProvenanceRun.start("Acme")
    provenance.record_policy_files(run)
    status, head, count = run.finish()
    assert count == 3
    assert status == "ok"


def test_kind_mapping_uses_allowed_provtrail_kinds():
    from provtrail.ledger import ALLOWED_KINDS
    for kind in provenance._KIND_BY_SOURCE.values():
        assert kind in ALLOWED_KINDS


# --- Fix #4: snapshot path exists and its sha256 == content_sha256, per provider ---
# ext_by_source is the ledger's own source of truth for the file extension; each
# provider module's Evidence.snapshot_path already assumes this exact extension
# (see that module's `_finalise`/Evidence construction) - a mismatch here is
# precisely the bug REVIEW-6bA-verified.md #4 fixed (the ext used to default to
# "txt" for every source_type not explicitly listed).
@pytest.mark.parametrize("source_type, ext", sorted(provenance._EXT_BY_SOURCE.items()))
def test_snapshot_path_exists_and_hash_matches_for_every_source_type(source_type, ext):
    run = provenance.ProvenanceRun.start("Acme")
    eid = run.next_evidence_id()
    raw = f"payload for {source_type}".encode()
    ev = Evidence(
        id=eid, source_type=source_type, url="https://example.com",
        observed_at="2026-09-19T00:00:00+00:00", content_sha256=hashlib.sha256(raw).hexdigest(),
        strength="WEAK", snippet="test", snapshot_path=f"snapshots/{run.run_id}/{eid}.{ext}",
        family="corporate_identity",
    )
    run.record(ev, raw, stage="test")
    status, _, count = run.finish()
    assert status == "ok"
    assert count == 1

    snapshot_file = provenance.SNAPSHOT_DIR / run.run_id / f"{eid}.{ext}"
    assert snapshot_file.exists(), f"expected snapshot at {snapshot_file} for source_type={source_type!r}"
    assert hashlib.sha256(snapshot_file.read_bytes()).hexdigest() == ev.content_sha256
