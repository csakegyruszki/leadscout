"""process_lead's provenance terminal markers: a normal run appends run_completed,
an aborted run appends run_aborted then re-raises - offline, no network/LLM calls.

Everything the pipeline calls out to (research_lead, screen, score_fit, write_row,
notify) is faked; only the provenance side effects and control flow are under test.
"""
from dataclasses import dataclass
from pathlib import Path

import pytest

from leadscout import pipeline, provenance
from leadscout.models import ComplianceResult, FitResult, Lead, Research


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    ledger_dir = tmp_path / "provenance"
    monkeypatch.setattr(provenance, "LEDGER_DIR", ledger_dir)
    monkeypatch.setattr(provenance, "LEDGER_PATH", ledger_dir / "ledger.jsonl")
    monkeypatch.setattr(provenance, "SNAPSHOT_DIR", ledger_dir / "snapshots")

    @dataclass
    class _FakeSettings:
        out_dir: Path = tmp_path / "out"
        tracker_path: Path = tmp_path / "out" / "leads_tracker.xlsx"
        fit_threshold: int = 60
        profile: str = "demo"

    monkeypatch.setattr(pipeline, "settings", _FakeSettings())


def _lead() -> Lead:
    return Lead(name="Ann", email="ann@x.com", company="X Corp", website="https://x.com")


def _fake_research(lead, run):
    return Research(summary="s", industry="SaaS", headquarters_country="US")


def _fake_compliance(lead, research, run):
    return ComplianceResult(False, "clear", [], "ok")


def _fake_fit(lead, research):
    return FitResult(80, "HIGH", 25, 40, 15, "r", "1k-10k", "reasoning")


def _patch_happy_path(monkeypatch):
    monkeypatch.setattr(pipeline, "research_lead", _fake_research)
    monkeypatch.setattr(pipeline, "screen", _fake_compliance)
    monkeypatch.setattr(pipeline, "score_fit", _fake_fit)
    monkeypatch.setattr(pipeline, "write_row", lambda outcome, path: 2)
    monkeypatch.setattr(pipeline, "notify", lambda outcome: Path("out/outbox/x-corp.eml"))


def test_successful_run_marks_completed(monkeypatch):
    _patch_happy_path(monkeypatch)
    outcome = pipeline.process_lead(_lead(), verbose=False)
    assert outcome.sales_ready is True
    assert provenance.run_status(outcome.run_id) == "completed"


def test_the_result_json_records_which_code_produced_it(monkeypatch):
    """F-15: the run, not the proof manifest written afterwards, is the authority on
    which source produced an artefact - so the binding has to be in the artefact."""
    import json

    from leadscout import runtime_identity

    _patch_happy_path(monkeypatch)
    outcome = pipeline.process_lead(_lead(), verbose=False)
    ctx = runtime_identity.context()
    assert outcome.proof_run["id"] == ctx.id
    assert outcome.proof_run["source_fingerprint"] == ctx.source_fingerprint

    written = json.loads(
        (pipeline.settings.out_dir / "results" / "x-corp.json").read_text(encoding="utf-8"))
    assert written["proof_run"]["id"] == ctx.id
    assert written["proof_run"]["source_fingerprint"] == ctx.source_fingerprint


def test_every_lead_in_one_batch_carries_the_same_binding(monkeypatch):
    """The context is created once per run and never replaced, so two leads processed
    in one batch cannot disagree about which code they ran on."""
    _patch_happy_path(monkeypatch)
    first = pipeline.process_lead(_lead(), verbose=False)
    second = pipeline.process_lead(_lead(), verbose=False)
    assert first.proof_run == second.proof_run
    assert first.proof_run["id"]


def test_aborted_run_marks_aborted_and_reraises(monkeypatch):
    monkeypatch.setattr(pipeline, "research_lead", _fake_research)

    def _boom(lead, research, run):
        raise RuntimeError("compliance screen exploded")

    monkeypatch.setattr(pipeline, "screen", _boom)

    captured_run_id = {}
    real_start = provenance.ProvenanceRun.start

    def _start(company):
        run = real_start(company)
        captured_run_id["run_id"] = run.run_id
        return run

    monkeypatch.setattr(provenance.ProvenanceRun, "start", staticmethod(_start))

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.process_lead(_lead(), verbose=False)

    assert excinfo.value.leadscout_run_id == captured_run_id["run_id"]
    assert provenance.run_status(captured_run_id["run_id"]) == "aborted"

    # the ledger's last record for this run is the abort marker, not a stray
    # earlier evidence record from record_policy_files()
    records = [r for r in provenance.Ledger(str(provenance.LEDGER_PATH)).records()
               if r.get("session_id") == captured_run_id["run_id"]]
    assert records[-1]["extra"]["stage"] == "run_aborted"
    assert records[-1]["extra"]["reason"] == "RuntimeError"
    assert records[-1]["extra"]["run_id"] == captured_run_id["run_id"]


def test_run_status_is_unknown_for_unseen_run_id():
    assert provenance.run_status("no-such-run") == "unknown"
