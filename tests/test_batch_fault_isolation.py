"""One lead's failure must not be every later lead's failure.

Measured before the guard in `cli.main`: a hard LLM failure on lead 2 of 3 propagated
out of the batch loop, so lead 3 was never attempted and the run ended in a traceback
instead of a report. The leads already written survived on disk, but nothing said that
the rest had silently not happened - and the process still exited 0.

Nothing here writes to `out/`: `process_lead` is replaced entirely, because what is
under test is the loop's fault handling, not the pipeline.
"""
from __future__ import annotations

import json

import pytest

from leadscout import cli
from leadscout.llm import LLMError

LEADS = [
    {"name": "n1", "email": "a@leada.example", "company": "LeadA", "website": "https://leada.example"},
    {"name": "n2", "email": "b@leadb.example", "company": "LeadB", "website": "https://leadb.example"},
    {"name": "n3", "email": "c@leadc.example", "company": "LeadC", "website": "https://leadc.example"},
]


@pytest.fixture
def leads_file(tmp_path):
    path = tmp_path / "leads.json"
    path.write_text(json.dumps(LEADS), encoding="utf-8")
    return path


def _run_with(monkeypatch, leads_file, failing_company, exc):
    processed: list[str] = []

    def fake_process_lead(lead, verbose=True):
        processed.append(lead.company)
        if lead.company == failing_company:
            raise exc
        return None

    monkeypatch.setattr(cli, "process_lead", fake_process_lead)
    code = cli.main(["--quiet", "batch", str(leads_file)])
    return processed, code


def test_a_failing_lead_does_not_stop_the_ones_after_it(monkeypatch, leads_file, capsys):
    processed, code = _run_with(monkeypatch, leads_file, "LeadB",
                               LLMError("every model in the chain failed"))
    assert processed == ["LeadA", "LeadB", "LeadC"], processed
    assert code == 1, "a partial batch must not report success"
    err = capsys.readouterr().err
    assert "LeadB" in err and "1 of 3 leads failed" in err
    assert "LeadA" not in err and "LeadC" not in err


@pytest.mark.parametrize("exc", [
    LLMError("all models in the fallback chain failed"),
    TimeoutError("read timeout"),
    ConnectionError("connection reset"),
    ValueError("schema-invalid response"),
    KeyError("missing field"),
])
def test_the_loop_survives_every_failure_shape_a_lead_can_produce(monkeypatch, leads_file, exc):
    processed, code = _run_with(monkeypatch, leads_file, "LeadB", exc)
    assert processed == ["LeadA", "LeadB", "LeadC"]
    assert code == 1


def test_a_batch_with_no_failures_still_reports_success(monkeypatch, leads_file, capsys):
    processed, code = _run_with(monkeypatch, leads_file, "NoSuchCompany", RuntimeError("never raised"))
    assert processed == ["LeadA", "LeadB", "LeadC"]
    assert code is None or code == 0
    assert "failed" not in capsys.readouterr().err


def test_every_lead_failing_is_still_reported_rather_than_raised(monkeypatch, leads_file, capsys):
    """The degenerate case - e.g. no LLM credentials at all - has to produce a report,
    not a traceback, because that is the state a first-time user is most likely in."""
    def always_fails(lead, verbose=True):
        raise LLMError("no usable provider credentials for the configured chain")

    monkeypatch.setattr(cli, "process_lead", always_fails)
    code = cli.main(["--quiet", "batch", str(leads_file)])
    assert code == 1
    err = capsys.readouterr().err
    assert "3 of 3 leads failed" in err
    for company in ("LeadA", "LeadB", "LeadC"):
        assert company in err


def test_the_summary_names_the_counts_and_the_failed_lead_unconditionally(monkeypatch, leads_file, capsys):
    """A failure visible only as an exit code is one a reader can miss."""
    _run_with(monkeypatch, leads_file, "LeadB", LLMError("chain exhausted"))
    out = capsys.readouterr().out
    assert "attempted=3" in out and "succeeded=2" in out and "failed=1" in out
    assert "failed lead: LeadB" in out


def test_a_clean_batch_also_prints_its_counts(monkeypatch, leads_file, capsys):
    _run_with(monkeypatch, leads_file, "NoSuchCompany", RuntimeError("never raised"))
    out = capsys.readouterr().out
    assert "attempted=3" in out and "succeeded=3" in out and "failed=0" in out
    assert "failed lead:" not in out
