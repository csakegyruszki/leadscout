"""Shared helpers for the tests/test_inbound_*.py files (not collected as tests itself).

`mail_env` points the inbound service at a tmp out dir, a real default profile, no LLM,
and replaces smtplib so that ANY attempt to open an SMTP connection is recorded (and
fails the test that did not expect it).
"""
from __future__ import annotations

import smtplib
from pathlib import Path
from types import SimpleNamespace

import pytest

from leadscout.inbound import service
from leadscout.models import ComplianceResult, FitResult, LeadOutcome, Research
from leadscout.profile import load_profile

FIXTURES = Path(__file__).parent / "fixtures" / "mail"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def make_outcome(lead, status: str = "clear") -> LeadOutcome:
    return LeadOutcome(
        lead=lead,
        research=Research(summary="s", industry="SaaS", headquarters_country="US"),
        compliance=ComplianceResult(status != "clear", status, [], "ok"),
        fit=FitResult(80, "HIGH", 25, 40, 15, "r", "1k-10k", "reasoning"),
        sales_ready=status == "clear", llm_calls=2, llm_cost_usd=0.0001, llm_latency_ms=500,
        run_id="run-1", evidence_count=3, provenance_status="ok", provenance_head="1:abcdef",
    )


class Runner:
    """A stand-in for pipeline.process_lead that records the leads it was given."""

    def __init__(self, status: str = "clear", exc: Exception | None = None) -> None:
        self.status, self.exc, self.leads, self.stems = status, exc, [], []

    def __call__(self, lead, *, verbose=True, artifact_stem=None):
        self.leads.append(lead)
        self.stems.append(artifact_stem)
        if self.exc:
            raise self.exc
        return make_outcome(lead, self.status)


class _NoSmtp:
    calls: list = []

    def __init__(self, *a, **k):
        type(self).calls.append((a, k))
        raise AssertionError("smtplib was used but this test expects no network delivery")


@pytest.fixture
def mail_env(tmp_path, monkeypatch):
    out = tmp_path / "out"
    cfg = SimpleNamespace(out_dir=out, proof_mode=False, send_email=False, smtp_host="", smtp_port=587,
                          smtp_user="", smtp_password="")
    monkeypatch.setattr(service, "get_settings", lambda: cfg)
    profile = load_profile("default")
    monkeypatch.setattr(service, "get_profile", lambda: profile)
    monkeypatch.setattr(service, "default_llm_fill", lambda: None)

    def _no_real_pipeline(*args, **kwargs):
        raise AssertionError("the real pipeline was reached: this test must inject a runner "
                             "(an unpatched run writes to the repository's out/provenance ledger)")
    monkeypatch.setattr(service, "process_lead", _no_real_pipeline)
    monkeypatch.delenv("LEADSCOUT_INBOUND_MAX_BYTES", raising=False)
    _NoSmtp.calls = []
    monkeypatch.setattr(smtplib, "SMTP", _NoSmtp)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _NoSmtp)
    return SimpleNamespace(cfg=cfg, out=out, inbound=out / "inbound", profile=profile, smtp_calls=_NoSmtp.calls)
