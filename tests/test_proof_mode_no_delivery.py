"""A proof build must be incapable of sending mail, not merely unconfigured.

The committed demo batch carries invented contact names on real corporate domains
(samples/leads.json: a Zapier address, a Masterplast address). "The SMTP settings happened
to be empty when the proof was built" is not a good enough reason for no message having
reached them - one credential left in the environment during a proof rebuild would be. So
the refusal lives in the code path, above the credential check.
"""
from __future__ import annotations

import dataclasses

import pytest

from leadscout import notify


def _outcome():
    from leadscout.models import (
        CloudUsageAssessment,
        ComplianceResult,
        FitResult,
        Lead,
        LeadOutcome,
        Research,
    )
    research = Research(cloud_usage=CloudUsageAssessment(state="UNKNOWN"))
    return LeadOutcome(
        lead=Lead("Dana", "dana@zapier.com", "Zapier", "https://zapier.com"),
        research=research,
        compliance=ComplianceResult(flagged=False, status="clear", matches=[], reasoning="ok"),
        fit=FitResult(score=65, confidence="MEDIUM", reasoning="ok"),
        sales_ready=True,
    )


@pytest.fixture
def _outbox(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "settings",
                        dataclasses.replace(notify.settings, outbox_dir=tmp_path / "outbox"))
    return tmp_path / "outbox"


def test_proof_mode_refuses_to_send_even_with_credentials_present(_outbox, monkeypatch):
    """The invariant: credentials present, proof mode on, smtplib MUST NOT be touched."""
    monkeypatch.setattr(notify, "settings", dataclasses.replace(
        notify.settings, outbox_dir=_outbox, proof_mode=True,
        smtp_host="smtp.example.test", smtp_user="user", smtp_password="secret"))

    def explode(*args, **kwargs):
        raise AssertionError("proof mode attempted a network delivery")

    monkeypatch.setattr(notify.smtplib, "SMTP", explode)

    path = notify.notify(_outcome())
    assert path.is_file(), "the .eml must still be written"
    assert path.read_bytes(), "the .eml must have content"


def test_without_proof_mode_the_credential_check_still_governs(_outbox, monkeypatch):
    """The guard must not silently disable real delivery for everyone else: with proof
    mode off and credentials present, the send path IS taken."""
    monkeypatch.setattr(notify, "settings", dataclasses.replace(
        notify.settings, outbox_dir=_outbox, proof_mode=False, send_email=True,
        smtp_host="smtp.example.test", smtp_user="user", smtp_password="secret"))
    attempted: list[str] = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            attempted.append(host)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            pass

        def login(self, *a):
            pass

        def send_message(self, msg):
            attempted.append("sent")

    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    notify.notify(_outcome())
    assert attempted == ["smtp.example.test", "sent"]


def test_no_credentials_means_no_send_regardless_of_proof_mode(_outbox, monkeypatch):
    for proof_mode in (True, False):
        monkeypatch.setattr(notify, "settings", dataclasses.replace(
            notify.settings, outbox_dir=_outbox, proof_mode=proof_mode,
            smtp_host="", smtp_user="", smtp_password=""))

        def explode(*args, **kwargs):
            raise AssertionError("sent with no credentials configured")

        monkeypatch.setattr(notify.smtplib, "SMTP", explode)
        assert notify.notify(_outcome()).is_file()


# --- the flag's SPELLING is part of the invariant ----------------------------------
# The tests above set `proof_mode` directly, so they could not see that
# `LEADSCOUT_PROOF_MODE=true` never reached them: the flag was parsed as `== "1"`, so an
# ordinary spelling left this whole refusal switched off. These two run the real
# environment variable through a real config import, in a subprocess, because every
# config field is resolved at import time.

_PROBE = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {repo!r})
from leadscout import notify
from leadscout.config import settings

touched = []


def _explode(*a, **k):
    touched.append("SMTP")
    raise AssertionError("network delivery attempted")


notify.smtplib.SMTP = _explode
object.__setattr__(settings, "smtp_host", "smtp.example.test")
object.__setattr__(settings, "smtp_user", "user")
object.__setattr__(settings, "smtp_password", "secret")
object.__setattr__(settings, "outbox_dir", Path(tempfile.mkdtemp()))

from leadscout.models import ComplianceResult, FitResult, Lead, LeadOutcome, Research

outcome = LeadOutcome(
    lead=Lead("Dana", "dana@zapier.com", "Zapier", "https://zapier.com"),
    research=Research(),
    compliance=ComplianceResult(flagged=False, status="clear", matches=[], reasoning="ok"),
    fit=FitResult(score=65, confidence="MEDIUM", reasoning="ok"),
    sales_ready=True,
)
path = notify.notify(outcome)
print(json.dumps({{"proof_mode": settings.proof_mode, "touched": touched,
                  "eml": Path(path).is_file()}}))
"""


def _probe(value: str):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["LEADSCOUT_PROOF_MODE"] = value
    proc = subprocess.run([sys.executable, "-c", _PROBE.format(repo=str(repo))],
                          cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    payload = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    return proc, (json.loads(payload[-1]) if payload else None)


def test_an_ordinary_true_spelling_switches_the_refusal_ON_not_off():
    """Credentials configured, LEADSCOUT_PROOF_MODE=true: smtplib MUST NOT be touched."""
    proc, result = _probe("true")
    assert proc.returncode == 0, proc.stderr
    assert result["proof_mode"] is True
    assert result["touched"] == [], "an outbound delivery was attempted in proof mode"
    assert result["eml"] is True


def test_a_misspelled_flag_fails_loudly_instead_of_silently_sending():
    """`tru` is not `true`. The run must not start at all - and must certainly not
    reach a send with the operator believing proof mode is on."""
    proc, result = _probe("tru")
    assert proc.returncode != 0
    assert "ConfigError" in proc.stderr and "LEADSCOUT_PROOF_MODE" in proc.stderr
    assert result is None
    assert "SMTP" not in proc.stdout


def test_credentials_alone_do_not_send(_outbox, monkeypatch):
    """Sending is opt-in: SMTP credentials present but LEADSCOUT_SEND_EMAIL off must not
    touch smtplib (a mailbox password added for the IMAP poller is not consent to send)."""
    monkeypatch.setattr(notify, "settings", dataclasses.replace(
        notify.settings, outbox_dir=_outbox, proof_mode=False, send_email=False,
        smtp_host="smtp.example.test", smtp_user="user", smtp_password="secret"))

    def explode(*a, **k):
        raise AssertionError("smtplib.SMTP was called with LEADSCOUT_SEND_EMAIL off")

    monkeypatch.setattr(notify.smtplib, "SMTP", explode)
    assert notify.notify(_outcome()).is_file()
