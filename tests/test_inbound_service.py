"""Dedupe store, service orchestration, reply drafts, and the end-to-end path (offline)."""
import json
import threading
from dataclasses import dataclass
from email import message_from_bytes, policy
from pathlib import Path

import pytest
from inbound_support import Runner, fixture_bytes, mail_env  # noqa: F401 - mail_env is a fixture

from leadscout import pipeline, provenance
from leadscout.inbound import service
from leadscout.inbound.service import RECORD_KEYS, process_message
from leadscout.inbound.store import Store, content_hash
from leadscout.models import ComplianceResult, FitResult, Research

# --- store ----------------------------------------------------------------------------------------


def test_claim_is_idempotent_on_message_id(tmp_path):
    s = Store(tmp_path)
    first = s.claim("<a@x>", "h1", "t")
    again = s.claim("<a@x>", "h1", "t")
    assert first.is_new and not again.is_new and again.record_id == first.record_id


def test_same_content_under_a_new_message_id_is_a_duplicate(tmp_path):
    s = Store(tmp_path)
    first = s.claim("<a@x>", "same-hash", "t")
    second = s.claim("<b@x>", "same-hash", "t")
    assert not second.is_new and second.duplicate_of == first.record_id and second.reason == "content_hash"
    # and the second id is remembered, so its own redelivery resolves the same way
    assert s.claim("<b@x>", "other", "t").duplicate_of == first.record_id


def test_failed_claim_can_be_retried_but_a_finished_one_cannot(tmp_path):
    s = Store(tmp_path)
    s.claim("<a@x>", "h", "t")
    s.finish("<a@x>", "failed")
    retry = s.claim("<a@x>", "h", "t")
    assert retry.is_new and retry.reason == "retry"
    s.finish("<a@x>", "processed")
    assert not s.claim("<a@x>", "h", "t").is_new


def test_stale_claim_is_taken_over(tmp_path, monkeypatch):
    s = Store(tmp_path)
    s.claim("<a@x>", "h", "t")
    assert not s.claim("<a@x>", "h", "t").is_new                    # in progress: not stolen
    monkeypatch.setattr("leadscout.inbound.store.STALE_CLAIM_SECONDS", -1)
    assert s.claim("<a@x>", "h", "t").is_new


def test_concurrent_claims_have_exactly_one_winner(tmp_path):
    Store(tmp_path)
    results, barrier = [], threading.Barrier(8)

    def worker():
        store = Store(tmp_path)
        barrier.wait()
        results.append(store.claim("<race@x>", "race-hash", "t").is_new)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(True) == 1 and len(results) == 8


def test_content_hash_ignores_case_whitespace_and_reply_prefix():
    a = content_hash("A@X.test", "Re: Hello  there", "Some   body\ntext")
    assert a == content_hash("a@x.test", "hello there", "some body text")
    assert a != content_hash("a@x.test", "hello there", "different")


# --- service ------------------------------------------------------------------------------------------


def test_process_message_writes_record_and_draft(mail_env):
    runner = Runner("clear")
    rec = process_message(fixture_bytes("plain.eml"), source="test", runner=runner)
    assert rec["status"] == "processed" and set(RECORD_KEYS) == set(rec)
    assert runner.leads[0].company == "Fabrikam Logistics GmbH" and runner.leads[0].website.startswith("https://")
    on_disk = json.loads((mail_env.inbound / "records" / f"{rec['record_id']}.json").read_text(encoding="utf-8"))
    assert on_disk["lead"]["email"] == "anna.keller@fabrikam-logistics.test"
    assert on_disk["outcome"]["compliance"]["status"] == "clear" and on_disk["next_action"]
    assert on_disk["mail"]["message_id"] == "<plain-001@fabrikam-logistics.test>"
    assert on_disk["auth_results"]["spf"] == "pass" and on_disk["injection_flags"] == []
    assert on_disk["extraction"]["methods"]["website"] == "sender_domain"

    draft = message_from_bytes(Path(rec["draft"]["path"]).read_bytes(), policy=policy.default)
    assert draft["To"] == "anna.keller@fabrikam-logistics.test" and draft["From"] == "leadscout@example.test"
    assert draft["Subject"] == "Re: Cloud cost review for our logistics platform"
    assert draft["In-Reply-To"] == "<plain-001@fabrikam-logistics.test>"
    assert "Hi Anna," in draft.get_content()
    assert mail_env.smtp_calls == []


def test_duplicate_message_returns_earlier_record_id_without_rerunning_pipeline(mail_env):
    runner = Runner()
    first = process_message(fixture_bytes("plain.eml"), source="t", runner=runner)
    second = process_message(fixture_bytes("plain.eml"), source="t", runner=runner)
    assert second["status"] == "duplicate" and second["record_id"] == first["record_id"]
    assert len(runner.leads) == 1


def test_resend_with_new_message_id_is_caught_by_content_hash(mail_env):
    runner = Runner()
    raw = fixture_bytes("plain.eml")
    first = process_message(raw, source="t", runner=runner)
    resent = raw.replace(b"<plain-001@fabrikam-logistics.test>", b"<plain-001-resent@fabrikam-logistics.test>")
    second = process_message(resent, source="t", runner=runner)
    assert second["status"] == "duplicate" and second["duplicate_of"] == first["record_id"]
    assert second["reason"] == "duplicate by content_hash" and len(runner.leads) == 1


def test_no_usable_website_goes_to_needs_review_and_pipeline_is_not_run(mail_env):
    runner = Runner()
    rec = process_message(fixture_bytes("freemail_nosite.eml"), source="t", runner=runner)
    assert rec["status"] == "needs_review" and not runner.leads and rec["outcome"] is None
    assert rec["draft"]["path"] is None
    assert Path(mail_env.inbound / "records" / f"{rec['record_id']}.json").exists()


def test_oversize_message_is_rejected_with_a_recorded_reason(mail_env, monkeypatch):
    raw = fixture_bytes("attachment.eml")
    monkeypatch.setenv("LEADSCOUT_INBOUND_MAX_BYTES", "2000")
    runner = Runner()
    rec = process_message(raw, source="t", runner=runner)
    assert rec["status"] == "rejected" and "message_too_large" in rec["reason"] and not runner.leads
    assert process_message(raw, source="t", runner=runner)["status"] == "duplicate"


def test_message_under_the_cap_is_processed_and_attachment_only_recorded(mail_env, monkeypatch):
    monkeypatch.setenv("LEADSCOUT_INBOUND_MAX_BYTES", "20000")
    runner = Runner()
    rec = process_message(fixture_bytes("attachment.eml"), source="t", runner=runner)
    assert rec["status"] == "processed"
    assert rec["attachments"] == [{"filename": "architecture.pdf", "content_type": "application/pdf", "size": 6000}]
    assert "JVBER" not in json.dumps(rec)


def test_pipeline_failure_is_recorded_not_raised_and_can_be_retried(mail_env):
    boom = Runner(exc=RuntimeError("provider exploded with key sk-secret-123456789"))
    rec = process_message(fixture_bytes("plain.eml"), source="t", runner=boom)
    assert rec["status"] == "failed" and "RuntimeError" in rec["reason"] and "sk-secret" not in json.dumps(rec)
    ok = Runner()
    again = process_message(fixture_bytes("plain.eml"), source="t", runner=ok)
    assert again["status"] == "processed" and len(ok.leads) == 1


@pytest.mark.parametrize("status,expect_draft", [("clear", True), ("review", False), ("blocked", False)])
def test_draft_only_for_statuses_in_draft_on(mail_env, status, expect_draft):
    rec = process_message(fixture_bytes("plain.eml"), source="t", runner=Runner(status))
    assert (rec["draft"]["path"] is not None) is expect_draft
    if not expect_draft:
        assert status in rec["draft"]["suppressed_reason"]


def test_injection_flagged_mail_is_needs_review_and_the_pipeline_is_not_run(mail_env):
    runner = Runner("clear")
    rec = process_message(fixture_bytes("malicious.eml"), source="t", runner=runner)
    assert rec["status"] == "needs_review" and not runner.leads and rec["outcome"] is None
    assert "addressed_to_ai" in rec["injection_flags"] and "injection" in rec["reason"]
    assert rec["draft"]["path"] is None
    assert not (mail_env.inbound / "drafts").exists() or not list((mail_env.inbound / "drafts").iterdir())


def test_draft_goes_to_original_sender_for_forwarded_mail_and_reply_to_otherwise(mail_env):
    rec = process_message(fixture_bytes("forwarded.eml"), source="t", runner=Runner())
    draft = message_from_bytes(Path(rec["draft"]["path"]).read_bytes(), policy=policy.default)
    assert draft["To"] == "priya.nair@zenith-cloud.test" and draft["Subject"] == "Re: Partnership enquiry"
    reply_to = b"Reply-To: Sales Desk <desk@fabrikam-logistics.test>\r\nMessage-ID:"
    raw = fixture_bytes("plain.eml").replace(b"Message-ID:", reply_to)
    raw = raw.replace(b"plain-001", b"plain-002").replace(b"Cloud cost review", b"Cloud cost review v2")
    rec2 = process_message(raw, source="t", runner=Runner())
    assert message_from_bytes(Path(rec2["draft"]["path"]).read_bytes(), policy=policy.default)["To"] \
        == "desk@fabrikam-logistics.test"


def test_subject_that_already_starts_with_re_is_not_prefixed_again(mail_env):
    rec = process_message(fixture_bytes("reply_chain.eml"), source="t", runner=Runner())
    draft = message_from_bytes(Path(rec["draft"]["path"]).read_bytes(), policy=policy.default)
    assert draft["Subject"] == "Re: Following up on our call"
    assert draft["References"] == "<orig-700@vendor.test> <orig-777@vendor.test> <reply-001@lumen-retail.test>"


def test_no_reply_sender_gets_no_draft(mail_env):
    raw = fixture_bytes("plain.eml").replace(b"anna.keller@", b"noreply@").replace(b"plain-001", b"plain-nr")
    raw = raw.replace(b"Cloud cost review", b"Auto cost review")
    rec = process_message(raw, source="t", runner=Runner())
    assert rec["draft"]["path"] is None and "no-reply" in rec["draft"]["suppressed_reason"]


def test_missing_message_id_record_flags_the_synthesised_id(mail_env):
    rec = process_message(fixture_bytes("no_message_id.eml"), source="t", runner=Runner())
    assert rec["mail"]["message_id_synthesised"] is True and rec["status"] == "processed"


def test_no_pipeline_dry_run_claims_nothing(mail_env):
    runner = Runner()
    dry = process_message(fixture_bytes("plain.eml"), source="t", pipeline=False, runner=runner)
    assert dry["status"] == "extracted" and not runner.leads and dry["lead"]["company"]
    real = process_message(fixture_bytes("plain.eml"), source="t", runner=runner)
    assert real["status"] == "processed" and len(runner.leads) == 1


# --- end to end ---------------------------------------------------------------------------------------------


@dataclass
class _FakePipelineSettings:
    out_dir: Path
    tracker_path: Path
    fit_threshold: int = 60
    profile: str = "demo"


@pytest.fixture
def real_pipeline(mail_env, monkeypatch):
    """The real pipeline.process_lead with its providers stubbed (same seam as test_pipeline.py),
    writing into the tmp out dir, never the repository's out/."""
    ledger = mail_env.out / "provenance"
    monkeypatch.setattr(provenance, "LEDGER_DIR", ledger)
    monkeypatch.setattr(provenance, "LEDGER_PATH", ledger / "ledger.jsonl")
    monkeypatch.setattr(provenance, "SNAPSHOT_DIR", ledger / "snapshots")
    monkeypatch.setattr(service, "process_lead", pipeline.process_lead)
    monkeypatch.setattr(pipeline, "settings", _FakePipelineSettings(mail_env.out, mail_env.out / "tracker.xlsx"))
    monkeypatch.setattr(pipeline, "research_lead",
                        lambda lead, run: Research(summary="s", industry="SaaS", headquarters_country="DE"))
    monkeypatch.setattr(pipeline, "screen", lambda lead, research, run: ComplianceResult(False, "clear", [], "ok"))
    monkeypatch.setattr(pipeline, "score_fit",
                        lambda lead, research: FitResult(80, "HIGH", 25, 40, 15, "r", "1k-10k", "reasoning"))
    monkeypatch.setattr(pipeline, "write_row", lambda outcome, path: 2)
    monkeypatch.setattr(pipeline, "notify", lambda outcome, **kw: mail_env.out / "outbox" / "lead.eml")
    return mail_env


def test_end_to_end_clean_mail_with_stub_llm(real_pipeline):
    asked = []

    def stub_llm(system, user):
        asked.append(user)
        return {"job_title": "Logistics Manager"}
    rec = process_message(fixture_bytes("no_message_id.eml"), source="e2e", ask=stub_llm)
    assert rec["status"] == "processed" and rec["outcome"]["lead"]["job_title"] == "Logistics Manager"
    assert rec["extraction"]["methods"]["job_title"] == "llm_fill_in" and asked
    assert rec["outcome"]["run_id"] and rec["next_action"]
    assert Path(rec["draft"]["path"]).exists()
    assert (real_pipeline.out / "results" / f"{rec['record_id']}.json").exists()   # named by record id, not company
    assert real_pipeline.smtp_calls == []


def test_end_to_end_malicious_mail_is_flagged_not_run_and_sends_nothing(real_pipeline):
    rec = process_message(fixture_bytes("malicious.eml"), source="e2e",
                          ask=lambda s, u: pytest.fail("flagged mail must not reach the LLM"))
    assert rec["status"] == "needs_review" and rec["outcome"] is None
    assert {"ignore_instructions", "addressed_to_ai", "secret_exfiltration"} <= set(rec["injection_flags"])
    assert rec["draft"]["path"] is None and rec["draft"]["sent"] is False
    drafts = real_pipeline.inbound / "drafts"
    assert not drafts.exists() or list(drafts.iterdir()) == []
    assert not (real_pipeline.out / "results").exists()                 # the pipeline never ran
    assert real_pipeline.smtp_calls == []
