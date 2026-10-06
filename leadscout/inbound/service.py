"""Orchestration: raw message -> parse -> dedupe -> extract -> pipeline -> record (+ draft).

Split in two so the webhook can answer fast: `prepare` (size cap, parse, claim, raw copy - the
synchronous part that decides "new or duplicate") and `run_prepared` (extraction, pipeline,
record, draft - the slow part, safe to run in a background task).

Test seams: `get_settings`, `get_profile`, `default_llm_fill` are module functions and the
pipeline runner is a parameter, so a test never has to touch the real out/ directory,
profile cache or network.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..config import settings
from ..models import Lead
from ..notify import next_action
from ..pipeline import process_lead
from ..profile import active_profile
from .draft import build_draft, draft_decision, reply_address, send_decision, send_draft
from .extract import blocking, extract_lead, field_flags, injection_flags
from .parse import DEFAULT_MAX_BYTES, ParsedMessage, parse_message
from .store import Claim, Store, content_hash, inbound_dir, max_attempts, safe_id

logger = logging.getLogger("leadscout")

RECORD_KEYS = (
    "record_id", "status", "reason", "source", "processed_at", "mail", "body_preview", "content_hash",
    "lead", "extraction", "injection_flags", "auth_results", "attachments", "outcome", "next_action", "draft",
)


def get_settings():
    return settings


def get_profile():
    return active_profile()


def default_llm_fill():
    """The LLM callable used for field fill-in, or None. Tests replace this."""
    from .. import llm

    def ask(system: str, user: str) -> dict:
        return llm.ask_json(system, user, purpose="inbound_extract")
    return ask


def max_message_bytes() -> int:
    raw = os.getenv("LEADSCOUT_INBOUND_MAX_BYTES", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_MAX_BYTES
    except ValueError:
        return DEFAULT_MAX_BYTES
    return value if value > 0 else DEFAULT_MAX_BYTES


_SLOTS: dict[int, threading.BoundedSemaphore] = {}
_SLOTS_LOCK = threading.Lock()


def concurrency() -> int:
    raw = os.getenv("LEADSCOUT_INBOUND_CONCURRENCY", "").strip()
    try:
        value = int(raw) if raw else 1
    except ValueError:
        return 1
    return value if value > 0 else 1


@contextlib.contextmanager
def pipeline_slot():
    """Bounds concurrent extraction + pipeline runs. One at a time by default: the pipeline
    drains a process-global LLM telemetry list by slice (llm.telemetry), so two runs in one
    process would mix each other's call counts and costs, and each run is network-heavy."""
    n = concurrency()
    with _SLOTS_LOCK:
        sem = _SLOTS.setdefault(n, threading.BoundedSemaphore(n))
    with sem:
        yield


@dataclass
class Prepared:
    claim: Claim
    source: str
    parsed: ParsedMessage | None = None
    pipeline: bool = True
    # Set when the message was settled at prepare time (rejected / duplicate).
    record: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.record is not None

    @property
    def record_id(self) -> str:
        return self.claim.record_id


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _skeleton(record_id: str, status: str, reason: str, source: str) -> dict:
    rec = dict.fromkeys(RECORD_KEYS)
    rec.update(record_id=record_id, status=status, reason=reason, source=source,
               processed_at=datetime.now(UTC).isoformat(timespec="seconds"),
               injection_flags=[], attachments=[], mail={})
    return rec


def _settle(cfg, store: Store, message_id: str, rec: dict) -> dict:
    _write_json(inbound_dir(cfg.out_dir) / "records" / f"{rec['record_id']}.json", rec)
    store.finish(message_id, rec["status"])
    return rec


def _reject(cfg, store: Store, synth: str, sha: str, size: int, source: str, reason: str) -> Prepared:
    claim = store.claim(synth, sha, source, status="rejected")
    if not claim.is_new:
        return Prepared(claim, source, record=_duplicate_record(claim, source))
    rec = _skeleton(claim.record_id, "rejected", reason, source)
    rec["mail"] = {"message_id": synth, "message_id_synthesised": True, "size_bytes": size, "raw_sha256": sha}
    return Prepared(claim, source, record=_settle(cfg, store, synth, rec))


def reject_oversize(ref: str, size: int, *, source: str) -> dict:
    """Record a message that was NOT downloaded because its advertised size exceeds the cap
    (IMAP RFC822.SIZE). `ref` identifies it (mailbox + uid); there is no Message-ID yet."""
    cfg = get_settings()
    synth = f"<{hashlib.sha256(ref.encode()).hexdigest()}@leadscout.local>"
    cap = max_message_bytes()
    prep = _reject(cfg, Store(cfg.out_dir), synth, hashlib.sha256(ref.encode()).hexdigest(), size, source,
                   f"message_too_large: {size} bytes > cap {cap} (not downloaded)")
    return prep.record


def prepare(raw: bytes, *, source: str, pipeline: bool = True) -> Prepared:
    cfg = get_settings()
    store = Store(cfg.out_dir)
    cap = max_message_bytes()

    sha = hashlib.sha256(raw).hexdigest()
    synth = f"<{sha}@leadscout.local>"
    if len(raw) > cap:
        return _reject(cfg, store, synth, sha, len(raw), source, f"message_too_large: {len(raw)} bytes > cap {cap}")

    try:
        parsed = parse_message(raw)
    except Exception as exc:  # noqa: BLE001 - the parser is lenient, but never let a bad mail kill a poll
        return _reject(cfg, store, synth, sha, len(raw), source, f"unparseable: {type(exc).__name__}")

    chash = content_hash(parsed.from_addr, parsed.subject, parsed.body)
    if not pipeline:
        claim = Claim(True, safe_id(parsed.message_id), "dry_run")    # a dry run claims nothing
    else:
        claim = store.claim(parsed.message_id, chash, source)
        if not claim.is_new:
            return Prepared(claim, source, parsed=parsed, record=_duplicate_record(claim, source))
        # Persisted at claim time, before the caller (a webhook) acknowledges: if the process
        # dies afterwards, `mail resume` can re-drive the message from this copy.
        store.save_raw(claim.record_id, raw)
    return Prepared(claim, source, parsed=parsed, pipeline=pipeline)


def _duplicate_record(claim: Claim, source: str) -> dict:
    return {"record_id": claim.record_id, "status": "duplicate", "duplicate": True,
            "duplicate_of": claim.duplicate_of, "reason": f"duplicate by {claim.reason}",
            "earlier_status": claim.status, "source": source}


def run_prepared(prep: Prepared, *, runner=None, ask="default") -> dict:
    """Extraction, pipeline, record, draft. Never raises for a bad mail or a pipeline
    failure: the outcome is a record with status `failed` / `dead_letter` / `needs_review`."""
    if prep.done:
        return prep.record
    cfg, profile = get_settings(), get_profile()
    store = Store(cfg.out_dir)
    parsed = prep.parsed
    runner = runner or process_lead
    if ask == "default":
        ask = default_llm_fill() if prep.pipeline else None   # a dry run makes no network calls

    flags = injection_flags(parsed)
    rid = prep.record_id
    rec = _skeleton(rid, "needs_review", "", prep.source)
    rec.update(
        mail=parsed.meta(), body_preview=parsed.body[:500],
        content_hash=content_hash(parsed.from_addr, parsed.subject, parsed.body),
        auth_results=parsed.auth_results, attachments=[a.__dict__ for a in parsed.attachments],
        draft={"path": None, "suppressed_reason": "pipeline not run", "sent": False},
    )

    def settle() -> dict:
        records = inbound_dir(cfg.out_dir) / "records"
        # A dry run must never overwrite the record of a real run of the same message.
        name = f"{rid}.json" if prep.pipeline else f"{rid}.dryrun.json"
        _write_json(records / name, rec)
        if prep.pipeline:
            store.finish(parsed.message_id, rec["status"])
        return rec

    outcome = None
    with pipeline_slot():
        ex = extract_lead(parsed, ask, allow_llm=not flags)       # flagged mail never reaches the LLM fill-in
        flags = [*flags, *field_flags(ex.lead)]
        rec.update(
            lead=ex.lead, injection_flags=flags,
            extraction={"methods": ex.methods, "llm_used": ex.llm_used, "llm_note": ex.llm_note,
                        "missing": ex.missing, "notes": parsed.notes})
        if blocking(flags):
            rec["reason"] = "prompt-injection flags present; pipeline not run (needs a human)"
            return settle()
        if not ex.usable:
            rec["reason"] = "no usable website/company could be derived from the mail (needs a human)"
            return settle()
        if not prep.pipeline:
            rec["status"], rec["reason"] = "extracted", "pipeline skipped (--no-pipeline)"
            return settle()
        try:
            # The mail-derived company name must not decide an artefact's file name: the record id does.
            outcome = runner(Lead(**ex.lead), verbose=False, artifact_stem=rid)
        except Exception as exc:  # noqa: BLE001 - one bad lead must not stop the poller or lose the record
            attempts = prep.claim.attempts
            dead = attempts >= max_attempts()
            rec["status"] = "dead_letter" if dead else "failed"
            rec["reason"] = (f"pipeline error: {type(exc).__name__} (run_id={getattr(exc, 'leadscout_run_id', None)}),"
                             f" attempt {attempts} of {max_attempts()}" + ("; giving up" if dead else ""))
            logger.error("inbound %s: pipeline failed (%s), attempt %s", rid, type(exc).__name__, attempts)
            return settle()

    rec["status"], rec["reason"] = "processed", "ok"
    rec["outcome"] = outcome.to_dict()
    rec["next_action"] = next_action(outcome)

    to_addr = reply_address(parsed)
    ok, why = draft_decision(profile=profile, outcome=outcome, flags=flags,
                             auto_generated=parsed.auto_generated, to_addr=to_addr)
    rec["draft"] = {"path": None, "suppressed_reason": None if ok else why, "sent": False}
    if ok:
        msg = build_draft(parsed, ex.lead, profile, to_addr)
        path = inbound_dir(cfg.out_dir) / "drafts" / f"{rid}.eml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(msg))
        rec["draft"]["path"] = str(path)
        may_send, send_why = send_decision(parsed, to_addr, cfg)
        if may_send:
            try:
                rec["draft"]["sent"] = send_draft(msg, cfg)
            except Exception as exc:  # noqa: BLE001 - the draft file already exists; report, do not fail the lead
                rec["draft"]["send_error"] = type(exc).__name__
        else:
            rec["draft"]["send_blocked_reason"] = send_why
    return settle()


def process_message(raw: bytes, *, source: str, pipeline: bool = True, runner=None, ask="default") -> dict:
    return run_prepared(prepare(raw, source=source, pipeline=pipeline), runner=runner, ask=ask)


def resume(*, runner=None, ask="default") -> dict:
    """Re-drive messages that were received but never got a result: `claimed` rows older than
    the stale window (the process died after the claim, e.g. after a webhook's 202) and `failed`
    rows with attempts left. Works from the raw copy saved at claim time."""
    cfg = get_settings()
    store = Store(cfg.out_dir)
    counts: dict[str, int] = {}
    for row in store.resumable():
        raw = store.load_raw(row["record_id"])
        if raw is None:
            rec = _skeleton(row["record_id"], "dead_letter", "raw copy missing; cannot resume", row["source"])
            _write_json(inbound_dir(cfg.out_dir) / "records" / f"{row['record_id']}.json", rec)
            store.finish(row["message_id"], "dead_letter")
            status = "dead_letter"
        else:
            status = process_message(raw, source="resume", runner=runner, ask=ask)["status"]
        counts[status] = counts.get(status, 0) + 1
    return {"counts": counts, "total": sum(counts.values()),
            "errors": counts.get("failed", 0) + counts.get("dead_letter", 0)}
