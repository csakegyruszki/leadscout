"""Idempotency store: one SQLite file under <out>/inbound/.

Primary key is the normalised Message-ID; a second, content-based key catches the same
mail re-sent under a fresh Message-ID. A message is CLAIMED (inserted) before any work
starts, inside a write transaction, so concurrent deliveries of one mail - webhook
retries, two poller runs - produce exactly one winner; everyone else gets the winner's
record id back.

A claim that never finished (process killed) is taken over after STALE_CLAIM_SECONDS,
and a record that ended `failed` may be claimed again while it has attempts left (the
service turns the last failed attempt into the terminal `dead_letter`). The raw message is
written to <out>/inbound/raw/<id>.eml at claim time, which is what lets `mail resume`
re-drive a message whose process died after the webhook had already answered 202.
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

STALE_CLAIM_SECONDS = 900
DEFAULT_MAX_ATTEMPTS = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id   TEXT PRIMARY KEY,
    record_id    TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status       TEXT NOT NULL,
    duplicate_of TEXT,
    source       TEXT NOT NULL DEFAULT '',
    claimed_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS messages_content_hash ON messages(content_hash);
"""


def max_attempts() -> int:
    raw = os.getenv("LEADSCOUT_INBOUND_MAX_ATTEMPTS", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_MAX_ATTEMPTS
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS
    return value if value > 0 else DEFAULT_MAX_ATTEMPTS


def inbound_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "inbound"


def safe_id(message_id: str) -> str:
    return hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:24]


def content_hash(from_addr: str, subject: str, body: str) -> str:
    """Normalised sender + subject + body, so a re-send that differs only in headers,
    whitespace, case, a `Re:`/`Fwd:` prefix or the signature-less quoted tail collapses."""
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower()).strip()
    subj = re.sub(r"^(?:(?:re|fw|fwd|aw|wg)\s*:\s*)+", "", norm(subject))
    return hashlib.sha256(f"{norm(from_addr)}\n{subj}\n{norm(body)}".encode()).hexdigest()


@dataclass
class Claim:
    is_new: bool
    record_id: str
    status: str                  # status of the row now holding this message
    duplicate_of: str | None = None
    reason: str = ""
    attempts: int = 1


class Store:
    def __init__(self, out_dir: Path) -> None:
        self.dir = inbound_dir(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.dir / "raw"
        self.path = self.dir / "inbound.sqlite3"
        with closing(self._connect()) as db:
            db.executescript(_SCHEMA)
            cols = {r["name"] for r in db.execute("PRAGMA table_info(messages)")}
            if "attempts" not in cols:          # a store created before attempt counting
                db.execute("ALTER TABLE messages ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)  # explicit transactions
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout = 30000")
        return db

    def claim(self, message_id: str, chash: str, source: str = "", *, status: str = "claimed") -> Claim:
        rid, now = safe_id(message_id), time.time()
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")          # takes the write lock: the check and the insert are one step
            try:
                row = db.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,)).fetchone()
                if row is not None:
                    retry = (row["status"] == "failed" and row["attempts"] < max_attempts()) or (
                        row["status"] == "claimed" and now - row["claimed_at"] > STALE_CLAIM_SECONDS)
                    if retry:
                        attempts = row["attempts"] + 1
                        db.execute("UPDATE messages SET status=?, claimed_at=?, updated_at=?, source=?, attempts=? "
                                   "WHERE message_id=?", (status, now, now, source, attempts, message_id))
                        db.execute("COMMIT")
                        return Claim(True, row["record_id"], status, reason="retry", attempts=attempts)
                    db.execute("COMMIT")
                    return Claim(False, row["duplicate_of"] or row["record_id"], row["status"],
                                 duplicate_of=row["duplicate_of"] or row["record_id"], reason="message_id",
                                 attempts=row["attempts"])
                twin = db.execute(
                    "SELECT record_id FROM messages WHERE content_hash = ? AND duplicate_of IS NULL "
                    "AND status NOT IN ('failed', 'rejected', 'dead_letter') ORDER BY claimed_at LIMIT 1",
                    (chash,)).fetchone()
                if twin is not None:
                    db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)",
                               (message_id, rid, chash, "duplicate", twin["record_id"], source, now, now, 1))
                    db.execute("COMMIT")
                    return Claim(False, twin["record_id"], "duplicate", duplicate_of=twin["record_id"],
                                 reason="content_hash")
                db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)",
                           (message_id, rid, chash, status, None, source, now, now, 1))
                db.execute("COMMIT")
                return Claim(True, rid, status)
            except BaseException:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    def finish(self, message_id: str, status: str) -> None:
        with closing(self._connect()) as db:
            db.execute("UPDATE messages SET status=?, updated_at=? WHERE message_id=?",
                       (status, time.time(), message_id))

    def get(self, message_id: str) -> dict | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,)).fetchone()
        return dict(row) if row else None

    # --- raw copies, for resume ---------------------------------------------------------------

    def save_raw(self, record_id: str, raw: bytes) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"{record_id}.eml"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, path)
        return path

    def load_raw(self, record_id: str) -> bytes | None:
        path = self.raw_dir / f"{record_id}.eml"
        return path.read_bytes() if path.is_file() else None

    def resumable(self) -> list[dict]:
        """Rows that were received but have no result: `claimed` longer than the stale window,
        and `failed` with attempts left. Oldest first."""
        now = time.time()
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT * FROM messages WHERE duplicate_of IS NULL AND "
                "((status = 'claimed' AND ? - claimed_at > ?) OR (status = 'failed' AND attempts < ?)) "
                "ORDER BY claimed_at", (now, STALE_CLAIM_SECONDS, max_attempts())).fetchall()
        return [dict(r) for r in rows]
