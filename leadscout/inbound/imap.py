"""IMAP poller (stdlib imaplib) behind the small `MailSource` interface.

Messages are fetched with BODY.PEEK[] so reading never sets \\Seen; a message is marked
\\Seen (and optionally moved) only by `ack(ref, ok=True)`, which the poller calls after
the record has been written. The password is read from the environment into the login
call and nowhere else: it is never logged and never part of an exception message.
"""
from __future__ import annotations

import imaplib
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from . import service

logger = logging.getLogger("leadscout")


class MailSource(Protocol):
    def fetch(self) -> Iterator[tuple[str, bytes | int | None]]:
        """Yield (ref, raw RFC 822 bytes); (ref, size) for a message over the cap that was not
        downloaded; (ref, None) for a failed fetch. `ref` is opaque to the poller."""

    def ack(self, ref: str, ok: bool) -> None:
        """Called once per fetched message after processing; ok=False leaves it for a retry."""


class ImapConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ImapConfig:
    host: str
    user: str
    password: str = ""
    port: int = 993
    folder: str = "INBOX"
    processed_folder: str = ""
    max_per_poll: int = 25

    @classmethod
    def from_env(cls) -> ImapConfig:
        host = os.getenv("LEADSCOUT_IMAP_HOST", "").strip()
        user = os.getenv("LEADSCOUT_IMAP_USER", "").strip()
        password = os.getenv("LEADSCOUT_IMAP_PASSWORD", "")
        if not (host and user and password):
            raise ImapConfigError(
                "set LEADSCOUT_IMAP_HOST, LEADSCOUT_IMAP_USER and LEADSCOUT_IMAP_PASSWORD to poll a mailbox")

        def num(name: str, raw: str, default: int) -> int:
            try:
                return int(raw.strip()) if raw.strip() else default
            except ValueError as exc:
                raise ImapConfigError(f"{name} must be an integer") from exc
        return cls(host=host, user=user, password=password,
                   port=num("LEADSCOUT_IMAP_PORT", os.getenv("LEADSCOUT_IMAP_PORT", ""), 993),
                   folder=os.getenv("LEADSCOUT_IMAP_FOLDER", "").strip() or "INBOX",
                   processed_folder=os.getenv("LEADSCOUT_IMAP_PROCESSED_FOLDER", "").strip(),
                   max_per_poll=max(1, num("LEADSCOUT_IMAP_MAX_PER_POLL",
                                           os.getenv("LEADSCOUT_IMAP_MAX_PER_POLL", ""), 25)))

    def __repr__(self) -> str:   # the password must not survive a stray log line or traceback
        return (f"ImapConfig(host={self.host!r}, user={self.user!r}, port={self.port}, "
                f"folder={self.folder!r}, processed_folder={self.processed_folder!r})")


def _quote(folder: str) -> str:
    return '"' + folder.replace("\\", "\\\\").replace('"', '\\"') + '"'


_SIZE = re.compile(rb"RFC822\.SIZE\s+(\d+)", re.I)


class ImapSource:
    def __init__(self, config: ImapConfig, client_factory=None) -> None:
        self.cfg = config
        self._factory = client_factory
        self._client = None

    def _connect(self):
        if self._client is None:
            factory = self._factory or (lambda h, p: imaplib.IMAP4_SSL(h, p, timeout=30))
            client = factory(self.cfg.host, self.cfg.port)
            try:
                client.login(self.cfg.user, self.cfg.password)
            except imaplib.IMAP4.error:
                # imaplib's message can quote the server's reply; keep the cause out of the log.
                raise ImapConfigError("IMAP login failed (check user / app password)") from None
            typ, _ = client.select(_quote(self.cfg.folder))
            if typ != "OK":
                raise ImapConfigError(f"cannot select IMAP folder {self.cfg.folder!r}")
            self._client = client
        return self._client

    def _advertised_size(self, c, ref: str) -> int | None:
        typ, parts = c.uid("FETCH", ref, "(RFC822.SIZE)")
        if typ != "OK":
            return None
        for p in parts or []:
            blob = p[0] if isinstance(p, tuple) else p
            m = _SIZE.search(blob if isinstance(blob, bytes) else str(blob).encode())
            if m:
                return int(m.group(1))
        return None

    def fetch(self) -> Iterator[tuple[str, bytes | int | None]]:
        """Yields (ref, raw bytes); (ref, size:int) for a message over the size cap that was NOT
        downloaded; (ref, None) when the fetch failed."""
        c = self._connect()
        typ, data = c.uid("SEARCH", None, "UNSEEN")
        if typ != "OK" or not data or not data[0]:
            return
        cap = service.max_message_bytes()
        for uid in data[0].split()[: self.cfg.max_per_poll]:
            ref = uid.decode() if isinstance(uid, bytes) else str(uid)
            size = self._advertised_size(c, ref)
            if size is not None and size > cap:
                yield ref, size                      # decided before the body crosses the wire
                continue
            typ, parts = c.uid("FETCH", ref, "(BODY.PEEK[])")
            raw = next((p[1] for p in parts or [] if isinstance(p, tuple) and len(p) > 1), None) \
                if typ == "OK" else None
            if not raw:
                logger.warning("imap: fetch of uid %s failed", ref)
                yield ref, None
                continue
            yield ref, bytes(raw)

    def ack(self, ref: str, ok: bool) -> None:
        if not ok or self._client is None:
            return
        c = self._client
        c.uid("STORE", ref, "+FLAGS", "(\\Seen)")
        if self.cfg.processed_folder:
            typ, _ = c.uid("COPY", ref, _quote(self.cfg.processed_folder))
            if typ != "OK":
                logger.warning("imap: copy of uid %s to %r failed; left in place", ref, self.cfg.processed_folder)
                return
            c.uid("STORE", ref, "+FLAGS", "(\\Deleted)")
            # Plain EXPUNGE removes EVERY \Deleted message in the folder, including ones someone
            # else flagged. Only UID EXPUNGE (UIDPLUS) is scoped to this message.
            if "UIDPLUS" in {str(x).upper() for x in getattr(c, "capabilities", ())}:
                c.uid("EXPUNGE", ref)
            else:
                logger.warning("imap: server lacks UIDPLUS; uid %s is flagged \\Deleted but not expunged", ref)

    def close(self) -> None:
        c, self._client = self._client, None
        if c is None:
            return
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            c.logout()
        except Exception:  # noqa: BLE001
            pass


def poll_once(source: MailSource | None = None, *, process=None) -> dict:
    """One pass over the mailbox. Returns counts by record status plus errors. A message is
    acknowledged (marked \\Seen) when a record was written for it, including the terminal
    `dead_letter`; `failed` (attempts left) and fetch failures are left for the next pass."""
    process = process or (lambda raw: service.process_message(raw, source="imap"))
    own = source is None
    source = source or ImapSource(ImapConfig.from_env())
    counts: dict[str, int] = {}
    errors = 0
    try:
        for ref, raw in source.fetch():
            try:
                if raw is None:
                    rec, ok = {"status": "error"}, False
                elif isinstance(raw, int):
                    rec = service.reject_oversize(f"{getattr(source, 'cfg', None)!r}:{ref}", raw, source="imap")
                    ok = True
                else:
                    rec = process(raw)
                    ok = rec.get("status") != "failed"
            except Exception as exc:  # noqa: BLE001 - one message must not stop the pass
                logger.error("imap: processing uid %s raised %s", ref, type(exc).__name__)
                rec, ok = {"status": "error"}, False
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            errors += 0 if ok else 1
            source.ack(ref, ok)
    finally:
        if own and hasattr(source, "close"):
            source.close()
    return {"counts": counts, "errors": errors, "total": sum(counts.values())}
