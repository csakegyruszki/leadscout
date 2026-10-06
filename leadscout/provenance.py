"""Evidence chain-of-custody via provtrail (the author's own package: pip install
provtrail; https://pypi.org/project/provtrail/).
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import UTC, datetime

from provtrail.ledger import Ledger

from .config import settings
from .models import Evidence
from .profile import active_profile
from .profile_config import compliance_config

# --- Design notes -----------------------------------------------------------------
# Every external fact used in a lead's compliance/fit decision - the website text, the
# Wikipedia summary, the OpenSanctions hits, the policy files that defined the
# do-not-engage/sanctioned-jurisdiction lists - gets a snapshot on disk and an
# append-only, hash-chained ledger record proving what was fetched, when, and its exact
# content hash. A decision can then be traced back to the bytes it was made from, not
# just re-described from memory.
#
# What the ledger does NOT prove: that the *source itself* was accurate (a company's
# own website can lie about its size), or that the content hasn't changed since it was
# captured (only that THIS snapshot, at THIS hash, existed at THIS time) - see the
# "Provenance" section in the README for the same caveat in provtrail's own words.
#
# Never records the lead's contact name or email: `extra["company"]` carries only the
# company name, and the snapshot bytes are always the source's own content (a website's
# about text, an OpenSanctions hit, a policy file), never the submitted lead payload.

logger = logging.getLogger("leadscout")

LEDGER_DIR = settings.out_dir / "provenance"
LEDGER_PATH = LEDGER_DIR / "ledger.jsonl"
SNAPSHOT_DIR = LEDGER_DIR / "snapshots"

# provtrail's ALLOWED_KINDS is {"url", "search", "scrape", "file", "manual"} - there
# is no "policy" kind, so a policy YAML file (no fetchable URL) is recorded as "file".
_KIND_BY_SOURCE = {
    "website": "url", "website_subpage": "url", "wikipedia": "url",
    "opensanctions": "search", "policy": "file", "hunter": "url",
}
# Explicit per-provider mapping (REVIEW-6bA-verified.md #4): the old default-to-"txt"
# fallback silently mismatched every provider that declares a ".json" extension in
# its own Evidence.snapshot_path (ats/github/gleif/wikidata), so the file provenance
# actually wrote never existed at the path the Evidence claimed. Each entry here is
# picked to match what that provider's Evidence.snapshot_path already says, not
# guessed independently - see each provider module's `_finalise`/Evidence construction.
_EXT_BY_SOURCE = {
    "website": "txt", "website_subpage": "txt", "wikipedia": "txt",
    "opensanctions": "json", "policy": "yaml",
    "ats": "json", "github": "json", "gleif": "json", "wikidata": "json",
    "footprint": "txt", "trust_pages": "txt", "vendor": "html", "hunter": "json",
}
# OpenSanctions match is a POST endpoint, not a browsable page, but it IS the real,
# stable URL the data came from, so it satisfies the ledger's source_url requirement.
_OPENSANCTIONS_URL = "https://api.opensanctions.org/match/default"


class ProvenanceRun:
    """One provenance run per lead. Create with `start()`, call `record()` for each
    piece of external evidence, call `finish()` once at the end of the lead."""

    def __init__(self, run_id: str, company: str) -> None:
        self.run_id = run_id
        self.company = company
        self._ledger = Ledger(str(LEDGER_PATH))
        self._count = 0
        self._had_error = False
        self._next_seq = 1

    @classmethod
    def start(cls, company: str) -> ProvenanceRun:
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        return cls(run_id=str(uuid.uuid4()), company=company)

    def next_evidence_id(self) -> str:
        eid = f"ev-{self._next_seq:03d}"
        self._next_seq += 1
        return eid

    def record(self, evidence: Evidence, raw_bytes: bytes, *, stage: str = "") -> None:
        """Write the snapshot file and append one ledger record.

        Never raises: any failure here (disk, lock timeout, a malformed record) is
        logged and flips this run to "degraded" in finish() - the pipeline's
        compliance/fit results are computed before this is ever called and are
        never touched by a provenance failure.
        """
        try:
            ext = _EXT_BY_SOURCE.get(evidence.source_type, "txt")
            run_dir = SNAPSHOT_DIR / self.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = run_dir / f"{evidence.id}.{ext}"
            snapshot_path.write_bytes(raw_bytes)
            rel_path = str(snapshot_path.relative_to(LEDGER_DIR)).replace("\\", "/")

            source_url = evidence.url or (
                _OPENSANCTIONS_URL if evidence.source_type == "opensanctions" else None)

            self._ledger.add(
                source_url=source_url,
                content_path=str(snapshot_path),
                content_root=str(LEDGER_DIR),
                path=rel_path,
                kind=_KIND_BY_SOURCE.get(evidence.source_type, "manual"),
                tool=f"leadscout:{evidence.source_type}",
                title=evidence.url or evidence.source_type,
                snippet=evidence.snippet[:200],
                session_id=self.run_id,
                extra={
                    "stage": stage,
                    "evidence_type": evidence.source_type,
                    "strength": evidence.strength,
                    "company": self.company,  # company name only - never contact PII
                },
            )
            self._count += 1
        except Exception:  # noqa: BLE001 - provenance must never break the pipeline
            self._had_error = True
            logger.exception("provenance record failed for %s (%s)", evidence.id, self.company)

    def finish(self) -> tuple[str, str, int]:
        """Verify the ledger (with file checks) and return (status, head, count)."""
        try:
            report = self._ledger.verify(check_files=True)
            head = self._ledger.head()
            head_str = f"{head[0]}:{head[1][:16]}" if head else "none"
            status = "ok" if (report.ok and not self._had_error) else "degraded"
            return status, head_str, self._count
        except Exception:  # noqa: BLE001
            logger.exception("provenance verify failed (%s)", self.company)
            return "degraded", "none", self._count

    def _mark_terminal(self, *, claim: str, extra: dict) -> None:
        """Append one terminal-state marker record. Never raises: same contract as
        record() - a marker-write failure must not take down the pipeline, it only
        degrades the run (mirrored via _had_error, same as record())."""
        try:
            self._ledger.add(
                content=claim,  # no source_url for a marker; content_hash satisfies
                               # provtrail's "source_url or content_hash" requirement
                kind="manual",
                tool="leadscout:run",
                claim=claim,
                session_id=self.run_id,
                extra=extra,
            )
        except Exception:  # noqa: BLE001 - provenance must never break the pipeline
            self._had_error = True
            logger.exception("provenance terminal marker failed for run %s (%s)", self.run_id, self.company)

    def mark_aborted(self, reason: str) -> None:
        """Append the run_aborted marker. Called from pipeline.process_lead's
        except-clause when a lead's run raises after a ProvenanceRun has started."""
        self._mark_terminal(
            claim="run aborted",
            extra={"stage": "run_aborted", "reason": reason, "run_id": self.run_id},
        )

    def mark_completed(self) -> None:
        """Append the run_completed marker. Called at the end of a normal,
        successful run so both terminal states are explicit in the ledger."""
        self._mark_terminal(
            claim="run completed",
            extra={"stage": "run_completed", "run_id": self.run_id},
        )


def run_status(run_id: str) -> str:
    """Return "completed"|"aborted"|"unknown" for `run_id` by scanning the ledger
    for its run_completed/run_aborted marker record (see ProvenanceRun.mark_*).

    Scans the whole ledger (not just this run's records) because the ledger is a
    single shared append-only file across all runs - there is no per-run file to
    open instead. "unknown" covers both "no marker was ever written" (e.g. the
    process was killed, not just an exception) and "run_id was never seen".
    """
    ledger = Ledger(str(LEDGER_PATH))
    status = "unknown"
    for rec in ledger.records():
        if rec.get("session_id") != run_id:
            continue
        stage = (rec.get("extra") or {}).get("stage")
        if stage == "run_completed":
            status = "completed"
        elif stage == "run_aborted":
            status = "aborted"
    return status


def record_policy_files(run: ProvenanceRun) -> None:
    """Record both policy files (do-not-engage list, sanctioned jurisdictions) as
    evidence at the start of a run, so the ledger proves which policy VERSION this
    lead's decision used - not just that a policy existed."""
    # The policy files THIS run's profile points at, plus the profile itself: a profile
    # changes outcomes (fit weights, competitor list, prompts) as much as a list does.
    profile = active_profile()
    cfg = compliance_config(profile)
    paths = [p for p in (cfg.competitors_path, cfg.jurisdictions_path) if p is not None]
    paths.append(profile.path)
    for path in paths:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        eid = run.next_evidence_id()
        evidence = Evidence(
            id=eid,
            source_type="policy",
            url=f"file://{path.as_posix()}",
            observed_at=datetime.now(UTC).isoformat(),
            content_sha256=hashlib.sha256(raw).hexdigest(),
            strength="STRONG",
            snippet=raw[:200].decode("utf-8", errors="replace"),
            snapshot_path=f"out/provenance/snapshots/{run.run_id}/{eid}.yaml",
            family="policy",
        )
        run.record(evidence, raw, stage="policy")
