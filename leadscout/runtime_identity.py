"""Which code is running, decided at run start - not at manifest-write time (F-15).

`scripts/proof_manifest.py` used to be the only witness to "these artefacts came from
this source", and it is written AFTER a proof build, from whatever is on disk at that
moment. Measured: edit a runtime file, re-run `proof_manifest.py write` without running
a single lead, and the gate goes green over artefacts the new code never produced. The
manifest was attesting to its own write time.

So the authority moves into the run. A `RunContext` is created when the run starts,
carries the fingerprint of the runtime source as it was THEN, and is stamped into every
result JSON the run writes. The manifest still records what it sees, but it can no
longer invent the binding: `verify()` reads the fingerprint out of the artefacts
themselves and compares it with the source on disk, so old artefacts plus a new
manifest cannot agree.

The context is immutable and computed once per process. An edit made mid-run therefore
does not retroactively change what the already-started run claims about itself - it
changes the source on disk, which is exactly what the next `verify()` will notice.

The fingerprint covers PATHS as well as bytes, so a rename or deletion changes it too.
It is meaningful over a source tree; a deployment that ships only the package (no
`requirements.txt`/`samples/`) fingerprints the subset it has, and `verify()` always
compares a tree against itself.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Everything whose bytes can change a lead's outcome. The prompts live inside these
# .py files, so they need no separate entry. `leadscout/static/form.html` is
# deliberately excluded: it is the web form's markup, and a proof batch never reads it.
def runtime_files(root: Path | None = None) -> list[Path]:
    root = root or ROOT
    files = sorted(root.joinpath("leadscout").rglob("*.py"))
    # recursive: config/profiles/*.yaml change outcomes just as the policy lists do
    files += sorted(root.joinpath("config").rglob("*.yaml"))
    files += [root / "requirements.txt", root / "samples" / "leads.json"]
    return [f for f in files if f.is_file() and "__pycache__" not in f.parts]


def source_fingerprint(root: Path | None = None) -> tuple[str, dict[str, str]]:
    """(aggregate, per-file) over the runtime source, path-sorted."""
    root = root or ROOT
    per_file = {
        f.relative_to(root).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in runtime_files(root)
    }
    agg = hashlib.sha256()
    for rel, digest in sorted(per_file.items()):
        agg.update(rel.encode())
        agg.update(digest.encode())
    return agg.hexdigest(), per_file


@dataclass(frozen=True)
class RunContext:
    """Immutable identity of one run of this code. `id` distinguishes two runs of the
    same source; `source_fingerprint` is what makes an artefact checkable against a
    tree later."""
    id: str
    source_fingerprint: str
    started_at: str

    def as_binding(self) -> dict:
        return {"id": self.id, "source_fingerprint": self.source_fingerprint,
                "started_at": self.started_at}


_CONTEXT: RunContext | None = None


def context() -> RunContext:
    """The run context for this process, created on first use and never replaced."""
    global _CONTEXT
    if _CONTEXT is None:
        agg, _ = source_fingerprint()
        _CONTEXT = RunContext(id=f"prf-{secrets.token_hex(6)}", source_fingerprint=agg,
                              started_at=datetime.now(UTC).isoformat())
    return _CONTEXT


def reset_context_for_tests() -> None:
    """Only tests call this: a new process is the only other way to get a new context,
    and a test that needs two runs should not have to spawn one."""
    global _CONTEXT
    _CONTEXT = None
