#!/usr/bin/env python
"""Bind the proof artefacts to the exact runtime source that produced them.

    python scripts/proof_manifest.py write     # after a proof build
    python scripts/proof_manifest.py verify    # gate; also run by tests/test_proof_freshness.py

WHY THIS EXISTS. A readme or a result set can only prove it matches itself. It cannot
prove the results equal the output of the code that is in the tree, so a sample batch can
silently predate later decision-relevant commits while every other check stays green.

WHY NOT A GIT SHA. A fresh history (a squash, a mirror, a fork) has a different commit id
for byte-identical code. A fingerprint over the runtime FILES survives that, and makes the
three-way equality checkable:

    proof manifest fingerprint == repository fingerprint == deployed image fingerprint

WHY THE MANIFEST IS NOT THE AUTHORITY (F-15). Hashing whatever is on disk at write
time and then checking it later proves only that nothing moved SINCE the write. It said
nothing about whether the artefacts came from that code: editing a runtime file and
re-running `write`, with no batch in between, turned the gate green over artefacts the
new code never produced. The binding now travels inside the artefacts - each result JSON
carries the `proof_run` block stamped by `leadscout/runtime_identity.py` when the run
started - and both `write` and `verify` read it from there. This file records and checks;
it no longer certifies itself.

WHAT THE MANIFEST DOES AND DOES NOT CLAIM. The sample-batch leads are built from LIVE
provider and model calls, so the proof is not deterministically reproducible and this file
does not pretend otherwise. Read it as three separate claims:

    source_fingerprint   code identity      - the runtime bytes that produced the proof
    artefacts            observed proof     - what those bytes actually produced, once
    generated_at         observation time   - when the outside world looked like that
    execution            audit metadata     - interpreter, profile, model chain, which
                                             optional providers were enabled

What IS reproducible: the same code, the same artefacts, and the same decisions
in the frozen replay corpus. Re-running the live batch on another day may legitimately
differ - a provider's answer changed, a model chose different words - and the fingerprint
plus the execution block is what makes that explainable rather than suspicious.

The manifest never hashes itself.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR_ENV = os.getenv("LEADSCOUT_OUTPUT_DIR", "").strip()
OUT_DIR = Path(_OUT_DIR_ENV).expanduser().resolve() if _OUT_DIR_ENV else ROOT / "out"
MANIFEST = OUT_DIR / "proof_manifest.json"
PROOF_IMAGES = ROOT / "docs" / "proof"

# Run as a script from anywhere, so the package this gate is about is importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from leadscout import runtime_identity  # noqa: E402, I001


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _runtime_files() -> list[Path]:
    return runtime_identity.runtime_files(ROOT)


def source_fingerprint() -> tuple[str, dict[str, str]]:
    """(aggregate, per-file) over the runtime source, path-sorted.

    One definition, in `leadscout/runtime_identity.py`, because the RUN now computes
    the same fingerprint at its start and stamps it into every result JSON - two copies
    of this rule would let the two halves of the F-15 check drift apart. `ROOT` is read
    at call time so a test can point this module at a copied tree.
    """
    return runtime_identity.source_fingerprint(ROOT)


def _result_bindings() -> dict[str, dict]:
    """The `proof_run` block each result JSON carries, keyed by file name.

    This is the evidence the freshness check is built on: it was written BY the run,
    before this script existed in the sequence. A result with no block (or an unreadable
    one) is recorded as an empty dict and fails the checks below - "no binding" is not
    "binding matches", and a pre-F-15 artefact must not read as fresh.
    """
    out: dict[str, dict] = {}
    for path in sorted((OUT_DIR / "results").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            block = data.get("proof_run") or {}
        except (OSError, json.JSONDecodeError):
            block = {}
        out[path.name] = block if isinstance(block, dict) else {}
    return out


def _binding_problems(agg: str) -> list[str]:
    """Every way the artefacts' own account of themselves can fail against `agg`."""
    bindings = _result_bindings()
    problems: list[str] = []
    if not bindings:
        problems.append("no result JSONs found in out/results - there is nothing to bind")
    for name, block in sorted(bindings.items()):
        if not block.get("source_fingerprint") or not block.get("id"):
            problems.append(
                f"results/{name} carries no proof_run binding: it cannot say which code "
                f"produced it, so it cannot be proof for any code. Re-run the proof build.")
        elif block["source_fingerprint"] != agg:
            problems.append(
                f"results/{name} was produced by source_fingerprint "
                f"{block['source_fingerprint'][:16]}..., the tree is now {agg[:16]}... - "
                f"these artefacts are not this code's output. Re-run the proof build.")
    run_ids = {b.get("id") for b in bindings.values() if b.get("id")}
    if len(run_ids) > 1:
        problems.append(
            f"the results come from {len(run_ids)} different runs ({', '.join(sorted(run_ids))}) - "
            f"a proof batch is one run, so this set was assembled, not produced")
    return problems


def _artefact_hashes() -> dict[str, dict[str, str]]:
    groups = {
        "results": sorted((OUT_DIR / "results").glob("*.json")),
        "notifications": sorted((OUT_DIR / "outbox").glob("*.eml")),
        "tracker": [OUT_DIR / "leads_tracker.xlsx"],
        "screenshots": sorted(PROOF_IMAGES.glob("*.png")),
    }
    out: dict[str, dict[str, str]] = {}
    for name, paths in groups.items():
        out[name] = {p.name: _sha256(p) for p in paths if p.is_file()}
    return out


def _execution_metadata() -> dict:
    """What the proof was observed IN, as opposed to what it was produced BY.

    Deliberately NOT part of `source_fingerprint`: the fingerprint answers "is this the
    same code", and mixing the environment into it would invalidate the proof whenever an
    interpreter patch version changed. This block answers the other question - why the
    same code can legitimately observe something different later, because the sample-batch
    leads are built from LIVE provider and model calls.

    Read from `settings` rather than re-read from the environment, so it records what the
    run actually used. Values are identifiers and enable/disable flags only: no key
    material, and `bool(...)` on a credential records only WHETHER one was present.
    """
    from leadscout.config import settings  # local: keeps `verify` usable without a full import

    return {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "profile": settings.profile,
        "model_chain": list(settings.models),
        "llm_hop_timeout": settings.llm_hop_timeout,
        "fit_threshold": settings.fit_threshold,
        "proof_mode": bool(getattr(settings, "proof_mode", False)),
        "providers_enabled": {
            "opensanctions": bool(settings.opensanctions_api_key),
            "hunter": bool(settings.hunter_api_key),
            "diffbot": bool(settings.diffbot_token),
            "brave_vendor_search": bool(settings.brave_api_key),
            "github_token": bool(getattr(settings, "github_token", "")),
            "js_render": os.getenv("LEADSCOUT_JS_RENDER", "") == "1",
        },
        "smtp_configured": bool(settings.smtp_host and settings.smtp_user and settings.smtp_password),
    }


def build() -> dict:
    agg, per_file = source_fingerprint()
    # F-15: a manifest may only be written over artefacts that already say, themselves,
    # that this source produced them. Refusing here is the point - "write the manifest
    # again" was precisely the move that used to re-freshen a stale proof.
    problems = _binding_problems(agg)
    if problems:
        raise SystemExit("cannot write a proof manifest:\n- " + "\n- ".join(problems))
    bindings = _result_bindings()
    samples = ROOT / "samples" / "leads.json"
    expected = [lead["company"] for lead in json.loads(samples.read_text(encoding="utf-8"))]
    return {
        "schema": "leadscout/proof-manifest/1",
        "generated_at": datetime.now(UTC).isoformat(),
        # Code identity. Equal fingerprints mean equal runtime bytes, whatever the commit
        # id - which is what makes a repo with fresh history checkable against
        # this proof and against the deployed image.
        "source_fingerprint": agg,
        "source_files": per_file,
        # Copied OUT of the artefacts, never minted here (F-15): the run said this, the
        # manifest only repeats it, and `verify` re-reads the artefacts rather than
        # trusting this line.
        "proof_run": sorted({b["id"] for b in bindings.values()})[0],
        "expected_leads": sorted(expected),
        "proof_input_sha256": _sha256(samples),
        # Actual observed proof: these hashes are what the run produced, at that moment.
        "artefacts": _artefact_hashes(),
        # Audit metadata, never part of the fingerprint.
        "execution": _execution_metadata(),
    }


def write() -> int:
    manifest = build()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {_rel(MANIFEST) if MANIFEST.is_relative_to(ROOT) else MANIFEST}")
    print(f"  source_fingerprint {manifest['source_fingerprint']}")
    print(f"  {len(manifest['source_files'])} runtime files, "
          f"{sum(len(v) for v in manifest['artefacts'].values())} artefacts")
    return 0


def verify() -> int:
    """Non-zero, with the specific reason, on any drift."""
    if not MANIFEST.is_file():
        print(f"MISSING: {MANIFEST} - run `python scripts/proof_manifest.py write` "
              f"after a proof build", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems: list[str] = []

    agg, per_file = source_fingerprint()
    label = "proof freshness"

    # F-15, first and decisive: what the ARTEFACTS say about the code that produced
    # them, checked against the tree. This does not consult the manifest at all, so a
    # freshly written manifest over stale results cannot make it pass.
    problems.extend(_binding_problems(agg))
    recorded_run = manifest.get("proof_run")
    artefact_runs = {b.get("id") for b in _result_bindings().values() if b.get("id")}
    if recorded_run and artefact_runs and recorded_run not in artefact_runs:
        problems.append(
            f"the manifest names run {recorded_run}, the results were produced by "
            f"{', '.join(sorted(artefact_runs))} - the manifest describes a different build")

    if agg != manifest.get("source_fingerprint"):
        changed = sorted(k for k in set(per_file) | set(manifest.get("source_files", {}))
                         if per_file.get(k) != manifest.get("source_files", {}).get(k))
        problems.append(
            "the runtime source changed since the proof was generated, so the "
            "artefacts no longer describe this code. Re-run the proof build.\n  "
            + "\n  ".join(changed[:25]) + ("\n  ..." if len(changed) > 25 else ""))

    for group, recorded in manifest.get("artefacts", {}).items():
        current = _artefact_hashes().get(group, {})
        for name, digest in recorded.items():
            if name not in current:
                problems.append(f"{group}/{name} is recorded in the manifest but missing on disk")
            elif current[name] != digest:
                problems.append(f"{group}/{name} changed after the manifest was written")
        for name in current:
            if name not in recorded:
                problems.append(f"{group}/{name} exists on disk but is not in the manifest")

    if problems:
        print(f"{label} FAILED:", file=sys.stderr)
        for p in problems:
            print(f"- {p}", file=sys.stderr)
        return 1
    print(f"{label} OK: source_fingerprint {agg}")
    return 0


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if action not in ("write", "verify"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(write() if action == "write" else verify())
