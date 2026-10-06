"""A run must not be able to write into a source tree it was not pointed at.

`config.ROOT` comes from `__file__`, so before `LEADSCOUT_OUTPUT_DIR` existed a run
started from anywhere still wrote into the repository's own `out/`. Measured while
testing batch fault isolation from a temp directory: the run appended rows to the
committed tracker and the provenance ledger, and left two result files and three
snapshot directories behind in the repo. That is the artefact contamination the proof
pipeline has to be safe from, and an explicit-input proof generator does not prevent it -
it only stops the extra files being READ.

These run the CLI in a subprocess, because the output location is resolved at import
time (provenance.py, sanctions.py, ranges.py and providers/footprint.py all capture
`settings.out_dir` when they load).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REPO_OUT = REPO / "out"


def _tree_digest(root: Path) -> str:
    """Content hash of every file under `root`, path-sorted - so an added, removed or
    edited file all change it."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).replace("\\", "/").encode())
        h.update(path.read_bytes())
    return h.hexdigest()


@pytest.mark.skipif(not REPO_OUT.exists(), reason="no committed out/ tree to protect")
def test_an_explicit_output_root_leaves_the_source_tree_untouched(tmp_path):
    before = _tree_digest(REPO_OUT)

    env = dict(os.environ)
    env["LEADSCOUT_OUTPUT_DIR"] = str(tmp_path / "artefacts")
    # No LLM endpoint: the lead will fail, which is fine - what is under test is WHERE
    # anything it does manage to write lands, including the provenance ledger that is
    # opened before the first model call.
    env.pop("OPENROUTER_API_KEY", None)
    env.pop("CLOUDFLARE_API_TOKEN", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("GEMINI_API_KEY", None)
    env["LEADSCOUT_MODELS"] = "openai:nonexistent-model"
    env["LEADSCOUT_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"   # refused fast, never reachable

    subprocess.run(
        [sys.executable, "-m", "leadscout.cli", "--quiet", "lead",
         "--name", "T", "--email", "t@isolation.example",
         "--company", "IsolationProbe", "--website", "https://isolation.example"],
        cwd=REPO, env=env, capture_output=True, timeout=300,
    )

    assert _tree_digest(REPO_OUT) == before, "the run wrote into the repository's out/"


def test_the_output_root_is_where_the_artefacts_actually_land(tmp_path):
    """The other half: pointing it somewhere else has to WORK, not just be ignored."""
    target = tmp_path / "artefacts"
    env = dict(os.environ)
    env["LEADSCOUT_OUTPUT_DIR"] = str(target)
    probe = (
        "import json;"
        "from leadscout.config import settings;"
        "print(json.dumps({"
        "'out': str(settings.out_dir),"
        "'tracker': str(settings.tracker_path),"
        "'outbox': str(settings.outbox_dir)}))"
    )
    proc = subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    paths = json.loads(proc.stdout.strip().splitlines()[-1])
    assert Path(paths["out"]) == target.resolve()
    assert Path(paths["tracker"]) == target.resolve() / "leads_tracker.xlsx"
    assert Path(paths["outbox"]) == target.resolve() / "outbox"


def test_the_default_is_unchanged_when_the_variable_is_absent():
    """A user who sets nothing must still get the documented `out/`."""
    env = {k: v for k, v in os.environ.items() if k != "LEADSCOUT_OUTPUT_DIR"}
    probe = "from leadscout.config import settings; print(settings.out_dir)"
    proc = subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip().splitlines()[-1]) == REPO_OUT
