"""A proof batch must be provably the output of the code that is in the tree.

A readme or result set can only prove it matches itself, not that it came from this code.
The fingerprint is over the runtime FILES, not a commit id, so a repository with a fresh
history and byte-identical code gets the same value.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_proof_manifest():
    spec = importlib.util.spec_from_file_location(
        "proof_manifest", ROOT / "scripts" / "proof_manifest.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["proof_manifest"] = module
    spec.loader.exec_module(module)
    return module


pm = _load_proof_manifest()


def test_the_fingerprint_covers_every_file_that_can_change_an_outcome():
    _, per_file = pm.source_fingerprint()
    assert "requirements.txt" in per_file
    assert "samples/leads.json" in per_file
    assert "config/do_not_engage.yaml" in per_file
    assert "config/restricted_jurisdictions.yaml" in per_file
    # The decision modules and the prompts that live inside them.
    for module in ("compliance", "research", "cloud", "fit", "pipeline", "tracker", "notify"):
        assert f"leadscout/{module}.py" in per_file, module
    assert any(k.startswith("leadscout/providers/") for k in per_file)
    assert not any("__pycache__" in k for k in per_file)


def test_the_fingerprint_is_stable_across_calls():
    first, _ = pm.source_fingerprint()
    second, _ = pm.source_fingerprint()
    assert first == second


def test_editing_a_runtime_file_changes_the_fingerprint(tmp_path, monkeypatch):
    """The whole point: a change to the code that decides outcomes must invalidate the
    proof. Exercised on a copy so the real tree is never touched."""
    import shutil

    copy_root = tmp_path / "repo"
    for rel in ("leadscout", "config"):
        shutil.copytree(ROOT / rel, copy_root / rel,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for rel in ("requirements.txt", "samples/leads.json"):
        (copy_root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, copy_root / rel)

    monkeypatch.setattr(pm, "ROOT", copy_root)
    before, _ = pm.source_fingerprint()

    target = copy_root / "leadscout" / "compliance.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# a decision-relevant edit\n",
                      encoding="utf-8")
    after, _ = pm.source_fingerprint()
    assert after != before, "editing compliance.py did not change the fingerprint"


def test_renaming_a_runtime_file_changes_the_fingerprint(tmp_path, monkeypatch):
    """The aggregate covers paths as well as contents, so a rename or a deletion is
    detected even when no surviving file's bytes changed."""
    import shutil

    copy_root = tmp_path / "repo"
    shutil.copytree(ROOT / "leadscout", copy_root / "leadscout",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "config", copy_root / "config")
    for rel in ("requirements.txt", "samples/leads.json"):
        (copy_root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, copy_root / rel)

    monkeypatch.setattr(pm, "ROOT", copy_root)
    before, _ = pm.source_fingerprint()
    (copy_root / "leadscout" / "util.py").rename(copy_root / "leadscout" / "utils.py")
    after, _ = pm.source_fingerprint()
    assert after != before


def _bound_proof_tree(tmp_path, monkeypatch, *, run_id: str = "prf-testrun0001") -> Path:
    """A copied tree whose result JSONs carry a proof_run binding for THAT tree's own
    source - what a proof build leaves behind once the run stamps its context in.
    Nothing here writes into the real repository."""
    import shutil

    root = tmp_path / "repo"
    for rel in ("leadscout", "config"):
        shutil.copytree(ROOT / rel, root / rel, ignore=shutil.ignore_patterns("__pycache__"))
    for rel in ("requirements.txt", "samples/leads.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, root / rel)
    results = root / "out" / "results"
    results.mkdir(parents=True)
    for i in range(3):
        (results / f"lead-{i}.json").write_text(json.dumps({"lead": f"lead-{i}"}), encoding="utf-8")

    monkeypatch.setattr(pm, "ROOT", root)
    monkeypatch.setattr(pm, "OUT_DIR", root / "out")
    monkeypatch.setattr(pm, "MANIFEST", root / "out" / "proof_manifest.json")
    monkeypatch.setattr(pm, "PROOF_IMAGES", root / "docs" / "proof")

    agg, _ = pm.source_fingerprint()
    for path in (root / "out" / "results").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["proof_run"] = {"id": run_id, "source_fingerprint": agg,
                            "started_at": "2026-09-20T00:00:00+00:00"}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return root


def _edit_runtime_source(root: Path) -> None:
    target = root / "leadscout" / "compliance.py"
    target.write_text(target.read_text(encoding="utf-8")
                      + "\n# a decision-relevant edit made after the proof was produced\n",
                      encoding="utf-8")


def test_a_bound_proof_verifies(tmp_path, monkeypatch):
    """The control: artefacts that say they came from this source, and did."""
    _bound_proof_tree(tmp_path, monkeypatch)
    assert pm.write() == 0
    assert pm.verify() == 0


def test_rewriting_the_manifest_cannot_refresh_a_stale_proof(tmp_path, monkeypatch):
    """F-15, the finding itself. Editing a runtime file must invalidate the proof, and
    re-running `write` - with no batch in between, so the artefacts are byte-identical -
    must not be able to declare it fresh again. This sequence used to go green: the
    manifest was the only witness, and it was attesting to its own write time."""
    root = _bound_proof_tree(tmp_path, monkeypatch)
    assert pm.write() == 0
    assert pm.verify() == 0

    _edit_runtime_source(root)
    assert pm.verify() == 1

    with pytest.raises(SystemExit):
        pm.write()
    assert pm.verify() == 1, "a rewritten manifest refreshed artefacts the new code never produced"


def test_a_result_without_a_binding_is_not_proof(tmp_path, monkeypatch):
    """A pre-F-15 artefact says nothing about which code produced it: "no binding" must
    not read as "binding matches"."""
    root = _bound_proof_tree(tmp_path, monkeypatch)
    victim = sorted((root / "out" / "results").glob("*.json"))[0]
    data = json.loads(victim.read_text(encoding="utf-8"))
    del data["proof_run"]
    victim.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SystemExit):
        pm.write()


def test_results_from_two_runs_are_not_one_proof_batch(tmp_path, monkeypatch):
    """Same source, two runs: a set assembled from several builds is not the output of
    the build the manifest describes, even when every fingerprint agrees."""
    root = _bound_proof_tree(tmp_path, monkeypatch)
    agg, _ = pm.source_fingerprint()
    victim = sorted((root / "out" / "results").glob("*.json"))[0]
    data = json.loads(victim.read_text(encoding="utf-8"))
    data["proof_run"] = {"id": "prf-anotherrun", "source_fingerprint": agg,
                        "started_at": "2026-09-20T01:00:00+00:00"}
    victim.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SystemExit):
        pm.write()


def test_the_manifest_does_not_hash_itself(tmp_path, monkeypatch):
    _bound_proof_tree(tmp_path, monkeypatch)
    assert pm.write() == 0
    manifest = json.loads(pm.MANIFEST.read_text(encoding="utf-8"))
    recorded = {name for group in manifest.get("artefacts", {}).values() for name in group}
    assert pm.MANIFEST.name not in recorded
    assert pm.MANIFEST.name not in manifest.get("source_files", {})
