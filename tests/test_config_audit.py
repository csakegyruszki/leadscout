"""Config audit: unknown LEADSCOUT_* keys warn (not fatal), typed known keys raise
a named ConfigError on a bad value, and .env.example stays reconcilable with what
the code actually reads.

Triggering case: a local .env had `LEADSCOUT_MODEL=x` (singular) while config.py
reads `LEADSCOUT_MODELS` (plural, see config.py:76-78) - the typo had NO effect and
NO warning, so the run silently fell through to the default profile chain instead
of using the intended model. These tests run config-import in a subprocess because
every field here is resolved at import time (config.py's dataclass defaults call
os.getenv() when the class body executes, and _warn_unknown_leadscout_keys() runs
at module load), not per-Settings()-call.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO / ".env.example"


def _run(probe: str, env_overrides: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-W", "always", "-c", probe],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )


# --- unknown key: warns, not fatal --------------------------------------------------

def test_leadscout_model_singular_typo_warns_by_name_and_does_not_activate():
    """The exact triggering case: LEADSCOUT_MODEL=x must not be silently ignored,
    and must not make the system believe model "x" is the active one."""
    proc = _run(
        "from leadscout.config import settings; print(repr(settings.models))",
        {"LEADSCOUT_MODEL": "x", "LEADSCOUT_MODELS": ""},
    )
    assert proc.returncode == 0, proc.stderr
    assert "LEADSCOUT_MODEL" in proc.stderr
    assert "RuntimeWarning" in proc.stderr
    # the process must still run (not fatal)
    models = proc.stdout.strip().splitlines()[-1]
    # "x" was never a recognised model id and must not appear as if it were active
    assert "'x'" not in models and '"x"' not in models
    # falls through to the documented default ("demo" profile) chain
    assert "openrouter:deepseek/deepseek-chat-v3.1" in models


def test_unknown_leadscout_key_warns_but_process_still_runs():
    proc = _run(
        "from leadscout.config import settings; print('ok')",
        {"LEADSCOUT_TOTALLY_MADE_UP": "1"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "ok"
    assert "LEADSCOUT_TOTALLY_MADE_UP" in proc.stderr
    assert "RuntimeWarning" in proc.stderr


def test_known_leadscout_key_never_warns():
    """Setting a KNOWN key must never itself be reported as unknown - checked by
    name, not by "no warnings at all", because this repo's own .env legitimately
    carries the LEADSCOUT_MODEL typo this feature exists to catch, and that
    warning firing here too is correct, not a regression."""
    proc = _run(
        "from leadscout.config import settings; print('ok')",
        {"LEADSCOUT_OUTPUT_DIR": "", "LEADSCOUT_PROFILE": "batch"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "'LEADSCOUT_OUTPUT_DIR'" not in proc.stderr
    assert "'LEADSCOUT_PROFILE'" not in proc.stderr


# --- known key, invalid typed value: configuration error, not silent default -------

@pytest.mark.parametrize("name,bad", [
    ("LEADSCOUT_LLM_HOP_TIMEOUT", "not-a-number"),
    ("LEADSCOUT_MAX_HTTP_CALLS", "not-a-number"),
    ("FIT_THRESHOLD", "not-a-number"),
])
def test_known_typed_key_with_invalid_value_raises_configuration_error(name, bad):
    proc = _run("from leadscout.config import settings", {name: bad})
    assert proc.returncode != 0, f"{name}={bad!r} should have failed import, but it did not"
    assert "ConfigError" in proc.stderr
    assert name in proc.stderr


@pytest.mark.parametrize("name,value,attr,expected", [
    ("LEADSCOUT_LLM_HOP_TIMEOUT", "45", "llm_hop_timeout", "45.0"),
    ("LEADSCOUT_MAX_HTTP_CALLS", "12", "max_http_calls", "12"),
    ("FIT_THRESHOLD", "80", "fit_threshold", "80"),
])
def test_known_typed_key_with_valid_value_is_honoured(name, value, attr, expected):
    proc = _run(
        f"from leadscout.config import settings; print(settings.{attr})", {name: value},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == expected


# --- .env.example reconcilability ---------------------------------------------------

def _env_example_keys() -> list[str]:
    keys = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "=" in line, f"non-comment .env.example line without '=': {line!r}"
        key, _, value = line.partition("=")
        keys.append((key.strip(), value.strip()))
    return keys


def test_env_example_has_no_non_placeholder_values():
    for key, value in _env_example_keys():
        allowed_defaults = {
            "LEADSCOUT_PROFILE": "demo",
            "SALES_REP_EMAIL": "sales-rep@example.test",
            "SMTP_PORT": "587",
            "LEADSCOUT_LLM_HOP_TIMEOUT": "30",
            "LEADSCOUT_MAX_HTTP_CALLS": "60",
            "FIT_THRESHOLD": "60",
        }
        if key in allowed_defaults:
            assert value == allowed_defaults[key], f"{key} default drifted from the documented one"
        else:
            assert value == "", f"{key}={value!r} in .env.example is not an empty placeholder"


def test_every_leadscout_prefixed_env_example_key_is_a_known_key():
    from leadscout.config import KNOWN_LEADSCOUT_KEYS

    for key, _ in _env_example_keys():
        if key.startswith("LEADSCOUT_"):
            assert key in KNOWN_LEADSCOUT_KEYS, (
                f".env.example documents {key!r}, which config.KNOWN_LEADSCOUT_KEYS does not know about"
            )


# --- the known-keys set itself must be complete, verified by grepping the source ---

def test_known_leadscout_keys_matches_every_os_getenv_in_the_source_tree():
    """The set in config.py must not be able to silently drift from what the code
    actually reads - grep leadscout/ and scripts/ for every `os.getenv("LEADSCOUT_...")`
    / `os.environ["LEADSCOUT_..."]` call and diff against KNOWN_LEADSCOUT_KEYS."""
    from leadscout.config import KNOWN_LEADSCOUT_KEYS

    # Matches both a direct os.getenv/os.environ[...] read AND config.py's own
    # _typed_env("NAME", caster, default) helper, which wraps os.getenv for the
    # keys that need type validation - either counts as "the code reads this key".
    pattern = re.compile(
        r"""(?:os\.(?:getenv|environ\.get|environ)\s*(?:\[|\()|_typed_env\()\s*["']([A-Za-z_][A-Za-z0-9_]*)["']"""
    )
    found: set[str] = set()
    for base in (REPO / "leadscout", REPO / "scripts"):
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for name in pattern.findall(text):
                if name.startswith("LEADSCOUT_"):
                    found.add(name)

    assert found, "the grep found nothing - the pattern itself is broken, not the code"
    missing_from_known = found - KNOWN_LEADSCOUT_KEYS
    stale_in_known = KNOWN_LEADSCOUT_KEYS - found
    assert not missing_from_known, f"read in code but not in KNOWN_LEADSCOUT_KEYS: {missing_from_known}"
    assert not stale_in_known, f"in KNOWN_LEADSCOUT_KEYS but never read in code: {stale_in_known}"


# --- LEADSCOUT_PROOF_MODE: a safety flag is on, off, or an error -------------------
# It used to be `os.getenv(...) == "1"`, so `LEADSCOUT_PROOF_MODE=true` - an ordinary
# spelling - left the send-refusal DISABLED while the operator believed it was on. With
# SMTP credentials present that is an unintended outbound notification during the very
# build the flag exists to protect. The delivery half of this is pinned in
# tests/test_proof_mode_no_delivery.py; here it is the parse.

@pytest.mark.parametrize("value,expected", [
    ("1", "True"), ("true", "True"), ("TRUE", "True"), ("True", "True"),
    ("yes", "True"), ("Yes", "True"), ("on", "True"), (" true ", "True"),
    ("0", "False"), ("false", "False"), ("FALSE", "False"), ("no", "False"),
    ("off", "False"), ("", "False"),
])
def test_proof_mode_accepts_the_ordinary_boolean_spellings(value, expected):
    """Empty is absence, not a wrong spelling: .env.example ships every optional key
    empty, and an empty flag has always meant "not set"."""
    proc = _run("from leadscout.config import settings; print(settings.proof_mode)",
                {"LEADSCOUT_PROOF_MODE": value})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == expected


@pytest.mark.parametrize("bad", ["treu", "tru", "maybe", "2", "enabled", "-1"])
def test_proof_mode_with_an_unrecognised_value_fails_loudly(bad):
    """Not "not the magic word, so off" - the same ConfigError every other known typed
    key raises, because a safety flag that cannot be honoured must not default to the
    unsafe state."""
    proc = _run("from leadscout.config import settings", {"LEADSCOUT_PROOF_MODE": bad})
    assert proc.returncode != 0, f"LEADSCOUT_PROOF_MODE={bad!r} should have failed import"
    assert "ConfigError" in proc.stderr
    assert "LEADSCOUT_PROOF_MODE" in proc.stderr
