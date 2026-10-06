"""Runtime configuration. Everything secret comes from the environment / .env."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CONFIG_DIR = ROOT / "config"

# Every LEADSCOUT_* variable actually read anywhere in the codebase (leadscout/ +
# scripts/), named explicitly so the unknown-key audit below can be derived from
# it instead of a second, hand-maintained list. tests/test_config_audit.py greps
# the source tree for `os.getenv("LEADSCOUT_...")` and asserts this set matches,
# so it cannot silently drift the way LEADSCOUT_MODEL (singular - never read;
# config.py reads LEADSCOUT_MODELS, plural) drifted in a local .env with no
# warning and no error, just a silent fall-through to the default chain.
KNOWN_LEADSCOUT_KEYS = frozenset({
    "LEADSCOUT_OUTPUT_DIR",       # config.py - out_dir override
    "LEADSCOUT_PROFILE",          # config.py - "demo" | "batch" model-chain profile
    "LEADSCOUT_MODELS",           # config.py - explicit model chain, overrides the profile
    "LEADSCOUT_LLM_HOP_TIMEOUT",  # config.py - per-hop timeout (seconds)
    "LEADSCOUT_MAX_HTTP_CALLS",   # config.py - footprint.py's HTTP fan-out cap
    "LEADSCOUT_BUILD",            # api.py - /health build identifier, cosmetic only
    "LEADSCOUT_LLM_BASE_URL",     # llm.py - openai:-compatible endpoint base URL
    "LEADSCOUT_LLM_API_KEY",      # llm.py - openai:-compatible endpoint key
    "LEADSCOUT_JS_RENDER",        # providers/website.py - "1" enables Crawl4AI escalation
    "LEADSCOUT_USER_AGENT",       # util.py - outbound User-Agent override (contact URL)
    "LEADSCOUT_PROOF_MODE",       # config.py - "1" forbids network delivery in code
    "LEADSCOUT_ORG_PROFILE",      # profile.py - organisation profile name or YAML path
    "LEADSCOUT_SEND_EMAIL",       # config.py - "1" allows network e-mail delivery (default off)
    "LEADSCOUT_IMAP_HOST",        # inbound/imap.py - mailbox to poll (unset = poller disabled)
    "LEADSCOUT_IMAP_PORT",        # inbound/imap.py - default 993 (IMAP over TLS)
    "LEADSCOUT_IMAP_USER",        # inbound/imap.py - mailbox login
    "LEADSCOUT_IMAP_PASSWORD",    # inbound/imap.py - app password; never logged
    "LEADSCOUT_IMAP_FOLDER",      # inbound/imap.py - default INBOX
    "LEADSCOUT_IMAP_PROCESSED_FOLDER",  # inbound/imap.py - optional folder a processed mail is moved to
    "LEADSCOUT_IMAP_MAX_PER_POLL",      # inbound/imap.py - messages handled per poll (default 25)
    "LEADSCOUT_INBOUND_TOKEN",    # inbound/webhook.py - shared secret for POST /inbound/mail (unset = 404)
    "LEADSCOUT_SEND_REPLIES",     # config.py - "1" (with SEND_EMAIL) lets inbound reply drafts be sent
    "LEADSCOUT_INBOUND_CONCURRENCY",    # inbound/service.py - concurrent pipeline runs (default 1)
    "LEADSCOUT_INBOUND_MAX_ATTEMPTS",   # inbound/service.py - failed runs before dead_letter (default 3)
    "LEADSCOUT_INBOUND_MAX_BYTES",      # inbound/service.py - raw message size cap (default 10 MiB)
})


class ConfigError(ValueError):
    """A known LEADSCOUT_*/FIT_THRESHOLD env var is set to a value of the wrong
    shape (not parseable as the type the code actually casts it to). Distinct
    from an unrecognised key, which only warns - this is a hard error because
    the setting cannot be honoured at all, so a silent default would be a lie."""


def _warn_unknown_leadscout_keys(environ: dict = os.environ) -> list[str]:
    """An unknown LEADSCOUT_* key is NOT fatal - a hosting platform may inject
    one nothing here reads - but it must never fail silently, because an unread
    key with no effect looks identical to a working one otherwise. Returns the
    names warned about, for tests."""
    unknown = sorted(
        name for name in environ
        if name.startswith("LEADSCOUT_") and name not in KNOWN_LEADSCOUT_KEYS
    )
    for name in unknown:
        warnings.warn(
            f"Unknown environment variable {name!r} is set but leadscout does not "
            "read it anywhere - it has NO EFFECT on this run. Did you mean one of: "
            f"{', '.join(sorted(KNOWN_LEADSCOUT_KEYS))}?",
            RuntimeWarning,
            stacklevel=2,
        )
    return unknown


def _typed_env(name: str, caster, default: str):
    """Read `name`, cast with `caster`, and turn a bad value into a named
    ConfigError instead of a bare ValueError/TypeError pointing at this line."""
    raw = os.getenv(name, default)
    try:
        return caster(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name}={raw!r} is not a valid value ({exc}).") from exc


_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def _env_bool(raw: str) -> bool:
    """A flag is on, off, or a configuration error - never "not the magic word, so
    off". `proof_mode` used to be `os.getenv(...) == "1"`, so `LEADSCOUT_PROOF_MODE=true`
    - an entirely ordinary spelling - silently left the send-refusal DISABLED while the
    operator believed it was on. With SMTP credentials in the environment that is an
    unintended outbound notification to the invented contact names in samples/leads.json,
    on real corporate domains, during the proof build the flag exists to protect.

    Cast, so `_typed_env` raises ConfigError for anything outside the two vocabularies,
    exactly as it does for a non-numeric timeout: a known key whose value cannot be
    honoured fails loudly rather than defaulting.
    """
    word = raw.strip().lower()
    if not word:
        # Absence, not a wrong spelling: `.env.example` ships every optional key with an
        # empty value, and an empty flag has always meant "not set". Off, silently.
        return False
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise ValueError(
        f"expected one of {sorted(_TRUE_WORDS)} or {sorted(_FALSE_WORDS)} (case-insensitive)")


_warn_unknown_leadscout_keys()

# Where every generated artefact goes: the tracker, the .eml files, the per-lead JSON,
# the provenance ledger and snapshots, and the provider caches.
#
# `ROOT` is derived from `__file__`, not from the working directory, so a run started
# from anywhere still wrote into the SOURCE TREE's out/. Measured while testing batch
# fault isolation from a temp directory: the run appended rows to the repository's
# committed tracker and ledger and left two result files and three snapshot directories
# behind. Committed proof artefacts must not be reachable by an unrelated run, so the
# location is injectable. `provenance.py`, `sanctions.py`, `ranges.py` and
# `providers/footprint.py` read `settings.out_dir` at import time, which is why this is
# resolved from the environment here rather than assigned after the fact.
_OUT_DIR_ENV = os.getenv("LEADSCOUT_OUTPUT_DIR", "").strip()
_OUT_DIR = Path(_OUT_DIR_ENV).expanduser().resolve() if _OUT_DIR_ENV else ROOT / "out"
DO_NOT_ENGAGE_PATH = CONFIG_DIR / "do_not_engage.yaml"
RESTRICTED_JURISDICTIONS_PATH = CONFIG_DIR / "restricted_jurisdictions.yaml"


def _load_yaml_list(path: Path, key: str) -> tuple[str, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return tuple(data.get(key, []))


# Model ids carry a provider prefix ("openrouter:<id>" / "cloudflare:<id>"); no
# prefix means "openrouter:" (see llm._parse_model). Chain order is a deployment
# profile, not a code change: latency matters for a live "demo" run and doesn't for
# a nightly "batch" job - both measured 2026-09-19 (see README "Model chain").
# nvidia/nemotron-3-nano-30b-a3b:free is 404 on OpenRouter and never included.
#
# `cloudflare:@cf/openai/gpt-oss-120b` is DROPPED from both profiles (Part 4,
# 2026-09-19): the per-model regression measured it at only 47.8%
# schema-valid (12 of 23 cases errored) - a model that fails to produce usable
# output on more than half its calls is not a reliable chain member regardless of
# its false-clear count on the smaller successful subset (see the design notes
# "Model chain" for the full table and reasoning).
_PROFILE_MODELS = {
    "demo": (
        "openrouter:deepseek/deepseek-chat-v3.1,"               # paid, fastest in a live demo run
        "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast,"  # free, false-clear 0, 3260ms
        "openrouter:deepseek/deepseek-v4-flash-0731:free,"      # free, false-clear 1, 9657ms
        "openrouter:nvidia/nemotron-3-super-120b-a12b:free"     # free, false-clear 2, 4144ms
    ),
    # Ordered by the measured per-model regression (evals/
    # model_comparison.md, 2026-09-19, 23 labelled compliance cases each):
    # false-clear count ascending (the PRIMARY safety metric - a false "clear" on
    # a case that should have been review/blocked), tie-break by latency.
    "batch": (
        "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast,"  # false-clear 0, 100% schema-valid, 3260ms
        "openrouter:deepseek/deepseek-v4-flash-0731:free,"      # false-clear 1, 100% schema-valid, 9657ms
        "openrouter:nvidia/nemotron-3-super-120b-a12b:free,"    # false-clear 2, 100% schema-valid, 4144ms
        "openrouter:deepseek/deepseek-chat-v3.1"                # paid, last resort
    ),
}
_DEFAULT_PROFILE = "demo"


def _load_profile() -> str:
    profile = os.getenv("LEADSCOUT_PROFILE", _DEFAULT_PROFILE)
    return profile if profile in _PROFILE_MODELS else _DEFAULT_PROFILE


def _load_models() -> tuple[str, ...]:
    raw = os.getenv("LEADSCOUT_MODELS") or _PROFILE_MODELS[_load_profile()]
    return tuple(m.strip() for m in raw.split(",") if m.strip())


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    models: tuple[str, ...] = field(default_factory=_load_models)
    profile: str = field(default_factory=_load_profile)
    cloudflare_account_id: str = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    cloudflare_api_token: str = os.getenv("CLOUDFLARE_API_TOKEN", "")
    # A hop exceeding this is abandoned and the next model in the chain is tried
    # (see llm._call_model); the existing fixed-backoff retry applies only to the
    # LAST model of the chain, independent of this per-hop timeout.
    llm_hop_timeout: float = _typed_env("LEADSCOUT_LLM_HOP_TIMEOUT", float, "30")
    opensanctions_api_key: str = os.getenv("OPENSANCTIONS_API_KEY", "")
    # Optional: raises the GitHub unauthenticated rate limit (60/hour); providers/
    # github.py works fine without it, just with a smaller MAX_CALLS budget margin.
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    # Optional: without it, providers/vendor.py's discovery queries are skipped
    # entirely (no key -> no Brave call -> status="skipped", never "degraded").
    brave_api_key: str = os.getenv("BRAVE_API_KEY", "")
    # Optional headcount source (Diffbot Knowledge Graph); see providers/headcount.py.
    diffbot_token: str = os.getenv("DIFFBOT_TOKEN", "")
    # Optional: without it, providers/hunter.py's contact-quality check is skipped
    # entirely (no key -> no Hunter call -> status="skipped", never "degraded").
    # Informational only - never affects fit score, cloud usage, or compliance.
    hunter_api_key: str = os.getenv("HUNTER_API_KEY", "")
    out_dir: Path = _OUT_DIR
    tracker_path: Path = _OUT_DIR / "leads_tracker.xlsx"
    outbox_dir: Path = _OUT_DIR / "outbox"
    # Empty = not overridden: notify.py then uses the profile's identity.rep_email.
    sales_rep_email: str = os.getenv("SALES_REP_EMAIL", "")
    # Optional real SMTP delivery; without these the mail is written as .eml only.
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    # A proof build must be incapable of sending, not merely unconfigured. The committed
    # demo batch carries invented contact names on real corporate domains, so "the SMTP
    # settings happened to be empty" is not a good enough reason for no message having
    # reached them - one credential left in the environment during a proof rebuild would
    # be. notify.py checks this ABOVE the credential check, so the only thing a proof
    # build can produce is the .eml file on disk.
    proof_mode: bool = _typed_env("LEADSCOUT_PROOF_MODE", _env_bool, "0")
    # Network delivery of ANY mail (the rep notification, an inbound-mail reply) is opt-in.
    # Credentials alone used to be the switch, so a mailbox password added for the IMAP
    # poller - or an SMTP block copied into a shared .env - would have started sending.
    # Off by default; proof_mode still wins over it.
    send_email: bool = _typed_env("LEADSCOUT_SEND_EMAIL", _env_bool, "0")
    http_timeout: float = 20.0
    # footprint.py's crt.sh/DoH/RIPEstat/range-file crawl over many subdomains is the
    # one provider that can fan out to dozens of HTTP calls for a single lead; this
    # caps it (REVIEW-6bB-verified.md #5). Configurable so a richer domain can be
    # given more budget without a code change. Raised 40 -> 60: a live run against a
    # domain with the full MAX_SUBDOMAINS=30 preferred-first list (each host costing
    # up to 2-3 DoH/RIPEstat/range calls) hit the old ceiling before covering even
    # half the preferred hosts (measured 2026-09-19, e.g. 40 calls for 18 hosts).
    max_http_calls: int = _typed_env("LEADSCOUT_MAX_HTTP_CALLS", int, "60")
    # Minimum fit_score (of 100) for sales_ready, alongside compliance=clear and
    # fit_confidence != LOW. Env-overridable so a sales team can tune the bar
    # without a code change.
    # When FIT_THRESHOLD is NOT set the organisation profile's `fit.threshold` applies
    # (fit.fit_threshold()); `fit_threshold` then only carries the historical default.
    fit_threshold: int = _typed_env("FIT_THRESHOLD", int, "60")
    fit_threshold_from_env: bool = bool(os.getenv("FIT_THRESHOLD", "").strip())

    # Do-not-engage list and sanctioned-jurisdiction list now live in config/*.yaml
    # (not hardcoded here) so the provenance ledger can record which policy version
    # a given decision used; same tuples exposed as before for backward compatibility.
    competitors: tuple[str, ...] = field(
        default_factory=lambda: _load_yaml_list(DO_NOT_ENGAGE_PATH, "competitors"))
    sanctioned_countries: tuple[str, ...] = field(
        default_factory=lambda: _load_yaml_list(RESTRICTED_JURISDICTIONS_PATH, "jurisdictions"))
    # Capital/major-city markers for the sanctioned jurisdictions above - used by
    # compliance.py's deterministic HQ-unknown safety net (Part 2b, 2026-09-19).
    restricted_jurisdiction_markers: tuple[str, ...] = field(
        default_factory=lambda: _load_yaml_list(RESTRICTED_JURISDICTIONS_PATH, "markers"))
    # Sending an inbound-mail reply draft needs THIS flag in addition to send_email (and SMTP
    # credentials, and proof_mode off): a reply goes to a stranger, a rep notification does not.
    send_replies: bool = _typed_env("LEADSCOUT_SEND_REPLIES", _env_bool, "0")


settings = Settings()
