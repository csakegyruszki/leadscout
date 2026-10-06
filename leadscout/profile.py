"""Organisation profile: who is selling what to whom, as data instead of code.

LeadScout started with one vendor's setup hardcoded, so its ICP, compliance lists and
mail wording were constants in the modules that used them. A profile is one YAML file
under `config/profiles/` that carries all of that for one deployment:

    identity    who the pipeline speaks for (product, sender, the rep who is notified)
    reply       whether and how a reply draft to an inbound mail is written
    ...         further sections (fit, compliance, notification) are read by the
                modules that own them, through `OrgProfile.section(name)`

Selected with `LEADSCOUT_ORG_PROFILE` - a profile name (`default`, `infra-vendor`, ...) or
a path to a YAML file. Unset means `default`, the neutral profile. A profile that cannot
be found or parsed is a hard error: silently falling back to another profile would score
leads against an ICP nobody chose.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROFILES_DIR = Path(__file__).resolve().parent.parent / "config" / "profiles"
DEFAULT_PROFILE = "default"


class ProfileError(ValueError):
    """The selected profile is missing, unreadable, or has a field of the wrong shape."""


@dataclass(frozen=True)
class Identity:
    product_name: str = "LeadScout"
    company_name: str = ""
    sender_name: str = "Sales team"
    sender_email: str = "leadscout@example.test"
    # Recipient of the internal lead notification; SALES_REP_EMAIL overrides it.
    rep_email: str = "sales-rep@example.test"
    signature: str = ""


@dataclass(frozen=True)
class ReplyConfig:
    """The reply DRAFT to an inbound mail. A draft is a file; sending is a separate,
    opt-in switch (LEADSCOUT_SEND_EMAIL) that this section cannot turn on."""
    enabled: bool = True
    # Compliance statuses that get a draft. A held or blocked lead gets none: the
    # first thing a prospect hears must not pre-empt the compliance review.
    draft_on: tuple[str, ...] = ("clear",)
    subject_prefix: str = "Re: "
    # str.format placeholders: {first_name} {name} {company} {product_name}
    # {company_name} {sender_name} {signature}
    template: str = (
        "Hi {first_name},\n\n"
        "thank you for reaching out. I have read your message and will come back to you "
        "shortly with next steps.\n\n"
        "Best regards,\n{sender_name}\n{signature}"
    )


@dataclass(frozen=True)
class OrgProfile:
    name: str
    path: Path
    identity: Identity = field(default_factory=Identity)
    reply: ReplyConfig = field(default_factory=ReplyConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    def section(self, name: str) -> dict[str, Any]:
        """A raw section for the module that owns it; {} when the profile omits it."""
        value = self.raw.get(name) or {}
        if not isinstance(value, dict):
            raise ProfileError(f"profile {self.name!r}: section {name!r} must be a mapping")
        return value


def _resolve(selector: str) -> Path:
    candidate = Path(selector).expanduser()
    if candidate.suffix in (".yaml", ".yml") or candidate.parent != Path("."):
        return candidate.resolve()
    return PROFILES_DIR / f"{selector}.yaml"


def _build(cls, data: Any, where: str):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ProfileError(f"{where} must be a mapping")
    known = set(cls.__dataclass_fields__)
    unknown = sorted(set(data) - known)
    if unknown:
        raise ProfileError(f"{where}: unknown keys {unknown}; expected a subset of {sorted(known)}")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
    return cls(**values)


def load_profile(selector: str | None = None) -> OrgProfile:
    """Load a profile by name or path. Not cached - see `active_profile()`."""
    selector = (selector or os.getenv("LEADSCOUT_ORG_PROFILE", "") or DEFAULT_PROFILE).strip()
    path = _resolve(selector)
    if not path.is_file():
        raise ProfileError(f"LEADSCOUT_ORG_PROFILE={selector!r}: no profile at {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ProfileError(f"profile {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileError(f"profile {path} must be a mapping at the top level")
    name = str(raw.get("name") or path.stem)
    return OrgProfile(
        name=name,
        path=path,
        identity=_build(Identity, raw.get("identity"), f"profile {name!r}: identity"),
        reply=_build(ReplyConfig, raw.get("reply"), f"profile {name!r}: reply"),
        raw=raw,
    )


@lru_cache(maxsize=1)
def active_profile() -> OrgProfile:
    """The profile this process runs with, read once. Tests that switch profiles call
    `active_profile.cache_clear()` after changing LEADSCOUT_ORG_PROFILE."""
    return load_profile()
