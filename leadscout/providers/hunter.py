"""Hunter.io contact-quality check: an email-verifier + domain-search pair, used
strictly as sales-rep context - see design notes below for why it never touches
fit/cloud/compliance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlencode

import httpx

from ..config import settings
from ..models import Evidence, ProviderResult
from ._base import UA, fallback_evidence_id, snapshot_path

# --- Design notes -----------------------------------------------------------------
# INFORMATIONAL ONLY (author's explicit requirement): Hunter never changes fit
# score, cloud-usage state, compliance status or sales_ready. The one Evidence item
# it can produce is `family="corporate_identity"`, `strength="WEAK"`, `provider=None`
# - WEAK so cloud.py's family/strength thresholds (which drive the cloud-usage
# waterfall) and fit.py's HQ-based scoring are structurally unaffected regardless of
# what Hunter returns; see tests/test_hunter.py's byte-identical-outcomes test.
# `contact_quality`/`contact_quality_note` (models.Research) are the actual product
# of this provider - advisory text a sales rep reads, never a scored signal.
#
# No `HUNTER_API_KEY` -> skipped, zero calls (same convention as providers/vendor.py
# with BRAVE_API_KEY): an optional enrichment step must never turn into a hard
# dependency for a lead that would otherwise process fine.
#
# API key hygiene: the key travels as a query parameter on both Hunter endpoints,
# so every URL this module builds for an Evidence/snapshot/error path is assembled
# by hand from the non-secret params only (`_public_url`) - never taken from the
# request/response object, which would carry it straight through.
#
# PII: the verifier call's query IS the contact's email, so Hunter already sees it
# server-side - nothing new is leaked by calling it. But the raw verifier response
# also echoes that email back in its body, and this pipeline's snapshot ledger is
# meant to hold company-level facts, not the lead's personal contact string (see
# models.Evidence's docstring: "Never carries the lead's contact name or email").
# `_redact_email` replaces the exact email string in the raw JSON bytes with
# "<redacted>" before it is hashed/snapshotted/recorded - the parsed fields used for
# classification are read from the ORIGINAL response, only the stored copy differs.
# Documented again in the design notes' Hunter section.

VERIFIER_API = "https://api.hunter.io/v2/email-verifier"
DOMAIN_API = "https://api.hunter.io/v2/domain-search"

# Hunter's own verifier statuses that are ambiguous enough to need the score as a
# tie-breaker, rather than an outright accept/reject.
_AMBIGUOUS_STATUSES = {"accept_all", "webmail", "unknown"}
_INVALID_STATUSES = {"invalid", "disposable"}
_RISKY_SCORE_FLOOR = 70


def _public_url(base: str, params: dict) -> str:
    """The URL Hunter was actually asked, minus `api_key` - safe to put in an
    Evidence.url, a snapshot path label, or an error message."""
    clean = {k: v for k, v in params.items() if k != "api_key"}
    return f"{base}?{urlencode(clean)}" if clean else base


def _scrub(text: str) -> str:
    """Belt-and-braces: strip any literal api_key value that might have leaked
    into an exception message (e.g. httpx echoing the request URL back)."""
    key = settings.hunter_api_key
    return text.replace(key, "<redacted>") if key else text


def _redact_email(raw: bytes, email: str) -> bytes:
    """Replace the exact contact email in a raw JSON response with "<redacted>"
    before it is hashed/snapshotted - see design notes above."""
    if not email:
        return raw
    return raw.replace(email.encode("utf-8"), b"<redacted>").replace(
        email.lower().encode("utf-8"), b"<redacted>")


@dataclass
class HunterFacts:
    """What research.py needs beyond the (informational-only) Evidence: the
    classification consumed by Research.contact_quality/contact_quality_note."""
    contact_quality: str  # "valid"|"risky"|"invalid"|"unverified"
    contact_quality_note: str
    # The organisation this DOMAIN belongs to, as the provider names it. Measured
    # 2026-09-20 over five unseen domains: this is what the domain lookup is actually
    # good at - `lidl.hu` returns "Lidl Magyarorszag", the LOCAL operating entity,
    # where the company name alone resolves to the global group. Its headcount and
    # location were not reliable in the same measurement (Typeform placed in the US,
    # Docplanner in Amsterdam, Notion at 5K-10K), so only the name is carried, and
    # only as an identity anchor - never as a firmographic fact.
    domain_organisation: str = ""


def _unverified(note: str) -> HunterFacts:
    return HunterFacts(contact_quality="unverified", contact_quality_note=note)


def classify_contact_quality(status: str, score: int | None, mx_records: bool | None) -> tuple[str, str]:
    """(contact_quality, contact_quality_note) from the verifier's own fields.

    - "valid" - Hunter's own `status == "valid"`, or an ambiguous status
      (accept_all/webmail/unknown) backed by a score >= 70 (a high score on an
      ambiguous status is still a good deliverability signal, per Hunter's own
      scoring model).
    - "risky" - an ambiguous status with score < 70 (or no score at all - a missing
      score must not collapse into the better "valid" bucket).
    - "invalid" - `invalid` or `disposable`.
    Any other/unexpected status string is treated as "risky", not "valid" - an
    unrecognised Hunter status is a gap, not a green light.
    """
    score_txt = str(score) if score is not None else "n/a"
    mx_txt = "MX ok" if mx_records else ("no MX" if mx_records is False else "MX unknown")
    note = f"Hunter: {status}, score {score_txt}, {mx_txt}"
    if status == "valid":
        return "valid", note
    if status in _INVALID_STATUSES:
        return "invalid", note
    if status in _AMBIGUOUS_STATUSES:
        if score is not None and score >= _RISKY_SCORE_FLOOR:
            return "valid", note
        return "risky", note
    return "risky", note


def run(email: str, domain: str, run=None) -> tuple[ProviderResult, HunterFacts]:
    if not settings.hunter_api_key:
        return (ProviderResult(provider_name="hunter", status="skipped", note="no HUNTER_API_KEY"),
                _unverified("Hunter: skipped, no HUNTER_API_KEY"))

    calls = 0
    try:
        with httpx.Client(timeout=settings.http_timeout, headers={"User-Agent": UA}) as client:
            verify_params = {"email": email, "api_key": settings.hunter_api_key}
            vr = client.get(VERIFIER_API, params=verify_params)
            calls += 1
            if vr.status_code != 200:
                note = f"HTTP {vr.status_code} from email-verifier ({_public_url(VERIFIER_API, verify_params)})"
                return (ProviderResult(provider_name="hunter", status="degraded", note=_scrub(note), calls=calls),
                        _unverified(_scrub(note)))

            verify_raw = vr.content
            verify_json = vr.json()
            vdata = verify_json.get("data", {}) or {}
            status = vdata.get("status", "unknown")
            score = vdata.get("score")
            mx_records = vdata.get("mx_records")

            domain_params = {"domain": domain, "api_key": settings.hunter_api_key, "limit": 1}
            dr = client.get(DOMAIN_API, params=domain_params)
            calls += 1
            if dr.status_code != 200:
                note = f"HTTP {dr.status_code} from domain-search ({_public_url(DOMAIN_API, domain_params)})"
                return (ProviderResult(provider_name="hunter", status="degraded", note=_scrub(note), calls=calls),
                        _unverified(_scrub(note)))
            domain_raw = dr.content
            ddata = dr.json().get("data", {}) or {}
    except Exception as e:  # noqa: BLE001 - a malformed/unreachable response must degrade, never crash
        note = _scrub(f"{type(e).__name__}: {e}")
        return (ProviderResult(provider_name="hunter", status="degraded", note=note, calls=calls),
                _unverified(f"Hunter: {note}"))

    # Both raw payloads go through the ledger (provenance requires every fetched
    # byte to be chain-of-custody-tracked, see provenance.ProvenanceRun.record) -
    # but only ONE Evidence is ever surfaced downstream (`ev`, below, built from
    # the domain-search response). The verifier response is recorded for audit
    # purposes only, with the contact's own email redacted from the stored bytes
    # (see `_redact_email`'s design note); it is never appended to a provider's
    # `evidence` list and therefore never reaches cloud.py/fit.py/compliance.py.
    if run is not None:
        v_eid = fallback_evidence_id("hunter", run)
        run.record(Evidence(
            id=v_eid, source_type="hunter", url=_public_url(VERIFIER_API, verify_params),
            observed_at=datetime.now(UTC).isoformat(),
            content_sha256=hashlib.sha256(_redact_email(verify_raw, email)).hexdigest(),
            strength="WEAK", snippet="hunter email-verifier response (email redacted)",
            snapshot_path=snapshot_path(run, v_eid), family="corporate_identity",
        ), _redact_email(verify_raw, email), stage="hunter")

    quality, note = classify_contact_quality(status, score, mx_records)

    org = ddata.get("organization") or "unknown"
    country = ddata.get("country") or "unknown"
    headcount = ddata.get("headcount") or "unknown"
    company_type = ddata.get("company_type") or "unknown"
    industry = ddata.get("industry") or "unknown"
    snippet = (f"org={org}, country={country}, headcount={headcount}, "
               f"type={company_type}, industry={industry}")[:200]

    eid = fallback_evidence_id("hunter", run)
    ev = Evidence(
        id=eid, source_type="hunter", url=_public_url(DOMAIN_API, domain_params),
        observed_at=datetime.now(UTC).isoformat(), content_sha256=hashlib.sha256(domain_raw).hexdigest(),
        strength="WEAK", snippet=snippet, snapshot_path=snapshot_path(run, eid),
        family="corporate_identity", provider=None, origin="provider:hunter",
    )
    if run is not None:
        run.record(ev, domain_raw, stage="hunter")

    provider_note = f"{calls} call(s), contact_quality={quality} ({status}, score={score})"
    result = ProviderResult(provider_name="hunter", status="ok", evidence=[ev], note=provider_note, calls=calls)
    return result, HunterFacts(contact_quality=quality, contact_quality_note=note,
                               domain_organisation="" if org == "unknown" else str(org))
