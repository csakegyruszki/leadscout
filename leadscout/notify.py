"""Email notification to the sales rep.

Always writes the message as an .eml file to out/outbox/ (demoable without any
mail server). It is also sent over SMTP only when LEADSCOUT_SEND_EMAIL is on AND
SMTP_* is configured; LEADSCOUT_PROOF_MODE forbids sending whatever else is set.
"""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from pathlib import Path

from .cloud import counts_toward_cloud_state
from .config import settings
from .fit import fit_threshold
from .models import Evidence, LeadOutcome
from .profile import active_profile
from .profile_config import fit_config, notification_config

logger = logging.getLogger("leadscout")


def _sanctions_line(c) -> str:
    if not c.sanctions_hits:
        return f"Sanctions API: {c.sanctions_status}"
    top = max(c.sanctions_hits, key=lambda h: h["score"])
    topics = ", ".join(top.get("topics", [])) or "none"
    return (f"Sanctions API: {c.sanctions_status} - {top.get('caption', '')} "
            f"(score {top.get('score', 0):.2f}, topics: {topics}) {top.get('url', '')}")


_STRENGTH_ORDER = {"WEAK": 0, "MEDIUM": 1, "STRONG": 2, "DIRECT": 3}


def _cloud_evidence_lines(r) -> list[str]:
    """One line per family - the strongest representative Evidence item for that
    family - `- <STRENGTH> · <family> · <one-line observation> (ev-id)`
    (Part 4 addendum, 2026-09-19)."""
    by_family: dict[str, Evidence] = {}
    for ev in r.evidence:
        # Only evidence the classifier actually counts belongs under "Cloud usage";
        # identity/encyclopedic items would suggest more support than there is.
        if not ev.family or not counts_toward_cloud_state(ev):
            continue
        # WEAK items never move the state; listing them only adds noise for a sales rep.
        if ev.strength == "WEAK" and ev.family != "edge_delivery":
            continue
        cur = by_family.get(ev.family)
        if cur is None or _STRENGTH_ORDER.get(ev.strength, -1) > _STRENGTH_ORDER.get(cur.strength, -1):
            by_family[ev.family] = ev
    lines = []
    for family in sorted(by_family):
        ev = by_family[family]
        observation = " ".join((ev.snippet or "").split())[:140] or "(no snippet)"
        lines.append(f"  - {ev.strength} · {family} · {observation} ({ev.id})")
    return lines or ["  (no cloud-bearing evidence found)"]


def _cloud_block(r, f) -> str:
    """Cloud usage: <STATE> / Confidence: <fit_confidence> / Observed provider(s):
    ... / Evidence: one line per family / Boundary: <sentence> (Part 4 addendum).
    Fix #8: also shows `unknown_reason` (when set) and any degraded/missing
    provider channel - a sales rep should see WHY the state is UNKNOWN, not just
    that it is."""
    cu = r.cloud_usage
    providers_str = ", ".join(
        f"{p['provider']} ({p['state']}, {p.get('independent_family_count', 0)} independent famil"
        f"{'y' if p.get('independent_family_count', 0) == 1 else 'ies'})" for p in cu.providers
    ) or "none identified"
    lines = [
        f"Cloud usage: {cu.state}",
        f"Confidence: {f.confidence}",
        f"Observed provider(s): {providers_str}",
        "Evidence:",
    ]
    lines.extend(_cloud_evidence_lines(r))
    lines.append(f"Boundary: {cu.boundary}")
    if cu.unknown_reason:
        lines.append(f"Unknown reason: {cu.unknown_reason}")
    if cu.missing_channels:
        lines.append(f"Missing channels: {', '.join(cu.missing_channels)}")
    return "\n".join(lines)


def _infrastructure_block(r) -> str:
    """Infrastructure: one line per `InfrastructureFootprint` - scope · category ·
    provider (or "network holder X (operator not established)") · technologies ·
    confidence · ev-ids (Fix #8)."""
    lines = ["Infrastructure:"]
    if not r.infrastructure:
        lines.append("  (no infrastructure footprint observed)")
        return "\n".join(lines)
    for fp in r.infrastructure:
        if fp.provider:
            who = fp.provider
        elif fp.network_holder:
            who = f"network holder {fp.network_holder} (operator not established)"
        else:
            who = "unknown"
        techs = ", ".join(fp.technologies) or "none"
        ev_ids = ", ".join(fp.evidence_ids) or "no evidence"
        lines.append(f"  - {fp.scope} · {fp.category} · {who} · {techs} · {fp.confidence} · {ev_ids}")
    return "\n".join(lines)


def next_action(outcome: LeadOutcome) -> str:
    """The one thing the sales rep should do with this lead. Deterministic, compliance first -
    a high fit never outranks a compliance hold, and a qualified account with an undeliverable
    contact is still worth working (find another contact), not dropped."""
    c, f, r = outcome.compliance, outcome.fit, outcome.research
    if c.status == "blocked":
        return "DO NOT ENGAGE - compliance block (see Compliance below); no outreach"
    if c.status == "review":
        return "HOLD - compliance review needed before any outreach (see Compliance below)"
    if outcome.sales_ready and r.contact_quality == "valid":
        return "CONTACT NOW - qualified account, verified contact"
    if outcome.sales_ready:
        return "WORK THE ACCOUNT - qualified, but the submitted email failed verification: find a valid contact"
    size_known = bool(r.estimated_employees) or outcome.lead.company_size_band in fit_config().band_employees()
    cloud_seen = r.cloud_usage.state in ("CONFIRMED", "LIKELY", "POSSIBLE")
    if cloud_seen and not size_known:
        # The score is low only because size - the first fit criterion - is unknown: that is a
        # question to ask, not a reason to drop the lead.
        hint = f" (unverified hint: {r.employees_hint.split(' - ')[0]})" if r.employees_hint else ""
        return f"CONFIRM SIZE - public-cloud use is {r.cloud_usage.state} but company size is unknown{hint}"
    if f.confidence == "LOW":
        # Not "revisit if the lead supplies company size": a large company usually
        # publishes its headcount, so an unknown size is a gap in THIS run's evidence,
        # not something to wait for the applicant to volunteer (Lidl, 2026-09-20 - the
        # figure is on the company's own corporate site).
        missing = "company size" if not size_known else "public-cloud use"
        return (f"LOW PRIORITY - not qualified: {missing} could not be established from accepted "
                f"evidence in this run; recheck before discarding")
    if not cloud_seen:
        return "LOW PRIORITY - below the fit threshold; no public-cloud spend evidenced"
    return f"LOW PRIORITY - fit {f.score} is below the threshold ({fit_threshold()})"


def render(outcome: LeadOutcome) -> tuple[str, str]:
    lead, r, c, f = outcome.lead, outcome.research, outcome.compliance, outcome.fit
    flag = {"clear": "CLEAR", "review": "REVIEW NEEDED", "blocked": "DO NOT ENGAGE"}[c.status]
    subject = notification_config().subject_template.format(
        company=lead.company, score=f.score, confidence=f.confidence, flag=flag)
    matches = "\n".join(
        f"  - {m.get('kind')}: {m.get('term')} ({m.get('score')}) - {m.get('reason')}" for m in c.matches
    ) or "  none"
    # Same employee fallback as the fit score, so the header never contradicts it.
    if r.estimated_employees:
        employees = str(r.estimated_employees) + (f" - source: {r.employees_source}" if r.employees_source else "")
    elif lead.company_size_band in (bands := fit_config().band_employees()):
        employees = f"~{bands[lead.company_size_band]} (self-reported band {lead.company_size_band})"
    else:
        employees = "unknown"
    hint_line = f"\nHeadcount hint: {r.employees_hint}" if r.employees_hint else ""
    # Fix #8: ACCOUNT (fit+compliance) and CONTACT (contact quality) are two
    # different questions - a great account with a bad email, or a validated
    # contact at an account that isn't ready, both used to collapse into one
    # merged "SALES-READY" flag. `ACCOUNT` is exactly `outcome.sales_ready`
    # (pipeline.py already computes it from fit+compliance alone, never contact
    # quality); `CONTACT: ACTIONABLE` is conservative on purpose - "unverified"
    # (no Hunter key, or a degraded lookup) must never read as actionable, only
    # an explicit "valid" does (CLAUDE.md's "unknown must not collapse into valid").
    contact_actionable = r.contact_quality == "valid"
    account_line = (f"ACCOUNT: SALES-READY {'yes' if outcome.sales_ready else 'no'} "
                    f"(fit {f.score}/100 {f.confidence}, compliance {flag})")
    contact_line = (f"CONTACT: ACTIONABLE {'yes' if contact_actionable else 'no'} "
                    f"(quality: {r.contact_quality} — {r.contact_quality_note or 'n/a'})")
    body = f"""New inbound lead: {lead.company}
Contact: {lead.name} <{lead.email}>{(' - ' + lead.job_title) if lead.job_title else ''}
Website: {lead.website}

NEXT ACTION: {next_action(outcome)}
{account_line}
{contact_line}

Summary
{r.summary}
Industry: {r.industry} | HQ: {r.headquarters_country} | Employees (est.): {employees}{hint_line}
Research confidence: {r.confidence}

Compliance: {flag} - {c.reasoning}

--- Detail for audit (not needed to act) ---

Fit score: {f.score}/100 ({f.confidence})
{f.reasoning}

""" + _cloud_block(r, f) + "\n\n" + _infrastructure_block(r) + f"""

Spend prior (relative, illustrative): {f.cloud_spend_band} - {f.cloud_spend_reasoning}

Compliance: {flag}
{c.reasoning}
Matches:
{matches}

{_sanctions_line(c)}

Sources:
""" + "\n".join(f"  {s}" for s in r.sources) + "\n\n"
    if r.uncertainties:
        body += "Uncertainties:\n" + "\n".join(f"  - {u}" for u in r.uncertainties) + "\n\n"
    body += f"Pipeline: {outcome.llm_calls} LLM calls ({outcome.llm_models}, profile={outcome.llm_profile}), "
    body += f"${outcome.llm_cost_usd:.6f}, {outcome.llm_latency_ms} ms, {outcome.llm_hops} hops\n"
    prov_flag = "ok" if outcome.provenance_status == "ok" else "DEGRADED"
    body += (f"Provenance: {outcome.evidence_count} sources, ledger {prov_flag}, "
             f"head {outcome.provenance_head}\n")
    return subject, body


def notify(outcome: LeadOutcome, *, artifact_stem: str | None = None) -> Path:
    subject, body = render(outcome)
    msg = EmailMessage()
    identity = active_profile().identity
    msg["From"] = settings.smtp_user or identity.sender_email
    msg["To"] = settings.sales_rep_email or identity.rep_email
    msg["Subject"] = subject
    msg.set_content(body)

    settings.outbox_dir.mkdir(parents=True, exist_ok=True)
    # Default: derived from the company name (form/CLI/proof behaviour). A caller whose company
    # name is attacker-supplied (the inbound mail path) passes its own unique stem instead, so one
    # lead cannot overwrite another's outbox file by naming the same company.
    safe = artifact_stem or "".join(ch if ch.isalnum() else "_" for ch in outcome.lead.company)[:40]
    path = settings.outbox_dir / f"{safe}.eml"
    path.write_bytes(bytes(msg))

    # Proof mode cannot send, and not because the credentials happen to be blank. The
    # committed demo batch carries invented contact names on real corporate domains
    # (samples/leads.json), so "the SMTP settings were empty at the time" is not a good
    # enough reason for no message having reached them - one credential left in the
    # environment during a proof rebuild would be. The refusal sits above the credential
    # check, so the only thing a proof build can produce is the .eml file on disk.
    if settings.proof_mode:
        logger.info("proof mode: wrote %s; network delivery is disabled in code", path.name)
        return path
    if not settings.send_email:
        return path
    if settings.smtp_host and settings.smtp_user and settings.smtp_password:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as s:
            s.starttls()
            s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)
    return path
