"""Reply DRAFT to an inbound mail. A draft is a file. Sending is a SEPARATE, stricter gate than the
rep notification: LEADSCOUT_SEND_EMAIL and LEADSCOUT_SEND_REPLIES, SMTP credentials, not proof mode,
the reply going to the From address itself, and a DMARC/DKIM pass for that domain (see send_decision)."""
from __future__ import annotations

import re
import smtplib
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr

_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def _line(text: str) -> str:
    """One header-safe line from untrusted text."""
    return " ".join(_CTRL.sub(" ", text or "").split())


class _Safe(dict):
    def __missing__(self, key):
        return ""


def first_name(name: str) -> str:
    parts = _line(name).split()
    return parts[0] if parts else "there"


def draft_decision(*, profile, outcome, flags: list[str], auto_generated: bool, to_addr: str) -> tuple[bool, str]:
    """(should_draft, reason). The reason is recorded either way."""
    if not profile.reply.enabled:
        return False, "profile reply disabled"
    if flags:
        return False, "suppressed: prompt-injection flags " + ",".join(flags)
    if outcome is None:
        return False, "no pipeline outcome"
    if auto_generated:
        return False, "suppressed: automatic/no-reply sender"
    if not to_addr:
        return False, "no reply address"
    status = outcome.compliance.status
    if status not in profile.reply.draft_on:
        return False, f"compliance status {status!r} not in draft_on {list(profile.reply.draft_on)}"
    return True, "ok"


def reply_address(parsed) -> str:
    """Who a reply goes to: Reply-To, else From. For a forwarded mail the lead is the
    ORIGINAL sender, never the forwarder, and Reply-To of the forwarding hop is ignored."""
    addr = parsed.from_addr if parsed.forwarded_by else (parsed.reply_to or parsed.from_addr)
    _, addr = parseaddr(addr or "")
    return addr if re.fullmatch(r"[^@\s<>,;]+@[^@\s<>,;]+\.[^@\s<>,;]+", addr or "") else ""


def build_draft(parsed, lead: dict, profile, to_addr: str) -> EmailMessage:
    ident = profile.identity
    values = _Safe(
        first_name=first_name(lead.get("name", "")), name=_line(lead.get("name", "")) or "there",
        company=_line(lead.get("company", "")), product_name=ident.product_name,
        company_name=ident.company_name, sender_name=ident.sender_name, signature=ident.signature)
    msg = EmailMessage()
    msg["From"] = _line(ident.sender_email)
    msg["To"] = to_addr
    subject = _line(parsed.subject)
    prefix = profile.reply.subject_prefix
    msg["Subject"] = subject if re.match(r"(?i)^\s*re\s*:", subject) else f"{prefix}{subject}".strip()
    msg["Date"] = formatdate(localtime=False)
    msg["Message-ID"] = make_msgid(domain="leadscout.local")
    if not parsed.message_id_synthesised:
        msg["In-Reply-To"] = parsed.message_id
        msg["References"] = " ".join([*parsed.references, parsed.message_id])
    msg["X-LeadScout-Draft"] = "1"
    msg.set_content(profile.reply.template.format_map(values).rstrip() + "\n")
    return msg


def _domain(addr: str) -> str:
    return addr.rsplit("@", 1)[-1].lower()


def _aligned(d: str, from_domain: str) -> bool:
    return d == from_domain or from_domain.endswith("." + d) or d.endswith("." + from_domain)


def send_decision(parsed, to_addr: str, cfg) -> tuple[bool, str]:
    """(may_send, reason). Every condition must hold; the reason is recorded when one does not.

    The authentication condition is only as good as the hop that wrote the topmost
    Authentication-Results header: a deployment whose MTA does not strip/overwrite that header
    from inbound mail, and any webhook provider that does not add one, must be treated as
    "unauthenticated" - which is what the absence of a header yields here."""
    if cfg.proof_mode:
        return False, "proof mode"
    if not cfg.send_email:
        return False, "LEADSCOUT_SEND_EMAIL is off"
    if not getattr(cfg, "send_replies", False):
        return False, "LEADSCOUT_SEND_REPLIES is off"
    if not (cfg.smtp_host and cfg.smtp_user and cfg.smtp_password):
        return False, "SMTP is not configured"
    if parsed.forwarded_by:
        return False, "forwarded mail: the sender's authentication is not evidenced"
    if not parsed.from_addr or to_addr.lower() != parsed.from_addr.lower():
        return False, "reply address differs from the From address"
    dom = _domain(parsed.from_addr)
    top = parsed.auth_top or {}
    if any(r == "pass" and _aligned(d, dom) for r, d in top.get("dmarc", [])):
        return True, "dmarc pass"
    if any(r == "pass" and _aligned(d, dom) for r, d in top.get("dkim", [])):
        return True, "aligned dkim pass"
    return False, "no dmarc=pass or aligned dkim=pass for the From domain in the topmost Authentication-Results"


def send_draft(msg: EmailMessage, cfg) -> bool:
    """Hand the draft to SMTP. The caller has already passed `send_decision`."""
    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30) as s:
        s.starttls()
        s.login(cfg.smtp_user, cfg.smtp_password)
        s.send_message(msg)
    return True
