"""Webhook payload -> raw RFC 822 bytes, for the three shapes providers actually send.

  message/rfc822            Cloudflare Email Worker (and anything that can POST raw MIME)
  multipart/form-data or    Mailgun route forward() -> field `body-mime`;
  x-www-form-urlencoded     SendGrid Inbound Parse with "POST the raw, full MIME message" -> field `email`
  application/json          Postmark inbound webhook (FromFull / TextBody / HtmlBody / Headers ...)

Multipart is parsed with the stdlib `email` package (python-multipart is not a dependency).
"""
from __future__ import annotations

import base64
import hmac
import json
import os
from email import message_from_bytes, policy
from email.message import EmailMessage
from email.utils import formataddr
from urllib.parse import parse_qs

from .parse import DEFAULT_MAX_BYTES

RAW_FIELDS = ("body-mime", "email")

_POSTMARK_COPY_HEADERS = ("In-Reply-To", "References", "Authentication-Results", "Auto-Submitted", "Precedence",
                          "Reply-To", "Message-ID", "Date")


class BadPayload(ValueError):
    pass


def configured_token() -> str:
    return os.getenv("LEADSCOUT_INBOUND_TOKEN", "")


def token_matches(supplied: str | None) -> bool:
    expected = configured_token()
    if not expected or not supplied:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def request_body_cap(message_cap: int | None = None) -> int:
    """Largest request body read at all: a base64/form-encoded message is ~1.4x its raw size."""
    return int((message_cap or DEFAULT_MAX_BYTES) * 1.5) + 64 * 1024


def _from_form(content_type: str, body: bytes) -> bytes:
    if content_type.startswith("application/x-www-form-urlencoded"):
        fields = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
        for name in RAW_FIELDS:
            if fields.get(name):
                return fields[name][0].encode("utf-8")
        raise BadPayload("no raw-MIME field (body-mime / email) in the form")
    # multipart/form-data: wrap in a header block so the stdlib parser can read the boundary
    envelope = b"Content-Type: " + content_type.encode("latin-1", "replace") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    form = message_from_bytes(envelope, policy=policy.compat32)
    if not form.is_multipart():
        raise BadPayload("multipart body could not be parsed")
    for part in form.get_payload():
        name = part.get_param("name", header="content-disposition")
        if name in RAW_FIELDS and not part.get_filename():
            payload = part.get_payload(decode=True)
            if payload:
                return payload
        if name in RAW_FIELDS and part.get_filename():       # some setups send the MIME as a file part
            payload = part.get_payload(decode=True)
            if payload:
                return payload
    raise BadPayload("no raw-MIME field (body-mime / email) in the form")


def _from_postmark(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise BadPayload("invalid JSON") from exc
    if not isinstance(data, dict) or not ("FromFull" in data or "From" in data):
        raise BadPayload("not a Postmark inbound payload")
    msg = EmailMessage()
    full = data.get("FromFull") or {}
    msg["From"] = formataddr((str(full.get("Name") or ""), str(full.get("Email") or data.get("From") or "")))
    msg["To"] = str(data.get("To") or "")
    msg["Subject"] = str(data.get("Subject") or "")
    headers = {str(h.get("Name", "")).lower(): str(h.get("Value", ""))
               for h in data.get("Headers") or [] if isinstance(h, dict)}
    if data.get("Date"):
        msg["Date"] = str(data["Date"])
    if data.get("ReplyTo"):
        msg["Reply-To"] = str(data["ReplyTo"])
    for name in _POSTMARK_COPY_HEADERS:
        value = headers.get(name.lower())
        if value and name not in msg:
            msg[name] = " ".join(value.split())
    # Postmark's MessageID is its own UUID; the sender's real Message-ID arrives in Headers.
    if "Message-ID" not in msg and data.get("MessageID"):
        msg["Message-ID"] = f"<{data['MessageID']}@inbound.postmarkapp.com>"
    for name in ("received-spf",):
        if headers.get(name):
            msg["X-Received-SPF"] = " ".join(headers[name].split())
    text, html = data.get("TextBody") or "", data.get("HtmlBody") or ""
    if text or not html:
        msg.set_content(text or "")
        if html:
            msg.add_alternative(html, subtype="html")
    else:
        msg.set_content(html, subtype="html")
    for att in data.get("Attachments") or []:
        if not isinstance(att, dict):
            continue
        try:
            content = base64.b64decode(att.get("Content") or "", validate=False)
        except ValueError:
            content = b""
        ctype = str(att.get("ContentType") or "application/octet-stream")
        maintype, _, subtype = ctype.partition("/")
        msg.add_attachment(content, maintype=maintype or "application", subtype=subtype or "octet-stream",
                           filename=str(att.get("Name") or ""))
    return bytes(msg)


def to_raw_mime(content_type: str, body: bytes) -> bytes:
    ctype = (content_type or "").strip().lower()
    if ctype.startswith("message/rfc822") or ctype.startswith("text/plain"):
        return body
    if ctype.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        return _from_form(content_type, body)
    if ctype.startswith("application/json"):
        try:
            return _from_postmark(body)
        except BadPayload:
            raise
        except (ValueError, TypeError, KeyError, LookupError, AttributeError) as exc:
            # CR/LF in a header value, a malformed ContentType, a wrong-typed field: the sender's
            # fault, so a 400, not a 500.
            raise BadPayload(f"cannot build a message from the Postmark payload ({type(exc).__name__})") from exc
    raise BadPayload(f"unsupported content type {content_type!r}")
