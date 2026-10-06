"""Parse a raw RFC 822 / MIME message into the fields the pipeline needs.

Nothing here is trusted: From is spoofable, Authentication-Results is recorded and never
used as proof, attachments are measured and never decoded into text, and the body is
data for extraction (see extract.py), not instructions.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from email import message_from_bytes, policy
from email.message import Message
from email.utils import getaddresses, parseaddr
from html.parser import HTMLParser
from pathlib import Path

DEFAULT_MAX_BYTES = 10 * 1024 * 1024


@dataclass
class Attachment:
    filename: str
    content_type: str
    size: int


@dataclass
class ParsedMessage:
    message_id: str
    message_id_synthesised: bool
    raw_sha256: str
    size_bytes: int
    from_name: str = ""
    from_addr: str = ""
    reply_to: str = ""
    to: list[str] = field(default_factory=list)
    subject: str = ""
    date: str = ""
    in_reply_to: str = ""
    references: list[str] = field(default_factory=list)
    auth_results: dict = field(default_factory=dict)
    auth_top: dict = field(default_factory=dict)
    html_truncated: bool = False
    auto_generated: bool = False
    # Text. `full_text` is everything visible (used for injection scanning); `body` is
    # the sender's own new text with quoted history and signature removed.
    full_text: str = ""
    body: str = ""
    signature: str = ""
    links: list[str] = field(default_factory=list)
    hidden_text: str = ""
    forwarded_by: dict | None = None      # {"name", "addr"} of whoever forwarded it
    forwarded_note: str = ""              # the forwarder's own words above the forwarded block
    attachments: list[Attachment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def meta(self) -> dict:
        return {
            "message_id": self.message_id,
            "message_id_synthesised": self.message_id_synthesised,
            "from_name": self.from_name, "from_addr": self.from_addr,
            "reply_to": self.reply_to, "to": self.to, "subject": self.subject,
            "date": self.date, "in_reply_to": self.in_reply_to, "references": self.references,
            "forwarded_by": self.forwarded_by, "size_bytes": self.size_bytes,
            "raw_sha256": self.raw_sha256, "auto_generated": self.auto_generated,
        }


# --- helpers ------------------------------------------------------------------------

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(text: str) -> str:
    return _CTRL.sub("", text or "")


def _hdr(msg: Message, name: str) -> str:
    try:
        value = msg.get(name)
    except Exception:  # noqa: BLE001 - a malformed header must not abort the whole parse
        return ""
    return _clean(" ".join(str(value).split())) if value else ""


def normalise_message_id(value: str) -> str:
    value = (value or "").strip().strip("<>").strip()
    return f"<{value.lower()}>" if value else ""


def _addr(value: str) -> tuple[str, str]:
    name, addr = parseaddr(value or "")
    addr = addr.strip().strip("<>").lower()
    return _clean(name).strip(), addr if "@" in addr else ""


_AUTH = re.compile(r"\b(spf|dkim|dmarc)\s*=\s*([a-z]+)", re.I)


def _auth_results(msg: Message) -> dict:
    out: dict[str, str] = {}
    raw = []
    for value in msg.get_all("Authentication-Results") or []:
        raw.append(" ".join(str(value).split()))
        for key, verdict in _AUTH.findall(str(value)):
            out.setdefault(key.lower(), verdict.lower())
    # Recorded only. A "pass" is a claim made by whichever hop wrote the header, and a
    # header injected by the sender is indistinguishable from one added by our MTA.
    return {**out, "raw": [r[:300] for r in raw[:3]], "trusted": False}


_DKIM_RES = re.compile(r"\bdkim\s*=\s*([a-z]+)[^;]*?\bheader\.[di]\s*=\s*@?([a-z0-9.\-]+)", re.I)
_DMARC_RES = re.compile(r"\bdmarc\s*=\s*([a-z]+)[^;]*?\bheader\.from\s*=\s*([a-z0-9.\-]+)", re.I)


def topmost_auth(msg: Message) -> dict:
    """Verdicts of the TOPMOST Authentication-Results header only, with the domains they
    were evaluated for: {"dmarc": [(result, header.from)], "dkim": [(result, d)]}. Used to
    gate sending a reply; still only as trustworthy as the hop that wrote that header."""
    values = msg.get_all("Authentication-Results") or []
    top = " ".join(str(values[0]).split())[:2000] if values else ""
    return {
        "dmarc": [(r.lower(), d.lower()) for r, d in _DMARC_RES.findall(top)],
        "dkim": [(r.lower(), d.lower()) for r, d in _DKIM_RES.findall(top)],
    }


# --- HTML -> text ---------------------------------------------------------------------

_VOID = {"br", "img", "meta", "link", "input", "hr", "area", "base", "col", "embed", "source", "wbr"}
_BLOCK = {"p", "div", "br", "tr", "li", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6",
          "blockquote", "section", "article", "header", "footer"}
_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px|pt|em|%)?\s*(?:;|$)|"
    r"opacity\s*:\s*0(?:\.0+)?\s*(?:;|$)|color\s*:\s*(?:#fff(?:fff)?|white)\s*(?:;|$)|"
    r"(?:max-)?height\s*:\s*0(?:px)?\s*(?:;|$)", re.I)


MAX_HTML_CHARS = 1_000_000      # input beyond this is not needed for lead extraction
MAX_HTML_DEPTH = 400            # nesting beyond this is abuse, not layout
MAX_HTML_EVENTS = 400_000       # start/end tags + text nodes handled
MAX_TEXT_CHARS = 200_000        # text handed on to the extractors


class _Html2Text(HTMLParser):
    """HTML -> visible text + hidden text. O(1) per node: the hidden state is a counter, not
    a scan of the open-element stack (a scan was quadratic: "<div>" * 8000 took ~1.5 s)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.visible: list[str] = []
        self.hidden: list[str] = []
        self.links: list[str] = []
        self.truncated = False
        self._stack: list[tuple[str, bool]] = []   # (tag, hidden)
        self._open: dict[str, int] = {}            # open count per tag name
        self._hidden_open = 0                      # hidden entries currently on the stack
        self._skip = 0                             # inside script/style/head
        self._events = 0

    def _stop(self) -> None:
        self.truncated = True

    def _tick(self) -> bool:
        self._events += 1
        if self._events > MAX_HTML_EVENTS:
            self._stop()
        return not self.truncated

    def handle_starttag(self, tag, attrs):
        if not self._tick():
            return
        a = {k: (v or "") for k, v in attrs}
        if tag in ("script", "style", "head", "title"):
            self._skip += 1
        if tag == "a" and a.get("href", "").lower().startswith(("http://", "https://", "mailto:")):
            self.links.append(a["href"])
        if tag in _BLOCK and not self._hidden_open:
            self.visible.append("\n")
        if tag in _VOID:
            return
        if len(self._stack) >= MAX_HTML_DEPTH:
            self._stop()
            return
        hidden = "hidden" in a or bool(_HIDDEN_STYLE.search(a.get("style", "")))
        self._stack.append((tag, hidden))
        self._open[tag] = self._open.get(tag, 0) + 1
        self._hidden_open += hidden

    def handle_startendtag(self, tag, attrs):
        if self._tick() and tag in _BLOCK and not self._hidden_open:
            self.visible.append("\n")

    def handle_endtag(self, tag):
        if not self._tick():
            return
        if tag in ("script", "style", "head", "title") and self._skip:
            self._skip -= 1
        if self._open.get(tag):                    # an unmatched end tag costs O(1), not O(depth)
            while self._stack:
                name, hidden = self._stack.pop()
                self._open[name] -= 1
                self._hidden_open -= hidden
                if name == tag:
                    break
        if tag in _BLOCK and not self._hidden_open:
            self.visible.append("\n")

    def handle_data(self, data):
        if self._skip or not self._tick():
            return
        (self.hidden if self._hidden_open else self.visible).append(data)


def html_to_text(html: str) -> tuple[str, str, list[str], bool]:
    """(visible text, hidden text, link hrefs, truncated). Script/style/head content is dropped;
    `truncated` is set when the nesting/size caps were hit (itself a signal)."""
    p = _Html2Text()
    if len(html) > MAX_HTML_CHARS:
        html, p.truncated = html[:MAX_HTML_CHARS], True
    try:
        p.feed(html)
        p.close()
    except Exception:  # noqa: BLE001 - tag soup from the wild; keep what was read
        pass
    text = re.sub(r"[ \t\xa0]+", " ", "".join(p.visible))
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, " ".join(" ".join(p.hidden).split()), p.links, p.truncated


# --- body text handling -----------------------------------------------------------------

_REPLY_HEADERS = [
    re.compile(r"^On\b.{0,300}\bwrote:\s*$", re.I),
    re.compile(r"^Am\b.{0,300}\bschrieb\b.{0,200}:\s*$", re.I),
    re.compile(r"^Le\b.{0,300}\ba écrit\s*:\s*$", re.I),
    re.compile(r"^.{0,300}\bírta:\s*$", re.I),
    re.compile(r"^-{2,}\s*(original message|eredeti üzenet|ursprüngliche nachricht)\s*-{2,}\s*$", re.I),
    re.compile(r"^_{10,}\s*$"),
]
_OUTLOOK_FROM = re.compile(r"^\*?(from|feladó|von|de)\s*:\*?\s+\S", re.I)
_OUTLOOK_FOLLOW = re.compile(
    r"^\*?(sent|date|küldve|elküldve|gesendet|envoyé|to|címzett|subject|tárgy|cc)\s*:", re.I)


def _norm_nl(text: str) -> str:
    return _clean(text.replace("\r\n", "\n").replace("\r", "\n"))


def strip_quoted(text: str) -> tuple[str, bool]:
    """Remove quoted reply history. Returns (new text, whether anything was removed)."""
    lines = _norm_nl(text).split("\n")
    cut = None
    for i, line in enumerate(lines):
        s = line.strip()
        if any(p.match(s) for p in _REPLY_HEADERS):
            cut = i
            break
        # "On Mon, 3 Jun 2024, Name <a@b.c>\nwrote:" - the attribution wrapped onto two lines
        if re.match(r"^On\b", s, re.I) and i + 1 < len(lines) \
                and re.match(r"^wrote:\s*$", lines[i + 1].strip(), re.I):
            cut = i
            break
        if _OUTLOOK_FROM.match(s) and any(_OUTLOOK_FOLLOW.match(x.strip()) for x in lines[i + 1:i + 5]):
            cut = i
            break
    kept = lines[:cut] if cut is not None else lines
    n_before = len(kept)
    kept = [ln for ln in kept if not ln.lstrip().startswith(">")]
    changed = cut is not None or len(kept) != n_before
    return "\n".join(kept).strip(), changed


_FWD_MARKERS = [
    re.compile(r"^-{3,}\s*forwarded message\s*-{3,}\s*$", re.I),
    re.compile(r"^begin forwarded message\s*:\s*$", re.I),
    re.compile(r"^-{3,}\s*(továbbított üzenet|weitergeleitete nachricht|message transféré)\s*-{3,}\s*$", re.I),
]
_FWD_HEADER = re.compile(
    r"^\*?(from|feladó|von|de|sent|date|dátum|datum|subject|tárgy|betreff|to|címzett|an|cc|reply-to)"
    r"\s*:\*?\s*(.*)$", re.I)
_FROM_KEYS = ("from", "feladó", "von", "de")
_SUBJECT_KEYS = ("subject", "tárgy", "betreff")


def split_forwarded(text: str) -> tuple[str, dict, str] | None:
    """If `text` carries an inline forwarded block: (forwarder's note, original headers, original body)."""
    lines = _norm_nl(text).split("\n")
    for i, line in enumerate(lines):
        if not any(p.match(line.strip()) for p in _FWD_MARKERS):
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        headers: dict[str, str] = {}
        while j < len(lines):
            m = _FWD_HEADER.match(lines[j].strip())
            if not m:
                break
            headers.setdefault(m.group(1).lower(), m.group(2).strip().strip("*").strip())
            j += 1
        if not headers:
            continue
        return "\n".join(lines[:i]).strip(), headers, "\n".join(lines[j:]).strip()
    return None


_SIGNOFF = re.compile(
    r"^(best regards|kind regards|warm regards|regards|best wishes|best|thanks|thank you|many thanks|cheers|"
    r"sincerely|yours sincerely|yours faithfully|üdvözlettel|szívélyes üdvözlettel|tisztelettel|köszönettel|"
    r"üdv|mit freundlichen grüßen|freundliche grüße|viele grüße|beste grüße|cordialement|bien cordialement)"
    r"\s*[,.!]*\s*$", re.I)


def split_signature(text: str, sender_name: str = "") -> tuple[str, str]:
    """(body, signature). A "-- " delimiter and a sign-off line can both be present
    ("Best regards, Name, Title" above, a boilerplate block below the delimiter); the
    signature then starts at the sign-off. Without either, a trailing block that starts
    at the sender's own name is taken as the signature (Outlook-style mails)."""
    lines = _norm_nl(text).split("\n")
    after = ""
    for i, line in enumerate(lines):
        if line.startswith("--") and line.rstrip("  ") == "--":
            lines, after = lines[:i], "\n".join(lines[i + 1:]).strip()
            break
    head, sig = "\n".join(lines).strip(), ""
    for i in range(max(0, len(lines) - 14), len(lines)):
        if _SIGNOFF.match(lines[i].strip()):
            head, sig = "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
            break
    else:
        name = " ".join(sender_name.lower().split())
        if name:
            for i in range(max(0, len(lines) - 10), len(lines)):
                if " ".join(lines[i].lower().split()) == name:
                    head, sig = "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
                    break
    return head, "\n".join(x for x in (sig, after) if x).strip()


# --- MIME walk ------------------------------------------------------------------------------

def _part_text(part: Message) -> str:
    try:
        return part.get_content()
    except Exception:  # noqa: BLE001 - unknown charset or broken encoding: decode leniently
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")


def _is_body_part(part: Message) -> bool:
    return (part.get_content_type() in ("text/plain", "text/html")
            and part.get_content_disposition() != "attachment" and not part.get_filename())


def _walk(msg: Message):
    texts, htmls, attachments, rfc822 = [], [], [], []
    for part in msg.walk():
        ctype = part.get_content_type()
        if part.is_multipart() and ctype != "message/rfc822":
            continue
        if ctype == "message/rfc822":
            inner = part.get_payload()
            inner = inner[0] if isinstance(inner, list) and inner else None
            if inner is not None:
                rfc822.append(inner)
            try:
                size = len(part.as_bytes())
            except Exception:  # noqa: BLE001
                size = 0
            attachments.append(Attachment(_clean(part.get_filename() or "forwarded.eml"), ctype, size))
        elif _is_body_part(part):
            (texts if ctype == "text/plain" else htmls).append(part)
        else:
            # Measured, never decoded into text: the bytes go nowhere near the pipeline or an LLM.
            payload = part.get_payload(decode=True)
            attachments.append(Attachment(_clean(part.get_filename() or ""), ctype, len(payload or b"")))
    return texts, htmls, attachments, rfc822


def _body_of(msg: Message):
    texts, htmls, attachments, rfc822 = _walk(msg)
    hidden, links, truncated = "", [], False
    if texts:
        text = "\n".join(_norm_nl(_part_text(p)) for p in texts)
    elif htmls:
        parts = [html_to_text(_part_text(p)) for p in htmls]
        text = "\n".join(t for t, _, _, _ in parts)
        hidden = " ".join(h for _, h, _, _ in parts if h)
        links = [link for _, _, ls, _ in parts for link in ls]
        truncated = any(tr for _, _, _, tr in parts)
    else:
        text = ""
    if texts:   # the HTML alternative is not used for text, but its hidden content is still scanned
        for p in htmls:
            _, h, ls, tr = html_to_text(_part_text(p))
            hidden = (hidden + " " + h).strip()
            links += ls
            truncated = truncated or tr
    if len(text) > MAX_TEXT_CHARS:
        text, truncated = text[:MAX_TEXT_CHARS], True
    return text, hidden[:MAX_TEXT_CHARS], links[:200], attachments, rfc822, truncated


def parse_message(raw: bytes | str | Path, _depth: int = 0) -> ParsedMessage:
    if isinstance(raw, Path):
        raw = raw.read_bytes()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    msg = message_from_bytes(raw, policy=policy.default)
    sha = hashlib.sha256(raw).hexdigest()

    mid = normalise_message_id(_hdr(msg, "Message-ID"))
    synthesised = not mid
    if synthesised:
        mid = f"<{sha}@leadscout.local>"

    from_name, from_addr = _addr(_hdr(msg, "From"))
    _, reply_to = _addr(_hdr(msg, "Reply-To"))
    to = [a.lower() for _, a in getaddresses([_hdr(msg, "To")]) if "@" in a]
    refs = [normalise_message_id(r) for r in re.findall(r"<[^<>\s]+>", _hdr(msg, "References"))]
    auto = (_hdr(msg, "Auto-Submitted").lower() not in ("", "no")
            or _hdr(msg, "Precedence").lower() in ("bulk", "junk", "list", "auto_reply")
            or bool(re.match(r"(?i)(no-?reply|do-?not-?reply|mailer-daemon|postmaster)$",
                             from_addr.split("@")[0])))

    text, hidden, links, attachments, rfc822, truncated = _body_of(msg)
    p = ParsedMessage(
        message_id=mid, message_id_synthesised=synthesised, raw_sha256=sha, size_bytes=len(raw),
        from_name=from_name, from_addr=from_addr, reply_to=reply_to, to=to,
        subject=_hdr(msg, "Subject"), date=_hdr(msg, "Date"),
        in_reply_to=normalise_message_id(_hdr(msg, "In-Reply-To")), references=refs,
        auth_results=_auth_results(msg), auth_top=topmost_auth(msg), html_truncated=truncated,
        auto_generated=auto, links=links, hidden_text=hidden,
        attachments=attachments,
    )
    if synthesised:
        p.notes.append("no Message-ID header; synthesised from the raw bytes")
    if not from_addr:
        p.notes.append("no parseable From address")

    p.full_text = text
    new_text = text
    forwarded = split_forwarded(text)
    if forwarded:
        note, headers, orig_body = forwarded
        orig_name, orig_addr = _addr(
            next((headers[k] for k in _FROM_KEYS if k in headers), "").replace("mailto:", ""))
        if orig_addr:
            p.forwarded_by = {"name": p.from_name, "addr": p.from_addr}
            p.forwarded_note = note
            p.from_name, p.from_addr = orig_name, orig_addr
            p.subject = next((headers[k] for k in _SUBJECT_KEYS if headers.get(k)), p.subject)
            new_text = orig_body
        else:
            p.notes.append("forwarded block found but its From could not be read")
    elif rfc822 and _depth < 2:
        inner = parse_message(rfc822[0].as_bytes(), _depth + 1)
        p.html_truncated = p.html_truncated or inner.html_truncated
        if inner.from_addr:
            p.forwarded_by = {"name": p.from_name, "addr": p.from_addr}
            p.forwarded_note, new_text = text, inner.full_text
            p.from_name, p.from_addr = inner.from_name, inner.from_addr
            p.subject = inner.subject or p.subject
            p.links += inner.links
            p.hidden_text = (p.hidden_text + " " + inner.hidden_text).strip()
            p.full_text = text + "\n" + inner.full_text
    new_text, quoted = strip_quoted(new_text)
    if quoted:
        p.notes.append("quoted reply history removed")
    p.body, p.signature = split_signature(new_text, p.from_name)
    return p
