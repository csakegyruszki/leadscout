"""Lead extraction from a ParsedMessage, and the prompt-injection heuristic.

Deterministic first. An LLM may fill fields that are STILL missing afterwards, with the
body wrapped as untrusted evidence, the answer schema-validated and any proposed website
restricted to a plain http(s) URL whose host actually appears in the mail.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError, field_validator

from ..prompt_safety import INJECTION_WARNING, wrap_evidence
from ..util import registrable_domain
from .parse import ParsedMessage

SIZE_BANDS = ("1-10", "11-50", "51-200", "201-1000", "1000+")

FREE_MAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com", "msn.com", "yahoo.com",
    "yahoo.co.uk", "yahoo.de", "ymail.com", "gmx.com", "gmx.net", "gmx.de", "gmx.at", "web.de", "freemail.hu",
    "citromail.hu", "t-online.de", "freenet.de", "proton.me", "protonmail.com", "pm.me", "icloud.com",
    "me.com", "mac.com", "aol.com", "mail.com", "mail.ru", "yandex.com", "yandex.ru", "zoho.com", "fastmail.com",
    "tutanota.com", "tuta.io", "hushmail.com", "seznam.cz", "wp.pl", "o2.pl", "interia.pl", "libero.it",
    "orange.fr", "wanadoo.fr", "free.fr", "laposte.net", "indiamail.com", "qq.com", "163.com", "126.com",
    "vipmail.hu", "invitel.hu", "t-online.hu", "mailbox.org", "posteo.de", "hey.com",
})

# Hosts that are never "the company's website": shorteners, tracking redirectors, social
# networks, calendar/meeting links, mail-safety rewriters.
_NOISE_HOSTS = (
    "bit.ly", "t.co", "lnkd.in", "goo.gl", "tinyurl.com", "ow.ly", "buff.ly", "is.gd", "rebrand.ly",
    "linkedin.com", "facebook.com", "fb.com", "twitter.com", "x.com", "instagram.com", "youtube.com",
    "youtu.be", "tiktok.com", "github.com", "medium.com", "wa.me", "t.me", "calendly.com", "zoom.us",
    "teams.microsoft.com", "meet.google.com", "google.com", "goo.gl", "microsoft.com", "outlook.com",
    "safelinks.protection.outlook.com", "mailchi.mp", "list-manage.com", "sendgrid.net", "mandrillapp.com",
    "hubspotlinks.com", "hs-sites.com", "clicks.mlsend.com", "doubleclick.net", "mailtrack.io",
    "docs.google.com", "drive.google.com", "dropbox.com", "wetransfer.com", "xing.com", "w3.org",
    "example.com", "example.org", "example.net",
)

_URL = re.compile(
    r"(?i)\b(?:https?://|www\.)[a-z0-9][a-z0-9\-._~%]*(?:\.[a-z0-9\-]+)+(?::\d+)?(?:/[^\s<>\"'()\[\]]*)?")
_EMAIL = re.compile(r"(?i)\b[a-z0-9._%+\-]+@([a-z0-9\-]+(?:\.[a-z0-9\-]+)+)\b")
_BARE_DOMAIN = re.compile(r"(?i)^\W*([a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.[a-z]{2,})\W*$")
_LEGAL = re.compile(
    r"(?i)\b(?:kft|zrt|nyrt|bt|kkt|rt|gmbh|ag|kg|ug|ltd|limited|llc|l\.l\.c|inc|corp|corporation|co\.|plc|"
    r"s\.p\.a|s\.a\.|sa|sas|sarl|srl|s\.r\.l|spa|b\.v|bv|nv|oy|ab|as|a/s|aps|sp\. z o\.o|s\.r\.o|spol\.)\b\.?")
_TITLE = re.compile(
    r"(?i)\b(chief\b.{0,30}officer|ceo|cto|cio|coo|cfo|ciso|cpo|founder|co-?founder|owner|president|"
    r"vp\b|vice president|head of\b.{0,40}|director\b.{0,30}|manager\b.{0,30}|(?:team |tech |engineering )?lead|"
    r"engineer|architect|developer|consultant|devops|sre|ügyvezető|vezérigazgató|igazgató|tulajdonos|"
    r"alapító|vezető|menedzser|mérnök|fejlesztő|geschäftsführer|leiter(?:in)?)\b")
_EMP = re.compile(
    r"(?i)\b(?:about |around |approx(?:imately)?\.? |~|over |under |kb\.? |körülbelül |nagyjából |ca\.? )?"
    r"(\d[\d.,]{0,11})(?:\s*[-–]\s*(\d[\d.,]{0,11}))?\s*\+?\s*"
    r"(?:employees|people|staff|engineers|developers|team members|FTE|emberes|munkatárs\w*|dolgozó\w*|"
    r"fős|fő\b|alkalmazott\w*|mitarbeiter\w*)")
_TEAM_OF = re.compile(r"(?i)\b(?:team|company|cég|csapat)\s+(?:of|size)\s*[:\-]?\s*(\d[\d.,]{0,11})")
_LABEL = {
    "company": re.compile(r"(?im)^\s*(?:company|organi[sz]ation|cég(?:név)?|unternehmen|firma)\s*[:=]\s*(.+?)\s*$"),
    "job_title": re.compile(
        r"(?im)^\s*(?:job title|title|position|role|beosztás|pozíció|munkakör)\s*[:=]\s*(.+?)\s*$"),
}


def _host(url: str) -> str:
    u = url if "//" in url else "https://" + url
    try:
        return (urlparse(u).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def _is_noise(host: str) -> bool:
    return any(host == n or host.endswith("." + n) for n in _NOISE_HOSTS)


def is_free_mail(domain: str) -> bool:
    d = domain.lower()
    return d in FREE_MAIL_DOMAINS or any(d.endswith("." + f) for f in ("gmail.com", "outlook.com"))


def _site_root(host: str) -> str:
    """https://<host> for a business host. Marketing/mail subdomains of the sender
    (mail., smtp., e., em.) are dropped; anything else is kept as written."""
    labels = host.split(".")
    while len(labels) > 2 and labels[0] in ("mail", "smtp", "mx", "email", "e", "em", "send", "mailer", "news"):
        labels = labels[1:]
    return "https://" + ".".join(labels)


def band_for(n: int) -> str:
    for limit, band in ((10, "1-10"), (50, "11-50"), (200, "51-200"), (1000, "201-1000")):
        if n <= limit:
            return band
    return "1000+"


def _num(s: str) -> int | None:
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", s):      # thousands separators: 1,200 / 1.200
        s = s.replace(",", "").replace(".", "")
    else:
        s = s.split(",")[0].split(".")[0]
    return int(s) if s.isdigit() else None


# --- injection heuristic --------------------------------------------------------------------

_INJECTION = {
    "ignore_instructions": re.compile(
        r"(?i)\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|any|your|the|these)\b"
        r"[^.\n]{0,30}\b(?:instructions?|rules?|prompts?|guidelines?|policy|policies|restrictions?|safeguards?)\b"
        r"|hagyd figyelmen kívül|előző utasítás|ignoriere (?:alle )?(?:vorherigen|obigen) anweisungen"),
    "addressed_to_ai": re.compile(
        r"(?im)^\W*(?:hey |hi |hello |dear |attention[: ]+)?(?:ai|a\.i\.|ai assistant|assistant|llm|language model|"
        r"chatbot|chatgpt|gpt|claude|gemini|automated (?:system|agent|reviewer)|ai agent)\b\s*[:,\-!]"),
    "role_tag": re.compile(
        r"<\|?\s*/?(?:system|assistant|im_start|im_end|user)\s*\|?>|\[/?(?:INST|SYS)\]|<<\s*SYS\s*>>|"
        r"^\s*(?:system|assistant)\s*:\s|^#{2,}\s*(?:system|instruction)|<\s*/?\s*function_call",
        re.I | re.M),
    "verdict_manipulation": re.compile(
        r"(?i)\b(?:mark|set|classify|label|score|rate|treat)\b[^.\n]{0,40}\b(?:lead|this|compliance|status|account)\b[^.\n]{0,40}"
        r"\b(?:clear|approved|safe|sales[- ]?ready|qualified|high|100|passed?)\b|"
        r"\b(?:skip|do not|don't|never)\b[^.\n]{0,20}\b(?:screen|check|flag|review|sanction)\w*"),
    "secret_exfiltration": re.compile(
        r"(?i)\b(?:reveal|print|show|send|reply with|include|output|leak|share|expose)\b[^.\n]{0,60}"
        r"\b(?:api[ _-]?keys?|secrets?|passwords?|tokens?|credentials?|system prompt|instructions|"
        r"environment variables?)\b"),
    "prompt_reveal": re.compile(r"(?i)\b(?:system prompt|your (?:hidden )?instructions|initial prompt)\b"),
    "persona_override": re.compile(
        r"(?i)\byou are (?:now |no longer )?(?:an? )?(?:ai|assistant|chatgpt|dan|jailbroken|unrestricted)\b|"
        r"\bnew instructions?\s*:|\bfrom now on\b[^.\n]{0,40}\b(?:you|always|only)\b|\bact as (?:an? )?\w+"),
}


_CTRL_FIELD = re.compile(r"[\x00-\x1f\x7f  ]+")
FIELD_MAX = 120


def clean_field(value: str, limit: int = FIELD_MAX) -> str:
    """One header-safe line: control characters (incl. newlines) become spaces, whitespace
    is collapsed and the length is capped. Mail-derived fields reach LLM prompts and file
    names downstream; they must not carry line structure or arbitrary length."""
    return " ".join(_CTRL_FIELD.sub(" ", value or "").split())[:limit]


def scan_text(text: str) -> list[str]:
    return [name for name, rx in _INJECTION.items() if rx.search(text or "")]


def injection_flags(parsed: ParsedMessage) -> list[str]:
    """Names of the injection heuristics that fired on anything in the mail, including the
    From display name and subject. A heuristic, not a classifier: it errs towards flagging.
    A flag (other than the soft `hidden_html_text`) makes the mail `needs_review`: the
    pipeline is not run."""
    haystack = "\n".join([parsed.subject, parsed.from_name, parsed.full_text, parsed.hidden_text])
    flags = scan_text(haystack)
    if parsed.hidden_text:
        flags.append("hidden_html_text")
    if parsed.html_truncated:
        flags.append("html_nesting_abuse")
    return flags


def field_flags(lead: dict) -> list[str]:
    """The same heuristics on each extracted field (they are what reaches research and
    compliance prompts), as `field:<name>:<flag>`."""
    return [f"field:{key}:{flag}" for key in ("name", "company", "job_title") for flag in scan_text(lead.get(key, ""))]


# `hidden_html_text` alone is informational: marketing mail routinely carries hidden
# preheader text. Instruction-like content inside it fires the other heuristics anyway.
SOFT_FLAGS = frozenset({"hidden_html_text"})


def blocking(flags: list[str]) -> bool:
    return any(f not in SOFT_FLAGS for f in flags)


# --- extraction -----------------------------------------------------------------------------

@dataclass
class Extraction:
    lead: dict = field(default_factory=lambda: {
        "name": "", "email": "", "company": "", "website": "", "job_title": "", "company_size_band": ""})
    methods: dict = field(default_factory=dict)
    llm_used: bool = False
    llm_note: str = ""
    missing: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.lead["website"] and self.lead["company"] and self.lead["email"])


def _urls(text: str) -> list[str]:
    out = []
    for m in _URL.findall(text or ""):
        host = _host(m.rstrip(".,;:"))
        if host and not _is_noise(host) and not is_free_mail(host) and host not in out:
            out.append(host)
    return out


def _signature_hosts(parsed: ParsedMessage) -> list[str]:
    hosts: list[str] = []
    sig = parsed.signature
    for h in _urls(sig):
        hosts.append(h)
    for dom in _EMAIL.findall(sig):
        d = dom.lower()
        if not is_free_mail(d) and not _is_noise(d) and d not in hosts:
            hosts.append(d)
    for line in sig.splitlines():
        m = _BARE_DOMAIN.match(line)
        if m and "@" not in line:
            d = m.group(1).lower().removeprefix("www.")
            if not is_free_mail(d) and not _is_noise(d) and d not in hosts:
                hosts.append(d)
    return hosts


def _name_from(parsed: ParsedMessage) -> tuple[str, str]:
    name = re.sub(r"\s+", " ", parsed.from_name.strip().strip("\"'")).strip()
    if name and "@" not in name:
        return name, "from_header"
    local = parsed.from_addr.split("@")[0]
    return " ".join(w.capitalize() for w in re.split(r"[._\-+]+", local) if w and not w.isdigit()), "email_local_part"


def _company_from_signature(parsed: ParsedMessage) -> str:
    """First signature line that ends in a legal-form token (optionally followed by
    ", City"); a stray "as"/"ab"/"sa" mid-sentence does not qualify."""
    for line in parsed.signature.splitlines():
        s = line.strip(" \t|*_•·")
        if not (2 < len(s) <= 80) or "@" in s or _URL.search(s):
            continue
        if s.lower().startswith(("http", "tel", "phone", "mob")):
            continue
        for m in _LEGAL.finditer(s):
            rest = s[m.end():].strip()
            if not rest or rest.startswith(","):
                return s[:m.end()].strip(" ,")
    return ""


def _title_from_signature(parsed: ParsedMessage) -> str:
    for line in parsed.signature.splitlines():
        s = line.strip(" \t|*_•·")
        if 2 < len(s) <= 70 and _TITLE.search(s) and "@" not in s and not _URL.search(s) \
                and not re.search(r"\d{5,}", s):
            return s.split("|")[0].split(",")[0].strip() if len(s) > 45 else s
    return ""


def _size_from(text: str) -> str:
    for m in _EMP.finditer(text):
        lo, hi = _num(m.group(1)), _num(m.group(2)) if m.group(2) else None
        if lo is None:
            continue
        # Prefer the stated upper bound when a range is given, but a stated band wins as written.
        if hi and f"{lo}-{hi}" in SIZE_BANDS:
            return f"{lo}-{hi}"
        return band_for(hi or lo)
    m = _TEAM_OF.search(text)
    if m and (n := _num(m.group(1))) is not None:
        return band_for(n)
    return ""


def _finalise(ex: Extraction) -> Extraction:
    for key in ("name", "company", "job_title"):
        ex.lead[key] = clean_field(ex.lead[key])
    ex.lead["website"] = ex.lead["website"][:200]
    ex.missing = [k for k in ("website", "company", "job_title", "company_size_band") if not ex.lead[k]]
    return ex


def extract_deterministic(parsed: ParsedMessage) -> Extraction:
    ex = Extraction()
    lead, how = ex.lead, ex.methods
    lead["email"] = parsed.from_addr
    how["email"] = "forwarded_original_sender" if parsed.forwarded_by else "from_header"
    lead["name"], how["name"] = _name_from(parsed)

    domain = parsed.from_addr.split("@")[-1] if parsed.from_addr else ""
    if domain and not is_free_mail(domain) and not _is_noise(domain):
        lead["website"], how["website"] = _site_root(domain), "sender_domain"
    else:
        candidates = _signature_hosts(parsed)
        # Body links count after the signature: a link in the middle of the text is more
        # likely a reference to someone else's page than the sender's own site.
        for h in _urls(parsed.body) + [_host(x) for x in parsed.links]:
            if h and not _is_noise(h) and not is_free_mail(h) and h not in candidates:
                candidates.append(h)
        if candidates:
            lead["website"] = _site_root(candidates[0])
            how["website"] = "signature_or_body_url"

    text = parsed.body + "\n" + parsed.signature
    m = _LABEL["company"].search(text)
    if m:
        lead["company"], how["company"] = m.group(1)[:120], "labelled_in_body"
    elif (c := _company_from_signature(parsed)):
        lead["company"], how["company"] = c, "signature_legal_form"
    elif lead["website"]:
        host = _host(lead["website"])
        label = registrable_domain(host).split(".")[0]
        if label:
            lead["company"], how["company"] = label.replace("-", " ").title(), "website_domain_label"

    m = _LABEL["job_title"].search(text)
    if m:
        lead["job_title"], how["job_title"] = m.group(1)[:80], "labelled_in_body"
    elif (t := _title_from_signature(parsed)):
        lead["job_title"], how["job_title"] = t, "signature_title_word"

    if band := _size_from(text):
        lead["company_size_band"], how["company_size_band"] = band, "stated_headcount_regex"

    return _finalise(ex)


# --- optional LLM fill-in ---------------------------------------------------------------------

_SYSTEM = (
    "You extract contact details from one inbound sales e-mail. Reply with a single JSON object with the keys "
    "company, job_title, company_size_band, website. Use an empty string for anything the mail does not state "
    "explicitly. company_size_band must be one of 1-10, 11-50, 51-200, 201-1000, 1000+ or empty. website must be "
    "the sender's own company website exactly as written in the mail, or empty. Do not guess. "
    + INJECTION_WARNING
)


class LLMFill(BaseModel):
    company: str = ""
    job_title: str = ""
    company_size_band: str = ""
    website: str = ""

    @field_validator("company", "job_title", "company_size_band", "website", mode="before")
    @classmethod
    def _str(cls, v):
        if v is None:
            return ""
        if not isinstance(v, str):
            raise ValueError("must be a string")
        return v.strip()

    @field_validator("company")
    @classmethod
    def _company(cls, v: str) -> str:
        if len(v) > 120 or "\n" in v:
            raise ValueError("company too long or multi-line")
        return v

    @field_validator("job_title")
    @classmethod
    def _title(cls, v: str) -> str:
        if len(v) > 80 or "\n" in v:
            raise ValueError("job_title too long or multi-line")
        return v

    @field_validator("company_size_band")
    @classmethod
    def _band(cls, v: str) -> str:
        if v and v not in SIZE_BANDS:
            raise ValueError(f"company_size_band must be one of {SIZE_BANDS}")
        return v

    @field_validator("website")
    @classmethod
    def _site(cls, v: str) -> str:
        if not v:
            return v
        if not re.fullmatch(r"https?://[^\s<>\"']{3,200}", v):
            raise ValueError("website must be a plain http(s) URL")
        u = urlparse(v)
        if u.username or u.password or not u.hostname or "." not in u.hostname:
            raise ValueError("website has credentials or no usable host")
        return v


LLMCallable = Callable[[str, str], dict]


def fill_with_llm(parsed: ParsedMessage, ex: Extraction, ask: LLMCallable) -> Extraction:
    """Fill ONLY the still-missing fields. `ask(system, user) -> dict`. Any failure leaves
    the deterministic result untouched and is recorded in `llm_note`."""
    wanted = [k for k in ("website", "company", "job_title", "company_size_band") if not ex.lead[k]]
    if not wanted:
        return ex
    body = (f"Subject: {parsed.subject}\nFrom: {parsed.from_name} <{parsed.from_addr}>\n\n"
            f"{parsed.body}\n\n{parsed.signature}")
    try:
        raw = ask(_SYSTEM, wrap_evidence("inbound-mail", "inbound-email", body))
        fill = LLMFill.model_validate(raw)
    except (ValidationError, Exception) as exc:  # noqa: BLE001 - extraction must survive a bad model answer
        ex.llm_note = f"llm fill-in discarded: {type(exc).__name__}"
        return ex
    ex.llm_used = True
    haystack = (parsed.subject + "\n" + parsed.full_text).lower()
    for key in wanted:
        value = getattr(fill, key)
        if not value:
            continue
        if key == "website":
            host = _host(value)
            if not host or _is_noise(host) or is_free_mail(host) or host not in haystack:
                ex.llm_note = (ex.llm_note + "; " if ex.llm_note else "") + "proposed website rejected"
                continue
            value = _site_root(host)
        ex.lead[key] = value
        ex.methods[key] = "llm_fill_in"
    ex.missing = [k for k in ("website", "company", "job_title", "company_size_band") if not ex.lead[k]]
    return ex


def extract_lead(parsed: ParsedMessage, ask: LLMCallable | None = None, *, allow_llm: bool = True) -> Extraction:
    ex = extract_deterministic(parsed)
    if ask is not None and allow_llm and ex.missing:
        ex = fill_with_llm(parsed, ex, ask)
        if ex.lead["website"] and not ex.lead["company"]:   # derive from the (validated) website
            label = registrable_domain(_host(ex.lead["website"])).split(".")[0]
            ex.lead["company"], ex.methods["company"] = label.replace("-", " ").title(), "website_domain_label"
    return _finalise(ex)
