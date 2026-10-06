"""Small shared helpers with no dependencies on the rest of the package."""
from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse, urlunparse

import httpx


def registrable_domain(host_or_url: str) -> str:
    """Lowercase host with a leading "www." stripped.

    Not a full public-suffix-list registrable-domain computation - just the same
    "strip www." convention providers/vendor.py already used for its own allowlist
    check. footprint.py and trust_pages.py both key off this: querying crt.sh with
    a "www." host instead of the bare domain returns a near-empty certificate-
    transparency result that then gets cached forever (out/cache/crtsh/
    www.<domain>.json holding a single stale entry was the observed symptom), and
    trust_pages.py's status-page guess (`status.<domain>` / `<label>.statuspage.io`)
    turns into nonsense like `status.www.<domain>` / `www.statuspage.io` with the
    "www" label left in.

    Accepts either a full URL (`https://www.example.com/path`) or a bare host
    (`www.example.com`) - `research.py` already has a netloc string in hand at most
    call sites, but this stays permissive so it stays a drop-in for
    `urlparse(website_url).netloc` call sites too.
    """
    host = urlparse(host_or_url).netloc if "//" in host_or_url else host_or_url
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


class UnsafeTargetError(ValueError):
    """A URL whose target is not a public internet address, so it is never fetched.

    The submitted `website` field is attacker-controlled and the deployed app fetches
    it. Without this guard, `http://127.0.0.1:.../` was fetched and its body ended up
    in `research.website_text`, and from there in the LLM prompt, the tracker row and
    the notification e-mail - a read primitive against anything reachable from the
    container, including a cloud metadata endpoint. Measured against a local stand-in
    service before this existed: content leaked, both directly and through a redirect.
    """


_BLOCKED_PORTS: frozenset[int] = frozenset()  # ports are not the control here; addresses are


def _unmap(addr: ipaddress.IPv4Address | ipaddress.IPv6Address):
    """An IPv4-mapped IPv6 literal (`::ffff:127.0.0.1`) reports is_loopback False, so
    the checks below have to run on the mapped IPv4 address instead of the wrapper."""
    return getattr(addr, "ipv4_mapped", None) or addr


def _to_idna(host: str) -> str:
    """The host as the resolver will see it, or unchanged if it cannot be encoded.

    An unencodable host is left alone rather than rejected: the checks after this still
    run on it, and a name the resolver also cannot use fetches nothing anyway.
    """
    if host.isascii():
        return host
    try:
        return host.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        return host


def _as_ip_literal(host: str):
    """The host as an IP address if it IS one in any form the socket layer accepts.

    `ipaddress` is strict, which is right for parsing and wrong for refusing: it rejects
    the legacy IPv4 spellings `inet_aton` still accepts, so every one of these reached
    the DNS branch and was allowed through (measured, all of them loopback):

        2130706433            decimal
        0x7f.0x0.0x0.0x1      hex, dotted
        017700000001          octal
        127.1                 short form

    Whether they then resolve depends on the platform's resolver, which is exactly the
    kind of difference a guard must not depend on - on the Linux container they do.
    Returns None for a real hostname, which goes on to be resolved.
    """
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except (OSError, UnicodeError, ValueError):
        return None


def _is_public(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    a = _unmap(addr)
    if (a.is_loopback or a.is_private or a.is_link_local or a.is_multicast
            or a.is_reserved or a.is_unspecified):
        return False
    return bool(a.is_global)


def assert_public_url(url: str) -> str | None:
    """Raise UnsafeTargetError unless `url` is an http(s) URL whose host resolves only
    to public addresses. Returns the address the caller must connect to, or None.

    The return value is what closes the DNS-rebinding hole. Validating a name and then
    letting the socket layer resolve it a second time is a time-of-check/time-of-use
    race: measured against a local stand-in, a name answering 93.184.216.34 to the guard
    and 127.0.0.1 to httpx was fetched and its body returned (HTTP 200 from loopback).
    A second getaddrinfo() call immediately before the fetch does not fix that - it is
    the same race with a shorter window. So the address validated HERE is returned and
    `safe_get` connects to THAT address; the name is never resolved twice.

    Returns None when there is no resolved address to pin, which is two cases and
    neither is a hole:
      - the host is an IP literal: there is no name, so there is nothing to re-resolve;
      - the name does not resolve at all: the request then fails on its own, and
        treating "DNS said nothing" as "unsafe" would make every offline test with a
        mocked transport fail for the wrong reason. `unknown must not collapse into
        valid` does not apply in reverse here - the fetch still cannot reach anything.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise UnsafeTargetError(f"scheme {parsed.scheme!r} is not fetched (http/https only): {url}")
    if parsed.username or parsed.password:
        raise UnsafeTargetError(f"credentials in a URL are not fetched: {url}")
    host = parsed.hostname
    if not host:
        raise UnsafeTargetError(f"no host in {url!r}")
    # A single trailing dot is the same name, written fully qualified. "127.0.0.1." read
    # as a hostname to `ipaddress`, fell through to DNS, and was allowed.
    host = host.rstrip(".")
    if not host:
        raise UnsafeTargetError(f"no host in {url!r}")
    # IDNA first, so the literal check below sees what the resolver will see. Unicode
    # digit and separator forms fold onto an ASCII address: fullwidth "１２７.０.０.１", the
    # ideographic full stop in "127。0。0。1" and its halfwidth variant all IDNA-encode to
    # 127.0.0.1. Those were refused only because THIS machine's getaddrinfo happened to
    # resolve them - the platform dependence a guard must not rest on. A genuine
    # non-ASCII domain still passes: "еxample.com" with a Cyrillic e encodes to
    # xn--xample-2of.com, a different public domain, and is allowed.
    host = _to_idna(host)
    literal = _as_ip_literal(host)
    if literal is not None:
        if not _is_public(literal):
            raise UnsafeTargetError(f"{host} is not a public address")
        return None  # no name to re-resolve, so nothing to pin
    try:
        infos = socket.getaddrinfo(host, parsed.port or None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError):
        return None  # unresolvable: the request will fail by itself, see the docstring
    pin: str | None = None
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if not _is_public(addr):
            raise UnsafeTargetError(f"{host} resolves to the non-public address {addr}")
        if pin is None:
            pin = str(addr)
    # Every answer was checked, so pinning the first is not "trusting the first answer":
    # a single non-public answer already raised above.
    return pin


MAX_REDIRECTS = 5


def _pinned_url(url: str, addr: str) -> str:
    """`url` with its hostname replaced by the already-validated address `addr`."""
    parsed = urlparse(url)
    host = f"[{addr}]" if ":" in addr else addr          # IPv6 literals need brackets
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urlunparse(parsed._replace(netloc=netloc))


def safe_get(client, url: str, *, max_redirects: int = MAX_REDIRECTS, **kwargs):
    """`client.get(url)` with every hop checked by `assert_public_url`, and connected to
    the address that check actually validated.

    Redirects are followed here rather than by httpx, because httpx validates nothing:
    a public URL answering 302 to `http://169.254.169.254/` was followed before this.
    `follow_redirects` is forced off on the underlying call for the same reason - a
    caller that passes it is asking for the unchecked behaviour.

    When the host is a name that resolved, the request goes to the validated address
    literally, carrying the original `Host:` header and the original name as the TLS
    server name, so virtual hosting and certificate verification are unchanged. This is
    what makes the check binding: resolving the name again at connect time is a
    time-of-check/time-of-use race, and one measured rebinding answer was enough to
    fetch loopback through a guard that had just called the same name public.

    The returned response is re-labelled with the requested URL, not the pinned one, so
    `response.url` stays the address the caller asked for - `vendor.py` matches it
    against a host allowlist and `website.py` derives the redirected origin from it.
    """
    kwargs.pop("follow_redirects", None)
    current = url
    for _ in range(max_redirects + 1):
        pin = assert_public_url(current)
        if pin is None:
            response = client.get(current, follow_redirects=False, **kwargs)
        else:
            parsed = urlparse(current)
            call = dict(kwargs)
            headers = httpx.Headers(call.get("headers") or {})
            headers["Host"] = parsed.netloc            # credentials are refused by the guard
            # No keep-alive on a pinned fetch. httpx pools connections by origin, and the
            # pinned origin is the ADDRESS - so two names on one address are one origin,
            # and the SNI hostname is not part of the pool key. Measured: with a
            # certificate valid only for a.example, a follow-up request for b.example on
            # the same client reused the connection and returned 200 after a single
            # handshake, its name never validated against any certificate. Closing each
            # pinned connection costs a handshake per fetch and makes every name prove
            # itself. This crawler fetches a handful of pages per lead, so that is cheap.
            headers["Connection"] = "close"
            call["headers"] = headers
            extensions = dict(call.get("extensions") or {})
            extensions.setdefault("sni_hostname", parsed.hostname)
            call["extensions"] = extensions
            response = client.get(_pinned_url(current, pin), follow_redirects=False, **call)
            response.request.url = httpx.URL(current)
        location = response.headers.get("location") if response.is_redirect else None
        if not location:
            return response
        current = str(response.url.join(location))
    raise UnsafeTargetError(f"more than {max_redirects} redirects from {url}")


# One outbound identity, in one place. This was five hardcoded copies of
# "leadscout/0.1 (+https://github.com/csakegyruszki/leadscout)", sent as User-Agent to
# every crawled site and public API and as HTTP-Referer to the LLM provider - so the
# DEVELOPMENT repository's URL left the process on every real run. A fork or mirror
# is a different repository, which would make that header simply false, and a
# header naming a repo the reader cannot see is worse than one that names none. The
# version also said 0.1 while the project was at 0.2.x.
#
# No URL by default: a contact URL is good crawler manners only when it resolves, and a
# placeholder that does not is worse than its absence. LEADSCOUT_USER_AGENT overrides it
# with a real one for a deployment that has somewhere to point.
USER_AGENT = os.getenv("LEADSCOUT_USER_AGENT", "").strip() or "leadscout/0.2 (research prototype)"
