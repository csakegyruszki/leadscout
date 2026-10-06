"""The submitted `website` is attacker-controlled and the deployed app fetches it.

Measured before `util.assert_public_url`/`util.safe_get` existed, against a local
stand-in service: submitting `http://127.0.0.1:<port>/internal` fetched it and the
response body ended up in `research.website_text` - and from there in the LLM prompt,
the tracker row and the notification e-mail. A read primitive against anything
reachable from the container, including a cloud metadata endpoint. The redirect path
worked too, because every client in the codebase was built with
`follow_redirects=True` and httpx validates nothing about where it is being sent.

Scope of the guard these tests pin, stated so the tests are not read as claiming more:
it refuses non-public targets on the first hop and on every redirect hop, and the fetch
goes to the address the check validated instead of re-resolving the name at connect
time. That re-resolution was a real hole, not a theoretical one: measured against a
local server, a name answering 93.184.216.34 to the guard and 127.0.0.1 to httpx was
fetched and returned HTTP 200 from loopback.

Still out of scope, so this is not read as claiming more: an address that is genuinely
public at check time but attacker-operated, and whatever the peer does once the
connection is established.
"""
from __future__ import annotations

import httpx
import pytest

from leadscout import research
from leadscout.util import UnsafeTargetError, assert_public_url, safe_get

PRIVATE_URLS = [
    "http://127.0.0.1/",
    "http://127.0.0.1:8080/admin",
    "https://localhost/",
    "http://[::1]/",
    "http://[::ffff:127.0.0.1]/",          # IPv4-mapped IPv6: is_loopback is False on the wrapper
    "http://10.0.0.1/",
    "http://172.16.0.1/",
    "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/",   # AWS/Azure/GCP instance metadata
    "http://169.254.170.2/v2/credentials",        # ECS task metadata
    "http://0.0.0.0/",
    "http://224.0.0.1/",                          # multicast
    "http://[fe80::1]/",                          # IPv6 link-local
    "http://[fd00::1]/",                          # IPv6 unique-local
]

NON_HTTP_URLS = [
    "file:///C:/Windows/win.ini",
    "file:///etc/passwd",
    "ftp://127.0.0.1/",
    "gopher://127.0.0.1:70/",
    "data:text/html,<h1>x</h1>",
]


@pytest.mark.parametrize("url", PRIVATE_URLS)
def test_a_non_public_target_is_refused(url):
    with pytest.raises(UnsafeTargetError):
        assert_public_url(url)


@pytest.mark.parametrize("url", NON_HTTP_URLS)
def test_only_http_and_https_are_fetched(url):
    with pytest.raises(UnsafeTargetError):
        assert_public_url(url)


def test_credentials_in_the_url_are_refused():
    """`http://user:pass@host/` is how a fetcher gets talked into authenticating, and
    the userinfo also hides the real host from a careless reader."""
    with pytest.raises(UnsafeTargetError):
        assert_public_url("http://admin:secret@127.0.0.1/")
    with pytest.raises(UnsafeTargetError):
        assert_public_url("http://user@example.com/")


def test_a_public_host_is_allowed():
    """Offline this resolves to nothing and is allowed through (the request then fails
    on its own); online it resolves to public addresses. Either way it must not raise."""
    assert_public_url("https://example.com/")
    assert_public_url("http://93.184.216.34/")   # a public literal


def test_a_redirect_to_a_private_address_is_refused_mid_chain():
    """The first hop being public proves nothing about the second. This is the case a
    real attacker uses: a public URL they control, answering 302 to the metadata IP."""
    hops: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hops.append(str(request.url))
        if request.url.host == "attacker.example":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
        return httpx.Response(200, text="SHOULD NEVER BE REACHED")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(UnsafeTargetError):
            safe_get(client, "http://attacker.example/")

    assert hops == ["http://attacker.example/"], f"the private hop was attempted: {hops}"


def test_a_redirect_chain_to_a_public_address_still_works():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(301, headers={"location": "https://example.com/end"})
        return httpx.Response(200, text="arrived")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        r = safe_get(client, "https://example.com/start")
    assert r.status_code == 200 and r.text == "arrived"


def test_an_endless_redirect_loop_is_refused_rather_than_followed_forever():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.com/loop"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(UnsafeTargetError):
            safe_get(client, "https://example.com/loop")


def test_safe_get_ignores_a_callers_follow_redirects_argument():
    """A caller asking for httpx's own redirect handling is asking for the unchecked
    behaviour, so the argument is dropped rather than honoured."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "attacker.example":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/"})
        return httpx.Response(200, text="x")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(UnsafeTargetError):
            safe_get(client, "http://attacker.example/", follow_redirects=True)


def test_fetch_website_refuses_instead_of_reporting_the_site_as_unreachable():
    """`fetch_website` never raises, so the refusal has to be visible in what it returns -
    and it must not read as "unreachable", which would describe a site that was tried."""
    text, ok, raw = research.fetch_website("http://169.254.169.254/latest/meta-data/")
    assert ok is False
    assert raw == ""
    assert text == "[refused: not a public address]"


def test_a_bare_domain_still_gets_https_but_a_real_scheme_is_left_alone():
    """The scheme-prepending convenience must not launder a non-http scheme into an
    http one: "file:///etc/passwd" became "https://file:///etc/passwd" before the fix,
    a nonsense host that failed with ConnectError - refused by accident, and reported as
    "unreachable" about a URL that should never have been attempted."""
    assert research._normalise_url("zapier.com") == "https://zapier.com"
    assert research._normalise_url("  zapier.com  ") == "https://zapier.com"
    assert research._normalise_url("https://zapier.com") == "https://zapier.com"
    assert research._normalise_url("file:///etc/passwd") == "file:///etc/passwd"
    text, ok, _ = research.fetch_website("file:///etc/passwd")
    assert ok is False and text == "[refused: not a public address]"


# --- Legacy IPv4 spellings and normalisation bypasses ------------------------------
# `ipaddress` is strict, which is right for parsing and wrong for refusing: every one of
# these reached the DNS branch and was ALLOWED before `_as_ip_literal` existed (measured,
# all of them loopback). Whether they then resolve is platform-dependent, and on the
# Linux container they do - which is exactly the kind of difference a guard must not rest
# on.
BYPASS_SPELLINGS = [
    "http://127.0.0.1./",            # trailing dot: the same name, fully qualified
    "http://2130706433/",            # decimal
    "http://0x7f.0x0.0x0.0x1/",      # hex, dotted
    "http://017700000001/",          # octal
    "http://127.1/",                 # short form
    "http://[0:0:0:0:0:ffff:7f00:1]/",   # IPv4-mapped IPv6, written out
]


@pytest.mark.parametrize("url", BYPASS_SPELLINGS)
def test_a_legacy_spelling_of_a_private_address_is_refused_too(url):
    with pytest.raises(UnsafeTargetError):
        assert_public_url(url)


def test_every_resolved_address_has_to_be_acceptable_not_just_the_first(monkeypatch):
    """A host that answers with a public AND a private address is refused. Checking only
    the first answer would make the guard depend on resolver ordering, which an attacker
    controls by publishing both records."""
    import socket as socket_mod

    def mixed(host, port, *args, **kwargs):
        if host == "mixed.example":
            return [(socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
                    (socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
        raise socket_mod.gaierror("not found")

    monkeypatch.setattr("leadscout.util.socket.getaddrinfo", mixed)
    with pytest.raises(UnsafeTargetError):
        assert_public_url("http://mixed.example/")


def test_a_host_resolving_only_to_public_addresses_is_allowed(monkeypatch):
    import socket as socket_mod

    def public_only(host, port, *args, **kwargs):
        return [(socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
                (socket_mod.AF_INET6, socket_mod.SOCK_STREAM, 6, "", ("2606:2800:220:1:248:1893:25c8:1946", 80, 0, 0))]

    monkeypatch.setattr("leadscout.util.socket.getaddrinfo", public_only)
    assert_public_url("http://public.example/")


# --- Redirect shapes ---------------------------------------------------------------

@pytest.mark.parametrize("status,location", [
    (301, "http://127.0.0.1/"),
    (302, "/../../"),                      # relative, resolved against the current URL
    (302, "//169.254.169.254/latest/"),    # protocol-relative
    (303, "http://10.0.0.1/"),
    (307, "http://169.254.169.254/latest/meta-data/"),
    (308, "http://192.168.1.1/"),
])
def test_a_redirect_to_a_private_target_is_refused_whatever_its_shape(status, location):
    reached: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        reached.append(str(request.url))
        if request.url.host == "public.example" and request.url.path == "/start":
            return httpx.Response(status, headers={"location": location})
        return httpx.Response(200, text="SHOULD NEVER BE REACHED")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        try:
            response = safe_get(client, "http://public.example/start")
        except UnsafeTargetError:
            assert reached == ["http://public.example/start"], reached
            return
    # A relative Location that stays on the public host is legitimately followed; the
    # only thing this test forbids is REACHING a private address.
    assert all("127.0.0.1" not in u and "169.254" not in u and "10.0.0.1" not in u
               and "192.168" not in u for u in reached), reached
    assert response.status_code == 200


def test_the_redirect_budget_is_bounded():
    from leadscout.util import MAX_REDIRECTS
    hops: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hops.append(str(request.url))
        return httpx.Response(302, headers={"location": f"https://example.com/{len(hops)}"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(UnsafeTargetError):
            safe_get(client, "https://example.com/0")
    assert len(hops) == MAX_REDIRECTS + 1, hops


# --- Sibling-host discovery goes through the same guard ----------------------------

def test_sibling_host_discovery_cannot_reach_a_private_address(monkeypatch):
    """The crawl may cross to a sibling host under the same registrable domain. That
    widening must not also widen what the guard allows: a sibling whose name resolves
    into the private range is refused like any other target, and its content never
    reaches the crawl result."""
    import socket as socket_mod

    from leadscout.providers import website as website_provider

    def resolver(host, port, *args, **kwargs):
        if host == "careers.public.example":
            return [(socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
        return [(socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]

    monkeypatch.setattr("leadscout.util.socket.getaddrinfo", resolver)

    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text="<html><body><p>INTERNAL CAREERS PAGE</p></body></html>")

    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(website_provider.httpx, "Client", client_factory)

    landing = '<html><body><a href="https://careers.public.example/">Careers</a></body></html>'
    roots = website_provider._first_party_sibling_roots(landing, "https://public.example/")
    assert roots == ["https://careers.public.example/"], "the sibling was not even nominated"

    pages = website_provider.crawl_subpages("https://public.example/", landing)

    assert not any("careers.public.example" in str(p.get("url", "")) for p in pages), pages
    assert not any("INTERNAL CAREERS PAGE" in str(p.get("text", "")) for p in pages), pages
    assert all("careers.public.example" not in u for u in fetched), fetched


# --- DNS rebinding: the check must bind the connection, not just precede it ---------
#
# This used to be a documented limitation, asserted as such by a test that read the
# docstring. It was measured as an actual exploit instead: with a local HTTP server on
# 127.0.0.1 and a name answering 93.184.216.34 to the guard and 127.0.0.1 to httpx, the
# fetch returned HTTP 200 and the loopback body. A second getaddrinfo() immediately
# before the fetch would not have closed it - that is the same race, run twice. So the
# address the guard validated is now the address the request connects to.

REBIND_HOST = "rebind.example"
PUBLIC_ANSWER = "93.184.216.34"


def _rebinding_resolver(answers):
    """getaddrinfo that answers REBIND_HOST from `answers`, one per call, last repeating."""
    import socket as socket_mod
    calls = {"n": 0}

    def resolver(host, port, *args, **kwargs):
        if host != REBIND_HOST:
            return [(socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, "", (PUBLIC_ANSWER, port or 80))]
        ip = answers[min(calls["n"], len(answers) - 1)]
        calls["n"] += 1
        return [(socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, "", (ip, port or 80))]

    return resolver, calls


def test_a_rebinding_name_cannot_reach_a_loopback_server(monkeypatch):
    """The exploit itself, with a real socket and a real server: if anything regresses to
    resolving the name at connect time, this server records a request."""
    import socket as socket_mod
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "6")
            self.end_headers()
            self.wfile.write(b"SECRET")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        resolver, calls = _rebinding_resolver([PUBLIC_ANSWER, "127.0.0.1"])
        monkeypatch.setattr(socket_mod, "getaddrinfo", resolver)
        with httpx.Client(timeout=2.0) as client:
            try:
                response = safe_get(client, f"http://{REBIND_HOST}:{port}/secret")
            except (UnsafeTargetError, httpx.HTTPError):
                response = None
        assert hits == [], f"the fetch reached the loopback server: {hits}"
        if response is not None:
            assert "SECRET" not in response.text
        assert calls["n"] == 1, f"the name was resolved {calls['n']} times, so the check is not binding"
    finally:
        server.shutdown()


def test_the_request_is_addressed_to_the_validated_address(monkeypatch):
    """The same invariant without a socket: the transport must be handed the validated
    address, carrying the original name as `Host:` and as the TLS server name."""
    import socket as socket_mod

    resolver, _ = _rebinding_resolver([PUBLIC_ANSWER, "127.0.0.1"])
    monkeypatch.setattr(socket_mod, "getaddrinfo", resolver)
    # Recorded as values, not as the request object: `safe_get` re-labels the request
    # afterwards, so holding the object would show the restored URL and hide what the
    # transport was actually handed.
    seen: list[tuple[str, str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["Host"],
                     request.extensions.get("sni_hostname")))
        return httpx.Response(200, text="ok")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = safe_get(client, f"https://{REBIND_HOST}/page")

    url, host_header, sni = seen[0]
    assert url == f"https://{PUBLIC_ANSWER}/page", f"connected to the name, not the address: {url}"
    assert host_header == REBIND_HOST
    assert sni == REBIND_HOST, "TLS would be negotiated for the address"
    # Callers read `response.url`: vendor.py matches it against a host allowlist and
    # website.py derives the redirected origin from it, so it must stay the real name.
    assert str(response.url) == f"https://{REBIND_HOST}/page"


def test_an_ip_literal_target_is_passed_through_unchanged(monkeypatch):
    """There is no name to re-resolve, so there is nothing to pin and nothing to rewrite."""
    seen: list[tuple[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.extensions.get("sni_hostname")))
        return httpx.Response(200, text="ok")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        safe_get(client, f"http://{PUBLIC_ANSWER}/x")

    assert seen[0] == (f"http://{PUBLIC_ANSWER}/x", None)


def test_an_unresolvable_name_is_still_passed_through(monkeypatch):
    """Offline, and in every mocked-transport test in this suite, the name resolves to
    nothing. That must leave the URL alone rather than fail or rewrite it."""
    import socket as socket_mod

    def resolver(host, port, *args, **kwargs):
        raise socket_mod.gaierror("offline")

    monkeypatch.setattr(socket_mod, "getaddrinfo", resolver)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text="ok")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        safe_get(client, "https://public.example/x")

    assert seen == ["https://public.example/x"]


def test_two_pinned_names_do_not_share_one_connection(monkeypatch):
    """Pinning rewrites the origin to the ADDRESS, and httpx pools connections by origin
    while the SNI hostname is not part of the pool key. So two names on one address are
    one origin. Measured with a certificate valid only for `a.example`: a follow-up
    request for `b.example` on the same client reused the connection and returned 200
    after a single TLS handshake, its name never validated against any certificate - the
    pin had introduced a hole of its own.

    The property is asserted here at the TCP level, which needs no certificate machinery
    and therefore no dependency this project does not already declare: two pinned names
    must open two connections, because a connection that is never reused cannot carry a
    name that was never validated on it.
    """
    import socket as socket_mod
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    connections: list[tuple] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"          # keep-alive available unless refused

        def do_GET(self):  # noqa: N802
            body = b"BODY"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        def verify_request(self, request, client_address):
            connections.append(client_address)   # one call per accepted connection
            return True

    server = Server(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def resolver(host, port_, *args, **kwargs):
        return [(socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, "", ("127.0.0.1", port_ or 80))]

    monkeypatch.setattr(socket_mod, "getaddrinfo", resolver)
    # The address policy is not what this test is about: it needs two NAMES that pin to
    # one reachable address, and loopback is the only address a test may connect to.
    monkeypatch.setattr("leadscout.util._is_public", lambda addr: True)

    try:
        with httpx.Client(timeout=5.0) as client:
            safe_get(client, f"http://a.example:{port}/one")
            safe_get(client, f"http://b.example:{port}/two")
    finally:
        server.shutdown()

    assert len(connections) == 2, (
        f"the two names shared {len(connections)} connection(s); a pooled connection "
        f"validated for one name would carry the other")


def test_http2_is_not_in_play_for_the_connection_isolation_claim():
    """The scope of the test above, recorded so it is not read as covering more.

    One connection per name is sufficient isolation only while a connection carries one
    request at a time. HTTP/2 connection coalescing would let one connection serve
    several names by itself, and `Connection: close` is an HTTP/1.1 mechanism that does
    not apply to it. Neither is in play here: nothing in this project passes
    `http2=True`, and httpx cannot negotiate HTTP/2 without the `h2` package, which is
    not a dependency. This test states that as a fact rather than an assumption - if it
    ever fails, the isolation claim above needs rewriting, not patching."""
    import importlib.util
    from pathlib import Path

    assert importlib.util.find_spec("h2") is None, \
        "h2 is installed: httpx can now negotiate HTTP/2 and connection coalescing is back in scope"
    assert "http2" not in (Path(__file__).parents[1] / "requirements.txt").read_text(encoding="utf-8")


def test_a_pinned_request_refuses_keep_alive(monkeypatch):
    """The mechanism behind the test above, pinned separately so it cannot be removed as
    an apparently pointless header."""
    import socket as socket_mod

    resolver, _ = _rebinding_resolver([PUBLIC_ANSWER])
    monkeypatch.setattr(socket_mod, "getaddrinfo", resolver)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Connection", ""))
        return httpx.Response(200, text="ok")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        safe_get(client, f"https://{REBIND_HOST}/page")

    assert seen == ["close"]
