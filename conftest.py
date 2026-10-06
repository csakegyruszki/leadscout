# Present so that `pytest` finds the `leadscout` package.
#
# pytest puts a test file's first non-package parent directory on sys.path, which for
# `tests/test_*.py` (no `__init__.py`) is `tests/`, not the repository root. The package
# is not installed either - `requirements.txt` lists leadscout's dependencies, not
# leadscout itself - so `import leadscout` resolved only when the interpreter had already
# put the working directory on sys.path, which `python -m pytest` does and a bare
# `pytest` does not. `make test` used the first form and passed; CI used the second and
# failed every run with 53 collection errors.
#
# A conftest.py in the rootdir is what pytest's own rootdir/sys.path rules key off, so
# this file is the fix rather than a pytest.ini setting or a sys.path hack in each test.
from __future__ import annotations

import os

# The legacy suite encodes the infra-vendor ICP (cloud-evidence-driven fit, the fictional
# competitor list, its vendor wording). Select that profile BEFORE any leadscout
# import - several modules read settings or build prompts at import time. Tests of other
# profiles pass an OrgProfile explicitly (tests/test_profiles.py).
os.environ.setdefault("LEADSCOUT_ORG_PROFILE", "infra-vendor")

import socket  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_live_dns(monkeypatch, request):
    """Name resolution fails in the suite unless a test sets up its own resolver.

    The suite mocks the HTTP transport but used to leave DNS real, so what a test did
    depended on whether the machine running it had a working resolver. That was harmless
    while nothing read the resolved address - and stopped being harmless when `safe_get`
    began connecting to the address it validated: with a resolver present, a test naming
    a domain that really exists saw the pinned address at the transport instead of the
    URL it matched on, so `test_vendor.py::test_redirect_off_allowlist_is_ignored`
    passed offline and failed online. One suite, two answers, depending on the network -
    which is the property a test suite must not have.

    Tests that need resolution patch `socket.getaddrinfo` themselves; that patch is
    applied after this one and wins. The `live_dns` marker opts a test out entirely.

    The one name that still resolves is `localhost`, because it is a name rather than a
    literal and refusing to resolve it would quietly turn `https://localhost/` from a
    refused target into an allowed one - the guard lets an unresolvable name through on
    purpose. A stub that answered nothing at all would have made that security test pass
    for the wrong reason.
    """
    if request.node.get_closest_marker("live_dns"):
        return

    real = socket.getaddrinfo

    def stub(host, port, *args, **kwargs):
        name = str(host).rstrip(".").lower()
        if name == "localhost" or name.endswith(".localhost"):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port or 0)),
                    (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", port or 0, 0, 0))]
        try:                       # IP literals still have to work: httpx resolves those too
            socket.inet_pton(socket.AF_INET, name)
        except OSError:
            pass
        else:
            return real(name, port, *args, **kwargs)
        raise socket.gaierror(f"name resolution is disabled in tests: {host!r}")

    monkeypatch.setattr(socket, "getaddrinfo", stub)


def pytest_configure(config):
    config.addinivalue_line("markers", "live_dns: let this test use the machine's real resolver")
