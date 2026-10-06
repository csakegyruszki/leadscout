"""leadscout/util.py: the shared "www."-stripping domain helper (item 2) - used by
research.py to compute the registrable domain it hands to providers/footprint.py
(crt.sh) and providers/trust_pages.py (status-page guesses), both of which produced
wrong results when handed a "www.<domain>" host directly."""
from leadscout.util import registrable_domain


def test_strips_leading_www():
    assert registrable_domain("www.masterplast.hu") == "masterplast.hu"


def test_leaves_bare_domain_unchanged():
    assert registrable_domain("masterplast.hu") == "masterplast.hu"


def test_accepts_a_full_url():
    assert registrable_domain("https://www.masterplast.hu/path?x=1") == "masterplast.hu"


def test_lowercases():
    assert registrable_domain("WWW.Example.COM") == "example.com"


def test_only_strips_a_leading_www_not_an_embedded_one():
    assert registrable_domain("wwwstats.example.com") == "wwwstats.example.com"
