"""First-party research may cross sibling subdomains of the SAME registrable domain.

Lidl (2026-09-20): the lead submitted `lidl.hu`. Its own landing page nominates
`vallalat.lidl.hu` - the corporate site, which states the Hungarian headcount in the
first person - and `jobs.lidl.hu`, alongside `konyha.lidl.hu` (recipes) and
`ugyfelszolgalat.lidl.hu` (customer service). A same-origin-only crawl reached none of
them, and the run reported "no company size evident from the given sources" about a
company that publishes it. That is a research failure, not a missing enrichment key.

Two gates, both required: identity (same registrable domain, nominated by a
first-party page we fetched) and purpose (the label looks like corporate content).
"""
from leadscout.providers.website import (
    MAX_PAGES,
    MAX_SIBLING_HOSTS,
    _first_party_sibling_roots,
    _registrable,
    select_subpages,
)

_LIDL_LINKS = """
<a href="https://konyha.lidl.hu/">Konyha</a>
<a href="https://vallalat.lidl.hu">Vallalat</a>
<a href="https://jobs.lidl.hu">Karrier</a>
<a href="https://ugyfelszolgalat.lidl.hu/SelfServiceHU/s/">Ugyfelszolgalat</a>
<a href="https://cdn.lidl.hu/img/a.png">asset</a>
<a href="https://careers.competitor.example/jobs">someone else</a>
"""


def test_only_purpose_relevant_siblings_are_followed():
    roots = _first_party_sibling_roots(_LIDL_LINKS, "https://www.lidl.hu/")
    assert roots == ["https://vallalat.lidl.hu/", "https://jobs.lidl.hu/"]


def test_the_identity_gate_is_the_registrable_domain_not_the_host():
    """util.registrable_domain only strips "www.", so it would read a sibling as a
    different domain entirely - hence this module's own public-suffix-aware compare."""
    assert _registrable("vallalat.lidl.hu") == _registrable("www.lidl.hu") == "lidl.hu"
    assert _registrable("careers.bbc.co.uk") == "bbc.co.uk"   # two-level suffix
    assert _registrable("evil-lidl.hu") != "lidl.hu"


def test_a_foreign_domain_is_never_followed_however_relevant_it_looks():
    roots = _first_party_sibling_roots(
        '<a href="https://careers.competitor.example/jobs">Careers</a>', "https://www.lidl.hu/")
    assert roots == []


def test_sibling_hosts_are_capped():
    html = "".join(f'<a href="https://careers{i}.example.com/">c</a>' for i in range(6))
    assert len(_first_party_sibling_roots(html, "https://example.com/")) <= MAX_SIBLING_HOSTS


def test_siblings_never_outrank_the_sites_own_pages_or_widen_the_budget():
    """This changes WHERE the crawl may look, not HOW MUCH it may fetch: a site whose
    own tree answers the question keeps exactly its previous candidate list."""
    own_pages = "".join(
        f'<a href="https://www.lidl.hu/{p}">x</a>' for p in ("about", "company", "careers", "security", "legal"))
    picked = select_subpages(own_pages + _LIDL_LINKS, "https://www.lidl.hu/")
    assert len(picked) <= MAX_PAGES
    assert all(u.startswith("https://www.lidl.hu/") for u in picked), picked


def test_multi_tenant_platform_hosts_are_not_one_registrable_domain():
    """The identity gate is only an identity gate if the public suffix includes the
    PSL's PRIVATE section. tldextract defaults to ICANN-only, under which every tenant
    of a shared platform collapses to one "registrable domain" - measured before the
    fix: `companya.github.io` and `careers.github.io` both read as `github.io`, so an
    unrelated tenant's site passed the gate as first-party to the lead."""
    for a, b in (("companya.github.io", "careers.github.io"),
                 ("acme.herokuapp.com", "corporate.herokuapp.com"),
                 ("startup.vercel.app", "jobs.vercel.app"),
                 ("bucket-a.s3.amazonaws.com", "company-b.s3.amazonaws.com")):
        assert _registrable(a) != _registrable(b), f"{a} and {b} share a registrable domain"
    # ICANN-suffix answers are unchanged by reading the private section too.
    assert _registrable("vallalat.lidl.hu") == _registrable("www.lidl.hu") == "lidl.hu"
    assert _registrable("careers.bbc.co.uk") == "bbc.co.uk"


def test_an_unrelated_tenant_on_the_same_platform_is_not_a_first_party_sibling():
    html = ('<a href="https://careers.github.io/">Careers</a>'
            '<a href="https://jobs.github.io/">Jobs</a>')
    assert _first_party_sibling_roots(html, "https://companya.github.io/") == []
