"""Real sanctions screening: the pure `classify()` decision function (offline, no
network, tested against captured measured payloads) and `screen_sanctions()`'s disk
cache + graceful-degradation behaviour (isolated to a temp cache dir so these tests
never read or write the repo's committed out/cache/opensanctions/ seed files).
"""
import json

import httpx
import pytest

from leadscout import sanctions


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sanctions, "CACHE_DIR", tmp_path / "cache" / "opensanctions")


class _FakeSettingsWithKey:
    opensanctions_api_key = "test-key"
    http_timeout = 5.0


class _FakeSettingsNoKey:
    opensanctions_api_key = ""
    http_timeout = 5.0


def _patch_client(monkeypatch, handler, *, has_key=True):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sanctions.httpx, "Client", fake_client)
    monkeypatch.setattr(sanctions, "settings", _FakeSettingsWithKey() if has_key else _FakeSettingsNoKey())


def _response(results):
    return httpx.Response(200, json={"responses": {"q": {"results": results}}})


def _recently() -> str:
    """A timestamp inside the cache freshness window, computed rather than written down.
    These two cases are about the cache being USED, not about it ageing; a literal date
    would have quietly turned them into freshness tests that start failing one month
    after the day they were written."""
    from datetime import UTC, datetime, timedelta
    return (datetime.now(UTC) - timedelta(days=1)).isoformat()


def _hit(caption, score, match, schema="Company", topics=None, datasets=None):
    return {"caption": caption, "score": score, "match": match, "schema": schema,
            "datasets": datasets or [], "properties": {"topics": topics or []}}


# --- classify(): the pure decision function, against measured/captured payloads ----

def test_classify_rosneft_is_blocked():
    hits = [
        _hit("Open Joint-Stock Company Rosneft Oil Company", 1.0, True,
             topics=["sanction", "export.risk", "export.control", "debarment", "corp.public", "corp.disqual"]),
        _hit("JSC Rosneft", 0.88, True, topics=[]),
    ]
    assert sanctions.classify(hits, "Rosneft") == "blocked_evidence"


def test_classify_sovcombank_is_blocked():
    hits = [
        _hit("Sovcombank", 1.0, True,
             topics=["sanction", "export.control", "debarment", "fin.bank", "corp.public", "corp.disqual"]),
        _hit("JOINT STOCK COMPANY SOVCOMBANK LIFE", 0.75, True, topics=["debarment", "sanction", "export.control"]),
    ]
    assert sanctions.classify(hits, "Sovcombank") == "blocked_evidence"


def test_classify_rosneft_trading_kft_is_review_not_blocked():
    """Score 0.72 is below the 0.85 floor for blocked_evidence even with match=true."""
    hits = [_hit("ROSNEFT TRADE LIMITED", 0.72, True, topics=["debarment", "sanction", "export.control"])]
    assert sanctions.classify(hits, "Rosneft Trading Kft") == "review_evidence"


def test_classify_masterplast_is_info():
    hits = [
        _hit("MASTERPLAST Nyilvanosan mukodo Reszvenytarsasag", 0.88, True, topics=["corp.public"]),
        _hit("OOO MASTERPLAST", 0.68, True, topics=[]),
    ]
    assert sanctions.classify(hits, "Masterplast") == "info"


def test_classify_snapp_person_hit_is_none():
    hits = [_hit("Steven Allen Snapp", 0.6, True, schema="Person", topics=["reg.action"])]
    assert sanctions.classify(hits, "Snapp") == "none"


def test_classify_zapier_no_hits_is_none():
    assert sanctions.classify([], "Zapier") == "none"


def test_classify_ct_inc_false_positive_is_review_never_blocked():
    """The measured false positive that motivated the weak-name guard: a short,
    generic query name must not be enough to force `blocked` even against a real
    hard-topic, match=true hit on an unrelated company."""
    hits = [_hit("CT TRANSPORTATION COMPANY, INC", 0.8333333333333333, True, topics=["debarment"])]
    assert sanctions.classify(hits, "CT Inc") == "review_evidence"


def test_classify_cloud_solutions_ltd_false_positive_is_review_never_blocked():
    hits = [
        _hit("Cloud Solutions LLC", 0.73, True, topics=["debarment", "sanction"]),
        _hit("CTS Cloud Trading Solutions Ltd (ex Novox Capital Ltd)", 0.6622516556291391, True, topics=["reg.warn"]),
        _hit("CLOUD NINE PLATINUM LTD", 0.625, True, topics=[]),
    ]
    assert sanctions.classify(hits, "Cloud Solutions Ltd") == "review_evidence"


def test_classify_dissimilar_caption_is_review_despite_high_score_and_match():
    """Clearing score/match isn't enough either - the caption has to actually
    resemble the query name (name_sim), or a spurious high-confidence hit on a
    totally different entity would still force blocked."""
    hits = [_hit("Totally Unrelated Global Trading Enterprises", 0.9, True, topics=["sanction"])]
    assert sanctions.classify(hits, "Northwind Freight Partners") == "review_evidence"


def test_is_weak_query_short_and_generic_names():
    assert sanctions._is_weak_query("ct")
    assert sanctions._is_weak_query("cloud solutions")
    assert not sanctions._is_weak_query("rosneft")
    assert not sanctions._is_weak_query("sovcombank")
    assert not sanctions._is_weak_query("rosneft trading")


# --- screen_sanctions(): disk cache + graceful degradation, all offline -----------

def test_cache_miss_calls_api_and_writes_cache(monkeypatch):
    results = [_hit("Sovcombank", 1.0, True, topics=["sanction", "export.control", "debarment"])]
    _patch_client(monkeypatch, lambda req: _response(results))
    result = sanctions.screen_sanctions("Sovcombank", "Russia")
    assert result.status == "blocked_evidence"
    path = sanctions._cache_path("Sovcombank", "ru")
    assert path.exists()
    cached = json.loads(path.read_text(encoding="utf-8"))
    assert cached["source"] == "api"
    assert cached["query"] == {"name": "Sovcombank", "country": "ru"}


def test_cache_hit_is_used_without_calling_the_api(monkeypatch):
    def handler(request):
        raise AssertionError("should not call the API on a cache hit")

    _patch_client(monkeypatch, handler)
    path = sanctions._cache_path("Rosneft", "ru")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "query": {"name": "Rosneft", "country": "ru"},
        "fetched_at": _recently(),
        "source": "manual_probe_2026-09-19",
        "results": [_hit("Open Joint-Stock Company Rosneft Oil Company", 1.0, True,
                          topics=["sanction", "export.control", "debarment"])],
    }), encoding="utf-8")
    result = sanctions.screen_sanctions("Rosneft", "Russia")
    assert result.status == "blocked_evidence"
    assert "cache" in result.note
    assert result.hits[0]["origin"] == "opensanctions:match(cache)"


def test_cache_hit_works_even_without_an_api_key(monkeypatch):
    """The point of the cache: a deployment with no key still sees real, cached data."""
    monkeypatch.setattr(sanctions, "settings", _FakeSettingsNoKey())
    path = sanctions._cache_path("Zapier", "us")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "query": {"name": "Zapier", "country": "us"},
        "fetched_at": _recently(),
        "source": "manual_probe_2026-09-19",
        "results": [],
    }), encoding="utf-8")
    result = sanctions.screen_sanctions("Zapier", "United States")
    assert result.status == "none"
    assert "cache" in result.note


def test_cache_key_is_deterministic_and_country_sensitive():
    a = sanctions._cache_path("Sovcombank", "ru")
    b = sanctions._cache_path("Sovcombank", "ru")
    c = sanctions._cache_path("Sovcombank", None)
    assert a == b
    assert a != c


def test_seed_cache_key_matches_pipeline_lookup_key():
    """The exact scenario this package is about: a seed file written under the key
    the eval/batch pipeline will actually look up for a given (name, hq_country) -
    the ISO2 mapping must be applied identically on both sides."""
    assert sanctions._cache_key("Rosneft Trading Kft", sanctions._iso2("Hungary")) == \
        sanctions._cache_key("Rosneft Trading Kft", "hu")
    assert sanctions._cache_key("Sovcombank", sanctions._iso2("Russia")) == \
        sanctions._cache_key("Sovcombank", "ru")


def test_missing_key_no_cache_is_skipped(monkeypatch):
    monkeypatch.setattr(sanctions, "settings", _FakeSettingsNoKey())
    result = sanctions.screen_sanctions("Some Uncached Company", "")
    assert result.status == "skipped"
    assert result.note


def test_http_error_status_no_cache_is_skipped_not_raised(monkeypatch):
    _patch_client(monkeypatch, lambda req: httpx.Response(500, text="boom"))
    result = sanctions.screen_sanctions("Some Uncached Company", "")
    assert result.status == "skipped"
    assert "500" in result.note


def test_network_error_no_cache_is_skipped_not_raised(monkeypatch):
    def handler(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    _patch_client(monkeypatch, handler)
    result = sanctions.screen_sanctions("Some Uncached Company", "")
    assert result.status == "skipped"
    assert result.note


def test_iso2_lookup_known_and_unknown():
    assert sanctions._iso2("Hungary") == "hu"
    assert sanctions._iso2("russia") == "ru"
    assert sanctions._iso2("Narnia") is None
    assert sanctions._iso2("") is None


def test_every_hit_has_provenance_fields(monkeypatch):
    results = [_hit("Sovcombank", 1.0, True, topics=["sanction", "export.control"],
                     datasets=["us_ofac_sdn", "eu_fsf", "gb_hmt_sanctions", "extra_dataset"])]
    results[0]["id"] = "5"
    _patch_client(monkeypatch, lambda req: _response(results))
    result = sanctions.screen_sanctions("Sovcombank Fresh Query", "Russia")
    hit = result.hits[0]
    assert hit["origin"] == "opensanctions:match"
    assert hit["url"] == "https://www.opensanctions.org/entities/5/"
    assert len(hit["datasets"]) == 3  # capped at 3, per spec


# --- A failed measurement is not a clean screen -----------------------------------
#
# Measured before these existed, every case below except the first returned
# status="none" with hits=[] - the same answer as a real no-match - and was written to
# the disk cache as if it were one; the non-dict and non-JSON bodies instead raised
# straight out of a function whose docstring says it never raises.

UNUSABLE_BODIES = [
    ("empty object", httpx.Response(200, json={})),
    ("json list", httpx.Response(200, json=[])),
    ("not json at all", httpx.Response(200, text="{bad")),
    ("wrong shape", httpx.Response(200, json={"unexpected": 1})),
    ("container without results", httpx.Response(200, json={"responses": {"q": {}}})),
    ("results is not a list", httpx.Response(200, json={"responses": {"q": {"results": "none"}}})),
]


@pytest.mark.parametrize("label,response", UNUSABLE_BODIES, ids=[b[0] for b in UNUSABLE_BODIES])
def test_a_200_with_an_unusable_body_degrades_instead_of_reading_as_no_match(
        label, response, monkeypatch):
    _patch_client(monkeypatch, lambda req: response)
    result = sanctions.screen_sanctions("Unusable Body Co", "Hungary")
    assert result.status == "skipped", f"{label} was reported as a completed screen"
    assert result.hits == []
    assert result.note
    assert not sanctions._cache_path("Unusable Body Co", "hu").exists(), \
        f"{label} was cached as if it were a real result"


def test_a_genuine_empty_result_is_still_a_completed_screen(monkeypatch):
    """The control for the cases above: a well-formed zero-result answer must stay
    negative evidence, or degrading the unusable bodies would have cost nothing."""
    _patch_client(monkeypatch, lambda req: _response([]))
    result = sanctions.screen_sanctions("Clean Co", "Hungary")
    assert result.status == "none"
    assert sanctions._cache_path("Clean Co", "hu").exists()


# --- A cached screen ages ----------------------------------------------------------

def _seed_cache(name, iso2, results, fetched_at):
    path = sanctions._cache_path(name, iso2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "query": {"name": name, "country": iso2}, "fetched_at": fetched_at,
        "source": "seed", "results": results,
    }), encoding="utf-8")
    return path


_EXACT_HIT = [_hit("Rosneft", 1.0, True, topics=["sanction"])]


def test_a_fresh_cached_hit_still_decides(monkeypatch):
    from datetime import UTC, datetime
    _seed_cache("Rosneft", "ru", _EXACT_HIT, datetime.now(UTC).isoformat())
    monkeypatch.setattr(sanctions, "settings", _FakeSettingsNoKey())
    result = sanctions.screen_sanctions("Rosneft", "Russia")
    assert result.status == "blocked_evidence"
    assert result.hits[0]["origin"] == "opensanctions:match(cache)"


@pytest.mark.parametrize("fetched_at", ["2000-01-01T00:00:00Z", "not a date", ""])
def test_a_stale_cached_hit_is_not_a_current_screen(fetched_at, monkeypatch):
    """Designations are added daily, so an old answer is not an answer about now. It
    must become neither a current block nor a silent clear: it degrades, and the note
    still says an old answer exists, because that is not the same as never having asked."""
    _seed_cache("Rosneft", "ru", _EXACT_HIT, fetched_at)
    monkeypatch.setattr(sanctions, "settings", _FakeSettingsNoKey())
    result = sanctions.screen_sanctions("Rosneft", "Russia")
    assert result.status == "skipped"
    assert result.hits == []
    assert "revalidate" in result.note


def test_a_stale_cache_entry_is_refreshed_when_a_key_is_available(monkeypatch):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return _response([])

    _seed_cache("Rosneft", "ru", _EXACT_HIT, "2000-01-01T00:00:00Z")
    _patch_client(monkeypatch, handler)
    result = sanctions.screen_sanctions("Rosneft", "Russia")
    assert calls, "the stale entry was served instead of revalidated"
    assert result.status == "none"


def test_cached_evidence_is_dated_when_it_was_fetched_not_when_it_was_read(monkeypatch):
    """`observed_at` is an acquisition time. Stamping it with now() made a cached answer
    read, in the evidence trail, as a fresh observation."""
    from datetime import UTC, datetime, timedelta

    class _SpyRun:
        """A stand-in rather than a real ProvenanceRun: starting one writes to the
        repository's committed out/provenance/ledger.jsonl, and a test must not."""
        run_id = "test-run"

        def __init__(self):
            self.recorded = []

        def next_evidence_id(self):
            return f"ev-{len(self.recorded) + 1:03d}"

        def record(self, evidence, raw_bytes, *, stage=""):
            self.recorded.append(evidence)

    fetched_at = (datetime.now(UTC) - timedelta(days=3)).isoformat()
    _seed_cache("Rosneft", "ru", _EXACT_HIT, fetched_at)
    monkeypatch.setattr(sanctions, "settings", _FakeSettingsNoKey())
    run = _SpyRun()
    sanctions.screen_sanctions("Rosneft", "Russia", run)
    observed = [e.observed_at for e in run.recorded if e.source_type == "opensanctions"]
    assert observed == [fetched_at], observed


def test_skipped_says_whether_the_control_failed_or_was_never_configured(monkeypatch):
    """`skipped` alone collapsed "we tried and could not tell" into "we never asked".
    Compliance treats those differently, so the screen has to distinguish them."""
    from datetime import UTC, datetime

    monkeypatch.setattr(sanctions, "settings", _FakeSettingsNoKey())
    never_asked = sanctions.screen_sanctions("Nobody Has Ever Queried This", "Hungary")
    assert (never_asked.status, never_asked.reason) == ("skipped", "not_configured")

    _seed_cache("Rosneft", "ru", _EXACT_HIT, "2000-01-01T00:00:00Z")
    stale = sanctions.screen_sanctions("Rosneft", "Russia")
    assert (stale.status, stale.reason) == ("skipped", "degraded")

    _patch_client(monkeypatch, lambda req: httpx.Response(503, text="down"))
    outage = sanctions.screen_sanctions("Outage Co", "Hungary")
    assert (outage.status, outage.reason) == ("skipped", "degraded")

    _patch_client(monkeypatch, lambda req: httpx.Response(200, text="{bad"))
    unreadable = sanctions.screen_sanctions("Unreadable Co", "Hungary")
    assert (unreadable.status, unreadable.reason) == ("skipped", "degraded")

    _seed_cache("Fresh Co", "hu", [], datetime.now(UTC).isoformat())
    ran = sanctions.screen_sanctions("Fresh Co", "Hungary")
    assert (ran.status, ran.reason) == ("none", ""), "a screen that ran must carry no failure reason"
