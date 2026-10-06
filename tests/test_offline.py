"""Tests that need no network and no API key: fuzzy pre-screen, fit maths, tracker, email."""
from pathlib import Path

from leadscout.compliance import normalise_name, prescreen
from leadscout.fit import score_fit
from leadscout.models import CloudUsageAssessment, ComplianceResult, FitResult, Lead, LeadOutcome, Research
from leadscout.notify import render
from leadscout.tracker import write_row


def test_normalise_strips_suffix_and_punctuation():
    assert normalise_name("Cloud-Trim Ltd.") == "cloud trim"
    assert normalise_name("CloudTrim Inc") == "cloudtrim"


def test_prescreen_catches_respelled_competitor():
    hits = prescreen(Lead("a", "a@b.c", "Cloud-Trim Ltd.", "https://cloud-trim.io"), Research())
    assert any(h["kind"] == "competitor" and h["term"] == "CloudTrim Inc" for h in hits)


def test_prescreen_generic_cloud_word_alone_is_weak():
    hits = prescreen(Lead("a", "a@b.c", "Blue Cloud Bakery", "https://example.com"), Research())
    assert all(h["score"] < 95 for h in hits)


def test_prescreen_sanctions_from_hq_and_tld():
    r = Research(headquarters_country="Iran")
    hits = prescreen(Lead("a", "a@b.c", "Snapp", "https://snapp.ir"), r)
    kinds = {(h["kind"], h["term"], h["score"]) for h in hits}
    assert ("sanctions", "Iran", 100) in kinds


def test_prescreen_low_confidence_hq_skips_the_full_confidence_sanctions_hit():
    """Item 4: a ccTLD-derived (hq_confidence="low") HQ must not ALSO drive the
    full-confidence (score 100) prescreen:hq sanctions hit - that would double-count
    the exact same weak website-TLD signal the prescreen:tld check (score 80,
    below) already covers independently. The TLD-based hit still fires."""
    r = Research(headquarters_country="Iran", hq_source="website_tld", hq_confidence="low")
    hits = prescreen(Lead("a", "a@b.c", "Snapp", "https://snapp.ir"), r)
    origins = {h["origin"] for h in hits}
    assert "prescreen:hq" not in origins
    assert "prescreen:tld" in origins


def _offline_ev(eid: str):
    """Evidence the tech_signal citation resolves to (REVIEW-A §8)."""
    from leadscout.models import Evidence
    return Evidence(id=eid, source_type="website", url="https://x.com", observed_at="2026-09-19T00:00:00Z",
                    content_sha256="x", strength="STRONG", snippet="s", snapshot_path="",
                    family="first_party_statement")


def test_fit_score_uses_evidence_backed_signals():
    # cloud_usage=CONFIRMED (+40, the evidence-backed assessment, not a
    # keyword match) plus a workload-intensity bonus citing its own evidence id.
    r = Research(industry="SaaS", estimated_employees=800, confidence="low",
                 cloud_usage=CloudUsageAssessment(state="CONFIRMED"),
                 tech_signals=["kubernetes (ev-001)"], headquarters_country="US",
                 evidence=[_offline_ev("ev-001")])
    f = score_fit(Lead("a", "a@b.c", "X", "https://x.com"), r)
    assert f.score > 0
    assert f.confidence in {"HIGH", "MEDIUM", "LOW"}
    assert "ev-001" in f.reasoning


def test_fit_confidence_high_needs_hq_employees_and_two_signals():
    r = Research(industry="SaaS", estimated_employees=800, headquarters_country="US",
                 cloud_usage=CloudUsageAssessment(state="CONFIRMED"),
                 tech_signals=["Kubernetes (ev-002)"], evidence=[_offline_ev("ev-002")])
    f = score_fit(Lead("a", "a@b.c", "X", "https://x.com"), r)
    assert f.confidence == "HIGH"


def test_fit_confidence_low_with_nothing_known():
    r = Research()
    f = score_fit(Lead("a", "a@b.c", "X", "https://x.com"), r)
    assert f.confidence == "LOW"


def test_tracker_and_email(tmp_path: Path):
    lead = Lead("Ann", "ann@x.com", "X Corp", "https://x.com")
    out = LeadOutcome(lead, Research(summary="s", industry="SaaS", headquarters_country="US", sources=["https://x.com"]),
                      ComplianceResult(False, "clear", [], "ok"), FitResult(70, "HIGH", 25, 35, 10, "r"), True)
    p = tmp_path / "t.xlsx"
    assert write_row(out, p) == 2
    assert write_row(out, p) == 2  # same company updates, doesn't append
    subject, body = render(out)
    assert "X Corp" in subject and "ACCOUNT: SALES-READY yes" in body


def test_email_has_exactly_one_pipeline_summary_line():
    lead = Lead("Ann", "ann@x.com", "X Corp", "https://x.com")
    out = LeadOutcome(lead, Research(summary="s", sources=["https://x.com"]),
                      ComplianceResult(False, "clear", [], "ok"), FitResult(70, "HIGH", 25, 35, 10, "r"), True,
                      llm_calls=2, llm_cost_usd=0.0001, llm_latency_ms=500, llm_models="model-a")
    _, body = render(out)
    pipeline_lines = [ln for ln in body.splitlines() if ln.startswith("Pipeline:")]
    assert len(pipeline_lines) == 1
    assert "model-a" in pipeline_lines[0]
