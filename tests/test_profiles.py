"""The neutral default profile, and the profile plumbing around it.

These tests pass an OrgProfile explicitly instead of switching LEADSCOUT_ORG_PROFILE: the
rest of the suite runs under the infra-vendor profile (see conftest.py)."""
from __future__ import annotations

import pytest

from leadscout import compliance, notify, research
from leadscout.cloud import CloudUsageAssessment
from leadscout.fit import score_fit
from leadscout.models import ComplianceResult, FitResult, Lead, LeadOutcome, Research
from leadscout.profile import PROFILES_DIR, ProfileError, load_profile
from leadscout.profile_config import (
    compliance_config,
    fit_config,
    notification_config,
    research_config,
    size_bands,
)


@pytest.fixture(scope="module")
def default():
    return load_profile("default")


@pytest.fixture(scope="module")
def infra_vendor():
    return load_profile("infra-vendor")


def _lead(company="Acme Industries", website="https://acme-industries.example", band=""):
    return Lead(name="Ann", email="ann@acme-industries.example", company=company, website=website,
                company_size_band=band)


def test_default_profile_loads_and_every_section_parses(default):
    assert default.name == "default"
    assert fit_config(default).threshold == 55
    assert compliance_config(default).competitors == ()
    assert research_config(default).tech_signal_example
    assert "{company}" in notification_config(default).subject_template


def test_default_profile_names_no_vendor_and_no_cloud_cost_offering(default):
    text = (PROFILES_DIR / "default.yaml").read_text(encoding="utf-8").lower()
    assert "examplecloud" not in text
    assert "cloud cost" not in text
    system = compliance._build_system(compliance_config(default)).lower()
    assert "examplecloud" not in system and "cloud cost" not in system
    assert "cloud cost" not in research._build_system(research_config(default)).lower()
    assert "in substance" not in system  # the competitor_offering line is omitted when empty


def test_infra_vendor_profile_keeps_its_vendor_wording(infra_vendor):
    assert "cloud cost optimization" in compliance._build_system(compliance_config(infra_vendor))


def test_sizeable_company_without_cloud_evidence_is_not_low_confidence_by_default(default):
    r = Research(headquarters_country="Germany", estimated_employees=600, industry="manufacturing",
                 cloud_usage=CloudUsageAssessment(state="UNKNOWN"))
    fit = score_fit(_lead(), r, profile=default)
    assert fit.confidence == "MEDIUM"
    assert fit.cloud_signal_points == 0
    assert fit.score >= 30  # scale carries the score


def test_same_company_is_low_confidence_under_the_cloud_driven_profile(infra_vendor):
    r = Research(headquarters_country="Germany", estimated_employees=600, industry="manufacturing",
                 cloud_usage=CloudUsageAssessment(state="UNKNOWN"))
    assert score_fit(_lead(), r, profile=infra_vendor).confidence == "LOW"


def test_unknown_headcount_and_hq_stays_low_even_without_the_cloud_requirement(default):
    r = Research(cloud_usage=CloudUsageAssessment(state="UNKNOWN"))
    assert score_fit(_lead(), r, profile=default).confidence == "LOW"


def test_empty_competitor_list_never_blocks_on_competitor_grounds(default):
    # A name and a domain that would trip the infra-vendor list outright.
    lead = _lead(company="CloudTrim Inc", website="https://cloud-trim.com")
    r = Research(headquarters_country="Germany")
    assert not [h for h in compliance.prescreen(lead, r, profile=default) if h["kind"] == "competitor"]
    assert [h for h in compliance.prescreen(lead, r, profile=load_profile("infra-vendor"))
            if h["kind"] == "competitor"]


def test_size_bands_come_from_the_profile(default, infra_vendor):
    assert size_bands(default) == size_bands(infra_vendor) == ("1-10", "11-50", "51-200", "201-1000", "1000+")


def test_band_wins_over_a_contradicting_inferred_headcount_under_any_profile(default):
    r = Research(estimated_employees=5, headquarters_country="Germany")
    fit = score_fit(_lead(band="201-1000"), r, profile=default)
    assert "contradicts the self-reported band" in fit.reasoning


def test_notification_subject_and_identity_come_from_the_profile(tmp_path, monkeypatch, default):
    lead = _lead()
    outcome = LeadOutcome(
        lead=lead, research=Research(headquarters_country="Germany"),
        compliance=ComplianceResult(False, "clear", [], "ok"),
        fit=FitResult(60, "MEDIUM", 25, 10, 5, "r", "LOW", "x"), sales_ready=True)
    subject, _ = notify.render(outcome)
    assert subject == "[Lead] Acme Industries - fit 60/100 (MEDIUM) - compliance CLEAR"


def test_unknown_key_in_a_section_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nfit:\n  thresold: 60\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="thresold"):
        fit_config(load_profile(str(bad)))


def test_unknown_identity_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nidentity:\n  sendr_email: x@y.z\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="sendr_email"):
        load_profile(str(bad))


def test_missing_profile_raises():
    with pytest.raises(ProfileError):
        load_profile("no-such-profile")


def test_missing_policy_file_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ncompliance:\n  competitors_file: nope.yaml\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="competitors_file"):
        compliance_config(load_profile(str(bad)))


def test_scale_tiers_must_end_with_an_open_tier(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nfit:\n  scale:\n    tiers:\n      - {below: 10, points: 1}\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="LAST tier"):
        fit_config(load_profile(str(bad)))


def test_runtime_fingerprint_covers_profiles():
    from leadscout.runtime_identity import runtime_files
    names = {f.name for f in runtime_files()}
    assert {"default.yaml", "infra-vendor.yaml"} <= names


def test_policy_evidence_includes_the_active_profile(tmp_path, monkeypatch):
    from leadscout import provenance
    monkeypatch.setattr(provenance, "LEDGER_DIR", tmp_path / "provenance")
    monkeypatch.setattr(provenance, "LEDGER_PATH", tmp_path / "provenance" / "ledger.jsonl")
    monkeypatch.setattr(provenance, "SNAPSHOT_DIR", tmp_path / "provenance" / "snapshots")
    paths = []
    run = provenance.ProvenanceRun.start("Acme")
    orig = run.record
    run.record = lambda ev, raw, stage="": (paths.append(ev.url), orig(ev, raw, stage=stage))[1]
    provenance.record_policy_files(run)
    assert any(p.endswith("/config/profiles/infra-vendor.yaml") for p in paths)
