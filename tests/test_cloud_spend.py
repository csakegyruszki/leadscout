"""Spend prior: rough employees x industry-intensity heuristic, surfaced as a
relative tier (LOW/MEDIUM/HIGH/VERY_HIGH) rather than a dollar figure (item 5) - a
prioritisation signal, not an estimated cloud invoice.

Does not touch fit-score maths - checked here by asserting score_fit's tier is
unchanged from the existing fit tests while cloud-spend fields are also populated.
"""
from leadscout.fit import estimate_cloud_spend, score_fit
from leadscout.models import Lead, Research


def test_infra_heavy_mid_size_lands_in_medium_tier():
    r = Research(industry="SaaS", estimated_employees=500)
    tier, reasoning = estimate_cloud_spend(r)
    assert tier == "MEDIUM"
    assert "500" in reasoning


def test_low_infra_small_company_is_low_tier():
    r = Research(industry="bakery", estimated_employees=20)
    tier, _ = estimate_cloud_spend(r)
    assert tier == "LOW"


def test_neutral_industry_very_large_company_is_very_high_tier():
    r = Research(industry="manufacturing", estimated_employees=100_000)
    tier, _ = estimate_cloud_spend(r)
    assert tier == "VERY_HIGH"


def test_unknown_employees_defaults_to_lowest_tier_and_says_why():
    r = Research(industry="SaaS", estimated_employees=None)
    tier, reasoning = estimate_cloud_spend(r)
    assert tier == "LOW"
    assert "no employee estimate" in reasoning


def test_reasoning_never_states_a_dollar_figure_as_the_result():
    """The relative tier must not be dressed back up as an estimated invoice -
    only the internal (never user-facing) heuristic wording may still mention $."""
    r = Research(industry="SaaS", estimated_employees=500)
    tier, _ = estimate_cloud_spend(r)
    assert tier in {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}


def test_score_fit_populates_cloud_spend_independent_of_fit_score():
    r = Research(industry="SaaS", estimated_employees=800, confidence="high", tech_signals=["aws"])
    f = score_fit(Lead("a", "a@b.c", "X", "https://x.com"), r)
    assert 0 <= f.score <= 100
    assert f.cloud_spend_band in {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}
    assert f.cloud_spend_reasoning
