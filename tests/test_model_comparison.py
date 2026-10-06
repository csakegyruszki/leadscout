"""Per-model regression harness - offline: ask_model/screen_sanctions
monkeypatched, no real LLM/network calls. Verifies the single-model-chain override
(dataclasses.replace on llm.settings, restored after), the metrics computed per
model (schema-valid rate, strict/flagged-vs-clear accuracy, false-clear count,
median latency, cost), that one erroring case doesn't sink the model's whole run,
and the paid-DeepSeek baseline reader.
"""
import json

from leadscout import compliance, evals, llm
from leadscout.models import ComplianceVerdict
from leadscout.sanctions import SanctionsScreen

_CASES = [
    {"company": "CloudTrim, Inc.", "website": "https://cloudtrim.com", "hq_country": "United States",
     "expected": "blocked", "why": "exact competitor"},
    {"company": "Acme Retail", "website": "https://acme-retail.example", "hq_country": "Canada",
     "expected": "clear", "why": "no match"},
]


def _write_cases(tmp_path, cases=_CASES):
    path = tmp_path / "compliance_cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def test_run_model_comparison_uses_a_single_model_chain_and_restores_settings(monkeypatch, tmp_path):
    seen_models = []

    def fake_ask_model(system, user, schema, purpose=""):
        seen_models.append(tuple(llm.settings.models))
        llm.telemetry.append({"purpose": purpose, "model": llm.settings.models[0], "provider": "openrouter",
                               "hops": 1, "timeout_hops": 0, "prompt_tokens": 10, "completion_tokens": 5,
                               "cost_usd": 0.0001, "estimated": False, "source": "test", "latency_ms": 250})
        return ComplianceVerdict(status="blocked" if "Lead company: CloudTrim" in user else "clear", flagged=True)

    monkeypatch.setattr(compliance, "ask_model", fake_ask_model)
    monkeypatch.setattr(compliance, "screen_sanctions",
                         lambda company, hq_country, run=None: SanctionsScreen(status="none"))
    original_models = llm.settings.models

    path = _write_cases(tmp_path)
    result = evals.run_model_comparison(path, ["openrouter:fake-model-a", "cloudflare:fake-model-b"])

    assert llm.settings.models == original_models  # restored
    assert seen_models == [("openrouter:fake-model-a",), ("openrouter:fake-model-a",),
                            ("cloudflare:fake-model-b",), ("cloudflare:fake-model-b",)]
    assert len(result["models"]) == 2
    m0 = result["models"][0]
    assert m0["model"] == "openrouter:fake-model-a"
    assert m0["n_cases"] == 2
    assert m0["strict_accuracy"] == 1.0
    assert m0["schema_valid_rate"] == 1.0
    assert m0["false_clear_count"] == 0
    assert m0["median_latency_ms"] == 250


def test_run_model_comparison_one_erroring_case_does_not_sink_the_run(monkeypatch, tmp_path):
    def flaky_ask_model(system, user, schema, purpose=""):
        if "Lead company: CloudTrim" in user:
            raise RuntimeError("simulated model failure")
        llm.telemetry.append({"purpose": purpose, "model": "x", "provider": "openrouter", "hops": 1,
                               "timeout_hops": 0, "prompt_tokens": 1, "completion_tokens": 1,
                               "cost_usd": 0.0, "estimated": False, "source": "test", "latency_ms": 100})
        return ComplianceVerdict(status="clear", flagged=False)

    monkeypatch.setattr(compliance, "ask_model", flaky_ask_model)
    monkeypatch.setattr(compliance, "screen_sanctions",
                         lambda company, hq_country, run=None: SanctionsScreen(status="none"))
    path = _write_cases(tmp_path)
    result = evals.run_model_comparison(path, ["openrouter:flaky"])
    m = result["models"][0]
    assert m["n_cases"] == 2
    assert m["errors"] == 1
    assert any(r["got"] == "error" for r in m["rows"])


def test_false_clear_count_flags_expected_non_clear_got_clear(monkeypatch, tmp_path):
    """A case with no deterministic prescreen/sanctions signal at all (so none of
    screen()'s three safety nets fire) - the only way a false clear can reach the
    final output is the LLM itself getting it wrong, which is exactly what this
    metric is meant to catch."""
    monkeypatch.setattr(compliance, "ask_model",
                         lambda system, user, schema, purpose="": ComplianceVerdict(status="clear", flagged=False))
    monkeypatch.setattr(compliance, "screen_sanctions",
                         lambda company, hq_country, run=None: SanctionsScreen(status="none"))
    cases = [{"company": "Globex Analytics", "website": "https://globex-analytics.example",
              "hq_country": "Germany", "expected": "review", "why": "synthetic, no prescreen signal"}]
    path = _write_cases(tmp_path, cases)
    result = evals.run_model_comparison(path, ["openrouter:always-clear"])
    m = result["models"][0]
    assert m["false_clear_count"] == 1


def test_load_paid_deepseek_baseline_reads_existing_artifact(tmp_path):
    artifact = {
        "n_cases": 23, "strict_accuracy": 0.826, "flagged_vs_clear_accuracy": 0.913,
        "total_cost_usd": 0.0, "total_latency_ms": 295914,
        "confusion_matrix": {"clear": {"clear": 6, "review": 1, "blocked": 0},
                              "review": {"clear": 1, "review": 2, "blocked": 2},
                              "blocked": {"clear": 0, "review": 0, "blocked": 11}},
    }
    path = tmp_path / "compliance_results.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    baseline = evals.load_paid_deepseek_baseline(path)
    assert baseline["n_cases"] == 23
    assert baseline["false_clear_count"] == 1
    assert baseline["median_latency_ms"] == round(295914 / 23)


def test_load_paid_deepseek_baseline_missing_file_returns_none(tmp_path):
    assert evals.load_paid_deepseek_baseline(tmp_path / "does_not_exist.json") is None


def test_render_model_comparison_includes_baseline_row():
    result = {"models": [{
        "model": "openrouter:fake", "n_cases": 2, "schema_valid_rate": 1.0, "strict_accuracy": 0.5,
        "flagged_vs_clear_accuracy": 1.0, "false_clear_count": 0, "median_latency_ms": 100,
        "cost_usd": 0.0, "errors": 0, "rows": [],
    }]}
    baseline = {
        "model": "openrouter:deepseek/deepseek-chat-v3.1 (paid, existing baseline)", "n_cases": 23,
        "schema_valid_rate": None, "strict_accuracy": 0.826, "flagged_vs_clear_accuracy": 0.913,
        "false_clear_count": 1, "median_latency_ms": 12866, "cost_usd": 0.0, "errors": None, "rows": None,
    }
    report = evals.render_model_comparison(result, baseline)
    assert "openrouter:fake" in report
    assert "existing baseline" in report
    assert "PRIMARY SAFETY METRIC" in report
