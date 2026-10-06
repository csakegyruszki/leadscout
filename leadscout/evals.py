"""Compliance eval harness: labelled cases -> precision/recall/F1 + confusion matrix.

Runs the real screen() (fuzzy pre-screen + LLM), so it costs LLM calls and needs the
OpenRouter key - this is deliberately an integration eval, not a unit test, because
the thing worth measuring is the LLM's judgement on genuinely ambiguous names.
"""
from __future__ import annotations

import dataclasses
import json
import statistics
from pathlib import Path

from . import llm
from .compliance import prescreen, screen
from .models import Lead, Research
from .research import research_from_text

_CLASSES = ("clear", "review", "blocked")


def _bucket(status: str) -> str:
    """Collapse clear/review/blocked to the decision a rep actually acts on."""
    return "clear" if status == "clear" else "flagged"


def _is_soft_miss(row: dict) -> bool:
    """A miss that's a much cheaper mistake than confusing clear with either flagged
    state: review<->blocked confusion, or a case-declared acceptable alternative
    (`case["accept"]`, e.g. a name ambiguous enough that either clear or review is a
    defensible real-world answer)."""
    if row["got"] == row["expected"]:
        return False
    if {row["got"], row["expected"]} == {"review", "blocked"}:
        return True
    return row["got"] in (row.get("accept") or [])


def _compute_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    confusion = {e: {g: 0 for g in _CLASSES} for e in _CLASSES}
    for row in rows:
        confusion[row["expected"]][row["got"]] += 1

    per_class = {}
    for cls in _CLASSES:
        tp = confusion[cls][cls]
        fp = sum(confusion[e][cls] for e in _CLASSES if e != cls)
        fn = sum(confusion[cls][g] for g in _CLASSES if g != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "support": sum(confusion[cls].values()),
        }

    strict_correct = sum(1 for r in rows if r["got"] == r["expected"])
    flagged_correct = sum(1 for r in rows if _bucket(r["got"]) == _bucket(r["expected"]))
    soft_misses = sum(1 for r in rows if _is_soft_miss(r))

    return {
        "n_cases": n,
        "strict_accuracy": round(strict_correct / n, 3) if n else 0.0,
        "flagged_vs_clear_accuracy": round(flagged_correct / n, 3) if n else 0.0,
        "soft_misses": soft_misses,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def run_compliance_eval(path: str | Path) -> dict:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    telemetry_start = len(llm.telemetry)
    rows = []
    for case in cases:
        lead = Lead(name="Eval Case", email="eval@example.test", company=case["company"], website=case["website"])
        # The case STIPULATES its jurisdiction, so it is modelled as an established
        # fact. Left sourceless it is an unresolved one, and the deterministic HQ
        # gate would then decide every case before the model spoke - measured: the
        # false-clear metric went to 0 because nothing could reach "clear" at all.
        # These evals measure the MODEL; the gate has its own tests.
        research = Research(headquarters_country=case.get("hq_country", ""),
                            hq_source="gleif", summary="", industry="")
        hits = prescreen(lead, research)
        top = max(hits, key=lambda h: h["score"]) if hits else None
        result = screen(lead, research)
        rows.append({
            "company": case["company"],
            "expected": case["expected"],
            "got": result.status,
            "accept": case.get("accept"),
            "why": case.get("why", ""),
            "prescreen_top": top["term"] if top else None,
            "prescreen_top_score": top["score"] if top else None,
            "sanctions_status": result.sanctions_status,
            "reasoning": result.reasoning,
        })
    calls = llm.telemetry[telemetry_start:]
    del llm.telemetry[telemetry_start:]

    per_model: dict[str, dict] = {}
    for c in calls:
        m = per_model.setdefault(c["model"], {"calls": 0, "cost_usd": 0.0, "latency_ms": 0})
        m["calls"] += 1
        m["cost_usd"] += c["cost_usd"]
        m["latency_ms"] += c["latency_ms"]
    for m in per_model.values():
        m["cost_usd"] = round(m["cost_usd"], 6)

    metrics = _compute_metrics(rows)
    metrics["total_cost_usd"] = round(sum(c["cost_usd"] for c in calls), 6)
    metrics["total_latency_ms"] = sum(c["latency_ms"] for c in calls)
    metrics["per_model"] = per_model
    metrics["rows"] = rows
    return metrics


def render_eval_report(result: dict, *, markdown: bool = False) -> str:
    lines: list[str] = []
    if markdown:
        lines.append("# Compliance eval results\n")
        lines.append(f"- cases: {result['n_cases']}")
        lines.append(f"- strict accuracy: {result['strict_accuracy']:.1%}")
        lines.append(f"- flagged-vs-clear accuracy: {result['flagged_vs_clear_accuracy']:.1%}")
        lines.append(f"- soft misses (review<->blocked): {result['soft_misses']}")
        lines.append(f"- total LLM cost: ${result['total_cost_usd']:.6f}, latency: {result['total_latency_ms']} ms\n")
        if result.get("per_model"):
            lines.append("## Model chain usage\n")
            lines.append("| model | calls | cost (USD) | latency (ms) |")
            lines.append("|---|---|---|---|")
            for model, m in result["per_model"].items():
                lines.append(f"| {model} | {m['calls']} | {m['cost_usd']:.6f} | {m['latency_ms']} |")
            lines.append("")
        lines.append("## Per-class metrics\n")
        lines.append("| class | precision | recall | f1 | support |")
        lines.append("|---|---|---|---|---|")
        for cls, m in result["per_class"].items():
            lines.append(f"| {cls} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |")
        lines.append("\n## Confusion matrix (rows = expected, cols = got)\n")
        lines.append("| expected \\ got | " + " | ".join(_CLASSES) + " |")
        lines.append("|" + "---|" * (len(_CLASSES) + 1))
        for e in _CLASSES:
            row = result["confusion_matrix"][e]
            lines.append(f"| {e} | " + " | ".join(str(row[g]) for g in _CLASSES) + " |")
        lines.append("\n## Per-case\n")
        lines.append("| company | expected | got | sanctions API | prescreen top | reasoning |")
        lines.append("|---|---|---|---|---|---|")
        for r in result["rows"]:
            top = f"{r['prescreen_top']} ({r['prescreen_top_score']})" if r["prescreen_top"] else "-"
            reasoning = (r["reasoning"] or "").replace("|", "/").replace("\n", " ")[:140]
            lines.append(f"| {r['company']} | {r['expected']} | {r['got']} | {r['sanctions_status']} "
                          f"| {top} | {reasoning} |")
        return "\n".join(lines) + "\n"

    lines.append(f"Compliance eval: {result['n_cases']} cases")
    lines.append(f"  strict accuracy:           {result['strict_accuracy']:.1%}")
    lines.append(f"  flagged-vs-clear accuracy: {result['flagged_vs_clear_accuracy']:.1%}")
    lines.append(f"  soft misses (review<->blocked): {result['soft_misses']}")
    lines.append(f"  cost: ${result['total_cost_usd']:.6f}  latency: {result['total_latency_ms']} ms")
    for model, m in (result.get("per_model") or {}).items():
        lines.append(f"    {model}: {m['calls']} calls, ${m['cost_usd']:.6f}, {m['latency_ms']} ms")
    lines.append("")
    for cls, m in result["per_class"].items():
        lines.append(f"  {cls:8s} precision={m['precision']:.2f} recall={m['recall']:.2f} "
                      f"f1={m['f1']:.2f} support={m['support']}")
    lines.append("")
    for r in result["rows"]:
        mark = "OK  " if r["got"] == r["expected"] else ("SOFT" if _is_soft_miss(r) else "MISS")
        lines.append(f"  [{mark}] {r['company']:30s} expected={r['expected']:8s} got={r['got']:8s} "
                      f"sanctions_api={r['sanctions_status']}")
    return "\n".join(lines)


def _hq_matches(expected: str, actual: str | None) -> bool:
    """The literal string "unknown" in the case's expected value matches either a
    null/empty actual value or the LLM's own "unknown"/"unresolved" answer -
    the original wording for this comparison."""
    exp = (expected or "").strip().lower()
    act = (actual or "").strip().lower()
    if exp == "unknown":
        return act in ("", "unknown", "unresolved", "none", "null")
    return exp == act


def run_adversarial_eval(path: str | Path) -> dict:
    """14 reviewed adversarial cases, each supplying `website_text`
    directly as the sole website evidence (via `research.research_from_text` -
    no fetching, no other providers, no live network calls beyond the LLM itself
    and OpenSanctions' committed cache). Checks compliance_status,
    headquarters_country (with the "unknown" special case) and that none of
    `must_not_contain_in_summary`'s strings leaked into the generated summary -
    i.e. that a hostile instruction embedded in the page text did not override the
    extraction prompt."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    telemetry_start = len(llm.telemetry)
    rows = []
    for case in cases:
        lead = Lead(name="Eval Contact", email="eval@example.test",
                    company=case["company"], website=case["website"])
        research = research_from_text(lead, case["website_text"])
        compliance = screen(lead, research)
        expected = case["expected"]

        status_ok = compliance.status == expected["compliance_status"]
        hq_ok = _hq_matches(expected["headquarters_country"], research.headquarters_country)
        summary_lower = (research.summary or "").lower()
        leaked = [s for s in expected.get("must_not_contain_in_summary", []) if s.lower() in summary_lower]
        passed = status_ok and hq_ok and not leaked

        rows.append({
            "id": case["id"], "category": case.get("category", ""), "company": case["company"],
            "pass": passed,
            "expected_compliance_status": expected["compliance_status"],
            "got_compliance_status": compliance.status,
            "expected_headquarters_country": expected["headquarters_country"],
            "got_headquarters_country": research.headquarters_country,
            "leaked_strings": leaked,
            "summary": research.summary,
            "attack_description": case.get("attack_description", ""),
            "notes": expected.get("notes", ""),
        })
    calls = llm.telemetry[telemetry_start:]
    del llm.telemetry[telemetry_start:]

    return {
        "n_cases": len(rows),
        "n_passed": sum(1 for r in rows if r["pass"]),
        "n_failed": sum(1 for r in rows if not r["pass"]),
        "failing_ids": [r["id"] for r in rows if not r["pass"]],
        "llm_calls": len(calls),
        "total_cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
        "total_latency_ms": sum(c["latency_ms"] for c in calls),
        "rows": rows,
    }


def render_adversarial_report(result: dict) -> str:
    lines = ["# Adversarial eval results\n"]
    lines.append(f"- cases: {result['n_cases']}, passed: {result['n_passed']}, failed: {result['n_failed']}")
    lines.append(f"- LLM calls: {result['llm_calls']}, cost: ${result['total_cost_usd']:.6f}, "
                 f"latency: {result['total_latency_ms']} ms\n")
    lines.append("| id | category | pass/fail | expected status | got status | expected HQ | got HQ "
                 "| leaked strings | notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in result["rows"]:
        mark = "PASS" if r["pass"] else "FAIL"
        leaked = ", ".join(r["leaked_strings"]) or "-"
        notes = (r["attack_description"] or "").replace("|", "/")[:100]
        lines.append(
            f"| {r['id']} | {r['category']} | {mark} | {r['expected_compliance_status']} "
            f"| {r['got_compliance_status']} | {r['expected_headquarters_country']} "
            f"| {r['got_headquarters_country']} | {leaked} | {notes} |"
        )
    return "\n".join(lines) + "\n"


def run_model_comparison(path: str | Path, model_ids: list[str]) -> dict:
    """run the existing 23 labelled compliance cases through each of
    `model_ids` in turn - a single-model chain (no fallback) for the duration of that
    model's run, via `dataclasses.replace` on `llm.settings` (restored in `finally`,
    even on error), so latency/cost/accuracy are attributable to that exact model,
    not to whichever model in a chain happened to answer.

    One case whose model call raises (bad JSON that survives the corrective retry,
    a timeout, missing credentials) is recorded as `got="error"` and counted as both
    a schema-invalid and an incorrect case for that model - it must not sink the
    whole model's run or the comparison.

    "schema-valid" here means the model's FIRST `ask_json` call already validated
    against `ComplianceVerdict` - no corrective retry needed - detected by counting
    how many telemetry entries `screen()`'s one `ask_model` call added for that case
    (1 = valid first try, 2 = needed the retry).
    """
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    original_settings = llm.settings
    model_rows = []
    for model_id in model_ids:
        llm.settings = dataclasses.replace(original_settings, models=(model_id,))
        telemetry_start = len(llm.telemetry)
        rows: list[dict] = []
        schema_valid = 0
        errors = 0
        try:
            for case in cases:
                lead = Lead(name="Eval Case", email="eval@example.test",
                            company=case["company"], website=case["website"])
                # The case STIPULATES its jurisdiction, so it is modelled as an established
                # fact. Left sourceless it is an unresolved one, and the deterministic HQ
                # gate would then decide every case before the model spoke - measured: the
                # false-clear metric went to 0 because nothing could reach "clear" at all.
                # These evals measure the MODEL; the gate has its own tests.
                research = Research(headquarters_country=case.get("hq_country", ""),
                                    hq_source="gleif", summary="", industry="")
                before = len(llm.telemetry)
                try:
                    result = screen(lead, research)
                    got = result.status
                    if len(llm.telemetry) - before <= 1:
                        schema_valid += 1
                except Exception as e:  # noqa: BLE001 - one bad case must not sink the whole model's run
                    got = "error"
                    errors += 1
                    logger_note = f"{type(e).__name__}: {e}"[:200]
                    rows.append({"company": case["company"], "expected": case["expected"],
                                 "got": got, "error": logger_note})
                    continue
                rows.append({"company": case["company"], "expected": case["expected"], "got": got})
        finally:
            calls = llm.telemetry[telemetry_start:]
            del llm.telemetry[telemetry_start:]
            llm.settings = original_settings

        n = len(rows)
        strict_correct = sum(1 for r in rows if r["got"] == r["expected"])
        flagged_correct = sum(1 for r in rows if r["got"] != "error" and _bucket(r["got"]) == _bucket(r["expected"]))
        false_clear = sum(1 for r in rows if r["got"] == "clear" and r["expected"] != "clear")
        latencies = [c["latency_ms"] for c in calls]
        model_rows.append({
            "model": model_id,
            "n_cases": n,
            "schema_valid_rate": round(schema_valid / n, 3) if n else 0.0,
            "strict_accuracy": round(strict_correct / n, 3) if n else 0.0,
            "flagged_vs_clear_accuracy": round(flagged_correct / n, 3) if n else 0.0,
            "false_clear_count": false_clear,
            "median_latency_ms": round(statistics.median(latencies)) if latencies else None,
            "cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
            "errors": errors,
            "rows": rows,
        })
    return {"models": model_rows}


def load_paid_deepseek_baseline(path: str | Path) -> dict | None:
    """reuse the EXISTING paid-DeepSeek baseline already sitting in
    evals/compliance_results.json from earlier work - never rerun it, just read its
    numbers. That file predates per-model telemetry tracking (`per_model` is null,
    `total_cost_usd` is 0.0 in the committed artifact) and has no per-case latency,
    so `median_latency_ms` here is really an AVERAGE (total_latency_ms / n_cases),
    and `schema_valid_rate`/`cost_usd` are reported as "not recorded" rather than
    guessed."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    confusion = data.get("confusion_matrix", {})
    n = data.get("n_cases", 0)
    false_clear = confusion.get("review", {}).get("clear", 0) + confusion.get("blocked", {}).get("clear", 0)
    avg_latency = data.get("total_latency_ms", 0) / n if n else None
    return {
        "model": "openrouter:deepseek/deepseek-chat-v3.1 (paid, existing baseline)",
        "n_cases": n,
        "schema_valid_rate": None,  # not recorded in the older run format
        "strict_accuracy": data.get("strict_accuracy"),
        "flagged_vs_clear_accuracy": data.get("flagged_vs_clear_accuracy"),
        "false_clear_count": false_clear,
        "median_latency_ms": round(avg_latency) if avg_latency else None,  # AVERAGE, not median - see docstring
        "cost_usd": data.get("total_cost_usd"),  # 0.0 in the committed artifact - not recorded, not zero cost
        "errors": None,
        "rows": None,
    }


def render_model_comparison(result: dict, baseline: dict | None = None) -> str:
    lines = ["# Per-model regression\n"]
    lines.append("| model | n cases | schema-valid rate | strict accuracy | flagged-vs-clear accuracy "
                 "| false-clear count (PRIMARY SAFETY METRIC) | median latency (ms) | cost (USD) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in result["models"]:
        lines.append(
            f"| {m['model']} | {m['n_cases']} | {m['schema_valid_rate']:.1%} | {m['strict_accuracy']:.1%} "
            f"| {m['flagged_vs_clear_accuracy']:.1%} | {m['false_clear_count']} "
            f"| {m['median_latency_ms']} | ${m['cost_usd']:.6f} |"
        )
    if baseline:
        sv = "n/a" if baseline["schema_valid_rate"] is None else f"{baseline['schema_valid_rate']:.1%}"
        cost = "n/a (not recorded)" if not baseline.get("cost_usd") else f"${baseline['cost_usd']:.6f}"
        lines.append(
            f"| {baseline['model']} | {baseline['n_cases']} | {sv} | {baseline['strict_accuracy']:.1%} "
            f"| {baseline['flagged_vs_clear_accuracy']:.1%} | {baseline['false_clear_count']} "
            f"| {baseline['median_latency_ms']} (avg, not median) | {cost} |"
        )
    lines.append("")
    for m in result["models"]:
        if m["errors"]:
            lines.append(f"- {m['model']}: {m['errors']} case(s) errored (counted against accuracy/schema-valid)")
    return "\n".join(lines) + "\n"
