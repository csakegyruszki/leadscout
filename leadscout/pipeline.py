"""Orchestration: lead -> research -> compliance -> fit -> tracker -> email."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import llm, runtime_identity
from .compliance import screen
from .config import settings
from .fit import confidence_qualifies, fit_threshold, score_fit
from .models import Lead, LeadOutcome
from .notify import notify
from .provenance import ProvenanceRun, record_policy_files
from .research import research_lead
from .tracker import write_row

logger = logging.getLogger("leadscout")
if not logger.handlers:  # avoid duplicate handlers if the module is re-imported (tests)
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def process_lead(lead: Lead, *, verbose: bool = True, artifact_stem: str | None = None) -> LeadOutcome:
    """verbose controls the CLI's print output only; structured logging always runs."""
    log = print if verbose else (lambda *a, **k: None)
    log(f"\n=== {lead.company} ({lead.website}) ===")
    logger.info("start company=%s website=%s", lead.company, lead.website)

    telemetry_start = len(llm.telemetry)  # mark, so we can drain just this lead's calls below

    run = ProvenanceRun.start(lead.company)
    try:
        record_policy_files(run)

        research = research_lead(lead, run)
        log(f"research: website {'ok' if research.website_ok else 'FAILED'}, "
            f"wiki {'yes' if research.wikipedia_url else 'no'}, HQ={research.headquarters_country}, "
            f"employees={research.estimated_employees}, confidence={research.confidence}")
        logger.info("research done hq=%s employees=%s confidence=%s",
                    research.headquarters_country, research.estimated_employees, research.confidence)

        compliance = screen(lead, research, run)
        log(f"compliance: {compliance.status.upper()} - {compliance.reasoning}")
        logger.info("compliance status=%s", compliance.status)

        fit = score_fit(lead, research)
        log(f"fit: {fit.score}/100 ({fit.confidence}) - {fit.reasoning}")
        logger.info("fit score=%s confidence=%s cloud_spend=%s", fit.score, fit.confidence, fit.cloud_spend_band)

        calls = llm.telemetry[telemetry_start:]
        del llm.telemetry[telemetry_start:]  # drained: pipeline is the sole consumer of this window
        llm_calls = len(calls)
        llm_cost_usd = round(sum(c["cost_usd"] for c in calls), 6)
        llm_latency_ms = sum(c["latency_ms"] for c in calls)
        llm_models = ", ".join(sorted({c["model"] for c in calls}))
        llm_hops = sum(c.get("hops", 1) for c in calls)

        provenance_status, provenance_head, evidence_count = run.finish()
        logger.info("provenance run=%s status=%s head=%s evidence=%s",
                    run.run_id, provenance_status, provenance_head, evidence_count)

        sales_ready = (
            compliance.status == "clear"
            and fit.score >= fit_threshold()
            and confidence_qualifies(fit.confidence)
        )
        outcome = LeadOutcome(lead=lead, research=research, compliance=compliance, fit=fit,
                              sales_ready=sales_ready,
                              llm_calls=llm_calls, llm_cost_usd=llm_cost_usd, llm_latency_ms=llm_latency_ms,
                              llm_models=llm_models, llm_hops=llm_hops, llm_profile=settings.profile,
                              run_id=run.run_id, evidence_count=evidence_count,
                              provenance_status=provenance_status, provenance_head=provenance_head,
                              # F-15: stamped from the run context created at run start,
                              # never recomputed here - the binding must describe the code
                              # this lead actually ran on.
                              proof_run=runtime_identity.context().as_binding())
        row = write_row(outcome, settings.tracker_path)
        eml = notify(outcome, artifact_stem=artifact_stem) if artifact_stem else notify(outcome)
        log(f"tracker row {row} -> {settings.tracker_path.name}; email -> {eml.name}; "
            f"sales-ready={outcome.sales_ready}")
        logger.info("done row=%s sales_ready=%s llm_calls=%s llm_cost_usd=%.6f llm_latency_ms=%s",
                    row, outcome.sales_ready, llm_calls, llm_cost_usd, llm_latency_ms)

        (settings.out_dir / "results").mkdir(parents=True, exist_ok=True)
        (settings.out_dir / "results" / f"{artifact_stem or Path(eml).stem}.json").write_text(
            json.dumps(outcome.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        run.mark_completed()
        return outcome
    except Exception as exc:
        run.mark_aborted(type(exc).__name__)
        logger.exception("run aborted run=%s company=%s", run.run_id, lead.company)
        # Let a caller (e.g. the API layer) report which run aborted without
        # threading run_id through every exception type raised inside the pipeline.
        exc.leadscout_run_id = run.run_id
        raise
