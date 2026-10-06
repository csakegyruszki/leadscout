"""CLI entry: run one lead from flags, a batch from a JSON file, or the compliance eval."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import runtime_identity
from .models import Lead
from .pipeline import process_lead
from .profile_config import size_bands


def _mail(a) -> int:
    from .inbound import service
    if a.mail_cmd == "process":
        rec = service.process_message(a.file.read_bytes(), source=f"file:{a.file.name}",
                                      pipeline=not a.no_pipeline)
        if a.json:
            print(json.dumps(rec, indent=2, ensure_ascii=False))
        else:
            lead = rec.get("lead") or {}
            print(f"{rec['record_id']}: {rec['status']} - {rec.get('reason', '')}")
            if lead:
                print(f"  lead: {lead.get('name')} <{lead.get('email')}> {lead.get('company')} {lead.get('website')}")
            if rec.get("injection_flags"):
                print(f"  injection flags: {', '.join(rec['injection_flags'])}")
        return 1 if rec["status"] in ("failed", "rejected") else 0
    if a.mail_cmd == "resume":
        return _resume(service)
    from .inbound import imap
    code = 0
    try:
        result = imap.poll_once()
    except imap.ImapConfigError as exc:
        print(f"mail poll: {exc}", file=sys.stderr)
        code = 2
    else:
        print(f"mail poll: {result['total']} message(s), by status {result['counts']}, errors={result['errors']}")
        code = 1 if result["errors"] else 0
    if a.resume:                      # a poll that failed to log in must not also skip the resume pass
        code = max(code, _resume(service))
    return code


def _resume(service) -> int:
    result = service.resume()
    print(f"mail resume: {result['total']} message(s), by status {result['counts']}")
    return 1 if result["errors"] else 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="leadscout", description="Inbound lead research + compliance + fit scoring")
    p.add_argument("--quiet", action="store_true",
                    help="suppress the per-lead print output (structured logging still runs)")
    sub = p.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("lead", help="process a single lead")
    one.add_argument("--name", required=True)
    one.add_argument("--email", required=True)
    one.add_argument("--company", required=True)
    one.add_argument("--website", required=True)
    one.add_argument("--job-title", default="")
    one.add_argument("--size-band", default="", help=" | ".join(size_bands()))

    batch = sub.add_parser("batch", help="process every lead in a JSON list")
    batch.add_argument("file", type=Path)

    ev = sub.add_parser("eval", help="run the compliance eval harness against labelled cases")
    ev.add_argument("file", type=Path, nargs="?")
    ev.add_argument("--adversarial", type=Path, metavar="FILE",
                     help="run the adversarial prompt-injection eval instead")
    ev.add_argument("--models", type=str, metavar="LIST",
                     help="comma-separated model ids (each becomes a single-model chain, no fallback) "
                          "to run FILE's compliance cases through - one row per model in "
                          "evals/model_comparison.md")

    mail = sub.add_parser("mail", help="inbound e-mail: process one .eml file, or poll the IMAP mailbox")
    mail_sub = mail.add_subparsers(dest="mail_cmd", required=True)
    mp = mail_sub.add_parser("process", help="run one raw message (.eml) through the inbound pipeline")
    mp.add_argument("file", type=Path)
    mp.add_argument("--json", action="store_true", help="print the full record as JSON")
    mp.add_argument("--no-pipeline", action="store_true",
                    help="parse + extract + write the record only: no research, no LLM, no dedupe claim")
    mpoll = mail_sub.add_parser("poll", help="poll the IMAP mailbox (LEADSCOUT_IMAP_*)")
    mpoll.add_argument("--once", action="store_true", required=True,
                       help="one pass, then exit (schedule it with infra/systemd/leadscout-mail-poll.timer)")
    mpoll.add_argument("--resume", action="store_true",
                       help="after the poll, also re-drive received-but-unfinished messages (see `mail resume`)")
    mail_sub.add_parser("resume", help="re-drive messages that were received but never got a result "
                                       "(stale claims after a crash, failed ones with attempts left)")

    sub.add_parser("cache-list", help="list cached OpenSanctions API responses (name, country, fetched_at, source)")

    a = p.parse_args(argv)
    if a.cmd == "lead":
        process_lead(Lead(a.name, a.email, a.company, a.website, a.job_title, a.size_band), verbose=not a.quiet)
    elif a.cmd == "batch":
        # One lead's failure is that lead's failure. Measured before this guard: a hard
        # LLM failure on lead 2 of 3 propagated out of the loop, so lead 3 was never
        # processed and the run ended with a traceback instead of a report - every lead
        # after the first failure silently did not happen. The loop reports the lead that
        # failed, keeps the ones already written, continues, and exits non-zero so a
        # caller (or CI) cannot mistake a partial batch for a complete one.
        failed: list[tuple[str, str]] = []
        items = json.loads(a.file.read_text(encoding="utf-8"))
        # F-15: the run context is created HERE, before the first lead, and every
        # result JSON this batch writes carries it. Printing it makes the binding the
        # proof gate will check visible in the build log too.
        ctx = runtime_identity.context()
        print(f"run {ctx.id}: source_fingerprint {ctx.source_fingerprint}")
        for item in items:
            lead = Lead(**item)
            try:
                process_lead(lead, verbose=not a.quiet)
            except Exception as e:  # noqa: BLE001 - the next lead is still worth processing
                failed.append((lead.company, f"{type(e).__name__}: {e}"))
                print(f"FAILED {lead.company}: {type(e).__name__}: {e}", file=sys.stderr)
        # The summary is unconditional. A failure visible only as an exit code is one a
        # reader can miss, and a partial batch that says nothing about what is missing is
        # exactly how five rows get used as six leads' worth of evidence.
        print(f"\nbatch: attempted={len(items)} succeeded={len(items) - len(failed)} failed={len(failed)}")
        for company, reason in failed:
            print(f"  failed lead: {company}: {reason}")
        if failed:
            print(f"\n{len(failed)} of {len(items)} leads failed and produced no row:", file=sys.stderr)
            for company, reason in failed:
                print(f"  {company}: {reason}", file=sys.stderr)
            return 1
    elif a.cmd == "mail":
        return _mail(a)
    elif a.cmd == "cache-list":
        from .sanctions import list_cache
        rows = list_cache()
        if not rows:
            print("no cached OpenSanctions responses yet")
        for row in rows:
            print(f"{row['name']:24s} country={row['country'] or '-':4s} results={row['n_results']:2d}  "
                  f"fetched_at={row['fetched_at']}  source={row['source']}")
    elif a.cmd == "eval" and a.adversarial:
        from .evals import render_adversarial_report, run_adversarial_eval
        result = run_adversarial_eval(a.adversarial)
        print(render_adversarial_report(result))
        out_dir = a.adversarial.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "adversarial_results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "adversarial_results.md").write_text(render_adversarial_report(result), encoding="utf-8")
    elif a.cmd == "eval" and a.models:
        if a.file is None:
            p.error("eval --models: FILE (the labelled compliance cases) is required")
        from .evals import load_paid_deepseek_baseline, render_model_comparison, run_model_comparison
        model_ids = [m.strip() for m in a.models.split(",") if m.strip()]
        result = run_model_comparison(a.file, model_ids)
        baseline = load_paid_deepseek_baseline(a.file.parent / "compliance_results.json")
        report = render_model_comparison(result, baseline)
        print(report)
        out_dir = a.file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model_comparison.json").write_text(
            json.dumps({"models": result["models"], "baseline": baseline}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (out_dir / "model_comparison.md").write_text(report, encoding="utf-8")
    else:
        if a.cmd == "eval" and a.file is None:
            p.error("eval: FILE is required unless --adversarial is given")
        from .evals import render_eval_report, run_compliance_eval
        result = run_compliance_eval(a.file)
        print(render_eval_report(result))
        (a.file.parent / "compliance_results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (a.file.parent / "compliance_results.md").write_text(
            render_eval_report(result, markdown=True), encoding="utf-8")


def console_main() -> int:
    """Entry point of the `leadscout` console script (pyproject [project.scripts])."""
    return main() or 0


if __name__ == "__main__":
    # main() returns 1 when a batch had failures; without this the process still
    # exited 0 and a partial batch looked like a complete one.
    sys.exit(main() or 0)
