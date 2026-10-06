"""HTTP endpoint wrapping the same pipeline the CLI uses - same validation, same output.

Run with: uvicorn leadscout.api:app
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator
from starlette.concurrency import run_in_threadpool

from .config import settings
from .inbound import service as inbound_service
from .inbound import webhook as inbound_webhook
from .llm import LLMError
from .models import Lead
from .notify import next_action
from .pipeline import process_lead
from .profile_config import size_bands

_TOKEN_QS = re.compile(r"(?i)([?&]token=)[^&\s\"']*")


def redact_token(text: str) -> str:
    return _TOKEN_QS.sub(lambda m: m.group(1) + "REDACTED", text)


class RedactTokenFilter(logging.Filter):
    """The inbound webhook accepts `?token=` for providers that cannot set headers; uvicorn's
    access log would otherwise record it with every request line."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_token(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact_token(a) if isinstance(a, str) else a for a in record.args)
        return True


logging.getLogger("uvicorn.access").addFilter(RedactTokenFilter())

app = FastAPI(title="leadscout", description="Inbound lead research, compliance screening and fit scoring")

_SIZE_BANDS = size_bands()

# Defensive: LLMError messages are built from HTTP status/response text (see
# llm.ask_json), never from settings.*_api_key directly, but a provider's error
# body could in principle echo back an Authorization header or a token-shaped
# string. Redact anything that looks like one before it reaches the response body.
_SECRET_PATTERN = re.compile(
    r"(?i)(bearer\s+[a-z0-9._~+/-]+=*|api[_-]?key[\"'=:\s]+[a-z0-9._~+/-]{8,}|sk-[a-z0-9]{10,})"
)


def _strip_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub("[REDACTED]", text)


def _has_usable_llm_credential() -> bool:
    has_openrouter = bool(settings.openrouter_api_key)
    has_cloudflare = bool(settings.cloudflare_account_id and settings.cloudflare_api_token)
    # An `ollama:` model needs no credential at all - it is served by a local daemon -
    # and an `openai:` model is any other OpenAI-compatible endpoint, configured by
    # base URL. Demanding a third-party key before we will even try would make a fully
    # local, zero-cost configuration look like a misconfigured one.
    has_local = any(m.startswith("ollama:") for m in settings.models)
    keyed = {"openai:": "LEADSCOUT_LLM_BASE_URL", "anthropic:": "ANTHROPIC_API_KEY",
             "gemini:": "GEMINI_API_KEY"}
    has_other = any(m.startswith(prefix) and os.getenv(env)
                    for m in settings.models for prefix, env in keyed.items())
    return has_openrouter or has_cloudflare or has_local or has_other


class LeadIn(BaseModel):
    name: str
    email: str
    company: str
    website: str
    job_title: str = ""
    company_size_band: str = ""

    @field_validator("company_size_band")
    @classmethod
    def _valid_band(cls, v: str) -> str:
        if v and v not in _SIZE_BANDS:
            raise ValueError(f"company_size_band must be one of {_SIZE_BANDS} or empty")
        return v


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "build": os.getenv("LEADSCOUT_BUILD", "unknown")}


@app.exception_handler(LLMError)
def _llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"error": "llm_upstream", "detail": _strip_secrets(str(exc))})


@app.exception_handler(Exception)
def _internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    run_id = getattr(exc, "leadscout_run_id", None)
    return JSONResponse(status_code=500, content={"error": "internal", "run_id": run_id})


@app.post("/leads")
def create_lead(lead_in: LeadIn) -> dict:
    if not _has_usable_llm_credential():
        return JSONResponse(
            status_code=503,
            content={
                "error": "configuration",
                "detail": "no usable LLM provider credential; see .env.example",
            },
        )
    lead = Lead(**lead_in.model_dump())
    outcome = process_lead(lead, verbose=False)
    payload = outcome.to_dict()
    # The routed action is THE thing a rep acts on, and it is not derivable from the
    # rest of the payload without re-implementing notify.next_action's precedence
    # (compliance outranks fit; a qualified account with an unverifiable contact is
    # still worth working). A second copy of that rule in a caller - the demo page
    # first shipped with one in JavaScript - drifts from this one silently, so the
    # single implementation is served here.
    payload["next_action"] = next_action(outcome)
    return payload


@app.post("/inbound/mail", status_code=202)
@app.post("/inbound/mail/mime", status_code=202, include_in_schema=False)   # Mailgun routes ending in /mime
async def inbound_mail(request: Request, background: BackgroundTasks):
    """Inbound e-mail webhook (see docs/inbound-email.md). Disabled (404) until
    LEADSCOUT_INBOUND_TOKEN is set; the token is compared in constant time. Parsing and
    de-duplication happen here; the pipeline runs after the 202 is sent."""
    if not inbound_webhook.configured_token():
        return JSONResponse(status_code=404, content={"error": "not_found"})
    supplied = request.headers.get("x-leadscout-token") or request.query_params.get("token")
    if not inbound_webhook.token_matches(supplied):
        return JSONResponse(status_code=401, content={"error": "unauthorized"})

    cap = inbound_webhook.request_body_cap(inbound_service.max_message_bytes())
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > cap:
        return JSONResponse(status_code=413, content={"error": "payload_too_large"})
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > cap:
            return JSONResponse(status_code=413, content={"error": "payload_too_large"})
        chunks.append(chunk)
    try:
        raw = inbound_webhook.to_raw_mime(request.headers.get("content-type", ""), b"".join(chunks))
    except inbound_webhook.BadPayload as exc:
        return JSONResponse(status_code=400, content={"error": "bad_payload", "detail": str(exc)})

    # parse + claim are CPU/disk work: off the event loop, so one slow message cannot stall the server
    prep = await run_in_threadpool(inbound_service.prepare, raw, source="webhook")
    if prep.done:
        rec = prep.record
        return {"id": prep.record_id, "status": rec["status"], "duplicate": rec["status"] == "duplicate",
                "reason": rec.get("reason", "")}
    background.add_task(inbound_service.run_prepared, prep)
    return {"id": prep.record_id, "status": "accepted", "duplicate": False,
            "message_id": prep.parsed.message_id}


_FORM_HTML = Path(__file__).with_name("static") / "form.html"


@app.get("/", response_class=HTMLResponse)
def form() -> HTMLResponse:
    """The inbound form (four fields), as an internal demo console - see the
    page's own banner for why it is not the public-facing version."""
    return HTMLResponse(_FORM_HTML.read_text(encoding="utf-8"))
