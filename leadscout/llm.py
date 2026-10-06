"""Thin OpenRouter client with a model fallback chain.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import httpx
from pydantic import BaseModel, ValidationError

from .config import settings

# --- Design notes -----------------------------------------------------------------
# Also handles the cross-cutting concerns every call needs: falling back across a
# chain of models when one is unavailable or rate-limited, retrying transient
# failures on the last model in the chain, and recording what each call cost (in
# tokens, dollars and time) so the pipeline can put a real number in the tracker
# instead of a guess.
#
# Why a chain, not one model: OpenRouter's free-tier models are capped at roughly 50
# requests/day without >=$10 of account credit (see the design notes "Model chain"), so
# a single free model is a single point of failure. Trying two free models before
# falling back to a cheap paid one keeps the common case at $0 without betting the
# whole run on one rate limit.

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CLOUDFLARE_URL_TMPL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
logger = logging.getLogger("leadscout")

# 429 (rate limit), 404 (model unavailable) and 5xx are treated as "try the next
# model" on every hop but the last; a timeout is treated the same way.
_HOP_STATUS = {404, 429, 500, 502, 503, 504}
# Retry (fixed backoff) applies only to the LAST model in the chain: free tiers are
# rate-limited, so on an earlier hop a fast failover beats waiting out a backoff.
_BACKOFFS_S = (1, 2, 4)

# Hand-maintained price table (USD per 1M tokens) for paid models, so a cost still
# shows up when OpenRouter doesn't return usage.cost for a given key/model. Free
# models (":free" suffix) are always $0 - see _cost_and_source.
_PRICE_PER_MTOK = {
    "deepseek/deepseek-chat-v3.1": {"prompt": 0.14, "completion": 0.28},
}
_DEFAULT_PRICE = {"prompt": 0.5, "completion": 1.5}

# Drained per lead by pipeline.process_lead; module-level because ask_json/ask_model
# are called from research.py and compliance.py with no shared caller to thread state
# through.
telemetry: list[dict] = []


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in prose or code fences; pull out the OUTERMOST
    balanced object with a brace-depth scan (not first-`{`/last-`}`, which a fake
    JSON-closing fragment inside untrusted evidence text could otherwise exploit to
    truncate or splice the real object)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise LLMError(f"no JSON object in model output: {text[:200]!r}")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise LLMError(f"no balanced JSON object in model output: {text[:200]!r}")


# Provider -> wire dialect. The dialect decides how a request is built and how the
# reply is read; the provider decides where it is sent and how it authenticates. Most
# vendors speak OpenAI's chat-completions shape, but Anthropic and Google do not, so
# the pipeline is not restricted to "OpenAI-compatible" endpoints - see _build_payload
# and _content_from_body.
_PROVIDERS: dict[str, str] = {
    "openrouter": "openai",
    "cloudflare": "openai",
    "ollama": "openai",
    "openai": "openai",       # any other endpoint speaking that shape, via a base URL
    "anthropic": "anthropic",  # native Messages API
    "gemini": "gemini",        # native generateContent API
}


def _parse_model(model: str) -> tuple[str, str]:
    """"openrouter:<id>" / "cloudflare:<id>" / "ollama:<id>" / "openai:<id>" ->
    (provider, id); no prefix defaults to "openrouter" (back-compat with bare
    OpenRouter ids).

    They all speak the same OpenAI-compatible chat-completions API, so a prefix and a
    base URL are the whole difference:
      - `ollama:` needs no credential - it posts to a local daemon (`OLLAMA_HOST`,
        default http://localhost:11434), so the pipeline can run with no third-party
        LLM account and no per-lead cost;
      - `openai:` is the generic escape hatch for ANY other compatible endpoint
        (Groq, Together, DeepInfra, Mistral, a self-hosted vLLM, LM Studio, OpenAI
        itself): set `LEADSCOUT_LLM_BASE_URL`, plus `LEADSCOUT_LLM_API_KEY` if that
        endpoint wants one. Nothing about this pipeline is tied to one vendor.
    """
    for provider in _PROVIDERS:
        if model.startswith(f"{provider}:"):
            return provider, model[len(provider) + 1:]
    return "openrouter", model


def _cost_and_source(provider: str, model_id: str, usage: dict, prompt_tokens: int,
                      completion_tokens: int) -> tuple[float, bool, str]:
    """Returns (cost_usd, estimated, source)."""
    if provider == "cloudflare":
        # Workers AI free tier: no usage.cost, never estimated from a price table.
        return 0.0, False, "cloudflare free tier"
    if provider == "ollama":
        # Local inference: genuinely $0, not a $0 estimate. Falling through to the
        # price table would invent a charge for a model that was never billed.
        return 0.0, False, "local (ollama)"
    if provider in ("openai", "anthropic", "gemini") and usage.get("cost") is None:
        # An arbitrary compatible endpoint has no price we know. Say so rather than
        # pricing it from a table built for OpenRouter ids: a made-up number in the
        # telemetry is worse than an acknowledged gap.
        return 0.0, True, f"{provider}: cost not reported"
    if usage.get("cost") is not None:
        return float(usage["cost"]), False, "usage.cost"
    if model_id.endswith(":free"):
        return 0.0, False, "free tier"
    price = _PRICE_PER_MTOK.get(model_id, _DEFAULT_PRICE)
    cost = prompt_tokens / 1_000_000 * price["prompt"] + completion_tokens / 1_000_000 * price["completion"]
    return cost, True, "price table"


def _cloudflare_creds() -> tuple[str, str]:
    """getattr-guarded so a test Settings stub without Cloudflare fields still works."""
    return getattr(settings, "cloudflare_account_id", ""), getattr(settings, "cloudflare_api_token", "")


def _ollama_base() -> str:
    return (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")


def _compatible_base() -> str:
    """Base URL of a generic OpenAI-compatible endpoint, with or without the /v1."""
    base = (os.getenv("LEADSCOUT_LLM_BASE_URL") or "").rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _url_and_headers(provider: str, model_id: str) -> tuple[str, dict] | None:
    """Returns (url, headers) for the given provider, or None if the provider's
    credentials aren't configured (treated as an immediate hop failure, no call)."""
    if provider == "ollama":
        # No credential to check: a local daemon either answers or the hop fails like
        # any other. Deliberately not probed here - an unreachable host must behave
        # exactly like a timeout, not like a configuration error.
        return f"{_ollama_base()}/v1/chat/completions", {}
    if provider == "openai":
        base = _compatible_base()
        if not base:
            return None  # no base URL configured: an immediate hop failure, no call
        key = os.getenv("LEADSCOUT_LLM_API_KEY") or ""
        # Some compatible endpoints (a self-hosted vLLM, LM Studio) want no key at all.
        return f"{base}/chat/completions", ({"Authorization": f"Bearer {key}"} if key else {})
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY") or ""
        if not key:
            return None
        base = (os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com/v1").rstrip("/")
        # Its own auth header and a required version header - not a Bearer token.
        return f"{base}/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY") or ""
        if not key:
            return None
        base = (os.getenv("GEMINI_BASE_URL")
                or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        # The key goes in a header, never in the URL query string.
        return f"{base}/models/{model_id}:generateContent", {"x-goog-api-key": key}
    if provider == "cloudflare":
        account_id, api_token = _cloudflare_creds()
        if not account_id or not api_token:
            return None
        url = CLOUDFLARE_URL_TMPL.format(account_id=account_id)
        return url, {"Authorization": f"Bearer {api_token}"}
    if not settings.openrouter_api_key:
        return None
    return OPENROUTER_URL, {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        # OpenRouter attribution headers. No repository URL: see util.USER_AGENT.
        "X-Title": "leadscout",
    }


def _build_payload(dialect: str, system: str, user: str, temperature: float) -> dict:
    """One prompt, three wire shapes. Each asks for JSON in the way that provider
    supports: OpenAI has response_format, Anthropic and Google have their own field
    names and put the system prompt outside the message list."""
    if dialect == "anthropic":
        return {"max_tokens": 4096, "temperature": temperature, "system": system,
                "messages": [{"role": "user", "content": user}]}
    if dialect == "gemini":
        return {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": temperature,
                                     "responseMimeType": "application/json"}}
    return {
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "usage": {"include": True},  # ask OpenRouter for usage.cost when it can give it
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def _content_from_body(dialect: str, body: dict) -> str:
    """The generated text, wherever that dialect keeps it. A KeyError/IndexError here
    is caught by the caller and treated as a hop-worthy failure, same as any other
    unparseable body."""
    if dialect == "anthropic":
        return "".join(b.get("text", "") for b in body["content"] if b.get("type") == "text")
    if dialect == "gemini":
        return "".join(p.get("text", "") for p in body["candidates"][0]["content"]["parts"])
    return body["choices"][0]["message"]["content"]


def _usage_from_body(dialect: str, body: dict) -> tuple[dict, int, int]:
    """(raw usage, prompt tokens, completion tokens) - the three dialects name these
    differently, and telemetry must not silently report 0 tokens for two of them."""
    if dialect == "anthropic":
        u = body.get("usage") or {}
        return u, int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0)
    if dialect == "gemini":
        u = body.get("usageMetadata") or {}
        return u, int(u.get("promptTokenCount", 0) or 0), int(u.get("candidatesTokenCount", 0) or 0)
    u = body.get("usage") or {}
    return u, int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0)


def _call_model(provider: str, model_id: str, payload_base: dict, *, retry: bool) -> httpx.Response | None:
    """POST to one model. Returns None (never raises) on a timeout that exhausts
    its attempts, or when the provider's credentials are missing - the caller
    treats both the same as any other hop failure."""
    dest = _url_and_headers(provider, model_id)
    if dest is None:
        return None
    url, headers = dest
    # Gemini names the model in the URL path, not in the body; sending it as a field
    # is rejected as an unknown key.
    payload = payload_base if _PROVIDERS.get(provider) == "gemini" else dict(payload_base, model=model_id)
    attempts = len(_BACKOFFS_S) + 1 if retry else 1
    hop_timeout = getattr(settings, "llm_hop_timeout", 30.0)
    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=hop_timeout) as client:
                r = client.post(url, json=payload, headers=headers)
            if r.status_code in _HOP_STATUS and attempt < attempts - 1:
                time.sleep(_BACKOFFS_S[attempt])
                continue
            return r
        except httpx.TimeoutException:
            if attempt < attempts - 1:
                time.sleep(_BACKOFFS_S[attempt])
                continue
            return None
    return None


def ask_json(system: str, user: str, *, temperature: float = 0.1, purpose: str = "unknown") -> dict:
    models = settings.models
    providers_used = {_parse_model(m)[0] for m in models}
    account_id, api_token = _cloudflare_creds()
    has_openrouter_key = bool(settings.openrouter_api_key)
    has_cloudflare_creds = bool(account_id and api_token)
    usable = ("openrouter" in providers_used and has_openrouter_key) or \
             ("cloudflare" in providers_used and has_cloudflare_creds) or \
             "ollama" in providers_used or \
             ("openai" in providers_used and bool(_compatible_base())) or \
             ("anthropic" in providers_used and bool(os.getenv("ANTHROPIC_API_KEY"))) or \
             ("gemini" in providers_used and bool(os.getenv("GEMINI_API_KEY")))
    if not usable:
        raise LLMError("no usable provider credentials for the configured chain: need "
                        "OPENROUTER_API_KEY for an openrouter: model, CLOUDFLARE_ACCOUNT_ID/"
                        "CLOUDFLARE_API_TOKEN for a cloudflare: model, ANTHROPIC_API_KEY or "
                        "GEMINI_API_KEY for those, an ollama: model served by a local daemon, "
                        "or LEADSCOUT_LLM_BASE_URL for an openai: model (see .env.example)")
    hops = 0
    timeout_hops = 0
    for i, model in enumerate(models):
        provider, model_id = _parse_model(model)
        dialect = _PROVIDERS.get(provider, "openai")
        # Built per hop, not once: a chain may mix dialects (e.g. a local Ollama model
        # first, a hosted Anthropic model as the last-resort hop).
        payload_base = _build_payload(dialect, system, user, temperature)
        is_last = i == len(models) - 1
        hops += 1
        start = time.monotonic()
        r = _call_model(provider, model_id, payload_base, retry=is_last)
        latency_ms = int((time.monotonic() - start) * 1000)

        if r is None:
            timeout_hops += 1
            logger.info("llm hop %s/%s: %s timed out (or missing credentials)%s",
                        hops, len(models), model, "" if is_last else ", trying next model")
            if is_last:
                raise LLMError(f"all models in the chain failed; last ({model}) timed out "
                                f"or missing credentials")
            continue

        if r.status_code != 200:
            if r.status_code in _HOP_STATUS and not is_last:
                logger.info("llm hop %s/%s: %s -> HTTP %s, trying next model",
                            hops, len(models), model, r.status_code)
                continue
            raise LLMError(f"{provider} {r.status_code} from {model_id}: {r.text[:300]}")

        try:
            body = r.json()
            content = _content_from_body(dialect, body)
            if not content or not content.strip():
                raise LLMError("empty response body")
            parsed = _extract_json(content)
        except Exception as e:  # noqa: BLE001 - any parse failure is a hop-worthy failure
            if not is_last:
                logger.info("llm hop %s/%s: %s -> unparseable body (%s), trying next model",
                            hops, len(models), model, type(e).__name__)
                continue
            raise LLMError(f"could not parse response from {model_id}: {e}") from e

        usage, prompt_tokens, completion_tokens = _usage_from_body(dialect, body)
        cost_usd, estimated, source = _cost_and_source(provider, model_id, usage, prompt_tokens, completion_tokens)
        telemetry.append({
            "purpose": purpose,
            "model": model_id,
            "provider": provider,
            "hops": hops,
            "timeout_hops": timeout_hops,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(cost_usd, 6),
            "estimated": estimated,
            "source": source,
            "latency_ms": latency_ms,
        })
        return parsed

    raise LLMError("all models in the fallback chain failed")


def ask_model[TModel: BaseModel](
    system: str, user: str, schema: type[TModel], *, temperature: float = 0.1, purpose: str = "unknown",
) -> TModel:
    """ask_json + pydantic validation, with one corrective retry.

    Function calling would guarantee shape but not every OpenRouter model supports it
    reliably; a plain JSON prompt plus "here's what you got wrong, try again" is cheaper
    and, for the two schemas here, gets the model to the right shape on the first retry
    almost every time.
    """
    data = ask_json(system, user, temperature=temperature, purpose=purpose)
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        retry_user = f"{user}\n\nYour previous answer failed validation: {e}\nReturn corrected JSON only."
        data = ask_json(system, retry_user, temperature=temperature, purpose=purpose)
        return schema.model_validate(data)
