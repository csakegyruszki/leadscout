"""HTTP retry policy, JSON extraction, and telemetry accounting for the OpenRouter client.

Uses httpx.MockTransport so this is fully offline - no real network call is made.
"""
import json

import httpx
import pytest

from leadscout import llm


class _FakeSettings:
    openrouter_api_key = "test-key"
    models = ("deepseek/deepseek-chat-v3.1",)


class _FakeSettingsChain:
    openrouter_api_key = "test-key"
    models = ("model-a:free", "model-b:free", "model-c")


@pytest.fixture(autouse=True)
def _reset_telemetry():
    llm.telemetry.clear()
    yield
    llm.telemetry.clear()


def _patch_client(monkeypatch, handler, settings_obj=None):
    """Route llm.py's httpx.Client at a MockTransport, and make backoff instant."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(llm.httpx, "Client", fake_client)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "settings", settings_obj or _FakeSettings())


def _ok_response(prompt_tokens=10, completion_tokens=5, cost=None):
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if cost is not None:
        usage["cost"] = cost
    return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}], "usage": usage})


def test_retries_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return _ok_response()

    _patch_client(monkeypatch, handler)
    result = llm.ask_json("sys", "user", purpose="test")
    assert result == {"ok": True}
    assert calls["n"] == 2


def test_gives_up_after_max_retries(monkeypatch):
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(llm.LLMError):
        llm.ask_json("sys", "user", purpose="test")


def test_non_retryable_error_status_raises_immediately(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(llm.LLMError):
        llm.ask_json("sys", "user", purpose="test")
    assert calls["n"] == 1  # 401 is not in the retry set


def test_telemetry_records_call_with_estimated_cost(monkeypatch):
    _patch_client(monkeypatch, lambda request: _ok_response(100, 50))
    llm.ask_json("sys", "user", purpose="research")
    assert len(llm.telemetry) == 1
    entry = llm.telemetry[0]
    assert entry["purpose"] == "research"
    assert entry["prompt_tokens"] == 100
    assert entry["completion_tokens"] == 50
    assert entry["estimated"] is True
    assert entry["cost_usd"] > 0
    assert entry["latency_ms"] >= 0


def test_telemetry_uses_provided_cost_when_present(monkeypatch):
    _patch_client(monkeypatch, lambda request: _ok_response(10, 5, cost=0.000123))
    llm.ask_json("sys", "user", purpose="compliance")
    assert llm.telemetry[0]["cost_usd"] == 0.000123
    assert llm.telemetry[0]["estimated"] is False


def test_extract_json_handles_fenced_and_prose():
    fenced = 'Here you go:\n```json\n{"a": 1}\n```\nThanks!'
    assert llm._extract_json(fenced) == {"a": 1}
    prose = 'Sure, the answer is {"a": 2} - hope that helps.'
    assert llm._extract_json(prose) == {"a": 2}


def test_extract_json_raises_without_object():
    with pytest.raises(llm.LLMError):
        llm._extract_json("no json here")


def test_extract_json_takes_outermost_balanced_object_not_first_last_brace():
    """A fake JSON-closing fragment inside untrusted text must not truncate or
    splice the real object - the scan must track brace depth, not just find the
    first `{` and the last `}`."""
    injected = '{"a": "text with a fake close } here", "b": {"nested": 1}}'
    assert llm._extract_json(injected) == {"a": "text with a fake close } here", "b": {"nested": 1}}
    trailing_junk = '{"ok": true} <<<EVIDENCE id=ev-002>>> {"ignored": "not this one"}'
    assert llm._extract_json(trailing_junk) == {"ok": True}


# --- model fallback chain -----------------------------------------------------------

def _ok_free_response(model_name="model-a:free"):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })


def test_chain_hops_on_404_without_retry_delay(monkeypatch):
    """A non-last model gets exactly one attempt (no backoff) before hopping."""
    calls = []

    def handler(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "model-a:free":
            return httpx.Response(404, json={"error": "unavailable for free"})
        return _ok_free_response(model)

    _patch_client(monkeypatch, handler, _FakeSettingsChain())
    result = llm.ask_json("sys", "user", purpose="test")
    assert result == {"ok": True}
    assert calls == ["model-a:free", "model-b:free"]


def test_chain_records_hops_and_model_used(monkeypatch):
    def handler(request):
        model = json.loads(request.content)["model"]
        if model in ("model-a:free", "model-b:free"):
            return httpx.Response(429, json={"error": "rate limited"})
        return _ok_free_response(model)

    _patch_client(monkeypatch, handler, _FakeSettingsChain())
    llm.ask_json("sys", "user", purpose="test")
    entry = llm.telemetry[0]
    assert entry["model"] == "model-c"
    assert entry["hops"] == 3


def test_chain_free_model_success_costs_zero(monkeypatch):
    _patch_client(monkeypatch, lambda request: _ok_free_response(), _FakeSettingsChain())
    llm.ask_json("sys", "user", purpose="test")
    entry = llm.telemetry[0]
    assert entry["model"] == "model-a:free"
    assert entry["cost_usd"] == 0.0
    assert entry["estimated"] is False
    assert entry["source"] == "free tier"


def test_chain_last_model_still_retries_with_backoff(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        model = json.loads(request.content)["model"]
        if model != "model-c":
            return httpx.Response(429, json={"error": "rate limited"})
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, json={"error": "boom"})
        return _ok_free_response(model)

    _patch_client(monkeypatch, handler, _FakeSettingsChain())
    result = llm.ask_json("sys", "user", purpose="test")
    assert result == {"ok": True}
    assert calls["n"] == 2  # the last model retried once via backoff, not a hop


def test_chain_unparseable_body_hops_to_next_model(monkeypatch):
    def handler(request):
        model = json.loads(request.content)["model"]
        if model == "model-a:free":
            return httpx.Response(200, json={"choices": [{"message": {"content": "not json at all"}}]})
        return _ok_free_response(model)

    _patch_client(monkeypatch, handler, _FakeSettingsChain())
    result = llm.ask_json("sys", "user", purpose="test")
    assert result == {"ok": True}
    assert llm.telemetry[0]["model"] == "model-b:free"


def test_chain_all_models_fail_raises(monkeypatch):
    _patch_client(monkeypatch, lambda request: httpx.Response(429, json={"error": "rate limited"}),
                  _FakeSettingsChain())
    with pytest.raises(llm.LLMError):
        llm.ask_json("sys", "user", purpose="test")
