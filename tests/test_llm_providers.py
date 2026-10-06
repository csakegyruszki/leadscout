"""Sec 1: provider-prefixed model ids, Cloudflare Workers AI routing, per-hop
timeout, and profile-based default chains. Fully offline (httpx.MockTransport).
"""
import json

import httpx
import pytest

from leadscout import config, llm


class _FakeSettingsCloudflareOnly:
    openrouter_api_key = ""
    models = ("cloudflare:@cf/openai/gpt-oss-120b",)
    cloudflare_account_id = "acct123"
    cloudflare_api_token = "cf-token"
    llm_hop_timeout = 30.0


class _FakeSettingsMixedChain:
    openrouter_api_key = "or-key"
    models = ("cloudflare:@cf/missing-creds-model", "openrouter:deepseek/deepseek-chat-v3.1")
    cloudflare_account_id = ""
    cloudflare_api_token = ""
    llm_hop_timeout = 30.0


class _FakeSettingsNoCreds:
    openrouter_api_key = ""
    models = ("openrouter:deepseek/deepseek-chat-v3.1",)
    cloudflare_account_id = ""
    cloudflare_api_token = ""
    llm_hop_timeout = 30.0


@pytest.fixture(autouse=True)
def _reset_telemetry():
    llm.telemetry.clear()
    yield
    llm.telemetry.clear()


def _patch_client(monkeypatch, handler, settings_obj):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(llm.httpx, "Client", fake_client)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "settings", settings_obj)


def _ok_response():
    return httpx.Response(200, json={
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })


def test_parse_model_defaults_to_openrouter():
    assert llm._parse_model("deepseek/deepseek-chat-v3.1") == ("openrouter", "deepseek/deepseek-chat-v3.1")
    assert llm._parse_model("openrouter:deepseek/x") == ("openrouter", "deepseek/x")
    assert llm._parse_model("cloudflare:@cf/openai/gpt-oss-120b") == ("cloudflare", "@cf/openai/gpt-oss-120b")


def test_cloudflare_hop_hits_correct_endpoint_and_costs_zero(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["model"] = json.loads(request.content)["model"]
        return _ok_response()

    _patch_client(monkeypatch, handler, _FakeSettingsCloudflareOnly())
    result = llm.ask_json("sys", "user", purpose="test")
    assert result == {"ok": True}
    assert seen["url"] == "https://api.cloudflare.com/client/v4/accounts/acct123/ai/v1/chat/completions"
    assert seen["auth"] == "Bearer cf-token"
    assert seen["model"] == "@cf/openai/gpt-oss-120b"
    entry = llm.telemetry[0]
    assert entry["provider"] == "cloudflare"
    assert entry["cost_usd"] == 0.0
    assert entry["estimated"] is False
    assert entry["source"] == "cloudflare free tier"


def test_missing_cloudflare_creds_hops_to_next_model_without_a_call(monkeypatch):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return _ok_response()

    _patch_client(monkeypatch, handler, _FakeSettingsMixedChain())
    result = llm.ask_json("sys", "user", purpose="test")
    assert result == {"ok": True}
    # the cloudflare hop never reached the network - only the openrouter model was called
    assert calls == ["deepseek/deepseek-chat-v3.1"]
    assert llm.telemetry[0]["timeout_hops"] == 1


def test_no_usable_provider_credentials_raises_immediately(monkeypatch):
    monkeypatch.setattr(llm, "settings", _FakeSettingsNoCreds())
    with pytest.raises(llm.LLMError, match="no usable provider credentials"):
        llm.ask_json("sys", "user", purpose="test")


def test_hop_timeout_is_read_from_settings(monkeypatch):
    captured = {}
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        kwargs["transport"] = httpx.MockTransport(lambda request: _ok_response())
        return real_client(*args, **kwargs)

    monkeypatch.setattr(llm.httpx, "Client", fake_client)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    class _Settings:
        openrouter_api_key = "k"
        models = ("openrouter:deepseek/deepseek-chat-v3.1",)
        cloudflare_account_id = ""
        cloudflare_api_token = ""
        llm_hop_timeout = 7.5

    monkeypatch.setattr(llm, "settings", _Settings())
    llm.ask_json("sys", "user", purpose="test")
    assert captured["timeout"] == 7.5


# --- config.py: profile-based default chains ---------------------------------------

def test_demo_profile_is_latency_first_default(monkeypatch):
    monkeypatch.delenv("LEADSCOUT_MODELS", raising=False)
    monkeypatch.delenv("LEADSCOUT_PROFILE", raising=False)
    assert config._load_profile() == "demo"
    models = config._load_models()
    assert models[0] == "openrouter:deepseek/deepseek-chat-v3.1"
    assert "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast" in models
    # Dropped from both profiles (Part 4): only 47.8% schema-valid, see config.py.
    assert "cloudflare:@cf/openai/gpt-oss-120b" not in models


def test_batch_profile_is_free_first_ordered_by_false_clear_count(monkeypatch):
    """Part 3: the batch chain is free-first (paid DeepSeek stays
    last as the reliable fallback) AND, among the free/0-cost models, ordered by
    the measured per-model regression's false-clear count ascending - the
    Cloudflare llama-3.3-70b model (0 false-clears) leads, not whichever free model
    happened to be first before that measurement existed."""
    monkeypatch.setenv("LEADSCOUT_PROFILE", "batch")
    monkeypatch.delenv("LEADSCOUT_MODELS", raising=False)
    models = config._load_models()
    assert models[0] == "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    assert models[-1] == "openrouter:deepseek/deepseek-chat-v3.1"
    assert not models[-1].endswith(":free")  # paid model is the last-resort fallback
    monkeypatch.delenv("LEADSCOUT_PROFILE", raising=False)


def test_leadscout_models_overrides_either_profile(monkeypatch):
    monkeypatch.setenv("LEADSCOUT_PROFILE", "batch")
    monkeypatch.setenv("LEADSCOUT_MODELS", "cloudflare:@cf/only-model")
    assert config._load_models() == ("cloudflare:@cf/only-model",)
    monkeypatch.delenv("LEADSCOUT_PROFILE", raising=False)
    monkeypatch.delenv("LEADSCOUT_MODELS", raising=False)


def test_unknown_profile_falls_back_to_demo(monkeypatch):
    monkeypatch.setenv("LEADSCOUT_PROFILE", "nonexistent")
    assert config._load_profile() == "demo"
    monkeypatch.delenv("LEADSCOUT_PROFILE", raising=False)
