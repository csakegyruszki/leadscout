"""Any OpenAI-compatible endpoint can drive this pipeline - nothing is tied to one vendor.

OpenRouter, Cloudflare Workers AI, a local Ollama daemon and any other compatible API
all speak the same chat-completions protocol, so the provider prefix and the base URL
are the whole difference. `ollama:` needs no key; `openai:` is the generic escape
hatch (Groq, Together, DeepInfra, a self-hosted vLLM, OpenAI itself) configured by
`LEADSCOUT_LLM_BASE_URL` and an optional `LEADSCOUT_LLM_API_KEY`.
"""
import dataclasses

import pytest

from leadscout import api, llm


def test_each_provider_prefix_parses():
    assert llm._parse_model("ollama:qwen2.5-coder:7b") == ("ollama", "qwen2.5-coder:7b")
    assert llm._parse_model("openai:llama-3.3-70b-versatile") == ("openai", "llama-3.3-70b-versatile")
    assert llm._parse_model("cloudflare:@cf/meta/llama-3.3-70b") == ("cloudflare", "@cf/meta/llama-3.3-70b")
    assert llm._parse_model("openrouter:deepseek/deepseek-chat") == ("openrouter", "deepseek/deepseek-chat")
    # A bare id stays OpenRouter, for back-compatibility with existing chains.
    assert llm._parse_model("deepseek/deepseek-chat") == ("openrouter", "deepseek/deepseek-chat")


def test_ollama_needs_no_credential_and_honours_OLLAMA_HOST(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    url, headers = llm._url_and_headers("ollama", "gemma4:e2b")
    assert url == "http://localhost:11434/v1/chat/completions"
    assert headers == {}
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434/")
    url, _ = llm._url_and_headers("ollama", "gemma4:e2b")
    assert url == "http://10.0.0.5:11434/v1/chat/completions"


def test_any_compatible_endpoint_works_from_a_base_url(monkeypatch):
    monkeypatch.setenv("LEADSCOUT_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LEADSCOUT_LLM_API_KEY", "k")
    url, headers = llm._url_and_headers("openai", "llama-3.3-70b-versatile")
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert headers == {"Authorization": "Bearer k"}
    # An endpoint that wants no key (self-hosted vLLM, LM Studio) must still work.
    monkeypatch.delenv("LEADSCOUT_LLM_API_KEY")
    assert llm._url_and_headers("openai", "x")[1] == {}
    # A base URL already ending in the full path is accepted, not doubled.
    monkeypatch.setenv("LEADSCOUT_LLM_BASE_URL", "http://localhost:8000/v1/chat/completions")
    assert llm._url_and_headers("openai", "x")[0] == "http://localhost:8000/v1/chat/completions"
    # No base URL at all: an immediate hop failure, never a call to somewhere random.
    monkeypatch.delenv("LEADSCOUT_LLM_BASE_URL")
    assert llm._url_and_headers("openai", "x") is None


def test_native_anthropic_and_gemini_apis_are_supported(monkeypatch):
    """Not just "OpenAI-compatible": these two have their own request and response
    shapes, so the pipeline speaks all three rather than excluding them."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    url, headers = llm._url_and_headers("anthropic", "claude-x")
    assert url.endswith("/messages") and headers["x-api-key"] == "a"
    assert "anthropic-version" in headers

    monkeypatch.setenv("GEMINI_API_KEY", "g")
    url, headers = llm._url_and_headers("gemini", "gemini-x")
    assert url.endswith("/models/gemini-x:generateContent")
    # The key belongs in a header, never in a query string.
    assert headers["x-goog-api-key"] == "g" and "key=" not in url


def test_each_dialect_builds_its_own_request_shape():
    anthropic = llm._build_payload("anthropic", "SYS", "USER", 0.1)
    assert anthropic["system"] == "SYS" and anthropic["messages"][0]["content"] == "USER"
    assert "response_format" not in anthropic  # not a field Anthropic accepts

    gemini = llm._build_payload("gemini", "SYS", "USER", 0.1)
    assert gemini["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert gemini["generationConfig"]["responseMimeType"] == "application/json"

    openai = llm._build_payload("openai", "SYS", "USER", 0.1)
    assert openai["messages"][0] == {"role": "system", "content": "SYS"}
    assert openai["response_format"] == {"type": "json_object"}


def test_each_dialect_is_read_back_correctly():
    assert llm._content_from_body("anthropic", {"content": [{"type": "text", "text": '{"a":1}'}]}) == '{"a":1}'
    assert llm._content_from_body(
        "gemini", {"candidates": [{"content": {"parts": [{"text": '{"a":1}'}]}}]}) == '{"a":1}'
    assert llm._content_from_body(
        "openai", {"choices": [{"message": {"content": '{"a":1}'}}]}) == '{"a":1}'


def test_token_usage_is_read_from_each_dialects_own_field_names():
    """Reading only OpenAI's names would report 0 tokens for the other two."""
    _, p, c = llm._usage_from_body("anthropic", {"usage": {"input_tokens": 11, "output_tokens": 7}})
    assert (p, c) == (11, 7)
    _, p, c = llm._usage_from_body(
        "gemini", {"usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7}})
    assert (p, c) == (11, 7)
    _, p, c = llm._usage_from_body("openai", {"usage": {"prompt_tokens": 11, "completion_tokens": 7}})
    assert (p, c) == (11, 7)


def test_an_unpriced_custom_endpoint_reports_a_gap_not_a_number():
    cost, estimated, source = llm._cost_and_source("openai", "some-model", {}, 1000, 1000)
    assert cost == 0.0 and estimated is True and "not reported" in source


def test_a_local_model_is_zero_cost_not_a_zero_estimate():
    """The price table must not invent a charge for a call nobody billed."""
    cost, estimated, source = llm._cost_and_source("ollama", "gemma4:e2b", {}, 1000, 1000)
    assert cost == 0.0 and estimated is False and source == "local (ollama)"


def test_a_local_only_chain_is_not_a_missing_credential(monkeypatch):
    """Requiring a third-party key would make a fully local setup look misconfigured."""
    local_only = dataclasses.replace(api.settings, openrouter_api_key="", cloudflare_account_id="",
                                     cloudflare_api_token="", models=("ollama:gemma4:e2b",))
    monkeypatch.setattr(api, "settings", local_only)
    assert api._has_usable_llm_credential() is True


def test_no_provider_at_all_still_fails_loudly(monkeypatch):
    nothing = dataclasses.replace(api.settings, openrouter_api_key="", cloudflare_account_id="",
                                  cloudflare_api_token="", models=("openrouter:some/model",))
    monkeypatch.setattr(api, "settings", nothing)
    assert api._has_usable_llm_credential() is False
    monkeypatch.setattr(llm, "settings", nothing)
    with pytest.raises(llm.LLMError, match="no usable provider credentials"):
        llm.ask_json("s", "u")
