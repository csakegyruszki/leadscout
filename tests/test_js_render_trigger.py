"""Crawl4AI escalation trigger: needs_js_render's four independent signals, and
maybe_render_with_crawl4ai's env-var + import gate.
"""
from leadscout.providers import website


def test_short_text_triggers():
    should, trigger = website.needs_js_render("<html><body>Hi</body></html>", "Hi")
    assert should is True
    assert trigger in ("short_text", "js_shell_marker")


def test_plain_prose_page_does_not_trigger():
    html = "<html><body>" + ("<p>Real prose sentence about the company. </p>" * 20) + "</body></html>"
    text = ("This is a real sentence describing what the company does in detail. " * 20).strip()
    should, trigger = website.needs_js_render(html, text)
    assert should is False
    assert trigger == ""


def test_low_text_html_ratio_triggers():
    text = "A" * 400  # long enough to clear the short-text trigger on its own
    html = "<div>" + ("x" * 50000) + "</div>" + text  # huge markup, tiny text share
    should, trigger = website.needs_js_render(html, text)
    assert should is True
    assert trigger == "low_text_html_ratio"


def test_js_shell_marker_with_little_text_triggers():
    html = '<html><body><div id="root"></div></body></html>'
    should, trigger = website.needs_js_render(html, "")
    assert should is True
    assert trigger == "js_shell_marker"


def test_mostly_short_lines_triggers():
    text = "\n".join(["Home", "About", "Careers", "Blog", "Contact", "Login", "Sign up"] * 10)
    html = "<html>" + text + "</html>" * 200  # keep the byte-ratio trigger from firing first
    should, trigger = website.needs_js_render(html, text)
    assert should is True
    assert trigger in ("short_text", "mostly_short_lines", "low_text_html_ratio")


def test_maybe_render_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LEADSCOUT_JS_RENDER", raising=False)
    result = website.maybe_render_with_crawl4ai("https://example.com", "<div id='root'></div>", "")
    assert result is None


def test_maybe_render_disabled_when_no_trigger_fires(monkeypatch):
    monkeypatch.setenv("LEADSCOUT_JS_RENDER", "1")
    html = "<html><body>" + ("<p>Real prose sentence about the company. </p>" * 20) + "</body></html>"
    text = ("This is a real sentence describing what the company does in detail. " * 20).strip()
    result = website.maybe_render_with_crawl4ai("https://example.com", html, text)
    assert result is None


def test_maybe_render_replaces_shell_text_with_fake_renderer(monkeypatch):
    """REVIEW-6bB-verified.md #3: wired escalation, tested with a fake renderer
    (no real crawl4ai calls) - rendered text must REPLACE the shell text."""
    monkeypatch.setenv("LEADSCOUT_JS_RENDER", "1")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "crawl4ai":
            import types
            return types.ModuleType("crawl4ai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(website, "_crawl4ai_fetch", lambda url: "Real rendered prose from the JS app, at last.")
    result = website.maybe_render_with_crawl4ai("https://example.com", '<div id="root"></div>', "")
    assert result == "Real rendered prose from the JS app, at last."


def test_maybe_render_falls_back_when_fake_renderer_returns_nothing(monkeypatch):
    monkeypatch.setenv("LEADSCOUT_JS_RENDER", "1")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "crawl4ai":
            import types
            return types.ModuleType("crawl4ai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(website, "_crawl4ai_fetch", lambda url: None)
    result = website.maybe_render_with_crawl4ai("https://example.com", '<div id="root"></div>', "")
    assert result is None


def test_maybe_render_logs_and_returns_none_without_crawl4ai_installed(monkeypatch, caplog):
    monkeypatch.setenv("LEADSCOUT_JS_RENDER", "1")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "crawl4ai":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # leadscout/pipeline.py sets logger.propagate=False on the shared "leadscout"
    # logger at import time (module singleton, may already be imported by another
    # test module in the same run) - caplog's handler lives on the root logger, so
    # force propagation back on for this one call or the record never reaches it.
    monkeypatch.setattr(website.logger, "propagate", True)
    with caplog.at_level("INFO", logger="leadscout"):
        result = website.maybe_render_with_crawl4ai("https://example.com", '<div id="root"></div>', "")
    assert result is None
    assert "disabled" in caplog.text
