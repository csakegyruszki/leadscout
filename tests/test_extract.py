"""leadscout/extract.py: structured page extraction (Fix #4) - trafilatura main
text + FAQPage JSON-LD Q/A + <details>/<table> text, each with a locator. Offline,
no network - pure text/HTML fixtures.
"""
from leadscout import extract
from leadscout.cloud import assess_cloud_usage
from leadscout.providers import trust_pages

_ZAPIER_FAQ_HTML = """
<html><head>
<script type="application/ld+json">
{"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
  {"@type": "Question", "name": "Where is Zapier hosted?",
   "acceptedAnswer": {"@type": "Answer", "text": "Zapier's infrastructure is hosted on Amazon Web Services (AWS)."}}
]}
</script>
</head><body><p>Welcome to Zapier.</p></body></html>
"""


def test_extract_structured_returns_main_text_chunk():
    chunks = extract.extract_structured("<html></html>", "hello world")
    assert chunks[0].text == "hello world"
    assert chunks[0].locator == "main_text"


def test_extract_structured_finds_faq_page_jsonld_qa():
    chunks = extract.extract_structured(_ZAPIER_FAQ_HTML, "Welcome to Zapier.")
    faq = [c for c in chunks if c.locator.startswith("jsonld:FAQPage")]
    assert len(faq) == 1
    assert "hosted on Amazon Web Services" in faq[0].text
    assert faq[0].locator == "jsonld:FAQPage.mainEntity[0]"


def test_extract_structured_ignores_malformed_jsonld():
    html = '<html><script type="application/ld+json">{not valid json</script></html>'
    chunks = extract.extract_structured(html, "")
    assert chunks == []


def test_extract_structured_finds_details_and_table_text():
    html = (
        "<html><body>"
        "<details><summary>Subprocessors</summary>We use AWS for hosting.</details>"
        "<table><tr><td>Amazon Web Services</td><td>cloud subprocessor</td></tr></table>"
        "</body></html>"
    )
    chunks = extract.extract_structured(html, "")
    locators = {c.locator for c in chunks}
    assert "details[0]" in locators
    assert "table[0]" in locators
    table_chunk = next(c for c in chunks if c.locator == "table[0]")
    assert "Amazon Web Services" in table_chunk.text


def test_zapier_faq_jsonld_hosted_on_aws_direct_reaches_confirmed():
    """The v0.1.2 capability gap this fix closes: a FAQPage JSON-LD hosting
    statement was invisible to trafilatura's main-text extraction (JSON-LD lives
    inside a <script> tag, which trafilatura strips) - end to end, this must now
    reach trust_pages.HOSTED_ON (DIRECT) and cloud.py's CONFIRMED state."""
    chunks = extract.extract_structured(_ZAPIER_FAQ_HTML, "Welcome to Zapier.")
    pages = [{"url": "https://zapier.com", "text": c.text, "locator": c.locator} for c in chunks]
    result = trust_pages.run("zapier.com", pages)
    assert result.status == "ok"
    hosted_on = [ev for ev in result.evidence if ev.strength == "DIRECT" and ev.provider == "AWS"]
    assert len(hosted_on) == 1
    assessment = assess_cloud_usage(result.evidence)
    assert assessment.state == "CONFIRMED"
