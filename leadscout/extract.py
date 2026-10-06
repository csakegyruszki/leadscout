"""Structured page extraction (Fix #4, v0.2 PART B): one helper, used for both the
landing page and every crawled subpage, that returns trafilatura's main text PLUS
whatever FAQPage JSON-LD Q/A, visible `<details>`, and `<table>` element text the
raw HTML has - each tagged with a locator (a JSON path for JSON-LD, an element
index otherwise) so a downstream Evidence can cite exactly which structural
element it saw, not just "the page".

v0.1.2 fetched a Zapier-style FAQ block via trafilatura's own extraction but never
saw it: trafilatura strips `<script>` tags (where FAQPage JSON-LD always lives) and
does not walk `<details>`/`<table>` markup for a "main text" summary - a first-party
hosting statement living in an FAQ answer or a subprocessor table was invisible to
every downstream provider (research.py, providers/trust_pages.py). No new
dependency: stdlib `json`/`re` only, matching the manual-regex HTML handling
providers/trust_pages.py and providers/website.py already use.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_LD_JSON_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S)
_DETAILS_RE = re.compile(r"<details\b[^>]*>(.*?)</details>", re.I | re.S)
_TABLE_RE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.I | re.S)
_TR_CLOSE_RE = re.compile(r"(?i)</tr\s*>")


@dataclass(frozen=True)
class ExtractedChunk:
    text: str
    locator: str  # e.g. "main_text", "jsonld:FAQPage.mainEntity[2]", "details[0]", "table[1]"


def _strip_tags(html_fragment: str) -> str:
    """Visible text only, same policy as providers/trust_pages.py's `_visible_text`
    (detection must never run on raw markup - a tag/attribute soup can hide an
    accidental substring hit, or read as a sentence it never was)."""
    stripped = _SCRIPT_STYLE_RE.sub(" ", html_fragment)
    return re.sub(r"\s+", " ", unescape(_TAG_RE.sub(" ", stripped))).strip()


def _faq_page_nodes(data) -> list[dict]:
    """Every FAQPage-typed node in one parsed JSON-LD document - directly, or
    inside a top-level `@graph` array (both shapes appear in the wild)."""
    candidates = data if isinstance(data, list) else [data]
    nodes: list[dict] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        graph = cand.get("@graph")
        pool = graph if isinstance(graph, list) else [cand]
        for node in pool:
            if not isinstance(node, dict):
                continue
            node_type = node.get("@type")
            types = node_type if isinstance(node_type, list) else [node_type]
            if "FAQPage" in types:
                nodes.append(node)
    return nodes


def _faq_chunks(html: str) -> list[ExtractedChunk]:
    out: list[ExtractedChunk] = []
    for block in _LD_JSON_RE.findall(html):
        try:
            data = json.loads(block)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        for node in _faq_page_nodes(data):
            entities = node.get("mainEntity") or []
            if isinstance(entities, dict):
                entities = [entities]
            for i, q in enumerate(entities):
                if not isinstance(q, dict):
                    continue
                name = str(q.get("name", "")).strip()
                answer = q.get("acceptedAnswer")
                answer_text = str(answer.get("text", "")).strip() if isinstance(answer, dict) else ""
                combined = _strip_tags(f"{name} {answer_text}".strip())
                if combined:
                    out.append(ExtractedChunk(text=combined, locator=f"jsonld:FAQPage.mainEntity[{i}]"))
    return out


def _details_chunks(html: str) -> list[ExtractedChunk]:
    out = []
    for i, block in enumerate(_DETAILS_RE.findall(html)):
        text = _strip_tags(block)
        if text:
            out.append(ExtractedChunk(text=text, locator=f"details[{i}]"))
    return out


def _table_chunks(html: str) -> list[ExtractedChunk]:
    out = []
    for i, block in enumerate(_TABLE_RE.findall(html)):
        # Keep row boundaries as line breaks before stripping tags, so a
        # "provider | purpose" subprocessor-table row still reads as one line
        # (and one sentence, for providers/trust_pages.py's sentence splitter)
        # instead of merging into its neighbours.
        text = _strip_tags(_TR_CLOSE_RE.sub("\n", block))
        if text:
            out.append(ExtractedChunk(text=text, locator=f"table[{i}]"))
    return out


def extract_structured(html: str, main_text: str = "") -> list[ExtractedChunk]:
    """`main_text` is the caller's own already-computed trafilatura extraction
    (never recomputed here - one trafilatura call per page stays the caller's
    responsibility). `html` is that same page's raw fetched HTML; passing ""
    skips the FAQ/details/table passes (nothing to parse) and returns just the
    main-text chunk. Never raises: a malformed JSON-LD block is skipped, not
    fatal - see `_faq_chunks`."""
    chunks: list[ExtractedChunk] = []
    if main_text:
        chunks.append(ExtractedChunk(text=main_text, locator="main_text"))
    if html:
        chunks.extend(_faq_chunks(html))
        chunks.extend(_details_chunks(html))
        chunks.extend(_table_chunks(html))
    return chunks
