"""Compliance / do-not-engage screening.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from .llm import LLMError, ask_model
from .models import ComplianceResult, ComplianceVerdict, Lead, Research
from .profile import OrgProfile, active_profile
from .profile_config import ComplianceConfig, compliance_config
from .prompt_safety import INJECTION_WARNING, wrap_evidence
from .sanctions import screen_sanctions
from .util import registrable_domain

# --- Design notes -----------------------------------------------------------------
# Three layers, on purpose:
# 1. Deterministic fuzzy pre-screen (rapidfuzz) against the do-not-engage list, plus an
#    exact abbreviation check and a sanctioned-HQ/TLD check. Cheap, explainable. It
#    NEVER clears a lead on its own.
# 2. Real sanctions/PEP screening via the OpenSanctions match API (sanctions.py) -
#    actual watchlist data, not a hand-typed list.
# 3. LLM reasoning over the lead, the research and both sets of candidates. It decides
#    between clear / review / blocked and must explain partial matches. The model sees
#    the *whole* picture, so it can also flag things the deterministic layers missed.
#
# AUTHORITY ORDER. Two separate ladders, because a source can be good at one and not
# the other - the domain enrichment names the organisation behind a domain well, and
# places it on the wrong continent (measured 2026-09-20).
#
#   IDENTITY (who applied):  registry / verified legal record > first-party legal
#       notice > first-party structured Organization data > exact-domain enrichment
#       > name inference
#   HQ / JURISDICTION (where it is): registry / legal record > first-party legal
#       notice > first-party explicit corporate statement > enrichment > LLM inference
#
# A rung below "registry" yields a CANDIDATE, never an established fact: domain
# enrichment may say `organisation_hint = "Lidl Magyarorszag"`, it may not say
# `legal_entity = ...`. `Research.hq_source` records which rung produced the
# headquarters, and this module reads it rather than trusting the value alone.
#
# IDENTITY UNCERTAINTY ALONE MUST NOT CAUSE REVIEW. `review` means a human has to
# decide something; unresolved identity is data uncertainty, and only becomes a
# decision when a *resolved* candidate interpretation could change the competitor,
# sanctions-entity or jurisdiction answer. Every resolved candidate is screened in its
# own right and the worst outcome wins (blocked > review > clear), so new evidence can
# only ever make the verdict stricter, never softer.
#
# Name-cleaning provenance (inspired by the OpenSanctions article on name cleaning,
# linked from the README): every hit carries `original_value` (the raw submitted
# company string) and `origin` (which layer produced it), and competitor abbreviations
# are matched *exactly*, never fuzzily - a 2-4 letter acronym is one edit away from a
# dozen unrelated ones, so fuzzy-matching it is noise, not signal.

if TYPE_CHECKING:
    from .provenance import ProvenanceRun

# Legal-form suffixes to strip before any name-similarity comparison (competitor
# prescreen AND providers/gleif.py's GLEIF-candidate name match both call
# `normalise_name`, so both share this one list). International, not just
# English/Hungarian: a legal entity's OWN registered name commonly carries its
# home jurisdiction's full company-type wording (e.g. GLEIF returned "MASTERPLAST
# Nyilvánosan működő Részvénytársaság" for a company everyone just calls
# "Masterplast" - the English-only suffix list scored that well below the name-
# similarity threshold). Longer/dotted forms are listed before the single-token
# alternation so a multi-word phrase or a "S.A."-style dotted abbreviation isn't
# left partially matched by a shorter alternative first (`re.sub` tries
# alternatives in the order given, at each position).
_LEGAL_SUFFIX = re.compile(
    r"(?:"
    r"gmbh\s*&\s*co\.?\s*kg"                                   # GmbH & Co. KG
    r"|nyilvánosan\s+működő\s+részvénytársaság"                # HU: Nyrt, public co.
    r"|zártkörűen\s+működő\s+részvénytársaság"                 # HU: Zrt, private co.
    r"|korlátolt\s+felelősségű\s+társaság"                     # HU: Kft, LLC
    r"|sp\.?\s*z\s*o\.?\s*o\.?"                                # PL: Sp. z o.o.
    r"|\bs\.\s*p\.\s*a\.?"                                     # IT: S.p.A.
    r"|\bs\.\s*a\.\s*s\.?"                                     # FR: S.A.S.
    r"|\bs\.\s*r\.\s*l\.?"                                     # IT: S.r.l.
    r"|\bs\.\s*r\.\s*o\.?"                                     # CZ/SK: s.r.o.
    r"|\bd\.\s*o\.\s*o\.?"                                     # SI/HR/RS/etc: d.o.o.
    r"|\bb\.\s*v\.?"                                           # NL: B.V.
    r"|\bn\.\s*v\.?"                                           # NL/BE: N.V.
    r"|\ba\.\s*s\.?"                                           # NO/CZ/SK: A.S. / a.s.
    r"|\ba/s"                                                  # DK: A/S
    r"|\bs\.\s*a\.?"                                           # ES/FR/etc: S.A.
    r"|\b(?:inc|incorporated|ltd|limited|llc|co|corp|corporation|company|holdings|group"
    r"|gmbh|kft|zrt|nyrt|bt|kkt|kg|plc|sa|ag|bv|nv|sas|sarl|oy|ab|as|asa|aps"
    r"|o[üu]|sia|uab)\b\.?"
    r")", re.I)


def _hq_established(hq_country: str) -> bool:
    return (hq_country or "").strip().lower() not in ("", "unknown")


# How much weight the source of a jurisdiction fact can carry. The waterfall that
# produces `hq_source` lives in `research._resolve_headquarters`; this is what a
# compliance decision is allowed to conclude from each rung.
_HQ_AUTHORITY = {
    "gleif": "registry",        # government registry legal address, structured
    "wikidata": "enrichment",   # accepted third-party enrichment, website-matched
    "llm": "inference",         # a READING of the sources, not a source
    "website_tld": "hint",      # the market a site is positioned for, not an address
}


@dataclass(frozen=True)
class JurisdictionFact:
    """A headquarters claim together with what it is worth for a compliance decision."""
    value: str
    source: str
    authority: str
    screenable: bool
    why: str


def jurisdiction_fact(research) -> JurisdictionFact:
    """Whether the headquarters on file can close a compliance decision - either way.

    One predicate for both directions, which is the point. The block side already had
    an authority rule of its own (registry-backed, or the country named in the raw
    evidence) while the clear side had none at all: it asked only whether the country
    string was non-empty. So the same fact that was too weak to block was strong enough
    to clear. Measured: a `.hu` ccTLD guess and an LLM-asserted "Germany" with
    `evidence_ids: []` each produced a final status of `clear`, while the block side
    correctly refused to act on either.

    Screenable requires all three: the country is known, its source carries enough
    authority, and where the source is an inference, the raw evidence actually names
    that country. Anything else leaves the jurisdiction UNRESOLVED - which is not a
    finding about the company, and is for the decision-sensitivity policy to act on.
    """
    value = (research.headquarters_country or "").strip()
    source = (research.hq_source or "").strip()
    if not _hq_established(value):
        return JurisdictionFact(value, source, "none", False,
                                "no headquarters country was established")
    authority = _HQ_AUTHORITY.get(source, "hint" if source else "none")
    if (research.hq_confidence or "").strip().lower() == "low":
        return JurisdictionFact(value, source, authority, False,
                                f"the headquarters country '{value}' is recorded at low "
                                f"confidence (source: {source or 'none'})")
    if authority in ("registry", "enrichment"):
        return JurisdictionFact(value, source, authority, True,
                                f"headquarters '{value}' from {source}")
    if authority == "inference":
        if _names_country_in_evidence(value, research):
            return JurisdictionFact(value, source, authority, True,
                                    f"headquarters '{value}' extracted from evidence that names it")
        return JurisdictionFact(value, source, authority, False,
                                f"the research model reported headquarters '{value}', but no "
                                f"accepted source names that country")
    return JurisdictionFact(value, source, authority, False,
                            f"headquarters '{value}' rests only on {source or 'no source'}, "
                            f"which is not evidence of where a legal entity sits")


def _names_country_in_evidence(country: str, research) -> bool:
    """True when the raw evidence text itself names `country`.

    The bar for letting an LLM-extracted headquarters reach the deterministic block:
    the model may read a country out of the sources, but the sources have to contain
    it. Same conservative substring match as `_mentions_restricted_marker`, over the
    same raw texts, and deliberately not over the model's own summary of them - a
    summary that names the country only because the model put it there would make the
    check circular. Wikipedia's extract counts: it is an accepted source here, not the
    model's own output.
    """
    combined = " ".join(
        t or "" for t in (research.website_text, research.wikipedia_summary)
    ).lower()
    return country.lower() in combined


def _mentions_restricted_marker(*texts: str, profile: OrgProfile | None = None) -> str | None:
    """Deterministic (never LLM-decided) scan of raw evidence text for a listed
    sanctioned-jurisdiction name or one of its capital/major-city markers
    (`config/restricted_jurisdictions.yaml`'s `markers`, e.g. "Tehran" for Iran).
    Case-insensitive substring match on purpose: this is a conservative trip-wire,
    not a full entity extraction - it can over-fire on an unrelated mention of a
    city name, which is the safe direction to be wrong in for a compliance gate."""
    combined = " ".join(t or "" for t in texts).lower()
    cfg = compliance_config(profile)
    for name in (*cfg.jurisdictions, *cfg.markers):
        if name.lower() in combined:
            return name
    return None


def normalise_name(name: str) -> str:
    n = _LEGAL_SUFFIX.sub(" ", name.lower())
    n = re.sub(r"[^a-z0-9]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def _auto_abbreviation(name: str) -> str:
    """Acronym from the non-suffix tokens, camelCase-aware.

    "CloudTrim Inc" -> strip suffix -> "CloudTrim" -> tokens Cloud/Trim -> "CT".
    "SpendWise Cloud" -> tokens Spend/Wise/Cloud -> "SWC".
    "RightSize Cloud Co" -> strip suffix -> "RightSize Cloud" -> tokens
    Right/Size/Cloud -> "RSC".
    """
    stripped = _LEGAL_SUFFIX.sub(" ", name)
    tokens = re.findall(r"[A-Z][a-z0-9]*", stripped) or stripped.split()
    return "".join(t[0] for t in tokens if t).upper()


def _competitor_list(cfg: ComplianceConfig) -> list[dict]:
    """name + auto-derived abbreviation for each competitor the profile lists."""
    return [{"name": c, "abbreviation": _auto_abbreviation(c)} for c in cfg.competitors]


# Snapshot for the active profile (tests and callers that predate profiles import this).
_COMPETITORS = _competitor_list(compliance_config())

# --- One policy, one number ------------------------------------------------------
# These thresholds are the competitor policy. The LLM prompt below is BUILT from them
# rather than restating them, because it used to restate them wrongly: it told the
# model "a pre-screen score of 90 or more IS a match (blocked)" while the
# deterministic floors fired at 95 and only ever escalated to `review`. Two
# definitions of one policy, and the weaker surface was the one deciding: a name
# scoring 92 was blocked or cleared depending on how the model felt that run, with no
# floor underneath it either way. A threshold that appears twice will eventually
# disagree with itself, so it now appears once.
NAME_MATCH_FLOOR = 95      # fuzzy/exact COMPANY NAME: can never be silently cleared
DOMAIN_MATCH_FLOOR = 90    # submitted WEBSITE/E-MAIL domain label: a registered string,
                           # not a fuzzy rendering of a name, so it earns a lower floor
CANDIDATE_MIN_SCORE = 85   # below the floors: collected as a candidate for the model
                           # to weigh and reported in the output, never decisive itself
WEAK_NAME_SCORE_CAP = 89   # a generic/trivial name cannot earn a floor-reaching score
                           # however it fuzzy-matches (see _is_weak_name)


_SANCTIONED_TLDS = {"ir": "Iran", "cu": "Cuba", "kp": "North Korea", "sy": "Syria", "ru": "Russia", "by": "Belarus"}

# Business words generic enough that matching one proves nothing about identity -
# same concept as sanctions.py's weak-query guard. The list is the profile's
# `compliance.generic_name_tokens`.


def _is_weak_name(name_norm: str, generic_tokens: frozenset[str] | None = None) -> bool:
    """A weak company name can't earn a high-confidence identity match on its own.

    Caught by a Hypothesis property test: `token_set_ratio`/`partial_ratio` both
    score a single generic word (e.g. "cloud") at 100 against any competitor whose
    name contains that word, because a one-token query is trivially a subset/substring
    of the longer name. That is a fuzzy-matching artifact, not evidence of identity -
    so a weak query's score is capped below the "confident match" threshold (see
    `prescreen` below), the same way sanctions.py caps a weak query's ability to
    reach `blocked_evidence`.
    """
    tokens = name_norm.split()
    if not tokens:
        return True
    if generic_tokens is None:
        generic_tokens = compliance_config().generic_name_tokens
    if all(t in generic_tokens for t in tokens):
        return True
    if max(len(t) for t in tokens) <= 3:
        return True
    return False


def _competitor_hits(identity: str, original_value: str, origin: str, label: str,
                     *, check_abbreviation: bool, min_score: int = 70,
                     cfg: ComplianceConfig | None = None, competitors: list[dict] | None = None) -> list[dict]:
    """Fuzzy competitor hits for ONE identity string - a company name, or a domain
    label (see `_domain_identities`). `label` names the surface inside the hit's
    reason, so a rep reading the flag sees WHICH field matched."""
    hits: list[dict] = []
    cfg = cfg or compliance_config()
    competitors = _competitor_list(cfg) if competitors is None else competitors
    weak_query = _is_weak_name(identity, cfg.generic_name_tokens)
    for comp in competitors:
        comp_norm = normalise_name(comp["name"])
        # token_set_ratio is robust to word order and dropped legal suffixes;
        # partial_ratio catches a competitor's name embedded in a longer one.
        ts = fuzz.token_set_ratio(identity, comp_norm)
        pr = fuzz.partial_ratio(identity, comp_norm)
        score = max(ts, pr)
        if weak_query:
            # A generic/trivial query name can't earn high-confidence identity, no
            # matter how it scores - capped below every deterministic floor.
            score = min(score, WEAK_NAME_SCORE_CAP)
        if score >= min_score:
            hits.append({
                "kind": "competitor", "term": comp["name"], "score": int(score),
                "reason": f"{label} similarity {int(score)} (token_set={int(ts)}, partial={int(pr)})",
                "original_value": original_value, "origin": origin,
            })
        elif check_abbreviation and identity == comp["abbreviation"].lower():
            # Exact only, deliberately: fuzzy-matching a 2-4 letter acronym against
            # other acronyms produces near-random hits (the IKEA/IAEA problem).
            hits.append({
                "kind": "competitor", "term": comp["name"], "score": 80,
                "reason": f"exact match on abbreviation '{comp['abbreviation']}'",
                "original_value": original_value, "origin": "prescreen:abbreviation",
            })
    return hits


def _domain_identities(lead: Lead) -> list[tuple[str, str]]:
    """`(identity, registrable domain)` pairs for the submitted website and e-mail.

    A competitor does not have to submit its own company name. In the adversarial
    battery (2026-09-20, cases e/h) the lead arrived as company "Unimedia Technology"
    with website `cloud-trim.com` - Cloud-Trim being that company's own AWS
    cost-optimization product, i.e. the listed competitor under a different legal
    name. The name-only pre-screen produced no candidate at all, and the LLM layer,
    which IS shown the website URL and even summarised the page correctly as "a free
    tool focused on AWS cost optimization", still answered "clear". A domain is a
    second identity surface, independent of the name the submitter chose to type, so
    it is screened deterministically here instead of being left to the model to
    notice - the same reasoning as safety net 4's.

    Both a hyphen-stripped and a hyphen-as-space reading are produced, because which
    one recovers the brand differs per competitor: `cloud-trim` -> "cloudtrim" scores
    100 against "CloudTrim Inc", while `spendwise-cloud` -> "spendwise cloud" scores
    100 against "SpendWise Cloud" (both measured; see tests).
    """
    out: list[tuple[str, str]] = []
    email_domain = lead.email.rsplit("@", 1)[-1] if "@" in (lead.email or "") else ""
    for raw in (lead.website or "", email_domain):
        rd = registrable_domain(raw)
        if not rd or "." not in rd:
            continue
        label = rd.rsplit(".", 1)[0]
        for variant in dict.fromkeys((label.replace("-", ""), label.replace("-", " "))):
            identity = normalise_name(variant)
            if identity:
                out.append((identity, rd))
    return out


def prescreen(lead: Lead, research: Research, profile: OrgProfile | None = None) -> list[dict]:
    """Fuzzy full-name and exact abbreviation hits against competitors, the same fuzzy
    check against the submitted website/e-mail domains, plus sanctioned-jurisdiction
    hits from the extracted HQ field and the website TLD.
    Returns candidate matches only - this layer never clears a lead by itself.
    """
    cfg = compliance_config(profile)
    competitors = _competitor_list(cfg)
    name = normalise_name(lead.company)
    hits: list[dict] = _competitor_hits(name, lead.company, "prescreen:rapidfuzz", "name",
                                        check_abbreviation=True, cfg=cfg, competitors=competitors)
    # Strongest hit per (competitor, domain) only: the two readings of one domain are
    # two views of the same surface, not two independent pieces of evidence.
    best: dict[tuple[str, str], dict] = {}
    for identity, rd in _domain_identities(lead):
        # Higher candidate floor than the name path's 70: a domain label carries no
        # legal suffix and no word order, so a genuine brand match scores very high
        # (measured: 100 for `cloud-trim` vs "CloudTrim Inc"), while the mid-70s band
        # is pure generic-token overlap - `cloudtrim` scores 71 against BOTH other
        # competitors purely because all three contain the word "cloud".
        for h in _competitor_hits(identity, rd, "prescreen:domain", f"domain '{rd}'",
                                  check_abbreviation=False, min_score=CANDIDATE_MIN_SCORE,
                                  cfg=cfg, competitors=competitors):
            key = (h["term"], rd)
            if h["score"] > best.get(key, {"score": -1})["score"]:
                best[key] = h
    hits.extend(best.values())

    # A "low"-confidence HQ (research.py's ccTLD-derived fallback, item 4 - see
    # models.Research.hq_confidence) is established enough to satisfy the
    # "clear requires HQ" safety net below, but is NOT independently strong enough
    # to drive a full-confidence (score 100) sanctions hit on its own: it's the
    # exact same weak ccTLD signal the `prescreen:tld` check right below already
    # scores at 80, so treating it as an equally-confirmed `prescreen:hq` hit at
    # 100 would double-count one weak signal as if it were two independent ones.
    #
    # An LLM-extracted HQ is a READING of the sources, not a source. It may only reach
    # the automatic-block floor when the raw evidence text itself names that country,
    # because otherwise a single hallucinated country name would be enough to block a
    # company deterministically, with no human in the loop - a worse failure than
    # sending a genuinely restricted lead to review. Unsupported claims still get the
    # `prescreen:hq_unsupported` origin, which safety net 1 escalates to review (it
    # keys on the score, not the origin) while safety net 0 ignores it (it keys on the
    # origin). Corroboration is deliberately the country NAME in the evidence text: the
    # marker list in config/restricted_jurisdictions.yaml is flat, with no country
    # association, so a country named only through one of its cities ("Tehran" but never
    # "Iran") corroborates nothing here and the lead goes to review instead of blocked -
    # the safe direction, and safety net 4 reads those markers separately anyway.
    hq = (research.headquarters_country or "").strip().lower()
    if research.hq_confidence != "low":
        for country in cfg.jurisdictions:
            if country.lower() == hq:
                # The SAME predicate that decides whether this fact may clear a lead.
                # These were two separate rules and they disagreed: a fact too weak to
                # block here was still strong enough to clear in safety net 4.
                corroborated = jurisdiction_fact(research).screenable
                hits.append({
                    "kind": "sanctions", "term": country, "score": 100,
                    "reason": (f"extracted HQ is '{country}'" if corroborated else
                               f"the research model reported HQ '{country}', but no accepted "
                               f"source names that country - unresolved, not established"),
                    "original_value": lead.company,
                    "origin": "prescreen:hq" if corroborated else "prescreen:hq_unsupported",
                })

    # WHICH legal entity applied has to be settled before WHERE it is based. Lidl
    # (2026-09-20): the lead arrived as company "Lidl" from `lidl.hu`, and the screen
    # cleared it on "Germany is not a sanctioned jurisdiction" - a country the LLM had
    # inferred for the global parent, with GLEIF matching none of its 15 candidates.
    # Both countries were unsanctioned so the verdict was harmless; the same path
    # states a parent's country for a subsidiary where it is not.
    #
    # The first version of this check compared the website's ccTLD against the
    # extracted HQ. That was the wrong model: a ccTLD marks the market a site is
    # positioned for - IANA assigns those codes to countries and territories, not to
    # company headquarters - so `.hu` is no evidence about where a legal entity sits.
    # The signal used instead is direct: the organisation the SUBMITTED DOMAIN belongs
    # to, named independently of the typed company name (measured: `lidl.hu` ->
    # "Lidl Magyarorszag"). When that is a more specific entity than the lead applied
    # under, and the headquarters on file rests on inference rather than a registry,
    # the facts describe one entity while a different one applied.
    # Which identities exist is a separate question from whether the HQ on file is
    # solid, and folding them together let a candidate escape screening entirely. Two
    # conditions used to gate this:
    #
    #   `typed in resolved`  - the candidate only counted when the typed name was a
    #       SUBSTRING of the resolved one, i.e. only when it was the "more specific"
    #       version of the same name. Measured: company "Acme" with the domain resolving
    #       to "Rosneft" produced no candidate at all, and the sanctions stub - which
    #       blocked Rosneft - was called once, with "Acme". A materially different
    #       identity is exactly the case that most needs screening, and it was the one
    #       case the condition excluded.
    #
    #   `hq_source in (...)` - no candidate when the TYPED entity had a registry HQ. But
    #       a registry record for the typed company says nothing about which entity the
    #       submitted domain belongs to; that is a fact about a different question.
    #
    # What remains is the material one: the domain resolves to a name that is not the
    # name applied under. It is a candidate interpretation to screen in its own right,
    # never an established legal entity.
    entity = (research.domain_entity_name or "").strip()
    if entity:
        typed, resolved = normalise_name(lead.company), normalise_name(entity)
        if typed and resolved and resolved != typed:
            hits.append({
                "kind": "entity_scope", "term": entity, "score": 0,
                "reason": (f"domain-level enrichment associates the submitted site with '{entity}', "
                           f"a different identity from the '{lead.company}' this applied under "
                           f"(headquarters on file: '{research.headquarters_country or 'unknown'}', "
                           f"source: {research.hq_source or 'none'}) - a second candidate "
                           f"interpretation to screen, not an established legal entity"),
                "original_value": lead.website, "origin": "prescreen:entity_scope",
            })

    tld = lead.website.rstrip("/").rsplit(".", 1)[-1].lower()
    if tld in _SANCTIONED_TLDS:
        hits.append({
            "kind": "sanctions", "term": _SANCTIONED_TLDS[tld], "score": 80,
            "reason": f"website TLD .{tld}",
            "original_value": lead.company, "origin": "prescreen:tld",
        })
    return hits


_SYSTEM_TEMPLATE = ("""You are a compliance screening agent for a B2B sales team. Decide whether a lead may be
engaged. Return STRICT JSON: {"status": "clear"|"review"|"blocked", "flagged": bool,
"matches": [{"kind": "competitor"|"sanctions", "term": str, "score": 0-100, "reason": str}], "reasoning": str}.

Rules:
- "blocked": the company is, or is very likely, one of the listed competitors (including renamed,
  re-suffixed, abbreviated or misspelled forms), OR it is headquartered in a listed sanctioned jurisdiction.
- "review": a plausible partial match (shared distinctive tokens, a subsidiary/brand relationship,
  an office in a sanctioned country but HQ elsewhere, conflicting evidence). Explain what a human should check.
- "clear": no match or only a coincidental generic-word overlap__GENERIC_EXAMPLE__.
- Generic words (__GENERIC_WORDS__) never justify a match by themselves.
- Pre-screen scores are CANDIDATES. The deterministic layer that runs after you has exactly two
  floors: a company-NAME score of __NAME_FLOOR__ or more, and a submitted WEBSITE/E-MAIL DOMAIN score
  of __DOMAIN_FLOOR__ or more. Either one forces at least "review" whatever you answer; below them a
  score decides nothing on its own. Treat a name at __NAME_FLOOR__+ or a domain at __DOMAIN_FLOOR__+ as
  a match unless the evidence positively shows a different company (different industry, different
  country, different product) - a missing website is not evidence of a different company. Reserve
  "blocked" for a match the evidence supports: a domain match, or a name match with corroboration. A
  name on its own is a candidate identity, never a confirmed one.
- Pre-screen hits carry an `origin`: `prescreen:domain` means the submitted WEBSITE or E-MAIL domain
  matched a competitor, not the typed company name. A different legal name is NOT evidence of a
  different company when the domain matches: an operator submitting its own competing product's
  domain is exactly the case this catches, so treat it as at least "review" and explain the
  name-versus-domain relationship a human should verify.
__SUBSTANCE_LINE__- An `origin: prescreen:hq_conflict` hit means the submitted domain's country and the extracted
  headquarters disagree, i.e. it is unclear whether the applicant is a local subsidiary or the
  global parent. Name BOTH entities and say which one a human should confirm; do not resolve it
  by picking the parent's country. State the jurisdiction you actually screened.
- An exact abbreviation match alone (e.g. initials matching a competitor) is at most "review" -
  initials are ambiguous on their own; look for other supporting evidence before calling it "blocked".
- The OpenSanctions API hits, when present, are real watchlist/registry data. Weigh their `topics`,
  not their raw `match` flag: sanction/export.control/debarment/crime topics are the actual concern;
  corp.public/reg.action/fin.*/role.*/poi topics are routine registry facts, not sanctions.
- Base sanctions reasoning on headquarters country, not on where customers or a single office are.
- If the research summary or evidence reports CONFLICTING or disputed headquarters/registration
  claims (e.g. a marketing page names one country while a legal/footer/registration statement
  names another) and at least one of the candidate countries is a listed sanctioned jurisdiction,
  that is at minimum "review", never "clear" - do not silently resolve the conflict in the
  company's favor and clear it; say what a human should verify.
- flagged is true for "review" and "blocked".
Keep reasoning to 2-4 sentences, concrete, referring to the evidence given.

""" + INJECTION_WARNING)


def _build_system(cfg: ComplianceConfig) -> str:
    """The compliance prompt for one profile. The vendor-specific parts (what counts as a
    competitor "in substance", the generic-word example) come from the profile; with an
    empty `competitor_offering` the in-substance line is omitted altogether."""
    substance = (
        f"- A company whose own product or service IS {cfg.competitor_offering} is a competitor in "
        "substance,\n  whatever it is called; say so rather than clearing it because the name differs "
        "from the list.\n") if cfg.competitor_offering else ""
    example = (f' (e.g. the word "{cfg.generic_name_example}" alone is not a match)'
               if cfg.generic_name_example else "")
    return (_SYSTEM_TEMPLATE
            .replace("__NAME_FLOOR__", str(NAME_MATCH_FLOOR))
            .replace("__DOMAIN_FLOOR__", str(DOMAIN_MATCH_FLOOR))
            .replace("__GENERIC_EXAMPLE__", example)
            .replace("__GENERIC_WORDS__", cfg.generic_words_hint)
            .replace("__SUBSTANCE_LINE__", substance))


_SYSTEM = _build_system(compliance_config())


def screen(lead: Lead, research: Research, run: ProvenanceRun | None = None,
           profile: OrgProfile | None = None) -> ComplianceResult:
    profile = profile or active_profile()
    cfg = compliance_config(profile)
    hits = prescreen(lead, research, profile)
    sanctions = screen_sanctions(lead.company, research.headquarters_country, run)
    sanctions_hits = [dict(h, original_value=lead.company) for h in sanctions.hits]

    user = (
        f"Lead company: {lead.company}\nWebsite: {lead.website}\n"
        f"Extracted HQ country: {research.headquarters_country or 'unknown'}\n"
        f"Industry: {research.industry}\n"
        f"{wrap_evidence('research-summary', 'research:summary', research.summary)}\n\n"
        f"Do-not-engage competitors: {', '.join(cfg.competitors)}\n"
        f"Sanctioned jurisdictions (illustrative): {', '.join(cfg.jurisdictions)}\n\n"
        f"Fuzzy/abbreviation pre-screen hits (candidates, not verdicts): {hits or 'none'}\n"
        f"OpenSanctions API hits (real watchlist/registry data, weigh topics not match): "
        f"{sanctions_hits or f'none ({sanctions.status})'}\n"
    )
    # The deterministic floors below are the safety layer, so they have to run even when
    # the model does not. Before this, a hard LLM failure (no credentials, every hop in
    # the chain down, an unparseable body twice) propagated out of `screen` and the floors
    # never executed at all: measured, a lead with an Iran headquarters raised LLMError
    # instead of returning "blocked", and in `cli.py batch` that aborted the remaining
    # leads as well. An unavailable model is an unavailable channel, not a clearance, so
    # the verdict falls back to ComplianceVerdict()'s own fail-closed defaults (review,
    # flagged, no matches) and the floors can still escalate it to blocked.
    try:
        verdict = ask_model(_build_system(cfg), user, ComplianceVerdict, purpose="compliance")
        model_error = ""
    except LLMError as e:
        verdict = ComplianceVerdict()
        model_error = f"{type(e).__name__}: {e}"
    status = verdict.status
    matches = []
    for m in verdict.matches:
        m = dict(m)
        m.setdefault("original_value", lead.company)
        m.setdefault("origin", "llm:reasoning")
        matches.append(m)

    # Every deterministic floor that fires records WHY, so the explanation the rep
    # reads is generated from the decision that was actually made. Without this the
    # model's original text survived a floor that overruled it, and the status
    # contradicted its own reasoning (Lidl, 2026-09-20: status "review", explanation
    # "only flags a domain-country mismatch").
    floors: list[str] = []
    notes: list[str] = []  # recorded context that did NOT change the decision
    if model_error:
        floors.append("The compliance model could not be reached, so this lead was not "
                      "assessed by it and cannot be cleared on its say-so; the deterministic "
                      f"checks below decided the outcome ({model_error}).")
    # Safety net 0: an established headquarters in a configured restricted jurisdiction
    # is the core do-not-engage rule, so it is decided here and not left to the
    # model. Only a hit the pre-screen raised at full confidence counts - the ccTLD
    # rung deliberately never reaches this, being scored at 80 (see `prescreen`).
    _restricted = [h for h in hits if h.get("origin") == "prescreen:hq"
                   and h["score"] >= NAME_MATCH_FLOOR]
    if _restricted:
        status = "blocked"
        matches = matches or _restricted
        floors.append(f"Headquarters is in {_restricted[0]['term']}, a restricted jurisdiction "
                      f"on the do-not-engage list.")
    # Safety net 1: a 95+ fuzzy prescreen hit can never be silently cleared. The reason
    # has to name the surface that actually matched: this fired on a sanctioned-HQ hit
    # (score 100) and explained it as "a near-exact name match to a competitor", which
    # is a false statement to the person reading it (found 2026-09-20 by running the
    # audit's own repro).
    _confident = [h for h in hits if h["score"] >= NAME_MATCH_FLOOR]
    if status == "clear" and _confident:
        status = "review"
        matches = matches or hits
        kinds = {h.get("kind") for h in _confident}
        what = ("a do-not-engage competitor" if kinds == {"competitor"}
                else "a restricted jurisdiction" if kinds == {"sanctions"}
                else "a do-not-engage list entry")
        floors.append(f"A near-exact pre-screen match to {what} was found; a human must "
                      f"confirm it before any outreach.")
    # Safety net 2: an exact abbreviation hit is, at minimum, a review candidate -
    # initials alone shouldn't clear on the model's say-so.
    if status == "clear" and any(h.get("origin") == "prescreen:abbreviation" for h in hits):
        status = "review"
        matches = matches or [h for h in hits if h.get("origin") == "prescreen:abbreviation"]
        floors.append("The company name matches a competitor's abbreviation exactly, which "
                      "initials alone cannot settle either way.")
    # Safety net 2b: a confident competitor hit on the submitted domain can never be
    # silently cleared either. Net 1's threshold is 95 because a NAME is fuzzy by
    # nature (legal forms, word order); a domain label is the string the submitter
    # actually registered, so 90 is the right floor here - and the case that prompted
    # this (`cloud-trim.com` submitted under the name "Unimedia Technology") scored
    # 100 while the LLM still answered "clear".
    if status == "clear" and any(
            h.get("origin") == "prescreen:domain" and h["score"] >= DOMAIN_MATCH_FLOOR for h in hits):
        status = "review"
        matches = matches or [h for h in hits if h.get("origin") == "prescreen:domain"]
        floors.append("The submitted website or e-mail domain matches a do-not-engage "
                      "competitor, whatever name the lead was filed under.")
    # Safety net 3: real OpenSanctions evidence overrides the LLM either way.
    if sanctions.status == "blocked_evidence":
        status = "blocked"
        floors.append("A watchlist record matched this company; screening evidence, not "
                      "an inference, decides this outcome.")
    elif sanctions.status == "review_evidence" and status == "clear":
        status = "review"
        floors.append("A watchlist search returned a record that needs human assessment.")

    # Safety net 4 (Part 2b, 2026-09-19): "clear" requires an AFFIRMATIVELY
    # established headquarters country - an unknown/empty/unresolved HQ is never
    # enough evidence to clear a lead on its own, and if the raw evidence text also
    # names a restricted jurisdiction or one of its capital/major-city markers
    # (e.g. a legal footer naming "Tehran" while the marketing copy claims a
    # different, unsanctioned HQ - adv-03), that combination is a stronger, more
    # specific reason for the same "review" floor. Deterministic, never LLM-decided:
    # the adversarial eval showed a prompt-only fix cannot be trusted for a safety
    # rule (the LLM can still be talked into "clear" by evidence-embedded
    # instructions; this check reads the raw evidence text directly and cannot be).
    # Safety net 4b: identity ambiguity is screened, not escalated. An unresolved
    # "which entity applied" question is DATA uncertainty; `review` should mean
    # DECISION uncertainty, or the queue fills with rows a human cannot act on. So the
    # second candidate identity is screened in its own right, and the ambiguity only
    # forces review when a resolved candidate interpretation could actually change the
    # answer. "Resolved", deliberately: the pipeline can evaluate the interpretations
    # it found evidence for, and cannot claim to have enumerated every legal entity.
    entity_hits = [h for h in hits if h.get("origin") == "prescreen:entity_scope"]
    if entity_hits:
        alt = str(entity_hits[0]["term"])
        alt_competitor = [h for h in _competitor_hits(normalise_name(alt), alt,
                                                      "prescreen:entity_scope_competitor",
                                                      f"candidate identity '{alt}'",
                                                      check_abbreviation=False, min_score=CANDIDATE_MIN_SCORE,
                                                      cfg=cfg)]
        # Screened on its OWN jurisdiction, which here is simply not known. Passing the
        # typed lead's `headquarters_country` was the contamination this check exists to
        # prevent, in miniature: the second candidate is a different legal entity, so
        # filtering its watchlist query by the first one's country can suppress a real
        # hit. No country means no country filter - the wider, more conservative query.
        alt_sanctions = screen_sanctions(alt, None, run)
        # Worst case wins, and evidence never softens a finding already made.
        if alt_sanctions.status == "blocked_evidence" or any(h["score"] >= NAME_MATCH_FLOOR for h in alt_competitor):
            status = "blocked"
            matches = matches + alt_competitor
            floors.append(f"The second candidate identity for this domain ('{alt}') is itself a "
                          f"do-not-engage match.")
        elif alt_sanctions.status == "review_evidence" or alt_competitor:
            status = "review" if status != "blocked" else status
            matches = matches + alt_competitor
            floors.append(f"Screening the second candidate identity for this domain ('{alt}') "
                          f"returned a record that needs human assessment.")
        elif status == "clear":
            # Every resolved interpretation screens the same way, so which one the lead
            # meant cannot change the outcome. Recorded, not escalated.
            notes.append(f"Domain-level enrichment associates the submitted site with '{alt}', "
                         f"while company-level evidence refers to a broader group. Each resolved "
                         f"candidate identity was screened separately and none is a competitor or "
                         f"a sanctions match, so this identity ambiguity does not affect the "
                         f"outcome.")
    _jurisdiction = jurisdiction_fact(research)
    if status == "clear" and not _jurisdiction.screenable:
        marker = _mentions_restricted_marker(research.website_text, research.summary, research.wikipedia_summary,
                                             profile=profile)
        reason = (f"{_jurisdiction.why}, and a restricted-jurisdiction marker is present in evidence"
                  if marker else _jurisdiction.why)
        status = "review"
        matches = matches or [{
            "kind": "sanctions", "term": marker or "unknown", "score": 100 if marker else 0,
            "reason": reason, "original_value": lead.company, "origin": "safety_net:hq_unknown",
        }]
        floors.append("A lead is never cleared without an affirmatively established "
                      f"headquarters country ({reason}).")

    # Safety net 5: a control that FAILED is not a control that found nothing. A network
    # error, a non-200, a 200 whose body could not be read and a cached answer too old to
    # stand for the present all arrive here as "skipped", and "skipped" passed through
    # this function untouched, so any of them cleared the lead as long as the model said
    # clear. That is the negative-evidence fallacy with the provider's failure standing
    # in for the provider's answer.
    #
    # Not configured is a different state and is deliberately not treated the same way.
    # OpenSanctions entity screening is an optional extra capability; the core
    # do-not-engage rule is the restricted-jurisdiction check in safety net 0, which is
    # deterministic and needs no API key. Forcing every lead to review because an
    # optional control was never installed would turn a first run into a
    # queue of rows nobody can act on, and would say "we tried and could not tell" about
    # a control that was never claimed. It is recorded instead, so the absence stays
    # visible in the result rather than being inferred from silence.
    if status == "clear" and sanctions.status == "skipped" and sanctions.reason == "not_configured":
        notes.append("Watchlist entity screening is not configured for this run "
                     f"({sanctions.note}), so this decision rests on the deterministic "
                     f"restricted-jurisdiction and do-not-engage checks alone.")
    elif status == "clear" and sanctions.status == "skipped":
        status = "review"
        matches = matches or [{
            "kind": "sanctions", "term": "unknown", "score": 0,
            "reason": f"sanctions screening did not complete: {sanctions.note or 'no reason recorded'}",
            "original_value": lead.company, "origin": "safety_net:sanctions_not_screened",
        }]
        floors.append("Watchlist screening did not complete for this lead, so there is no "
                      "screening result to clear it on "
                      f"({sanctions.note or 'no reason recorded'}).")

    # The explanation follows the decision. When a deterministic rule overruled the
    # model, its reasoning is no longer the reason for the outcome, so it is reported
    # as what it now is - a secondary assessment that did not decide anything.
    model_reasoning = verdict.reasoning.strip()
    if floors:
        reasoning = " ".join(floors)
        if model_reasoning:
            reasoning += f" (Model assessment, which did not decide this outcome: {model_reasoning})"
    else:
        reasoning = model_reasoning
    # A clearance that rests on jurisdiction has to say how firmly that jurisdiction is
    # known. Left to itself the model writes "the HQ is confirmed as Germany" about a
    # country it inferred from a Wikipedia sentence (measured on the Lidl run), and
    # with no floor firing that sentence IS the rep-facing reasoning. The qualifier is
    # deterministic, so it cannot be phrased away.
    if status == "clear" and _hq_established(research.headquarters_country) \
            and research.hq_source in ("llm", "wikidata", "website_tld"):
        notes.append(f"Note: the headquarters used here ('{research.headquarters_country}') is "
                     f"inferred from research evidence, not established from a registry record.")
    if status == "clear":
        notes.append("Scope: no blocking or review-triggering issue was found by the configured "
                     "checks; this is not a KYC, AML or legal clearance.")
    if notes:
        reasoning = (reasoning + " " + " ".join(notes)).strip()

    return ComplianceResult(
        flagged=status != "clear",
        status=status,
        matches=matches,
        reasoning=reasoning,
        sanctions_status=sanctions.status,
        sanctions_hits=sanctions_hits,
    )
