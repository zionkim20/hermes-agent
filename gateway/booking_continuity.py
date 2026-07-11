"""Active booking-task continuity resolver (HUM-2199 / HUM-1918).

When a household has an in-flight external-vendor booking task, later
references like "that winery tour" must resolve against the *anchored* task
facts BEFORE any fresh web/search lookup, and a conflicting lookup result
must never silently overwrite the anchored vendor/offering/date/reservation
facts.

This module is deliberately transport- and gateway-agnostic: it holds only
pure functions over plain dicts so the behaviour is unit-testable without
booting the gateway or the LLM loop. Persistence lives in
``hermes_state.SessionDB`` (``*_active_booking_task`` / ``update_booking_status``);
the gateway wires ``build_preflight_note`` into the message path.

Origin: renata-family ticket HUM-1917 — Mia answered "that winery tour" with a
fresh lookup (Château Ferrand / Discovery Tour / Jul 4) instead of the anchored
task (Château La Dominique / Le Charme / 5 Jul / reservation.chateau-ladominique.com).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Allowed enum-ish values (kept lenient at the persistence layer; used by the
# resolver/tests and available to callers building task objects).
EVIDENCE_TYPES = (
    "voice_note",
    "screenshot",
    "vendor_reply",
    "link",
    "manual_user_statement",
)
BOOKING_STATUSES = (
    "not_booked",
    "held",
    "requested",
    "confirmed",
    "cancelled",
    "unknown",
)

# Deictic / continuation references that point at an already-discussed booking
# without naming the vendor. Matching any of these on inbound text means we
# should retrieve the anchored task before deciding to run a fresh lookup.
_DEICTIC_PATTERNS = [
    r"\bthat\s+(?:winery\s+|wine\s+|vineyard\s+)?tour\b",
    r"\bthe\s+(?:winery|vineyard|restaurant|venue|hotel|vendor)\b",
    r"\bthe\s+reservation\b",
    r"\bthe\s+booking\b",
    r"\bthat\s+booking\b",
    r"\bthat\s+reservation\b",
    r"\bthe\s+(?:tour|visit|tasting|table)\b",
    r"\bthe\s+thing\s+from\s+the\s+(?:screenshot|photo|voice\s?note|message)\b",
]

# Intents to continue / act on a previously-discussed vendor booking.
_INTENT_PATTERNS = [
    r"\bhelp\s+me\s+with\b",
    r"\b(?:continue|finish|complete)\b",
    r"\b(?:book|re-?book|confirm|reserve|pay|reply|respond)\b",
    r"\bcheck\s+(?:on\s+)?(?:availability|the\s+booking|the\s+reservation)\b",
    r"\bwhat(?:'s| is)\s+the\s+status\b",
]

# Explicit switch: user names a NEW vendor and asks to drop the current one.
# e.g. "actually ignore La Dominique and look for Château Ferrand on July 4".
_SWITCH_PATTERN = re.compile(
    r"\b(?:ignore|forget|drop|never\s*mind|cancel)\b.{0,60}?"
    r"\b(?:look(?:\s+for)?|search(?:\s+for)?|find|check|try)\b\s+"
    r"(?P<vendor>[A-Z0-9][\w'’.\- ]{2,60}?)"
    r"(?=\s+(?:on|for|instead|please|,|\.|$))",
    re.IGNORECASE,
)


def detect_continuity_signal(text: Optional[str], *, has_new_evidence: bool = False) -> bool:
    """True when inbound text (or a new evidence event) should trigger retrieval.

    ``has_new_evidence`` covers the "new screenshot/voice note arrived in a
    thread that already has an unresolved booking task" case, where the text
    itself may not carry a deictic reference.
    """
    if has_new_evidence:
        return True
    if not text:
        return False
    lowered = text.strip()
    if not lowered:
        return False
    for pat in _DEICTIC_PATTERNS:
        if re.search(pat, lowered, re.IGNORECASE):
            return True
    for pat in _INTENT_PATTERNS:
        if re.search(pat, lowered, re.IGNORECASE):
            return True
    return False


def detect_explicit_switch(text: Optional[str]) -> Optional[str]:
    """Return the new vendor name if the user explicitly asks to switch, else None.

    A match means the *user* named a different vendor and asked to drop the
    current one. The caller still requires explicit confirmation before
    superseding the anchored task — this only detects the request.
    """
    if not text:
        return None
    m = _SWITCH_PATTERN.search(text)
    if not m:
        return None
    vendor = (m.group("vendor") or "").strip(" .,'’")
    return vendor or None


# ── Inbound evidence extraction (write half of the continuity guard) ─────────
# Deliberately conservative: we only ever *anchor* (create/enrich) a booking
# task from a message that carries a high-signal fact — a named venue entity or
# a reservation link. A bare date or the word "tour" is not enough. False
# negatives (missing an anchor) are safe; a false positive that anchors noise is
# not, so the bar to write is intentionally high and the write is conflict-
# guarded downstream (``detect_conflict`` + enrich-not-blank upsert).

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec"
)
_DATE_DAY_MONTH = re.compile(
    rf"\b(\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS}))\b", re.IGNORECASE
)
_DATE_MONTH_DAY = re.compile(
    rf"\b((?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?)\b", re.IGNORECASE
)

# A venue-type keyword followed by 1–4 capitalised tokens ("Château La
# Dominique", "Hotel Belvedere"). Requiring the keyword keeps this from
# grabbing arbitrary capitalised words.
_VENDOR_ENTITY = re.compile(
    r"\b((?:Château|Chateau|Domaine|Maison|Hôtel|Hotel|Restaurant|Villa|Casa|"
    r"Vineyard|Winery|Estate|Club|Spa|Resort|Bistro|Brasserie|Trattoria)\s+"
    r"[A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){0,3})"
)

# 1–2 capitalised tokens immediately before an activity noun ("Le Charme tour",
# "Discovery Tour"). The leading tokens stay case-sensitive (the offering name
# must be capitalised), but the activity noun is matched case-insensitively via a
# scoped inline flag so common title-cased vendor phrasing ("Discovery Tour",
# "Prestige Tasting") is captured, not silently dropped. A global re.IGNORECASE
# would also relax the leading [A-Z] and let it grab lowercase junk ("the tour").
_OFFERING = re.compile(
    r"\b([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+)?)\s+"
    r"(?i:tour|tasting|experience|package|menu|visit|session|dinner|lunch)\b"
)

_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)


def _status_from_text(text: str) -> Optional[str]:
    """Best-effort booking-status read from inbound text; None if unclear."""
    low = text.lower()
    if re.search(r"\bcancel(?:led|ed|ing)?\b|\bcalled?\s+off\b", low):
        return "cancelled"
    if re.search(r"\bnot\s+(?:yet\s+)?booked\b|still\s+needs?\s+(?:to\s+be\s+)?book|"
                 r"needs\s+booking|isn'?t\s+booked", low):
        return "not_booked"
    if re.search(r"\bconfirmed\b|\ball\s+set\b|we'?re\s+booked\b|"
                 r"booking\s+(?:is\s+)?confirmed\b", low):
        return "confirmed"
    if re.search(r"\bon\s+hold\b|\bheld\b|\bholding\b", low):
        return "held"
    if re.search(r"\brequested\b|\breached\s+out\b|sent\s+(?:a\s+)?request", low):
        return "requested"
    return None


def extract_booking_evidence(
    text: Optional[str], *, has_media: bool = False
) -> Dict[str, Any]:
    """Pull anchorable booking facts out of one inbound message.

    Returns a dict of whitelisted ``active_booking_task`` fields. It is
    *anchorable* (safe to create/enrich a task from) only when it contains a
    ``vendor_entity`` or a ``reservation_url_or_contact``; otherwise the caller
    must not create a task from it. A bare status word (e.g. "we're confirmed
    now") is returned on its own so the caller can route it to a status update
    of an already-anchored task, but never to a fresh anchor.
    """
    text = text or ""
    facts: Dict[str, Any] = {}

    url_m = _URL.search(text)
    if url_m:
        facts["reservation_url_or_contact"] = url_m.group(0).rstrip(".,);]")

    vendor_m = _VENDOR_ENTITY.search(text)
    if vendor_m:
        facts["vendor_entity"] = vendor_m.group(1).strip()

    off_m = _OFFERING.search(text)
    if off_m:
        cand = off_m.group(1).strip()
        # Don't mistake the vendor's own tail for the offering.
        if cand and cand not in facts.get("vendor_entity", ""):
            facts["offering_name"] = cand

    date_m = _DATE_DAY_MONTH.search(text) or _DATE_MONTH_DAY.search(text)
    if date_m:
        facts["date"] = date_m.group(1).strip()

    status = _status_from_text(text)

    anchorable = bool(
        facts.get("vendor_entity") or facts.get("reservation_url_or_contact")
    )
    if not anchorable:
        # No entity/link → not enough to anchor. Surface only a status signal.
        return {"booking_status": status} if status else {}

    facts["booking_status"] = status or "not_booked"
    if has_media:
        facts["source_evidence_type"] = "screenshot"
    elif facts.get("reservation_url_or_contact"):
        facts["source_evidence_type"] = "vendor_reply"
    else:
        facts["source_evidence_type"] = "manual_user_statement"
    facts["source_evidence_summary"] = text.strip()[:280]
    return facts


def _norm(value: Optional[str]) -> str:
    """Normalise a fact for comparison: lowercase, collapse whitespace, strip
    a leading scheme/www so URLs compare on host+path."""
    if not value:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.rstrip("/")
    s = re.sub(r"\s+", " ", s)
    return s


# Fields whose disagreement between anchored task and a fresh lookup counts as
# a conflict that must not silently overwrite.
_CONFLICT_FIELDS = ("vendor_entity", "offering_name", "date", "reservation_url_or_contact")


@dataclass
class ConflictResult:
    conflict: bool
    fields: List[str] = field(default_factory=list)
    detail: Dict[str, Dict[str, str]] = field(default_factory=dict)


def detect_conflict(task: Dict[str, Any], lookup: Dict[str, Any]) -> ConflictResult:
    """Compare a fresh lookup result against anchored task facts.

    Only compares fields present (non-empty) on BOTH sides — a lookup that
    merely *adds* a missing fact (e.g. a phone number the task lacked) is
    enrichment, not a conflict.
    """
    fields_in_conflict: List[str] = []
    detail: Dict[str, Dict[str, str]] = {}
    for key in _CONFLICT_FIELDS:
        anchored = _norm(task.get(key))
        fresh = _norm(lookup.get(key))
        if anchored and fresh and anchored != fresh:
            fields_in_conflict.append(key)
            detail[key] = {"anchored": task.get(key), "lookup": lookup.get(key)}
    return ConflictResult(
        conflict=bool(fields_in_conflict),
        fields=fields_in_conflict,
        detail=detail,
    )


def _pretty_status(status: Optional[str]) -> str:
    return {
        "not_booked": "not booked yet",
        "held": "on hold",
        "requested": "requested (waiting on the vendor)",
        "confirmed": "confirmed",
        "cancelled": "cancelled",
        "unknown": "status unclear",
    }.get((status or "unknown"), "status unclear")


def build_anchored_facts_line(task: Dict[str, Any]) -> str:
    """One plain-language sentence naming the anchored facts, no internal IDs.

    Safe to surface to the household channel (HUM-500 sanitization class): it
    contains only the vendor/offering/date/link the user themselves supplied.
    """
    parts: List[str] = []
    vendor = task.get("vendor_entity")
    offering = task.get("offering_name")
    date = task.get("date")
    if vendor and offering:
        parts.append(f"{vendor} — {offering}")
    elif vendor:
        parts.append(str(vendor))
    elif offering:
        parts.append(str(offering))
    if date:
        parts.append(f"on {date}")
    party = task.get("party_size")
    if party:
        parts.append(f"for {party}")
    lead = " ".join(parts) if parts else "your booking"
    tail = []
    url = task.get("reservation_url_or_contact")
    if url:
        tail.append(f"Reservation link: {_norm(url)}")
    tail.append(f"It's {_pretty_status(task.get('booking_status'))}")
    return f"{lead}. " + ". ".join(tail) + "."


def build_preflight_note(task: Dict[str, Any]) -> str:
    """Build the system note prepended to the agent context before it decides
    whether to run web/search tools.

    Instructs the agent to use the anchored facts first and to apply the
    conflict guard — it never leaks internal issue IDs or logs.
    """
    facts = build_anchored_facts_line(task)
    return (
        "[Active booking task on file for this conversation] "
        f"{facts} "
        "Use these anchored facts first. Only run a fresh web/search lookup if "
        "you still need to complete this booking. If a lookup disagrees with the "
        "anchored vendor, tour/offering, date, or reservation link above, do NOT "
        "replace the anchored facts — state the anchored facts or ask one brief "
        "clarifying question. Switch to a different vendor only if the user "
        "explicitly confirms they want a different one."
    )
