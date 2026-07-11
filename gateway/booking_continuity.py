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
