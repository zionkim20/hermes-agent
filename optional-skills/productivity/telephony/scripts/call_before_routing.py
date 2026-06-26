"""Governed call-before-routing capability (HUM-1848).

A narrow, checklist-backed governance layer that sits ON TOP of the existing
telephony primitive (``telephony.py`` ``ai-call`` / ``ai-status`` via Bland.ai or
Vapi). It does NOT place calls itself and is NOT general phone autonomy. It
decides *whether* a vendor must be verified by phone before a household is routed
to a high-cost-of-failure stop, turns a raw provider call result into a
structured, sanitized record, and gates the household-facing recommendation so a
failed / no-answer call can never silently become a confident routing.

Scope (V1): restaurants / vendors / travel stops, especially during travel with
a young child, and especially European restaurants in June/July/August where
summer-vacation closures are common and Google Maps open-state is unreliable.

Design goals (see product brief 2026-06-24-mia-call-before-routing.md):
  - pure stdlib, no network, fully unit-testable
  - data minimization: the generated script and stored note never disclose
    child name, exact accommodation, personal schedule, or internal context
  - plain, non-technical household language
  - Maps/open-state alone is never sufficient in the Europe-summer scenario

The actual placement is delegated to ``telephony.py`` (``ai-call``) and the raw
status (``ai-status``) is fed back into :func:`build_call_result`. The telephony
provider/credential provisioning for a given household is an operational concern
owned by Infra, not by this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

EUROPE_SUMMER_MONTHS = frozenset({6, 7, 8})

# Lowercased markers used to recognise a European location from a free-text
# region/country string. Intentionally broad — when in doubt about Europe in
# summer we prefer to require verification rather than route on stale Maps data.
_EUROPE_MARKERS = frozenset(
    {
        "europe",
        "eu",
        "france",
        "french",
        "italy",
        "italian",
        "spain",
        "spanish",
        "portugal",
        "portuguese",
        "germany",
        "german",
        "greece",
        "greek",
        "netherlands",
        "dutch",
        "belgium",
        "austria",
        "switzerland",
        "swiss",
        "croatia",
        "croatian",
        "denmark",
        "sweden",
        "norway",
        "finland",
        "ireland",
        "poland",
        "czech",
        "hungary",
        "paris",
        "lyon",
        "nice",
        "rome",
        "milan",
        "florence",
        "venice",
        "barcelona",
        "madrid",
        "lisbon",
        "porto",
        "berlin",
        "munich",
        "amsterdam",
        "vienna",
        "athens",
        "santorini",
    }
)


class AnsweredBy(str, Enum):
    """Normalised answer status, independent of the upstream provider."""

    HUMAN = "human"
    RECORDING = "recording"
    VOICEMAIL = "voicemail"
    NO_ANSWER = "no_answer"
    FAILED = "failed"


class OpenStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNCERTAIN = "uncertain"


class ServiceStatus(str, Enum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class Confidence(str, Enum):
    VERIFIED = "verified"          # a human confirmed current service
    UNVERIFIED = "unverified"      # could not confirm; do not route confidently
    CONTRADICTED = "contradicted"  # live verification contradicts online data


# Only a live human answer can produce a confident, route-able verification.
_LIVE_HUMAN_ANSWERS = frozenset({AnsweredBy.HUMAN})


@dataclass
class CallContext:
    """Everything the policy needs to decide on, and to script, a verification call.

    Only ``vendor_name`` and ``phone_number`` are operationally required to place
    a call; the rest drive the policy decision and the Europe-summer rule.
    """

    vendor_name: str
    phone_number: str = ""
    region: str = ""                 # free-text region/country/city
    month: Optional[int] = None      # 1-12, local month of the planned stop
    is_traveling: bool = False
    child_present: bool = False
    timing_tight: bool = False
    online_open_state: Optional[str] = None   # what Maps/online currently claims
    online_data_conflicting: bool = False
    holiday_or_vacation_prone: bool = False
    mission_critical: bool = False
    # Fields that MUST NOT leak into the call script or stored note.
    sensitive_terms: tuple[str, ...] = ()

    def is_europe(self) -> bool:
        text = (self.region or "").lower()
        return any(marker in text for marker in _EUROPE_MARKERS)

    def is_europe_summer(self) -> bool:
        return self.is_europe() and self.month in EUROPE_SUMMER_MONTHS


def is_europe_summer_high_risk(ctx: CallContext) -> bool:
    """Europe in Jun/Jul/Aug => restaurants are high closure-risk by default.

    In this window Google Maps / online open-state alone is explicitly NOT
    sufficient; current service must be confirmed by live phone / direct site /
    social / door evidence before routing confidently.
    """

    return ctx.is_europe_summer()


def requires_verification(ctx: CallContext) -> tuple[bool, list[str]]:
    """Decide whether call-before-routing is required, with human-readable reasons."""

    reasons: list[str] = []

    if is_europe_summer_high_risk(ctx):
        reasons.append(
            "European restaurant in summer (Jun-Aug): summer-vacation closures are "
            "common and online open-state is unreliable."
        )
    if ctx.online_data_conflicting:
        reasons.append("Online availability data is contradictory.")
    if ctx.holiday_or_vacation_prone:
        reasons.append("Stop is holiday/vacation-prone.")
    if ctx.mission_critical:
        reasons.append("Stop is mission-critical to the plan.")
    if ctx.is_traveling and (ctx.child_present or ctx.timing_tight):
        reasons.append(
            "Travelling with a child or under tight timing — a wasted stop is high cost."
        )

    return (bool(reasons), reasons)


def build_call_script(ctx: CallContext) -> str:
    """Build a plain-language, data-minimized call checklist.

    The script asks the brief's required questions and adds the Europe-summer
    closure question when applicable. It deliberately never references the
    household's child, accommodation, schedule, or any internal context.
    """

    lines = [
        f"Hello, I'm calling to check a few details about {ctx.vendor_name} before "
        "recommending it to someone visiting today.",
        "1. Are you open now, and until what time?",
        "2. Is the kitchen / service open right now?",
        "3. Are you taking walk-ins or reservations today?",
        "4. Do you have a suitable high-protein / child-friendly option available?",
        "5. Any closure, private event, or limited menu today?",
    ]
    if is_europe_summer_high_risk(ctx):
        lines.append(
            "6. Are you currently on any summer vacation or holiday closure, and is "
            "the kitchen open today?"
        )
    script = "\n".join(lines)
    return sanitize_note(script, ctx)


# Phrases that, if present in a free-text note, indicate disclosure of private
# household context. Kept conservative; pair with explicit ``sensitive_terms``.
_SENSITIVE_PATTERNS = (
    re.compile(r"\bhotel\b[^.\n]*", re.IGNORECASE),
    re.compile(r"\bairbnb\b[^.\n]*", re.IGNORECASE),
    re.compile(r"\bstaying at\b[^.\n]*", re.IGNORECASE),
    re.compile(r"\broom\s*\d+\b", re.IGNORECASE),
)


def sanitize_note(text: str, ctx: Optional[CallContext] = None) -> str:
    """Remove private household context from any text bound for logs or scripts."""

    if not text:
        return ""
    out = text
    if ctx is not None:
        for term in ctx.sensitive_terms:
            term = (term or "").strip()
            if term:
                out = re.sub(re.escape(term), "[redacted]", out, flags=re.IGNORECASE)
    for pat in _SENSITIVE_PATTERNS:
        out = pat.sub("[redacted]", out)
    return out


@dataclass
class CallResult:
    """Structured, sanitized record of a verification call (brief schema)."""

    vendor_name: str
    phone_number: str
    call_time: str                       # ISO 8601 timestamp
    timezone: str
    answered_by: AnsweredBy
    open_status: OpenStatus
    kitchen_or_service_available: ServiceStatus
    reservation_or_walkin_status: str
    fit_notes: str
    confidence: Confidence
    next_action: str
    short_note: str                      # sanitized raw note

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Enum):
                d[key] = value.value
        # asdict does not unwrap enum values inside dataclass; do it explicitly.
        d["answered_by"] = self.answered_by.value
        d["open_status"] = self.open_status.value
        d["kitchen_or_service_available"] = self.kitchen_or_service_available.value
        d["confidence"] = self.confidence.value
        return d


# Map upstream provider answer strings to our normalised taxonomy.
_PROVIDER_ANSWER_MAP = {
    "human": AnsweredBy.HUMAN,
    "person": AnsweredBy.HUMAN,
    "voicemail": AnsweredBy.VOICEMAIL,
    "voice-mail": AnsweredBy.VOICEMAIL,
    "machine": AnsweredBy.VOICEMAIL,
    "answering_machine": AnsweredBy.VOICEMAIL,
    "recording": AnsweredBy.RECORDING,
    "ivr": AnsweredBy.RECORDING,
    "no-answer": AnsweredBy.NO_ANSWER,
    "no_answer": AnsweredBy.NO_ANSWER,
    "noanswer": AnsweredBy.NO_ANSWER,
    "busy": AnsweredBy.NO_ANSWER,
    "failed": AnsweredBy.FAILED,
    "error": AnsweredBy.FAILED,
    "canceled": AnsweredBy.FAILED,
}


def normalize_answered_by(value: Optional[str]) -> AnsweredBy:
    if not value:
        return AnsweredBy.NO_ANSWER
    return _PROVIDER_ANSWER_MAP.get(str(value).strip().lower(), AnsweredBy.FAILED)


def derive_confidence(
    answered_by: AnsweredBy,
    open_status: OpenStatus,
    ctx: CallContext,
    *,
    contradicts_online: bool = False,
) -> Confidence:
    """Confidence gating — the safety core of the capability.

    Rules:
      - Only a live human answer can yield a VERIFIED (route-able) result.
      - A live human result that contradicts online data is CONTRADICTED
        (still actionable: the verified phone result overrides stale Maps).
      - Anything else (voicemail/recording/no-answer/failed, or human +
        uncertain open status) is UNVERIFIED: never a confident recommendation.
    """

    if answered_by not in _LIVE_HUMAN_ANSWERS:
        return Confidence.UNVERIFIED
    if open_status is OpenStatus.UNCERTAIN:
        return Confidence.UNVERIFIED
    if contradicts_online:
        return Confidence.CONTRADICTED
    return Confidence.VERIFIED


def build_call_result(
    ctx: CallContext,
    *,
    provider_status: dict[str, Any],
    call_time: str,
    timezone: str,
    open_status: OpenStatus,
    kitchen_or_service_available: ServiceStatus,
    reservation_or_walkin_status: str = "uncertain",
    fit_notes: str = "",
    raw_note: str = "",
) -> CallResult:
    """Turn a raw telephony ``ai-status`` payload into a structured CallResult.

    ``open_status`` / ``kitchen_or_service_available`` are the interpreted answers
    to the call script (parsed from ``provider_status['transcript']`` /
    ``['analysis']`` by the caller). This function owns normalisation,
    confidence gating, the next-action decision, and sanitization.
    """

    answered_by = normalize_answered_by(
        provider_status.get("answered_by") or provider_status.get("status")
    )

    contradicts_online = _contradicts_online(ctx, open_status)
    confidence = derive_confidence(
        answered_by, open_status, ctx, contradicts_online=contradicts_online
    )
    next_action = _next_action(confidence, open_status, ctx)

    return CallResult(
        vendor_name=ctx.vendor_name,
        phone_number=ctx.phone_number,
        call_time=call_time,
        timezone=timezone,
        answered_by=answered_by,
        open_status=open_status,
        kitchen_or_service_available=kitchen_or_service_available,
        reservation_or_walkin_status=reservation_or_walkin_status,
        fit_notes=sanitize_note(fit_notes, ctx),
        confidence=confidence,
        next_action=next_action,
        short_note=sanitize_note(raw_note, ctx),
    )


def _contradicts_online(ctx: CallContext, open_status: OpenStatus) -> bool:
    online = (ctx.online_open_state or "").strip().lower()
    if not online:
        return False
    if online in {"open", "open now"} and open_status is OpenStatus.CLOSED:
        return True
    if online in {"closed", "closed now"} and open_status is OpenStatus.OPEN:
        return True
    return False


def _next_action(confidence: Confidence, open_status: OpenStatus, ctx: CallContext) -> str:
    if confidence is Confidence.VERIFIED and open_status is OpenStatus.OPEN:
        return "route"
    if confidence is Confidence.CONTRADICTED:
        if open_status is OpenStatus.OPEN:
            return "route_using_verified_phone_result"
        return "do_not_route_use_verified_phone_result"
    if open_status is OpenStatus.CLOSED:
        return "do_not_route_offer_alternative"
    return "do_not_route_verification_failed_offer_alternative"


def derive_recommendation(result: CallResult, ctx: CallContext) -> dict[str, Any]:
    """Whether the household may be routed, and the plain-language message."""

    routable = result.next_action in {
        "route",
        "route_using_verified_phone_result",
    }
    return {
        "routable": routable,
        "confidence": result.confidence.value,
        "next_action": result.next_action,
        "household_message": household_message(result, ctx),
    }


def household_message(result: CallResult, ctx: CallContext) -> str:
    """Plain, non-technical message for the household. No jargon, no internal context."""

    name = result.vendor_name
    if result.answered_by not in _LIVE_HUMAN_ANSWERS:
        return (
            f"I couldn't reach {name} to confirm they're open, so I won't treat it as "
            "verified. I'll suggest a place I can confirm instead."
        )
    if result.confidence is Confidence.CONTRADICTED:
        if result.open_status is OpenStatus.OPEN:
            return (
                f"Online listings looked off, but I called {name} and they confirmed "
                "they're open and serving now, so it's good to go."
            )
        return (
            f"Online said {name} was open, but I called and they're actually closed "
            "today — I won't send you there. I'll find a verified alternative."
        )
    if result.confidence is Confidence.VERIFIED and result.open_status is OpenStatus.OPEN:
        return f"I called {name} and they confirmed they're open and serving now."
    if result.open_status is OpenStatus.CLOSED:
        return (
            f"I called {name} and they're closed today, so I won't recommend it. "
            "I'll suggest a verified alternative."
        )
    return (
        f"I called {name} but couldn't get a clear confirmation they're serving now, "
        "so I won't treat it as verified. I'll suggest a place I can confirm instead."
    )
