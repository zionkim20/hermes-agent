"""QA for the governed call-before-routing capability (HUM-1848).

Covers the three acceptance QA scenarios plus the structural and safety
invariants from the product brief:
  1. stale-map data overridden by a verified phone result
  2. Europe summer-vacation closure where Maps says open but direct
     verification is required before routing
  3. no-answer falls back to safer alternatives, never a confident recommendation
  4. structured result captures every required field
  5. failed / voicemail / recording answers are never confident
  6. data minimization: child name / accommodation never leak into script or note

Run with the in-repo venv from the telephony skill scripts dir:
  venv/bin/python -m pytest \
    optional-skills/productivity/telephony/scripts/test_call_before_routing.py \
    -o addopts="" -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from call_before_routing import (  # noqa: E402
    AnsweredBy,
    CallContext,
    Confidence,
    OpenStatus,
    ServiceStatus,
    build_call_result,
    build_call_script,
    derive_recommendation,
    is_europe_summer_high_risk,
    normalize_answered_by,
    requires_verification,
    sanitize_note,
)

REQUIRED_RESULT_FIELDS = {
    "vendor_name",
    "phone_number",
    "call_time",
    "timezone",
    "answered_by",
    "open_status",
    "kitchen_or_service_available",
    "reservation_or_walkin_status",
    "fit_notes",
    "confidence",
    "next_action",
    "short_note",
}


def _result(ctx, provider_status, open_status, service, raw_note=""):
    return build_call_result(
        ctx,
        provider_status=provider_status,
        call_time="2026-07-15T12:30:00",
        timezone="Europe/Paris",
        open_status=open_status,
        kitchen_or_service_available=service,
        raw_note=raw_note,
    )


# --- Scenario 1: stale-map data overridden by verified phone result -----------

def test_stale_map_open_but_phone_says_closed_overrides():
    ctx = CallContext(
        vendor_name="Chez Marcel",
        phone_number="+33123456789",
        region="Lyon, France",
        month=7,
        is_traveling=True,
        child_present=True,
        online_open_state="open",
    )
    result = _result(
        ctx,
        {"answered_by": "human", "status": "completed"},
        OpenStatus.CLOSED,
        ServiceStatus.NO,
    )
    assert result.confidence is Confidence.CONTRADICTED
    rec = derive_recommendation(result, ctx)
    assert rec["routable"] is False
    assert "closed" in rec["household_message"].lower()
    assert result.next_action == "do_not_route_use_verified_phone_result"


def test_stale_map_closed_but_phone_says_open_overrides_and_routes():
    ctx = CallContext(
        vendor_name="Trattoria Sole",
        phone_number="+390612345678",
        region="Rome, Italy",
        month=8,
        online_open_state="closed",
    )
    result = _result(
        ctx,
        {"answered_by": "human", "status": "completed"},
        OpenStatus.OPEN,
        ServiceStatus.YES,
    )
    assert result.confidence is Confidence.CONTRADICTED
    rec = derive_recommendation(result, ctx)
    assert rec["routable"] is True
    assert result.next_action == "route_using_verified_phone_result"


# --- Scenario 2: Europe summer closure, Maps-open is not enough ----------------

def test_europe_summer_requires_verification_even_if_maps_open():
    ctx = CallContext(
        vendor_name="Le Petit Jardin",
        region="Nice, France",
        month=7,
        is_traveling=True,
        child_present=True,
        online_open_state="open",
    )
    assert is_europe_summer_high_risk(ctx) is True
    required, reasons = requires_verification(ctx)
    assert required is True
    assert any("summer" in r.lower() for r in reasons)
    # The script must add the summer-vacation closure question.
    script = build_call_script(ctx)
    assert "summer vacation" in script.lower()


def test_europe_summer_human_confirms_open_is_verified():
    ctx = CallContext(
        vendor_name="Bistro Marais",
        region="Paris, France",
        month=6,
        online_open_state="open",
    )
    result = _result(
        ctx,
        {"answered_by": "human", "status": "completed"},
        OpenStatus.OPEN,
        ServiceStatus.YES,
    )
    # online "open" agrees with phone "open" -> verified, not merely contradicted
    assert result.confidence is Confidence.VERIFIED
    assert derive_recommendation(result, ctx)["routable"] is True


def test_non_europe_or_off_season_not_forced_by_summer_rule():
    # Same month, but not Europe.
    ctx_us = CallContext(vendor_name="Joe's Diner", region="Austin, Texas", month=7)
    assert is_europe_summer_high_risk(ctx_us) is False
    # Europe but off-season.
    ctx_winter = CallContext(vendor_name="Chez X", region="Paris, France", month=2)
    assert is_europe_summer_high_risk(ctx_winter) is False


# --- Scenario 3: no-answer falls back to safer alternatives --------------------

def test_no_answer_never_confident_offers_alternative():
    ctx = CallContext(
        vendor_name="Osteria Vela",
        region="Venice, Italy",
        month=8,
        online_open_state="open",
    )
    result = _result(
        ctx,
        {"answered_by": "no_answer", "status": "no-answer"},
        OpenStatus.UNCERTAIN,
        ServiceStatus.UNCERTAIN,
    )
    assert result.confidence is Confidence.UNVERIFIED
    rec = derive_recommendation(result, ctx)
    assert rec["routable"] is False
    msg = rec["household_message"].lower()
    assert "couldn't reach" in msg or "could not reach" in msg
    assert "alternative" in msg or "instead" in msg


# --- Invariant: failed / voicemail / recording are never confident ------------

def test_failed_and_machine_answers_never_confident():
    ctx = CallContext(vendor_name="V", region="Madrid, Spain", month=7, online_open_state="open")
    for ans in ("failed", "voicemail", "recording", "busy", "error"):
        result = _result(
            ctx,
            {"answered_by": ans, "status": ans},
            OpenStatus.OPEN,            # even if a machine "claims" open
            ServiceStatus.YES,
        )
        assert result.confidence is Confidence.UNVERIFIED, ans
        assert derive_recommendation(result, ctx)["routable"] is False, ans


def test_human_but_uncertain_open_is_not_confident():
    ctx = CallContext(vendor_name="V", region="Berlin, Germany", month=7)
    result = _result(
        ctx,
        {"answered_by": "human", "status": "completed"},
        OpenStatus.UNCERTAIN,
        ServiceStatus.UNCERTAIN,
    )
    assert result.confidence is Confidence.UNVERIFIED
    assert derive_recommendation(result, ctx)["routable"] is False


def test_normalize_answered_by_unknown_is_failed():
    assert normalize_answered_by("something-weird") is AnsweredBy.FAILED
    assert normalize_answered_by("") is AnsweredBy.NO_ANSWER
    assert normalize_answered_by(None) is AnsweredBy.NO_ANSWER


# --- Structured result completeness ------------------------------------------

def test_structured_result_has_all_required_fields():
    ctx = CallContext(vendor_name="Full Fields", phone_number="+15551230000", region="Paris", month=7)
    result = _result(
        ctx,
        {"answered_by": "human", "status": "completed"},
        OpenStatus.OPEN,
        ServiceStatus.YES,
        raw_note="They are open until 22:00.",
    )
    d = result.to_dict()
    assert REQUIRED_RESULT_FIELDS.issubset(d.keys())
    # enum values serialised to plain strings for persistence
    assert d["answered_by"] == "human"
    assert d["open_status"] == "open"
    assert d["confidence"] in {"verified", "unverified", "contradicted"}


# --- Data minimization --------------------------------------------------------

def test_script_and_note_do_not_leak_sensitive_context():
    ctx = CallContext(
        vendor_name="Cafe Lumiere",
        region="Paris, France",
        month=7,
        sensitive_terms=("Zaya", "Hotel Lutetia"),
    )
    script = build_call_script(ctx)
    assert "zaya" not in script.lower()
    assert "lutetia" not in script.lower()

    note = sanitize_note(
        "Told them Zaya needs a high-protein meal; we are staying at Hotel Lutetia room 304.",
        ctx,
    )
    assert "zaya" not in note.lower()
    assert "lutetia" not in note.lower()
    assert "room 304" not in note.lower()
    assert "[redacted]" in note


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
