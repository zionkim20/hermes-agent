"""Booking-task continuity resolver + HUM-1917 repro (HUM-2199 / HUM-1918).

T1 active-task retrieval, T2 conflict guard, T3 explicit switch, plus the
end-to-end HUM-1917 acceptance: a voice-note "that winery tour" reference
resolves to Château La Dominique / Le Charme / 5 July and a conflicting fresh
lookup (Château Ferrand / Discovery Tour / July 4) cannot overwrite it.
"""
import pytest

from hermes_state import SessionDB
from gateway.booking_continuity import (
    detect_continuity_signal,
    detect_explicit_switch,
    detect_conflict,
    build_anchored_facts_line,
    build_preflight_note,
)

HOUSEHOLD = "whatsapp:renata-family"
THREAD = "wa-thread-1"

# The fresh lookup that WRONGLY hijacked the thread in the original ticket.
FERRAND_LOOKUP = {
    "vendor_entity": "Château Ferrand",
    "offering_name": "Discovery Tour",
    "date": "Saturday July 4",
    "reservation_url_or_contact": "https://chateau-ferrand.example/tours",
}


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


@pytest.fixture
def seeded_db(db):
    """DB with the anchored La Dominique task (post vendor-reply ingestion)."""
    db.upsert_active_booking_task(
        HOUSEHOLD,
        THREAD,
        requester="Renata",
        task_type="winery_tour_booking",
        vendor_entity="Château La Dominique",
        offering_name="Le Charme",
        date="5 July",
        party_size="4",
        source_evidence_type="vendor_reply",
        source_evidence_summary="Vendor can welcome the family for Le Charme on 5 July.",
        reservation_url_or_contact="https://reservation.chateau-ladominique.com/",
        booking_status="not_booked",
        confidence="high",
    )
    return db


# ── Signal detection ────────────────────────────────────────────────────────
class TestSignalDetection:
    @pytest.mark.parametrize("text", [
        "Can you help me with that winery tour?",
        "what's the status on the reservation?",
        "can you finish the booking",
        "reply to the winery please",
        "the thing from the screenshot",
    ])
    def test_positive_signals(self, text):
        assert detect_continuity_signal(text) is True

    @pytest.mark.parametrize("text", [
        "what's the weather in Bordeaux tomorrow?",
        "thanks so much!",
        "",
        None,
    ])
    def test_negative_signals(self, text):
        assert detect_continuity_signal(text) is False

    def test_new_evidence_forces_signal(self):
        # A screenshot/voice note arriving with no deictic text still triggers.
        assert detect_continuity_signal("here", has_new_evidence=True) is True


# ── T1: active-task retrieval before lookup ─────────────────────────────────
class TestT1Retrieval:
    def test_voice_note_retrieves_anchored_task(self, seeded_db):
        inbound = "Can you help me with that winery tour?"
        assert detect_continuity_signal(inbound) is True
        task = seeded_db.get_active_booking_task(HOUSEHOLD, THREAD)
        assert task is not None
        assert task["vendor_entity"] == "Château La Dominique"
        note = build_preflight_note(task)
        # The note that steers the agent names the anchored facts, no internal IDs.
        assert "Château La Dominique" in note
        assert "Le Charme" in note
        assert "5 July" in note
        assert "reservation.chateau-ladominique.com" in note
        assert "HUM-" not in note
        assert "2199" not in note


# ── T2: conflict guard ──────────────────────────────────────────────────────
class TestT2ConflictGuard:
    def test_conflict_detected_on_all_fields(self, seeded_db):
        task = seeded_db.get_active_booking_task(HOUSEHOLD, THREAD)
        result = detect_conflict(task, FERRAND_LOOKUP)
        assert result.conflict is True
        assert set(result.fields) >= {"vendor_entity", "offering_name", "date"}

    def test_anchored_facts_unchanged_after_conflicting_lookup(self, seeded_db):
        task = seeded_db.get_active_booking_task(HOUSEHOLD, THREAD)
        result = detect_conflict(task, FERRAND_LOOKUP)
        # The guard's contract: on conflict we do NOT persist the lookup.
        assert result.conflict is True
        # Persistence untouched — still La Dominique.
        reread = seeded_db.get_active_booking_task(HOUSEHOLD, THREAD)
        assert reread["vendor_entity"] == "Château La Dominique"
        assert reread["date"] == "5 July"
        # The household-facing anchored line references La Dominique, never Ferrand.
        line = build_anchored_facts_line(reread)
        assert "Château La Dominique" in line
        assert "Ferrand" not in line
        assert "Discovery Tour" not in line

    def test_enrichment_is_not_conflict(self, seeded_db):
        task = seeded_db.get_active_booking_task(HOUSEHOLD, THREAD)
        # A lookup that only adds a phone number (task lacks one) is enrichment.
        enrich = {"reservation_url_or_contact": task["reservation_url_or_contact"],
                  "time_window": "afternoon"}
        assert detect_conflict(task, enrich).conflict is False


# ── T3: explicit user switch ────────────────────────────────────────────────
class TestT3ExplicitSwitch:
    def test_explicit_switch_detected(self):
        text = "Actually ignore La Dominique and look for Château Ferrand on July 4"
        vendor = detect_explicit_switch(text)
        assert vendor is not None
        assert "Ferrand" in vendor

    def test_non_switch_not_flagged(self):
        assert detect_explicit_switch("help me with that winery tour") is None

    def test_switch_supersedes_only_after_confirmation(self, seeded_db):
        # A detected switch request does NOT itself mutate state. The prior task
        # remains anchored until the caller explicitly supersedes (post-confirm).
        text = "Actually ignore La Dominique and look for Château Ferrand on July 4"
        assert detect_explicit_switch(text) is not None
        # Nothing superseded yet.
        assert seeded_db.get_active_booking_task(HOUSEHOLD, THREAD)["vendor_entity"] == "Château La Dominique"
        # Simulate explicit user confirmation → caller supersedes + creates new.
        seeded_db.supersede_active_booking_task(HOUSEHOLD, THREAD)
        seeded_db.upsert_active_booking_task(
            HOUSEHOLD, THREAD, vendor_entity="Château Ferrand",
            offering_name="Discovery Tour", date="4 July", booking_status="not_booked",
        )
        now = seeded_db.get_active_booking_task(HOUSEHOLD, THREAD)
        assert now["vendor_entity"] == "Château Ferrand"


# ── HUM-1917 acceptance repro ───────────────────────────────────────────────
class TestHUM1917Repro:
    def test_repro_returns_ladominique_not_ferrand(self, seeded_db):
        """Full continuity path: vendor-reply ingestion → voice-note continuation
        → conflicting fresh lookup → anchored response."""
        # 1. Voice-note continuation.
        inbound = "Can you help me with that winery tour?"
        assert detect_continuity_signal(inbound) is True

        # 2. Retrieve anchored task before any lookup.
        task = seeded_db.get_active_booking_task(HOUSEHOLD, THREAD)
        assert task is not None

        # 3. A fresh lookup returns the conflicting Ferrand facts.
        conflict = detect_conflict(task, FERRAND_LOOKUP)
        assert conflict.conflict is True

        # 4. The anchored, household-facing response is produced from the task,
        #    NOT the lookup.
        response = build_anchored_facts_line(task)
        assert "Château La Dominique" in response
        assert "Le Charme" in response
        assert "5 July" in response
        assert "reservation.chateau-ladominique.com" in response
        # Must NOT be the hijacking lookup.
        assert "Château Ferrand" not in response
        assert "Discovery Tour" not in response
        assert "July 4" not in response
