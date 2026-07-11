"""Production wiring for the active booking-task continuity guard (HUM-2204).

These drive the *production entry point* — the gateway resolver
``GatewayRunner._resolve_booking_continuity`` that ``_handle_message_with_agent``
calls on every inbound message (gateway/run.py) — and assert the
``active_booking_task`` row is actually written / enriched / status-updated /
superseded on live messages, plus the attachment-continuation preflight path.

HUM-2203 review blockers this covers:
  * Blocker 1 — the write/conflict/switch path is exercised via the handler
    resolver, not a direct unit call to the SessionDB helper.
  * Blocker 2 — an attachment-only continuation (media_urls, no continuity
    text) still triggers the preflight via ``has_new_evidence``.
"""
import types

import pytest

import gateway.run as gateway_run
from gateway.booking_continuity import (
    detect_continuity_signal,
    build_preflight_note,
)
from gateway.config import Platform
from gateway.session import SessionSource
from hermes_state import SessionDB

HOUSEHOLD_CHAT = "renata-family"
THREAD = "wa-thread-1"
# household_id the resolver derives from (platform, chat_id).
HH_KEY = f"{Platform.WHATSAPP.value}:{HOUSEHOLD_CHAT}"

VENDOR_REPLY = (
    "Château La Dominique can welcome your family for the Le Charme tour on "
    "5 July. Reserve at https://reservation.chateau-ladominique.com/"
)


def _runner(db):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_db = db
    return runner


def _source(thread=THREAD):
    return SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=HOUSEHOLD_CHAT,
        user_id="renata",
        user_name="Renata",
        thread_id=thread,
    )


def _event(text="", media_urls=None):
    return types.SimpleNamespace(text=text, media_urls=list(media_urls or []))


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


# ── Acceptance 1: the live inbound path writes / enriches / updates / supersedes
class TestResolverWritePath:
    def test_vendor_reply_persists_active_task(self, db):
        runner = _runner(db)
        task = runner._resolve_booking_continuity(_event(VENDOR_REPLY), _source())
        # Row written by the production resolver, not a direct helper call.
        assert task is not None
        assert task["vendor_entity"] == "Château La Dominique"
        assert task["offering_name"] == "Le Charme"
        assert task["date"] == "5 July"
        assert task["reservation_url_or_contact"] == "https://reservation.chateau-ladominique.com/"
        assert task["booking_status"] == "not_booked"
        # And it is retrievable as the anchored task for the thread.
        assert db.get_active_booking_task(HH_KEY, THREAD)["id"] == task["id"]

    def test_enrich_not_blank_and_conflict_guard(self, db):
        runner = _runner(db)
        runner._resolve_booking_continuity(_event(VENDOR_REPLY), _source())
        # A later message names a DIFFERENT vendor/date (a bad fresh lookup
        # leaking into text) but also adds a genuinely new fact (party size is
        # not a conflict field). Anchored vendor/date must NOT be overwritten;
        # the message must not blank existing fields.
        conflicting = (
            "Château Ferrand has a Discovery Tour on July 4, phone 555-1234."
        )
        task = runner._resolve_booking_continuity(_event(conflicting), _source())
        assert task["vendor_entity"] == "Château La Dominique"  # kept
        assert task["date"] == "5 July"                          # kept
        assert task["offering_name"] == "Le Charme"              # not blanked
        # Only one active row — enriched in place, not a second task.
        assert task["superseded"] == 0

    def test_status_only_update_on_existing_task(self, db):
        runner = _runner(db)
        runner._resolve_booking_continuity(_event(VENDOR_REPLY), _source())
        # A pure status message (no new vendor/link) routes to
        # update_booking_status, not a fresh anchor.
        task = runner._resolve_booking_continuity(
            _event("great news — the winery confirmed our booking, we're all set!"),
            _source(),
        )
        assert task["booking_status"] == "confirmed"
        assert task["last_confirmed_at"] is not None

    def test_explicit_switch_supersedes_anchor(self, db):
        runner = _runner(db)
        runner._resolve_booking_continuity(_event(VENDOR_REPLY), _source())
        # User explicitly directs Mia to drop the anchored vendor.
        result = runner._resolve_booking_continuity(
            _event("Actually ignore La Dominique and look for Château Ferrand on July 4"),
            _source(),
        )
        # Anchor retired → resolver returns None and no active row remains.
        assert result is None
        assert db.get_active_booking_task(HH_KEY, THREAD) is None

    def test_ordinary_chatter_writes_nothing(self, db):
        runner = _runner(db)
        # No vendor, no link → nothing anchored, resolver returns None.
        assert runner._resolve_booking_continuity(
            _event("thanks so much, talk soon!"), _source()
        ) is None
        assert db.get_active_booking_task(HH_KEY, THREAD) is None


# ── Acceptance 2: attachment-only continuation triggers the preflight ────────
class TestAttachmentContinuity:
    def test_media_only_continuation_injects_anchored_note(self, db):
        runner = _runner(db)
        # Seed an anchored task via the production resolver.
        runner._resolve_booking_continuity(_event(VENDOR_REPLY), _source())

        # A follow-up screenshot with NO continuity text arrives.
        media_event = _event(text="", media_urls=["/tmp/screenshot.jpg"])
        anchored = runner._resolve_booking_continuity(media_event, _source())
        # The anchored task is still returned (media alone doesn't overwrite it).
        assert anchored is not None
        assert anchored["vendor_entity"] == "Château La Dominique"

        # This is the exact gate _handle_message_with_agent uses to inject the
        # preflight note (gateway/run.py, booking-continuity preflight block).
        has_new_evidence = bool(media_event.media_urls)
        assert detect_continuity_signal(media_event.text or "", has_new_evidence=has_new_evidence) is True
        note = build_preflight_note(anchored)
        assert "Château La Dominique" in note
        assert "Le Charme" in note
        assert "5 July" in note
        assert "reservation.chateau-ladominique.com" in note
        # No internal IDs leak to the household channel.
        assert "HUM-" not in note

    def test_text_continuation_without_media_still_injects(self, db):
        runner = _runner(db)
        runner._resolve_booking_continuity(_event(VENDOR_REPLY), _source())
        text_event = _event("can you help me with that winery tour?")
        anchored = runner._resolve_booking_continuity(text_event, _source())
        assert anchored is not None
        assert detect_continuity_signal(text_event.text, has_new_evidence=False) is True
