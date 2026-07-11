"""Tests for the active_booking_task persistence layer (HUM-2199 / HUM-1918).

Covers the storage half of Mia's booking-task continuity guard: upsert/enrich,
single-active-task-per-key invariant, status updates, and supersede-on-switch.
"""
import pytest

from hermes_state import SessionDB

HOUSEHOLD = "whatsapp:renata-family"
THREAD = "wa-thread-1"


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed_ladominique(db):
    return db.upsert_active_booking_task(
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


class TestUpsert:
    def test_insert_then_get(self, db):
        tid = _seed_ladominique(db)
        assert isinstance(tid, int)
        task = db.get_active_booking_task(HOUSEHOLD, THREAD)
        assert task is not None
        assert task["vendor_entity"] == "Château La Dominique"
        assert task["offering_name"] == "Le Charme"
        assert task["date"] == "5 July"
        assert task["booking_status"] == "not_booked"
        assert task["reservation_url_or_contact"].startswith("https://reservation.chateau-ladominique.com")

    def test_upsert_enriches_without_blanking(self, db):
        tid = _seed_ladominique(db)
        # A later evidence event adds party detail but omits the vendor.
        tid2 = db.upsert_active_booking_task(
            HOUSEHOLD, THREAD, time_window="afternoon"
        )
        assert tid2 == tid  # same active row, not a new one
        task = db.get_active_booking_task(HOUSEHOLD, THREAD)
        assert task["time_window"] == "afternoon"
        # Omitted fields must NOT be blanked by the enrich.
        assert task["vendor_entity"] == "Château La Dominique"
        assert task["date"] == "5 July"

    def test_single_active_task_per_key(self, db):
        _seed_ladominique(db)
        db.upsert_active_booking_task(
            HOUSEHOLD, THREAD, offering_name="Le Charme (updated)"
        )
        # Still exactly one active task for the key.
        task = db.get_active_booking_task(HOUSEHOLD, THREAD)
        assert task["offering_name"] == "Le Charme (updated)"

    def test_get_missing_returns_none(self, db):
        assert db.get_active_booking_task(HOUSEHOLD, "no-such-thread") is None

    def test_unknown_kwarg_is_ignored(self, db):
        # A caller typo / injected column name is silently dropped, not written.
        tid = db.upsert_active_booking_task(
            HOUSEHOLD, THREAD, vendor_entity="Test", bogus_col="x"
        )
        task = db.get_active_booking_task(HOUSEHOLD, THREAD)
        assert task["vendor_entity"] == "Test"
        assert "bogus_col" not in task


class TestStatus:
    def test_update_status(self, db):
        _seed_ladominique(db)
        task = db.get_active_booking_task(HOUSEHOLD, THREAD)
        db.update_booking_status(task["id"], "requested")
        assert db.get_active_booking_task(HOUSEHOLD, THREAD)["booking_status"] == "requested"

    def test_update_status_confirmed_stamps_last_confirmed(self, db):
        _seed_ladominique(db)
        task = db.get_active_booking_task(HOUSEHOLD, THREAD)
        assert task["last_confirmed_at"] is None
        db.update_booking_status(task["id"], "confirmed", mark_confirmed=True)
        after = db.get_active_booking_task(HOUSEHOLD, THREAD)
        assert after["booking_status"] == "confirmed"
        assert after["last_confirmed_at"] is not None


class TestSupersede:
    def test_supersede_hides_task_and_frees_key(self, db):
        _seed_ladominique(db)
        db.supersede_active_booking_task(HOUSEHOLD, THREAD)
        # No active task after supersede.
        assert db.get_active_booking_task(HOUSEHOLD, THREAD) is None
        # A fresh upsert on the same key starts a NEW active row.
        new_id = db.upsert_active_booking_task(
            HOUSEHOLD, THREAD, vendor_entity="Château Ferrand",
            offering_name="Discovery Tour", date="4 July",
            booking_status="not_booked",
        )
        task = db.get_active_booking_task(HOUSEHOLD, THREAD)
        assert task["id"] == new_id
        assert task["vendor_entity"] == "Château Ferrand"
