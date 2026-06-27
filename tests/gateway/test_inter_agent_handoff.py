"""Tests for inter-agent (agent→agent) durable handoff injection.

Origin: HUM-2078. An inter-agent handoff addressed to a destination agent
(e.g. Ari → Mia) must reach that agent as an **actionable incoming turn**,
independent of any chat transport. Telegram's Bot API silently drops bot→bot
messages, so a shared chat can never carry a handoff between two bots; the
durable ``inter_agent_handoffs`` queue + gateway injection is the path that
survives that drop.

Two surfaces under test:

1. ``SessionDB`` queue CRUD — mirrors the session-handoff state machine:
   pending → running → (completed | failed).
2. ``GatewayRunner._process_inter_agent_handoff`` — proves the enqueued
   handoff surfaces as an actionable (``internal=True``, not observe-only)
   inbound turn in the destination agent's session, and that ingestion is
   logged.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hermes_state import SessionDB


# ── DB queue CRUD ──────────────────────────────────────────────────────────


class TestInterAgentHandoffDB:
    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        return SessionDB(db_path=home / "state.db")

    def test_table_exists(self, db):
        db._conn.execute(
            "SELECT id, target_platform, target_agent, origin_agent, text, "
            "state, error, created_at, updated_at FROM inter_agent_handoffs "
            "LIMIT 0"
        )

    def test_enqueue_returns_id_and_marks_pending(self, db):
        hid = db.enqueue_inter_agent_handoff(
            "telegram", "Mia, can you confirm pickup time?",
            target_agent="mia", origin_agent="ari",
        )
        assert isinstance(hid, int) and hid > 0

        pending = db.list_pending_inter_agent_handoffs()
        assert len(pending) == 1
        row = pending[0]
        assert row["id"] == hid
        assert row["target_platform"] == "telegram"
        assert row["target_agent"] == "mia"
        assert row["origin_agent"] == "ari"
        assert row["text"] == "Mia, can you confirm pickup time?"
        assert row["state"] == "pending"

    def test_platform_is_normalized(self, db):
        hid = db.enqueue_inter_agent_handoff("  TELEGRAM ", "hi")
        row = db.list_pending_inter_agent_handoffs()[0]
        assert row["id"] == hid
        assert row["target_platform"] == "telegram"

    def test_claim_is_atomic(self, db):
        hid = db.enqueue_inter_agent_handoff("telegram", "hi")
        assert db.claim_inter_agent_handoff(hid) is True
        # Second claim is a no-op (state is now running, not pending).
        assert db.claim_inter_agent_handoff(hid) is False

    def test_list_pending_excludes_running_and_terminal(self, db):
        a = db.enqueue_inter_agent_handoff("telegram", "a")
        b = db.enqueue_inter_agent_handoff("telegram", "b")
        c = db.enqueue_inter_agent_handoff("telegram", "c")
        d = db.enqueue_inter_agent_handoff("telegram", "d")

        db.claim_inter_agent_handoff(c)  # running
        db.claim_inter_agent_handoff(d)
        db.complete_inter_agent_handoff(d)  # terminal

        pending_ids = {r["id"] for r in db.list_pending_inter_agent_handoffs()}
        assert pending_ids == {a, b}

    def test_pending_ordered_oldest_first(self, db):
        first = db.enqueue_inter_agent_handoff("telegram", "first")
        second = db.enqueue_inter_agent_handoff("telegram", "second")
        ids = [r["id"] for r in db.list_pending_inter_agent_handoffs()]
        assert ids == [first, second]

    def test_complete_clears_error(self, db):
        hid = db.enqueue_inter_agent_handoff("telegram", "hi")
        db.claim_inter_agent_handoff(hid)
        db.fail_inter_agent_handoff(hid, "transient")
        # Retry path: re-enqueue + claim + complete.
        hid2 = db.enqueue_inter_agent_handoff("telegram", "hi again")
        db.claim_inter_agent_handoff(hid2)
        db.complete_inter_agent_handoff(hid2)
        row = next(
            r for r in db._conn.execute(
                "SELECT * FROM inter_agent_handoffs WHERE id = ?", (hid2,)
            )
        )
        assert row["state"] == "completed"
        assert row["error"] is None

    def test_fail_records_and_truncates_reason(self, db):
        hid = db.enqueue_inter_agent_handoff("telegram", "hi")
        db.claim_inter_agent_handoff(hid)
        db.fail_inter_agent_handoff(hid, "x" * 1000)
        row = next(
            r for r in db._conn.execute(
                "SELECT * FROM inter_agent_handoffs WHERE id = ?", (hid,)
            )
        )
        assert row["state"] == "failed"
        assert len(row["error"]) <= 500


# ── Gateway injection ────────────────────────────────────────────────────


class TestInterAgentHandoffInjection:
    """``_process_inter_agent_handoff`` must inject an actionable turn."""

    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        return SessionDB(db_path=home / "state.db")

    def _make_runner(self, db, *, response=None, home=True):
        """Build a GatewayRunner stub exercising only the handoff path."""
        from gateway.run import GatewayRunner
        from gateway.config import Platform, HomeChannel

        runner = object.__new__(GatewayRunner)
        runner._session_db = db
        runner._running = True

        home_channel = (
            HomeChannel(
                platform=Platform.TELEGRAM,
                chat_id="-5288400624",
                name="Mia Home",
            )
            if home
            else None
        )
        runner.config = SimpleNamespace(
            get_home_channel=lambda p: home_channel,
        )

        # Adapter only needs .send (used when the agent emits a reply).
        adapter = SimpleNamespace(
            send=AsyncMock(return_value=SimpleNamespace(success=True)),
        )
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._adapter = adapter

        # Capture the synthetic turn handed to the runner.
        runner._handle_message = AsyncMock(return_value=response)
        runner.session_store = SimpleNamespace(
            get_or_create_session=lambda source: None,
        )
        return runner

    @pytest.mark.asyncio
    async def test_handoff_surfaces_as_actionable_turn(self, db, caplog):
        hid = db.enqueue_inter_agent_handoff(
            "telegram",
            "Mia — handoff: confirm the 4pm pickup with the family.",
            target_agent="mia",
            origin_agent="ari",
        )
        runner = self._make_runner(db)

        with caplog.at_level(logging.INFO):
            row = db.list_pending_inter_agent_handoffs()[0]
            assert db.claim_inter_agent_handoff(hid) is True
            await runner._process_inter_agent_handoff(row)
            db.complete_inter_agent_handoff(hid)

        # The handoff was dispatched through the actionable runner path
        # exactly once — NOT dropped, NOT downgraded to observe-only.
        runner._handle_message.assert_awaited_once()
        event = runner._handle_message.await_args.args[0]

        # Actionable: internal synthetic turn, system:handoff author (bypasses
        # user-auth gating; never observe-only).
        assert event.internal is True
        assert event.source.user_id == "system:handoff"
        assert event.source.chat_id == "-5288400624"

        # The destination agent's next input visibly contains the handoff body
        # and the addressing frame.
        assert "confirm the 4pm pickup" in event.text
        assert "ari" in event.text and "mia" in event.text

        # Ingestion is logged (required acceptance: prove ingestion happened).
        assert any(
            "Inter-agent handoff" in r.getMessage()
            and "injecting actionable turn" in r.getMessage()
            for r in caplog.records
        )

        # Queue row reached terminal completed state.
        final = next(
            r for r in db._conn.execute(
                "SELECT state FROM inter_agent_handoffs WHERE id = ?", (hid,)
            )
        )
        assert final["state"] == "completed"

    @pytest.mark.asyncio
    async def test_agent_reply_is_sent_back(self, db):
        hid = db.enqueue_inter_agent_handoff("telegram", "ping", origin_agent="ari")
        runner = self._make_runner(db, response="on it")
        row = db.list_pending_inter_agent_handoffs()[0]
        db.claim_inter_agent_handoff(hid)
        await runner._process_inter_agent_handoff(row)
        runner._adapter.send.assert_awaited_once()
        assert runner._adapter.send.await_args.kwargs["content"] == "on it"

    @pytest.mark.asyncio
    async def test_no_home_channel_raises(self, db):
        hid = db.enqueue_inter_agent_handoff("telegram", "ping")
        runner = self._make_runner(db, home=False)
        row = db.list_pending_inter_agent_handoffs()[0]
        db.claim_inter_agent_handoff(hid)
        with pytest.raises(RuntimeError, match="no home channel"):
            await runner._process_inter_agent_handoff(row)

    @pytest.mark.asyncio
    async def test_empty_text_raises(self, db):
        # Enqueue bypasses validation; the processor must reject empty bodies
        # so a malformed row never silently no-ops.
        db.enqueue_inter_agent_handoff("telegram", "   ")
        runner = self._make_runner(db)
        row = db.list_pending_inter_agent_handoffs()[0]
        with pytest.raises(RuntimeError, match="text is empty"):
            await runner._process_inter_agent_handoff(row)
