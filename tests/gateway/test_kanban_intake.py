import json
import os
import sqlite3

from gateway import kanban_intake
from hermes_cli import kanban_db


def test_kanban_intake_captures_declarative_action_and_mental_load(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("HERMES_INTAKE_LLM_ALWAYS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    kanban_db.init_db(db_path)

    cards = kanban_intake.capture_inbound(
        "renata wants to start packing tomorrow and i need to figure out the chef filming plan",
        session_key="whatsapp:family",
        event_message_id="msg-403pm",
    )

    assert any(c["card_type"] == "action" for c in cards)
    assert any(c["card_type"] in {"absorbed_load", "decision_with_default"} for c in cards)
    assert all(c["id"] for c in cards)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT kind, card_type, mental_load_category, visibility, proposed_default, source_message_id "
        "FROM tasks WHERE source_message_id = ?",
        ("msg-403pm",),
    ).fetchall()
    assert len(rows) == len(cards)
    assert all(r["kind"] == "household" for r in rows)
    assert any(r["card_type"] == "action" for r in rows)
    assert any(r["mental_load_category"] == "anticipation" for r in rows)
    assert all(r["source_message_id"] == "msg-403pm" for r in rows)

    context = kanban_intake.format_pre_turn_context(cards)
    assert "captured BEFORE this turn" in context
    assert "Do NOT re-discover" in context


def test_kanban_intake_is_idempotent_by_source_message(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    kanban_db.init_db(db_path)

    msg = "we should discuss travel logistics tomorrow"
    first = kanban_intake.capture_inbound(msg, session_key="s", event_message_id="same-msg")
    second = kanban_intake.capture_inbound(msg, session_key="s", event_message_id="same-msg")

    assert [c["id"] for c in second] == [c["id"] for c in first]
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM tasks WHERE source_message_id = 'same-msg'").fetchone()[0]
    assert count == len(first)


def test_kanban_intake_captures_clean_batched_household_asks(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    kanban_db.init_db(db_path)

    cards = kanban_intake.capture_inbound(
        "\n".join(
            [
                "ok so for this week we still need to see what help renata needs with moving to make sure this is done",
                "also when is that party I have to go to for zaya",
                "i think filming will happen on wednesday",
                "and what time are we leaving saturday",
            ]
        ),
        session_key="whatsapp:zion",
        event_message_id="clean-batch",
    )

    action_titles = [c["title"].lower() for c in cards if c["card_type"] == "action"]
    assert len(action_titles) >= 4
    assert any("renata needs with moving" in title for title in action_titles)
    assert any("party i have to go to for zaya" in title for title in action_titles)
    assert any("filming" in title and "wednesday" in title for title in action_titles)
    assert any("leaving saturday" in title for title in action_titles)


def test_kanban_intake_preserves_source_unit_provenance(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    kanban_db.init_db(db_path)

    units = [
        {
            "unit_id": "wa-1",
            "source_message_id": "wa-1",
            "source_type": "user_text",
            "sender_id": "zion",
            "channel": "whatsapp",
            "chat_id": "family",
            "timestamp": "2026-06-08T19:55:00-03:00",
            "text": "ok so for this week we still need to see what help renata needs with moving to make sure this is done",
        },
        {
            "unit_id": "wa-2",
            "source_message_id": "wa-2",
            "source_type": "user_text",
            "sender_id": "zion",
            "channel": "whatsapp",
            "chat_id": "family",
            "timestamp": "2026-06-08T19:55:10-03:00",
            "text": "also when is that party I have to go to for zaya",
        },
        {
            "unit_id": "wa-3",
            "source_message_id": "wa-3",
            "source_type": "user_text",
            "sender_id": "zion",
            "channel": "whatsapp",
            "chat_id": "family",
            "timestamp": "2026-06-08T19:55:20-03:00",
            "text": "i think filming will happen on wednesday",
        },
        {
            "unit_id": "wa-4",
            "source_message_id": "wa-4",
            "source_type": "user_text",
            "sender_id": "zion",
            "channel": "whatsapp",
            "chat_id": "family",
            "timestamp": "2026-06-08T19:55:30-03:00",
            "text": "and what time are we leaving saturday",
        },
    ]

    cards = kanban_intake.capture_inbound(
        "\n".join(unit["text"] for unit in units),
        session_key="whatsapp:family",
        event_message_id="wa-batch",
        provenance_units=units,
    )

    action_cards = [card for card in cards if card["card_type"] == "action"]
    action_source_ids = {card["source_message_id"] for card in action_cards}
    assert {"wa-1", "wa-2", "wa-3", "wa-4"}.issubset(action_source_ids)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT title, body, source_message_id FROM tasks WHERE source_message_id IN ('wa-1', 'wa-2', 'wa-3', 'wa-4')"
    ).fetchall()
    assert len(rows) >= 4
    for row in rows:
        metadata_json = row["body"].split("[Intake metadata]\n", 1)[1]
        metadata = json.loads(metadata_json)
        assert metadata["source_message_id"] == row["source_message_id"]
        assert metadata["source_unit"]["unit_id"] == row["source_message_id"]
        assert metadata["source_unit"]["source_type"] == "user_text"


def test_kanban_intake_captures_hum743_live_multitask_batch(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    kanban_db.init_db(db_path)

    units = [
        {
            "unit_id": "wa-hotel",
            "source_message_id": "wa-hotel",
            "source_type": "user_text",
            "text": "Confirm the hotel again on saturday",
        },
        {
            "unit_id": "wa-film",
            "source_message_id": "wa-film",
            "source_type": "user_text",
            "text": "i think I found someone to film on wed",
        },
        {
            "unit_id": "wa-bananas",
            "source_message_id": "wa-bananas",
            "source_type": "user_text",
            "text": "we need bananas",
        },
        {
            "unit_id": "wa-return",
            "source_message_id": "wa-return",
            "source_type": "user_text",
            "text": "i want to know when we are coming back on sunday",
        },
        {
            "unit_id": "wa-dinner",
            "source_message_id": "wa-dinner",
            "source_type": "user_text",
            "text": "i think renata said she wanted to do a dinner for her sister here at the house on monday",
        },
    ]

    cards = kanban_intake.capture_inbound(
        "\n".join(unit["text"] for unit in units),
        session_key="whatsapp:zion",
        event_message_id="wa-live-batch",
        provenance_units=units,
    )

    action_cards = [card for card in cards if card["card_type"] == "action"]
    action_titles = [card["title"].lower() for card in action_cards]
    action_sources = {card["source_message_id"] for card in action_cards}
    assert {"wa-hotel", "wa-film", "wa-bananas", "wa-return", "wa-dinner"}.issubset(action_sources)
    assert any("hotel" in title and "saturday" in title for title in action_titles)
    assert any("film" in title and "wed" in title for title in action_titles)
    assert any("bananas" in title for title in action_titles)
    assert any("coming back on sunday" in title for title in action_titles)
    assert any("dinner" in title and "sister" in title and "monday" in title for title in action_titles)


def test_kanban_intake_captures_fast_fact_burst_grocery_preference(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    kanban_db.init_db(db_path)

    units = [
        {
            "unit_id": "msg-zaya",
            "source_message_id": "msg-zaya",
            "source_type": "user_text",
            "text": "Zaya’s full legal name: Zaya De Melo Kim",
        },
        {
            "unit_id": "msg-mom",
            "source_message_id": "msg-mom",
            "source_type": "user_text",
            "text": "My mom’s full name: Bernadete Regina Barboza de Melo",
        },
        {
            "unit_id": "msg-passport",
            "source_message_id": "msg-passport",
            "source_type": "user_text",
            "text": "Passport: FX019281",
        },
        {
            "unit_id": "msg-grocery",
            "source_message_id": "msg-grocery",
            "source_type": "user_text",
            "text": "I usually need bananas + 0% yogurt here",
        },
    ]

    cards = kanban_intake.capture_inbound(
        "\n".join(unit["text"] for unit in units),
        session_key="whatsapp:zion",
        event_message_id="wa-fast-facts",
        provenance_units=units,
    )

    preference_cards = [card for card in cards if card["card_type"] == "household_preference"]
    assert any(
        card["source_message_id"] == "msg-grocery"
        and "bananas" in card["title"].lower()
        and "0% yogurt" in card["title"].lower()
        for card in preference_cards
    )


def test_kanban_intake_ignores_screenshot_ui_ocr_fragments(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    kanban_db.init_db(db_path)

    cards = kanban_intake.capture_inbound(
        "Screenshot description: outgoing green bubbles are visible. "
        "Do: bubbles and outgoing green ones. Do: composer. "
        "Do: delivery status. Do: There is a gray/white status line with an hourglass. "
        "I still have to figure out when I can book my video shoot, then loop in renata with her calendar.",
        session_key="whatsapp:zion",
        event_message_id="ocr-noise",
    )

    titles = [c["title"].lower() for c in cards]
    assert any("video shoot" in title for title in titles)
    assert not any("bubbles" in title for title in titles)
    assert not any("composer" in title for title in titles)
    assert not any("delivery status" in title for title in titles)
    assert not any("hourglass" in title for title in titles)


def test_kanban_intake_does_not_fallback_untrusted_provenance_units(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    kanban_db.init_db(db_path)

    cards = kanban_intake.capture_inbound(
        "please book the bubbles and outgoing green ones",
        session_key="whatsapp:zion",
        event_message_id="image-caption",
        provenance_units=[
            {
                "unit_id": "image-caption",
                "source_message_id": "image-caption",
                "source_type": "image_caption",
                "trusted_for_intake": False,
                "text": "please book the bubbles and outgoing green ones",
            }
        ],
    )

    assert cards == []
