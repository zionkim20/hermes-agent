from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionContext, SessionSource, build_session_context_prompt, build_session_key
from gateway.stream_consumer import GatewayStreamConsumer


class _SilentResponseAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.WHATSAPP)
        self.sent_messages = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent_messages.append((chat_id, content, reply_to, metadata))
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


def test_group_prompt_includes_turn_taking_and_silent_policy():
    context = SessionContext(
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="120363408749876118@g.us",
            chat_name="Colin's visit",
            chat_type="group",
            user_name="Zion",
        ),
        connected_platforms=[Platform.WHATSAPP],
        home_channels={},
        shared_multi_user_session=False,
    )

    prompt = build_session_context_prompt(context)

    assert "overheard group instruction" in prompt
    assert "reply with exactly `[SILENT]`" in prompt
    assert "@mention" in prompt
    assert "permission to answer only the agent-relevant part" in prompt
    assert "already in the chat" in prompt
    assert "do not restate" in prompt


@pytest.mark.asyncio
async def test_silent_marker_suppresses_gateway_delivery():
    adapter = _SilentResponseAdapter()
    event = MessageEvent(
        text="Colin - bring bathing suit and comfortable clothes to walk",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="group-1@g.us",
            chat_name="Colin's visit",
            chat_type="group",
            user_id="zion",
            user_name="Zion",
        ),
        message_id="msg-1",
    )
    adapter._message_handler = AsyncMock(return_value="  [SILENT]  ")

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent_messages == []


@pytest.mark.asyncio
async def test_streamed_silent_marker_suppresses_gateway_delivery():
    adapter = _SilentResponseAdapter()
    consumer = GatewayStreamConsumer(adapter, "group-1@g.us")

    consumer.on_delta(" [SILENT] ")
    consumer.finish()
    await consumer.run()

    assert consumer.final_response_sent is True
    assert adapter.sent_messages == []
