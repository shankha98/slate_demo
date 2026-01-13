from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents import AgentLogger, MultiAgentSystem


@pytest.mark.asyncio
async def test_agent_process_simple_message(run_id):
    # Mock Logger
    mock_log_callback = AsyncMock()
    logger = AgentLogger(mock_log_callback)

    # Mock External Clients
    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        # Setup Slate Mock
        mock_slate = MockSlate.return_value
        mock_slate.focus.return_value = MagicMock(id="test-focus-id")
        mock_slate.drift.return_value = MagicMock(items=[])
        mock_slate.reminisce.return_value = MagicMock(traces=[])

        # Setup GenAI Mock
        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        # Scenario: User says "Hello"
        # Manager answers directly
        mock_response = MagicMock()
        mock_response.function_calls = []
        mock_response.text = "Hello there!"
        mock_chat.send_message.return_value = mock_response

        system = MultiAgentSystem(run_id, logger)
        response = await system.process_message("Hello")

        assert response == "Hello there!"

        # Verify Logs
        assert mock_log_callback.call_count > 0
        # Verify Slate Commit
        mock_slate.commit.assert_called()


@pytest.mark.asyncio
async def test_agent_delegation(run_id):
    mock_log_callback = AsyncMock()
    logger = AgentLogger(mock_log_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value
        mock_genai = MockGenAI.return_value

        manager_chat = MagicMock()
        specialist_chat = MagicMock()
        mock_genai.chats.create.side_effect = [manager_chat, specialist_chat]

        # Manager Flow
        # 1. Manager calls remember("DELEGATE: ...")
        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "remember"
        func_call.args = {"content": "DELEGATE: task"}
        msg1.function_calls = [func_call]
        msg1.text = None

        # 2. Manager receives tool result and says "Delegated"
        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "Delegated."

        manager_chat.send_message.side_effect = [msg1, msg2]

        # Specialist Flow
        # 1. Specialist answers "Done"
        msg3 = MagicMock()
        msg3.function_calls = []
        msg3.text = "Task Done"
        specialist_chat.send_message.return_value = msg3

        # Slate Drift Mocking
        item_delegate = MagicMock()
        item_delegate.content = "DELEGATE: task"
        item_delegate.relevance = 1.0

        resp_delegate = MagicMock()
        resp_delegate.items = [item_delegate]

        # We return the delegation item for all drift calls to ensure logic picks it up.
        # Calls are: 1. Manager Context, 2. Check Delegation, 3. Specialist Context (if needed)  # noqa: E501
        mock_slate.drift.side_effect = [
            resp_delegate,
            resp_delegate,
            resp_delegate,
        ]

        system = MultiAgentSystem(run_id, logger)
        response = await system.process_message("Complex Task")

        assert response == "Task Done"
        assert mock_slate.focus.called
