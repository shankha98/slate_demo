from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents import AgentLogger, SingleAgentSystem


@pytest.mark.asyncio
async def test_agent_process_simple_message(run_id):
    """Test SingleAgentSystem processes a simple message without tools."""
    mock_log_callback = AsyncMock()
    logger = AgentLogger(mock_log_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value
        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        # Scenario: User says "Hello", Assistant answers directly
        mock_response = MagicMock()
        mock_response.function_calls = []
        mock_response.text = "Hello there!"
        mock_chat.send_message.return_value = mock_response

        system = SingleAgentSystem(run_id, logger)
        response = await system.process_message("Hello")

        assert response == "Hello there!"
        assert mock_log_callback.call_count > 0


@pytest.mark.asyncio
async def test_agent_remember_fact(run_id):
    """Test SingleAgentSystem stores a fact using remember_fact tool."""
    mock_log_callback = AsyncMock()
    logger = AgentLogger(mock_log_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value
        mock_slate.commit.return_value = MagicMock(id="test-commit-id")
        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        # Scenario: Agent calls remember_fact tool
        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "remember_fact"
        func_call.args = {"fact": "User likes pizza", "topic": "preferences"}
        msg1.function_calls = [func_call]
        msg1.text = None

        # After tool result, agent responds
        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "I'll remember that you like pizza!"

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = SingleAgentSystem(run_id, logger)
        response = await system.process_message("I really like pizza")

        assert response == "I'll remember that you like pizza!"
        mock_slate.commit.assert_called_once()


@pytest.mark.asyncio
async def test_agent_recall_facts_cold_start(run_id):
    """Test SingleAgentSystem recall_facts returns empty on cold start."""
    mock_log_callback = AsyncMock()
    logger = AgentLogger(mock_log_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value
        # Simulate empty traces (cold start)
        mock_slate.reminisce.return_value = MagicMock(traces=[])

        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        # Agent calls recall_facts
        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "recall_facts"
        func_call.args = {"topic": "preferences"}
        msg1.function_calls = [func_call]
        msg1.text = None

        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "I don't have any information about your preferences yet."

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = SingleAgentSystem(run_id, logger)
        response = await system.process_message("What do you know about my preferences?")

        assert "don't have any information" in response.lower()


@pytest.mark.asyncio
async def test_agent_recall_facts_hit(run_id):
    """Test SingleAgentSystem recall_facts returns stored facts."""
    mock_log_callback = AsyncMock()
    logger = AgentLogger(mock_log_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value

        # Simulate returning a trace
        trace = MagicMock()
        trace.outcome = "User likes pizza"
        mock_slate.reminisce.return_value = MagicMock(traces=[trace])

        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "recall_facts"
        func_call.args = {"topic": "preferences"}
        msg1.function_calls = [func_call]
        msg1.text = None

        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "I remember you like pizza!"

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = SingleAgentSystem(run_id, logger)
        response = await system.process_message("What do you know about my preferences?")

        assert "pizza" in response.lower()


@pytest.mark.asyncio
async def test_agent_error_handling(run_id):
    """Test SingleAgentSystem handles errors gracefully."""
    mock_log_callback = AsyncMock()
    logger = AgentLogger(mock_log_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value
        # Simulate Slate connection error
        mock_slate.reminisce.side_effect = Exception("Connection failed")

        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "recall_facts"
        func_call.args = {"topic": "test"}
        msg1.function_calls = [func_call]
        msg1.text = None

        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "I had trouble accessing my memory."

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = SingleAgentSystem(run_id, logger)
        response = await system.process_message("What do you remember?")

        # Agent should still respond despite tool error
        assert response is not None
