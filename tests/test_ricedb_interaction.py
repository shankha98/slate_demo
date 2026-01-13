from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents import AgentLogger, MultiAgentSystem


@pytest.mark.asyncio
async def test_search_history_cold_start(run_id):
    """Test Case 1: RiceDB returns no results (Cold Start)."""
    # Mock Logger
    mock_callback = AsyncMock()
    logger = AgentLogger(mock_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value
        # Simulate RiceDB returning empty traces
        mock_slate.reminisce.return_value = MagicMock(traces=[])
        # Also need to mock drift to prevent delegation and keep Manager active
        mock_slate.drift.return_value = MagicMock(items=[])

        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        # Scenario: Agent calls search_history
        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "search_history"
        func_call.args = {"query": "test query"}
        msg1.function_calls = [func_call]
        msg1.text = None

        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "Done"

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = MultiAgentSystem(run_id, logger)
        await system.process_message("test input")

        # Check logger mock for expected tool result
        found_tool_result = False
        for call in mock_callback.call_args_list:
            data = call.args[0]
            # Use 'in' to match message + latency
            if (
                data["type"] == "tool_result"
                and "No relevant past experiences found." in data["content"]
            ):
                found_tool_result = True
                break

        assert found_tool_result, (
            "Expected 'No relevant past experiences found.' in logs."
        )


@pytest.mark.asyncio
async def test_search_history_hit(run_id):
    """Test Case 2: RiceDB returns valid traces."""
    mock_callback = AsyncMock()
    logger = AgentLogger(mock_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value
        mock_slate.drift.return_value = MagicMock(items=[])

        # Simulate RiceDB returning a trace
        trace = MagicMock()
        trace.action = "Test Action"
        trace.outcome = "Test Outcome"
        mock_slate.reminisce.return_value = MagicMock(traces=[trace])

        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "search_history"
        func_call.args = {"query": "test query"}
        msg1.function_calls = [func_call]
        msg1.text = None

        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "Done"

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = MultiAgentSystem(run_id, logger)
        await system.process_message("test input")

        expected_result = "- Action: Test Action | Outcome: Test Outcome"

        found_tool_result = False
        for call in mock_callback.call_args_list:
            data = call.args[0]
            if data["type"] == "tool_result" and expected_result in data["content"]:
                found_tool_result = True
                break

        assert found_tool_result, f"Expected '{expected_result}' in tool result."


@pytest.mark.asyncio
async def test_search_history_error(run_id):
    """Test Case 3: RiceDB connection failure or error."""
    mock_callback = AsyncMock()
    logger = AgentLogger(mock_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value
        mock_slate.drift.return_value = MagicMock(items=[])

        # Simulate RiceDB Error
        mock_slate.reminisce.side_effect = Exception("ACL error")

        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "search_history"
        func_call.args = {"query": "test query"}
        msg1.function_calls = [func_call]
        msg1.text = None

        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "Done"

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = MultiAgentSystem(run_id, logger)
        await system.process_message("test input")

        expected_result = "Error: ACL error"

        found_tool_result = False
        for call in mock_callback.call_args_list:
            data = call.args[0]
            if data["type"] == "tool_result" and expected_result in data["content"]:
                found_tool_result = True
                break

        assert found_tool_result, f"Expected error message '{expected_result}' in logs."
