from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents import AgentLogger, SingleAgentSystem


@pytest.mark.asyncio
async def test_recall_facts_cold_start(run_id):
    """Test Case 1: Slate returns no results (Cold Start)."""
    mock_callback = AsyncMock()
    logger = AgentLogger(mock_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value
        # Simulate Slate returning empty traces
        mock_slate.reminisce.return_value = MagicMock(traces=[])

        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        # Scenario: Agent calls recall_facts
        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "recall_facts"
        func_call.args = {"topic": "test topic"}
        msg1.function_calls = [func_call]
        msg1.text = None

        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "Done"

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = SingleAgentSystem(run_id, logger)
        await system.process_message("test input")

        # Check logger mock for expected tool result
        found_tool_result = False
        for call in mock_callback.call_args_list:
            data = call.args[0]
            if (
                data["type"] == "tool_result"
                and "No relevant facts found" in data["content"]
            ):
                found_tool_result = True
                break

        assert found_tool_result, "Expected 'No relevant facts found' in logs."


@pytest.mark.asyncio
async def test_recall_facts_hit(run_id):
    """Test Case 2: Slate returns valid traces."""
    mock_callback = AsyncMock()
    logger = AgentLogger(mock_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value

        # Simulate Slate returning a trace
        trace = MagicMock()
        trace.outcome = "User prefers Python over JavaScript"
        mock_slate.reminisce.return_value = MagicMock(traces=[trace])

        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "recall_facts"
        func_call.args = {"topic": "programming languages"}
        msg1.function_calls = [func_call]
        msg1.text = None

        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "Done"

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = SingleAgentSystem(run_id, logger)
        await system.process_message("test input")

        expected_result = "User prefers Python over JavaScript"

        found_tool_result = False
        for call in mock_callback.call_args_list:
            data = call.args[0]
            if data["type"] == "tool_result" and expected_result in data["content"]:
                found_tool_result = True
                break

        assert found_tool_result, f"Expected '{expected_result}' in tool result."


@pytest.mark.asyncio
async def test_recall_facts_error(run_id):
    """Test Case 3: Slate connection failure or error."""
    mock_callback = AsyncMock()
    logger = AgentLogger(mock_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value

        # Simulate Slate Error
        mock_slate.reminisce.side_effect = Exception("Connection error")

        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "recall_facts"
        func_call.args = {"topic": "test topic"}
        msg1.function_calls = [func_call]
        msg1.text = None

        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "Done"

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = SingleAgentSystem(run_id, logger)
        await system.process_message("test input")

        expected_result = "Error recalling facts"

        found_tool_result = False
        for call in mock_callback.call_args_list:
            data = call.args[0]
            if data["type"] == "tool_result" and expected_result in data["content"]:
                found_tool_result = True
                break

        assert found_tool_result, f"Expected error message '{expected_result}' in logs."


@pytest.mark.asyncio
async def test_remember_fact_success(run_id):
    """Test Case 4: Successfully store a fact in Slate."""
    mock_callback = AsyncMock()
    logger = AgentLogger(mock_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value
        mock_slate.commit.return_value = MagicMock(id="commit-id-123")

        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "remember_fact"
        func_call.args = {"fact": "Lives in San Francisco", "topic": "location"}
        msg1.function_calls = [func_call]
        msg1.text = None

        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "Got it, I'll remember that."

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = SingleAgentSystem(run_id, logger)
        await system.process_message("I live in San Francisco")

        # Verify commit was called
        mock_slate.commit.assert_called_once_with(
            input="location",
            outcome="Lives in San Francisco",
            action="remember_fact",
            agent_id="assistant",
        )

        # Check for success message in logs
        found_tool_result = False
        for call in mock_callback.call_args_list:
            data = call.args[0]
            if (
                data["type"] == "tool_result"
                and "Fact remembered" in data["content"]
            ):
                found_tool_result = True
                break

        assert found_tool_result, "Expected 'Fact remembered' in logs."


@pytest.mark.asyncio
async def test_remember_fact_error(run_id):
    """Test Case 5: Error storing a fact in Slate."""
    mock_callback = AsyncMock()
    logger = AgentLogger(mock_callback)

    with (
        patch("app.agents.CortexClient") as MockSlate,
        patch("app.agents.genai.Client") as MockGenAI,
    ):
        mock_slate = MockSlate.return_value
        mock_slate.commit.side_effect = Exception("Write failed")

        mock_genai = MockGenAI.return_value
        mock_chat = MagicMock()
        mock_genai.chats.create.return_value = mock_chat

        msg1 = MagicMock()
        func_call = MagicMock()
        func_call.name = "remember_fact"
        func_call.args = {"fact": "Some fact", "topic": "test"}
        msg1.function_calls = [func_call]
        msg1.text = None

        msg2 = MagicMock()
        msg2.function_calls = []
        msg2.text = "Okay"

        mock_chat.send_message.side_effect = [msg1, msg2]

        system = SingleAgentSystem(run_id, logger)
        await system.process_message("Remember this")

        expected_result = "Error remembering fact"

        found_tool_result = False
        for call in mock_callback.call_args_list:
            data = call.args[0]
            if data["type"] == "tool_result" and expected_result in data["content"]:
                found_tool_result = True
                break

        assert found_tool_result, f"Expected error message '{expected_result}' in logs."
