import asyncio
import inspect
import os
import time
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

from slate_client import CortexClient

load_dotenv()

# Configuration
GEMINI_MODEL = "gemini-2.5-flash"

SLATE_ADDRESS = os.getenv("SLATE_ADDRESS", "localhost:50051")
SLATE_TOKEN = os.getenv("SLATE_TOKEN", "")


class AgentLogger:
    def __init__(self, callback: Callable[[dict], Any]):
        self.callback = callback

    async def log(
        self, agent_name: str, event_type: str, content: str, details: Any = None
    ):
        """
        event_type: 'activation', 'thinking', 'tool_call', 'tool_result',
                    'slate_call', 'output', 'input'
        """
        data = {
            "agent": agent_name,
            "type": event_type,
            "content": content,
            "details": details,
        }
        try:
            if inspect.iscoroutinefunction(self.callback):
                await self.callback(data)
            else:
                self.callback(data)
        except Exception as e:
            print(f"AgentLogger Error: {e}")


class MultiAgentSystem:
    def __init__(self, run_id: str, logger: AgentLogger):
        self.run_id = run_id
        self.logger = logger

        # Initialize Slate Client
        self.slate = CortexClient(
            address=SLATE_ADDRESS, token=SLATE_TOKEN, run_id=run_id
        )

        # Initialize Gemini Client
        api_key = os.getenv("GEMINI_API_KEY")
        vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT")

        if vertex_project and os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "true":
            self.genai_client = genai.Client(
                vertexai=True,
                project=vertex_project,
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
        else:
            self.genai_client = genai.Client(api_key=api_key)

    async def _slate_call(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def process_message(self, user_message: str):
        try:
            # 1. User Input
            await self.logger.log("System", "input", f"User: {user_message}")

            # 2. Commit User Input to Echoes
            start = time.time()
            await self._slate_call(
                self.slate.commit, input=user_message, outcome="", agent_id="user"
            )
            duration = (time.time() - start) * 1000
            await self.logger.log(
                "System",
                "slate_call",
                f"Committing user input to Echoes (Memory) ({duration:.2f}ms)",
            )

            # Define Tools Wrapper (Note: These must be synchronous for Gemini SDK, but we wrap the calls internally if possible?
            # NO, Gemini SDK tool calls are synchronous in the loop I wrote.
            # I call `tools_map[fn_name](...)`.
            # I need to make my tool implementations use async?
            # But `_run_agent_loop` calls them synchronously: `result = tools_map[fn_name](**args_dict)`.
            # So I should make `_run_agent_loop` await them if they are coroutines.
            # Or I keep them synchronous but use `run_in_executor` INSIDE them?
            # `run_in_executor` is async (awaitable). I cannot await inside a sync function.
            # So I must make tools Async.
            # And `_run_agent_loop` must await them.
            # Let's do that.

            async def remember(content: str) -> str:
                """
                Stores a piece of information in working memory (Flux).
                Use this to keep track of important details, context,
                or to DELEGATE tasks to the Specialist.
                If delegating, start content with "DELEGATE:".
                """
                try:
                    t_start = time.time()
                    resp = await self._slate_call(self.slate.focus, content)
                    t_dur = (time.time() - t_start) * 1000
                    return f"Stored. ID: {resp.id} ({t_dur:.2f}ms)"
                except Exception as e:
                    return f"Error: {e}"

            async def recall_context() -> str:
                """
                Retrieves the current working memory (Flux) sorted by relevance.
                Call this to see what you were working on or to get context.
                """
                try:
                    t_start = time.time()
                    resp = await self._slate_call(self.slate.drift)
                    t_dur = (time.time() - t_start) * 1000
                    if not hasattr(resp, "items") or not resp.items:
                        return f"Memory is empty. ({t_dur:.2f}ms)"
                    items = [
                        f"- {item.content} ({item.relevance:.2f})"
                        for item in resp.items
                    ]
                    return "\n".join(items) + f"\n(Latency: {t_dur:.2f}ms)"
                except Exception as e:
                    return f"Error: {e}"

            async def save_experience(action: str, outcome: str) -> str:
                """
                Saves an interaction to long-term memory (Echoes).
                Use this after completing a significant step or action.
                """
                try:
                    t_start = time.time()
                    await self._slate_call(
                        self.slate.commit,
                        input=action,
                        outcome=outcome,
                        action=action,
                        agent_id="specialist",
                    )
                    t_dur = (time.time() - t_start) * 1000
                    return f"Experience saved. ({t_dur:.2f}ms)"
                except Exception as e:
                    return f"Error: {e}"

            async def search_history(query: str) -> str:
                """
                Searches long-term memory (Echoes) for past similar experiences.
                Use this before acting to see if we've done this before.
                """
                try:
                    t_start = time.time()
                    resp = await self._slate_call(self.slate.reminisce, query, limit=3)
                    t_dur = (time.time() - t_start) * 1000
                    if not hasattr(resp, "traces") or not resp.traces:
                        return f"No relevant past experiences found. ({t_dur:.2f}ms)"
                    traces = [
                        f"- Action: {t.action} | Outcome: {t.outcome}"
                        for t in resp.traces
                    ]
                    return "\n".join(traces) + f"\n(Latency: {t_dur:.2f}ms)"
                except Exception as e:
                    return f"Error: {e}"

            # --- Phase 1: Manager Agent ---
            await self.logger.log("Manager", "activation", "Manager agent active")

            manager_tools = [remember, search_history]
            manager_sys_instruct = """
You are the Manager Agent.
Your goal is to handle the user's request.
1. Search history (`search_history`) to see if we've handled similar requests.
2. If simple, Answer directly.
3. If complex, DELEGATE to the Specialist by using `remember`
   with "DELEGATE: <task details>".
Do NOT execute complex tasks yourself.
"""

            # Initial prompt
            chat_history = []  # noqa: F841

            # Run Manager Loop
            # We use a manual loop to handle tool calls and delegation detection
            manager_response = await self._run_agent_loop(
                agent_name="Manager",
                model=GEMINI_MODEL,
                system_instruction=manager_sys_instruct,
                tools=manager_tools,
                prompt=user_message,
                tools_map={"remember": remember, "search_history": search_history},
            )

            # Check if delegation happened
            # We check if "DELEGATE:" string appears in Flux (via drift).
            # Simpler: check if `remember` was called with DELEGATE during the loop.
            # But `_run_agent_loop` returns the final text.
            # Let's check the context (Flux) to see if there is a pending task.

            # Read Flux to see if there is a DELEGATE task
            start_drift = time.time()
            drift_resp = await self._slate_call(self.slate.drift)
            dur_drift = (time.time() - start_drift) * 1000
            await self.logger.log(
                "System",
                "slate_call",
                f"Checking for delegation in Flux ({dur_drift:.2f}ms)",
            )

            delegated_task = None
            if hasattr(drift_resp, "items"):
                completed_tasks = set()
                for item in drift_resp.items:
                    if item.content.startswith("Task Completed: "):
                        completed_tasks.add(
                            item.content.replace("Task Completed: ", "")
                        )

                for item in drift_resp.items:
                    if item.content.startswith("DELEGATE:"):
                        if item.content not in completed_tasks:
                            delegated_task = item.content
                            break

            if delegated_task:
                await self.logger.log(
                    "Manager", "output", f"Delegating task: {delegated_task}"
                )

                # --- Phase 2: Specialist Agent ---
                await self.logger.log(
                    "Specialist", "activation", "Specialist agent active"
                )

                specialist_tools = [recall_context, save_experience, search_history]
                specialist_sys_instruct = """
You are the Specialist Agent.
1. Start by calling `recall_context` to see the delegated task (look for "DELEGATE:").
2. Execute the task.
3. Save your result using `save_experience`.
4. Return the final answer to the user.
"""

                specialist_response = await self._run_agent_loop(
                    agent_name="Specialist",
                    model=GEMINI_MODEL,
                    system_instruction=specialist_sys_instruct,
                    tools=specialist_tools,
                    prompt="The Manager has delegated a task to you. "
                    "Check context and execute.",
                    tools_map={
                        "recall_context": recall_context,
                        "save_experience": save_experience,
                        "search_history": search_history,
                    },
                )

                # Mark task as done in Flux to prevent re-execution in future turns
                await self._slate_call(
                    self.slate.focus, f"Task Completed: {delegated_task}"
                )

                return specialist_response
            else:
                return manager_response

        except Exception as e:
            error_msg = str(e)
            if (
                "Connection refused" in error_msg
                or "StatusCode.UNAVAILABLE" in error_msg
            ):
                error_msg = (
                    f"Could not connect to Slate server at {SLATE_ADDRESS}. "
                    "Please ensure the server is running and accessible."
                )

            await self.logger.log("System", "error", error_msg)
            return f"Error: {error_msg}"

    async def _run_agent_loop(
        self, agent_name, model, system_instruction, tools, prompt, tools_map
    ):
        """
        Executes the Gemini model with manual tool handling loop.
        """
        config = types.GenerateContentConfig(
            tools=tools,
            system_instruction=system_instruction,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        chat = self.genai_client.chats.create(model=model, config=config)

        # Initial message
        await self.logger.log(agent_name, "thinking", "Processing...")

        # We need to handle the turn loop manually
        # Send message
        response = chat.send_message(prompt)

        max_turns = 10
        current_turn = 0

        while current_turn < max_turns:
            current_turn += 1

            # Check for function calls
            if response.function_calls:
                for func_call in response.function_calls:
                    fn_name = func_call.name or "unknown"
                    fn_args = func_call.args

                    await self.logger.log(
                        agent_name, "tool_call", f"{fn_name}({fn_args})"
                    )

                    # Execute tool
                    if fn_name in tools_map:
                        try:
                            # Convert args to dict
                            args_dict = {k: v for k, v in fn_args.items()}  # ty:ignore[possibly-missing-attribute]
                            # Check if tool is async
                            tool_func = tools_map[fn_name]
                            if inspect.iscoroutinefunction(tool_func):
                                result = await tool_func(**args_dict)
                            else:
                                result = tool_func(**args_dict)
                        except Exception as e:
                            result = f"Error executing tool: {e}"
                    else:
                        result = f"Error: Tool {fn_name} not found."

                    await self.logger.log(agent_name, "tool_result", str(result))

                    # Feed result back to model
                    # Using the function response part
                    func_resp_part = types.Part.from_function_response(
                        name=fn_name, response={"result": result}
                    )

                    # Send tool response
                    response = chat.send_message([func_resp_part])
            else:
                # Text response
                text_content = response.text
                if text_content:
                    await self.logger.log(agent_name, "output", text_content)
                    return text_content
                else:
                    # Empty response?
                    return "No response generated."

        return "Max turns reached."
