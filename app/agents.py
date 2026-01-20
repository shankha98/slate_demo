import asyncio
import inspect
import os
import time
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rice_sdk import Client

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


class SingleAgentSystem:
    def __init__(
        self,
        run_id: str,
        logger: AgentLogger,
        gemini_key: str | None = None,
        slate_token: str | None = None,
        slate_address: str | None = None,
    ):
        self.run_id = run_id
        self.logger = logger

        # Configuration Priorities:
        # 1. Passed arguments (from Frontend/WebSocket)
        # 2. Environment Variables (from .env)

        self.slate_address = slate_address or SLATE_ADDRESS
        self.slate_token = slate_token or SLATE_TOKEN
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")

        # Initialize Rice Client
        # We set environment variables so Client().connect() can find them.
        if self.slate_address:
            os.environ["STATE_INSTANCE_URL"] = self.slate_address
        if self.slate_token:
            os.environ["STATE_AUTH_TOKEN"] = self.slate_token

        # Initialize Rice Client
        self.client = Client(run_id=run_id)
        self.client.connect()

        # Initialize Gemini Client
        api_key = self.gemini_key
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

            # Define Tools Wrapper
            async def remember_fact(fact: str, topic: str) -> str:
                """
                Stores a specific fact or piece of information into long-term memory.
                Use this when the user tells you something new that you should remember.
                """
                try:
                    t_start = time.time()
                    # We map this to Slate's 'commit' (Echoes)
                    await self._slate_call(
                        self.client.state.commit,
                        input_text=topic,  # We use topic as input key
                        output=fact,
                        action="remember_fact",
                        agent_id="assistant",
                    )
                    t_dur = (time.time() - t_start) * 1000
                    return f"Fact remembered: '{fact}' ({t_dur:.2f}ms)"
                except Exception as e:
                    return f"Error remembering fact: {e}"

            async def recall_facts(topic: str) -> str:
                """
                Searches long-term memory for facts related to a specific topic.
                Use this when you need to answer a question based on past conversations.
                """
                try:
                    t_start = time.time()
                    # We map this to Slate's 'reminisce'
                    traces = await self._slate_call(
                        self.client.state.reminisce, query=topic, limit=5
                    )
                    t_dur = (time.time() - t_start) * 1000

                    if not traces:
                        return (
                            f"No relevant facts found about '{topic}'. ({t_dur:.2f}ms)"
                        )

                    facts = [f"- {t.outcome}" for t in traces]
                    return "\n".join(facts) + f"\n(Latency: {t_dur:.2f}ms)"
                except Exception as e:
                    return f"Error recalling facts: {e}"

            # Agent Execution
            await self.logger.log("Assistant", "activation", "Assistant active")

            tools = [remember_fact, recall_facts]
            system_instruction = """
You are a helpful AI assistant with long-term memory powered by Slate.
- When the user tells you something, use `remember_fact` to store it.
- When the user asks a question, use `recall_facts` to search your memory first.
- Always check memory before saying you don't know.
"""

            # Run Agent Loop
            response = await self._run_agent_loop(
                agent_name="Assistant",
                model=GEMINI_MODEL,
                system_instruction=system_instruction,
                tools=tools,
                prompt=user_message,
                tools_map={
                    "remember_fact": remember_fact,
                    "recall_facts": recall_facts,
                },
            )

            return response

        except Exception as e:
            error_msg = str(e)
            await self.logger.log("System", "error", error_msg)
            return f"Error: {error_msg}"

    async def _run_agent_loop(
        self, agent_name, model, system_instruction, tools, prompt, tools_map
    ):
        config = types.GenerateContentConfig(
            tools=tools,
            system_instruction=system_instruction,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        chat = self.genai_client.chats.create(model=model, config=config)
        await self.logger.log(agent_name, "thinking", "Processing...")

        response = chat.send_message(prompt)
        max_turns = 10
        current_turn = 0

        while current_turn < max_turns:
            current_turn += 1
            if response.function_calls:
                for func_call in response.function_calls:
                    fn_name = func_call.name or "unknown"
                    fn_args = func_call.args
                    await self.logger.log(
                        agent_name, "tool_call", f"{fn_name}({fn_args})"
                    )
                    tool_start = time.time()
                    if fn_name in tools_map:
                        try:
                            args_dict = {k: v for k, v in fn_args.items()}  # ty:ignore[possibly-missing-attribute]
                            tool_func = tools_map[fn_name]
                            if inspect.iscoroutinefunction(tool_func):
                                result = await tool_func(**args_dict)
                            else:
                                result = tool_func(**args_dict)
                        except Exception as e:
                            result = f"Error executing tool: {e}"
                    else:
                        result = f"Error: Tool {fn_name} not found."
                    tool_latency_ms = (time.time() - tool_start) * 1000
                    await self.logger.log(
                        agent_name,
                        "tool_result",
                        str(result),
                        details={
                            "latency_ms": round(tool_latency_ms, 2),
                            "tool": fn_name,
                        },
                    )
                    func_resp_part = types.Part.from_function_response(
                        name=fn_name, response={"result": result}
                    )
                    response = chat.send_message([func_resp_part])
            else:
                text_content = response.text
                if text_content:
                    await self.logger.log(agent_name, "output", text_content)
                    return text_content
                else:
                    return "No response generated."
        return "Max turns reached."
