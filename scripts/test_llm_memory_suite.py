#!/usr/bin/env python3
"""
Standalone Memory Test with LLM + Slate

This script tests memory scenarios by directly using Google GenAI SDK
with Slate (CortexClient) as tools. Does NOT require the FastAPI server.

Usage:
    # Run a specific test
    uv run scripts/test_llm_memory_suite.py test_data/01_basic_identity.json

    # Run all tests
    uv run scripts/test_llm_memory_suite.py --all

    # Run with verbose output (shows tool calls)
    uv run scripts/test_llm_memory_suite.py --all -v

    # Use a specific model
    uv run scripts/test_llm_memory_suite.py --all --model gemini-2.0-flash

Environment Variables:
    GEMINI_API_KEY      - Google Gemini API key (required)
    SLATE_ADDRESS       - Slate server address (default: localhost:50051)
    SLATE_TOKEN         - Slate authentication token
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from slate_client import CortexClient

load_dotenv()

# Defaults
DEFAULT_MODEL = "gemini-2.5-flash"
SLATE_ADDRESS = os.getenv("SLATE_ADDRESS", "localhost:50051")
SLATE_TOKEN = os.getenv("SLATE_TOKEN", "")


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


class MemoryTestAgent:
    """
    A standalone agent that uses Google GenAI with Slate memory tools.
    """

    def __init__(
        self,
        run_id: str,
        model: str = DEFAULT_MODEL,
        verbose: bool = False,
    ):
        self.run_id = run_id
        self.model = model
        self.verbose = verbose
        self.tool_calls: list[dict] = []

        # Initialize Slate Client
        self.slate = CortexClient(
            address=SLATE_ADDRESS,
            token=SLATE_TOKEN,
            run_id=run_id,
        )

        # Initialize Gemini Client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        self.genai_client = genai.Client(api_key=api_key)

        # System instruction for the agent
        self.system_instruction = """
You are a helpful AI assistant with long-term memory powered by Slate.

IMPORTANT RULES:
1. When the user tells you personal information, facts, or anything worth remembering, 
   use `remember_fact` to store it in your memory.
2. When the user asks a question about something they may have told you before, 
   use `recall_facts` to search your memory FIRST before responding.
3. Always check your memory before saying "I don't know" or "I don't have that information".
4. Be concise in your responses.
5. If memory returns relevant facts, use them to answer the question accurately.
"""

    def _remember_fact(self, fact: str, topic: str) -> str:
        """
        Stores a fact in Slate long-term memory.

        Args:
            fact: The specific information to remember.
            topic: A short topic/category for the fact.
        """
        try:
            t_start = time.time()
            self.slate.commit(
                input=topic,
                outcome=fact,
                action="remember_fact",
                agent_id="assistant",
            )
            latency_ms = (time.time() - t_start) * 1000

            self.tool_calls.append(
                {
                    "tool": "remember_fact",
                    "args": {"fact": fact, "topic": topic},
                    "latency_ms": round(latency_ms, 2),
                    "result": "success",
                }
            )

            if self.verbose:
                print(
                    f"   {Colors.GREEN}📝 Stored:{Colors.RESET} '{fact[:60]}...' [{latency_ms:.1f}ms]"
                )

            return f"Fact stored successfully: '{fact}'"
        except Exception as e:
            self.tool_calls.append(
                {
                    "tool": "remember_fact",
                    "args": {"fact": fact, "topic": topic},
                    "error": str(e),
                }
            )
            return f"Error storing fact: {e}"

    def _recall_facts(self, topic: str) -> str:
        """
        Searches Slate memory for facts related to a topic.

        Args:
            topic: The topic or query to search for.
        """
        try:
            t_start = time.time()
            resp = self.slate.reminisce(topic, limit=5)
            latency_ms = (time.time() - t_start) * 1000

            if not hasattr(resp, "traces") or not resp.traces:
                self.tool_calls.append(
                    {
                        "tool": "recall_facts",
                        "args": {"topic": topic},
                        "latency_ms": round(latency_ms, 2),
                        "result": "no_results",
                    }
                )

                if self.verbose:
                    print(
                        f"   {Colors.YELLOW}🔍 Recall:{Colors.RESET} '{topic}' → No results [{latency_ms:.1f}ms]"
                    )

                return f"No relevant facts found about '{topic}'."

            facts = [f"- {t.outcome}" for t in resp.traces]
            result = "\n".join(facts)

            self.tool_calls.append(
                {
                    "tool": "recall_facts",
                    "args": {"topic": topic},
                    "latency_ms": round(latency_ms, 2),
                    "result": f"{len(resp.traces)} facts found",
                }
            )

            if self.verbose:
                print(
                    f"   {Colors.CYAN}🔍 Recall:{Colors.RESET} '{topic}' → {len(resp.traces)} facts [{latency_ms:.1f}ms]"
                )
                for fact in facts[:3]:
                    print(f"      {Colors.DIM}{fact[:70]}...{Colors.RESET}")

            return result
        except Exception as e:
            self.tool_calls.append(
                {
                    "tool": "recall_facts",
                    "args": {"topic": topic},
                    "error": str(e),
                }
            )
            return f"Error recalling facts: {e}"

    def process_message(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.
        """
        self.tool_calls = []  # Reset tool calls for this turn

        # Define tools for Gemini
        tools = [self._remember_fact, self._recall_facts]

        config = types.GenerateContentConfig(
            tools=tools,
            system_instruction=self.system_instruction,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True  # We handle function calling manually
            ),
        )

        chat = self.genai_client.chats.create(model=self.model, config=config)
        response = chat.send_message(user_message)

        # Agent loop - handle function calls
        max_turns = 10
        for _ in range(max_turns):
            if response.function_calls:
                for func_call in response.function_calls:
                    fn_name = func_call.name or "unknown"
                    fn_args = dict(func_call.args) if func_call.args else {}

                    # Execute the function
                    if fn_name == "remember_fact":
                        result = self._remember_fact(**fn_args)
                    elif fn_name == "recall_facts":
                        result = self._recall_facts(**fn_args)
                    else:
                        result = f"Unknown function: {fn_name}"

                    # Send function response back to model
                    func_resp_part = types.Part.from_function_response(
                        name=fn_name,
                        response={"result": result},
                    )
                    response = chat.send_message([func_resp_part])
            else:
                # No more function calls, return the text response
                return response.text or "No response generated."

        return "Max turns reached without final response."


def run_test_scenario(
    filepath: Path,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    pause_between_turns: float = 0.5,
) -> tuple[bool, str, list[dict]]:
    """
    Run a single test scenario from a JSON file.

    Returns: (success, summary, results)
    """
    # Unique run_id for this test scenario
    run_id = f"test-{filepath.stem}-{uuid.uuid4().hex[:8]}"

    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}📋 Test:{Colors.RESET} {filepath.name}")
    print(f"{Colors.BOLD}🧠 Memory ID:{Colors.RESET} {run_id}")
    print(f"{Colors.BOLD}🤖 Model:{Colors.RESET} {model}")
    print(f"{'=' * 70}")

    try:
        with open(filepath, "r") as f:
            messages = json.load(f)
    except FileNotFoundError:
        return False, f"File not found: {filepath}", []
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", []

    # Create the agent
    try:
        agent = MemoryTestAgent(run_id=run_id, model=model, verbose=verbose)
    except Exception as e:
        return False, f"Failed to create agent: {e}", []

    results = []

    try:
        for i, user_msg in enumerate(messages, 1):
            print(f"\n{Colors.BLUE}--- Turn {i}/{len(messages)} ---{Colors.RESET}")
            print(f"{Colors.MAGENTA}👤 User:{Colors.RESET} {user_msg}")

            t_start = time.time()
            bot_response = agent.process_message(user_msg)
            total_time = (time.time() - t_start) * 1000

            print(f"{Colors.GREEN}🤖 Bot:{Colors.RESET} {bot_response}")

            if verbose:
                print(
                    f"   {Colors.DIM}⏱️ Total: {total_time:.0f}ms | Tools: {len(agent.tool_calls)}{Colors.RESET}"
                )

            results.append(
                {
                    "turn": i,
                    "user": user_msg,
                    "bot": bot_response,
                    "tool_calls": agent.tool_calls.copy(),
                    "total_time_ms": round(total_time, 2),
                }
            )

            # Small pause to avoid rate limiting
            if i < len(messages):
                time.sleep(pause_between_turns)

        print(f"\n{Colors.GREEN}✅ Test completed: {filepath.name}{Colors.RESET}")
        return True, f"Completed {len(messages)} turns successfully", results

    except Exception as e:
        return False, f"Error during test: {e}", results


def run_all_tests(
    test_dir: Path,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run all JSON test files in the test_data directory."""
    results = {}

    test_files = sorted(test_dir.glob("*.json"))

    if not test_files:
        print(f"No test files found in {test_dir}")
        return results

    print(
        f"\n{Colors.BOLD}🚀 Running {len(test_files)} test scenarios with {model}{Colors.RESET}"
    )

    for filepath in test_files:
        success, summary, _ = run_test_scenario(filepath, model=model, verbose=verbose)
        results[filepath.name] = {"success": success, "summary": summary}

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"{Colors.BOLD}📊 TEST SUMMARY{Colors.RESET}")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r["success"])
    failed = len(results) - passed

    for name, result in results.items():
        status = (
            f"{Colors.GREEN}✅{Colors.RESET}"
            if result["success"]
            else f"{Colors.RED}❌{Colors.RESET}"
        )
        print(f"{status} {name}: {result['summary']}")

    color = Colors.GREEN if failed == 0 else Colors.YELLOW if passed > 0 else Colors.RED
    print(
        f"\n{color}🏁 Total: {passed} passed, {failed} failed out of {len(results)} tests{Colors.RESET}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Standalone memory test using LLM + Slate directly"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to a specific test JSON file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests in test_data directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed tool calls and results",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="Pause between turns in seconds (default: 0.5)",
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print(
            f"{Colors.RED}Error: GEMINI_API_KEY environment variable is required{Colors.RESET}"
        )
        print("Set it with: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)

    if args.all:
        test_dir = Path("test_data")
        if not test_dir.exists():
            print(f"Error: {test_dir} directory not found")
            sys.exit(1)
        run_all_tests(test_dir, model=args.model, verbose=args.verbose)
    elif args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: {filepath} not found")
            sys.exit(1)
        success, summary, _ = run_test_scenario(
            filepath,
            model=args.model,
            verbose=args.verbose,
            pause_between_turns=args.pause,
        )
        if not success:
            print(f"\n{Colors.RED}❌ Test failed: {summary}{Colors.RESET}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
