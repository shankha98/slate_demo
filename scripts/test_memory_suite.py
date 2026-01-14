#!/usr/bin/env python3
"""
Comprehensive Memory Test Runner

Runs multiple test datasets against the Slate memory system to validate
persistence, recall, updates, and edge cases.

Usage:
    # Run a specific test
    uv run scripts/test_memory_suite.py test_data/01_basic_identity.json

    # Run all tests
    uv run scripts/test_memory_suite.py --all

    # Run all tests with verbose output
    uv run scripts/test_memory_suite.py --all --verbose
"""

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

import websockets


async def run_test_scenario(
    filepath: Path,
    verbose: bool = False,
    pause_between_turns: float = 1.0,
) -> tuple[bool, str]:
    """
    Run a single test scenario from a JSON file.

    Returns (success: bool, summary: str)
    """
    uri = "ws://localhost:8000/ws"

    # Unique run_id for this test scenario
    run_id = f"test-{filepath.stem}-{uuid.uuid4().hex[:8]}"

    print(f"\n{'=' * 60}")
    print(f"📋 Test: {filepath.name}")
    print(f"🧠 Memory ID: {run_id}")
    print("=" * 60)

    try:
        with open(filepath, "r") as f:
            messages = json.load(f)
    except FileNotFoundError:
        return False, f"File not found: {filepath}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    results = []

    try:
        for i, user_msg in enumerate(messages, 1):
            print(f"\n--- Turn {i}/{len(messages)} ---")
            print(f"👤 User: {user_msg}")

            # Generate a unique session_id for each connection
            session_id = f"session-{uuid.uuid4().hex[:8]}"
            uri_with_params = f"{uri}?run_id={run_id}&session_id={session_id}"

            async with websockets.connect(uri_with_params) as websocket:
                # Wait for session_info
                session_msg = await websocket.recv()
                session_data = json.loads(session_msg)

                if verbose:
                    print(
                        f"   📡 Session: {session_data.get('session_id')} | Memory: {session_data.get('run_id')}"  # noqa: E501
                    )

                # Send message
                await websocket.send(
                    json.dumps({"type": "message", "content": user_msg})
                )

                # Collect tool calls for verbose output
                tool_calls = []

                # Wait for response
                while True:
                    try:
                        response_str = await websocket.recv()
                        response = json.loads(response_str)

                        if response.get("type") == "final_response":
                            bot_response = response.get("content", "")
                            print(f"🤖 Bot: {bot_response}")
                            results.append(
                                {
                                    "turn": i,
                                    "user": user_msg,
                                    "bot": bot_response,
                                    "tool_calls": tool_calls,
                                }
                            )
                            break
                        elif response.get("agent") and verbose:
                            event_type = response.get("type", "")
                            content = response.get("content", "")
                            details = response.get("details", {})

                            if event_type == "tool_call":
                                tool_calls.append(content)
                                print(f"   🔧 Tool: {content}")
                            elif event_type == "tool_result":
                                latency = details.get("latency_ms", "?")
                                print(f"   ⚡ Result ({latency}ms): {content[:100]}...")

                    except websockets.exceptions.ConnectionClosed:
                        return False, f"Connection closed unexpectedly at turn {i}"

            await asyncio.sleep(pause_between_turns)

        print(f"\n✅ Test completed: {filepath.name}")
        return True, f"Completed {len(messages)} turns successfully"

    except Exception as e:
        return False, f"Error: {e}"


async def run_all_tests(test_dir: Path, verbose: bool = False) -> dict:
    """Run all JSON test files in the test_data directory."""
    results = {}

    test_files = sorted(test_dir.glob("*.json"))

    if not test_files:
        print(f"No test files found in {test_dir}")
        return results

    print(f"\n🚀 Running {len(test_files)} test scenarios...")

    for filepath in test_files:
        success, summary = await run_test_scenario(filepath, verbose=verbose)
        results[filepath.name] = {"success": success, "summary": summary}

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r["success"])
    failed = len(results) - passed

    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"{status} {name}: {result['summary']}")

    print(f"\n🏁 Total: {passed} passed, {failed} failed out of {len(results)} tests")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Run memory test scenarios")
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
        "--pause",
        type=float,
        default=1.0,
        help="Pause between turns in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    if args.all:
        test_dir = Path("test_data")
        if not test_dir.exists():
            print(f"Error: {test_dir} directory not found")
            sys.exit(1)
        await run_all_tests(test_dir, verbose=args.verbose)
    elif args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: {filepath} not found")
            sys.exit(1)
        success, summary = await run_test_scenario(
            filepath, verbose=args.verbose, pause_between_turns=args.pause
        )
        if not success:
            print(f"\n❌ Test failed: {summary}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
