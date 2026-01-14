import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from google import genai
from google.genai import types

from slate_client import CortexClient

# Load env vars
load_dotenv()

SLATE_ADDRESS = os.getenv("SLATE_ADDRESS", "localhost:50051")
SLATE_TOKEN = os.getenv("SLATE_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


async def main():
    print("Starting LLM + Slate Memory Test...")
    run_id = f"test-llm-memory-{uuid.uuid4().hex[:8]}"

    # 1. Initialize Clients
    slate_client = CortexClient(address=SLATE_ADDRESS, token=SLATE_TOKEN, run_id=run_id)
    genai_client = genai.Client(api_key=GEMINI_API_KEY)

    # 2. Define Tools for the LLM
    def remember(fact: str) -> str:
        """Stores a fact in long-term memory."""
        print(f"[Tool] Remembering: {fact}")
        slate_client.commit(
            input="fact", outcome=fact, action="remember", agent_id="tester"
        )
        return "Fact stored successfully."

    def recall(query: str) -> str:
        """Searches long-term memory."""
        print(f"[Tool] Recalling: {query}")
        resp = slate_client.reminisce(query, limit=3)
        if hasattr(resp, "traces"):
            print(f"  -> RiceDB returned {len(resp.traces)} traces:")
            for t in resp.traces:
                print(
                    f"     * {t.outcome} (score: {t.relevance if hasattr(t, 'relevance') else 'N/A'})"  # noqa: E501
                )

        if not hasattr(resp, "traces") or not resp.traces:
            return "No memories found."
        return "\n".join([t.outcome for t in resp.traces])

    tools = [remember, recall]

    # 3. Create Chat Session
    chat = genai_client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False
            ),
            system_instruction="You are a memory tester. Use the 'remember' tool to store facts provided by the user. Use the 'recall' tool to retrieve them when asked.",  # noqa: E501
        ),
    )

    # 4. Turn 1: Teach the LLM a fact
    secret_code = f"Omega-{uuid.uuid4().hex[:4]}"
    prompt1 = f"The secret code is {secret_code}. Please remember this."
    print(f"\nUser: {prompt1}")

    # Using synchronous methods for simplicity here as we are in a script,
    # but inside an async function we could use the async client.
    # However, google-genai client methods like send_message are synchronous unless using aio client.  # noqa: E501
    # Since I instantiated genai.Client(), it's sync. I should remove 'await'.

    response1 = chat.send_message(prompt1)
    print(f"Agent: {response1.text}")

    # Wait for indexing (critical step for vector DBs)
    print("Waiting 5s for indexing...")
    await asyncio.sleep(5)

    # 5. Turn 2: Ask the LLM to recall (New Session)
    print("\n--- Starting New Chat Session (to verify persistence) ---")
    chat2 = genai_client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False
            ),
            system_instruction="You are a memory tester. Use the 'remember' tool to store facts provided by the user. Use the 'recall' tool to retrieve them when asked.",  # noqa: E501
        ),
    )

    prompt2 = "What is the secret code I told you?"
    print(f"\nUser: {prompt2}")

    response2 = chat2.send_message(prompt2)
    print(f"Agent: {response2.text}")

    # 6. Verification
    if response2.text and secret_code in response2.text:
        print(
            "\n✅ SUCCESS: LLM successfully stored and recalled the secret code via Slate!"  # noqa: E501
        )
    else:
        print("\n❌ FAILURE: LLM failed to recall the correct code.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
