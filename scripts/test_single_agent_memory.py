import os
import sys
import time
import uuid

from dotenv import load_dotenv

from rice_sdk import Client

# Load env vars
load_dotenv()

SLATE_ADDRESS = os.getenv("SLATE_ADDRESS", "localhost:50051")
SLATE_TOKEN = os.getenv("SLATE_TOKEN", "")


def main():
    print(f"Connecting to Slate at {SLATE_ADDRESS}...")
    run_id = f"test-single-agent-{uuid.uuid4().hex[:8]}"

    if SLATE_ADDRESS:
        os.environ["STATE_INSTANCE_URL"] = SLATE_ADDRESS
    if SLATE_TOKEN:
        os.environ["STATE_AUTH_TOKEN"] = SLATE_TOKEN

    try:
        client = Client(run_id=run_id)
        client.connect()

        print("\n--- Test Phase 1: Store Information ---")
        fact = "The project code name is Project Chimera."
        print(f"Committing fact: '{fact}'")

        # Simulating an agent storing a fact
        client.state.commit(
            input_text="What is the project code name?",
            output=fact,
            action="store_information",
            agent_id="single-agent",
        )

        print("Commit successful. Waiting for indexing (5s)...")
        time.sleep(5)

        print("\n--- Test Phase 2: Recall Information ---")
        query = "project code name"
        print(f"Searching for: '{query}'")

        traces = client.state.reminisce(query, limit=5)

        found = False
        if traces:
            print(f"Found {len(traces)} traces:")
            for trace in traces:
                print(f"  - Action: {trace.action} | Outcome: {trace.outcome}")
                if "Project Chimera" in trace.outcome:
                    found = True

        if found:
            print("\n✅ SUCCESS: Memory working correctly in single agent mode!")
        else:
            print("\n❌ FAILURE: Could not recall the stored fact.")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
