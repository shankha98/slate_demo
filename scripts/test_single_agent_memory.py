import os
import sys
import time
import uuid

from dotenv import load_dotenv

from slate_client import CortexClient

# Load env vars
load_dotenv()

SLATE_ADDRESS = os.getenv("SLATE_ADDRESS", "localhost:50051")
SLATE_TOKEN = os.getenv("SLATE_TOKEN", "")


def main():
    print(f"Connecting to Slate at {SLATE_ADDRESS}...")
    run_id = f"test-single-agent-{uuid.uuid4().hex[:8]}"

    try:
        client = CortexClient(address=SLATE_ADDRESS, token=SLATE_TOKEN, run_id=run_id)

        print("\n--- Test Phase 1: Store Information ---")
        fact = "The project code name is Project Chimera."
        print(f"Committing fact: '{fact}'")

        # Simulating an agent storing a fact
        client.commit(
            input="What is the project code name?",
            outcome=fact,
            action="store_information",
            reasoning="Storing important project details",
            agent_id="single-agent",
        )

        print("Commit successful. Waiting for indexing (5s)...")
        time.sleep(5)

        print("\n--- Test Phase 2: Recall Information ---")
        query = "project code name"
        print(f"Searching for: '{query}'")

        resp = client.reminisce(query, limit=5)

        found = False
        if hasattr(resp, "traces"):
            print(f"Found {len(resp.traces)} traces:")
            for trace in resp.traces:
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
