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
    # run_id = f"test-persistence-{uuid.uuid4().hex[:8]}"
    run_id = "test-persistence-ricedb"  # Fixed ID for easier debugging
    print(f"Connecting to Slate at {SLATE_ADDRESS} with run_id={run_id}...")

    try:
        client = CortexClient(address=SLATE_ADDRESS, token=SLATE_TOKEN, run_id=run_id)

        # Test 1: EXACTLY like the working script but with Alice content
        print("\n--- Test 1: Mimic Working Script (Long String) ---")
        fact = f"Name: Alice UniqueString-{uuid.uuid4()}"
        print(f"Committing fact: '{fact}'")

        # client.commit(
        #     input="test input",
        #     outcome=fact,
        #     action="test_action",
        #     reasoning="testing ricedb",
        # )

        print("Waiting 5s...")
        time.sleep(5)

        print("Searching 'Alice'...")
        resp = client.reminisce("Name", limit=5)
        if hasattr(resp, "traces"):
            print(f"Debug: Found {len(resp.traces)} traces")
            for t in resp.traces:
                print(f" - {t.outcome} (Agent: {t.agent_id})")

        if hasattr(resp, "traces") and any("Alice" in t.outcome for t in resp.traces):
            print("✅ SUCCESS: Found 'Alice' (Test 1)")
        else:
            print("❌ FAILURE: Did not find 'Alice' (Test 1)")

        # Test 2: Use the Agent parameters
        print("\n--- Test 2: Agent Parameters ---")
        action = "User provided name"
        outcome = "Name: Bob"
        print(f"Committing: Action='{action}', Outcome='{outcome}'")

        # client.commit(
        #     input=action, outcome=outcome, action=action, agent_id="specialist"
        # )

        print("Waiting 5s...")
        time.sleep(5)

        print("Searching 'Bob'...")
        resp = client.reminisce("Name", limit=5)
        if hasattr(resp, "traces"):
            print(f"Debug: Found {len(resp.traces)} traces")
            for t in resp.traces:
                print(f" - {t.outcome} (Agent: {t.agent_id})")

        if hasattr(resp, "traces") and any("Bob" in t.outcome for t in resp.traces):
            print("✅ SUCCESS: Found 'Bob' (Test 2)")
        else:
            print("❌ FAILURE: Did not find 'Bob' (Test 2)")

        print("Searching 'user's name'...")
        resp = client.reminisce("user's name", limit=5)
        if hasattr(resp, "traces"):
            print(f"Debug: Found {len(resp.traces)} traces")
            for t in resp.traces:
                print(f" - {t.outcome} (Agent: {t.agent_id})")

        if hasattr(resp, "traces") and any("Bob" in t.outcome for t in resp.traces):
            print("✅ SUCCESS: Found 'Bob' via semantic query (Test 2)")
        else:
            print("❌ FAILURE: Did not find 'Bob' via semantic query (Test 2)")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
