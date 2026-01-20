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
    # run_id = f"test-persistence-{uuid.uuid4().hex[:8]}"
    run_id = "test-persistence-ricedb"  # Fixed ID for easier debugging
    print(f"Connecting to Slate at {SLATE_ADDRESS} with run_id={run_id}...")

    if SLATE_ADDRESS:
        os.environ["STATE_INSTANCE_URL"] = SLATE_ADDRESS
    if SLATE_TOKEN:
        os.environ["STATE_AUTH_TOKEN"] = SLATE_TOKEN

    try:
        client = Client(run_id=run_id)
        client.connect()

        # Test 1: EXACTLY like the working script but with Alice content
        print("\n--- Test 1: Mimic Working Script (Long String) ---")
        fact = f"Name: Alice UniqueString-{uuid.uuid4()}"
        print(f"Committing fact: '{fact}'")

        # client.state.commit(
        #     input_text="test input",
        #     output=fact,
        #     action="test_action",
        # )

        print("Waiting 5s...")
        time.sleep(5)

        print("Searching 'Alice'...")
        traces = client.state.reminisce("Name", limit=5)
        if traces:
            print(f"Debug: Found {len(traces)} traces")
            for t in traces:
                print(f" - {t.outcome} (Agent: {t.agent_id})")

        if traces and any("Alice" in t.outcome for t in traces):
            print("✅ SUCCESS: Found 'Alice' (Test 1)")
        else:
            print("❌ FAILURE: Did not find 'Alice' (Test 1)")

        # Test 2: Use the Agent parameters
        print("\n--- Test 2: Agent Parameters ---")
        action = "User provided name"
        outcome = "Name: Bob"
        print(f"Committing: Action='{action}', Outcome='{outcome}'")

        # client.state.commit(
        #     input_text=action, output=outcome, action=action, agent_id="specialist"
        # )

        print("Waiting 5s...")
        time.sleep(5)

        print("Searching 'Bob'...")
        traces = client.state.reminisce("Name", limit=5)
        if traces:
            print(f"Debug: Found {len(traces)} traces")
            for t in traces:
                print(f" - {t.outcome} (Agent: {t.agent_id})")

        if traces and any("Bob" in t.outcome for t in traces):
            print("✅ SUCCESS: Found 'Bob' (Test 2)")
        else:
            print("❌ FAILURE: Did not find 'Bob' (Test 2)")

        print("Searching 'user's name'...")
        traces = client.state.reminisce("user's name", limit=5)
        if traces:
            print(f"Debug: Found {len(traces)} traces")
            for t in traces:
                print(f" - {t.outcome} (Agent: {t.agent_id})")

        if traces and any("Bob" in t.outcome for t in traces):
            print("✅ SUCCESS: Found 'Bob' via semantic query (Test 2)")
        else:
            print("❌ FAILURE: Did not find 'Bob' via semantic query (Test 2)")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
