import os
import sys
import time

from dotenv import load_dotenv

from rice_sdk import Client

# Load env vars
load_dotenv()

SLATE_ADDRESS = os.getenv("SLATE_ADDRESS", "localhost:50051")
SLATE_TOKEN = os.getenv("SLATE_TOKEN", "")


def main():
    print(f"Connecting to Slate at {SLATE_ADDRESS}...")
    run_id = "test-ricedb-real-v1"

    if SLATE_ADDRESS:
        os.environ["STATE_INSTANCE_URL"] = SLATE_ADDRESS
    if SLATE_TOKEN:
        os.environ["STATE_AUTH_TOKEN"] = SLATE_TOKEN

    try:
        client = Client(run_id=run_id)
        client.connect()

        # 1. Commit a unique fact
        fact = f"RiceDB Test Timestamp {time.time()}"
        print(f"Committing fact: '{fact}'")
        client.state.commit(
            input_text="test input",
            output=fact,
            action="test_action",
        )

        print("Commit successful (no exception raised).")

        # 2. Wait a moment for indexing (if needed, though commit might be sync)
        time.sleep(1)

        # 3. Reminisce (Search)
        print("Searching for fact...")
        traces = client.state.reminisce("RiceDB Test", limit=5)

        found = False
        if traces:
            for trace in traces:
                print(f"Found trace: {trace.outcome}")
                if fact in trace.outcome:
                    found = True

        if found:
            print("SUCCESS: Fact found in RiceDB!")
        else:
            print("FAILURE: Fact NOT found in RiceDB.")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
