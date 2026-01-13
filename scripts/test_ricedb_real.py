import os
import sys
import time

from dotenv import load_dotenv

from slate_client import CortexClient

# Load env vars
load_dotenv()

SLATE_ADDRESS = os.getenv("SLATE_ADDRESS", "localhost:50051")
SLATE_TOKEN = os.getenv("SLATE_TOKEN", "")


def main():
    print(f"Connecting to Slate at {SLATE_ADDRESS}...")
    run_id = "test-ricedb-real-v1"

    try:
        client = CortexClient(address=SLATE_ADDRESS, token=SLATE_TOKEN, run_id=run_id)

        # 1. Commit a unique fact
        fact = f"RiceDB Test Timestamp {time.time()}"
        print(f"Committing fact: '{fact}'")
        client.commit(
            input="test input",
            outcome=fact,
            action="test_action",
            reasoning="testing ricedb",
        )

        print("Commit successful (no exception raised).")

        # 2. Wait a moment for indexing (if needed, though commit might be sync)
        time.sleep(1)

        # 3. Reminisce (Search)
        print("Searching for fact...")
        resp = client.reminisce("RiceDB Test", limit=5)

        found = False
        if hasattr(resp, "traces"):
            for trace in resp.traces:
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
