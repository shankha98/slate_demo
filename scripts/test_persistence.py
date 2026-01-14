import asyncio
import json
import sys
import uuid

import websockets
from dotenv import load_dotenv

load_dotenv()


async def run_persistence_test():
    uri = "ws://localhost:8000/ws"
    run_id = f"test-persistence-{uuid.uuid4().hex[:8]}"
    uri_with_id = f"{uri}?run_id={run_id}"

    print(f"--- Session 1 (ID: {run_id}) ---")
    async with websockets.connect(uri_with_id) as websocket:
        # Check session info
        msg = json.loads(await websocket.recv())
        print(f"Received: {msg}")

        # Send fact
        print("User: My name is Alice.")
        await websocket.send(
            json.dumps({"type": "message", "content": "My name is Alice."})
        )

        # Wait for response
        while True:
            resp = json.loads(await websocket.recv())
            if resp.get("type") == "final_response":
                print(f"Bot: {resp['content']}")
                break
            elif resp.get("agent"):
                print(f"  [{resp['agent']}] {resp['type']}: {resp['content'][:50]}...")

    print("\n--- Disconnected ---")
    # print("--- Waiting 10s for indexing ---")
    # await asyncio.sleep(10)
    print("--- Reconnecting to Session 2 ---")

    async with websockets.connect(uri_with_id) as websocket:
        # Check session info
        msg = json.loads(await websocket.recv())
        print(f"Received: {msg}")

        # Ask question
        print("User: What is my name?")
        await websocket.send(
            json.dumps({"type": "message", "content": "What is my name?"})
        )

        # Wait for response
        while True:
            resp = json.loads(await websocket.recv())
            if resp.get("type") == "final_response":
                print(f"Bot: {resp['content']}")
                content = resp["content"]
                if "Alice" in content:
                    print("SUCCESS: Persistence verified!")
                else:
                    print("FAILURE: Did not recall name.")
                    sys.exit(1)
                break
            elif resp.get("agent"):
                print(f"  [{resp['agent']}] {resp['type']}: {resp['content'][:50]}...")


if __name__ == "__main__":
    asyncio.run(run_persistence_test())
