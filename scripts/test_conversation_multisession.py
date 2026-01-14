import asyncio
import json
import sys
import uuid

import websockets


async def run_conversation_multisession():
    uri = "ws://localhost:8000/ws"

    # Generate a consistent run_id to simulate the same Slate memory context across different chat sessions
    run_id = f"test-multisession-{uuid.uuid4().hex[:8]}"
    uri_with_run_id = f"{uri}?run_id={run_id}"

    print("Starting Multi-Session Persistence Test")
    print(f"Shared Run ID: {run_id}")
    print(
        "Each turn will be a NEW WebSocket connection (simulating a new chat session)."
    )

    try:
        with open("sample_conversation.json", "r") as f:
            messages = json.load(f)
    except FileNotFoundError:
        print("Error: sample_conversation.json not found.")
        sys.exit(1)

    try:
        for i, user_msg in enumerate(messages, 1):
            print(f"\n--- Turn {i} (New Connection) ---")
            print(f"User: {user_msg}")

            # Connect for THIS turn only
            async with websockets.connect(uri_with_run_id) as websocket:
                # Wait for session_info
                session_msg = await websocket.recv()
                # print(f"DEBUG: {session_msg}")

                # Send message
                await websocket.send(
                    json.dumps({"type": "message", "content": user_msg})
                )

                # Wait for response
                while True:
                    try:
                        response_str = await websocket.recv()
                        # print(f"DEBUG RX: {response_str}")
                        response = json.loads(response_str)

                        if response.get("type") == "final_response":
                            print(f"Bot: {response.get('content')}")
                            break
                        elif response.get("agent"):
                            # print(f"  [{response['agent']}] {response['type']}: {response['content'][:50]}...")
                            pass

                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed unexpectedly.")
                        sys.exit(1)

            # Simulate a small pause between sessions to ensure we aren't hitting race conditions in the test script itself,
            # and to allow Slate backend (RiceDB) to index if necessary (though our code handles that).
            # The agent itself handles indexing wait, but a small buffer here mimics real user behavior.
            await asyncio.sleep(1)

        print("\nConversation completed successfully across multiple sessions.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_conversation_multisession())
