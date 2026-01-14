import asyncio
import json
import sys

import websockets


async def run_conversation():
    uri = "ws://localhost:8000/ws"

    try:
        with open("sample_conversation.json", "r") as f:
            messages = json.load(f)
    except FileNotFoundError:
        print("Error: sample_conversation.json not found.")
        sys.exit(1)

    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected.")

            for i, user_msg in enumerate(messages, 1):
                print(f"\n--- Turn {i} ---")
                print(f"User: {user_msg}")

                # Send message
                await websocket.send(
                    json.dumps({"type": "message", "content": user_msg})
                )

                # Wait for response
                while True:
                    try:
                        response_str = await websocket.recv()
                        print(f"DEBUG RX: {response_str}")
                        response = json.loads(response_str)

                        if response.get("type") == "final_response":
                            print(f"Bot: {response.get('content')}")
                            break
                        elif response.get("agent"):
                            # It's a log message
                            # print(f"  [{response['agent']}] {response['type']}: {response['content'][:50]}...")
                            pass  # suppress logs for cleaner output

                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed unexpectedly.")
                        sys.exit(1)

            print("\nConversation completed successfully.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_conversation())
