import os
import sys
import time

# Ensure we can import the local package if running from here
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from slate_client import CortexClient

def main():
    address = os.environ.get("SLATE_INSTANCE_URL", "localhost:50051")
    token = os.environ.get("SLATE_AUTH_TOKEN", "dev_secret")
    print(f"Connecting to {address} with token: {token}")
    
    client = CortexClient(address=address, token=token, run_id="python-client-demo")
    
    try:
        response = client.focus("Hello Secure World")
        print(f"Focused: {response.id}")

        print("Committing memory...")
        client.commit("User says hello", "Agent replied", reasoning="Greeting")
        
        time.sleep(2)

        print("Reminiscing...")
        memories = client.reminisce("hello", 1).traces
        for m in memories:
             print(f"Memory: {m.input} -> {m.outcome}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
