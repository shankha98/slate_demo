import os
import sys

# Ensure slate_client is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from dotenv import load_dotenv

load_dotenv()
from google import genai
from google.genai import types

from slate_client import CortexClient

# Shared Memory
slate_url = os.environ.get("SLATE_INSTANCE_URL", "localhost:50051")
slate_token = os.environ.get("SLATE_AUTH_TOKEN", "dev_secret")
print(f"Connecting to Slate at {slate_url} with token: {slate_token[:5]}...")

slate = CortexClient(address=slate_url, token=slate_token, run_id="research-team-run")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable not set.")
    exit(1)

client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
model_id = "gemini-3-flash-preview"


# --- Researcher Agent ---
def search(query: str) -> str:
    """Simulates a web search engine. Use this to find information."""
    print(f"[Tool] Searching for: {query}")
    # Mock knowledge for the example topic
    if "rust" in query.lower():
        return (
            "Rust was born at Mozilla Research. Graydon Hoare started it in 2006. "
            "Version 1.0 was released in 2015. "
            "It introduced the concept of ownership and borrowing for memory safety without garbage collection. "
            "The Rust Foundation was formed in 2021."
        )
    return "No results found."


def store_finding(fact: str) -> str:
    """Stores a research finding in shared working memory."""
    print(f"[Tool] Storing finding: {fact}")
    try:
        slate.focus(f"[FINDING] {fact}")
        # Also commit to long-term for persistence
        slate.commit(
            fact,
            "Research Finding",
            action="Research",
            reasoning="Important fact found",
        )
        return "Finding stored."
    except Exception as e:
        print(f"[Tool Error] store_finding failed: {e}")
        return f"Error: {e}"


def researcher_task(topic):
    print(f"\n--- Researcher working on: {topic} ---")
    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            tools=[store_finding, search],
            system_instruction="You are a Researcher. Your goal is to find interesting facts about the topic "
            "and store them using 'store_finding'. Use the 'search' tool to find information. "
            "You do not write the final report. "
            "Just find 2-3 distinct facts and store them.",
        ),
    )
    resp = chat.send_message(f"Research this topic: {topic}")
    print(f"Researcher: {resp.text}")


# --- Writer Agent ---
def read_memory() -> str:
    """Reads all findings from working memory."""
    print("[Tool] Reading memory...")
    try:
        resp = slate.drift()
        findings = [item.content for item in resp.items if "[FINDING]" in item.content]
        if not findings:
            print("[Tool] No findings found in memory.")
            return "No findings yet."
        return "\n".join(findings)
    except Exception as e:
        print(f"[Tool Error] read_memory failed: {e}")
        return f"Error: {e}"


def writer_task(topic):
    print(f"\n--- Writer working on: {topic} ---")
    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            tools=[read_memory],
            system_instruction="You are a Writer. Your goal is to write a short summary based ONLY on findings "
            "available in memory. Use 'read_memory' to get facts. "
            "Do not hallucinate facts not in memory.",
        ),
    )
    resp = chat.send_message(f"Write a report on: {topic}")
    print(f"Writer: {resp.text}")


def main():
    topic = "The history of Rust programming language"

    # Step 1: Researcher populates memory
    researcher_task(topic)

    # Step 2: Writer consumes memory
    writer_task(topic)


if __name__ == "__main__":
    main()
