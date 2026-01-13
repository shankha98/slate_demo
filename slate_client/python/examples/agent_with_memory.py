import os
import sys

# Ensure slate_client is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from dotenv import load_dotenv

load_dotenv()
from google import genai
from google.genai import types

from slate_client import CortexClient

# Initialize Slate Client
# CortexClient takes address and token.
slate = CortexClient(
    address=os.environ.get("SLATE_INSTANCE_URL", "localhost:50051"),
    token=os.environ.get("SLATE_AUTH_TOKEN", "dev_secret"),
    run_id="agent-memory-run",
)


# Define Tools for Gemini
def remember(content: str) -> str:
    """Stores a piece of information in working memory."""
    try:
        resp = slate.focus(content)
        return f"Stored in working memory. ID: {resp.id}"
    except Exception as e:
        return f"Error storing memory: {e}"


def recall_context() -> str:
    """Retrieves current working memory context."""
    try:
        resp = slate.drift()
        if not resp.items:
            return "Working memory is empty."
        items = [
            f"- {item.content} (Relevance: {item.relevance:.2f})" for item in resp.items
        ]
        return "\n".join(items)
    except Exception as e:
        return f"Error retrieving memory: {e}"


def save_experience(input_text: str, action: str, outcome: str) -> str:
    """Saves an interaction experience to long-term episodic memory."""
    try:
        slate.commit(input_text, outcome, action=action)
        return "Experience saved to long-term memory."
    except Exception as e:
        return f"Error saving experience: {e}"


def recall_past(query: str) -> str:
    """Searches long-term episodic memory for similar past experiences."""
    try:
        resp = slate.reminisce(query, limit=3)
        if not resp.traces:
            return "No relevant past experiences found."
        traces = [f"- Input: {t.input} -> Outcome: {t.outcome}" for t in resp.traces]
        return "\n".join(traces)
    except Exception as e:
        return f"Error searching memory: {e}"


def run_agent():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    model_id = "gemini-3-flash-preview"

    print(f"Agent starting with model {model_id}...")

    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            tools=[remember, recall_context, save_experience, recall_past],
            system_instruction="You are an intelligent agent with access to external memory tools (Slate). "
            "Use 'remember' to store temporary context. Use 'recall_context' to see what you are working on. "
            "Use 'save_experience' to log important actions. Use 'recall_past' to learn from history. "
            "Always check your memory before acting.",
        ),
    )

    print("\n--- Interaction 1: Context Setting ---")
    user_msg = "I am planning a surprise party for Alice next Friday."
    print(f"User: {user_msg}")
    response = chat.send_message(user_msg)
    if response.text:
        print(f"Agent: {response.text}")
    else:
        print(f"Agent (No Text): {response.candidates[0].content.parts}")

    # Check if it called tool
    # SDK handles tool calls automatically by default.

    print("\n--- Interaction 2: Recall ---")
    user_msg = "Who is the party for and when?"
    print(f"User: {user_msg}")
    response = chat.send_message(user_msg)
    print(f"Agent: {response.text}")


if __name__ == "__main__":
    run_agent()
