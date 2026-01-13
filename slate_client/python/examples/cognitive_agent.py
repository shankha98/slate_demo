import os
import sys

# Ensure slate_client is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from dotenv import load_dotenv
from google import genai
from google.genai import types

from slate_client import CortexClient

# Load env
load_dotenv()

# --- Slate Client Wrapper ---
# Connects to the Cognitive Slate server
slate = CortexClient(
    address=os.environ.get("SLATE_INSTANCE_URL", "localhost:50051"),
    token=os.environ.get("SLATE_AUTH_TOKEN", "dev_secret"),
    run_id="cognitive-agent-run",
)


def log_internal_action(action_type: str, details: str):
    """Logs internal cognitive actions for transparency."""
    print(f"\n[INTERNAL ACTION] {action_type}: {details}")


# --- Cognitive Actions ---
# These map directly to the framework's internal action space.


def reasoning(context: str) -> str:
    """
    Internal Action: Reasoning.
    Updates working memory by processing current information.
    """
    log_internal_action("Reasoning", f"Processing context: {context[:50]}...")
    try:
        resp = slate.focus(f"[REASONING] {context}")
        return f"Reasoning stored in Working Memory. ID: {resp.id}"
    except Exception as e:
        return f"Error: {e}"


def retrieval(query: str) -> str:
    """
    Internal Action: Retrieval.
    Reads from long-term memory (Episodic/Semantic) into Working Memory.
    """
    log_internal_action("Retrieval", f"Querying long-term memory for: {query}")
    try:
        resp = slate.reminisce(query, limit=3)
        if not resp.traces:
            return "No relevant long-term memories found."

        memories = [f"- {t.input} -> {t.outcome}" for t in resp.traces]
        result = "\n".join(memories)

        # Store retrieved info into Working Memory (standard Cognitive flow)
        slate.focus(f"[RETRIEVED] {result}")
        return f"Retrieved and stored in Working Memory:\n{result}"
    except Exception as e:
        return f"Error: {e}"


def learning(experience: str, outcome: str) -> str:
    """
    Internal Action: Learning.
    Writes new experiences to long-term memory (Episodic).
    """
    log_internal_action("Learning", f"Committing experience: {experience[:50]}...")
    try:
        slate.commit(
            experience, outcome, action="Learning", reasoning="Cognitive Update"
        )
        return "Experience learned (saved to Episodic Memory)."
    except Exception as e:
        return f"Error: {e}"


def grounding_web_search(query: str) -> str:
    """
    External Action: Grounding (Digital).
    Simulates a web search environment interaction.
    """
    print(f"\n[EXTERNAL ACTION] Grounding: Web Search for '{query}'")
    # Simulation of external environment feedback
    return f"Search Results for {query}: [Simulated Result 1], [Simulated Result 2]"


# --- Main Agent Loop (The Decision Procedure) ---


def run_cognitive_agent():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set.")
        return

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    model_id = "gemini-3-flash-preview"

    # Define the Cognitive Agent System Prompt
    system_prompt = (
        "You are a Cognitive Language Agent. Your architecture consists of:\n"
        "1. Working Memory (Context)\n"
        "2. Long-term Memory (Episodic/Semantic)\n"
        "3. Action Space: Internal (Reasoning, Retrieval, Learning) and External (Grounding).\n\n"
        "Your Decision Cycle:\n"
        "1. PLAN: Use Retrieval and Reasoning to understand the situation.\n"
        "2. ACT: Choose an External Action (Grounding) or Internal Action (Learning).\n"
        "3. REFLECT: Update memory with the results.\n\n"
        "Goal: Research 'Cognitive Architectures' and summarize key insights."
    )

    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            tools=[reasoning, retrieval, learning, grounding_web_search],
            system_instruction=system_prompt,
        ),
    )

    print(f"--- Starting Cognitive Agent (Model: {model_id}) ---\n")

    # Step 1: User Input (Environmental Stimulus)
    user_input = "Start research on Cognitive Architectures."
    print(f"User: {user_input}")

    # The model acts as the Decision Procedure
    response = chat.send_message(user_input)
    print(f"\nAgent: {response.text}")

    # Step 2: Follow-up (Simulating the loop)
    # In a real loop, we would automatically feed observations back.
    # Here we prompt for the next step in the cycle.
    next_step_prompt = (
        "Based on your previous actions, what is the next step in your decision cycle?"
    )
    print(f"\n[System Loop]: {next_step_prompt}")

    response = chat.send_message(next_step_prompt)
    print(f"\nAgent: {response.text}")


if __name__ == "__main__":
    run_cognitive_agent()
