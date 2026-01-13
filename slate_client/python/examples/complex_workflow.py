import logging
import os
import sys
import time

# Ensure slate_client is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from dotenv import load_dotenv
from google import genai
from google.genai import types

from slate_client import CortexClient

# Load env
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SlateAgent")


# --- Slate Client & Tool Wrapper ---
class SlateTools:
    def __init__(self):
        token = os.environ.get("SLATE_AUTH_TOKEN", "dev_secret")
        address = os.environ.get("SLATE_INSTANCE_URL", "localhost:50051")
        self.client = CortexClient(
            address=address, token=token, run_id="complex-workflow-run"
        )
        logger.info(f"Connected to Slate Cortex at {address}")

    def log_tool_use(self, tool_name: str, **kwargs):
        """Logs the tool usage to console and optionally to Slate itself as a meta-trace."""
        logger.info(f"🛠️ TOOL CALLED: {tool_name} | Args: {kwargs}")

    # --- Working Memory (Flux) ---
    def store_context(self, content: str) -> str:
        """Stores temporary context or scratchpad notes in working memory."""
        self.log_tool_use("store_context", content=content)
        try:
            resp = self.client.focus(content)
            return f"Stored context ID: {resp.id}"
        except Exception as e:
            logger.error(f"Failed to store context: {e}")
            return f"Error: {e}"

    def retrieve_context(self) -> str:
        """Retrieves currently active context from working memory, sorted by relevance."""
        self.log_tool_use("retrieve_context")
        try:
            resp = self.client.drift()
            if not resp.items:
                return "Working memory is empty."

            # Format nicely
            result = ["Current Working Memory (Most Relevant First):"]
            for item in resp.items:
                result.append(f"- {item.content} (Score: {item.relevance:.2f})")
            return "\n".join(result)
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return f"Error: {e}"

    # --- Episodic Memory (Echoes) ---
    def record_experience(self, action: str, outcome: str, reasoning: str = "") -> str:
        """Records an action and its outcome into long-term episodic memory."""
        self.log_tool_use(
            "record_experience", action=action, outcome=outcome, reasoning=reasoning
        )
        try:
            # We use action as 'input' conceptually here for the trace
            self.client.commit(
                input=action, outcome=outcome, action=action, reasoning=reasoning
            )
            return "Experience committed to long-term memory."
        except Exception as e:
            logger.error(f"Failed to record experience: {e}")
            return f"Error: {e}"

    def recall_experiences(self, query: str) -> str:
        """Searches long-term memory for similar past experiences."""
        self.log_tool_use("recall_experiences", query=query)
        try:
            resp = self.client.reminisce(query, limit=3)
            if not resp.traces:
                return "No relevant past experiences found."

            result = [f"Recall results for '{query}':"]
            for t in resp.traces:
                result.append(
                    f"- Action: {t.action} | Outcome: {t.outcome} | Reasoning: {t.reasoning}"
                )
            return "\n".join(result)
        except Exception as e:
            logger.error(f"Failed to recall experiences: {e}")
            return f"Error: {e}"

    # --- Procedural Memory (Reflex) ---
    def execute_skill(self, skill_name: str) -> str:
        """Executes a compiled skill (WASM) stored in procedural memory."""
        self.log_tool_use("execute_skill", skill_name=skill_name)
        try:
            resp = self.client.trigger(skill_name)
            return f"Skill '{skill_name}' executed. Result: {resp.result}"
        except Exception as e:
            return f"Error executing skill: {e}"


slate_tools = SlateTools()

# --- Agent Defintions ---


def create_agent(client, model_id, system_prompt, tools):
    return client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            tools=tools, system_instruction=system_prompt
        ),
    )


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    model_id = "gemini-3-flash-preview"

    # --- Scenario: Software Development Task Force ---
    # Agent 1: Architect - Plans the system, stores decisions in Flux.
    # Agent 2: Developer - Writes code based on plan, checks for past similar bugs (Echoes).
    # Agent 3: QA - Tests the code, records pass/fail (Echoes).

    logger.info("--- Starting Complex Workflow: Software Dev Team ---")

    # 1. Architect Planning
    architect_prompt = (
        "You are a Software Architect. Your job is to design a scalable API for a 'Todo App'. "
        "1. Outline the high-level architecture. "
        "2. Store the key design decisions in working memory using 'store_context' so the developer can see them. "
        "3. Check if we have built similar apps before using 'recall_experiences' to avoid pitfalls."
    )

    architect = create_agent(
        client,
        model_id,
        architect_prompt,
        [slate_tools.store_context, slate_tools.recall_experiences],
    )

    logger.info(">>> Architect is thinking...")
    response = architect.send_message("Please design the Todo App backend.")
    print(f"\n[Architect]: {response.text}\n")

    # 2. Developer Implementation
    developer_prompt = (
        "You are a Senior Developer. "
        "1. Retrieve the architecture plan from working memory ('retrieve_context'). "
        "2. Based on the plan, propose the database schema (SQL). "
        "3. Commit your implementation details to long-term memory ('record_experience') so QA knows what to test."
    )

    # Pause to allow previous context storage to propagate and avoid API rate limits/race conditions
    time.sleep(2)

    developer = create_agent(
        client,
        model_id,
        developer_prompt,
        [slate_tools.retrieve_context, slate_tools.record_experience],
    )

    logger.info(">>> Developer is coding...")
    response = developer.send_message(
        "Implement the database schema based on the architect's plan."
    )
    print(f"\n[Developer]: {response.text}\n")

    # 3. QA Testing
    qa_prompt = (
        "You are a QA Engineer. "
        "1. Retrieve the implementation details from long-term memory using 'recall_experiences' (search for 'schema' or 'database'). "
        "2. Create a test plan. "
        "3. Store the test plan in working memory ('store_context')."
    )

    qa = create_agent(
        client,
        model_id,
        qa_prompt,
        [slate_tools.recall_experiences, slate_tools.store_context],
    )

    logger.info(">>> QA is testing...")
    response = qa.send_message("Create a test plan for the database.")
    print(f"\n[QA]: {response.text}\n")


if __name__ == "__main__":
    main()
