# Slate Agent Examples (Python)

This directory contains examples of using Slate with Google GenAI SDK to build agentic systems.

## Setup

1.  Ensure you have `python` and `pip` installed.
2.  Install dependencies (from `clients/python` root):

    ```bash
    pip install -r requirements.txt
    ```

    This installs `slate-client` dependencies, `google-genai`, and `python-dotenv`.

    _Note: These examples import `slate-client` directly from `../src/slate_client` using a sys.path hack for development convenience. In a production environment, you would install the package via `pip install ..`._

3.  Set environment variables:
    Create a `.env` file in `clients/python` (this is already ignored in git) or export them:
    ```bash
    export GEMINI_API_KEY="your-gemini-key"
    # Ensure Slate server is running separately (e.g. `make run-ricedb`)
    export SLATE_AUTH_TOKEN="dev_secret"
    ```

## Examples

Run these examples from `clients/python` directory (or inside `examples` but ensure `.env` is loaded).

### 1. Agent with Memory (`agent_with_memory.py`)

A single agent that can remember context, commit experiences, and recall past interactions using Slate tools.

```bash
python examples/agent_with_memory.py
```

### 2. Research Team (`research_team.py`)

A multi-agent system where a "Researcher" agent gathers facts and stores them in Slate, and a "Writer" agent retrieves them to write a report.

```bash
python examples/research_team.py
```

### 3. Complex Workflow (`complex_workflow.py`)

A long-running workflow with Architect, Developer, and QA agents.

```bash
python examples/complex_workflow.py
```

### 4. Cognitive Agent (`cognitive_agent.py`)

Implementation of a Cognitive Architecture loop (Plan/Act/Reflect).

```bash
python examples/cognitive_agent.py
```
