# MetaOS

A Python framework for building, running, and automatically optimizing multi-step AI agent workflows.

The fastest way to use it is to run `python cli.py` and chat with the **MetaOS Agent** — describe a goal and it plans, writes, tests, and optimizes the full pipeline for you. You can also use every component individually as a Python library.

---

## What's included

| Component | What it does |
|-----------|-------------|
| `LLM_API` | LiteLLM-backed LLM wrapper with retries, history, context compaction, tool calling, and metrics |
| Operators | `Predict`, `CoT`, `React`, `Router` — all support self-consistency (`n` parallel calls + majority vote) |
| `Workflow` | JSON-defined DAG executor with parallel steps, conditional branching, prompt files, run logging, and checkpointing |
| `Dataset` | Labeled example store with train/test split and evaluator factory (exact match or LLM judge) |
| `PromptOptimizer` | Backward-pass loop that finds failing steps and rewrites their prompts |
| `WorkflowOptimizer` | Structural optimizer that adds, removes, or rewires DAG steps |
| `Planner` | Recursively decomposes a goal into a `TodoTree` using structured LLM output |
| `TodoTree` | Hierarchical goal decomposition tree — records what was tried, what failed, and what worked |
| `ToolSynthesizer` | LLM-generates a Python function from a description, sandbox-tests it, iterates on failures |
| `ToolRegistry` | Persistent tool store — save, load, search, and compile synthesized tools |
| `ToolOptimizer` | Iteratively improves an existing tool against test cases (accept/revert loop) |
| `DataSynthesizer` | Generates synthetic labeled (inputs, expected) pairs for optimization; optionally grounded in web search |
| `WorkflowRegistry` | Persistent workflow store — save, load, and search workflow configs by name/tag |
| **`MetaOSAgent`** | **Top-level conversational orchestrator — chat to design, build, and optimize full AI pipelines** |
| **`cli.py`** | **Textual terminal UI for MetaOSAgent — multi-panel chat, live sidebar, slash commands, session save/load** |

---

## Installation

```bash
git clone https://github.com/anishsan/MetaOS.git
cd MetaOS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your API key(s):

```
GROQ_API_KEY=your_key_here
# OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...
```

Model names use a `provider/model` prefix — LiteLLM routes to the right API automatically:

```
"groq/llama-3.3-70b-versatile"
"openai/gpt-4o"
"anthropic/claude-sonnet-4-6"
"ollama/llama3"
```

---

## Quick start — run a workflow

**1. Define `workflow.json`:**

```json
{
  "name": "support",
  "operators": {
    "classifier": {
      "type": "router",
      "model": "groq/llama-3.3-70b-versatile",
      "input_cpm": 0.59,
      "output_cpm": 0.79
    },
    "writer": {
      "type": "cot",
      "model": "groq/llama-3.3-70b-versatile",
      "input_cpm": 0.59,
      "output_cpm": 0.79
    }
  },
  "steps": [
    {
      "id": "classify",
      "operator": "classifier",
      "prompt": "Classify this message: {message}",
      "args": {"routes": ["billing", "technical", "general"]},
      "output": "department"
    },
    {
      "id": "respond",
      "operator": "writer",
      "prompt": "Write a {department} support reply to: {message}",
      "depends_on": ["classify"],
      "output": "reply"
    }
  ]
}
```

**2. Run it:**

```python
from src import Workflow

workflow = Workflow.from_json("workflow.json")
result = workflow.run({"message": "I was charged twice"})

print(result["department"])           # "billing"
print(result["reply"]["answer"])      # the written reply
print(workflow.metrics["total_cost"]) # total cost in $
```

---

## MetaOS Agent — chat to build AI systems

`MetaOSAgent` is the top-level interface. Give it a goal in plain English; it plans, builds, tests, and optimizes the full pipeline without writing config by hand.

### Terminal UI

```bash
python cli.py
```

Opens a multi-panel TUI: chat on the left, live tree / tool / workflow sidebar on the right, inline cost tracker, and tool-call progress as the agent works.

#### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `groq/llama-3.3-70b-versatile` | LiteLLM model string |
| `--input-cpm` | `0.59` | Cost per million input tokens (USD) |
| `--output-cpm` | `0.79` | Cost per million output tokens (USD) |
| `--session` | — | Path to a saved session JSON to load on startup |
| `--tools-dir` | `tools/` | Directory for the tool registry |
| `--workflows-dir` | `workflows/` | Directory for the workflow registry |

#### Slash commands

| Command | Action |
|---------|--------|
| `/help` | Show all commands |
| `/tree` | Print the current planning tree |
| `/tools` | List every tool in the registry |
| `/workflows` | List every workflow in the registry |
| `/cost` | Show session cost, API call count, tool call count |
| `/save [name]` | Save conversation to `sessions/{name}.json` |
| `/load [name]` | Load a saved session |
| `/reset` | Start a fresh session (keeps tool/workflow registries) |

Keyboard shortcuts: `Ctrl+S` save · `Ctrl+R` reset · `Ctrl+C` quit.

### Python API

```python
from src import MetaOSAgent

agent = MetaOSAgent("groq/llama-3.3-70b-versatile", 0.59, 0.79)

print(agent.chat("Build a customer support pipeline"))
print(agent.chat("The plan looks good, go ahead and build it"))
print(agent.chat("Generate 20 test examples and optimize"))
```

**What the agent does automatically:**

- Searches registries before creating anything new
- Decomposes goals into a `TodoTree` and presents it for approval
- Builds tools bottom-up, sandbox-tests each one
- Generates workflow JSON configs with operator docs embedded in context
- Generates synthetic test data (or uses web search as grounding)
- Runs `PromptOptimizer` / `WorkflowOptimizer` / `ToolOptimizer` per node

---

## Tool Forge — agentic tool creation

```python
from src import ToolRegistry, ToolSynthesizer

registry = ToolRegistry("tools/")
synth = ToolSynthesizer("groq/llama-3.3-70b-versatile", 0.59, 0.79, registry)

record = synth.synthesize(
    description="Calculate compound interest given principal, annual rate, and years",
    signature="(principal: float, rate: float, years: int) -> float",
    test_cases=[
        {"args": {"principal": 1000, "rate": 0.05, "years": 1}, "expected": 1050.0},
    ],
    tags=["finance"],
)

fn = registry.load_as_callable(record.name)
print(fn(1000, 0.05, 3))   # 1157.625
```

---

## Multi-provider support

MetaOS uses [LiteLLM](https://github.com/BerriAI/litellm) so any provider works without code changes:

```
"groq/llama-3.3-70b-versatile"   → Groq
"openai/gpt-4o"                  → OpenAI
"anthropic/claude-sonnet-4-6"    → Anthropic
"ollama/llama3"                  → local Ollama
```
