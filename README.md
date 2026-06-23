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
git clone https://github.com/your-username/MetaOS.git
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

## Operators

| Type | Description |
|------|-------------|
| `predict` | Single LLM call. With `n > 1`: majority vote across N parallel calls. |
| `cot` | Two-call chain: reason first, then answer. With `n > 1`: self-consistent CoT. |
| `react` | Multi-turn tool-use loop (ReAct pattern). Calls tools until it has enough info. |
| `router` | Classifies input into one of N categories using JSON schema output. |

All operators accept `n` (self-consistency) and `system_prompt` (override default):

```json
{
  "type": "cot",
  "model": "groq/llama-3.3-70b-versatile",
  "input_cpm": 0.59,
  "output_cpm": 0.79,
  "n": 5,
  "system_prompt": "You are a medical triage assistant."
}
```

Use operators directly in Python too:

```python
from src import CoT, React, Router

cot = CoT("analyser", "groq/llama-3.3-70b-versatile", 0.59, 0.79, n=3)
reasoning, answer = cot("Is this argument logically valid? Argument: ...")

router = Router("classifier", "groq/llama-3.3-70b-versatile", 0.59, 0.79)
route = router("Classify this message", routes=["billing", "technical", "general"])
```

---

## Workflow features

### Prompt files

Keep long prompts in `.md` files instead of inline JSON:

```json
{ "id": "classify", "operator": "classifier", "prompt_file": "prompts/classify.md", "output": "department" }
```

`{key}` template variables work identically in `.md` files.

### Parallel execution

Steps with no dependency on each other run simultaneously — detected automatically from `depends_on`.

### Conditional branching

```json
{
  "id": "handle_billing",
  "depends_on": ["classify"],
  "condition": "{department} == 'billing'",
  "output": "resolution"
}
```

### Run logging

```python
result = workflow.run(inputs, log_dir="runs/")
# writes runs/20260623_143022/inputs.json, state.json, metrics.json
```

### Checkpointing

```python
workflow.save_checkpoint("checkpoints/v1.json", description="baseline")
workflow = Workflow.from_checkpoint("checkpoints/v1.json", tools={...})
```

### Tools (React)

```python
def lookup_invoice(invoice_id: str) -> str:
    """Look up invoice details by ID."""
    return f"Invoice {invoice_id}: $49.99, paid 2026-05-01"

workflow = Workflow.from_json("workflow.json", tools={"lookup_invoice": lookup_invoice})
```

---

## Optimization

### 1. Label your examples

```python
from src import Dataset

ds = Dataset.from_json("data/examples.json")
# or: ds = Dataset.from_list([{"inputs": {...}, "expected": "..."}, ...])
train, test = ds.split(test_size=0.2)
```

### 2. Create an evaluator

```python
# Exact match — good for classification
evaluator = train.make_evaluator(output_key="department", mode="exact_match")

# LLM judge — good for open-ended outputs
evaluator = train.make_evaluator(
    output_key="final_reply",
    mode="llm_judge",
    judge_model="groq/llama-3.3-70b-versatile",
    input_cpm=0.59,
    output_cpm=0.79,
)
```

### 3. Optimize prompts

```python
from src import PromptOptimizer

optimizer = PromptOptimizer(
    workflow=workflow,
    evaluator=evaluator,
    optimizer_model="groq/llama-3.3-70b-versatile",
    input_cpm=0.59,
    output_cpm=0.79,
)

workflow = optimizer.optimize(
    examples=train.inputs,
    checkpoint_dir="checkpoints/",
    max_rounds=3,
    experiment_dir="experiments/",  # writes a .md log per run
)
```

### 4. Optimize structure (optional)

```python
from src import WorkflowOptimizer

wopt = WorkflowOptimizer(workflow, evaluator, "groq/llama-3.3-70b-versatile", 0.59, 0.79)
workflow = wopt.optimize(train.inputs, "checkpoints/", max_rounds=3, experiment_dir="experiments/")
```

### 5. Evaluate on test set

```python
results = test.evaluate(workflow, output_key="department", evaluator=evaluator)
print(results)
# {"avg_score": 0.92, "n_examples": 10, "n_passing": 9, "scores": [...]}
```

---

## File layout

```
cli.py             — Textual TUI entry point (python cli.py)

src/
  __init__.py      — public API
  llm_api.py       — LiteLLM wrapper with history, retries, metrics
  agent.py         — Predict, CoT, React, Router operators
  workflow.py      — DAG workflow engine
  optimizer.py     — PromptOptimizer, WorkflowOptimizer, Dataset
  planner.py       — TodoNode, TodoTree, Planner
  tool_forge.py    — ToolRecord, ToolRegistry, ToolSynthesizer, ToolOptimizer
  data_synth.py    — DataSynthesizer (synthetic data generation + web grounding)
  meta_agent.py    — MetaOSAgent, WorkflowRegistry, WorkflowBuilder

docs/
  workflow.md      — full workflow reference
  optimization.md  — optimization, evaluation, and planning reference

sessions/          — saved TUI sessions (ui_messages + llm_history)
plan/              — TodoTree JSON and linked workflow files
tools/             — synthesized tool store (one .json per tool + _index.json)
workflows/         — workflow registry (one .json per workflow + _index.json)
tests/tools/       — auto-generated pytest files for synthesized tools
experiments/       — auto-generated experiment logs (one .md per run)
checkpoints/       — auto-generated workflow version snapshots
runs/              — auto-generated per-run state and metrics logs
```

---

## Multi-provider support

MetaOS uses [LiteLLM](https://github.com/BerriAI/litellm) so any provider works without code changes. Set the relevant API key in `.env` and use the provider prefix in your model string:

```
"groq/llama-3.3-70b-versatile"   → Groq
"openai/gpt-4o"                  → OpenAI
"anthropic/claude-sonnet-4-6"    → Anthropic
"ollama/llama3"                  → local Ollama
```

---

## MetaOS Agent — chat to build AI systems

`MetaOSAgent` is the top-level interface. Give it a goal in plain English; it plans, builds, tests, and optimizes the full pipeline — tools, workflows, prompts — without you writing any config by hand.

### Quickest start — terminal UI

```bash
python cli.py
```

This opens a multi-panel TUI: chat on the left, live tree / tool / workflow sidebar on the right, inline cost tracker, and tool-call progress as the agent works.

```
┌─ MetaOS ─────────────────────────────── $0.00421 • 3 calls ─┐
│                          │  Planning Tree                    │
│  You › Build me a        │  ─────────────────────────────    │
│  support pipeline        │  [in_progress] Customer Support   │
│                          │    [done] classify_message        │
│  MetaOS ›                │    [pending] write_reply          │
│    → plan_goal           │                                   │
│    → search_tools        │  Tools (1)                        │
│    → build_tool          │  ─────────────────────────────    │
│                          │  classify_message(msg) → str      │
│  Here's your plan:       │                                   │
│  ...                     │  Workflows (0)                    │
│                          │  None yet                         │
├──────────────────────────┴───────────────────────────────────┤
│ > Type a message or /help for commands…           [Ctrl+C]   │
└──────────────────────────────────────────────────────────────┘
```

#### CLI options

```bash
python cli.py --model anthropic/claude-sonnet-4-6
python cli.py --model openai/gpt-4o --session sessions/my_session.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `groq/llama-3.3-70b-versatile` | LiteLLM model string |
| `--input-cpm` | `0.59` | Cost per million input tokens (USD) |
| `--output-cpm` | `0.79` | Cost per million output tokens (USD) |
| `--session` | — | Path to a saved session JSON to load on startup |
| `--tools-dir` | `tools/` | Directory for the tool registry |
| `--workflows-dir` | `workflows/` | Directory for the workflow registry |

Default model and CPM values can also be set in `.env`:

```
METAOS_MODEL=anthropic/claude-sonnet-4-6
METAOS_INPUT_CPM=3.0
METAOS_OUTPUT_CPM=15.0
```

#### Slash commands

| Command | Action |
|---------|--------|
| `/help` | Show all commands |
| `/tree` | Print the current planning tree |
| `/tools` | List every tool in the registry |
| `/workflows` | List every workflow in the registry |
| `/cost` | Show session cost, API call count, tool call count |
| `/save [name]` | Save conversation to `sessions/{name}.json` |
| `/load [name]` | Load a saved session (lists available sessions if no name given) |
| `/reset` | Start a fresh session (keeps tool/workflow registries) |

Keyboard shortcuts: `Ctrl+S` save · `Ctrl+R` reset · `Ctrl+C` quit.

#### Session persistence

Sessions are saved to `sessions/{name}.json` and include the full conversation display, the raw LLM history (so the agent remembers context), and a pointer to the planning tree. Resume any past session with `/load` or `--session`.

---

### Python API

Use `MetaOSAgent` directly if you want to integrate it into a script or notebook:

```python
from src import MetaOSAgent

agent = MetaOSAgent("groq/llama-3.3-70b-versatile", 0.59, 0.79)

# 1. Give a goal — agent plans and waits for approval
print(agent.chat(
    "Build a customer support pipeline that classifies messages and writes tailored replies"
))

# 2. Approve the plan — agent builds tools and workflows bottom-up
print(agent.chat("The plan looks good, go ahead and build it"))

# 3. Generate test data and optimize
print(agent.chat(
    "Generate 20 test examples and run prompt optimization on the classifier"
))

# 4. Reuse in a new session (registries persist across resets)
agent.reset_session()
print(agent.chat(
    "I need another pipeline that handles refunds — reuse the classifier if possible"
))
# Agent calls search_tools() → finds the existing tool → reuses it
```

**What the agent does automatically:**
- Searches the tool and workflow registries before creating anything new
- Decomposes goals into a `TodoTree` and presents it for approval
- Builds tools bottom-up (leaf nodes first), sandbox-tests each one
- Generates workflow JSON configs with operator docs embedded in context
- Generates synthetic test data (or uses web search as grounding)
- Runs `PromptOptimizer` / `WorkflowOptimizer` / `ToolOptimizer` per node
- Updates the `TodoTree` with pass/fail status and experiment log references
- Writes pytest files for every synthesized tool

---

## Tool Forge — agentic tool creation

Agents can generate, test, and persist their own Python utility functions at runtime.

### Synthesize a tool from a description

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

The synthesizer generates Python source, runs it in a subprocess sandbox, retries on failure, and saves the validated function to the registry.

### React agent with live tool creation

Give a React agent a `registry` and it gains a built-in `forge_tool` meta-tool. If the agent determines it needs a function that doesn't exist, it can create one mid-reasoning:

```python
from src import React, ToolRegistry

registry = ToolRegistry("tools/")
agent = React("agent", "groq/llama-3.3-70b-versatile", 0.59, 0.79)

result = agent(
    "Calculate compound interest on $5000 at 7% for 10 years",
    tools=[],
    registry=registry,
)
# Agent calls forge_tool("calculate compound interest", "(p: float, r: float, n: int) -> float")
# then calls the newly created function in the same session
```

### Workflow with registry auto-resolution

Tools in a workflow config are auto-loaded from the registry if not passed manually:

```python
from src import Workflow, ToolRegistry

registry = ToolRegistry("tools/")
workflow = Workflow.from_json("workflow.json", registry=registry)
# Any tool name in operator config resolved from registry automatically
```

### Optimize an existing tool

```python
from src import ToolOptimizer, ToolRegistry

registry = ToolRegistry("tools/")
opt = ToolOptimizer("groq/llama-3.3-70b-versatile", 0.59, 0.79, registry)

record = opt.optimize(
    "compound_interest",
    test_cases=[
        {"args": {"principal": 1000, "rate": 0.1, "years": 3}, "expected": 1331.0},
    ],
    max_rounds=3,
)
```

---

## Documentation

- [Workflow reference](docs/workflow.md) — JSON schema, state, templates, tools, async, metrics
- [Optimization reference](docs/optimization.md) — Dataset, evaluators, PromptOptimizer, WorkflowOptimizer, experiment logs
- [Operator reference](docs/operators.md) — compact field reference for all operator types with JSON examples