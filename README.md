# MetaOS

A Python framework for building, running, and automatically optimizing multi-step AI agent workflows.

Define your pipeline as a JSON file. MetaOS executes it as a DAG, handles parallelism, routes between agents, and provides an optimizer that rewrites prompts and restructures the workflow automatically using labeled examples.

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
src/
  __init__.py      — public API
  llm_api.py       — LiteLLM wrapper with history, retries, metrics
  agent.py         — Predict, CoT, React, Router operators
  workflow.py      — DAG workflow engine
  optimizer.py     — PromptOptimizer, WorkflowOptimizer, Dataset

docs/
  workflow.md      — full workflow reference
  optimization.md  — optimization and evaluation reference

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

## Documentation

- [Workflow reference](docs/workflow.md) — JSON schema, state, templates, tools, async, metrics
- [Optimization reference](docs/optimization.md) — Dataset, evaluators, PromptOptimizer, WorkflowOptimizer, experiment logs