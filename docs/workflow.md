# Workflow

A `Workflow` chains agentic operators into a directed execution graph defined in a JSON file. Each step calls one operator, writes its output to shared state, and later steps can reference that output in their prompts. Independent steps run in parallel automatically.

---

## Quick Start

**1. Define your workflow in JSON:**

```json
{
  "name": "my_workflow",
  "operators": {
    "classifier": {
      "type": "router",
      "model": "llama-3.3-70b-versatile",
      "input_cpm": 0.59,
      "output_cpm": 0.79
    },
    "writer": {
      "type": "cot",
      "model": "llama-3.3-70b-versatile",
      "input_cpm": 0.59,
      "output_cpm": 0.79
    }
  },
  "steps": [
    {
      "id": "classify",
      "operator": "classifier",
      "prompt": "Classify this text: {input_text}",
      "args": { "routes": ["positive", "negative", "neutral"] },
      "output": "sentiment"
    },
    {
      "id": "respond",
      "operator": "writer",
      "prompt": "Write a {sentiment} response to: {input_text}",
      "depends_on": ["classify"],
      "output": "reply"
    }
  ]
}
```

**2. Run it from Python:**

```python
from src.workflow import Workflow

workflow = Workflow.from_json("my_workflow.json")
result = workflow.run({"input_text": "Your product is amazing!"})

print(result["sentiment"])       # "positive"
print(result["reply"])           # CoT output dict
print(result["reply"]["answer"]) # the final written response
print(workflow.metrics)          # tokens, cost, latency
```

---

## JSON File Structure

```
{
  "name": str,           — workflow name (used in logging)
  "operators": { ... },  — operator instances available to steps
  "steps": [ ... ]       — ordered list of steps to execute
}
```

### Operator Fields

Each key under `"operators"` is a name you choose. Steps reference it by that name.

| Field | Required | Description |
|-------|----------|-------------|
| `type` | yes | `"cot"`, `"react"`, `"router"`, or `"predict"` |
| `model` | yes | model name string (e.g. `"llama-3.3-70b-versatile"`) |
| `input_cpm` | yes | cost per million input tokens (float) |
| `output_cpm` | yes | cost per million output tokens (float) |
| `n` | no | number of independent calls for self-consistency. Default: `1` (single call). When `n > 1`, calls run in parallel and the majority answer wins. |
| `system_prompt` | no | overrides the operator's built-in default system prompt |
| `tools` | no | list of tool name strings — React only. Names must match keys in the `tools={}` dict passed to `from_json` |
| `max_iterations` | no | max tool loop iterations — React only. Default: `5` |

### Step Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | unique name for this step |
| `operator` | yes | must match a key defined under `"operators"` |
| `prompt` | no* | inline template string — `{key}` placeholders are filled from state |
| `prompt_file` | no* | path to a `.md` prompt file, relative to the workflow JSON's directory. `prompt_file` takes precedence if both are present. |
| `output` | yes | the state key under which this step's result is stored |
| `depends_on` | no | list of step `id`s that must complete before this step runs. Default: `[]` |
| `condition` | no | expression like `"{sentiment} == 'billing'"` — step is skipped if false, output set to `null` |
| `args` | no | extra kwargs passed to the operator call. Use `routes` for Router, or `max_iterations` to override React's default per step |

*Either `prompt` or `prompt_file` must be provided.

---

## State and Templates

Every value in the workflow — initial inputs and step outputs — lives in a single flat dict called **state**. As steps complete, their output is added to state under their `output` key.

```
Initial state:  { "query": "What is X?", "language": "English" }
After step 1:   { "query": "...", "language": "...", "category": "factual" }
After step 2:   { "query": "...", "language": "...", "category": "...", "findings": "..." }
```

Reference any state value in a prompt using `{key}`:

```json
"prompt": "Research this {category} topic in {language}: {query}"
```

### CoT outputs

`CoT` steps store their output as a dict with two fields. Use subscript syntax to access them:

| Template | Resolves to |
|----------|-------------|
| `{report}` | the full dict (e.g. `{'reasoning': '...', 'answer': '...'}`) |
| `{report[answer]}` | just the final answer string |
| `{report[reasoning]}` | just the reasoning string |

```json
"prompt": "Summarize this answer concisely: {report[answer]}"
```

### State is append-only

Once a key is written to state it cannot be overwritten. The workflow raises at:
- **Init time** — if two steps share the same `output` key
- **Run time** — if any input key you pass to `run()` conflicts with a step's `output` key

This ensures the full state at the end of a run is a complete, unambiguous trace of every value produced.

---

## Execution Model

### Sequential steps

A step with `depends_on` waits for those steps to finish first:

```json
{ "id": "step_a", "output": "result_a", ... },
{ "id": "step_b", "depends_on": ["step_a"], "output": "result_b", ... }
```

### Parallel steps

Steps with no dependency on each other run simultaneously. The workflow detects this automatically from the `depends_on` graph:

```json
{ "id": "summarise_a", "output": "summary_a", ... },
{ "id": "summarise_b", "output": "summary_b", ... },
{
  "id": "compare",
  "depends_on": ["summarise_a", "summarise_b"],
  "prompt": "Compare:\nA: {summary_a[answer]}\nB: {summary_b[answer]}",
  "output": "comparison"
}
```

`summarise_a` and `summarise_b` fire at the same time. `compare` waits until both finish.

> **Note:** `CoT` and `React` operators are stateful (they use internal conversation history). Two parallel steps must **not** share the same operator instance. Define separate operator entries for each parallel step.

```json
"operators": {
  "writer_a": { "type": "cot", ... },
  "writer_b": { "type": "cot", ... }
}
```

### Conditional steps

A step with a `condition` is skipped (output = `null`) if the condition is false. Use conditions to implement branching after a Router step:

```json
{
  "id": "billing_handler",
  "depends_on": ["classify"],
  "condition": "{department} == 'billing'",
  "output": "resolution"
}
```

Conditions must use `==` or `!=` comparisons on string values. The `{key}` placeholder is replaced with the actual state value before evaluation.

When branching on a Router, make your conditions mutually exclusive and ensure they all write to the same `output` key — exactly one branch runs, so only one write happens.

---

## Prompt Files

Long or structured prompts can live in separate `.md` files instead of inline in the JSON. Markdown lets you use headers, lists, and code blocks to write clearer prompts.

```
my_workflow/
  workflow.json
  prompts/
    classify.md
    handle_billing.md
    write_reply.md
```

Reference a prompt file in a step using `prompt_file` (path relative to the workflow JSON):

```json
{
  "id": "classify",
  "operator": "classifier",
  "prompt_file": "prompts/classify.md",
  "output": "category"
}
```

`prompts/classify.md`:
```markdown
## Role
You are a customer support classifier.

## Task
Classify the following message into exactly one of the provided categories.

**Message:** {message}
**Customer tier:** {tier}

## Rules
- Be decisive — return only the category name
- If ambiguous between billing and technical, prefer billing
```

`{key}` template variables work identically in `.md` files. Literal `{` or `}` must be escaped as `{{` or `}}`.

---

## Self-Consistency (`n`)

Every operator supports running `n` independent calls in parallel and returning the majority answer. Useful when reliability matters more than latency.

```json
"operators": {
  "fact_checker": {
    "type": "predict",
    "model": "llama-3.3-70b-versatile",
    "input_cpm": 0.59,
    "output_cpm": 0.79,
    "n": 5
  },
  "analyser": {
    "type": "cot",
    "model": "llama-3.3-70b-versatile",
    "input_cpm": 0.59,
    "output_cpm": 0.79,
    "n": 3
  }
}
```

- **`Predict`** (new): a single plain LLM call with no reasoning step. With `n > 1` it becomes self-consistent prediction — the simplest form of majority vote.
- **`CoT`** with `n > 1`: runs `n` full reasoning chains, votes on the final `answer` field, returns `(reasoning, winning_answer)` from the first run that produced the winning answer.
- **`Router`** and **`React`** with `n > 1`: majority vote on the output string.

---

## Run Logging

Pass `log_dir` to `run()` to write a timestamped directory of files after each run:

```python
result = workflow.run(inputs, log_dir="runs/")
```

```
runs/
  20260623_143022/
    inputs.json    — inputs passed to run()
    state.json     — full flat state (every intermediate + final value)
    metrics.json   — per-operator token counts, cost, latency
```

Each run gets its own directory, so logs never overwrite each other. `state.json` is the primary input for the optimizer — it records the full chain of intermediate outputs.

```python
# Also works with arun()
result = await workflow.arun(inputs, log_dir="runs/")
```

---

## Checkpointing

A checkpoint captures the complete frozen state of a workflow version — structure, all resolved prompt text, and optional performance metadata — in a single self-contained JSON file.

```python
# Save current workflow as a checkpoint
workflow.save_checkpoint("checkpoints/v1.json", description="baseline")

# Attach eval metrics when saving
workflow.save_checkpoint(
    "checkpoints/v2.json",
    description="after prompt opt round 1",
    metrics={"avg_score": 0.87, "n_eval_runs": 50}
)

# Load a checkpoint (fully self-contained — no separate prompt files needed)
workflow = Workflow.from_checkpoint("checkpoints/v1.json", tools={...})
```

Checkpoint file format:
```json
{
  "version": "v1",
  "timestamp": "2026-06-23T14:30:00",
  "description": "baseline",
  "workflow": { ...full workflow.json content... },
  "prompts": {
    "classify":       "resolved prompt text...",
    "handle_billing": "resolved prompt text..."
  },
  "metrics": {
    "avg_score": 0.82,
    "n_eval_runs": 50
  }
}
```

Prompts are stored as resolved text (not file paths), so checkpoints are fully self-contained: shareable, diffable, and git-committable. `from_checkpoint` is identical to `from_json` except it reads the embedded workflow and prompts from the checkpoint block.

---

## Using Tools with React

Tools are Python callables registered in Python (they can't go in JSON). Pass them as a dict to `from_json`:

```python
def lookup_invoice(invoice_id: str) -> str:
    """Look up invoice details by ID."""
    return f"Invoice {invoice_id}: $49.99, paid on 2026-05-01."

def check_system_status(service: str) -> str:
    """Check the operational status of a service."""
    return f"{service} is operational."

workflow = Workflow.from_json(
    "support.json",
    tools={
        "lookup_invoice": lookup_invoice,
        "check_system_status": check_system_status,
    }
)
```

In the JSON, reference them by their dict key:

```json
"operators": {
  "support_agent": {
    "type": "react",
    "model": "llama-3.3-70b-versatile",
    "input_cpm": 0.59,
    "output_cpm": 0.79,
    "tools": ["lookup_invoice", "check_system_status"],
    "max_iterations": 6
  }
}
```

Tool functions **must have type annotations and a docstring** — the schema sent to the LLM is generated from these automatically.

---

## Full Example: Customer Support Pipeline

Classifies an incoming message, routes it to the right specialist agent, and writes a polished reply.

```json
{
  "name": "customer_support",

  "operators": {
    "classifier": {
      "type": "router",
      "model": "llama-3.3-70b-versatile",
      "input_cpm": 0.59,
      "output_cpm": 0.79,
      "system_prompt": "You are a customer support classifier. Be decisive."
    },
    "billing_agent": {
      "type": "react",
      "model": "llama-3.3-70b-versatile",
      "input_cpm": 0.59,
      "output_cpm": 0.79,
      "tools": ["lookup_invoice", "check_subscription"],
      "max_iterations": 6
    },
    "tech_agent": {
      "type": "react",
      "model": "llama-3.3-70b-versatile",
      "input_cpm": 0.59,
      "output_cpm": 0.79,
      "tools": ["search_docs", "check_system_status"],
      "max_iterations": 8
    },
    "general_agent": {
      "type": "cot",
      "model": "llama-3.3-70b-versatile",
      "input_cpm": 0.59,
      "output_cpm": 0.79
    },
    "responder": {
      "type": "cot",
      "model": "llama-3.3-70b-versatile",
      "input_cpm": 0.59,
      "output_cpm": 0.79,
      "system_prompt": "You write polite, concise customer support replies."
    }
  },

  "steps": [
    {
      "id": "classify",
      "operator": "classifier",
      "prompt": "Classify this customer message: {message}",
      "args": { "routes": ["billing", "technical", "general"] },
      "output": "department"
    },
    {
      "id": "handle_billing",
      "operator": "billing_agent",
      "prompt": "Resolve this billing issue for customer {customer_id}: {message}",
      "depends_on": ["classify"],
      "condition": "{department} == 'billing'",
      "output": "resolution"
    },
    {
      "id": "handle_tech",
      "operator": "tech_agent",
      "prompt": "Diagnose and resolve this technical issue: {message}",
      "depends_on": ["classify"],
      "condition": "{department} == 'technical'",
      "output": "resolution"
    },
    {
      "id": "handle_general",
      "operator": "general_agent",
      "prompt": "Answer this general enquiry: {message}",
      "depends_on": ["classify"],
      "condition": "{department} == 'general'",
      "output": "resolution"
    },
    {
      "id": "write_reply",
      "operator": "responder",
      "prompt": "Write a customer-facing reply based on this resolution:\n{resolution}",
      "depends_on": ["handle_billing", "handle_tech", "handle_general"],
      "output": "final_reply"
    }
  ]
}
```

```python
result = workflow.run({
    "message": "I was charged twice this month",
    "customer_id": "USR-9921"
})

print(result["department"])                 # "billing"
print(result["resolution"])                 # React agent's findings string
print(result["final_reply"]["answer"])      # polished customer reply
print(workflow.metrics["total_cost"])       # total $ spent across all operators
```

---

## Async Usage

`run()` uses `asyncio.run()` internally and works in scripts. In an environment that already has a running event loop (e.g. Jupyter), use `arun()` instead:

```python
result = await workflow.arun({"message": "..."})
```

---

## Accessing Metrics

After a run, `workflow.metrics` aggregates token usage and cost across all operator instances:

```python
m = workflow.metrics

m["total_input_tokens"]   # int
m["total_output_tokens"]  # int
m["total_cost"]           # float, $ total
m["total_api_time"]       # float, seconds
m["total_calls"]          # int, total LLM calls made
m["total_tool_calls"]     # int
m["per_operator"]         # dict keyed by operator name with individual breakdowns
```

`workflow.state` holds the complete flat state dict after the run — every input and every step output, in one place.
