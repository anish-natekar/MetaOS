# Operator Reference

Operators are the LLM-backed reasoning units in MetaOS workflows. Each operator type has a different reasoning pattern. All support self-consistency (`n > 1`).

---

## Available Operators

### `predict`
Single LLM call. Use for simple extraction, formatting, or generation where no reasoning chain is needed.

```json
{
  "type": "predict",
  "model": "groq/llama-3.3-70b-versatile",
  "input_cpm": 0.59,
  "output_cpm": 0.79
}
```

**Output**: string (or structured model if `response_basemodel` is set)

---

### `cot` — Chain of Thought
Two-call chain: reason step-by-step first, then give a final answer. Use for complex analysis, judgment, or tasks that benefit from explicit reasoning.

```json
{
  "type": "cot",
  "model": "groq/llama-3.3-70b-versatile",
  "input_cpm": 0.59,
  "output_cpm": 0.79
}
```

**Output**: `{"reasoning": "...", "answer": "..."}` (the step output key will hold this dict)

---

### `react` — Reasoning + Acting
Multi-turn tool-use loop. LLM reasons, calls tools, incorporates results, and repeats until it has a final answer. Use when the task requires looking something up, running a calculation, or calling an API.

```json
{
  "type": "react",
  "model": "groq/llama-3.3-70b-versatile",
  "input_cpm": 0.59,
  "output_cpm": 0.79,
  "tools": ["lookup_invoice", "check_status"],
  "max_iterations": 5
}
```

**Output**: string. Tools listed under `"tools"` must be registered in the `Workflow(tools={...})` dict or present in a `ToolRegistry`.

---

### `router`
Classifies input into exactly one of N categories using structured output. Use for branching, routing, and classification tasks.

```json
{
  "type": "router",
  "model": "groq/llama-3.3-70b-versatile",
  "input_cpm": 0.59,
  "output_cpm": 0.79
}
```

Step `args` must include `"routes"`:
```json
{ "args": { "routes": ["billing", "technical", "general"] } }
```

**Output**: string matching one of the provided routes

---

## Common Operator Fields

| Field | Required | Description |
|-------|----------|-------------|
| `type` | yes | `"predict"`, `"cot"`, `"react"`, or `"router"` |
| `model` | yes | LiteLLM model name, e.g. `"groq/llama-3.3-70b-versatile"` |
| `input_cpm` | yes | Cost per million input tokens (float) |
| `output_cpm` | yes | Cost per million output tokens (float) |
| `n` | no | Self-consistency: run N times, return majority vote (default: 1) |
| `system_prompt` | no | Override the default system prompt for this operator |
| `tools` | no | List of tool names (React only) |
| `max_iterations` | no | Max tool-use loop iterations (React only, default: 5) |

---

## Workflow Step Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Unique step identifier |
| `operator` | yes | Name of an operator defined under `"operators"` |
| `prompt` | yes* | Template string with `{key}` placeholders referencing input or prior step outputs |
| `prompt_file` | yes* | Path to a `.md` file containing the prompt template (use instead of `prompt`) |
| `output` | yes | State key where this step's result is stored |
| `depends_on` | no | List of step IDs that must complete before this step runs |
| `condition` | no | Python-like expression, e.g. `"{category} == 'billing'"` — skips step if false |
| `args` | no | Extra kwargs passed to the operator (e.g. `routes` for router) |

*Either `prompt` or `prompt_file` is required, not both.

---

## Choosing an Operator

| Task type | Best operator |
|-----------|--------------|
| Classify / route | `router` |
| Simple extraction or short generation | `predict` |
| Analysis, judgment, or multi-step reasoning | `cot` |
| Needs to call functions / look things up | `react` |

---

## Model Name Convention

MetaOS uses LiteLLM — prefix the model with the provider:

| Provider | Example |
|----------|---------|
| Groq | `"groq/llama-3.3-70b-versatile"` |
| OpenAI | `"openai/gpt-4o"` |
| Anthropic | `"anthropic/claude-sonnet-4-6"` |
| Local (Ollama) | `"ollama/llama3"` |
