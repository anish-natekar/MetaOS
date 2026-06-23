# Optimization

MetaOS includes an automated optimization loop that improves workflows using labeled examples. Two optimizers are available: `PromptOptimizer` rewrites individual step prompts; `WorkflowOptimizer` changes the DAG structure itself.

For building workflows from scratch, see [Planning with TodoTree](#planning-with-todotree) — the `Planner` agent decomposes a goal into a `TodoTree` and the optimizers can write back to it as an institutional memory of what was tried.

---

## Dataset

A `Dataset` holds labeled `(inputs, expected_output)` pairs used for training and evaluation.

```python
from src import Dataset

# Load from a JSON file
ds = Dataset.from_json("data/examples.json")

# Or build from a list
ds = Dataset.from_list([
    {"inputs": {"message": "I was charged twice"}, "expected": "billing"},
    {"inputs": {"message": "My app keeps crashing"}, "expected": "technical"},
])

# Split into train / test
train, test = ds.split(test_size=0.2, seed=42)

print(train)   # Dataset(8 examples)
print(test)    # Dataset(2 examples)
```

### JSON file format

```json
[
  {
    "inputs": {"message": "I was charged twice this month", "customer_id": "USR-001"},
    "expected": "billing"
  },
  {
    "inputs": {"message": "The app crashes on startup", "customer_id": "USR-002"},
    "expected": "technical"
  }
]
```

If your file uses different key names, pass `inputs_key` and `expected_key`:

```python
ds = Dataset.from_json("data.json", inputs_key="query", expected_key="label")
```

---

## Evaluators

An evaluator is a function `(inputs: dict, state: dict) -> float` that scores a single workflow run from 0.0 (completely wrong) to 1.0 (perfect). `Dataset.make_evaluator()` creates one automatically.

### Exact match

Best for classification tasks — Router routing, short categorical answers.

```python
evaluator = train.make_evaluator(
    output_key="department",    # which state key to score
    mode="exact_match",
)
```

Returns 1.0 if `state["department"]` matches the expected string (case-insensitive), else 0.0.

### LLM judge

Best for open-ended outputs — CoT answers, written replies.

```python
evaluator = train.make_evaluator(
    output_key="final_reply",
    mode="llm_judge",
    judge_model="groq/llama-3.3-70b-versatile",
    input_cpm=0.59,
    output_cpm=0.79,
)
```

An LLM compares the produced output to the expected output and returns a score from 0.0 to 1.0.

### Custom evaluator

Write your own — just match the signature:

```python
def my_evaluator(inputs: dict, state: dict) -> float:
    reply = state.get("final_reply", {})
    answer = reply.get("answer", "") if isinstance(reply, dict) else reply
    # compare to expected, run checks, etc.
    return 1.0 if "sorry" not in answer.lower() else 0.0
```

---

## Evaluating a workflow

Run every example through the workflow and get aggregate metrics:

```python
results = test.evaluate(
    workflow=workflow,
    output_key="department",
    evaluator=evaluator,    # optional — defaults to exact_match
)

print(results)
# {
#   "avg_score": 0.87,
#   "n_examples": 10,
#   "n_passing": 9,       # score >= 0.7
#   "scores": [1.0, 0.5, 1.0, ...]
# }
```

---

## PromptOptimizer

Runs a backward-pass loop: finds which step first went wrong in failing runs, rewrites that step's prompt, re-evaluates, and keeps the change only if the overall score improved.

```python
from src import PromptOptimizer, Dataset, Workflow

workflow = Workflow.from_json("my_workflow/workflow.json")
ds = Dataset.from_json("data/examples.json")
train, test = ds.split()

evaluator = train.make_evaluator(
    output_key="final_reply",
    mode="llm_judge",
    judge_model="groq/llama-3.3-70b-versatile",
    input_cpm=0.59,
    output_cpm=0.79,
)

optimizer = PromptOptimizer(
    workflow=workflow,
    evaluator=evaluator,
    optimizer_model="groq/llama-3.3-70b-versatile",
    input_cpm=0.59,
    output_cpm=0.79,
)

improved = optimizer.optimize(
    examples=train.inputs,
    checkpoint_dir="checkpoints/",
    max_rounds=3,
    score_threshold=0.7,       # runs below this score are "failing"
    min_examples_to_fix=3,     # stop if fewer than this many failures remain
    experiment_dir="experiments/",  # optional — writes a .md log per run
)

# Evaluate on held-out test set
print(test.evaluate(improved, output_key="final_reply", evaluator=evaluator))
```

### What happens each round

```
1. Run all training examples → collect (inputs, state, score) per run
2. Split into passing (score ≥ threshold) and failing
3. For each failing run: ask LLM judge which step first went wrong
4. Pick the step with the most attributed failures
5. Call optimizer LLM → generate improved prompt for that step
6. Patch the workflow with the new prompt
7. Re-run all examples → measure new overall score
8. Accept (save checkpoint) if score improved, revert otherwise
9. Repeat for max_rounds
```

### Checkpoints

Every accepted round is saved as a checkpoint. `current.txt` always points to the best version:

```
checkpoints/
  v1.json       ← baseline
  v2.json       ← after round 1 (avg_score=0.84)
  v3.json       ← after round 2 (avg_score=0.89)
  current.txt   ← "checkpoints/v3.json"
```

Manual rollback:

```python
workflow = Workflow.from_checkpoint("checkpoints/v2.json", tools={...})
```

---

## WorkflowOptimizer

When prompt optimization plateaus, `WorkflowOptimizer` proposes structural changes to the DAG: adding steps, removing steps, or rewiring dependencies.

```python
from src import WorkflowOptimizer

wopt = WorkflowOptimizer(
    workflow=workflow,
    evaluator=evaluator,
    optimizer_model="groq/llama-3.3-70b-versatile",
    input_cpm=0.59,
    output_cpm=0.79,
)

improved = wopt.optimize(
    examples=train.inputs,
    checkpoint_dir="checkpoints/",
    max_rounds=3,
    score_threshold=0.7,
    experiment_dir="experiments/",
)
```

### Supported structural changes

The optimizer proposes one of three change types per round:

**`add_step`** — insert a new step that uses an existing operator:
```json
{
  "type": "add_step",
  "step": {
    "id": "fetch_history",
    "operator": "data_agent",
    "prompt": "Fetch account history for customer {customer_id}.",
    "output": "account_history",
    "depends_on": ["classify"],
    "condition": null
  },
  "rationale": "Billing agent needs account history to resolve disputes."
}
```

**`remove_step`** — remove a step, providing new dependencies for anything that depended on it:
```json
{
  "type": "remove_step",
  "step_id": "intermediate_check",
  "update_depends_on": {
    "write_reply": ["handle_billing"]
  },
  "rationale": "The check step adds latency without improving accuracy."
}
```

**`change_depends_on`** — rewire a step to fire earlier or later:
```json
{
  "type": "change_depends_on",
  "step_id": "write_reply",
  "new_depends_on": ["handle_billing", "handle_tech", "handle_general", "fetch_history"],
  "rationale": "write_reply should also see the account history."
}
```

All proposals are validated (no cycles, no duplicate IDs, no missing operators) before being applied. Invalid proposals are logged and skipped.

### Typical workflow

```python
# 1. Prompt optimization first
p_optimizer = PromptOptimizer(workflow, evaluator, model, cpm_in, cpm_out)
workflow = p_optimizer.optimize(train.inputs, "checkpoints/", max_rounds=5)

# 2. Structural optimization if prompt opt plateaued
w_optimizer = WorkflowOptimizer(workflow, evaluator, model, cpm_in, cpm_out)
workflow = w_optimizer.optimize(train.inputs, "checkpoints/", max_rounds=3)

# 3. Evaluate final result on test set
print(test.evaluate(workflow, output_key="final_reply", evaluator=evaluator))
```

---

## Experiment Logs

Pass `experiment_dir` to either optimizer to get a human-readable `.md` log per run:

```
experiments/
  exp_20260623_143022.md    ← PromptOptimizer run
  wexp_20260624_091500.md   ← WorkflowOptimizer run
```

Each file records, per round:
- Optimization type and target step
- Issues detected (failing examples)
- Prompt or structural change proposed
- Metrics before and after (avg score, failing count, optimizer cost)
- Decision: ACCEPTED or REVERTED

Logs are written incrementally — a crash mid-run preserves all completed rounds.

---

## Planning with TodoTree

`Planner` takes a goal string and recursively decomposes it into a hierarchical `TodoTree`. Each node records *what* it does, *why* it exists, and its current status. Failed branches are kept permanently as institutional memory so the optimizer doesn't repeat them.

### Build a plan from a goal

```python
from src import Planner, TodoTree

planner = Planner("groq/llama-3.3-70b-versatile", input_cpm=0.59, output_cpm=0.79)
tree = planner.plan(
    goal="Customer support pipeline that classifies, resolves, and replies to messages",
    context="Tools available: lookup_invoice, check_system_status",
    max_depth=4,
)
tree.save("plan/todo_tree.json")
print(tree.get_subtree_text())
```

Each node in the tree has:

| Field | Purpose |
|-------|---------|
| `name` | Short title (≤8 words) |
| `description` | Abstracted solution (≤3 sentences) |
| `why` | What problem this addresses (≤3 sentences) |
| `status` | `pending` / `in_progress` / `done` / `failed` |
| `is_leaf` | `True` when directly implementable by a single workflow or tool |
| `implementation_note` | Leaf only — what implements this |
| `workflow_ref` | Path to a linked workflow JSON |
| `experiment_refs` | Paths to all experiment logs trialled at this node |

### Link a workflow to a leaf node

```python
tree = TodoTree.from_json("plan/todo_tree.json")
leaf_id = "a3f1"   # get from tree.get_leaves() or get_subtree_text()

tree.attach_workflow(leaf_id, "plan/workflows/classify.json")
tree.save("plan/todo_tree.json")
```

### Optimize with tree tracking

Pass `todo_tree`, `tree_node_id`, and `tree_path` to either optimizer to have it write back results automatically:

```python
from src import PromptOptimizer, Dataset

ds = Dataset.from_json("data/examples.json")
train, test = ds.split()
evaluator = train.make_evaluator("final_reply", mode="llm_judge",
                                  judge_model="groq/llama-3.3-70b-versatile",
                                  input_cpm=0.59, output_cpm=0.79)

workflow = Workflow.from_json("plan/workflows/classify.json")
optimizer = PromptOptimizer(workflow, evaluator, "groq/llama-3.3-70b-versatile", 0.59, 0.79)

workflow = optimizer.optimize(
    examples=train.inputs,
    checkpoint_dir="checkpoints/",
    experiment_dir="experiments/",
    todo_tree=tree,
    tree_node_id=leaf_id,
    tree_path="plan/todo_tree.json",   # saved after every round that changes the tree
)
```

After optimization:
- Every experiment log path is appended to `node.experiment_refs`
- If any round was accepted → `node.status = "done"`
- If no round improved → `node.status = "failed"` (branch kept, never deleted)

### Inspect the tree

```python
# Full tree render
print(tree.get_subtree_text())

# Compact one-level summary (useful as optimizer context)
print(tree.summary())

# Leaves that still need workflows
leaves = tree.get_leaves()

# All failed branches — what was tried and didn't work
failures = tree.get_failed_branches()
for node in failures:
    print(f"{node.name}: tried {node.experiment_refs}")
```

### Extend a node manually

Add child nodes to an existing node to model a new sub-problem discovered during optimization:

```python
child_id = tree.add_node(
    parent_id=leaf_id,
    name="Handle invoice lookup failures",
    description="Gracefully recover when lookup_invoice returns an error.",
    why="5% of billing disputes fail because the invoice ID is missing or malformed.",
    is_leaf=True,
    implementation_note="Add a fallback React step that prompts user for a valid ID.",
)
tree.save("plan/todo_tree.json")
```

Or use `Planner.extend()` to decompose an existing node automatically:

```python
tree = planner.extend(tree, node_id=leaf_id, context="lookup_invoice can return None")
tree.save("plan/todo_tree.json")
```

### Loading an existing tree

```python
tree = TodoTree.from_json("plan/todo_tree.json")
```