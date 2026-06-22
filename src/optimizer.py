import os
import json
import random
import copy
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .llm_api import LLM_API
from .workflow import Workflow, WorkflowStep


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------

class RunResult:
    def __init__(self, inputs: dict, state: dict, score: float):
        self.inputs = inputs
        self.state = state
        self.score = score


# ---------------------------------------------------------------------------
# Experiment log
# ---------------------------------------------------------------------------

class _ExperimentLog:
    """Writes one .md file per optimize() call, appended incrementally."""

    def __init__(self, path: str, workflow_name: str, baseline_checkpoint: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w")
        self._rounds: list[dict] = []
        self._start_score: float | None = None
        self._end_score: float | None = None

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._f.write(f"# Experiment: {ts}\n")
        self._f.write(f"**Type:** prompt optimization\n")
        self._f.write(f"**Workflow:** {workflow_name}\n")
        self._f.write(f"**Baseline checkpoint:** {baseline_checkpoint}\n\n")
        self._f.flush()

    def set_baseline(self, score: float, n: int):
        self._start_score = score
        self._f.write(f"**Baseline score:** {score:.3f} over {n} examples\n\n")
        self._f.flush()

    def write_round(
        self,
        round_num: int,
        step_id: str,
        issues: str,
        old_prompt: str,
        new_prompt: str,
        score_before: float,
        score_after: float,
        failing_before: int,
        failing_after: int,
        n_total: int,
        accepted: bool,
        checkpoint_path: str = None,
        optimizer_cost: float = 0.0,
    ):
        self._f.write(f"---\n\n## Round {round_num}\n")
        self._f.write(f"**Optimization type:** prompt\n")
        self._f.write(f"**Target step:** `{step_id}`\n\n")
        self._f.write(f"### Issues Detected\n\n{issues}\n\n")
        self._f.write(f"### Prompt Change\n\n**Before:**\n```\n{old_prompt.strip()}\n```\n\n")
        self._f.write(f"**After:**\n```\n{new_prompt.strip()}\n```\n\n")
        self._f.write(f"### Metrics\n\n")
        self._f.write(f"| Metric | Before | After |\n|--------|--------|-------|\n")
        self._f.write(f"| Avg score | {score_before:.3f} | {score_after:.3f} |\n")
        self._f.write(f"| Failing runs | {failing_before}/{n_total} | {failing_after}/{n_total} |\n")
        self._f.write(f"| Optimizer cost | ${optimizer_cost:.4f} | — |\n\n")

        if accepted:
            self._f.write(f"### Decision: **ACCEPTED** → saved as `{checkpoint_path}`\n\n")
            self._end_score = score_after
        else:
            self._f.write(
                f"### Decision: **REVERTED** — score {score_after:.3f} "
                f"did not improve on {score_before:.3f}\n\n"
            )
        self._f.flush()
        self._rounds.append({"accepted": accepted})

    def close(self, final_checkpoint: str = None):
        accepted = sum(1 for r in self._rounds if r["accepted"])
        reverted = len(self._rounds) - accepted
        end = self._end_score if self._end_score is not None else self._start_score

        self._f.write(f"---\n\n## Summary\n\n")
        self._f.write(
            f"- Rounds attempted: {len(self._rounds)} | "
            f"Accepted: {accepted} | Reverted: {reverted}\n"
        )
        if self._start_score is not None:
            self._f.write(f"- Score: {self._start_score:.3f} → {end:.3f}\n")
        if final_checkpoint:
            self._f.write(f"- Final checkpoint: `{final_checkpoint}`\n")
        self._f.close()


# ---------------------------------------------------------------------------
# PromptOptimizer
# ---------------------------------------------------------------------------

class PromptOptimizer:
    """
    Runs a backward-pass prompt optimization loop over a Workflow.

    Usage:
        optimizer = PromptOptimizer(
            workflow=workflow,
            evaluator=my_score_fn,          # (inputs, state) -> float 0-1
            optimizer_model="groq/llama-3.3-70b-versatile",
            input_cpm=0.59,
            output_cpm=0.79,
        )
        improved = optimizer.optimize(
            examples=[{"query": "..."}, ...],
            checkpoint_dir="checkpoints/",
            max_rounds=3,
            score_threshold=0.7,
            min_examples_to_fix=3,
            experiment_dir="experiments/",  # optional
        )
    """

    _JUDGE_SYSTEM = (
        "You are an AI workflow quality evaluator. "
        "Your job is to determine whether a single step in an AI pipeline produced correct output. "
        "Reply with exactly one word: 'yes' or 'no'."
    )

    _OPTIMIZER_SYSTEM = (
        "You are an expert prompt engineer. "
        "You receive a failing prompt template and examples of bad outputs, "
        "and rewrite the prompt to fix the failures. "
        "Return ONLY the improved prompt template text — no preamble, no explanation, "
        "no code fences."
    )

    def __init__(
        self,
        workflow: Workflow,
        evaluator: Callable[[dict, dict], float],
        optimizer_model: str,
        input_cpm: float,
        output_cpm: float,
    ):
        self.workflow = workflow
        self.evaluator = evaluator
        self._judge = LLM_API(
            model_name=optimizer_model,
            system_prompt=self._JUDGE_SYSTEM,
            input_token_cpm=input_cpm,
            output_token_cpm=output_cpm,
        )
        self._optimizer = LLM_API(
            model_name=optimizer_model,
            system_prompt=self._OPTIMIZER_SYSTEM,
            input_token_cpm=input_cpm,
            output_token_cpm=output_cpm,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def optimize(
        self,
        examples: list[dict],
        checkpoint_dir: str,
        max_rounds: int = 3,
        score_threshold: float = 0.7,
        min_examples_to_fix: int = 3,
        experiment_dir: str = None,
    ) -> Workflow:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Baseline checkpoint
        baseline_path = os.path.join(checkpoint_dir, "v1.json")
        self.workflow.save_checkpoint(baseline_path, description="baseline")
        best_path = baseline_path

        # Experiment log
        log: _ExperimentLog | None = None
        if experiment_dir:
            os.makedirs(experiment_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log = _ExperimentLog(
                path=os.path.join(experiment_dir, f"exp_{ts}.md"),
                workflow_name=self.workflow.name,
                baseline_checkpoint=baseline_path,
            )

        # Initial run
        runs = self._run_all(examples)
        best_score = sum(r.score for r in runs) / len(runs)
        if log:
            log.set_baseline(best_score, len(runs))

        version = 1

        for round_num in range(1, max_rounds + 1):
            current_score = best_score
            failing = [r for r in runs if r.score < score_threshold]

            if len(failing) < min_examples_to_fix:
                break

            # Attribute each failure to a culprit step (expensive — one judge call per step per run)
            culprit_of: dict[int, str | None] = {
                id(r): self._find_culprit(r) for r in failing
            }
            culprits = [c for c in culprit_of.values() if c]
            if not culprits:
                break

            # Step with the most failures
            target_id = Counter(culprits).most_common(1)[0][0]
            step = next(s for s in self.workflow.steps if s.id == target_id)
            failing_for_step = [r for r in failing if culprit_of[id(r)] == target_id]

            old_prompt = step.prompt_template
            cost_before = self._optimizer_cost()

            new_prompt = self._generate_prompt(step, failing_for_step[:5])
            optimizer_cost = self._optimizer_cost() - cost_before

            # Patch, re-run everything, measure
            self._patch_prompt(target_id, new_prompt)
            new_runs = self._run_all(examples)
            new_score = sum(r.score for r in new_runs) / len(new_runs)
            failing_after = sum(1 for r in new_runs if r.score < score_threshold)

            accepted = new_score > current_score

            if accepted:
                version += 1
                ckpt_path = os.path.join(checkpoint_dir, f"v{version}.json")
                self.workflow.save_checkpoint(
                    ckpt_path,
                    description=f"round {round_num}: improved step '{target_id}'",
                    metrics={"avg_score": new_score, "n_eval_runs": len(examples)},
                )
                best_score = new_score
                best_path = ckpt_path
                runs = new_runs
            else:
                self._patch_prompt(target_id, old_prompt)  # revert
                ckpt_path = None

            if log:
                issues = self._summarise_issues(
                    [self._describe_failure(r, target_id) for r in failing_for_step]
                )
                log.write_round(
                    round_num=round_num,
                    step_id=target_id,
                    issues=issues,
                    old_prompt=old_prompt,
                    new_prompt=new_prompt,
                    score_before=current_score,
                    score_after=new_score,
                    failing_before=len(failing_for_step),
                    failing_after=failing_after,
                    n_total=len(examples),
                    accepted=accepted,
                    checkpoint_path=ckpt_path,
                    optimizer_cost=optimizer_cost,
                )

        # Write current.txt pointing to the best accepted checkpoint
        with open(os.path.join(checkpoint_dir, "current.txt"), "w") as f:
            f.write(best_path + "\n")

        if log:
            log.close(final_checkpoint=best_path)

        return self.workflow

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_all(self, examples: list[dict]) -> list[RunResult]:
        results = []
        for inputs in examples:
            state = self.workflow.run(inputs)
            score = self.evaluator(inputs, state)
            results.append(RunResult(inputs=inputs, state=state, score=score))
        return results

    def _find_culprit(self, run: RunResult) -> str | None:
        """Return the ID of the first step that produced a bad output, or None."""
        for step in self.workflow.steps:
            output = run.state.get(step.output_key)
            if output is None:
                continue  # step was skipped (conditional)

            prompt = (
                f"Step '{step.id}' in an AI workflow pipeline produced the following output.\n\n"
                f"Step's task (prompt template): {step.prompt_template[:400]}\n\n"
                f"Step's output: {str(output)[:600]}\n\n"
                f"The overall workflow scored {run.score:.2f}/1.0 for this input "
                f"(1.0 = perfect, 0.0 = completely wrong).\n\n"
                f"Was this step's output correct and appropriate for its task? "
                f"Reply with exactly one word: 'yes' or 'no'."
            )
            verdict = self._judge(prompt).strip().lower()
            self._judge.reset_history()

            if verdict.startswith("no"):
                return step.id
        return None

    def _describe_failure(self, run: RunResult, step_id: str) -> str:
        step = next(s for s in self.workflow.steps if s.id == step_id)
        output = run.state.get(step.output_key, "")
        inputs_summary = json.dumps(run.inputs, default=str)[:300]
        output_summary = str(output)[:300]
        return f"Input: {inputs_summary}\nOutput: {output_summary}"

    def _summarise_issues(self, descriptions: list[str]) -> str:
        if not descriptions:
            return "No specific issues captured."
        items = "\n\n".join(f"**Example {i+1}:**\n{d}" for i, d in enumerate(descriptions[:5]))
        return f"{len(descriptions)} failing example(s):\n\n{items}"

    def _downstream_context(self, step_id: str) -> str:
        downstream = [s for s in self.workflow.steps if step_id in s.depends_on]
        if not downstream:
            return "This is the final step — its output is the workflow's result."
        parts = [
            f"Step '{s.id}' ({type(self.workflow.operators[s.operator_id]).__name__}) "
            f"uses this output in its prompt."
            for s in downstream
        ]
        return "\n".join(parts)

    def _generate_prompt(self, step: WorkflowStep, failing_runs: list[RunResult]) -> str:
        examples_text = "\n\n".join(
            f"Example {i+1}:\n{self._describe_failure(r, step.id)}"
            for i, r in enumerate(failing_runs)
        )
        downstream = self._downstream_context(step.id)
        prompt = (
            f"CURRENT PROMPT TEMPLATE:\n{step.prompt_template}\n\n"
            f"WHAT THE NEXT STEP EXPECTS:\n{downstream}\n\n"
            f"FAILING EXAMPLES (input → bad output):\n{examples_text}\n\n"
            f"Write an improved prompt template that fixes these failures. "
            f"Use the same {{placeholder}} syntax for template variables. "
            f"Return ONLY the new prompt template text."
        )
        result = self._optimizer(prompt)
        self._optimizer.reset_history()
        return result

    def _patch_prompt(self, step_id: str, prompt: str):
        """Patch a step's prompt in-place (also updates _config so checkpoints capture it)."""
        step = next(s for s in self.workflow.steps if s.id == step_id)
        step.prompt_template = prompt
        for cfg_step in self.workflow._config.get("steps", []):
            if cfg_step["id"] == step_id:
                cfg_step["prompt"] = prompt
                cfg_step.pop("prompt_file", None)
                break

    def _optimizer_cost(self) -> float:
        m = self._optimizer.metrics
        return m.input_cost + m.output_cost


# ---------------------------------------------------------------------------
# Dataset — supervised evaluation and training data
# ---------------------------------------------------------------------------

@dataclass
class Example:
    inputs: dict
    expected: Any


class Dataset:
    """
    A collection of labeled examples for training and evaluating workflows.

    Each example is an (inputs, expected_output) pair. The dataset can be
    split into train/test subsets and used to generate evaluator functions
    compatible with PromptOptimizer.

    Usage:
        ds = Dataset.from_json("data/examples.json")
        train, test = ds.split(test_size=0.2)

        evaluator = ds.make_evaluator(
            output_key="final_reply",
            mode="llm_judge",
            judge_model="groq/llama-3.3-70b-versatile",
            input_cpm=0.59,
            output_cpm=0.79,
        )

        optimizer = PromptOptimizer(workflow, evaluator, ...)
        optimizer.optimize(examples=train.inputs, ...)

        results = test.evaluate(workflow, output_key="final_reply", evaluator=evaluator)
    """

    def __init__(self, examples: list[Example]):
        self.examples = examples

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_list(
        cls,
        data: list[dict],
        inputs_key: str = "inputs",
        expected_key: str = "expected",
    ) -> "Dataset":
        """Build from a list of dicts with `inputs` and `expected` keys."""
        return cls([Example(inputs=d[inputs_key], expected=d[expected_key]) for d in data])

    @classmethod
    def from_json(
        cls,
        path: str,
        inputs_key: str = "inputs",
        expected_key: str = "expected",
    ) -> "Dataset":
        """Load from a JSON file containing a list of {inputs, expected} objects."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_list(data, inputs_key, expected_key)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(self, test_size: float = 0.2, seed: int = 42) -> tuple["Dataset", "Dataset"]:
        """Return (train, test) datasets. Shuffled deterministically."""
        examples = list(self.examples)
        rng = random.Random(seed)
        rng.shuffle(examples)
        n_test = max(1, int(len(examples) * test_size))
        return Dataset(examples[n_test:]), Dataset(examples[:n_test])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def inputs(self) -> list[dict]:
        """List of input dicts — pass directly to optimizer.optimize(examples=...)."""
        return [e.inputs for e in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __repr__(self) -> str:
        return f"Dataset({len(self.examples)} examples)"

    # ------------------------------------------------------------------
    # Evaluator factory
    # ------------------------------------------------------------------

    def make_evaluator(
        self,
        output_key: str,
        mode: str = "llm_judge",
        judge_model: str = None,
        input_cpm: float = 0.0,
        output_cpm: float = 0.0,
    ) -> Callable[[dict, dict], float]:
        """
        Return a scoring function (inputs, state) -> float compatible with PromptOptimizer.

        mode="exact_match": 1.0 if output matches expected (case-insensitive), else 0.0.
                            Best for classification tasks (Router, short answers).

        mode="llm_judge":   An LLM rates output quality against expected on 0.0–1.0.
                            Best for open-ended tasks (CoT answers, written replies).
        """
        expected_map = {
            json.dumps(e.inputs, sort_keys=True, default=str): e.expected
            for e in self.examples
        }

        def _get_output(state: dict) -> str:
            val = state.get(output_key, "")
            if isinstance(val, dict):
                val = val.get("answer", val)
            return str(val).strip()

        if mode == "exact_match":
            def exact_evaluator(inputs: dict, state: dict) -> float:
                key = json.dumps(inputs, sort_keys=True, default=str)
                expected = expected_map.get(key)
                if expected is None:
                    return 0.0
                return 1.0 if _get_output(state).lower() == str(expected).strip().lower() else 0.0
            return exact_evaluator

        elif mode == "llm_judge":
            if judge_model is None:
                raise ValueError("judge_model must be provided for mode='llm_judge'")
            judge_llm = LLM_API(
                model_name=judge_model,
                system_prompt=(
                    "You are a quality evaluator for AI outputs. "
                    "Compare the produced output to the expected output and rate similarity/quality "
                    "on a scale from 0.0 (completely wrong) to 1.0 (perfect). "
                    "Reply with ONLY a decimal number between 0.0 and 1.0."
                ),
                input_token_cpm=input_cpm,
                output_token_cpm=output_cpm,
            )

            def llm_evaluator(inputs: dict, state: dict) -> float:
                key = json.dumps(inputs, sort_keys=True, default=str)
                expected = expected_map.get(key)
                output = _get_output(state)
                prompt = (
                    f"Expected output:\n{expected}\n\n"
                    f"Produced output:\n{output}\n\n"
                    f"Score (0.0–1.0):"
                )
                verdict = judge_llm(prompt).strip()
                judge_llm.reset_history()
                try:
                    return max(0.0, min(1.0, float(verdict)))
                except (ValueError, TypeError):
                    return 0.0

            return llm_evaluator

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'exact_match' or 'llm_judge'.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        workflow: Workflow,
        output_key: str,
        evaluator: Callable[[dict, dict], float] = None,
    ) -> dict:
        """
        Run all examples through the workflow and return aggregate metrics.

        If evaluator is None, falls back to exact_match against self.expected.
        """
        if evaluator is None:
            evaluator = self.make_evaluator(output_key, mode="exact_match")

        scores = []
        for example in self.examples:
            state = workflow.run(example.inputs)
            scores.append(evaluator(example.inputs, state))

        return {
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "n_examples": len(scores),
            "n_passing": sum(1 for s in scores if s >= 0.7),
            "scores": scores,
        }


# ---------------------------------------------------------------------------
# WorkflowOptimizer — structural DAG optimization
# ---------------------------------------------------------------------------

class WorkflowOptimizer:
    """
    Optimizes a workflow's structure (not just prompts) by proposing and
    evaluating DAG-level changes: adding steps, removing steps, or rewiring
    dependencies.

    Run PromptOptimizer first to squeeze out prompt-level gains, then use
    WorkflowOptimizer when prompt optimization has plateaued.

    Usage:
        wopt = WorkflowOptimizer(
            workflow=workflow,
            evaluator=my_evaluator,
            optimizer_model="groq/llama-3.3-70b-versatile",
            input_cpm=0.59,
            output_cpm=0.79,
        )
        improved = wopt.optimize(
            examples=[...],
            checkpoint_dir="checkpoints/",
            max_rounds=3,
            score_threshold=0.7,
            experiment_dir="experiments/",
        )

    Supported change types (proposed by the LLM, validated before applying):
        add_step          — insert a new step using an existing operator
        remove_step       — remove a step (only if nothing depends on it)
        change_depends_on — rewire a step's dependencies
    """

    _PROPOSER_SYSTEM = (
        "You are an AI workflow architect. "
        "You receive a description of an AI pipeline and its failure patterns, "
        "and propose ONE structural change to improve performance. "
        "You must respond with ONLY a valid JSON object — no explanation, no code fences."
    )

    _CHANGE_SCHEMA = """
Propose exactly one of these change types as a JSON object:

1. Add a new step (uses an existing operator):
{
  "type": "add_step",
  "step": {
    "id": "<unique_step_id>",
    "operator": "<existing_operator_name>",
    "prompt": "<prompt template with {placeholder} syntax>",
    "output": "<unique_output_key>",
    "depends_on": ["<step_id>", ...],
    "condition": null
  },
  "rationale": "<why this helps>"
}

2. Remove a step (only if no other step depends on it):
{
  "type": "remove_step",
  "step_id": "<step_id>",
  "update_depends_on": {
    "<step_id_that_depended_on_removed>": ["<new_dep_1>", ...]
  },
  "rationale": "<why this helps>"
}

3. Change a step's dependencies:
{
  "type": "change_depends_on",
  "step_id": "<step_id>",
  "new_depends_on": ["<step_id>", ...],
  "rationale": "<why this helps>"
}
"""

    def __init__(
        self,
        workflow: Workflow,
        evaluator: Callable[[dict, dict], float],
        optimizer_model: str,
        input_cpm: float,
        output_cpm: float,
    ):
        self.workflow = workflow
        self.evaluator = evaluator
        self._proposer = LLM_API(
            model_name=optimizer_model,
            system_prompt=self._PROPOSER_SYSTEM,
            input_token_cpm=input_cpm,
            output_token_cpm=output_cpm,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def optimize(
        self,
        examples: list[dict],
        checkpoint_dir: str,
        max_rounds: int = 3,
        score_threshold: float = 0.7,
        experiment_dir: str = None,
    ) -> Workflow:
        os.makedirs(checkpoint_dir, exist_ok=True)

        baseline_path = os.path.join(checkpoint_dir, "wopt_v1.json")
        self.workflow.save_checkpoint(baseline_path, description="workflow-opt baseline")
        best_path = baseline_path

        log: _ExperimentLog | None = None
        if experiment_dir:
            os.makedirs(experiment_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log = _ExperimentLog(
                path=os.path.join(experiment_dir, f"wexp_{ts}.md"),
                workflow_name=self.workflow.name,
                baseline_checkpoint=baseline_path,
            )

        runs = self._run_all(examples)
        best_score = sum(r.score for r in runs) / len(runs)
        if log:
            log.set_baseline(best_score, len(runs))

        version = 1

        for round_num in range(1, max_rounds + 1):
            current_score = best_score
            failing = [r for r in runs if r.score < score_threshold]

            failure_summary = self._summarise_failures(failing)
            workflow_desc = self._describe_workflow()
            proposal_json = self._propose_change(workflow_desc, failure_summary)

            if proposal_json is None:
                break

            change_type = proposal_json.get("type", "")
            rationale = proposal_json.get("rationale", "")

            # Save config snapshot for rollback
            config_snapshot = copy.deepcopy(self.workflow._config)

            try:
                self._apply_change(proposal_json)
            except (ValueError, KeyError) as e:
                if log:
                    self._log_invalid(log, round_num, change_type, str(e), rationale)
                self.workflow._config = config_snapshot
                self._rebuild_workflow()
                continue

            new_runs = self._run_all(examples)
            new_score = sum(r.score for r in new_runs) / len(new_runs)
            failing_after = sum(1 for r in new_runs if r.score < score_threshold)
            accepted = new_score > current_score

            if accepted:
                version += 1
                ckpt_path = os.path.join(checkpoint_dir, f"wopt_v{version}.json")
                self.workflow.save_checkpoint(
                    ckpt_path,
                    description=f"wopt round {round_num}: {change_type}",
                    metrics={"avg_score": new_score, "n_eval_runs": len(examples)},
                )
                best_score = new_score
                best_path = ckpt_path
                runs = new_runs
            else:
                # Revert
                self.workflow._config = config_snapshot
                self._rebuild_workflow()
                ckpt_path = None

            if log:
                log.write_round(
                    round_num=round_num,
                    step_id=f"[{change_type}]",
                    issues=failure_summary,
                    old_prompt=workflow_desc,
                    new_prompt=f"Proposed: {json.dumps(proposal_json, indent=2)}",
                    score_before=current_score,
                    score_after=new_score,
                    failing_before=len(failing),
                    failing_after=failing_after,
                    n_total=len(examples),
                    accepted=accepted,
                    checkpoint_path=ckpt_path,
                )

        with open(os.path.join(checkpoint_dir, "current.txt"), "w") as f:
            f.write(best_path + "\n")

        if log:
            log.close(final_checkpoint=best_path)

        return self.workflow

    # ------------------------------------------------------------------
    # Workflow description and failure summarisation
    # ------------------------------------------------------------------

    def _describe_workflow(self) -> str:
        lines = [f"Workflow: '{self.workflow.name}'", "", "Operators:"]
        for name, op in self.workflow.operators.items():
            lines.append(f"  {name}: {type(op).__name__}")
        lines.append("", "Steps (in execution order):")
        for step in self.workflow.steps:
            op_type = type(self.workflow.operators[step.operator_id]).__name__
            line = (
                f"  {step.id} — operator: '{step.operator_id}' ({op_type}), "
                f"output: '{step.output_key}'"
            )
            if step.depends_on:
                line += f", depends_on: {step.depends_on}"
            if step.condition:
                line += f", condition: '{step.condition}'"
            lines.append(line)
        return "\n".join(lines)

    def _summarise_failures(self, failing: list[RunResult]) -> str:
        if not failing:
            return "No failing runs."
        samples = failing[:5]
        parts = [f"{len(failing)} failing run(s). Sample inputs and scores:"]
        for r in samples:
            parts.append(f"  score={r.score:.2f} | inputs={json.dumps(r.inputs, default=str)[:200]}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def _propose_change(self, workflow_desc: str, failure_summary: str) -> dict | None:
        prompt = (
            f"CURRENT WORKFLOW:\n{workflow_desc}\n\n"
            f"FAILURE PATTERNS:\n{failure_summary}\n\n"
            f"VALID CHANGE TYPES:\n{self._CHANGE_SCHEMA}\n\n"
            f"Propose ONE change that would most improve the workflow's performance."
        )
        raw = self._proposer(prompt)
        self._proposer.reset_history()
        try:
            # Strip markdown code fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Change application and validation
    # ------------------------------------------------------------------

    def _apply_change(self, proposal: dict):
        change_type = proposal["type"]
        config = self.workflow._config
        step_ids = {s["id"] for s in config["steps"]}
        op_ids = set(config.get("operators", {}))

        if change_type == "add_step":
            step_cfg = proposal["step"]
            if step_cfg["id"] in step_ids:
                raise ValueError(f"Step id '{step_cfg['id']}' already exists.")
            if step_cfg.get("output") in {s["output"] for s in config["steps"]}:
                raise ValueError(f"Output key '{step_cfg['output']}' already in use.")
            if step_cfg.get("operator") not in op_ids:
                raise ValueError(f"Operator '{step_cfg.get('operator')}' not defined.")
            for dep in step_cfg.get("depends_on", []):
                if dep not in step_ids:
                    raise ValueError(f"Dependency '{dep}' does not exist.")
            config["steps"].append(step_cfg)

        elif change_type == "remove_step":
            step_id = proposal["step_id"]
            if step_id not in step_ids:
                raise ValueError(f"Step '{step_id}' does not exist.")
            dependents = [s["id"] for s in config["steps"] if step_id in s.get("depends_on", [])]
            update_map: dict = proposal.get("update_depends_on", {})
            unresolved = [d for d in dependents if d not in update_map]
            if unresolved:
                raise ValueError(
                    f"Steps {unresolved} depend on '{step_id}' but no update_depends_on provided."
                )
            config["steps"] = [s for s in config["steps"] if s["id"] != step_id]
            for sid, new_deps in update_map.items():
                for s in config["steps"]:
                    if s["id"] == sid:
                        s["depends_on"] = new_deps

        elif change_type == "change_depends_on":
            step_id = proposal["step_id"]
            new_deps = proposal["new_depends_on"]
            if step_id not in step_ids:
                raise ValueError(f"Step '{step_id}' does not exist.")
            for dep in new_deps:
                if dep not in step_ids:
                    raise ValueError(f"Dependency '{dep}' does not exist.")
            for s in config["steps"]:
                if s["id"] == step_id:
                    s["depends_on"] = new_deps
                    break

        else:
            raise ValueError(f"Unknown change type '{change_type}'.")

        self._rebuild_workflow()
        # _rebuild raises if the new config is invalid (cycles, conflicts, etc.)

    def _rebuild_workflow(self):
        """Rebuild the Workflow in-place from the current _config."""
        from .workflow import Workflow as _Workflow
        rebuilt = _Workflow(self.workflow._config, self.workflow.tools)
        self.workflow.steps = rebuilt.steps
        self.workflow.operators = rebuilt.operators
        self.workflow._op_call_kwargs = rebuilt._op_call_kwargs
        self.workflow.state = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_all(self, examples: list[dict]) -> list[RunResult]:
        results = []
        for inputs in examples:
            state = self.workflow.run(inputs)
            score = self.evaluator(inputs, state)
            results.append(RunResult(inputs=inputs, state=state, score=score))
        return results

    def _log_invalid(
        self,
        log: _ExperimentLog,
        round_num: int,
        change_type: str,
        error: str,
        rationale: str,
    ):
        log.write_round(
            round_num=round_num,
            step_id=f"[{change_type}]",
            issues=f"Proposed change was invalid: {error}",
            old_prompt=self._describe_workflow(),
            new_prompt=f"Rationale: {rationale}",
            score_before=0.0,
            score_after=0.0,
            failing_before=0,
            failing_after=0,
            n_total=0,
            accepted=False,
        )