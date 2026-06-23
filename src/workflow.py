import json
import os
import re
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel
from .agent import Agentic_Operator, CoT, React, Router, Predict


OPERATOR_REGISTRY: dict[str, type[Agentic_Operator]] = {
    "cot":     CoT,
    "react":   React,
    "router":  Router,
    "predict": Predict,
}


@dataclass
class WorkflowStep:
    id: str
    operator_id: str
    prompt_template: str
    output_key: str
    args: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    condition: str | None = None


class WorkflowState:
    def __init__(self, inputs: dict):
        self._data: dict = dict(inputs)

    def set(self, key: str, value):
        if key in self._data:
            raise ValueError(
                f"State key '{key}' is already set and cannot be overwritten. "
                f"Each step must write to a unique output key."
            )
        self._data[key] = value

    def resolve(self, template: str) -> str:
        try:
            return template.format_map(self._data)
        except KeyError as e:
            raise KeyError(
                f"Template references undefined key {e}. "
                f"Available keys: {sorted(self._data)}"
            )

    def eval_condition(self, expr: str) -> bool:
        def replacer(m):
            key_expr = m.group(1)
            if "[" in key_expr:
                base, sub = key_expr.rstrip("]").split("[", 1)
                val = self._data[base][sub]
            else:
                val = self._data[key_expr]
            return repr(str(val))
        resolved = re.sub(r'\{([^}]+)\}', replacer, expr)
        try:
            return bool(eval(resolved, {"__builtins__": {}}, {}))
        except Exception as e:
            raise ValueError(f"Cannot evaluate condition '{expr}' (resolved: '{resolved}'): {e}")

    @property
    def snapshot(self) -> dict:
        return dict(self._data)


class Workflow:
    def __init__(self, config: dict, tools: dict[str, callable] = None, base_dir: str = ".", registry=None):
        self.name = config.get("name", "workflow")
        self.tools = tools or {}
        self.registry = registry
        self._config = config
        self.base_dir = base_dir
        self.steps = [self._parse_step(s) for s in config["steps"]]
        self.operators, self._op_call_kwargs = self._build_operators(config.get("operators", {}))
        self._validate()
        self.state: dict = {}

    @classmethod
    def from_json(cls, path: str, tools: dict[str, callable] = None, registry=None) -> "Workflow":
        with open(path) as f:
            return cls(json.load(f), tools, base_dir=os.path.dirname(os.path.abspath(path)), registry=registry)

    @classmethod
    def from_checkpoint(cls, path: str, tools: dict[str, callable] = None, registry=None) -> "Workflow":
        with open(path) as f:
            data = json.load(f)
        config = data["workflow"]
        prompts = data.get("prompts", {})
        for step in config.get("steps", []):
            if step["id"] in prompts:
                step["prompt"] = prompts[step["id"]]
                step.pop("prompt_file", None)
        return cls(config, tools, registry=registry)

    def _parse_step(self, cfg: dict) -> WorkflowStep:
        if "prompt_file" in cfg:
            path = os.path.join(self.base_dir, cfg["prompt_file"])
            with open(path) as f:
                prompt_template = f.read()
        else:
            prompt_template = cfg.get("prompt", "")
        return WorkflowStep(
            id=cfg["id"],
            operator_id=cfg["operator"],
            prompt_template=prompt_template,
            output_key=cfg["output"],
            args=cfg.get("args", {}),
            depends_on=cfg.get("depends_on", []),
            condition=cfg.get("condition"),
        )

    def _build_operators(self, ops_cfg: dict) -> tuple[dict, dict]:
        operators: dict[str, Agentic_Operator] = {}
        call_kwargs: dict[str, dict] = {}

        for name, cfg in ops_cfg.items():
            op_type = cfg["type"]
            if op_type not in OPERATOR_REGISTRY:
                raise ValueError(f"Unknown operator type '{op_type}'. Available: {list(OPERATOR_REGISTRY)}")

            init_kw = {}
            if "system_prompt" in cfg:
                init_kw["system_prompt"] = cfg["system_prompt"]

            operators[name] = OPERATOR_REGISTRY[op_type](
                name=name,
                model_name=cfg["model"],
                input_token_cpm=cfg["input_cpm"],
                output_token_cpm=cfg["output_cpm"],
                n=cfg.get("n", 1),
                **init_kw
            )

            runtime_kw = {}
            if "tools" in cfg:
                resolved: list[callable] = []
                missing: list[str] = []
                for t in cfg["tools"]:
                    if t in self.tools:
                        resolved.append(self.tools[t])
                    elif self.registry is not None and self.registry.has(t):
                        fn = self.registry.load_as_callable(t)
                        self.tools[t] = fn  # cache so subsequent lookups are O(1)
                        resolved.append(fn)
                    else:
                        missing.append(t)
                if missing:
                    raise ValueError(
                        f"Operator '{name}' references unregistered tools: {missing}. "
                        "Pass these callables via the tools= param or register them in a ToolRegistry."
                    )
                runtime_kw["tools"] = resolved
            if "max_iterations" in cfg:
                runtime_kw["max_iterations"] = cfg["max_iterations"]
            call_kwargs[name] = runtime_kw

        return operators, call_kwargs

    def _validate(self):
        step_ids = {s.id for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in step_ids:
                    raise ValueError(f"Step '{s.id}' depends on unknown step '{dep}'")
            if s.operator_id not in self.operators:
                raise ValueError(f"Step '{s.id}' references unknown operator '{s.operator_id}'")
        self._check_output_key_conflicts()
        self._check_cycles()
        self._check_parallel_stateful_conflicts()

    def _check_output_key_conflicts(self):
        seen = {}
        for step in self.steps:
            if step.output_key in seen:
                raise ValueError(
                    f"Steps '{seen[step.output_key]}' and '{step.id}' both write to "
                    f"output key '{step.output_key}'. Each step must use a unique output key."
                )
            seen[step.output_key] = step.id

    def _check_cycles(self):
        steps_by_id = {s.id: s for s in self.steps}
        visited, in_stack = set(), set()

        def dfs(sid):
            visited.add(sid)
            in_stack.add(sid)
            for dep in steps_by_id[sid].depends_on:
                if dep not in visited:
                    dfs(dep)
                elif dep in in_stack:
                    raise ValueError(f"Circular dependency detected involving step '{dep}'")
            in_stack.remove(sid)

        for step in self.steps:
            if step.id not in visited:
                dfs(step.id)

    def _all_deps(self, step_id: str) -> set[str]:
        steps_by_id = {s.id: s for s in self.steps}
        deps, queue = set(), list(steps_by_id[step_id].depends_on)
        while queue:
            d = queue.pop()
            if d not in deps:
                deps.add(d)
                queue.extend(steps_by_id[d].depends_on)
        return deps

    def _check_parallel_stateful_conflicts(self):
        stateful = (CoT, React)
        for i, a in enumerate(self.steps):
            for b in self.steps[i + 1:]:
                if a.operator_id != b.operator_id:
                    continue
                if not isinstance(self.operators[a.operator_id], stateful):
                    continue
                if b.id not in self._all_deps(a.id) and a.id not in self._all_deps(b.id):
                    raise ValueError(
                        f"Steps '{a.id}' and '{b.id}' both use stateful operator "
                        f"'{a.operator_id}' ({type(self.operators[a.operator_id]).__name__}) "
                        f"and may run in parallel. Define separate operator entries for each."
                    )

    def _normalize(self, result) -> any:
        if isinstance(result, tuple) and len(result) == 2:
            reasoning, answer = result
            return {
                "reasoning": reasoning,
                "answer": answer.model_dump() if isinstance(answer, BaseModel) else answer,
            }
        if isinstance(result, BaseModel):
            return result.model_dump()
        return result

    def _run_step(self, step: WorkflowStep, state: WorkflowState):
        if step.condition and not state.eval_condition(step.condition):
            return None
        prompt = state.resolve(step.prompt_template)
        kwargs = {**self._op_call_kwargs.get(step.operator_id, {}), **step.args}
        return self.operators[step.operator_id](prompt, **kwargs)

    async def _run_async(self, inputs: dict) -> dict:
        state = WorkflowState(inputs)
        completed: set[str] = set()
        all_ids = {s.id for s in self.steps}

        while completed != all_ids:
            ready = [
                s for s in self.steps
                if s.id not in completed
                and all(dep in completed for dep in s.depends_on)
            ]
            if not ready:
                raise RuntimeError(
                    "Workflow deadlock — no steps ready but not all complete. "
                    "Check for circular dependencies."
                )
            results = await asyncio.gather(
                *[asyncio.to_thread(self._run_step, s, state) for s in ready]
            )
            for step, result in zip(ready, results):
                state.set(step.output_key, self._normalize(result))
                completed.add(step.id)

        self.state = state._data
        return dict(state._data)

    def _write_run_log(self, inputs: dict, state: dict, log_dir: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(log_dir, ts)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "inputs.json"), "w") as f:
            json.dump(inputs, f, indent=2, default=str)
        with open(os.path.join(run_dir, "state.json"), "w") as f:
            json.dump(state, f, indent=2, default=str)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

    def run(self, inputs: dict, log_dir: str = None) -> dict:
        conflicts = set(inputs) & {s.output_key for s in self.steps}
        if conflicts:
            raise ValueError(
                f"Input keys conflict with step output keys: {conflicts}. "
                f"Rename either the inputs or the step outputs."
            )
        result = asyncio.run(self._run_async(inputs))
        if log_dir:
            self._write_run_log(inputs, result, log_dir)
        return result

    async def arun(self, inputs: dict, log_dir: str = None) -> dict:
        result = await self._run_async(inputs)
        if log_dir:
            self._write_run_log(inputs, result, log_dir)
        return result

    def save_checkpoint(self, path: str, description: str = "", metrics: dict = None):
        checkpoint = {
            "version": os.path.splitext(os.path.basename(path))[0],
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "workflow": self._config,
            "prompts": {s.id: s.prompt_template for s in self.steps},
        }
        if metrics is not None:
            checkpoint["metrics"] = metrics
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    @property
    def metrics(self) -> dict:
        totals = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "total_api_time": 0.0,
            "total_calls": 0,
            "total_tool_calls": 0,
            "total_retries": 0,
            "per_operator": {},
        }
        for name, op in self.operators.items():
            m = op.metrics
            totals["total_input_tokens"] += m.total_input_tokens
            totals["total_output_tokens"] += m.total_output_tokens
            totals["total_cost"] += m.input_cost + m.output_cost
            totals["total_api_time"] += m.total_api_time
            totals["total_calls"] += m.total_number_of_calls
            totals["total_tool_calls"] += m.total_tool_calls
            totals["total_retries"] += m.total_retries
            totals["per_operator"][name] = {
                "input_tokens": m.total_input_tokens,
                "output_tokens": m.total_output_tokens,
                "cost": m.input_cost + m.output_cost,
                "calls": m.total_number_of_calls,
            }
        return totals