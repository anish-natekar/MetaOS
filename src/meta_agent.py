"""
MetaOS Agent — conversational system builder.

Orchestrates the full MetaOS stack: planning → tool synthesis → workflow creation
→ optimization → data generation. Maintains conversation history across turns so
the user can chat naturally to design, build, and improve AI agent pipelines.
"""

import json
import os
from datetime import datetime

from .llm_api import LLM_API, ResponseType
from .planner import Planner, TodoTree, TodoNode
from .tool_forge import ToolRegistry, ToolSynthesizer, ToolOptimizer
from .optimizer import PromptOptimizer, WorkflowOptimizer, Dataset
from .workflow import Workflow
from .data_synth import DataSynthesizer


# ---------------------------------------------------------------------------
# WorkflowRegistry
# ---------------------------------------------------------------------------

class WorkflowRegistry:
    """
    Persistent store for workflow configs.
    Mirrors the ToolRegistry pattern: {name}.json + _index.json in a directory.

    Usage:
        registry = WorkflowRegistry("workflows/")
        registry.save("support_classifier", config_dict, description="Route support messages", tags=["support"])
        config = registry.load("support_classifier")
        names = registry.search("support")
    """

    _INDEX = "_index.json"

    def __init__(self, workflows_dir: str = "workflows/"):
        self.workflows_dir = workflows_dir
        os.makedirs(workflows_dir, exist_ok=True)
        self._index: dict[str, dict] = self._load_index()

    def save(self, name: str, config: dict, description: str = "", tags: list[str] = None) -> None:
        """Persist a workflow config. Overwrites any existing entry for that name."""
        tags = tags or []
        entry = {
            "config": config,
            "description": description,
            "tags": tags,
            "updated_at": datetime.now().isoformat(),
        }
        with open(self._path(name), "w") as f:
            json.dump(entry, f, indent=2)
        self._index[name] = {"description": description, "tags": tags}
        self._save_index()

    def load(self, name: str) -> dict:
        """Load a workflow config dict by name. Raises KeyError if not found."""
        path = self._path(name)
        if not os.path.exists(path):
            raise KeyError(f"No workflow named '{name}' in registry at '{self.workflows_dir}'")
        with open(path) as f:
            return json.load(f)["config"]

    def has(self, name: str) -> bool:
        return name in self._index

    def search(self, query: str) -> list[str]:
        q = query.lower()
        return [
            name for name, meta in self._index.items()
            if q in " ".join([name, meta.get("description", ""), *meta.get("tags", [])]).lower()
        ]

    def list_workflows(self) -> list[dict]:
        return [{"name": k, **v} for k, v in self._index.items()]

    def _path(self, name: str) -> str:
        return os.path.join(self.workflows_dir, f"{name}.json")

    def _load_index(self) -> dict:
        path = os.path.join(self.workflows_dir, self._INDEX)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(os.path.join(self.workflows_dir, self._INDEX), "w") as f:
            json.dump(self._index, f, indent=2)


# ---------------------------------------------------------------------------
# WorkflowBuilder — LLM-generates workflow JSON configs
# ---------------------------------------------------------------------------

# Embedded docs loaded once at import time so they're in the module scope
_OPERATORS_DOC = """\
AVAILABLE OPERATOR TYPES:
- predict: single LLM call; output = string
- cot: reason first then answer; output = {reasoning: str, answer: str}
- react: tool-use loop; output = string. Requires "tools": [name, ...] and optional "max_iterations"
- router: classify into categories; output = one route string. Step needs "args": {"routes": [...]}

OPERATOR FIELDS (all operators):
  type, model, input_cpm, output_cpm, n (optional, default 1), system_prompt (optional)

STEP FIELDS:
  id (unique), operator (name from operators block), prompt (use {key} for state refs),
  output (key to store result), depends_on (list of step ids), condition (optional expression),
  args (extra kwargs like routes or max_iterations)

OUTPUT FORMAT:
  cot steps: state[output_key] = {"reasoning": "...", "answer": "..."}
  all other steps: state[output_key] = the string result
  To reference a cot step's answer in a later prompt: {step_output[answer]}

RULES:
- Each step MUST have a unique output key
- Use depends_on to express ordering; independent steps run in parallel
- Only reference state keys that exist in inputs or prior step outputs
- For router steps, include "args": {"routes": [...]} at the step level
- For react steps, list tool names under operator config "tools": [...]
"""


class WorkflowBuilder:
    """
    Generates valid workflow JSON configs for a given goal using LLM.
    Validates by constructing a Workflow instance; retries on validation error.
    """

    _SYSTEM = (
        "You are a MetaOS workflow architect. Your job is to design workflow JSON configurations "
        "that chain agentic operators into a pipeline to accomplish a goal.\n\n"
        + _OPERATORS_DOC +
        "\nReturn ONLY valid JSON matching the workflow format. No prose outside the JSON."
    )

    def __init__(self, model: str, input_cpm: float, output_cpm: float, max_attempts: int = 3):
        self._llm = LLM_API(
            model_name=model,
            system_prompt=self._SYSTEM,
            input_token_cpm=input_cpm,
            output_token_cpm=output_cpm,
        )
        self.max_attempts = max_attempts
        self._model = model
        self._input_cpm = input_cpm
        self._output_cpm = output_cpm

    def build(
        self,
        workflow_name: str,
        goal: str,
        input_description: str,
        available_tools: list[str] = None,
        error_feedback: str = None,
    ) -> dict:
        """
        Generate a workflow config dict for the given goal.
        Returns a validated config dict ready to pass to WorkflowRegistry.save() or Workflow().

        Raises RuntimeError if validation fails after max_attempts.
        """
        available_tools = available_tools or []
        last_error = error_feedback

        for attempt in range(1, self.max_attempts + 1):
            config = self._generate(workflow_name, goal, input_description, available_tools, last_error)
            validation_error = self._validate(config)
            if validation_error is None:
                self._llm.reset_history()
                return config
            last_error = validation_error

        self._llm.reset_history()
        raise RuntimeError(
            f"WorkflowBuilder failed to generate a valid config for '{goal}' "
            f"after {self.max_attempts} attempts. Last error: {last_error}"
        )

    def _generate(
        self,
        name: str,
        goal: str,
        input_description: str,
        available_tools: list[str],
        error_feedback: str | None,
    ) -> dict:
        parts = [
            f"Design a workflow named '{name}' that accomplishes this goal:\n{goal}",
            f"\nInput fields available: {input_description}",
        ]
        if available_tools:
            parts.append(f"Available tools (use in react operator 'tools' list): {available_tools}")
        parts.append(
            f"\nDefault model to use: {self._model!r}, input_cpm: {self._input_cpm}, output_cpm: {self._output_cpm}"
        )
        if error_feedback:
            parts.append(f"\nPrevious attempt was invalid — fix this error:\n{error_feedback}")
        parts.append("\nReturn ONLY the workflow JSON object.")

        raw = self._llm("\n".join(parts), response_type=ResponseType.TEXT)

        # Strip markdown code fences
        import re
        raw = re.sub(r'^```(?:json)?\s*\n', '', raw.strip())
        raw = re.sub(r'\n```\s*$', '', raw)

        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw:\n{raw[:500]}")

    def _validate(self, config: dict) -> str | None:
        """Try to construct a Workflow from the config. Return error string or None."""
        try:
            Workflow(config)
            return None
        except Exception as e:
            return str(e)


# ---------------------------------------------------------------------------
# MetaOSAgent — the main conversational orchestrator
# ---------------------------------------------------------------------------

_AGENT_SYSTEM = """\
You are MetaOS, an AI system architect. You help users design, build, optimize, and maintain AI agent pipelines through natural conversation.

YOUR WORKFLOW (follow this order):
1. PLAN — call plan_goal() to decompose the user's goal into a structured planning tree
2. SHOW — call show_tree() to display the plan; wait for user approval before building anything
3. BUILD (bottom-up, starting from leaf nodes):
   a. For each leaf tool node → call build_tool() to synthesize and test it
   b. For each workflow node → call build_workflow() to create the config, then save_workflow()
   c. Combine workflows if needed (create a parent workflow that orchestrates sub-workflows)
4. TEST — call generate_data() or web_search_and_generate() to create labeled test data
5. OPTIMIZE — call optimize_node() per node with the dataset to improve quality

RULES:
- Always call search_tools() and search_workflows() BEFORE creating anything new
- Always show the plan tree and wait for user confirmation before starting to build
- After building each tool, show the user whether it passed its test cases
- After generating a workflow config, show it to the user before saving
- If tool synthesis fails after retries, report the error and ask the user what to do
- If optimization plateaus, suggest trying a different optimizer type or adding more data
- Reuse existing tools and workflows whenever possible
- Track everything: use attach_tool() and attach_workflow() to link artifacts to tree nodes

""" + _OPERATORS_DOC + """

WORKFLOW JSON FORMAT:
{
  "name": "workflow_name",
  "operators": {
    "op_name": {"type": "predict|cot|react|router", "model": "...", "input_cpm": N, "output_cpm": N}
  },
  "steps": [
    {"id": "step1", "operator": "op_name", "prompt": "Do {something} with {input}", "output": "result_key"}
  ]
}
"""


class MetaOSAgent:
    """
    Conversational MetaOS system builder. Chat to design and build AI pipelines.

    Maintains full conversation history across chat() calls. All tools (plan_goal,
    build_tool, build_workflow, etc.) are available to the underlying LLM as callable
    functions. The LLM decides when to call them based on the conversation context.

    Usage:
        agent = MetaOSAgent("groq/llama-3.3-70b-versatile", 0.59, 0.79)

        print(agent.chat("Build me a customer support pipeline"))
        # → agent plans, shows tree, waits for approval

        print(agent.chat("Looks good, start building"))
        # → agent synthesizes tools, builds workflows, shows progress

        print(agent.chat("Generate test data and optimize"))
        # → agent generates data, runs optimization, reports results

        agent.reset_session()   # start a new goal, keep registry
    """

    def __init__(
        self,
        model: str,
        input_cpm: float,
        output_cpm: float,
        tool_registry_dir: str = "tools/",
        workflow_registry_dir: str = "workflows/",
        pytest_dir: str = "tests/tools/",
        plan_dir: str = "plan/",
        on_tool_call: callable = None,
    ):
        self.model = model
        self.input_cpm = input_cpm
        self.output_cpm = output_cpm
        self.pytest_dir = pytest_dir
        self.plan_dir = plan_dir

        self.on_tool_call = on_tool_call

        # Sub-components
        self.tool_registry = ToolRegistry(tool_registry_dir)
        self.workflow_registry = WorkflowRegistry(workflow_registry_dir)
        self.planner = Planner(model, input_cpm, output_cpm)
        self.tool_synth = ToolSynthesizer(model, input_cpm, output_cpm, self.tool_registry)
        self.tool_opt = ToolOptimizer(model, input_cpm, output_cpm, self.tool_registry)
        self.data_synth = DataSynthesizer(model, input_cpm, output_cpm)
        self.workflow_builder = WorkflowBuilder(model, input_cpm, output_cpm)

        # Session state
        self.tree: TodoTree | None = None
        self.datasets: dict[str, Dataset] = {}
        self._tree_path: str = os.path.join(plan_dir, "todo_tree.json")

        # Persistent LLM — history is NEVER reset between chat() calls
        self._llm = LLM_API(
            model_name=model,
            system_prompt=_AGENT_SYSTEM,
            input_token_cpm=input_cpm,
            output_token_cpm=output_cpm,
            keep_history=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, message: str) -> str:
        """
        Send a message. History is maintained across all chat() calls in this session.
        The agent may call tools internally before returning its response.
        """
        tools = self._get_tools()
        result = self._llm(message, tools=tools)

        # Manual React loop — keep going until the LLM returns a string response
        max_tool_rounds = 20
        for _ in range(max_tool_rounds):
            if isinstance(result, str):
                break
            result = self._llm("Continue based on the tool results above.", tools=tools)

        return result if isinstance(result, str) else str(result)

    def reset_session(self) -> None:
        """Clear current tree and conversation history. Tool/workflow registries are preserved."""
        self.tree = None
        self.datasets = {}
        self._llm.reset_history()

    @property
    def metrics(self) -> dict:
        return {
            "total_cost": self._llm.metrics.input_cost + self._llm.metrics.output_cost,
            "total_calls": self._llm.metrics.total_number_of_calls,
            "total_tool_calls": self._llm.metrics.total_tool_calls,
        }

    # ------------------------------------------------------------------
    # Tool methods (called by the LLM via tool calling)
    # ------------------------------------------------------------------

    def plan_goal(self, goal: str, context: str = "") -> str:
        """Decompose a goal into a structured planning tree (TodoTree).
        goal: the high-level objective to achieve.
        context: optional description of available tools or constraints.
        Returns the tree as indented text."""
        self.tree = self.planner.plan(goal, context=context)
        os.makedirs(self.plan_dir, exist_ok=True)
        self.tree.save(self._tree_path)
        return f"Planning tree created:\n\n{self.tree.get_subtree_text()}"

    def show_tree(self, node_id: str = "") -> str:
        """Show the current planning tree (or a subtree starting at node_id).
        node_id: optional node to start from (empty string = root)."""
        if self.tree is None:
            return "No planning tree yet. Call plan_goal() first."
        nid = node_id.strip() or None
        return self.tree.get_subtree_text(nid)

    def get_node_status(self) -> str:
        """Show a compact summary of all nodes and their current status."""
        if self.tree is None:
            return "No planning tree yet."
        return self.tree.summary()

    def build_tool(
        self,
        node_id: str,
        description: str,
        signature: str,
        test_cases_json: str,
    ) -> str:
        """Synthesize and sandbox-test a Python tool function for a leaf node.
        node_id: TodoTree node ID this tool implements.
        description: one-sentence description of what the function should do.
        signature: Python signature, e.g. '(message: str) -> str'.
        test_cases_json: JSON array of test cases, e.g. '[{"args": {"x": 1}, "expected": 2}]'.
        Returns pass/fail status and the generated source."""
        try:
            test_cases = json.loads(test_cases_json) if isinstance(test_cases_json, str) else test_cases_json
        except json.JSONDecodeError as e:
            return f"Invalid test_cases_json: {e}"

        try:
            record = self.tool_synth.synthesize(
                description=description,
                signature=signature,
                test_cases=test_cases,
                pytest_dir=self.pytest_dir,
            )
            if self.tree and node_id and node_id in self.tree.nodes:
                self.tree.nodes[node_id].implementation_note = f"tool:{record.name}"
                self.tree.nodes[node_id].is_leaf = True
                self.tree.mark_status(node_id, "done")
                self.tree.save(self._tree_path)
            return (
                f"Tool '{record.name}' created successfully.\n"
                f"Version: {record.version} | Tests: {len(test_cases)} passed\n"
                f"Source:\n```python\n{record.source}\n```"
                + (f"\nPytest file: {self.pytest_dir}/test_{record.name}.py" if self.pytest_dir else "")
            )
        except RuntimeError as e:
            if self.tree and node_id and node_id in self.tree.nodes:
                self.tree.mark_failed(node_id)
                self.tree.save(self._tree_path)
            return f"Tool synthesis failed: {e}"

    def build_workflow(
        self,
        node_id: str,
        workflow_name: str,
        goal: str,
        input_description: str,
    ) -> str:
        """Generate a workflow JSON config for a node.
        node_id: TodoTree node ID this workflow implements.
        workflow_name: snake_case name for this workflow.
        goal: what this workflow should accomplish.
        input_description: comma-separated list of input field names, e.g. 'message, customer_id'.
        Returns the workflow JSON for review. Call save_workflow() to persist it."""
        available_tools = self.tool_registry.search("")  # all tools
        available_workflows = [w["name"] for w in self.workflow_registry.list_workflows()]

        try:
            config = self.workflow_builder.build(
                workflow_name=workflow_name,
                goal=goal,
                input_description=input_description,
                available_tools=available_tools,
            )
            # Store temporarily in session so save_workflow() can access it
            self._pending_workflow = (node_id, workflow_name, config)
            return (
                f"Workflow '{workflow_name}' generated. Review the config below, "
                f"then call save_workflow('{workflow_name}') to persist it.\n\n"
                f"```json\n{json.dumps(config, indent=2)}\n```"
            )
        except RuntimeError as e:
            return f"Workflow generation failed: {e}"

    def save_workflow(self, workflow_name: str, tags_json: str = "[]") -> str:
        """Save the most recently built workflow to the workflow registry.
        workflow_name: must match the name used in build_workflow().
        tags_json: optional JSON array of tag strings, e.g. '["support", "classifier"]'."""
        if not hasattr(self, "_pending_workflow") or self._pending_workflow[1] != workflow_name:
            return f"No pending workflow named '{workflow_name}'. Call build_workflow() first."

        node_id, name, config = self._pending_workflow
        try:
            tags = json.loads(tags_json) if isinstance(tags_json, str) else tags_json
        except json.JSONDecodeError:
            tags = []

        # Extract description from workflow config or goal
        description = config.get("name", name)

        self.workflow_registry.save(name, config, description=description, tags=tags)

        # Link to tree node
        if self.tree and node_id and node_id in self.tree.nodes:
            wf_path = os.path.join(self.workflow_registry.workflows_dir, f"{name}.json")
            self.tree.attach_workflow(node_id, wf_path)
            self.tree.mark_status(node_id, "done")
            self.tree.save(self._tree_path)

        del self._pending_workflow
        return f"Workflow '{name}' saved to registry."

    def generate_data(
        self,
        dataset_name: str,
        goal: str,
        input_keys_json: str,
        output_description: str,
        n_examples: int = 10,
        context: str = "",
    ) -> str:
        """Generate synthetic labeled test data for a goal.
        dataset_name: key to store this dataset (used in optimize_node).
        goal: what the workflow is supposed to accomplish.
        input_keys_json: JSON array of input field names, e.g. '["message"]'.
        output_description: what the expected output looks like, e.g. 'billing, technical, or general'.
        n_examples: number of examples to generate.
        context: optional extra context (domain docs, examples, etc.).
        Returns a sample of the generated examples."""
        try:
            input_keys = json.loads(input_keys_json) if isinstance(input_keys_json, str) else input_keys_json
        except json.JSONDecodeError:
            input_keys = [input_keys_json]

        dataset = self.data_synth.generate(
            goal=goal,
            input_keys=input_keys,
            output_description=output_description,
            n_examples=n_examples,
            context=context,
        )
        self.datasets[dataset_name] = dataset

        sample = dataset.inputs[:3]
        return (
            f"Dataset '{dataset_name}' created with {len(dataset.inputs)} examples.\n"
            f"Sample (first 3):\n{json.dumps(sample, indent=2, default=str)}"
        )

    def web_search_and_generate(
        self,
        dataset_name: str,
        web_query: str,
        goal: str,
        input_keys_json: str,
        output_description: str,
        n_examples: int = 10,
    ) -> str:
        """Search the web for context and generate synthetic test data from it.
        dataset_name: key to store this dataset.
        web_query: what to search for on the web.
        goal: what the workflow is supposed to accomplish.
        input_keys_json: JSON array of input field names.
        output_description: what the expected output looks like.
        n_examples: number of examples to generate."""
        try:
            input_keys = json.loads(input_keys_json) if isinstance(input_keys_json, str) else input_keys_json
        except json.JSONDecodeError:
            input_keys = [input_keys_json]

        dataset = self.data_synth.generate_from_web(
            web_query=web_query,
            goal=goal,
            input_keys=input_keys,
            output_description=output_description,
            n_examples=n_examples,
        )
        self.datasets[dataset_name] = dataset

        sample = dataset.inputs[:3]
        return (
            f"Dataset '{dataset_name}' created from web context with {len(dataset.inputs)} examples.\n"
            f"Sample (first 3):\n{json.dumps(sample, indent=2, default=str)}"
        )

    def optimize_node(
        self,
        node_id: str,
        workflow_name: str,
        dataset_name: str,
        output_key: str,
        optimizer_type: str = "prompt",
        max_rounds: int = 3,
    ) -> str:
        """Run an optimizer on a workflow node using a named dataset.
        node_id: TodoTree node ID (for tree updates).
        workflow_name: name of the workflow in WorkflowRegistry to optimize.
        dataset_name: name of a dataset created by generate_data().
        output_key: which workflow output field to evaluate.
        optimizer_type: 'prompt' or 'workflow' (default: 'prompt').
        max_rounds: optimization rounds (default: 3)."""
        if dataset_name not in self.datasets:
            return f"Dataset '{dataset_name}' not found. Call generate_data() first."
        if not self.workflow_registry.has(workflow_name):
            return f"Workflow '{workflow_name}' not in registry."

        dataset = self.datasets[dataset_name]
        config = self.workflow_registry.load(workflow_name)

        # Resolve tools from registry for any react operators
        workflow = Workflow(config, registry=self.tool_registry)

        evaluator = dataset.make_evaluator(output_key=output_key, mode="llm_judge",
                                           judge_model=self.model,
                                           input_cpm=self.input_cpm,
                                           output_cpm=self.output_cpm)

        checkpoint_dir = f"checkpoints/{workflow_name}"
        experiment_dir = f"experiments/{workflow_name}"
        tree_path = self._tree_path if self.tree else None

        if optimizer_type == "workflow":
            opt = WorkflowOptimizer(workflow, evaluator, self.model, self.input_cpm, self.output_cpm)
            improved = opt.optimize(
                dataset.inputs, checkpoint_dir, max_rounds=max_rounds,
                experiment_dir=experiment_dir,
                todo_tree=self.tree, tree_node_id=node_id if self.tree else None,
                tree_path=tree_path,
            )
        else:
            opt = PromptOptimizer(workflow, evaluator, self.model, self.input_cpm, self.output_cpm)
            improved = opt.optimize(
                dataset.inputs, checkpoint_dir, max_rounds=max_rounds,
                experiment_dir=experiment_dir,
                todo_tree=self.tree, tree_node_id=node_id if self.tree else None,
                tree_path=tree_path,
            )

        # Save the optimized workflow back to registry
        self.workflow_registry.save(workflow_name, improved._config,
                                    description=f"optimized ({optimizer_type})")

        return (
            f"Optimization complete for '{workflow_name}' ({optimizer_type} optimizer, {max_rounds} rounds).\n"
            f"Checkpoints: {checkpoint_dir}/\n"
            f"Experiment log: {experiment_dir}/"
        )

    def optimize_tool(
        self,
        tool_name: str,
        test_cases_json: str,
        max_rounds: int = 3,
    ) -> str:
        """Improve an existing tool's implementation against updated test cases.
        tool_name: name of a tool in the ToolRegistry.
        test_cases_json: JSON array of test cases to pass.
        max_rounds: optimization rounds."""
        if not self.tool_registry.has(tool_name):
            return f"Tool '{tool_name}' not found in registry."
        try:
            test_cases = json.loads(test_cases_json) if isinstance(test_cases_json, str) else test_cases_json
        except json.JSONDecodeError as e:
            return f"Invalid test_cases_json: {e}"

        record = self.tool_opt.optimize(tool_name, test_cases, max_rounds=max_rounds)
        return (
            f"Tool '{tool_name}' optimized to version {record.version}.\n"
            f"Source:\n```python\n{record.source}\n```"
        )

    def search_tools(self, query: str) -> str:
        """Search the tool registry for existing tools matching a query.
        query: keyword(s) to search by name, description, or tags."""
        names = self.tool_registry.search(query)
        if not names:
            return f"No tools found matching '{query}'."
        tools = self.tool_registry.list_tools()
        matches = [t for t in tools if t["name"] in names]
        lines = [f"Found {len(matches)} tool(s) matching '{query}':"]
        for t in matches:
            lines.append(f"  {t['name']}{t['signature']} — {t['description']}")
        return "\n".join(lines)

    def search_workflows(self, query: str) -> str:
        """Search the workflow registry for existing workflows matching a query.
        query: keyword(s) to search by name, description, or tags."""
        names = self.workflow_registry.search(query)
        if not names:
            return f"No workflows found matching '{query}'."
        workflows = self.workflow_registry.list_workflows()
        matches = [w for w in workflows if w["name"] in names]
        lines = [f"Found {len(matches)} workflow(s) matching '{query}':"]
        for w in matches:
            lines.append(f"  {w['name']} — {w.get('description', '')}")
        return "\n".join(lines)

    def list_tools(self) -> str:
        """List all tools currently available in the tool registry."""
        tools = self.tool_registry.list_tools()
        if not tools:
            return "No tools in registry yet."
        lines = [f"Registry contains {len(tools)} tool(s):"]
        for t in tools:
            lines.append(f"  {t['name']}{t['signature']} v{t['version']} — {t['description']}")
        return "\n".join(lines)

    def list_workflows(self) -> str:
        """List all workflows currently saved in the workflow registry."""
        workflows = self.workflow_registry.list_workflows()
        if not workflows:
            return "No workflows in registry yet."
        lines = [f"Registry contains {len(workflows)} workflow(s):"]
        for w in workflows:
            lines.append(f"  {w['name']} — {w.get('description', '')}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_tools(self) -> list[callable]:
        import functools
        raw = [
            self.plan_goal,
            self.show_tree,
            self.get_node_status,
            self.build_tool,
            self.build_workflow,
            self.save_workflow,
            self.generate_data,
            self.web_search_and_generate,
            self.optimize_node,
            self.optimize_tool,
            self.search_tools,
            self.search_workflows,
            self.list_tools,
            self.list_workflows,
        ]
        if self.on_tool_call is None:
            return raw

        def _wrap(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                self.on_tool_call(fn.__name__)
                return fn(*args, **kwargs)
            return wrapper

        return [_wrap(t) for t in raw]
