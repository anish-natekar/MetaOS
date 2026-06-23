import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Literal

from pydantic import BaseModel

from .llm_api import LLM_API, ResponseType


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------

class SubgoalSchema(BaseModel):
    name: str
    description: str          # ≤3 sentences, abstracted solution
    why: str                  # ≤3 sentences, what problem this addresses
    is_leaf: bool             # True if directly implementable by a single tool/workflow
    implementation_note: str | None = None  # leaf only: what solves this


class DecompositionSchema(BaseModel):
    subgoals: list[SubgoalSchema]


# ---------------------------------------------------------------------------
# TodoNode
# ---------------------------------------------------------------------------

@dataclass
class TodoNode:
    id: str
    name: str
    description: str                  # ≤3 sentences
    why: str                          # ≤3 sentences
    parent_id: str | None
    children_ids: list[str] = field(default_factory=list)
    status: str = "pending"           # "pending" | "in_progress" | "done" | "failed"
    is_leaf: bool = False
    implementation_note: str | None = None
    workflow_ref: str | None = None   # path to linked workflow JSON
    experiment_refs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TodoTree
# ---------------------------------------------------------------------------

class TodoTree:
    """
    Hierarchical decomposition of a goal into subgoals, down to leaf nodes
    that are directly implementable. Serves as:
      - Planning artifact: why and what at every level
      - Optimizer memory: failed branches are kept and tagged to avoid repeating mistakes
      - Audit trail: every experiment is linked to the node it was testing

    Storage: a single flat JSON file (plan/todo_tree.json by default).
    """

    def __init__(self, nodes: dict[str, TodoNode], root_id: str):
        self.nodes = nodes
        self.root_id = root_id

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def new(cls, goal_name: str, description: str) -> "TodoTree":
        """Create a tree with a single root node."""
        root_id = cls._make_id()
        root = TodoNode(
            id=root_id,
            name=goal_name,
            description=description,
            why="Root goal — the outcome this entire plan works toward.",
            parent_id=None,
            status="in_progress",
        )
        return cls(nodes={root_id: root}, root_id=root_id)

    @classmethod
    def from_json(cls, path: str) -> "TodoTree":
        with open(path) as f:
            data = json.load(f)
        nodes = {nid: TodoNode(**nd) for nid, nd in data["nodes"].items()}
        return cls(nodes=nodes, root_id=data["root_id"])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "root_id": self.root_id,
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_node(
        self,
        parent_id: str,
        name: str,
        description: str,
        why: str,
        is_leaf: bool = False,
        implementation_note: str | None = None,
    ) -> str:
        """Add a child node under parent_id. Returns the new node's id."""
        node_id = self._make_id()
        node = TodoNode(
            id=node_id,
            name=name,
            description=description,
            why=why,
            parent_id=parent_id,
            is_leaf=is_leaf,
            implementation_note=implementation_note,
        )
        self.nodes[node_id] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node_id)
        return node_id

    def mark_status(self, node_id: str, status: str):
        """Set status on a node. Valid values: pending, in_progress, done, failed."""
        self.nodes[node_id].status = status

    def mark_failed(self, node_id: str, experiment_ref: str | None = None):
        """Mark a node as failed (branch was tried and reverted). Node is never deleted."""
        node = self.nodes[node_id]
        node.status = "failed"
        if experiment_ref and experiment_ref not in node.experiment_refs:
            node.experiment_refs.append(experiment_ref)

    def attach_workflow(self, node_id: str, workflow_path: str):
        """Link a workflow JSON file to a leaf node."""
        self.nodes[node_id].workflow_ref = workflow_path
        self.nodes[node_id].is_leaf = True

    def link_experiment(self, node_id: str, experiment_path: str, accepted: bool):
        """Record an experiment trial against a node. Accepted rounds don't change status."""
        node = self.nodes[node_id]
        if experiment_path not in node.experiment_refs:
            node.experiment_refs.append(experiment_path)
        if accepted and node.status not in ("done",):
            node.status = "done"

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_path_to_root(self, node_id: str) -> list[TodoNode]:
        """Return the chain from node up to root (inclusive), root last."""
        path = []
        current = self.nodes.get(node_id)
        while current:
            path.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        return path

    def get_leaves(self) -> list[TodoNode]:
        return [n for n in self.nodes.values() if not n.children_ids]

    def get_failed_branches(self) -> list[TodoNode]:
        return [n for n in self.nodes.values() if n.status == "failed"]

    def get_children(self, node_id: str) -> list[TodoNode]:
        return [self.nodes[cid] for cid in self.nodes[node_id].children_ids if cid in self.nodes]

    # ------------------------------------------------------------------
    # Text rendering
    # ------------------------------------------------------------------

    def get_subtree_text(self, node_id: str = None, _indent: int = 0) -> str:
        """Render the tree (or a subtree) as indented human-readable text."""
        node_id = node_id or self.root_id
        node = self.nodes[node_id]
        pad = "  " * _indent
        prefix = f"[{node.status}] " if node.status != "pending" else ""
        lines = [f"{pad}{prefix}{node.name}"]
        lines.append(f"{pad}  Desc: {node.description}")
        lines.append(f"{pad}  Why:  {node.why}")
        if node.workflow_ref:
            lines.append(f"{pad}  Workflow: {node.workflow_ref}")
        if node.implementation_note:
            lines.append(f"{pad}  Impl: {node.implementation_note}")
        if node.experiment_refs:
            for ref in node.experiment_refs:
                lines.append(f"{pad}  Experiment: {ref}")
        for child_id in node.children_ids:
            lines.append(self.get_subtree_text(child_id, _indent + 1))
        return "\n".join(lines)

    def summary(self) -> str:
        """One-level overview of root + direct children — compact optimizer context."""
        root = self.nodes[self.root_id]
        lines = [f"Goal: {root.name}", f"  {root.description}", ""]
        for child_id in root.children_ids:
            child = self.nodes[child_id]
            lines.append(f"  [{child.status}] {child.name}: {child.description}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id() -> str:
        return uuid.uuid4().hex[:6]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """
    Recursively decomposes a goal into a TodoTree using structured LLM output.

    Usage:
        planner = Planner("groq/llama-3.3-70b-versatile", 0.59, 0.79)
        tree = planner.plan(
            goal="Customer support pipeline",
            context="Available tools: lookup_invoice, check_system_status",
        )
        tree.save("plan/todo_tree.json")
        print(tree.get_subtree_text())
    """

    _SYSTEM = (
        "You are a technical planning agent. Your job is to decompose a goal into subgoals.\n"
        "For each subgoal provide:\n"
        "  name: short title (≤8 words)\n"
        "  description: abstracted solution in ≤3 sentences\n"
        "  why: what problem this subgoal addresses, in ≤3 sentences\n"
        "  is_leaf: true ONLY if this subgoal can be solved directly by a single "
        "workflow, tool, or code implementation — no further planning needed\n"
        "  implementation_note: if is_leaf=true, briefly describe what implements it\n"
        "Return 2–5 subgoals per decomposition. Be concrete and technical."
    )

    def __init__(self, model: str, input_cpm: float, output_cpm: float):
        self._llm = LLM_API(
            model_name=model,
            system_prompt=self._SYSTEM,
            input_token_cpm=input_cpm,
            output_token_cpm=output_cpm,
        )

    def plan(
        self,
        goal: str,
        context: str = "",
        max_depth: int = 5,
    ) -> TodoTree:
        """Build a complete TodoTree from a goal string."""
        tree = TodoTree.new(goal, goal)
        self._decompose(tree, tree.root_id, context, depth=0, max_depth=max_depth)
        return tree

    def extend(
        self,
        tree: TodoTree,
        node_id: str,
        context: str = "",
        max_depth: int = 3,
    ) -> TodoTree:
        """Decompose an existing node further — used when the optimizer finds a new issue."""
        self._decompose(tree, node_id, context, depth=0, max_depth=max_depth)
        return tree

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decompose(
        self,
        tree: TodoTree,
        node_id: str,
        context: str,
        depth: int,
        max_depth: int,
    ):
        if depth >= max_depth:
            return

        node = tree.nodes[node_id]
        siblings_summary = ""
        if node.parent_id:
            siblings = [
                tree.nodes[cid].name
                for cid in tree.nodes[node.parent_id].children_ids
                if cid != node_id and cid in tree.nodes
            ]
            if siblings:
                siblings_summary = f"\nOther subgoals already planned at this level: {', '.join(siblings)}"

        prompt = (
            f"Decompose this goal into subgoals:\n\n"
            f"Goal name: {node.name}\n"
            f"Goal description: {node.description}\n"
            f"Why this goal exists: {node.why}\n"
        )
        if context:
            prompt += f"\nContext / available capabilities:\n{context}\n"
        if siblings_summary:
            prompt += siblings_summary

        raw = self._llm(
            prompt,
            response_type=ResponseType.JSON_SCHEMA,
            response_basemodel=DecompositionSchema,
        )
        self._llm.reset_history()

        try:
            decomposition = DecompositionSchema.model_validate_json(raw)
        except Exception:
            return

        for sg in decomposition.subgoals:
            child_id = tree.add_node(
                parent_id=node_id,
                name=sg.name,
                description=sg.description,
                why=sg.why,
                is_leaf=sg.is_leaf,
                implementation_note=sg.implementation_note,
            )
            if not sg.is_leaf:
                self._decompose(tree, child_id, context, depth + 1, max_depth)