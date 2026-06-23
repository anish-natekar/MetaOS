"""Unit tests for TodoNode and TodoTree — no LLM calls required."""
import pytest
from src import TodoTree, TodoNode


def _make_tree():
    """Helper: a tree with a root and two children."""
    tree = TodoTree.new("Build pipeline", "Create an end-to-end AI pipeline")
    child_a = tree.add_node(tree.root_id, "Classify", "Classify input text", "Need routing", is_leaf=True)
    child_b = tree.add_node(tree.root_id, "Write reply", "Draft a response", "User needs answer")
    return tree, child_a, child_b


def test_new_creates_root():
    tree = TodoTree.new("Goal", "Description of goal")
    assert len(tree.nodes) == 1
    root = tree.nodes[tree.root_id]
    assert root.name == "Goal"
    assert root.status == "in_progress"
    assert root.parent_id is None


def test_add_node_parent_link():
    tree, child_a, child_b = _make_tree()
    assert child_a in tree.nodes[tree.root_id].children_ids
    assert child_b in tree.nodes[tree.root_id].children_ids
    assert tree.nodes[child_a].parent_id == tree.root_id


def test_add_node_is_leaf_flag():
    tree, child_a, child_b = _make_tree()
    assert tree.nodes[child_a].is_leaf is True
    assert tree.nodes[child_b].is_leaf is False


def test_mark_status():
    tree, child_a, _ = _make_tree()
    tree.mark_status(child_a, "done")
    assert tree.nodes[child_a].status == "done"


def test_mark_failed_sets_status_and_ref():
    tree, child_a, _ = _make_tree()
    tree.mark_failed(child_a, "experiments/run1.md")
    node = tree.nodes[child_a]
    assert node.status == "failed"
    assert "experiments/run1.md" in node.experiment_refs


def test_mark_failed_no_duplicate_refs():
    tree, child_a, _ = _make_tree()
    tree.mark_failed(child_a, "experiments/run1.md")
    tree.mark_failed(child_a, "experiments/run1.md")
    assert tree.nodes[child_a].experiment_refs.count("experiments/run1.md") == 1


def test_link_experiment_accepted_sets_done():
    tree, child_a, _ = _make_tree()
    tree.link_experiment(child_a, "experiments/run1.md", accepted=True)
    assert tree.nodes[child_a].status == "done"
    assert "experiments/run1.md" in tree.nodes[child_a].experiment_refs


def test_link_experiment_rejected_leaves_status():
    tree, child_a, _ = _make_tree()
    tree.link_experiment(child_a, "experiments/run1.md", accepted=False)
    assert tree.nodes[child_a].status == "pending"


def test_attach_workflow():
    tree, child_a, _ = _make_tree()
    tree.attach_workflow(child_a, "workflows/classify.json")
    node = tree.nodes[child_a]
    assert node.workflow_ref == "workflows/classify.json"
    assert node.is_leaf is True


def test_get_leaves_returns_childless_nodes():
    tree, child_a, child_b = _make_tree()
    leaves = tree.get_leaves()
    leaf_ids = {n.id for n in leaves}
    assert child_a in leaf_ids
    assert child_b in leaf_ids
    assert tree.root_id not in leaf_ids


def test_get_failed_branches():
    tree, child_a, child_b = _make_tree()
    tree.mark_failed(child_a)
    failed = tree.get_failed_branches()
    assert any(n.id == child_a for n in failed)
    assert not any(n.id == child_b for n in failed)


def test_get_subtree_text_contains_names():
    tree, _, _ = _make_tree()
    text = tree.get_subtree_text()
    assert "Build pipeline" in text
    assert "Classify" in text
    assert "Write reply" in text


def test_get_subtree_text_shows_status():
    tree, child_a, _ = _make_tree()
    tree.mark_status(child_a, "done")
    text = tree.get_subtree_text()
    assert "[done]" in text


def test_summary_contains_root_and_children():
    tree, _, _ = _make_tree()
    s = tree.summary()
    assert "Build pipeline" in s
    assert "Classify" in s
    assert "Write reply" in s


def test_save_and_load_roundtrip(tmp_path):
    tree, child_a, child_b = _make_tree()
    tree.mark_status(child_a, "done")
    path = str(tmp_path / "tree.json")
    tree.save(path)

    loaded = TodoTree.from_json(path)
    assert loaded.root_id == tree.root_id
    assert set(loaded.nodes.keys()) == set(tree.nodes.keys())
    assert loaded.nodes[child_a].status == "done"
    assert loaded.nodes[child_b].name == "Write reply"


def test_save_creates_parent_dirs(tmp_path):
    tree = TodoTree.new("Test", "desc")
    path = str(tmp_path / "nested" / "dir" / "tree.json")
    tree.save(path)
    loaded = TodoTree.from_json(path)
    assert loaded.root_id == tree.root_id


def test_get_path_to_root():
    tree, child_a, _ = _make_tree()
    path = tree.get_path_to_root(child_a)
    assert path[0].id == child_a
    assert path[-1].id == tree.root_id
