"""Integration tests for MetaOSAgent — requires a live API key."""
import pytest
from src import MetaOSAgent

pytestmark = pytest.mark.integration


def _make_agent(tmp_path, model, cpm):
    in_cpm, out_cpm = cpm
    return MetaOSAgent(
        model=model,
        input_cpm=in_cpm,
        output_cpm=out_cpm,
        tool_registry_dir=str(tmp_path / "tools"),
        workflow_registry_dir=str(tmp_path / "workflows"),
        plan_dir=str(tmp_path / "plan"),
    )


def test_metrics_initial_state(tmp_path, model, cpm):
    agent = _make_agent(tmp_path, model, cpm)
    m = agent.metrics
    assert m["total_cost"] == 0.0
    assert m["total_calls"] == 0
    assert m["total_tool_calls"] == 0


def test_reset_session_clears_tree(tmp_path, model, cpm):
    agent = _make_agent(tmp_path, model, cpm)
    assert agent.tree is None
    agent.reset_session()
    assert agent.tree is None


def test_search_tools_empty(tmp_path, model, cpm):
    agent = _make_agent(tmp_path, model, cpm)
    result = agent.search_tools("anything")
    assert isinstance(result, str)
    assert "no" in result.lower() or "0" in result or "found" in result.lower()


def test_search_workflows_empty(tmp_path, model, cpm):
    agent = _make_agent(tmp_path, model, cpm)
    result = agent.search_workflows("anything")
    assert isinstance(result, str)


def test_list_tools_empty(tmp_path, model, cpm):
    agent = _make_agent(tmp_path, model, cpm)
    result = agent.list_tools()
    assert isinstance(result, str)


def test_list_workflows_empty(tmp_path, model, cpm):
    agent = _make_agent(tmp_path, model, cpm)
    result = agent.list_workflows()
    assert isinstance(result, str)


def test_chat_returns_string(tmp_path, model, cpm):
    agent = _make_agent(tmp_path, model, cpm)
    response = agent.chat("Hello, what can you help me with? Just reply in one sentence.")
    assert isinstance(response, str)
    assert len(response) > 0


def test_metrics_tracked_after_chat(tmp_path, model, cpm):
    agent = _make_agent(tmp_path, model, cpm)
    agent.chat("Say the word 'hello' and nothing else.")
    m = agent.metrics
    assert m["total_cost"] > 0
    assert m["total_calls"] >= 1
