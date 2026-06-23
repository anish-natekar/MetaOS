"""Unit tests for Workflow, WorkflowStep, WorkflowState — no LLM calls required."""
import pytest
from src import Workflow, WorkflowStep, WorkflowState


MINIMAL_CONFIG = {
    "name": "test_wf",
    "operators": {
        "pred": {
            "type": "predict",
            "model": "groq/llama-3.3-70b-versatile",
            "input_cpm": 0.59,
            "output_cpm": 0.79,
        }
    },
    "steps": [
        {"id": "s1", "operator": "pred", "prompt": "Say hello to {name}", "output": "greeting"}
    ],
}

MULTI_STEP_CONFIG = {
    "name": "multi",
    "operators": {
        "pred": {
            "type": "predict",
            "model": "groq/llama-3.3-70b-versatile",
            "input_cpm": 0.59,
            "output_cpm": 0.79,
        }
    },
    "steps": [
        {"id": "step_a", "operator": "pred", "prompt": "First: {x}", "output": "a_out"},
        {"id": "step_b", "operator": "pred", "prompt": "Second: {a_out}", "output": "b_out", "depends_on": ["step_a"]},
    ],
}


def test_construct_from_config(sample_workflow_config):
    wf = Workflow(sample_workflow_config)
    assert wf.name == "test_wf"
    assert len(wf.steps) == 1


def test_step_fields_parsed():
    wf = Workflow(MINIMAL_CONFIG)
    step = wf.steps[0]
    assert step.id == "s1"
    assert step.operator_id == "pred"
    assert step.output_key == "greeting"
    assert "{name}" in step.prompt_template


def test_step_depends_on_parsed():
    wf = Workflow(MULTI_STEP_CONFIG)
    assert wf.steps[1].depends_on == ["step_a"]


def test_step_no_depends_on_defaults_empty():
    wf = Workflow(MINIMAL_CONFIG)
    assert wf.steps[0].depends_on == []


def test_operator_built():
    wf = Workflow(MINIMAL_CONFIG)
    assert "pred" in wf.operators


def test_workflow_from_json(tmp_path):
    import json
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(MINIMAL_CONFIG))
    wf = Workflow.from_json(str(path))
    assert wf.name == "test_wf"


def test_workflow_from_checkpoint(tmp_path):
    import json
    checkpoint = {
        "workflow": MINIMAL_CONFIG,
        "prompts": {"s1": "Customized prompt for {name}"},
    }
    path = tmp_path / "ckpt.json"
    path.write_text(json.dumps(checkpoint))
    wf = Workflow.from_checkpoint(str(path))
    assert "Customized" in wf.steps[0].prompt_template


def test_condition_field_parsed():
    config = {
        "name": "cond_wf",
        "operators": {
            "pred": {"type": "predict", "model": "groq/llama-3.3-70b-versatile", "input_cpm": 0.59, "output_cpm": 0.79}
        },
        "steps": [
            {"id": "s1", "operator": "pred", "prompt": "x", "output": "out",
             "condition": "{category} == 'billing'"},
        ],
    }
    wf = Workflow(config)
    assert wf.steps[0].condition == "{category} == 'billing'"


# ---------------------------------------------------------------------------
# WorkflowState
# ---------------------------------------------------------------------------

def test_state_set_and_get():
    state = WorkflowState({"x": 10})
    state.set("y", 20)
    assert state.snapshot["x"] == 10
    assert state.snapshot["y"] == 20


def test_state_duplicate_key_raises():
    state = WorkflowState({"x": 1})
    with pytest.raises(ValueError, match="already set"):
        state.set("x", 2)


def test_state_resolve_template():
    state = WorkflowState({"name": "Alice"})
    result = state.resolve("Hello {name}!")
    assert result == "Hello Alice!"


def test_state_resolve_missing_key_raises():
    state = WorkflowState({"x": 1})
    with pytest.raises(KeyError):
        state.resolve("{missing_key}")


def test_state_eval_condition_true():
    state = WorkflowState({"category": "billing"})
    assert state.eval_condition("{category} == 'billing'") is True


def test_state_eval_condition_false():
    state = WorkflowState({"category": "general"})
    assert state.eval_condition("{category} == 'billing'") is False


def test_state_snapshot_is_copy():
    state = WorkflowState({"x": 1})
    snap = state.snapshot
    snap["x"] = 999
    assert state.snapshot["x"] == 1
