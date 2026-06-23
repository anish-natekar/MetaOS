"""Unit tests for WorkflowRegistry — no LLM calls required."""
import json
import pytest
from src import WorkflowRegistry


SAMPLE_CONFIG = {
    "name": "support",
    "operators": {
        "pred": {"type": "predict", "model": "groq/llama-3.3-70b-versatile", "input_cpm": 0.59, "output_cpm": 0.79}
    },
    "steps": [{"id": "s1", "operator": "pred", "prompt": "Hi", "output": "out"}],
}


def test_save_creates_files(tmp_path):
    reg = WorkflowRegistry(str(tmp_path))
    reg.save("support", SAMPLE_CONFIG, description="Support pipeline")
    assert (tmp_path / "support.json").exists()
    assert (tmp_path / "_index.json").exists()


def test_has_after_save(tmp_path):
    reg = WorkflowRegistry(str(tmp_path))
    assert not reg.has("support")
    reg.save("support", SAMPLE_CONFIG)
    assert reg.has("support")


def test_load_returns_config(tmp_path):
    reg = WorkflowRegistry(str(tmp_path))
    reg.save("support", SAMPLE_CONFIG)
    config = reg.load("support")
    assert config["name"] == "support"
    assert "operators" in config
    assert "steps" in config


def test_load_missing_raises(tmp_path):
    reg = WorkflowRegistry(str(tmp_path))
    with pytest.raises(KeyError, match="missing"):
        reg.load("missing")


def test_list_workflows_empty(tmp_path):
    reg = WorkflowRegistry(str(tmp_path))
    assert reg.list_workflows() == []


def test_list_workflows_after_save(tmp_path):
    reg = WorkflowRegistry(str(tmp_path))
    reg.save("wf_a", SAMPLE_CONFIG, description="First")
    reg.save("wf_b", SAMPLE_CONFIG, description="Second")
    names = [w["name"] for w in reg.list_workflows()]
    assert "wf_a" in names
    assert "wf_b" in names


def test_list_workflows_has_required_keys(tmp_path):
    reg = WorkflowRegistry(str(tmp_path))
    reg.save("wf", SAMPLE_CONFIG, description="desc", tags=["nlp"])
    wf = reg.list_workflows()[0]
    assert "name" in wf
    assert "description" in wf
    assert "tags" in wf


def test_search_by_name(tmp_path):
    reg = WorkflowRegistry(str(tmp_path))
    reg.save("support_classifier", SAMPLE_CONFIG, description="classify")
    reg.save("email_writer", SAMPLE_CONFIG, description="write email")
    results = reg.search("support")
    assert "support_classifier" in results
    assert "email_writer" not in results


def test_search_by_description(tmp_path):
    reg = WorkflowRegistry(str(tmp_path))
    reg.save("wf", SAMPLE_CONFIG, description="sentiment analysis pipeline")
    assert "wf" in reg.search("sentiment")


def test_search_by_tag(tmp_path):
    reg = WorkflowRegistry(str(tmp_path))
    reg.save("wf", SAMPLE_CONFIG, tags=["nlp", "classification"])
    assert "wf" in reg.search("nlp")


def test_search_case_insensitive(tmp_path):
    reg = WorkflowRegistry(str(tmp_path))
    reg.save("wf", SAMPLE_CONFIG, description="NLP pipeline")
    assert "wf" in reg.search("nlp")


def test_index_persists_across_instances(tmp_path):
    reg1 = WorkflowRegistry(str(tmp_path))
    reg1.save("wf", SAMPLE_CONFIG, description="test")

    reg2 = WorkflowRegistry(str(tmp_path))
    assert reg2.has("wf")
    assert len(reg2.list_workflows()) == 1


def test_index_no_name_collision(tmp_path):
    """Index entry should not contain a 'name' key that conflicts with list_workflows name injection."""
    reg = WorkflowRegistry(str(tmp_path))
    reg.save("my_wf", SAMPLE_CONFIG, description="test")

    index_path = tmp_path / "_index.json"
    index = json.loads(index_path.read_text())
    assert "name" not in index["my_wf"]

    result = reg.list_workflows()[0]
    assert result["name"] == "my_wf"
