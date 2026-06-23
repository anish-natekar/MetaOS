"""Unit tests for ToolRegistry — no LLM calls required."""
import json
import pytest
from src import ToolRecord, ToolRegistry


def _make_record(name="add_numbers"):
    from datetime import datetime
    return ToolRecord(
        name=name,
        description="Add two integers.",
        source=f"def {name}(a: int, b: int) -> int:\n    return a + b\n",
        signature="(a: int, b: int) -> int",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        version=1,
        test_cases=[{"args": {"a": 1, "b": 2}, "expected": 3}],
        tags=["math"],
        synthesis_model="test",
    )


def test_save_creates_files(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    reg.save(_make_record())
    assert (tmp_path / "add_numbers.json").exists()
    assert (tmp_path / "_index.json").exists()


def test_has_after_save(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    assert not reg.has("add_numbers")
    reg.save(_make_record())
    assert reg.has("add_numbers")


def test_load_roundtrip(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    original = _make_record()
    reg.save(original)
    loaded = reg.load("add_numbers")
    assert loaded.name == "add_numbers"
    assert loaded.description == "Add two integers."
    assert loaded.tags == ["math"]
    assert loaded.version == 1


def test_load_missing_raises(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    with pytest.raises(KeyError, match="no_tool"):
        reg.load("no_tool")


def test_load_as_callable(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    reg.save(_make_record())
    fn = reg.load_as_callable("add_numbers")
    assert callable(fn)
    assert fn(3, 4) == 7


def test_load_as_callable_has_docstring(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    reg.save(_make_record())
    fn = reg.load_as_callable("add_numbers")
    assert fn.__doc__ is not None


def test_list_tools_empty(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    assert reg.list_tools() == []


def test_list_tools_after_save(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    reg.save(_make_record("tool_a"))
    reg.save(_make_record("tool_b"))
    tools = reg.list_tools()
    names = [t["name"] for t in tools]
    assert "tool_a" in names
    assert "tool_b" in names


def test_list_tools_has_required_keys(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    reg.save(_make_record())
    tool = reg.list_tools()[0]
    assert "name" in tool
    assert "description" in tool
    assert "signature" in tool
    assert "tags" in tool
    assert "version" in tool


def test_search_by_name(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    reg.save(_make_record("string_reverse"))
    reg.save(_make_record("add_numbers"))
    results = reg.search("string")
    assert "string_reverse" in results
    assert "add_numbers" not in results


def test_search_by_tag(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    record = _make_record()
    record.tags = ["finance", "math"]
    reg.save(record)
    assert "add_numbers" in reg.search("finance")


def test_search_case_insensitive(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    reg.save(_make_record())
    assert "add_numbers" in reg.search("MATH")


def test_delete(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    reg.save(_make_record())
    assert reg.has("add_numbers")
    reg.delete("add_numbers")
    assert not reg.has("add_numbers")
    assert not (tmp_path / "add_numbers.json").exists()


def test_delete_updates_index(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    reg.save(_make_record())
    reg.delete("add_numbers")
    index_path = tmp_path / "_index.json"
    index = json.loads(index_path.read_text())
    assert "add_numbers" not in index


def test_delete_nonexistent_does_not_raise(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    reg.delete("never_saved")   # should not raise


def test_index_persists_across_instances(tmp_path):
    reg1 = ToolRegistry(str(tmp_path))
    reg1.save(_make_record())

    reg2 = ToolRegistry(str(tmp_path))
    assert reg2.has("add_numbers")
    tools = reg2.list_tools()
    assert len(tools) == 1


def test_save_overwrites_existing(tmp_path):
    reg = ToolRegistry(str(tmp_path))
    r = _make_record()
    reg.save(r)
    r.version = 2
    r.description = "Updated description."
    reg.save(r)
    loaded = reg.load("add_numbers")
    assert loaded.version == 2
    assert loaded.description == "Updated description."
