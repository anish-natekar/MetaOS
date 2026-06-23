"""Integration tests for ToolSynthesizer and ToolOptimizer — requires a live API key."""
import pytest
from src import ToolRegistry, ToolSynthesizer, ToolOptimizer

pytestmark = pytest.mark.integration


def test_synthesize_basic(tmp_path, model, cpm):
    in_cpm, out_cpm = cpm
    registry = ToolRegistry(str(tmp_path / "tools"))
    synth = ToolSynthesizer(model, in_cpm, out_cpm, registry)

    record = synth.synthesize(
        description="Reverse a string and return the result",
        signature="(s: str) -> str",
        test_cases=[{"args": {"s": "hello"}, "expected": "olleh"}],
        tags=["string"],
    )

    assert record.name
    assert callable(registry.load_as_callable(record.name))
    fn = registry.load_as_callable(record.name)
    assert fn("hello") == "olleh"


def test_synthesize_saves_to_registry(tmp_path, model, cpm):
    in_cpm, out_cpm = cpm
    registry = ToolRegistry(str(tmp_path / "tools"))
    synth = ToolSynthesizer(model, in_cpm, out_cpm, registry)

    record = synth.synthesize(
        description="Return True if a number is even, False otherwise",
        signature="(n: int) -> bool",
        test_cases=[
            {"args": {"n": 2}, "expected": True},
            {"args": {"n": 3}, "expected": False},
        ],
    )

    assert registry.has(record.name)


def test_optimize_tool(tmp_path, model, cpm):
    in_cpm, out_cpm = cpm
    registry = ToolRegistry(str(tmp_path / "tools"))
    synth = ToolSynthesizer(model, in_cpm, out_cpm, registry)

    # Synthesize a tool first
    record = synth.synthesize(
        description="Compute the factorial of a non-negative integer",
        signature="(n: int) -> int",
        test_cases=[{"args": {"n": 5}, "expected": 120}],
    )

    opt = ToolOptimizer(model, in_cpm, out_cpm, registry)
    improved = opt.optimize(
        record.name,
        test_cases=[
            {"args": {"n": 0}, "expected": 1},
            {"args": {"n": 5}, "expected": 120},
        ],
        max_rounds=2,
    )

    fn = registry.load_as_callable(improved.name)
    assert fn(5) == 120
    assert fn(0) == 1
