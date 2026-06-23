"""Unit tests for Dataset and RunResult — no LLM calls required."""
import json
import pytest
from src import Dataset, RunResult


def test_from_list_basic():
    ds = Dataset.from_list([
        {"inputs": {"q": "hello"}, "expected": "world"},
    ])
    assert len(ds) == 1
    assert ds.examples[0].inputs == {"q": "hello"}
    assert ds.examples[0].expected == "world"


def test_from_list_custom_keys():
    ds = Dataset.from_list(
        [{"in": {"x": 1}, "out": "one"}],
        inputs_key="in",
        expected_key="out",
    )
    assert ds.examples[0].inputs == {"x": 1}
    assert ds.examples[0].expected == "one"


def test_from_json(tmp_path):
    data = [
        {"inputs": {"k": "a"}, "expected": "A"},
        {"inputs": {"k": "b"}, "expected": "B"},
    ]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))

    ds = Dataset.from_json(str(path))
    assert len(ds) == 2
    assert ds.examples[1].expected == "B"


def test_repr():
    ds = Dataset.from_list([{"inputs": {}, "expected": "x"}])
    assert "1" in repr(ds)


def test_split_sizes():
    ds = Dataset.from_list([{"inputs": {"i": i}, "expected": str(i)} for i in range(10)])
    train, test = ds.split(test_size=0.2)
    assert len(test) == 2
    assert len(train) == 8
    assert len(train) + len(test) == 10


def test_split_deterministic():
    ds = Dataset.from_list([{"inputs": {"i": i}, "expected": str(i)} for i in range(10)])
    train_a, test_a = ds.split(seed=99)
    train_b, test_b = ds.split(seed=99)
    assert [e.expected for e in test_a.examples] == [e.expected for e in test_b.examples]


def test_split_different_seeds():
    ds = Dataset.from_list([{"inputs": {"i": i}, "expected": str(i)} for i in range(10)])
    _, test_a = ds.split(seed=1)
    _, test_b = ds.split(seed=2)
    assert [e.expected for e in test_a.examples] != [e.expected for e in test_b.examples]


def test_inputs_property():
    ds = Dataset.from_list([
        {"inputs": {"x": 1}, "expected": "a"},
        {"inputs": {"x": 2}, "expected": "b"},
    ])
    inputs = ds.inputs
    assert inputs == [{"x": 1}, {"x": 2}]


def test_exact_match_evaluator_hit():
    ds = Dataset.from_list([{"inputs": {"q": "hi"}, "expected": "hello"}])
    ev = ds.make_evaluator("answer", mode="exact_match")
    score = ev({"q": "hi"}, {"answer": "hello"})
    assert score == 1.0


def test_exact_match_evaluator_miss():
    ds = Dataset.from_list([{"inputs": {"q": "hi"}, "expected": "hello"}])
    ev = ds.make_evaluator("answer", mode="exact_match")
    score = ev({"q": "hi"}, {"answer": "wrong"})
    assert score == 0.0


def test_exact_match_case_insensitive():
    ds = Dataset.from_list([{"inputs": {"q": "x"}, "expected": "Yes"}])
    ev = ds.make_evaluator("answer", mode="exact_match")
    assert ev({"q": "x"}, {"answer": "yes"}) == 1.0
    assert ev({"q": "x"}, {"answer": "YES"}) == 1.0


def test_exact_match_unknown_input_returns_zero():
    ds = Dataset.from_list([{"inputs": {"q": "a"}, "expected": "b"}])
    ev = ds.make_evaluator("answer", mode="exact_match")
    assert ev({"q": "UNKNOWN"}, {"answer": "b"}) == 0.0


def test_exact_match_dict_output_uses_answer_key():
    ds = Dataset.from_list([{"inputs": {"q": "x"}, "expected": "42"}])
    ev = ds.make_evaluator("result", mode="exact_match")
    assert ev({"q": "x"}, {"result": {"answer": "42"}}) == 1.0


def test_llm_judge_requires_model():
    ds = Dataset.from_list([{"inputs": {}, "expected": "x"}])
    with pytest.raises(ValueError, match="judge_model"):
        ds.make_evaluator("out", mode="llm_judge")


def test_unknown_mode_raises():
    ds = Dataset.from_list([{"inputs": {}, "expected": "x"}])
    with pytest.raises(ValueError, match="Unknown mode"):
        ds.make_evaluator("out", mode="bad_mode")


def test_run_result_attributes():
    rr = RunResult(inputs={"x": 1}, state={"out": "val"}, score=0.75)
    assert rr.inputs == {"x": 1}
    assert rr.state["out"] == "val"
    assert rr.score == 0.75
