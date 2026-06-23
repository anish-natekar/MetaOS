"""Unit tests for _run_sandbox — no LLM calls required."""
import pytest
from src.tool_forge import _run_sandbox


def test_success_returns_none():
    source = "def add(a, b):\n    return a + b\n"
    result = _run_sandbox(source, "add", [{"args": {"a": 1, "b": 2}, "expected": 3}], timeout=10)
    assert result is None


def test_wrong_output_returns_error():
    source = "def always_zero(x):\n    return 0\n"
    result = _run_sandbox(source, "always_zero", [{"args": {"x": 5}, "expected": 5}], timeout=10)
    assert result is not None
    assert "failed" in result.lower() or "expected" in result.lower()


def test_syntax_error_returns_error():
    source = "def bad(: pass"
    result = _run_sandbox(source, "bad", [], timeout=10)
    assert result is not None


def test_runtime_exception_returns_error():
    source = "def explode(x):\n    raise ValueError('boom')\n"
    result = _run_sandbox(source, "explode", [{"args": {"x": 1}}], timeout=10)
    assert result is not None
    assert "ValueError" in result or "boom" in result


def test_no_test_cases_success():
    source = "def noop():\n    return None\n"
    result = _run_sandbox(source, "noop", [], timeout=10)
    assert result is None


def test_no_test_cases_wrong_name():
    source = "def correct_name():\n    return 1\n"
    result = _run_sandbox(source, "wrong_name", [], timeout=10)
    # wrong_name is not defined — subprocess will get NameError
    assert result is not None


def test_multiple_test_cases_all_pass():
    source = "def double(n):\n    return n * 2\n"
    cases = [
        {"args": {"n": 1}, "expected": 2},
        {"args": {"n": 5}, "expected": 10},
        {"args": {"n": 0}, "expected": 0},
    ]
    assert _run_sandbox(source, "double", cases, timeout=10) is None


def test_multiple_test_cases_one_fails():
    source = "def double(n):\n    return n * 2\n"
    cases = [
        {"args": {"n": 1}, "expected": 2},
        {"args": {"n": 5}, "expected": 99},  # wrong expected
    ]
    result = _run_sandbox(source, "double", cases, timeout=10)
    assert result is not None


def test_timeout():
    # Module-level infinite loop so it triggers even with empty test_cases
    source = "while True: pass\ndef spin():\n    return 1\n"
    result = _run_sandbox(source, "spin", [{"args": {}, "expected": 1}], timeout=2)
    assert result is not None
    assert "timed out" in result.lower() or "timeout" in result.lower()


def test_with_imports():
    source = "import math\ndef circle_area(r):\n    return math.pi * r * r\n"
    result = _run_sandbox(source, "circle_area", [{"args": {"r": 0}, "expected": 0.0}], timeout=10)
    assert result is None


def test_expected_none_skips_assertion():
    # When expected is None, sandbox only checks it runs without error
    source = "def greet(name):\n    return f'hello {name}'\n"
    result = _run_sandbox(source, "greet", [{"args": {"name": "world"}}], timeout=10)
    assert result is None
