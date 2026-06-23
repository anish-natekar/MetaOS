"""Integration tests for Predict, CoT, React, Router — requires a live API key."""
import pytest
from src import Predict, CoT, React, Router

pytestmark = pytest.mark.integration


def test_predict(model, cpm):
    in_cpm, out_cpm = cpm
    pred = Predict("test", model, in_cpm, out_cpm)
    result = pred("What is 2 + 2? Answer with just the number.")
    assert "4" in result


def test_cot(model, cpm):
    in_cpm, out_cpm = cpm
    cot = CoT("test", model, in_cpm, out_cpm)
    reasoning, answer = cot("What is 15 * 4? Show your work briefly then give the final answer.")
    assert reasoning
    assert "60" in answer


def test_react_with_tool(model, cpm):
    in_cpm, out_cpm = cpm

    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    react = React("test", model, in_cpm, out_cpm)
    result = react("What is 7 multiplied by 9?", tools=[multiply])
    assert "63" in result


def test_router(model, cpm):
    in_cpm, out_cpm = cpm
    router = Router("test", model, in_cpm, out_cpm)
    route = router(
        "My invoice is wrong and I was charged twice.",
        routes=["billing", "technical", "general"],
    )
    assert route == "billing"
