"""Integration tests for LLM_API — requires a live API key."""
import pytest
from src import LLM_API, ResponseType
from pydantic import BaseModel

pytestmark = pytest.mark.integration

_SYSTEM = "You are a concise assistant. Answer in as few words as possible."


def test_basic_text_call(model, cpm):
    in_cpm, out_cpm = cpm
    llm = LLM_API(model, _SYSTEM, in_cpm, out_cpm)
    response = llm("Say the word 'pineapple' and nothing else.")
    assert "pineapple" in response.lower()


def test_metrics_accumulate(model, cpm):
    in_cpm, out_cpm = cpm
    llm = LLM_API(model, _SYSTEM, in_cpm, out_cpm)
    llm("Say hi.")
    llm("Say bye.")
    assert llm.metrics.total_number_of_calls == 2
    assert llm.metrics.input_cost > 0
    assert llm.metrics.output_cost > 0


def test_history_accumulates(model, cpm):
    in_cpm, out_cpm = cpm
    llm = LLM_API(model, _SYSTEM, in_cpm, out_cpm, keep_history=True)
    llm("My favorite color is blue.")
    reply = llm("What is my favorite color?")
    assert "blue" in reply.lower()


def test_reset_history(model, cpm):
    in_cpm, out_cpm = cpm
    llm = LLM_API(model, _SYSTEM, in_cpm, out_cpm, keep_history=True)
    llm("My favorite color is blue.")
    llm.reset_history()
    non_system = [m for m in llm.history if m.get("role") != "system"]
    assert non_system == []


def test_json_schema_output(model, cpm):
    in_cpm, out_cpm = cpm

    class Color(BaseModel):
        name: str
        hex_code: str

    llm = LLM_API(model, _SYSTEM, in_cpm, out_cpm)
    result = llm(
        "Give me the color red as JSON.",
        response_type=ResponseType.JSON_SCHEMA,
        response_basemodel=Color,
    )
    assert isinstance(result, Color)
    assert "red" in result.name.lower()


def test_tool_calling(model, cpm):
    in_cpm, out_cpm = cpm

    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return f"Sunny, 25°C in {city}"

    llm = LLM_API(model, _SYSTEM, in_cpm, out_cpm)
    response = llm("What is the weather in Paris?", tools=[get_weather])
    assert "paris" in response.lower() or "25" in response or "sunny" in response.lower()
