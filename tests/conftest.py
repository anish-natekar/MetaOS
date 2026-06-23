import json
import pytest


@pytest.fixture
def sample_workflow_config():
    return {
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
            {
                "id": "step1",
                "operator": "pred",
                "prompt": "Say hi to {name}",
                "output": "greeting",
            }
        ],
    }


@pytest.fixture
def sample_dataset():
    from src import Dataset
    return Dataset.from_list([
        {"inputs": {"x": "hello"}, "expected": "hi"},
        {"inputs": {"x": "bye"}, "expected": "goodbye"},
        {"inputs": {"x": "thanks"}, "expected": "welcome"},
        {"inputs": {"x": "yes"}, "expected": "no"},
        {"inputs": {"x": "up"}, "expected": "down"},
    ])


@pytest.fixture
def sample_tool_record():
    from src import ToolRecord
    from datetime import datetime
    return ToolRecord(
        name="add_numbers",
        description="Add two integers and return their sum.",
        source="def add_numbers(a: int, b: int) -> int:\n    return a + b\n",
        signature="(a: int, b: int) -> int",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        version=1,
        test_cases=[{"args": {"a": 1, "b": 2}, "expected": 3}],
        tags=["math", "arithmetic"],
        synthesis_model="test",
    )
