"""Integration tests for PromptOptimizer — requires a live API key."""
import pytest
from src import Workflow, Dataset, PromptOptimizer

pytestmark = pytest.mark.integration

WORKFLOW_CONFIG = {
    "name": "classify_sentiment",
    "operators": {
        "router": {
            "type": "router",
            "model": "groq/llama-3.3-70b-versatile",
            "input_cpm": 0.59,
            "output_cpm": 0.79,
        }
    },
    "steps": [
        {
            "id": "classify",
            "operator": "router",
            "prompt": "Classify the sentiment of this text: {text}",
            "output": "sentiment",
            "args": {"routes": ["positive", "negative", "neutral"]},
        }
    ],
}

EXAMPLES = [
    {"inputs": {"text": "I love this product!"}, "expected": "positive"},
    {"inputs": {"text": "This is terrible."}, "expected": "negative"},
    {"inputs": {"text": "It arrived on time."}, "expected": "neutral"},
    {"inputs": {"text": "Amazing experience!"}, "expected": "positive"},
]


def test_prompt_optimizer_one_round(tmp_path, model, cpm):
    in_cpm, out_cpm = cpm
    wf = Workflow(WORKFLOW_CONFIG)
    ds = Dataset.from_list(EXAMPLES)
    evaluator = ds.make_evaluator("sentiment", mode="exact_match")

    opt = PromptOptimizer(
        workflow=wf,
        evaluator=evaluator,
        optimizer_model=model,
        input_cpm=in_cpm,
        output_cpm=out_cpm,
    )

    result_wf = opt.optimize(
        examples=ds.inputs,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        experiment_dir=str(tmp_path / "experiments"),
        max_rounds=1,
    )

    assert result_wf is not None
    assert len(result_wf.steps) == len(wf.steps)
