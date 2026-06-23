"""Fixtures shared across integration tests."""
import os
import pytest

MODEL = os.getenv("METAOS_TEST_MODEL", "groq/llama-3.3-70b-versatile")
INPUT_CPM = float(os.getenv("METAOS_INPUT_CPM", "0.59"))
OUTPUT_CPM = float(os.getenv("METAOS_OUTPUT_CPM", "0.79"))

_skip_no_key = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"),
    reason="No LLM API key set (GROQ_API_KEY or OPENAI_API_KEY)",
)


@pytest.fixture(scope="session")
def model():
    return MODEL


@pytest.fixture(scope="session")
def cpm():
    return INPUT_CPM, OUTPUT_CPM


@pytest.fixture(autouse=True)
def skip_without_api_key():
    """Auto-skip every test in the integration package if no key is available."""
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No LLM API key available")
