import json
from pydantic import BaseModel

from .llm_api import LLM_API, ResponseType
from .optimizer import Dataset


# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------

class _ExampleSchema(BaseModel):
    inputs: dict   # maps to workflow input keys
    expected: str  # ground-truth expected output for the evaluator


class _DatasetSchema(BaseModel):
    examples: list[_ExampleSchema]
    notes: str     # LLM's note on coverage / quality


# ---------------------------------------------------------------------------
# DataSynthesizer
# ---------------------------------------------------------------------------

class DataSynthesizer:
    """
    Generates synthetic labeled (inputs, expected) pairs for use with Dataset
    and the MetaOS optimizers.

    Usage — basic:
        synth = DataSynthesizer("groq/llama-3.3-70b-versatile", 0.59, 0.79)
        dataset = synth.generate(
            goal="classify customer support messages",
            input_keys=["message"],
            output_description="category: 'billing', 'technical', or 'general'",
            n_examples=20,
        )
        train, test = dataset.split()

    Usage — from web context:
        dataset = synth.generate_from_web(
            web_query="common customer support complaint examples",
            goal="classify customer support messages",
            input_keys=["message"],
            output_description="category: 'billing', 'technical', or 'general'",
            n_examples=20,
        )
    """

    _SYSTEM = (
        "You are a synthetic data generator for AI evaluation. "
        "Given a goal and a description of the expected output, generate realistic and diverse "
        "(inputs, expected_output) pairs that a workflow would encounter in production. "
        "Ensure variety: different phrasing, edge cases, ambiguous cases, and easy cases. "
        "Keep expected outputs concise and consistent with the output_description format. "
        "Return ONLY the structured JSON matching the schema — no prose outside it."
    )

    def __init__(self, model: str, input_cpm: float, output_cpm: float):
        self._llm = LLM_API(
            model_name=model,
            system_prompt=self._SYSTEM,
            input_token_cpm=input_cpm,
            output_token_cpm=output_cpm,
        )

    def generate(
        self,
        goal: str,
        input_keys: list[str],
        output_description: str,
        n_examples: int = 10,
        context: str = "",
    ) -> Dataset:
        """
        Generate n_examples synthetic labeled examples for the given goal.

        Args:
            goal:               What the workflow is supposed to accomplish.
            input_keys:         Which input fields to populate (e.g. ["message", "customer_id"]).
            output_description: What the expected output looks like (e.g. "category: billing/technical/general").
            n_examples:         Number of examples to generate.
            context:            Optional additional context (domain info, web search results, etc.).

        Returns:
            Dataset with generated examples, ready for train/test split and evaluation.
        """
        prompt = self._build_prompt(goal, input_keys, output_description, n_examples, context)
        raw = self._llm(prompt, response_type=ResponseType.JSON_SCHEMA, response_basemodel=_DatasetSchema)
        self._llm.reset_history()

        schema = _DatasetSchema.model_validate_json(raw)
        records = [{"inputs": e.inputs, "expected": e.expected} for e in schema.examples]
        return Dataset.from_list(records)

    def generate_from_web(
        self,
        web_query: str,
        goal: str,
        input_keys: list[str],
        output_description: str,
        n_examples: int = 10,
    ) -> Dataset:
        """
        First search the web using an LLM web-search tool, then generate synthetic
        examples grounded in the retrieved context.

        The web search is performed via a tool call to the LLM (model must support tool use).
        Falls back to training-knowledge-based generation if web search is unavailable.
        """
        context = self._fetch_web_context(web_query)
        return self.generate(goal, input_keys, output_description, n_examples, context=context)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        goal: str,
        input_keys: list[str],
        output_description: str,
        n_examples: int,
        context: str,
    ) -> str:
        parts = [
            f"Goal: {goal}",
            f"Input fields to generate: {json.dumps(input_keys)}",
            f"Expected output format: {output_description}",
            f"Number of examples to generate: {n_examples}",
        ]
        if context:
            parts.append(f"\nDomain context / reference material:\n{context}")
        parts.append(
            "\nGenerate diverse, realistic examples. Include easy cases, hard cases, "
            "and edge cases. Vary the phrasing and scenario for each example."
        )
        return "\n".join(parts)

    def _fetch_web_context(self, query: str) -> str:
        """
        Use LLM tool-calling to perform a web search and extract relevant context.
        Returns the web search result as a plain text string.
        """
        def web_search(search_query: str) -> str:
            """Search the web for information. Returns relevant text snippets."""
            try:
                import urllib.request
                import urllib.parse
                # DuckDuckGo Instant Answer API (no auth required)
                q = urllib.parse.quote_plus(search_query)
                url = f"https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1"
                req = urllib.request.Request(url, headers={"User-Agent": "MetaOS/1.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode())
                snippets = []
                if data.get("AbstractText"):
                    snippets.append(data["AbstractText"])
                for r in data.get("RelatedTopics", [])[:5]:
                    if isinstance(r, dict) and r.get("Text"):
                        snippets.append(r["Text"])
                if snippets:
                    return "\n".join(snippets)
                return f"No results found for: {search_query}"
            except Exception as e:
                return f"Web search unavailable ({e}). Using training knowledge."

        prompt = (
            f"Search the web for information relevant to this topic and summarize the key findings:\n"
            f"{query}\n\n"
            "Call the web_search tool with a specific query, then summarize what you found."
        )

        result = self._llm(prompt, tools=[web_search])
        # If LLM returned tool results dict, extract the text
        if isinstance(result, dict):
            # Tool was called; LLM hasn't summarized yet — ask for summary
            summary = self._llm("Summarize the search results above in a few sentences.")
        else:
            summary = result

        self._llm.reset_history()
        return summary if isinstance(summary, str) else str(summary)
