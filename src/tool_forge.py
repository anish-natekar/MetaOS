import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime

from pydantic import BaseModel

from .llm_api import LLM_API, ResponseType


# ---------------------------------------------------------------------------
# ToolRecord
# ---------------------------------------------------------------------------

@dataclass
class ToolRecord:
    name: str                    # valid Python identifier; used as filename key
    description: str             # one-sentence summary; becomes docstring
    source: str                  # complete source: import lines + def block
    signature: str               # e.g. "(x: int, y: int) -> int"
    created_at: str              # ISO-8601
    updated_at: str              # ISO-8601; bumped on each accepted optimization
    version: int                 # incremented on each accepted optimization
    test_cases: list[dict]       # [{"args": {"x": 1}, "expected": 2}, ...]
    tags: list[str]              # free-form labels for search
    synthesis_model: str         # model that generated this (audit trail)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Persistent store for synthesized tool functions.

    Each tool is saved as {registry_dir}/{name}.json (full ToolRecord).
    A lightweight {registry_dir}/_index.json maps name → {description, signature, tags, version}
    for fast listing and search without loading sources.

    Usage:
        registry = ToolRegistry("tools/")
        record = registry.load("compound_interest")
        fn = registry.load_as_callable("compound_interest")
    """

    _INDEX = "_index.json"

    def __init__(self, registry_dir: str = "tools/"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self._index: dict[str, dict] = self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, record: ToolRecord) -> None:
        """Persist a ToolRecord. Overwrites any existing record for that name."""
        record.updated_at = datetime.now().isoformat()
        path = self._path(record.name)
        data = asdict(record)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._update_index(record)

    def load(self, name: str) -> ToolRecord:
        """Load a ToolRecord by name. Raises KeyError if not found."""
        path = self._path(name)
        if not os.path.exists(path):
            raise KeyError(f"No tool named '{name}' in registry at '{self.registry_dir}'")
        with open(path) as f:
            data = json.load(f)
        return ToolRecord(**data)

    def load_as_callable(self, name: str) -> callable:
        """Load a ToolRecord and compile it into a live Python callable."""
        return self._compile(self.load(name))

    def has(self, name: str) -> bool:
        """O(1) check whether a tool exists in the registry."""
        return name in self._index

    def list_tools(self) -> list[dict]:
        """Return all index entries (no source loaded). Each entry has: name, description, signature, tags, version."""
        return [{"name": k, **v} for k, v in self._index.items()]

    def search(self, query: str) -> list[str]:
        """Case-insensitive substring search across name, description, and tags. Returns matching names."""
        q = query.lower()
        results = []
        for name, meta in self._index.items():
            haystack = " ".join([name, meta.get("description", ""), *meta.get("tags", [])])
            if q in haystack.lower():
                results.append(name)
        return results

    def delete(self, name: str) -> None:
        """Remove a tool from the registry."""
        path = self._path(name)
        if os.path.exists(path):
            os.unlink(path)
        self._index.pop(name, None)
        self._save_index()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compile(self, record: ToolRecord) -> callable:
        """exec the source in an isolated namespace and return the callable."""
        namespace: dict = {}
        try:
            exec(record.source, namespace)
        except Exception as e:
            raise RuntimeError(f"Failed to compile tool '{record.name}': {e}") from e
        fn = namespace.get(record.name)
        if fn is None or not callable(fn):
            raise RuntimeError(
                f"Tool source for '{record.name}' did not define a callable named '{record.name}'. "
                f"Defined names: {[k for k in namespace if not k.startswith('_')]}"
            )
        if not fn.__doc__:
            fn.__doc__ = record.description
        return fn

    def _update_index(self, record: ToolRecord) -> None:
        self._index[record.name] = {
            "description": record.description,
            "signature": record.signature,
            "tags": record.tags,
            "version": record.version,
        }
        self._save_index()

    def _load_index(self) -> dict:
        path = os.path.join(self.registry_dir, self._INDEX)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        path = os.path.join(self.registry_dir, self._INDEX)
        with open(path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _path(self, name: str) -> str:
        return os.path.join(self.registry_dir, f"{name}.json")


# ---------------------------------------------------------------------------
# Sandbox execution helper (module-level, shared by Synthesizer + Optimizer)
# ---------------------------------------------------------------------------

def _build_test_script(source: str, fn_name: str, test_cases: list[dict]) -> str:
    lines = [source, ""]
    if not test_cases:
        # Just verify the function is importable and callable
        lines += [
            "import sys",
            f"fn = {fn_name}",
            "if not callable(fn):",
            f"    print('{fn_name} is not callable', file=sys.stderr)",
            "    sys.exit(1)",
        ]
        return "\n".join(lines)

    lines += [
        "import sys, json",
        f"fn = {fn_name}",
        f"_test_cases = json.loads({repr(json.dumps(test_cases))})",
        "for _i, _tc in enumerate(_test_cases):",
        "    try:",
        "        _result = fn(**_tc['args'])",
        "        _expected = _tc.get('expected')",
        "        if _expected is not None and _result != _expected:",
        "            print(f'Test {_i} failed: expected {_expected!r}, got {_result!r}', file=sys.stderr)",
        "            sys.exit(1)",
        "    except Exception as _e:",
        "        print(f'Test {_i} raised {type(_e).__name__}: {_e}', file=sys.stderr)",
        "        sys.exit(1)",
    ]
    return "\n".join(lines)


def _run_sandbox(
    source: str,
    fn_name: str,
    test_cases: list[dict],
    timeout: int = 5,
) -> str | None:
    """
    Run source in a subprocess with test_cases.
    Returns None on success, or an error string on failure.
    """
    script = _build_test_script(source, fn_name, test_cases)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            tmp_path = f.name
            f.write(script)
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return proc.stderr.decode("utf-8", errors="replace").strip()
        return None
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout}s"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# ToolSynthesizer
# ---------------------------------------------------------------------------

class _GeneratedTool(BaseModel):
    name: str           # valid Python identifier
    description: str    # one sentence; becomes the docstring
    source: str         # complete Python source: import lines then def block
    explanation: str    # brief implementation note (audit only)


class ToolSynthesizer:
    """
    Generates new Python tool functions via LLM, validates them in a subprocess
    sandbox, and persists them in a ToolRegistry.

    Usage:
        synth = ToolSynthesizer("groq/llama-3.3-70b-versatile", 0.59, 0.79, registry)
        record = synth.synthesize(
            description="Calculate compound interest",
            signature="(principal: float, rate: float, years: int) -> float",
            test_cases=[{"args": {"principal": 1000, "rate": 0.05, "years": 1}, "expected": 1050.0}],
        )
        fn = registry.load_as_callable(record.name)
    """

    _SYSTEM = (
        "You are an expert Python programmer who writes clean, self-contained tool functions.\n"
        "Requirements for every function you write:\n"
        "1. All import statements appear BEFORE the def block\n"
        "2. Parameters are type-annotated; return type is annotated\n"
        "3. The function has a concise one-sentence docstring\n"
        "4. The function name is a valid Python identifier in snake_case\n"
        "5. No network calls, no filesystem writes, no external APIs\n"
        "6. No global state — the function must be fully self-contained\n"
        "Return ONLY the structured JSON matching the schema — no prose outside it."
    )

    def __init__(
        self,
        model: str,
        input_cpm: float,
        output_cpm: float,
        registry: ToolRegistry,
        sandbox_timeout: int = 5,
        max_attempts: int = 3,
    ):
        self._llm = LLM_API(
            model_name=model,
            system_prompt=self._SYSTEM,
            input_token_cpm=input_cpm,
            output_token_cpm=output_cpm,
        )
        self.registry = registry
        self.sandbox_timeout = sandbox_timeout
        self.max_attempts = max_attempts
        self._model = model

    def synthesize(
        self,
        description: str,
        signature: str = None,
        test_cases: list[dict] = None,
        tags: list[str] = None,
        pytest_dir: str = None,
    ) -> ToolRecord:
        """
        Generate, test, and register a new tool. Returns the saved ToolRecord.
        Raises RuntimeError if synthesis fails after max_attempts.

        Args:
            pytest_dir: if provided, writes a pytest-compatible test file to this directory.
        """
        test_cases = test_cases or []
        tags = tags or []
        error_feedback: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            generated = self._generate(description, signature, error_feedback)
            fn_name = self._extract_fn_name(generated)

            error = _run_sandbox(generated.source, fn_name, test_cases, self.sandbox_timeout)
            if error is None:
                now = datetime.now().isoformat()
                record = ToolRecord(
                    name=fn_name,
                    description=generated.description,
                    source=generated.source,
                    signature=signature or self._extract_signature(generated.source, fn_name),
                    created_at=now,
                    updated_at=now,
                    version=1,
                    test_cases=test_cases,
                    tags=tags,
                    synthesis_model=self._model,
                )
                self.registry.save(record)
                if pytest_dir:
                    self._write_pytest_file(record, pytest_dir)
                return record

            error_feedback = error

        raise RuntimeError(
            f"Failed to synthesize tool for '{description}' after {self.max_attempts} attempts. "
            f"Last error: {error_feedback}"
        )

    def _write_pytest_file(self, record: ToolRecord, pytest_dir: str) -> None:
        """Write a pytest-compatible test file for the synthesized tool."""
        os.makedirs(pytest_dir, exist_ok=True)
        lines = [
            "import pytest",
            "from src import ToolRegistry",
            "",
            f"registry = ToolRegistry('{self.registry.registry_dir}')",
            "",
            "@pytest.fixture",
            f"def {record.name}():",
            f"    return registry.load_as_callable('{record.name}')",
            "",
        ]
        for i, tc in enumerate(record.test_cases):
            args_repr = repr(tc["args"])
            if "expected" in tc:
                lines += [
                    f"def test_{record.name}_case_{i}({record.name}):",
                    f"    assert {record.name}(**{args_repr}) == {tc['expected']!r}",
                    "",
                ]
            else:
                lines += [
                    f"def test_{record.name}_case_{i}_no_exception({record.name}):",
                    f"    {record.name}(**{args_repr})  # should not raise",
                    "",
                ]
        path = os.path.join(pytest_dir, f"test_{record.name}.py")
        with open(path, "w") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate(self, description: str, signature: str | None, error_feedback: str | None) -> _GeneratedTool:
        parts = [f"Create a Python tool function that: {description}"]
        if signature:
            parts.append(f"The function signature must be: def <name>{signature}")
        if error_feedback:
            parts.append(f"\nPrevious attempt failed with this error — fix it:\n{error_feedback}")
        prompt = "\n".join(parts)

        raw = self._llm(prompt, response_type=ResponseType.JSON_SCHEMA, response_basemodel=_GeneratedTool)
        self._llm.reset_history()
        return _GeneratedTool.model_validate_json(raw)

    def _extract_fn_name(self, generated: _GeneratedTool) -> str:
        """Use the LLM-provided name, but verify it appears in the source as a def."""
        name = generated.name
        match = re.search(r'^def\s+(\w+)', generated.source, re.MULTILINE)
        if match:
            source_name = match.group(1)
            if source_name != name:
                name = source_name  # trust the actual source over the name field
        return name

    def _extract_signature(self, source: str, fn_name: str) -> str:
        """Extract the signature string from source as a fallback when caller didn't provide one."""
        match = re.search(rf'def\s+{re.escape(fn_name)}\s*(\([^)]*\)[^:]*)', source)
        return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# ToolOptimizer
# ---------------------------------------------------------------------------

class ToolOptimizer:
    """
    Iteratively improves an existing tool's implementation against test cases.
    Uses an accept/revert loop: only keeps changes that reduce the number of failures.

    Usage:
        opt = ToolOptimizer("groq/llama-3.3-70b-versatile", 0.59, 0.79, registry)
        record = opt.optimize("compound_interest", test_cases=[...], max_rounds=3)
    """

    _SYSTEM = (
        "You are an expert Python programmer improving an existing tool function.\n"
        "You will receive the current implementation and a list of failing test cases with error messages.\n"
        "Rewrite the implementation to fix all failures.\n"
        "Rules:\n"
        "1. Preserve the function name and signature exactly\n"
        "2. Keep any import statements that are correct; add new ones as needed (before the def block)\n"
        "3. Do not add network calls, filesystem writes, or external API calls\n"
        "Return ONLY the complete updated Python source string (imports + def block). No explanation."
    )

    def __init__(
        self,
        model: str,
        input_cpm: float,
        output_cpm: float,
        registry: ToolRegistry,
        sandbox_timeout: int = 5,
    ):
        self._llm = LLM_API(
            model_name=model,
            system_prompt=self._SYSTEM,
            input_token_cpm=input_cpm,
            output_token_cpm=output_cpm,
        )
        self.registry = registry
        self.sandbox_timeout = sandbox_timeout

    def optimize(
        self,
        name: str,
        test_cases: list[dict],
        max_rounds: int = 3,
    ) -> ToolRecord:
        """
        Improve tool `name` to pass more of the given test_cases.
        Returns the (possibly updated) ToolRecord.
        """
        record = self.registry.load(name)
        failures = self._run_tests(record.source, record.name, test_cases)

        for _ in range(max_rounds):
            if not failures:
                break
            new_source = self._rewrite(record, failures)
            new_failures = self._run_tests(new_source, record.name, test_cases)

            if len(new_failures) < len(failures):
                record.source = new_source
                record.version += 1
                record.test_cases = test_cases
                self.registry.save(record)
                failures = new_failures

        return record

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_tests(self, source: str, fn_name: str, test_cases: list[dict]) -> list[str]:
        """Run each test case individually and collect failure messages."""
        failures = []
        for i, tc in enumerate(test_cases):
            err = _run_sandbox(source, fn_name, [tc], self.sandbox_timeout)
            if err is not None:
                failures.append(f"Test {i} ({tc}): {err}")
        return failures

    def _rewrite(self, record: ToolRecord, failures: list[str]) -> str:
        failure_block = "\n".join(failures)
        prompt = (
            f"Function name: {record.name}\n"
            f"Description: {record.description}\n\n"
            f"Current source:\n```python\n{record.source}\n```\n\n"
            f"Failing test cases:\n{failure_block}\n\n"
            "Rewrite the function source to fix all failures."
        )
        new_source = self._llm(prompt, response_type=ResponseType.TEXT)
        self._llm.reset_history()
        # Strip markdown code fences if the model wraps the response
        new_source = re.sub(r'^```python\s*\n', '', new_source.strip())
        new_source = re.sub(r'\n```\s*$', '', new_source)
        return new_source.strip()
