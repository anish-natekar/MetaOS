"""Unit tests for cli.py helper functions — no LLM calls required."""
import sys
import os
import pytest

# cli.py is at the repo root — ensure it's importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cli import _split_content


def test_plain_text_no_fences():
    segments = _split_content("Hello world")
    assert len(segments) == 1
    kind, value = segments[0]
    assert kind == "text"
    assert value == "Hello world"


def test_single_code_block():
    content = "Here is code:\n```python\nprint('hi')\n```\nDone."
    segments = _split_content(content)
    kinds = [s[0] for s in segments]
    assert "code" in kinds
    code_seg = next(s for s in segments if s[0] == "code")
    lang, code = code_seg[1]
    assert lang == "python"
    assert "print" in code


def test_code_block_no_lang():
    content = "```\nsome code\n```"
    segments = _split_content(content)
    code_seg = next(s for s in segments if s[0] == "code")
    lang, code = code_seg[1]
    assert lang == "text"
    assert "some code" in code


def test_text_before_and_after_code():
    content = "Before\n```json\n{}\n```\nAfter"
    segments = _split_content(content)
    kinds = [s[0] for s in segments]
    assert kinds.count("text") == 2
    assert kinds.count("code") == 1


def test_multiple_code_blocks():
    content = "Intro\n```python\nx=1\n```\nMiddle\n```bash\nls\n```\nEnd"
    segments = _split_content(content)
    code_segs = [s for s in segments if s[0] == "code"]
    assert len(code_segs) == 2
    assert code_segs[0][1][0] == "python"
    assert code_segs[1][1][0] == "bash"


def test_empty_string():
    segments = _split_content("")
    assert segments == []


def test_only_whitespace():
    segments = _split_content("   \n   ")
    # All whitespace strips to nothing — no segments
    assert all(s[1] == "" or s[1].strip() == "" for s in segments if s[0] == "text")


def test_code_block_only():
    content = "```python\ndef foo():\n    pass\n```"
    segments = _split_content(content)
    assert any(s[0] == "code" for s in segments)
    assert not any(s[0] == "text" and s[1] for s in segments)
