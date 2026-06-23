"""
MetaOS CLI — Textual TUI for the MetaOS Agent.

Usage:
    python cli.py
    python cli.py --model groq/llama-3.3-70b-versatile
    python cli.py --session sessions/my_session.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Input, RichLog, Static

load_dotenv()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("METAOS_MODEL", "groq/llama-3.3-70b-versatile")
DEFAULT_INPUT_CPM = float(os.getenv("METAOS_INPUT_CPM", "0.59"))
DEFAULT_OUTPUT_CPM = float(os.getenv("METAOS_OUTPUT_CPM", "0.79"))
SESSIONS_DIR = Path("sessions")

HELP_TEXT = """\
**Available commands:**

| Command | Description |
|---------|-------------|
| `/reset` | Start a new session (keeps tool/workflow registries) |
| `/tree` | Show the current planning tree |
| `/tools` | List all tools in the registry |
| `/workflows` | List all workflows in the registry |
| `/save [name]` | Save this session (default: timestamp) |
| `/load [name]` | Load a saved session |
| `/cost` | Show API usage and cost breakdown |
| `/help` | Show this message |

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `Ctrl+S` | Save current session |
| `Ctrl+R` | Reset session |
| `Ctrl+C` | Quit |
"""

# ---------------------------------------------------------------------------
# Message rendering helpers
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)


def _split_content(content: str) -> list:
    """Split agent response into a list of (type, value) segments for rendering."""
    segments = []
    last = 0
    for m in _CODE_BLOCK.finditer(content):
        if m.start() > last:
            segments.append(("text", content[last:m.start()].strip()))
        lang = m.group(1) or "text"
        code = m.group(2)
        segments.append(("code", (lang, code)))
        last = m.end()
    if last < len(content):
        tail = content[last:].strip()
        if tail:
            segments.append(("text", tail))
    return segments


# ---------------------------------------------------------------------------
# Textual App
# ---------------------------------------------------------------------------

CSS = """
Screen {
    layout: horizontal;
}

#chat-panel {
    width: 2fr;
    height: 100%;
    padding: 0 1;
}

#sidebar {
    width: 1fr;
    height: 100%;
    border-left: solid $accent-darken-2;
    padding: 0 1;
    overflow-y: auto;
}

#sidebar Static {
    margin-bottom: 1;
}

#input {
    dock: bottom;
    height: 3;
    border-top: solid $accent-darken-2;
}

Header {
    background: $accent-darken-3;
}
"""


class MetaOSTUI(App):
    CSS = CSS
    TITLE = "MetaOS"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+r", "cmd_reset", "New session", show=True),
        Binding("ctrl+s", "cmd_save", "Save session", show=True),
    ]

    def __init__(self, agent, model_name: str, initial_session: Path | None = None):
        super().__init__()
        self.agent = agent
        self.model_name = model_name
        self._ui_messages: list[dict] = []   # {role, content, timestamp}
        self._initial_session = initial_session
        self._busy = False

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            with VerticalScroll(id="chat-panel"):
                yield RichLog(id="chat-log", markup=True, highlight=True, wrap=True)
            with VerticalScroll(id="sidebar"):
                yield Static(id="tree-view", markup=True)
                yield Static(id="tools-view", markup=True)
                yield Static(id="workflows-view", markup=True)
        yield Input(placeholder="Type a message or /help for commands…", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self._update_header()
        self._update_sidebar()
        log = self.query_one("#chat-log", RichLog)
        log.write(Text.from_markup(
            f"[bold green]MetaOS[/bold green] [dim]— {self.model_name}[/dim]\n"
            "Type a goal to get started, or [bold]/help[/bold] for commands.\n"
        ))
        if self._initial_session:
            self._load_session(self._initial_session)
        else:
            self._offer_resume()
        self.query_one("#input", Input).focus()

    # ------------------------------------------------------------------
    # Input handler
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return
        self.query_one("#input", Input).clear()

        if message.startswith("/"):
            self._handle_slash(message)
        else:
            self._send_message(message)

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _handle_slash(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=1)
        name = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        log = self.query_one("#chat-log", RichLog)

        if name == "/help":
            log.write(Markdown(HELP_TEXT))

        elif name == "/reset":
            self.action_cmd_reset()

        elif name == "/tree":
            if self.agent.tree is None:
                log.write(Text.from_markup("[dim]No planning tree yet.[/dim]\n"))
            else:
                log.write(Text.from_markup(f"[bold]Planning Tree[/bold]\n"))
                log.write(Text(self.agent.tree.get_subtree_text()))
                log.write("")

        elif name == "/tools":
            tools = self.agent.tool_registry.list_tools()
            if not tools:
                log.write(Text.from_markup("[dim]No tools in registry yet.[/dim]\n"))
            else:
                t = Table("Name", "Signature", "Description", title="Tool Registry")
                for tool in tools:
                    t.add_row(tool["name"], tool.get("signature", ""), tool.get("description", ""))
                log.write(t)
                log.write("")

        elif name == "/workflows":
            wflows = self.agent.workflow_registry.list_workflows()
            if not wflows:
                log.write(Text.from_markup("[dim]No workflows in registry yet.[/dim]\n"))
            else:
                t = Table("Name", "Description", title="Workflow Registry")
                for w in wflows:
                    t.add_row(w["name"], w.get("description", ""))
                log.write(t)
                log.write("")

        elif name == "/save":
            self.action_cmd_save(arg.strip() or None)

        elif name == "/load":
            self._cmd_load(arg.strip() or None)

        elif name == "/cost":
            m = self.agent.metrics
            log.write(Text.from_markup(
                f"[bold]Session Cost[/bold]\n"
                f"  Total cost:   [green]${m['total_cost']:.6f}[/green]\n"
                f"  API calls:    {m['total_calls']}\n"
                f"  Tool calls:   {m['total_tool_calls']}\n"
            ))

        else:
            log.write(Text.from_markup(f"[red]Unknown command:[/red] {name} — type [bold]/help[/bold]\n"))

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _send_message(self, message: str) -> None:
        if self._busy:
            return
        self._busy = True
        self.query_one("#input", Input).disabled = True

        log = self.query_one("#chat-log", RichLog)
        log.write(Text.from_markup(f"[bold cyan]You ›[/bold cyan] {message}\n"))
        self._ui_messages.append({"role": "user", "content": message,
                                   "timestamp": datetime.now().isoformat()})
        self._run_chat(message)

    @work(thread=True)
    def _run_chat(self, message: str) -> None:
        response = self.agent.chat(message)
        self.call_from_thread(self._on_response, response)

    def _show_tool_call(self, tool_name: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(Text.from_markup(f"  [dim]→ {tool_name}[/dim]"))

    def _on_response(self, response: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(Text.from_markup("[bold green]MetaOS ›[/bold green]"))

        segments = _split_content(response)
        if not segments:
            log.write(Text(response))
        for kind, value in segments:
            if kind == "text" and value:
                log.write(Markdown(value))
            elif kind == "code":
                lang, code = value
                log.write(Syntax(code, lang if lang != "text" else "python",
                                 theme="monokai", line_numbers=False))
        log.write("")

        self._ui_messages.append({"role": "assistant", "content": response,
                                   "timestamp": datetime.now().isoformat()})
        self._update_header()
        self._update_sidebar()
        self.query_one("#input", Input).disabled = False
        self._busy = False
        self.query_one("#input", Input).focus()

    # ------------------------------------------------------------------
    # Actions (keybindings)
    # ------------------------------------------------------------------

    def action_cmd_reset(self) -> None:
        self.agent.reset_session()
        self._ui_messages = []
        log = self.query_one("#chat-log", RichLog)
        log.clear()
        log.write(Text.from_markup(
            "[bold green]MetaOS ›[/bold green] Session reset. Start with a new goal.\n"
        ))
        self._update_header()
        self._update_sidebar()
        self.query_one("#input", Input).disabled = False
        self._busy = False

    def action_cmd_save(self, name: str | None = None) -> None:
        SESSIONS_DIR.mkdir(exist_ok=True)
        filename = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        if not filename.endswith(".json"):
            filename += ".json"
        path = SESSIONS_DIR / filename
        session = {
            "version": 1,
            "saved_at": datetime.now().isoformat(),
            "model": self.model_name,
            "ui_messages": self._ui_messages,
            "llm_history": self.agent._llm.history,
            "metrics": self.agent.metrics,
            "tree_path": self.agent._tree_path,
        }
        with open(path, "w") as f:
            json.dump(session, f, indent=2, default=str)
        log = self.query_one("#chat-log", RichLog)
        log.write(Text.from_markup(f"[dim]Session saved → {path}[/dim]\n"))

    def _cmd_load(self, name: str | None) -> None:
        log = self.query_one("#chat-log", RichLog)
        if name is None:
            # Show available sessions
            files = sorted(SESSIONS_DIR.glob("*.json")) if SESSIONS_DIR.exists() else []
            if not files:
                log.write(Text.from_markup("[dim]No saved sessions found in sessions/[/dim]\n"))
                return
            log.write(Text.from_markup("[bold]Saved sessions:[/bold]"))
            for f in files:
                log.write(Text.from_markup(f"  [cyan]{f.stem}[/cyan]"))
            log.write(Text.from_markup("[dim]Use /load <name> to load one.[/dim]\n"))
            return

        filename = name if name.endswith(".json") else name + ".json"
        path = SESSIONS_DIR / filename
        if not path.exists():
            log.write(Text.from_markup(f"[red]Session not found:[/red] {path}\n"))
            return
        self._load_session(path)

    def _load_session(self, path: Path) -> None:
        log = self.query_one("#chat-log", RichLog)
        try:
            with open(path) as f:
                session = json.load(f)
        except Exception as e:
            log.write(Text.from_markup(f"[red]Failed to load session:[/red] {e}\n"))
            return

        # Restore LLM history
        self.agent._llm.history = session.get("llm_history", [])
        self._ui_messages = session.get("ui_messages", [])

        # Reload tree if it was saved
        tree_path = session.get("tree_path")
        if tree_path and Path(tree_path).exists():
            from src.planner import TodoTree
            self.agent.tree = TodoTree.from_json(tree_path)

        # Replay chat display
        log.clear()
        log.write(Text.from_markup(f"[dim]Session loaded from {path}[/dim]\n"))
        for msg in self._ui_messages:
            if msg["role"] == "user":
                log.write(Text.from_markup(f"[bold cyan]You ›[/bold cyan] {msg['content']}\n"))
            else:
                log.write(Text.from_markup("[bold green]MetaOS ›[/bold green]"))
                for kind, value in _split_content(msg["content"]):
                    if kind == "text" and value:
                        log.write(Markdown(value))
                    elif kind == "code":
                        lang, code = value
                        log.write(Syntax(code, lang if lang != "text" else "python",
                                         theme="monokai", line_numbers=False))
                log.write("")

        self._update_header()
        self._update_sidebar()

    def _offer_resume(self) -> None:
        """If sessions exist, print a hint to resume the most recent one."""
        if not SESSIONS_DIR.exists():
            return
        files = sorted(SESSIONS_DIR.glob("*.json"))
        if not files:
            return
        latest = files[-1]
        log = self.query_one("#chat-log", RichLog)
        log.write(Text.from_markup(
            f"[dim]Previous session found: [cyan]{latest.stem}[/cyan] "
            f"— type [bold]/load {latest.stem}[/bold] to resume it.[/dim]\n"
        ))

    # ------------------------------------------------------------------
    # Sidebar + header updates
    # ------------------------------------------------------------------

    def _update_header(self) -> None:
        m = self.agent.metrics
        self.sub_title = f"${m['total_cost']:.5f} • {m['total_calls']} calls"

    def _update_sidebar(self) -> None:
        # Tree
        tree_widget = self.query_one("#tree-view", Static)
        if self.agent.tree is not None:
            tree_text = self.agent.tree.get_subtree_text()
            tree_widget.update(
                Text.from_markup(f"[bold]Planning Tree[/bold]\n[dim]────────────[/dim]\n")
                .__add__(Text(tree_text))
            )
        else:
            tree_widget.update(Text.from_markup("[bold]Planning Tree[/bold]\n[dim]No plan yet[/dim]"))

        # Tools
        tools_widget = self.query_one("#tools-view", Static)
        tools = self.agent.tool_registry.list_tools()
        if tools:
            lines = [Text.from_markup(f"[bold]Tools ({len(tools)})[/bold]")]
            for t in tools:
                lines.append(Text.from_markup(
                    f"  [cyan]{t['name']}[/cyan][dim]{t.get('signature', '')}[/dim]"
                ))
            tools_widget.update(Text("\n").join(lines))
        else:
            tools_widget.update(Text.from_markup("[bold]Tools[/bold]\n[dim]None yet[/dim]"))

        # Workflows
        wf_widget = self.query_one("#workflows-view", Static)
        workflows = self.agent.workflow_registry.list_workflows()
        if workflows:
            lines = [Text.from_markup(f"[bold]Workflows ({len(workflows)})[/bold]")]
            for w in workflows:
                lines.append(Text.from_markup(f"  [green]{w['name']}[/green]"))
            wf_widget.update(Text("\n").join(lines))
        else:
            wf_widget.update(Text.from_markup("[bold]Workflows[/bold]\n[dim]None yet[/dim]"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MetaOS Agent — conversational AI system builder")
    p.add_argument("--model", default=DEFAULT_MODEL, help="LiteLLM model name")
    p.add_argument("--input-cpm", type=float, default=DEFAULT_INPUT_CPM)
    p.add_argument("--output-cpm", type=float, default=DEFAULT_OUTPUT_CPM)
    p.add_argument("--session", default=None, help="Path to a saved session JSON to load on startup")
    p.add_argument("--tools-dir", default="tools/")
    p.add_argument("--workflows-dir", default="workflows/")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Import here so startup is fast even if src has import errors
    try:
        from src import MetaOSAgent
    except ImportError as e:
        print(f"Failed to import MetaOS: {e}", file=sys.stderr)
        sys.exit(1)

    initial_session = Path(args.session) if args.session else None

    # Create agent with tool-call callback (set later once app is created)
    agent = MetaOSAgent(
        model=args.model,
        input_cpm=args.input_cpm,
        output_cpm=args.output_cpm,
        tool_registry_dir=args.tools_dir,
        workflow_registry_dir=args.workflows_dir,
    )

    app = MetaOSTUI(agent=agent, model_name=args.model, initial_session=initial_session)

    # Wire the tool-call callback now that we have both the agent and app
    agent.on_tool_call = lambda name: app.call_from_thread(app._show_tool_call, name)

    app.run()


if __name__ == "__main__":
    main()
