from __future__ import annotations

import argparse
from pathlib import Path

from .agent_loop import AgentLoop
from .config import load_config
from .memory_store import MemoryStore
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient
from .render import print_assistant, print_sessions, print_trace
from .trace import TraceWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vib-code")
    parser.add_argument("--cwd", default=".", help="Workspace directory for the session.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="Start a new interactive chat session.")
    chat_parser.add_argument("--prompt", help="Run one prompt non-interactively and exit.")
    chat_parser.add_argument("--title", default="Interactive chat", help="Session title.")

    resume_parser = subparsers.add_parser("resume", help="Resume an existing session.")
    resume_parser.add_argument("session_id", help="Session ID to resume.")
    resume_parser.add_argument("--prompt", help="Run one prompt non-interactively and exit.")

    subparsers.add_parser("sessions", help="List saved sessions.")
    trace_parser = subparsers.add_parser("trace", help="Print a formatted trace for a session.")
    trace_parser.add_argument("session_id", help="Session ID to inspect.")
    trace_parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw JSONL trace instead of the formatted view.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    workspace_dir = Path(args.cwd).resolve()
    config = load_config(base_dir=base_dir, workspace_dir=workspace_dir)
    memory_store = MemoryStore(
        Path(config.storage_dir),
        legacy_storage_dir=base_dir / ".vib-code",
    )
    if config.provider == "openai":
        client = OpenAIClient(config.openai_base_url, config.openai_api_key, config.model)
    else:
        client = OllamaClient(config.ollama_host, config.model)
    loop = AgentLoop(memory_store, client, config)

    if args.command == "sessions":
        print_sessions(memory_store.list_sessions())
        return

    if args.command == "trace":
        session = memory_store.load_session(args.session_id)
        print_trace(TraceWriter(Path(session.trace_path)).read_text(), raw=args.raw)
        return

    if args.command == "chat":
        session = loop.start_session(title=args.title)
        if args.prompt:
            print_assistant(loop.run_turn(session, args.prompt))
            print(session.session_id)
            return
        loop.run_interactive(session)
        return

    if args.command == "resume":
        session = memory_store.load_session(args.session_id)
        loop.resume_session(session)
        if args.prompt:
            print_assistant(loop.run_turn(session, args.prompt))
            return
        loop.run_interactive(session)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
