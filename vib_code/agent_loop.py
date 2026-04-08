from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .approvals import confirm_action
from .action_parser import ActionParseError, parse_action
from .action_schema import action_schema, blocked_action_schema
from .context_builder import build_chat_messages
from .memory_store import MemoryStore
from .model_client import ModelClient
from .models import ChatMessage, SessionRecord
from .patch_engine import PatchEngine
from .render import (
    print_agent_event,
    print_approval_preview,
    print_assistant,
    print_banner,
    print_sessions,
    print_tool_output,
    print_user_prompt,
)
from .trace import TraceWriter
from .tool_executor import execute_action


SUMMARY_REFRESH_EVERY_USER_TURNS = 7
MAX_STEPS_PER_TURN = 50
RUNTIME_TOOL_CONTEXT_LIMIT = 8


class AgentLoop:
    def __init__(
        self,
        memory_store: MemoryStore,
        client: ModelClient,
        config,
    ) -> None:
        self.memory_store = memory_store
        self.client = client
        self.config = config
        self.patch_engine = PatchEngine()

    def start_session(self, title: str = "Interactive chat") -> SessionRecord:
        session = self.memory_store.create_session(
            model=self.config.model,
            working_directory=self.config.workspace_dir,
            title=title,
        )
        self._trace(session).append_session_event(session, "session_started")
        return session

    def resume_session(self, session: SessionRecord) -> None:
        self._trace(session).append_session_event(session, "session_resumed")

    def run_interactive(self, session: SessionRecord) -> None:
        print_banner(session)
        while True:
            try:
                user_input = print_user_prompt()
            except KeyboardInterrupt:
                print()
                break
            if not user_input:
                continue
            if user_input in {"/exit", "/quit"}:
                break
            if user_input == "/sessions":
                print_sessions(self.memory_store.list_sessions())
                continue
            if user_input == "/summary":
                print(session.summary or "No summary yet.")
                continue
            if user_input == "/share":
                share_path = self._write_share_file(session)
                print(f"Saved share JSON to {share_path}")
                continue

            reply = self.run_turn(session, user_input)
            print_assistant(reply)

    def run_turn(self, session: SessionRecord, user_input: str) -> str:
        session.messages.append(ChatMessage(role="user", content=user_input))
        trace = self._trace(session)
        max_steps = MAX_STEPS_PER_TURN
        turn_goal = _analyze_turn_goal(user_input)
        previous_tool_signature: tuple[str, tuple[tuple[str, object], ...], str] | None = None
        tool_transcript: list[dict[str, Any]] = []
        runtime_state: dict[str, object] | None = None

        for step in range(max_steps):
            print_agent_event(f"step {step + 1}: sending request to model")
            blocked_mode = runtime_state is not None and runtime_state.get("phase") == "stalled_after_repeated_failure"
            schema = (
                blocked_action_schema(provider=self.config.provider)
                if blocked_mode
                else action_schema(provider=self.config.provider)
            )
            messages = build_chat_messages(
                self.config,
                session,
                runtime_state=runtime_state,
                blocked_mode=blocked_mode,
            )
            request_payload = self.client.build_chat_payload(messages, format_schema=schema)
            trace.append(
                session.session_id,
                "model_request",
                {"request": request_payload, "step": step + 1},
            )
            try:
                result = self.client.chat(messages, format_schema=schema)
            except RuntimeError as exc:
                trace.append(
                    session.session_id,
                    "model_error",
                    {
                        "request": request_payload,
                        "error": str(exc),
                        "step": step + 1,
                    },
                )
                raise
            trace.append(
                session.session_id,
                "model_response",
                {
                    "request": result.request_payload,
                    "response": result.response_body,
                    "step": step + 1,
                },
            )

            try:
                action = parse_action(result.content)
            except ActionParseError as exc:
                print_agent_event(f"step {step + 1}: model returned invalid action: {exc}")
                trace.append(
                    session.session_id,
                    "tool_result",
                    {
                        "action": "parse_error",
                        "ok": False,
                        "summary": str(exc),
                        "step": step + 1,
                    },
                )
                runtime_state = {
                    "phase": "retry_after_invalid_action",
                    "original_user_request": user_input,
                    "validation_error": str(exc),
                    "must_return_schema_valid_action": True,
                }
                continue

            if action.action == "final_answer":
                print_agent_event(f"step {step + 1}: model returned final_answer")
                message = str(action.args["message"])
                session.messages.append(ChatMessage(role="assistant", content=message))
                self._update_summary(session, user_input, message, tool_transcript)
                self.memory_store.save_session(session)
                return message

            if action.action == "ask_user":
                print_agent_event(f"step {step + 1}: model returned ask_user")
                question = str(action.args["question"]).strip()
                context = str(action.args.get("context", "")).strip()
                message = question if not context else f"{context}\n\n{question}"
                session.messages.append(ChatMessage(role="assistant", content=message))
                self._update_summary(session, user_input, message, tool_transcript)
                self.memory_store.save_session(session)
                return message

            print_agent_event(
                f"step {step + 1}: model requested tool_call {action.action} "
                f"with args={action.args}"
            )
            trace.append(
                session.session_id,
                "tool_request",
                {
                    "action": action.action,
                    "args": action.args,
                    "step": step + 1,
                },
            )
            if action.action in {"create_file", "replace_in_file", "write_patch", "run_command"}:
                preview_result = execute_action(
                    action,
                    self.config.workspace_dir,
                    preview_only=True,
                    patch_engine=self.patch_engine,
                )
                if preview_result.ok:
                    trace.append(
                        session.session_id,
                        "approval_requested",
                        {
                            "action": action.action,
                            "summary": preview_result.summary,
                            "step": step + 1,
                        },
                    )
                    print_agent_event(f"step {step + 1}: approval requested for {action.action}")
                    print_approval_preview(action.action, preview_result.stdout)
                    approved = confirm_action(action, preview_result.stdout)
                    trace.append(
                        session.session_id,
                        "approval_decision",
                        {
                            "action": action.action,
                            "approved": approved,
                            "summary": preview_result.summary,
                            "step": step + 1,
                        },
                    )
                    if approved:
                        tool_result = execute_action(
                            action,
                            self.config.workspace_dir,
                            patch_engine=self.patch_engine,
                        )
                    else:
                        tool_result = _approval_denied_result(action.action)
                else:
                    tool_result = preview_result
            else:
                tool_result = execute_action(action, self.config.workspace_dir)
            print_agent_event(
                f"step {step + 1}: tool {action.action} completed "
                f"ok={tool_result.ok} summary={tool_result.summary}"
            )
            print_tool_output(
                action.action,
                stdout=tool_result.stdout,
                stderr=tool_result.stderr,
                max_lines=None if action.action == "read_file" else 5,
            )
            trace.append(
                session.session_id,
                "tool_result",
                {
                    "action": action.action,
                    "ok": tool_result.ok,
                    "summary": tool_result.summary,
                    "stdout": tool_result.stdout,
                    "stderr": tool_result.stderr,
                    "exit_code": tool_result.exit_code,
                    "metadata": tool_result.metadata,
                    "step": step + 1,
                },
            )
            current_signature = (
                action.action,
                tuple(sorted(action.args.items())),
                str(tool_result.metadata.get("normalized_failure", "")) or tool_result.stdout,
            )
            latest_file_context = _latest_successful_read_result(
                tool_transcript,
                str(action.args.get("path", "")),
            )
            normalized_tool_result = _normalize_tool_result(tool_result)
            tool_transcript.append(
                {
                    "action": action.action,
                    "args": dict(action.args),
                    "result": normalized_tool_result,
                }
            )
            recent_tool_results = _runtime_tool_context(tool_transcript, limit=RUNTIME_TOOL_CONTEXT_LIMIT)
            read_progress = _build_read_progress(tool_transcript)
            repeat_pattern = _detect_repeat_pattern(tool_transcript)
            phase = "post_read_file_result" if action.action == "read_file" and tool_result.ok else "post_tool_result"
            if (
                action.action in {"write_patch", "replace_in_file"}
                and not tool_result.ok
                and latest_file_context is not None
            ):
                phase = "retry_after_edit_error"
            runtime_state = {
                "phase": phase,
                "original_user_request": user_input,
                "turn_goal": turn_goal,
                "last_action": {
                    "action": action.action,
                    "args": dict(action.args),
                },
                "last_tool_result": normalized_tool_result,
                "recent_tool_results": recent_tool_results,
                "read_progress": read_progress,
                "repeat_pattern": repeat_pattern,
                "decision_rule": {
                    "may_answer_now": tool_result.ok,
                    "repeat_same_tool_call_forbidden": previous_tool_signature == current_signature,
                    "choose_final_answer_if_sufficient": not (
                        turn_goal["multi_file_read_requested"]
                        and read_progress.get("remaining_files_count", 0)
                    ),
                },
            }
            if phase == "retry_after_edit_error":
                runtime_state["latest_file_context"] = latest_file_context
                runtime_state["decision_rule"] = {
                    "prefer_replace_in_file": True,
                    "must_not_repeat_last_action": True,
                }
            if repeat_pattern.get("detected"):
                runtime_state["decision_rule"]["repeat_detected"] = True
                if repeat_pattern.get("kind") in {"alternating_cycle", "same_file_reread"}:
                    runtime_state["decision_rule"]["must_choose_unread_file"] = True
                    runtime_state["decision_rule"]["must_not_repeat_last_action"] = True
            if (
                turn_goal["multi_file_read_requested"]
                and read_progress.get("remaining_files_count", 0)
            ):
                runtime_state["decision_rule"]["must_continue_reading"] = True
                if read_progress.get("next_unread_file"):
                    runtime_state["decision_rule"]["suggested_next_file"] = read_progress["next_unread_file"]
                if read_progress.get("read_files"):
                    runtime_state["decision_rule"]["must_not_reread_files"] = read_progress["read_files"]
            if previous_tool_signature == current_signature:
                runtime_state["decision_rule"]["repeat_detected"] = True
                stall_reason: str | None = None
                if tool_result.ok:
                    stall_reason = "repeated_no_progress_success"
                elif tool_result.metadata.get("retryable") is False:
                    stall_reason = "repeated_non_retryable_failure"

                if stall_reason is not None:
                    if (
                        stall_reason == "repeated_no_progress_success"
                        and read_progress.get("remaining_files_count", 0)
                        and turn_goal["multi_file_read_requested"]
                    ):
                        trace.append(
                            session.session_id,
                            "loop_stalled",
                            {
                                "step": step + 1,
                                "reason": stall_reason,
                                "action": action.action,
                                "args": dict(action.args),
                                "normalized_failure": tool_result.metadata.get("normalized_failure"),
                            },
                        )
                        runtime_state["decision_rule"]["must_not_repeat_last_action"] = True
                        runtime_state["decision_rule"]["must_choose_unread_file"] = True
                        runtime_state["decision_rule"]["choose_final_answer_if_sufficient"] = False
                        print_agent_event(
                            "stalled loop detected: forcing next action toward an unread file"
                        )
                        previous_tool_signature = current_signature
                        continue
                    allowed_actions = ["final_answer", "ask_user"]
                    blocked_phase = "stalled_after_repeated_failure"
                    if (
                        stall_reason == "repeated_no_progress_success"
                        and action.action == "read_file"
                    ):
                        allowed_actions = ["final_answer"]
                        blocked_phase = "post_read_file_result"
                    runtime_state = {
                        "phase": blocked_phase,
                        "original_user_request": user_input,
                        "turn_goal": turn_goal,
                        "last_action": {
                            "action": action.action,
                            "args": dict(action.args),
                        },
                        "last_tool_result": normalized_tool_result,
                        "recent_tool_results": recent_tool_results,
                        "read_progress": read_progress,
                        "repeat_pattern": repeat_pattern,
                        "stall_reason": stall_reason,
                        "decision_rule": {
                            "allowed_actions": allowed_actions,
                            "must_not_repeat_last_action": True,
                        },
                    }
                    if (
                        turn_goal["multi_file_read_requested"]
                        and read_progress.get("remaining_files_count", 0)
                    ):
                        runtime_state["decision_rule"]["must_continue_reading"] = True
                        runtime_state["decision_rule"]["choose_final_answer_if_sufficient"] = False
                        runtime_state["decision_rule"]["suggested_next_file"] = read_progress.get("next_unread_file", "")
                        runtime_state["decision_rule"]["must_not_reread_files"] = read_progress.get("read_files", [])
                    trace.append(
                        session.session_id,
                        "loop_stalled",
                        {
                            "step": step + 1,
                            "reason": stall_reason,
                            "action": action.action,
                            "args": dict(action.args),
                            "normalized_failure": tool_result.metadata.get("normalized_failure"),
                        },
                    )
                    print_agent_event(
                        "stalled loop detected: narrowing next action to final_answer or ask_user"
                    )
            previous_tool_signature = current_signature

        print_agent_event("step limit reached: switching to recovery answer from gathered tool results")
        fallback = self._recover_with_tool_context(session, user_input, tool_transcript)
        session.messages.append(ChatMessage(role="assistant", content=fallback))
        self._update_summary(session, user_input, fallback, tool_transcript)
        self.memory_store.save_session(session)
        return fallback

    def _update_summary(
        self,
        session: SessionRecord,
        user_input: str,
        assistant_output: str,
        tool_transcript: list[dict[str, Any]],
    ) -> None:
        user_turn_count = sum(1 for message in session.messages if message.role == "user")
        if user_turn_count % SUMMARY_REFRESH_EVERY_USER_TURNS != 0:
            return

        older_messages = session.messages[:-6]
        if not session.summary and len(older_messages) < 4 and not tool_transcript:
            return

        transcript_lines = []
        for message in older_messages[-8:]:
            content = " ".join(message.content.split())
            transcript_lines.append(f"{message.role}: {content[:240]}")

        payload = {
            "existing_summary": session.summary,
            "older_messages": transcript_lines,
            "latest_turn": {
                "user": user_input,
                "assistant": assistant_output,
                "tool_results": tool_transcript[-4:],
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the session state for vib-code.\n"
                    "Write a compact factual summary for future turns.\n"
                    "Use exactly these section headers when they have content:\n"
                    "Important Paths:\nConfirmed Files:\nChanges:\nOpen Questions:\n"
                    "Keep only durable facts, important paths, confirmed file names, user preferences, "
                    "and relevant tool outcomes.\n"
                    "Do not include raw chain-of-thought, repeated failures, or verbose tool output.\n"
                    "Under each header, use short bullet lines starting with '- '.\n"
                    "Omit empty sections.\n"
                    "Important Paths must contain only filesystem paths or directories, never tool names or actions.\n"
                    "Confirmed Files must contain only files or directories confirmed by successful tool results.\n"
                    "Changes must contain only actual edits, writes, created files, deleted files, or config changes.\n"
                    "Do not list read-only inspection, listing, or searching as a change.\n"
                    "If there were no actual edits or writes, omit the Changes section.\n"
                    "Open Questions must contain only unresolved user requests or ambiguities that still matter.\n"
                    "If there are no meaningful open questions, omit that section.\n"
                    "If a fact came from a successful tool result, keep it.\n"
                    "If a prior assistant statement was not tool-confirmed, omit it.\n"
                    "Output only the summary text."
                ),
            },
            {
                "role": "user",
                "content": "Session summary update payload:\n" + _json_dump(payload),
            },
        ]
        try:
            result = self.client.chat(messages)
        except RuntimeError:
            if transcript_lines:
                session.summary = "\n".join(transcript_lines[-4:])
            return

        summary = result.content.strip()
        if summary:
            session.summary = _sanitize_summary(summary, tool_transcript)

    def _trace(self, session: SessionRecord) -> TraceWriter:
        return TraceWriter(Path(session.trace_path))

    def _write_share_file(self, session: SessionRecord) -> Path:
        export = {
            "session_id": session.session_id,
            "model": session.model,
            "working_directory": session.working_directory,
            "system": {
                "system_prompt": self.config.system_prompt,
                "session_summary": session.summary,
            },
            "user_messages": [
                {
                    "created_at": message.created_at,
                    "content": message.content,
                }
                for message in session.messages
                if message.role == "user"
            ],
        }
        share_path = Path(self.config.workspace_dir) / f"vib-code-share-{session.session_id}.json"
        share_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
        return share_path

    def _recover_with_tool_context(
        self,
        session: SessionRecord,
        user_input: str,
        tool_transcript: list[dict[str, Any]],
    ) -> str:
        if not tool_transcript:
            return "I could not complete the request within the current step limit."

        trace = self._trace(session)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are vib-code finishing a turn after tool execution. "
                    "Answer the user directly in plain text using only the provided structured tool results. "
                    "Do not return JSON. Do not ask for more tools."
                ),
            },
            {"role": "user", "content": user_input},
            {
                "role": "system",
                "content": "Available gathered tool results JSON:\n\n" + _json_dump(tool_transcript[-4:]),
            },
        ]
        request_payload = self.client.build_chat_payload(messages)
        trace.append(
            session.session_id,
            "model_request",
            {"request": request_payload, "step": "recovery"},
        )
        print_agent_event("recovery: asking model to synthesize a plain-text answer from tool results")
        try:
            result = self.client.chat(messages)
        except RuntimeError:
            print_agent_event("recovery: model call failed")
            return "I could not complete the request within the current step limit."

        trace.append(
            session.session_id,
            "model_response",
            {
                "request": result.request_payload,
                "response": result.response_body,
                "step": "recovery",
            },
        )
        print_agent_event("recovery: synthesized final answer")
        return result.content


def _normalize_tool_result(tool_result) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": tool_result.ok,
        "summary": tool_result.summary,
        "exit_code": tool_result.exit_code,
        "metadata": tool_result.metadata,
    }
    if tool_result.action_type in {"list_files", "search_text"} and tool_result.stdout:
        result["stdout_lines"] = tool_result.stdout.splitlines()
    elif tool_result.stdout:
        result["stdout"] = tool_result.stdout
    if tool_result.stderr:
        result["stderr"] = tool_result.stderr
    return result


def _runtime_tool_context(
    tool_transcript: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    recent: list[dict[str, Any]] = []
    for item in tool_transcript[-limit:]:
        recent.append(
            {
                "action": item.get("action"),
                "args": item.get("args", {}),
                "result": _compact_runtime_tool_result(item.get("result", {})),
            }
        )
    return recent


def _compact_runtime_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "ok": bool(result.get("ok")),
        "summary": result.get("summary"),
        "exit_code": result.get("exit_code"),
        "metadata": result.get("metadata", {}),
    }
    stdout_lines = result.get("stdout_lines")
    if isinstance(stdout_lines, list):
        compact["stdout_lines"] = stdout_lines
    stdout = result.get("stdout")
    if isinstance(stdout, str) and stdout:
        if len(stdout) > 1200:
            compact["stdout_preview"] = stdout[:1200] + f"\n... truncated {len(stdout) - 1200} more characters"
            compact["stdout_truncated"] = True
        else:
            compact["stdout"] = stdout
    stderr = result.get("stderr")
    if isinstance(stderr, str) and stderr:
        if len(stderr) > 800:
            compact["stderr_preview"] = stderr[:800] + f"\n... truncated {len(stderr) - 800} more characters"
            compact["stderr_truncated"] = True
        else:
            compact["stderr"] = stderr
    return compact


def _build_read_progress(tool_transcript: list[dict[str, Any]]) -> dict[str, Any]:
    listed_files: list[str] = []
    listed_path = "."
    for item in reversed(tool_transcript):
        if item.get("action") != "list_files":
            continue
        result = item.get("result", {})
        stdout_lines = result.get("stdout_lines")
        if result.get("ok") and isinstance(stdout_lines, list):
            listed_files = [str(line) for line in stdout_lines]
            listed_path = str(item.get("args", {}).get("path", "."))
            break

    read_files: list[str] = []
    seen: set[str] = set()
    for item in tool_transcript:
        if item.get("action") != "read_file":
            continue
        result = item.get("result", {})
        path = str(item.get("args", {}).get("path", "")).strip()
        if not result.get("ok") or not path or path in seen:
            continue
        seen.add(path)
        read_files.append(path)

    remaining_files = [
        path
        for path in listed_files
        if not path.endswith("/") and path not in seen
    ]
    next_unread_file = _choose_next_unread_file(remaining_files)
    return {
        "listed_path": listed_path,
        "listed_files": listed_files,
        "read_files": read_files,
        "remaining_files": remaining_files,
        "remaining_files_count": len(remaining_files),
        "next_unread_file": next_unread_file,
    }


def _detect_repeat_pattern(tool_transcript: list[dict[str, Any]]) -> dict[str, Any]:
    if len(tool_transcript) < 2:
        return {"detected": False}

    last = tool_transcript[-1]
    previous = tool_transcript[-2]
    last_path = str(last.get("args", {}).get("path", ""))
    if (
        last.get("action") == "read_file"
        and previous.get("action") == "read_file"
        and last_path
        and last_path == str(previous.get("args", {}).get("path", ""))
    ):
        return {
            "detected": True,
            "kind": "same_file_reread",
            "details": f"Repeated read_file on {last_path}",
            "repeated_read_path": last_path,
        }

    if len(tool_transcript) < 4:
        return {"detected": False}

    recent = tool_transcript[-4:]
    signatures = [_tool_history_signature(item) for item in recent]
    if signatures[0] == signatures[2] and signatures[1] == signatures[3] and signatures[0] != signatures[1]:
        details = (
            f"Alternating cycle between {recent[0].get('action')} and {recent[1].get('action')}"
        )
        repeated_read_path = ""
        for item in recent:
            if item.get("action") == "read_file":
                repeated_read_path = str(item.get("args", {}).get("path", ""))
                break
        pattern: dict[str, Any] = {
            "detected": True,
            "kind": "alternating_cycle",
            "details": details,
        }
        if repeated_read_path:
            pattern["repeated_read_path"] = repeated_read_path
        return pattern

    return {"detected": False}


def _choose_next_unread_file(remaining_files: list[str]) -> str:
    for path in remaining_files:
        if path.startswith("__") and path.endswith(".py"):
            continue
        if path.endswith(".json"):
            continue
        return path
    return remaining_files[0] if remaining_files else ""


def _tool_history_signature(item: dict[str, Any]) -> tuple[str, tuple[tuple[str, object], ...], str]:
    action = str(item.get("action", ""))
    args = item.get("args", {})
    result = item.get("result", {})
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    outcome = ""
    if isinstance(metadata, dict):
        outcome = str(metadata.get("normalized_failure", ""))
    if not outcome and isinstance(result, dict):
        outcome = str(result.get("summary", ""))
    return (action, tuple(sorted(args.items())) if isinstance(args, dict) else tuple(), outcome)


def _json_dump(payload: object) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=True)


def _analyze_turn_goal(user_input: str) -> dict[str, bool]:
    lowered = user_input.lower()
    multi_file_read = any(
        phrase in lowered
        for phrase in (
            "read all files",
            "read all the files",
            "read the files",
            "read files",
            "list and read",
        )
    )
    directory_summary = any(
        phrase in lowered
        for phrase in (
            "summarise the directory",
            "summarize the directory",
            "summarise directory",
            "summarize directory",
            "summarise this directory",
            "summarize this directory",
        )
    )
    current_directory = "current directory" in lowered or "directory" in lowered
    return {
        "multi_file_read_requested": multi_file_read or (current_directory and "read" in lowered),
        "directory_summary_requested": directory_summary,
        "directory_listing_requested": current_directory and ("list" in lowered or "files" in lowered),
    }


def _approval_denied_result(action_type: str):
    from .models import ToolResult

    return ToolResult(
        ok=False,
        action_type=action_type,
        summary=f"User denied approval for {action_type}.",
        metadata={
            "error_type": "approval_denied",
            "retryable": True,
            "normalized_failure": "approval_denied",
        },
    )


def _sanitize_summary(summary: str, tool_transcript: list[dict[str, Any]]) -> str:
    mutating_actions = {"create_file", "replace_in_file", "write_patch", "run_command"}
    has_mutation = any(item.get("action") in mutating_actions for item in tool_transcript)
    if has_mutation:
        return summary

    lines = summary.splitlines()
    kept: list[str] = []
    skip_changes = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Changes:":
            skip_changes = True
            continue
        if skip_changes and stripped.endswith(":") and not stripped.startswith("- "):
            skip_changes = False
        if not skip_changes:
            kept.append(line)

    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept).strip()


def _latest_successful_read_result(
    tool_transcript: list[dict[str, Any]],
    path: str,
) -> dict[str, Any] | None:
    if not path:
        return None
    for item in reversed(tool_transcript):
        if item.get("action") != "read_file":
            continue
        if item.get("args", {}).get("path") != path:
            continue
        result = item.get("result", {})
        if result.get("ok"):
            return result
    return None
