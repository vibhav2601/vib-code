from __future__ import annotations

import json
from typing import Any

from .models import SessionRecord


def print_banner(session: SessionRecord) -> None:
    print(f"vib-code session {session.session_id} | model={session.model}")
    print("Commands: /exit, /sessions, /summary, /share")


def print_assistant(message: str) -> None:
    print(f"\nvib-code> {message}\n")


def print_user_prompt() -> str:
    return input("you> ").strip()


def print_sessions(sessions: list[SessionRecord]) -> None:
    if not sessions:
        print("No saved sessions.")
        return
    for session in sessions:
        print(f"{session.session_id}  {session.updated_at}  {session.title}")


def print_trace(trace_text: str, *, raw: bool = False) -> None:
    output = trace_text if raw else format_trace(trace_text)
    print(output, end="" if output.endswith("\n") else "\n")


def print_agent_event(message: str) -> None:
    print(f"[agent] {message}")


def print_tool_output(action: str, stdout: str = "", stderr: str = "", max_lines: int | None = 5) -> None:
    if stdout:
        preview = stdout if max_lines is None else _truncate_lines(stdout, max_lines=max_lines)
        print(f"[tool:{action}:stdout]\n{preview}")
    if stderr:
        preview = stderr if max_lines is None else _truncate_lines(stderr, max_lines=max_lines)
        print(f"[tool:{action}:stderr]\n{preview}")


def print_approval_preview(action: str, preview: str, max_chars: int = 1200) -> None:
    if not preview:
        return
    rendered = _truncate(preview, max_chars=max_chars)
    print(f"[approval:{action}:preview]\n{rendered}")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    remaining = len(text) - max_chars
    return text[:max_chars] + f"\n... truncated {remaining} more characters"


def _truncate_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    remaining = len(lines) - max_lines
    kept = "\n".join(lines[:max_lines])
    return kept + f"\n... truncated {remaining} more lines"


def format_trace(trace_text: str) -> str:
    sections: list[str] = []
    for line_number, line in enumerate(trace_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError as exc:
            sections.append(f"[invalid trace line {line_number}] {exc}: {stripped}")
            continue
        sections.append(_format_trace_event(event))
    if not sections:
        return "No trace events.\n"
    return "\n\n".join(sections) + "\n"


def _format_trace_event(event: dict[str, Any]) -> str:
    timestamp = str(event.get("timestamp", "?"))
    event_type = str(event.get("event_type", "unknown"))
    payload = event.get("payload")
    if not isinstance(payload, dict):
        payload = {}

    header_parts = [timestamp, event_type]
    step = payload.get("step")
    if step is not None:
        header_parts.append(f"step {step}")
    lines = [" | ".join(header_parts)]

    if event_type in {"session_started", "session_resumed"}:
        lines.extend(_format_key_value("title", payload.get("title")))
        lines.extend(_format_key_value("model", payload.get("model")))
        lines.extend(_format_key_value("working_directory", payload.get("working_directory")))
        return "\n".join(lines)

    if event_type == "model_request":
        request = payload.get("request")
        if isinstance(request, dict):
            lines.extend(_format_model_request(request))
        return "\n".join(lines)

    if event_type == "model_response":
        response = payload.get("response")
        if isinstance(response, dict):
            lines.extend(_format_model_response(response))
        return "\n".join(lines)

    if event_type == "tool_request":
        lines.extend(_format_tool_call(payload.get("action"), payload.get("args")))
        return "\n".join(lines)

    if event_type == "tool_result":
        lines.extend(_format_tool_result(payload))
        return "\n".join(lines)

    if event_type == "model_error":
        lines.extend(_format_key_value("error", payload.get("error")))
        request = payload.get("request")
        if isinstance(request, dict):
            lines.extend(_format_key_value("model", request.get("model")))
        return "\n".join(lines)

    if event_type == "loop_stalled":
        lines.extend(_format_loop_stalled(payload))
        return "\n".join(lines)

    if event_type in {"approval_requested", "approval_decision"}:
        lines.extend(_format_approval(payload))
        return "\n".join(lines)

    lines.append("  payload:")
    lines.extend(_indent_lines(_json_dump(payload).splitlines(), indent="    "))
    return "\n".join(lines)


def _format_chat_message(message: Any) -> list[str]:
    if not isinstance(message, dict):
        return [f"    ? {message!r}"]
    role = str(message.get("role", "?"))
    content = _format_message_content(message)
    content_lines = content.splitlines() or [""]
    lines = [f"    {role}: {content_lines[0]}"]
    lines.extend(_indent_lines(content_lines[1:], indent="      "))
    return lines


def _format_message_content(message: dict[str, Any]) -> str:
    role = str(message.get("role", "?"))
    if role == "system":
        return "<system_prompt>"
    content = message.get("content", "")
    if isinstance(content, str):
        return _prettify_json_string(content)
    return _json_dump(content)


def _format_model_request(request: dict[str, Any]) -> list[str]:
    lines = []
    lines.extend(_format_key_value("model", request.get("model")))

    messages = request.get("messages")
    parsed_runtime_state = None
    parsed_phase_instruction = None
    conversation: list[dict[str, str]] = []
    session_summary = None
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role != "system":
                if isinstance(content, str):
                    conversation.append({"role": str(role), "content": content})
                continue
            if not isinstance(content, str):
                continue
            if content.startswith("Runtime state JSON:\n"):
                parsed_runtime_state = _parse_runtime_state(content)
            elif content.startswith("Phase:"):
                parsed_phase_instruction = content
            elif content.startswith("Session summary:\n"):
                session_summary = content.removeprefix("Session summary:\n").strip()

    phase = None
    if isinstance(parsed_runtime_state, dict):
        phase = parsed_runtime_state.get("phase")
    if phase is None and parsed_phase_instruction:
        phase = parsed_phase_instruction.splitlines()[0].removeprefix("Phase: ").strip().rstrip(".")
    if phase:
        lines.append(f"  phase: {phase}")

    allowed_actions = _extract_allowed_actions(request.get("format"))
    if allowed_actions:
        lines.append(f"  allowed_actions: {', '.join(allowed_actions)}")

    if session_summary:
        lines.extend(_format_multiline_block("session_summary", session_summary))

    if conversation:
        lines.append("  conversation:")
        for message in conversation[-4:]:
            role = message["role"]
            content = _conversation_preview(message["content"])
            lines.append(f"    {role}: {content}")

    latest_user = _latest_conversation_message(conversation, "user")
    if latest_user is not None:
        lines.extend(_format_key_value("latest_user", latest_user))

    if isinstance(parsed_runtime_state, dict):
        lines.extend(_format_runtime_state_summary(parsed_runtime_state))

    return lines


def _format_model_response(response: dict[str, Any]) -> list[str]:
    lines = []
    lines.extend(_format_key_value("model", response.get("model")))
    message = response.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            lines.extend(_format_action_or_message(content))
        else:
            lines.append("  assistant:")
            lines.extend(_indent_lines(_format_message_content(message).splitlines(), indent="    "))
    lines.extend(_format_key_value("done_reason", response.get("done_reason")))
    token_summary = _token_summary(response)
    if token_summary:
        lines.append(f"  token_counts: {token_summary}")
    duration_summary = _duration_summary(response)
    if duration_summary:
        lines.append(f"  duration_ms: {duration_summary}")
    return lines


def _format_tool_call(action: Any, args: Any) -> list[str]:
    summary = _format_action_summary(str(action) if action is not None else "?", args if isinstance(args, dict) else None)
    lines = [f"  call: {summary}"]
    if isinstance(args, dict) and args and len(args) > 1:
        lines.extend(_format_key_value("args", args, as_json=True))
    return lines


def _format_tool_result(payload: dict[str, Any]) -> list[str]:
    action = payload.get("action")
    ok = payload.get("ok")
    summary = payload.get("summary")
    lines = [f"  result: {'ok' if ok else 'error'}"]
    if action is not None:
        lines.append(f"  action: {action}")
    if summary is not None:
        lines.append(f"  summary: {summary}")
    exit_code = payload.get("exit_code")
    if exit_code is not None:
        lines.append(f"  exit_code: {exit_code}")

    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata:
        important_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {"path"}
        }
        if important_metadata:
            lines.extend(_format_key_value("metadata", important_metadata, as_json=True))

    stdout = payload.get("stdout")
    if isinstance(stdout, str) and stdout:
        label = "entries" if action == "list_files" else "output"
        lines.extend(_format_multiline_block(label, stdout))

    stderr = payload.get("stderr")
    if isinstance(stderr, str) and stderr:
        lines.extend(_format_multiline_block("stderr", stderr))
    return lines


def _format_loop_stalled(payload: dict[str, Any]) -> list[str]:
    lines = []
    reason = payload.get("reason")
    if reason is not None:
        lines.append(f"  reason: {reason}")
    action = payload.get("action")
    args = payload.get("args")
    if action is not None:
        lines.append(f"  blocked_action: {_format_action_summary(str(action), args if isinstance(args, dict) else None)}")
    normalized_failure = payload.get("normalized_failure")
    if normalized_failure is not None:
        lines.append(f"  normalized_failure: {normalized_failure}")
    return lines


def _format_approval(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.extend(_format_key_value("action", payload.get("action")))
    lines.extend(_format_key_value("approved", payload.get("approved")))
    lines.extend(_format_key_value("summary", payload.get("summary")))
    return lines


def _format_key_value(key: str, value: Any, *, as_json: bool = False) -> list[str]:
    if value is None:
        return []
    if as_json:
        lines = _json_dump(value).splitlines()
    else:
        text = str(value)
        lines = text.splitlines() or [""]
    formatted = [f"  {key}: {lines[0]}"]
    formatted.extend(_indent_lines(lines[1:], indent="    "))
    return formatted


def _format_multiline_block(label: str, text: str) -> list[str]:
    lines = text.splitlines() or [""]
    formatted = [f"  {label}: {lines[0]}"]
    formatted.extend(_indent_lines(lines[1:], indent="    "))
    return formatted


def _format_runtime_state_summary(runtime_state: dict[str, Any]) -> list[str]:
    lines = []
    last_action = runtime_state.get("last_action")
    if isinstance(last_action, dict):
        action = last_action.get("action")
        args = last_action.get("args")
        if action is not None:
            lines.append(
                f"  last_action: {_format_action_summary(str(action), args if isinstance(args, dict) else None)}"
            )

    last_tool_result = runtime_state.get("last_tool_result")
    if isinstance(last_tool_result, dict):
        ok = last_tool_result.get("ok")
        summary = last_tool_result.get("summary")
        if summary is not None:
            status = "ok" if ok else "error"
            lines.append(f"  last_tool_result: {status} | {summary}")

    stall_reason = runtime_state.get("stall_reason")
    if stall_reason is not None:
        lines.append(f"  stall_reason: {stall_reason}")

    decision_rule = runtime_state.get("decision_rule")
    if isinstance(decision_rule, dict) and decision_rule:
        parts = []
        for key in (
            "may_answer_now",
            "choose_final_answer_if_sufficient",
            "must_not_repeat_last_action",
            "repeat_same_tool_call_forbidden",
            "repeat_detected",
        ):
            if key in decision_rule:
                parts.append(f"{key}={decision_rule[key]}")
        allowed_actions = decision_rule.get("allowed_actions")
        if isinstance(allowed_actions, list) and allowed_actions:
            parts.append("allowed_actions=" + ",".join(str(item) for item in allowed_actions))
        if parts:
            lines.append("  decision_rule: " + " | ".join(parts))
    return lines


def _extract_allowed_actions(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return []
    variants = schema.get("oneOf")
    if not isinstance(variants, list):
        return []
    actions: list[str] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        properties = variant.get("properties")
        if not isinstance(properties, dict):
            continue
        action_schema = properties.get("action")
        if not isinstance(action_schema, dict):
            continue
        action = action_schema.get("const")
        if isinstance(action, str):
            actions.append(action)
    return actions


def _parse_runtime_state(content: str) -> dict[str, Any] | None:
    prefix = "Runtime state JSON:\n"
    if not content.startswith(prefix):
        return None
    raw = content[len(prefix) :]
    suffix = "\nTreat this runtime state as authoritative."
    if raw.endswith(suffix):
        raw = raw[: -len(suffix)]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _latest_conversation_message(conversation: list[dict[str, str]], role: str) -> str | None:
    for message in reversed(conversation):
        if message["role"] == role:
            return message["content"]
    return None


def _format_action_or_message(content: str) -> list[str]:
    parsed = _parse_action_payload(content)
    if parsed is None:
        return _format_multiline_block("assistant", content)

    action = parsed.get("action")
    args = parsed.get("args")
    if not isinstance(action, str):
        return _format_multiline_block("assistant", _json_dump(parsed))
    if action == "final_answer" and isinstance(args, dict):
        message = args.get("message")
        if isinstance(message, str):
            return _format_multiline_block("final_answer", message)
    if action == "ask_user" and isinstance(args, dict):
        lines = [f"  ask_user: {args.get('question', '')}"]
        context = args.get("context")
        if context:
            lines.append(f"  context: {context}")
        return lines
    return [f"  decision: {_format_action_summary(action, args if isinstance(args, dict) else None)}"]


def _parse_action_payload(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _format_action_summary(action: str, args: dict[str, Any] | None) -> str:
    if not args:
        return f"{action}()"
    rendered_args = ", ".join(
        f"{key}={_short_value(value)}" for key, value in args.items()
    )
    return f"{action}({rendered_args})"


def _short_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    return _json_dump(value).replace("\n", " ")


def _conversation_preview(text: str) -> str:
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if not parts:
        return ""
    preview = " / ".join(parts)
    if len(preview) <= 140:
        return preview
    return preview[:137] + "..."


def _indent_lines(lines: list[str], *, indent: str) -> list[str]:
    return [f"{indent}{line}" if line else indent.rstrip() for line in lines]


def _token_summary(response: dict[str, Any]) -> str:
    parts = []
    if "prompt_eval_count" in response:
        parts.append(f"prompt={response['prompt_eval_count']}")
    if "eval_count" in response:
        parts.append(f"eval={response['eval_count']}")
    return " ".join(parts)


def _duration_summary(response: dict[str, Any]) -> str:
    fields = [
        ("total_duration", "total"),
        ("load_duration", "load"),
        ("prompt_eval_duration", "prompt_eval"),
        ("eval_duration", "eval"),
    ]
    parts = []
    for field_name, label in fields:
        value = response.get(field_name)
        if isinstance(value, int):
            parts.append(f"{label}={value // 1_000_000}")
    return " ".join(parts)


def _prettify_json_string(text: str) -> str:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return _json_dump(parsed)


def _json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True)
