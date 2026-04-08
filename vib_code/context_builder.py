from __future__ import annotations

import json

from .action_schema import action_schema_text, blocked_action_schema_text
from .models import ChatMessage, Config, SessionRecord


def build_chat_messages(
    config: Config,
    session: SessionRecord,
    runtime_state: dict[str, object] | None = None,
    *,
    turn_context: dict[str, object] | None = None,
    blocked_mode: bool = False,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": config.system_prompt.strip()},
    ]

    if session.summary:
        messages.append(
            {
                "role": "system",
                "content": f"Session summary:\n{session.summary}",
            }
        )

    messages.append(
        {
            "role": "system",
            "content": "Structured output JSON schema:\n"
            + (
                blocked_action_schema_text(provider=config.provider)
                if blocked_mode
                else action_schema_text(provider=config.provider)
            ),
        }
    )

    if runtime_state is not None:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Runtime state JSON:\n"
                    + json.dumps(runtime_state, indent=2, ensure_ascii=True)
                    + "\nTreat this runtime state as authoritative."
                ),
            }
        )

    if turn_context is not None:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Referenced path context for this turn:\n"
                    + json.dumps(turn_context, indent=2, ensure_ascii=True)
                    + "\nTreat referenced path context as authoritative for this turn."
                ),
            }
        )

    for message in _recent_messages(session.messages, runtime_state):
        messages.append({"role": message.role, "content": message.content})

    messages.append(
        {
            "role": "system",
            "content": _phase_instruction(runtime_state),
        }
    )
    return messages


def _recent_messages(
    messages: list[ChatMessage],
    runtime_state: dict[str, object] | None,
    limit: int = 8,
) -> list[ChatMessage]:
    recent = messages[-limit:] if len(messages) > limit else list(messages)
    if runtime_state is None:
        return recent

    phase = runtime_state.get("phase")
    if phase not in {"post_tool_result", "post_read_file_result", "retry_after_edit_error", "stalled_after_repeated_failure"}:
        return recent

    if phase in {"post_tool_result", "post_read_file_result", "retry_after_edit_error"}:
        return recent

    filtered: list[ChatMessage] = []
    user_budget = 3
    for message in reversed(recent):
        if message.role == "user":
            if user_budget > 0:
                filtered.append(message)
                user_budget -= 1
            continue
        if message.role == "assistant":
            stripped = message.content.strip()
            if (
                "Do you want" in stripped
                or "Would you like" in stripped
                or stripped.endswith("?")
            ):
                filtered.append(message)
            continue
    filtered.reverse()
    return filtered


def _phase_instruction(runtime_state: dict[str, object] | None) -> str:
    if runtime_state is None:
        return (
            "Phase: initial_decision.\n"
            "Question: What is the next step?\n"
            "For search_text, quoted queries are exact phrases, /.../ is raw regex, and plain multi-word queries use fuzzy matching.\n"
            "read_file reads exactly one file per action.\n"
            "Use create_file only for new files.\n"
            "For an existing file edit, use read_file first unless the latest runtime state already contains the file contents.\n"
            "For simple exact edits, prefer replace_in_file over write_patch.\n"
            "For whole-line deletion with replace_in_file, include the exact line indentation and trailing newline in the find text.\n"
            "When editing an existing file with write_patch, use unified diff hunks starting with @@.\n"
            "Return exactly one schema-valid JSON object and nothing else. Do not emit a second JSON object or any trailing text.\n"
            "If the user can be answered directly, use final_answer. "
            "Otherwise choose exactly one tool."
        )

    phase = runtime_state.get("phase")
    if phase == "post_tool_result":
        return (
            "Phase: post_tool_result.\n"
            "These are the results of the last tool call. What is the next step?\n"
            "Ground your next step in the latest tool result, not prior assistant text.\n"
            "If prior assistant text conflicts with the latest tool result, ignore the prior assistant text.\n"
            "Use Runtime state JSON fields like turn_goal, recent_tool_results, read_progress, and repeat_pattern.\n"
            "Preserve the user's current objective from the recent conversation.\n"
            "If the user asked you to implement, modify, inspect, or plan code changes, keep moving toward that objective.\n"
            "A directory listing alone is usually not sufficient to implement or plan a code change.\n"
            "If the latest tool result is only a file list and the user asked for a change, inspect likely files with read_file or search_text instead of stopping.\n"
            "If the user asked to read multiple files or summarize a directory, do not stop after one file.\n"
            "read_file reads exactly one file per action, so continue with additional read_file calls as needed.\n"
            "If read_progress.remaining_files_count is greater than 0, the task is not complete yet.\n"
            "If read_progress.next_unread_file is present, prefer that file over rereading an already-read file.\n"
            "Do not call list_files again when read_progress already contains a current directory listing unless you need a different directory.\n"
            "If repeat_pattern.detected is true, break the loop by choosing a different unread file.\n"
            "If the tool result is sufficient, return final_answer.\n"
            "Only choose another tool if information is still missing.\n"
            "Do not repeat the same tool call if the result already answers the user."
        )

    if phase == "post_read_file_result":
        return (
            "Phase: post_read_file_result.\n"
            "You have successfully read the requested file.\n"
            "Ground your next step in the latest file contents, not prior assistant text.\n"
            "If prior assistant text conflicts with the latest file contents, ignore the prior assistant text.\n"
            "Use Runtime state JSON fields like turn_goal, recent_tool_results, read_progress, and repeat_pattern.\n"
            "If the user asked what the file does, summarize the code directly from the file contents.\n"
            "If the user asked to read multiple files or summarize the directory, one read_file result is not enough.\n"
            "read_file reads exactly one file per action, so choose another relevant file instead of stopping early.\n"
            "If read_progress.remaining_files_count is greater than 0, continue reading files instead of returning final_answer.\n"
            "Prefer read_progress.next_unread_file when it is present.\n"
            "Do not choose any path that is already listed in read_progress.read_files unless the user explicitly asked to reread it.\n"
            "Do not call list_files again when the current listing is already available in read_progress.listed_files.\n"
            "If repeat_pattern.detected is true, break the loop by choosing a different unread file.\n"
            "If the user asked you to edit this file, prefer replace_in_file for simple exact text edits.\n"
            "For whole-line deletion with replace_in_file, include the exact line indentation and trailing newline in the find text.\n"
            "Use write_patch only when replace_in_file is not expressive enough.\n"
            "Unified diff hunks must start with headers like @@ -1,1 +1,2 @@ and use space/-/+ line prefixes.\n"
            "Return final_answer only when the user's request is already satisfied.\n"
            "Do not reread the same file if the current file contents are already present in the latest tool result."
        )

    if phase == "retry_after_edit_error":
        return (
            "Phase: retry_after_edit_error.\n"
            "Your last file edit failed.\n"
            "Do not repeat the same failed edit.\n"
            "Use the latest known file contents and exact text from the file.\n"
            "Prefer replace_in_file for simple deletions or substitutions of a known line or block.\n"
            "For whole-line deletion with replace_in_file, include the exact line indentation and trailing newline in the find text.\n"
            "Use write_patch only if an exact text replacement cannot express the change.\n"
            "Return exactly one schema-valid JSON object and nothing else."
        )

    if phase == "retry_after_invalid_action":
        return (
            "Phase: retry_after_invalid_action.\n"
            "Your last response did not match the schema.\n"
            "What is the next step?\n"
            "Return exactly one schema-valid JSON object and nothing else.\n"
            "Do not emit a second JSON object.\n"
            "Do not add prose, markdown, or trailing text."
        )

    if phase == "stalled_after_repeated_failure":
        stall_reason = str(runtime_state.get("stall_reason", ""))
        last_tool_result = runtime_state.get("last_tool_result")
        last_ok = False
        if isinstance(last_tool_result, dict):
            last_ok = bool(last_tool_result.get("ok"))

        if stall_reason == "repeated_no_progress_success" and last_ok:
            return (
                "Phase: stalled_after_repeated_failure.\n"
                "The last tool call succeeded, but repeating the same successful tool call made no progress.\n"
                "Do not claim that the tool returned no results.\n"
                "Base your next step on the actual last tool result.\n"
                "Ground your next step in the latest tool result, not prior assistant text.\n"
                "If prior assistant text conflicts with the latest tool result, ignore the prior assistant text.\n"
                "Use Runtime state JSON fields like turn_goal, recent_tool_results, read_progress, and repeat_pattern.\n"
                "If read_progress.remaining_files_count is greater than 0, do not stop yet.\n"
                "If read_progress.next_unread_file is present, choose that unread file instead of repeating the last tool call.\n"
                "You may only return final_answer or ask_user.\n"
                "Use final_answer if the existing result is enough.\n"
                "Use ask_user only to ask what to do next with the known result."
            )

        return (
            "Phase: stalled_after_repeated_failure.\n"
            "The last tool attempts did not make progress.\n"
            "Base your next step on the actual last tool result.\n"
            "Ground your next step in the latest tool result, not prior assistant text.\n"
            "If prior assistant text conflicts with the latest tool result, ignore the prior assistant text.\n"
            "You may only return final_answer or ask_user.\n"
            "If the user needs to guide the next action, use ask_user."
        )

    return (
        "Phase: generic_decision.\n"
        "What is the next step?\n"
        "Return exactly one schema-valid JSON object and nothing else."
    )
