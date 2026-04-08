from __future__ import annotations

import json
from dataclasses import dataclass


class ActionParseError(ValueError):
    pass


@dataclass
class ActionRequest:
    action: str
    args: dict[str, object]


ALLOWED_ACTIONS = {
    "final_answer",
    "ask_user",
    "list_files",
    "read_file",
    "search_text",
    "create_file",
    "replace_in_file",
    "write_patch",
    "run_command",
}


def _load_action_payload(text: str) -> object:
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"Model output was not valid JSON: {exc}") from exc

    index = end
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            return payload
        try:
            extra_payload, extra_end = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"Model output was not valid JSON: {exc}") from exc
        if extra_payload != payload:
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise ActionParseError(f"Model output was not valid JSON: {exc}") from exc
            raise ActionParseError("Model output contained multiple JSON objects.")
        index = extra_end


def parse_action(raw_text: str) -> ActionRequest:
    text = raw_text.strip()
    payload = _load_action_payload(text)

    if not isinstance(payload, dict):
        raise ActionParseError("Model output must be a JSON object.")

    action = payload.get("action")
    args = payload.get("args", {})
    if action not in ALLOWED_ACTIONS:
        raise ActionParseError(f"Unsupported action: {action!r}")
    if not isinstance(args, dict):
        raise ActionParseError("`args` must be a JSON object.")

    if action == "final_answer":
        if not isinstance(args.get("message"), str) or not args["message"].strip():
            raise ActionParseError("final_answer requires a non-empty string `message`.")
    elif action == "ask_user":
        if not isinstance(args.get("question"), str) or not args["question"].strip():
            raise ActionParseError("ask_user requires a non-empty string `question`.")
        context = args.get("context", "")
        if context is not None and not isinstance(context, str):
            raise ActionParseError("ask_user `context` must be a string when provided.")
    elif action in {"list_files", "read_file"}:
        if not isinstance(args.get("path"), str) or not args["path"].strip():
            raise ActionParseError(f"{action} requires a non-empty string `path`.")
    elif action == "search_text":
        if not isinstance(args.get("query"), str) or not args["query"].strip():
            raise ActionParseError("search_text requires a non-empty string `query`.")
        path = args.get("path", ".")
        if not isinstance(path, str) or not path.strip():
            raise ActionParseError("search_text `path` must be a non-empty string when provided.")
        args["path"] = path
    elif action == "create_file":
        if not isinstance(args.get("path"), str) or not args["path"].strip():
            raise ActionParseError("create_file requires a non-empty string `path`.")
        if not isinstance(args.get("content"), str):
            raise ActionParseError("create_file requires string `content`.")
    elif action == "replace_in_file":
        if not isinstance(args.get("path"), str) or not args["path"].strip():
            raise ActionParseError("replace_in_file requires a non-empty string `path`.")
        if not isinstance(args.get("find"), str) or not args["find"]:
            raise ActionParseError("replace_in_file requires a non-empty string `find`.")
        if not isinstance(args.get("replace"), str):
            raise ActionParseError("replace_in_file requires string `replace`.")
        expected_count = args.get("expected_count", 1)
        if not isinstance(expected_count, int) or expected_count < 1:
            raise ActionParseError("replace_in_file `expected_count` must be a positive integer when provided.")
        args["expected_count"] = expected_count
    elif action == "write_patch":
        if not isinstance(args.get("path"), str) or not args["path"].strip():
            raise ActionParseError("write_patch requires a non-empty string `path`.")
        if not isinstance(args.get("patch"), str) or not args["patch"].strip():
            raise ActionParseError("write_patch requires a non-empty string `patch`.")
    elif action == "run_command":
        if not isinstance(args.get("command"), str) or not args["command"].strip():
            raise ActionParseError("run_command requires a non-empty string `command`.")
        cwd = args.get("cwd", ".")
        if not isinstance(cwd, str) or not cwd.strip():
            raise ActionParseError("run_command `cwd` must be a non-empty string when provided.")
        args["cwd"] = cwd

    return ActionRequest(action=action, args=args)
