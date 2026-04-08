from __future__ import annotations

import json
from typing import Any


DEFAULT_PROVIDER = "ollama"
OPENAI_PROVIDER = "openai"


def action_schema(provider: str = DEFAULT_PROVIDER) -> dict[str, Any]:
    actions = [
        "final_answer",
        "list_files",
        "read_file",
        "search_text",
        "create_file",
        "replace_in_file",
        "write_patch",
        "run_command",
    ]
    return _schema_for_actions(actions, provider=provider)


def blocked_action_schema(provider: str = DEFAULT_PROVIDER) -> dict[str, Any]:
    return _schema_for_actions(["final_answer", "ask_user"], provider=provider)


def action_schema_text(provider: str = DEFAULT_PROVIDER) -> str:
    return json.dumps(action_schema(provider=provider), indent=2)


def blocked_action_schema_text(provider: str = DEFAULT_PROVIDER) -> str:
    return json.dumps(blocked_action_schema(provider=provider), indent=2)


def _schema_for_actions(actions: list[str], *, provider: str) -> dict[str, Any]:
    variants = [_variant_for_action(action) for action in actions]
    if provider == OPENAI_PROVIDER:
        return _openai_schema_for_variants(variants)
    return {
        "type": "object",
        "description": "Emit exactly one JSON object matching one allowed action variant. Do not emit multiple objects.",
        "oneOf": variants,
    }


def _variant_for_action(action: str) -> dict[str, Any]:
    if action == "final_answer":
        return {
            "type": "object",
            "description": "Answer the user directly when you already have enough information.",
            "properties": {
                "action": {"const": "final_answer"},
                "args": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "minLength": 1},
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
            "required": ["action", "args"],
            "additionalProperties": False,
        }

    if action == "ask_user":
        return {
            "type": "object",
            "description": "Ask the user a short clarification question when progress is blocked.",
            "properties": {
                "action": {"const": "ask_user"},
                "args": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                        "context": {"type": "string"},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            "required": ["action", "args"],
            "additionalProperties": False,
        }

    if action == "list_files":
        return {
            "type": "object",
            "description": "List files and directories under a workspace-relative path.",
            "properties": {
                "action": {"const": "list_files"},
                "args": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            "required": ["action", "args"],
            "additionalProperties": False,
        }

    if action == "read_file":
        return {
            "type": "object",
            "description": (
                "Read the contents of exactly one existing file inside the workspace. "
                "This action reads one file path at a time and cannot read multiple files in one call."
            ),
            "properties": {
                "action": {"const": "read_file"},
                "args": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            "required": ["action", "args"],
            "additionalProperties": False,
        }

    if action == "search_text":
        return {
            "type": "object",
            "description": (
                "Search for text in the workspace or a workspace-relative subtree. "
                "Quoted queries are exact phrases, /.../ is raw regex, and plain multi-word queries use fuzzy matching."
            ),
            "properties": {
                "action": {"const": "search_text"},
                "args": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "Search query. Use quotes for an exact phrase. Use /.../ for an explicit regex. "
                                "Otherwise multi-word queries are expanded into fuzzy matching."
                            ),
                        },
                        "path": {"type": "string", "minLength": 1},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            "required": ["action", "args"],
            "additionalProperties": False,
        }

    if action == "create_file":
        return {
            "type": "object",
            "description": "Create a brand-new file. Use this only when the file does not already exist.",
            "properties": {
                "action": {"const": "create_file"},
                "args": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Workspace-relative file path to create.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete contents for the new file.",
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
            "required": ["action", "args"],
            "additionalProperties": False,
        }

    if action == "replace_in_file":
        return {
            "type": "object",
            "description": (
                "Modify an existing file by replacing exact text. Prefer this over write_patch "
                "for simple edits like deleting or changing a known line or block. "
                "For whole-line deletion, include the indentation and trailing newline in the exact text to find."
            ),
            "properties": {
                "action": {"const": "replace_in_file"},
                "args": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Workspace-relative path to an existing file.",
                        },
                        "find": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "Exact text to find in the file. For whole-line deletion, include the full line "
                                "with indentation and trailing newline."
                            ),
                        },
                        "replace": {
                            "type": "string",
                            "description": "Replacement text. Use an empty string to delete the matched text.",
                        },
                        "expected_count": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Expected number of matches. Omit to require exactly one match.",
                        },
                    },
                    "required": ["path", "find", "replace"],
                    "additionalProperties": False,
                },
            },
            "required": ["action", "args"],
            "additionalProperties": False,
        }

    if action == "write_patch":
        return {
            "type": "object",
            "description": (
                "Modify an existing file by applying unified diff hunks. "
                "Use this only when replace_in_file is not expressive enough, and read_file first unless "
                "the latest tool result already contains the file contents."
            ),
            "properties": {
                "action": {"const": "write_patch"},
                "args": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Workspace-relative path to an existing file.",
                        },
                        "patch": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "Unified diff hunk text only. Each hunk must start with a header like "
                                "'@@ -1,1 +1,2 @@'. Prefix unchanged lines with a space, removed lines "
                                "with '-', and added lines with '+'. Do not send plain replacement text."
                            ),
                        },
                    },
                    "required": ["path", "patch"],
                    "additionalProperties": False,
                },
            },
            "required": ["action", "args"],
            "additionalProperties": False,
        }

    if action == "run_command":
        return {
            "type": "object",
            "description": "Run a simple non-shell command inside the workspace after user approval.",
            "properties": {
                "action": {"const": "run_command"},
                "args": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Command string without shell operators like pipes or redirects.",
                        },
                        "cwd": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Optional workspace-relative directory to run the command in.",
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
            "required": ["action", "args"],
            "additionalProperties": False,
        }

    raise ValueError(f"Unsupported action for schema generation: {action}")


def _openai_schema_for_variants(variants: list[dict[str, Any]]) -> dict[str, Any]:
    action_names: list[str] = []
    args_variants: list[dict[str, Any]] = []
    for variant in variants:
        action_schema = variant["properties"]["action"]
        if "const" in action_schema:
            action_names.append(str(action_schema["const"]))
        args_variants.append(_normalize_openai_args_schema(variant["properties"]["args"]))

    return {
        "type": "object",
        "description": (
            "Emit exactly one JSON object with keys action and args. "
            "Do not emit a second JSON object or any trailing text."
        ),
        "properties": {
            "action": {
                "type": "string",
                "enum": action_names,
                "description": "Choose exactly one action from the allowed set.",
            },
            "args": {
                "anyOf": args_variants,
                "description": "Arguments for the chosen action. Match the selected action as closely as possible.",
            },
        },
        "required": ["action", "args"],
        "additionalProperties": False,
    }


def _normalize_openai_args_schema(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(schema))
    properties = normalized.get("properties")
    if not isinstance(properties, dict):
        return normalized

    normalized["required"] = list(properties.keys())
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        if name == "path":
            prop.setdefault(
                "description",
                "Workspace-relative path. When unsure, use '.' or a confirmed file path from the latest tool result.",
            )
        elif name == "context":
            prop.setdefault(
                "description",
                "Context for the question. Use an empty string when there is no extra context.",
            )
        elif name == "cwd":
            prop.setdefault(
                "description",
                "Workspace-relative directory to run the command in. Use '.' when not needed.",
            )
        elif name == "expected_count":
            prop.setdefault(
                "description",
                "Expected number of matches. Use 1 when there should be exactly one match.",
            )
    return normalized
