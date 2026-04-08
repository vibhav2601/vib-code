from __future__ import annotations

print("hello")

import difflib
import re
import shlex
import subprocess
from pathlib import Path

from .action_parser import ActionRequest
from .models import ToolResult
from .patch_engine import PatchEngine, PatchError


def execute_action(
    action: ActionRequest,
    workspace_dir: str,
    *,
    preview_only: bool = False,
    patch_engine: PatchEngine | None = None,
) -> ToolResult:
    if action.action == "list_files":
        return _list_files(str(action.args["path"]), workspace_dir)
    if action.action == "read_file":
        return _read_file(str(action.args["path"]), workspace_dir)
    if action.action == "search_text":
        return _search_text(str(action.args["query"]), str(action.args.get("path", ".")), workspace_dir)
    if action.action == "create_file":
        return _create_file(str(action.args["path"]), str(action.args["content"]), workspace_dir, preview_only=preview_only)
    if action.action == "replace_in_file":
        return _replace_in_file(
            str(action.args["path"]),
            str(action.args["find"]),
            str(action.args["replace"]),
            int(action.args.get("expected_count", 1)),
            workspace_dir,
            preview_only=preview_only,
        )
    if action.action == "write_patch":
        engine = patch_engine or PatchEngine()
        return _write_patch(
            str(action.args["path"]),
            str(action.args["patch"]),
            workspace_dir,
            engine,
            preview_only=preview_only,
        )
    if action.action == "run_command":
        return _run_command(
            str(action.args["command"]),
            str(action.args.get("cwd", ".")),
            workspace_dir,
            preview_only=preview_only,
        )

    return ToolResult(
        ok=False,
        action_type=action.action,
        summary=f"Unsupported executable action: {action.action}",
    )


def _resolve_path(raw_path: str, workspace_dir: str) -> Path:
    workspace = Path(workspace_dir).resolve()
    candidate = (workspace / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {raw_path}") from exc
    return candidate


def _list_files(raw_path: str, workspace_dir: str) -> ToolResult:
    try:
        target = _resolve_path(raw_path, workspace_dir)
    except ValueError as exc:
        return ToolResult(
            ok=False,
            action_type="list_files",
            summary=str(exc),
            metadata={"error_type": "workspace_boundary", "retryable": False, "normalized_failure": "workspace_boundary"},
        )

    if not target.exists():
        return ToolResult(
            ok=False,
            action_type="list_files",
            summary=f"Path does not exist: {raw_path}",
            metadata={"error_type": "not_found", "retryable": False, "normalized_failure": "not_found"},
        )
    if not target.is_dir():
        return ToolResult(
            ok=False,
            action_type="list_files",
            summary=f"Not a directory: {raw_path}",
            metadata={"error_type": "not_directory", "retryable": False, "normalized_failure": "not_directory"},
        )

    entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    rendered = []
    for entry in entries[:200]:
        suffix = "/" if entry.is_dir() else ""
        rendered.append(f"{entry.name}{suffix}")
    if len(entries) > 200:
        rendered.append(f"... truncated {len(entries) - 200} more entries")

    return ToolResult(
        ok=True,
        action_type="list_files",
        summary=f"Listed {len(entries)} entries in {raw_path}",
        stdout="\n".join(rendered),
        metadata={"path": raw_path},
    )


def _create_file(raw_path: str, content: str, workspace_dir: str, *, preview_only: bool) -> ToolResult:
    try:
        target = _resolve_path(raw_path, workspace_dir)
    except ValueError as exc:
        return ToolResult(
            ok=False,
            action_type="create_file",
            summary=str(exc),
            metadata={"error_type": "workspace_boundary", "retryable": False, "normalized_failure": "workspace_boundary"},
        )

    if target.exists():
        return ToolResult(
            ok=False,
            action_type="create_file",
            summary=f"File already exists: {raw_path}",
            metadata={"error_type": "already_exists", "retryable": False, "normalized_failure": "already_exists"},
        )

    preview = "\n".join(
        difflib.unified_diff(
            [],
            content.splitlines(),
            fromfile=str(target),
            tofile=str(target),
            lineterm="",
        )
    )
    if preview_only:
        return ToolResult(
            ok=True,
            action_type="create_file",
            summary=f"Preview create file {raw_path}",
            stdout=preview,
            metadata={"path": raw_path, "preview_only": True},
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return ToolResult(
        ok=True,
        action_type="create_file",
        summary=f"Created file {raw_path}",
        metadata={"path": raw_path},
    )


def _replace_in_file(
    raw_path: str,
    find_text: str,
    replace_text: str,
    expected_count: int,
    workspace_dir: str,
    *,
    preview_only: bool,
) -> ToolResult:
    try:
        target = _resolve_path(raw_path, workspace_dir)
    except ValueError as exc:
        return ToolResult(
            ok=False,
            action_type="replace_in_file",
            summary=str(exc),
            metadata={"error_type": "workspace_boundary", "retryable": False, "normalized_failure": "workspace_boundary"},
        )

    if not target.exists():
        return ToolResult(
            ok=False,
            action_type="replace_in_file",
            summary=f"File does not exist: {raw_path}",
            metadata={"error_type": "not_found", "retryable": False, "normalized_failure": "not_found"},
        )
    if not target.is_file():
        return ToolResult(
            ok=False,
            action_type="replace_in_file",
            summary=f"Not a file: {raw_path}",
            metadata={"error_type": "not_file", "retryable": False, "normalized_failure": "not_file"},
        )

    original_text = target.read_text(encoding="utf-8", errors="replace")
    match_count = original_text.count(find_text)
    if match_count == 0:
        return ToolResult(
            ok=False,
            action_type="replace_in_file",
            summary=f"Could not find the exact text in {raw_path}.",
            metadata={
                "error_type": "replace_not_found",
                "retryable": False,
                "normalized_failure": "replace_not_found",
                "path": raw_path,
                "expected_count": expected_count,
            },
        )
    if match_count != expected_count:
        return ToolResult(
            ok=False,
            action_type="replace_in_file",
            summary=(
                f"Expected {expected_count} match(es) for replacement in {raw_path}, "
                f"but found {match_count}."
            ),
            metadata={
                "error_type": "replace_count_mismatch",
                "retryable": False,
                "normalized_failure": "replace_count_mismatch",
                "path": raw_path,
                "match_count": match_count,
                "expected_count": expected_count,
            },
        )

    updated_text = original_text.replace(find_text, replace_text)
    if updated_text == original_text:
        return ToolResult(
            ok=False,
            action_type="replace_in_file",
            summary="Replacement would not change the file.",
            metadata={"error_type": "replace_no_change", "retryable": False, "normalized_failure": "replace_no_change"},
        )

    preview = "\n".join(
        difflib.unified_diff(
            original_text.splitlines(),
            updated_text.splitlines(),
            fromfile=str(target),
            tofile=str(target),
            lineterm="",
        )
    )
    if preview_only:
        return ToolResult(
            ok=True,
            action_type="replace_in_file",
            summary=f"Preview replace text in {raw_path}",
            stdout=preview,
            metadata={"path": raw_path, "match_count": match_count, "preview_only": True},
        )

    target.write_text(updated_text, encoding="utf-8")
    return ToolResult(
        ok=True,
        action_type="replace_in_file",
        summary=f"Replaced text in {raw_path}",
        metadata={"path": raw_path, "match_count": match_count},
    )


def _write_patch(
    raw_path: str,
    patch: str,
    workspace_dir: str,
    patch_engine: PatchEngine,
    *,
    preview_only: bool,
) -> ToolResult:
    try:
        target = _resolve_path(raw_path, workspace_dir)
    except ValueError as exc:
        return ToolResult(
            ok=False,
            action_type="write_patch",
            summary=str(exc),
            metadata={"error_type": "workspace_boundary", "retryable": False, "normalized_failure": "workspace_boundary"},
        )

    if not target.exists():
        return ToolResult(
            ok=False,
            action_type="write_patch",
            summary=f"File does not exist: {raw_path}",
            metadata={"error_type": "not_found", "retryable": False, "normalized_failure": "not_found"},
        )
    if not target.is_file():
        return ToolResult(
            ok=False,
            action_type="write_patch",
            summary=f"Not a file: {raw_path}",
            metadata={"error_type": "not_file", "retryable": False, "normalized_failure": "not_file"},
        )

    try:
        preview = patch_engine.preview(target, patch)
    except PatchError as exc:
        return ToolResult(
            ok=False,
            action_type="write_patch",
            summary=str(exc),
            metadata={"error_type": "patch_error", "retryable": False, "normalized_failure": "patch_error"},
        )

    if preview_only:
        return ToolResult(
            ok=True,
            action_type="write_patch",
            summary=f"Preview patch for {raw_path}",
            stdout=preview,
            metadata={"path": raw_path, "preview_only": True},
        )

    try:
        patch_engine.apply(target, patch)
    except PatchError as exc:
        return ToolResult(
            ok=False,
            action_type="write_patch",
            summary=str(exc),
            metadata={"error_type": "patch_error", "retryable": False, "normalized_failure": "patch_error"},
        )
    return ToolResult(
        ok=True,
        action_type="write_patch",
        summary=f"Applied patch to {raw_path}",
        metadata={"path": raw_path},
    )


def _run_command(raw_command: str, raw_cwd: str, workspace_dir: str, *, preview_only: bool) -> ToolResult:
    try:
        cwd = _resolve_path(raw_cwd, workspace_dir)
    except ValueError as exc:
        return ToolResult(
            ok=False,
            action_type="run_command",
            summary=str(exc),
            metadata={"error_type": "workspace_boundary", "retryable": False, "normalized_failure": "workspace_boundary"},
        )

    if not cwd.exists() or not cwd.is_dir():
        return ToolResult(
            ok=False,
            action_type="run_command",
            summary=f"Command cwd is not a directory: {raw_cwd}",
            metadata={"error_type": "not_directory", "retryable": False, "normalized_failure": "not_directory"},
        )

    try:
        argv = shlex.split(raw_command)
    except ValueError as exc:
        return ToolResult(
            ok=False,
            action_type="run_command",
            summary=f"Could not parse command: {exc}",
            metadata={"error_type": "invalid_command", "retryable": False, "normalized_failure": "invalid_command"},
        )
    if not argv:
        return ToolResult(
            ok=False,
            action_type="run_command",
            summary="Command was empty after parsing.",
            metadata={"error_type": "invalid_command", "retryable": False, "normalized_failure": "invalid_command"},
        )

    preview = f"cwd: {cwd}\ncommand: {' '.join(argv)}"
    if preview_only:
        return ToolResult(
            ok=True,
            action_type="run_command",
            summary=f"Preview command in {raw_cwd}",
            stdout=preview,
            metadata={"command": raw_command, "cwd": str(cwd), "preview_only": True},
        )

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            check=False,
        )
    except FileNotFoundError:
        return ToolResult(
            ok=False,
            action_type="run_command",
            summary=f"Command not found: {argv[0]}",
            metadata={"error_type": "command_not_found", "retryable": False, "normalized_failure": "command_not_found"},
        )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 0:
        return ToolResult(
            ok=True,
            action_type="run_command",
            summary=f"Command succeeded: {' '.join(argv)}",
            stdout=stdout,
            stderr=stderr,
            exit_code=completed.returncode,
            metadata={"command": raw_command, "cwd": str(cwd)},
        )
    return ToolResult(
        ok=False,
        action_type="run_command",
        summary=f"Command failed with exit code {completed.returncode}: {' '.join(argv)}",
        stdout=stdout,
        stderr=stderr,
        exit_code=completed.returncode,
        metadata={
            "command": raw_command,
            "cwd": str(cwd),
            "error_type": "command_failed",
            "retryable": True,
            "normalized_failure": f"command_failed:{completed.returncode}",
        },
    )


def _read_file(raw_path: str, workspace_dir: str) -> ToolResult:
    try:
        target = _resolve_path(raw_path, workspace_dir)
    except ValueError as exc:
        return ToolResult(
            ok=False,
            action_type="read_file",
            summary=str(exc),
            metadata={"error_type": "workspace_boundary", "retryable": False, "normalized_failure": "workspace_boundary"},
        )

    if not target.exists():
        return ToolResult(
            ok=False,
            action_type="read_file",
            summary=f"File does not exist: {raw_path}",
            metadata={"error_type": "not_found", "retryable": False, "normalized_failure": "not_found"},
        )
    if not target.is_file():
        return ToolResult(
            ok=False,
            action_type="read_file",
            summary=f"Not a file: {raw_path}",
            metadata={"error_type": "not_file", "retryable": False, "normalized_failure": "not_file"},
        )

    content = target.read_text(encoding="utf-8", errors="replace")
    return ToolResult(
        ok=True,
        action_type="read_file",
        summary=f"Read file {raw_path}",
        stdout=content,
        metadata={"path": raw_path, "truncated": False},
    )


def _search_text(query: str, raw_path: str, workspace_dir: str) -> ToolResult:
    try:
        target = _resolve_path(raw_path, workspace_dir)
    except ValueError as exc:
        return ToolResult(
            ok=False,
            action_type="search_text",
            summary=str(exc),
            metadata={"error_type": "workspace_boundary", "retryable": False, "normalized_failure": "workspace_boundary"},
        )

    base = str(target)
    search_mode, resolved_pattern = _resolve_search_query(query)
    commands = [
        (_rg_search_command(search_mode, resolved_pattern, base), "rg"),
        (
            _grep_search_command(search_mode, resolved_pattern, base),
            "grep",
        ),
    ]
    completed = None
    backend = ""
    for command, candidate_backend in commands:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=workspace_dir,
                check=False,
            )
            backend = candidate_backend
            break
        except FileNotFoundError:
            continue

    if completed is None:
        return ToolResult(
            ok=False,
            action_type="search_text",
            summary="Neither `rg` nor `grep` is installed, so search_text is unavailable.",
            metadata={"error_type": "tool_unavailable", "retryable": False, "normalized_failure": "tool_unavailable"},
        )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode not in {0, 1}:
        return ToolResult(
            ok=False,
            action_type="search_text",
            summary=f"search_text failed with exit code {completed.returncode}",
            stdout=stdout,
            stderr=stderr,
            exit_code=completed.returncode,
            metadata={
                "error_type": "execution_failed",
                "retryable": True,
                "normalized_failure": f"execution_failed:{completed.returncode}",
            },
        )

    lines = stdout.splitlines() if stdout else []
    filtered_lines = []
    for line in lines:
        if "Binary file " in line and " matches" in line:
            continue
        if "__pycache__/" in line or line.endswith(".pyc"):
            continue
        filtered_lines.append(line)
    lines = filtered_lines
    rendered = "\n".join(lines[:200])
    if len(lines) > 200:
        rendered += f"\n... truncated {len(lines) - 200} more matches"

    return ToolResult(
        ok=True,
        action_type="search_text",
        summary=(
            f"Found {len(lines)} matches for {query!r} in {raw_path} "
            f"(mode={search_mode}, pattern={resolved_pattern!r})"
        ),
        stdout=rendered,
        stderr=stderr,
        exit_code=completed.returncode,
        metadata={
            "query": query,
            "path": raw_path,
            "backend": backend,
            "mode": search_mode,
            "resolved_pattern": resolved_pattern,
        },
    )


def _resolve_search_query(raw_query: str) -> tuple[str, str]:
    query = raw_query.strip()
    if len(query) >= 2 and query[0] == query[-1] and query[0] in {'"', "'"}:
        return "exact", query[1:-1]
    if len(query) >= 3 and query[0] == "/" and query[-1] == "/":
        return "regex", query[1:-1]

    tokens = query.split()
    if len(tokens) > 1:
        escaped_tokens = [re.escape(token) for token in tokens]
        return "fuzzy", ".*".join(escaped_tokens)
    return "exact", query


def _rg_search_command(mode: str, pattern: str, base: str) -> list[str]:
    command = ["rg", "-n", "--no-heading", "--color", "never"]
    if mode == "exact":
        command.append("-F")
    command.extend([pattern, base])
    return command


def _grep_search_command(mode: str, pattern: str, base: str) -> list[str]:
    command = [
        "grep",
        "-R",
        "-n",
        "-H",
        "--exclude-dir=__pycache__",
        "--exclude=*.pyc",
    ]
    if mode == "exact":
        command.append("-F")
    else:
        command.append("-E")
    command.extend([pattern, base])
    return command
