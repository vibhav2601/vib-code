from __future__ import annotations

import re
from pathlib import Path
from typing import Any


MENTION_RE = re.compile(r"(?<!\S)@([^\s]+)")


def extract_path_mentions(user_input: str, workspace_dir: str) -> dict[str, Any] | None:
    workspace = Path(workspace_dir).resolve()
    raw_matches = MENTION_RE.findall(user_input)
    if not raw_matches:
        return None

    mentions: list[dict[str, Any]] = []
    for raw in raw_matches:
        mention = _inspect_path(raw, workspace)
        mentions.append(mention)

    return {
        "workspace_dir": str(workspace),
        "mentions": mentions,
    }


def _inspect_path(raw_path: str, workspace: Path) -> dict[str, Any]:
    raw_path = raw_path.rstrip(".,:;!?")
    candidate = (workspace / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
    result: dict[str, Any] = {"path": raw_path}
    try:
        candidate.relative_to(workspace)
    except ValueError:
        result["ok"] = False
        result["error"] = "Path escapes workspace"
        return result

    if not candidate.exists():
        result["ok"] = False
        result["error"] = "Path does not exist"
        return result

    result["ok"] = True
    if candidate.is_dir():
        entries = sorted(candidate.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        rendered = []
        for entry in entries[:100]:
            rendered.append(f"{entry.name}/" if entry.is_dir() else entry.name)
        if len(entries) > 100:
            rendered.append(f"... truncated {len(entries) - 100} more entries")
        result["type"] = "directory"
        result["entries"] = rendered
        return result

    content = candidate.read_text(encoding="utf-8", errors="replace")
    truncated = False
    if len(content) > 8000:
        content = content[:8000]
        truncated = True
    result["type"] = "file"
    result["content"] = content
    result["truncated"] = truncated
    return result
