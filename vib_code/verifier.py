from __future__ import annotations

from .models import ToolResult


def verify_completion() -> ToolResult:
    return ToolResult(
        ok=True,
        action_type="verify_placeholder",
        summary="Verification layer is planned but not implemented in this checkpoint.",
    )
