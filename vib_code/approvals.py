from __future__ import annotations

from .action_parser import ActionRequest


def confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def confirm_action(action: ActionRequest, preview: str) -> bool:
    print(f"[approval] action={action.action}")
    return confirm("Approve this action?")
