from __future__ import annotations

from typing import Any, Protocol

from .models import ModelCallResult


class ModelClient(Protocol):
    model: str

    def build_chat_payload(
        self,
        messages: list[dict[str, str]],
        *,
        format_schema: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        ...

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        format_schema: dict[str, Any] | None = None,
    ) -> ModelCallResult:
        ...
