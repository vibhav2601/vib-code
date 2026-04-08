from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .model_client import ModelClient
from .models import ModelCallResult


class OllamaClient(ModelClient):
    def __init__(self, host: str, model: str) -> None:
        self.host = host.rstrip("/")
        self.model = model

    def build_chat_payload(
        self,
        messages: list[dict[str, str]],
        *,
        format_schema: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        if format_schema is not None:
            payload["format"] = format_schema
        return payload

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        format_schema: dict[str, Any] | None = None,
    ) -> ModelCallResult:
        payload_dict = self.build_chat_payload(messages, format_schema=format_schema)
        payload = json.dumps(payload_dict).encode("utf-8")

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running?"
            ) from exc

        message = body.get("message", {})
        content = message.get("content", "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return ModelCallResult(
            request_payload=payload_dict,
            response_body=body,
            content=content,
        )
