from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .models import ModelCallResult


class OpenAIClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
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
            "temperature": 0,
        }
        if format_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "vib_code_action",
                    "schema": format_schema,
                    "strict": True,
                },
            }
        return payload

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        format_schema: dict[str, Any] | None = None,
    ) -> ModelCallResult:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        payload_dict = self.build_chat_payload(messages, format_schema=format_schema)
        payload = json.dumps(payload_dict).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach OpenAI at {self.base_url}."
            ) from exc

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI returned no choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            content = "".join(text_parts)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenAI returned an empty response.")

        return ModelCallResult(
            request_payload=payload_dict,
            response_body=body,
            content=content.strip(),
        )
