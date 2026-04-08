from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ChatMessage:
    role: str
    content: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class SessionRecord:
    session_id: str
    title: str
    model: str
    created_at: str
    updated_at: str
    working_directory: str
    trace_path: str
    storage_name: str = ""
    summary: str = ""
    messages: list[ChatMessage] = field(default_factory=list)


@dataclass
class Config:
    provider: str
    model: str
    ollama_host: str
    openai_base_url: str
    openai_api_key: str
    storage_dir: str
    system_prompt: str
    workspace_dir: str


@dataclass
class ToolResult:
    ok: bool
    action_type: str
    summary: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCallResult:
    request_payload: dict[str, Any]
    response_body: dict[str, Any]
    content: str
