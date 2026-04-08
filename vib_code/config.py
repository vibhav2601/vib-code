from __future__ import annotations

import json
import os
from pathlib import Path

from .models import Config


DEFAULT_SYSTEM_PROMPT = """You are vib-code, a local coding harness running in the terminal.
You are vib-code.
Be concise, practical, and deterministic.
Use the host-provided runtime state and schema as authoritative.
Do not invent repository facts when a tool is required.
Always return exactly one JSON object that matches the provided schema.
Do not emit a second JSON object.
Do not repeat the same object.
Do not add prose, markdown, code fences, or trailing text before or after the JSON object.
"""

DEFAULT_CONTEXT_WINDOW_TOKENS = 32768

MODEL_CONTEXT_WINDOW_TOKENS = {
    "gpt-5.4": 1_048_576,
    "gpt-5.4-mini": 400_000,
    "gpt-5.4-nano": 400_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "qwen2.5:3b": 32_768,
    "qwen2.5-coder:7b": 131_072,
}


def load_config(base_dir: Path, workspace_dir: Path | None = None) -> Config:
    config_path = base_dir / ".vib-code-config.json"
    file_config: dict[str, str] = {}
    if config_path.exists():
        file_config = json.loads(config_path.read_text())

    resolved_workspace = workspace_dir or base_dir
    storage_dir = (
        os.environ.get("VIB_CODE_STORAGE_DIR")
        or file_config.get("storage_dir")
        or str(Path.home() / ".vib-code")
    )
    model = os.environ.get("VIB_CODE_MODEL", file_config.get("model", "qwen2.5:3b"))
    context_window_tokens = (
        os.environ.get("VIB_CODE_CONTEXT_WINDOW_TOKENS")
        or file_config.get("context_window_tokens")
    )

    return Config(
        provider=os.environ.get("VIB_CODE_PROVIDER", file_config.get("provider", "ollama")),
        model=model,
        context_window_tokens=int(context_window_tokens)
        if context_window_tokens is not None
        else resolve_context_window_tokens(model),
        ollama_host=os.environ.get(
            "VIB_CODE_OLLAMA_HOST",
            file_config.get("ollama_host", "http://127.0.0.1:11434"),
        ),
        openai_base_url=os.environ.get(
            "OPENAI_BASE_URL",
            file_config.get("openai_base_url", "https://api.openai.com/v1"),
        ),
        openai_api_key=os.environ.get(
            "OPENAI_API_KEY",
            file_config.get("openai_api_key", ""),
        ),
        storage_dir=storage_dir,
        system_prompt=file_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        workspace_dir=str(resolved_workspace),
    )


def resolve_context_window_tokens(model: str) -> int:
    normalized_model = model.strip().lower()
    for alias, token_limit in sorted(
        MODEL_CONTEXT_WINDOW_TOKENS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if normalized_model == alias or normalized_model.startswith(f"{alias}-"):
            return token_limit
    return DEFAULT_CONTEXT_WINDOW_TOKENS
