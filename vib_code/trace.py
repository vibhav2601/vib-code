from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .models import SessionRecord, utc_now_iso


class TraceWriter:
    def __init__(self, trace_path: Path) -> None:
        self.trace_path = trace_path
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.touch(exist_ok=True)

    def append(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "event_id": uuid.uuid4().hex,
            "timestamp": utc_now_iso(),
            "session_id": session_id,
            "event_type": event_type,
            "payload": payload,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def append_session_event(self, session: SessionRecord, event_type: str) -> None:
        self.append(
            session.session_id,
            event_type,
            {
                "title": session.title,
                "model": session.model,
                "working_directory": session.working_directory,
                "trace_path": session.trace_path,
            },
        )

    def read_text(self) -> str:
        return self.trace_path.read_text(encoding="utf-8")
