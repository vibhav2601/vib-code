from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .models import ChatMessage, SessionRecord, utc_now_iso


class MemoryStore:
    def __init__(self, storage_dir: Path, legacy_storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir
        self.sessions_dir = storage_dir / "sessions"
        self.traces_dir = storage_dir / "traces"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_storage_dir(legacy_storage_dir)
        self._migrate_legacy_storage_names()

    def create_session(self, model: str, working_directory: str, title: str = "New session") -> SessionRecord:
        now = utc_now_iso()
        session_id = uuid.uuid4().hex[:12]
        storage_name = self._storage_name(now, session_id)
        trace_path = self.traces_dir / f"{storage_name}.jsonl"
        trace_path.touch(exist_ok=True)
        session = SessionRecord(
            session_id=session_id,
            title=title,
            model=model,
            created_at=now,
            updated_at=now,
            working_directory=working_directory,
            trace_path=str(trace_path),
            storage_name=storage_name,
        )
        self.save_session(session)
        return session

    def save_session(self, session: SessionRecord) -> None:
        session.updated_at = utc_now_iso()
        if not session.storage_name:
            session.storage_name = self._storage_name(session.created_at, session.session_id)
        session.trace_path = str(self.traces_dir / f"{session.storage_name}.jsonl")
        path = self.sessions_dir / f"{session.storage_name}.json"
        path.write_text(json.dumps(asdict(session), indent=2))

    def load_session(self, session_id: str) -> SessionRecord:
        path = self._find_session_file(session_id)
        data = json.loads(path.read_text())
        storage_name = data.setdefault("storage_name", path.stem)
        data.setdefault("trace_path", str(self.traces_dir / f"{storage_name}.jsonl"))
        messages = [ChatMessage(**item) for item in data.pop("messages", [])]
        return SessionRecord(messages=messages, **data)

    def list_sessions(self) -> list[SessionRecord]:
        sessions: list[SessionRecord] = []
        for path in sorted(self.sessions_dir.glob("*.json")):
            data = json.loads(path.read_text())
            session_id = data.get("session_id", path.stem)
            storage_name = data.setdefault("storage_name", path.stem)
            data.setdefault("trace_path", str(self.traces_dir / f"{storage_name}.jsonl"))
            messages = [ChatMessage(**item) for item in data.pop("messages", [])]
            sessions.append(SessionRecord(messages=messages, **data))
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions

    def trace_path_for_session(self, session_id: str) -> Path:
        return Path(self.load_session(session_id).trace_path)

    def _find_session_file(self, session_id: str) -> Path:
        direct_path = self.sessions_dir / f"{session_id}.json"
        if direct_path.exists():
            return direct_path

        matches = sorted(self.sessions_dir.glob(f"*_{session_id}.json"))
        if matches:
            return matches[-1]

        for path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if data.get("session_id") == session_id:
                return path
        raise FileNotFoundError(f"No saved session found for id {session_id}")

    def _migrate_legacy_storage_dir(self, legacy_storage_dir: Path | None) -> None:
        if legacy_storage_dir is None:
            return
        try:
            legacy_dir = legacy_storage_dir.resolve()
        except FileNotFoundError:
            return
        if legacy_dir == self.storage_dir.resolve() or not legacy_dir.exists():
            return

        for name in ("sessions", "traces"):
            legacy_subdir = legacy_dir / name
            target_subdir = self.storage_dir / name
            if not legacy_subdir.exists():
                continue
            target_subdir.mkdir(parents=True, exist_ok=True)
            for path in legacy_subdir.iterdir():
                destination = target_subdir / path.name
                if destination.exists():
                    continue
                path.rename(destination)

        if legacy_dir.exists() and not any(legacy_dir.iterdir()):
            legacy_dir.rmdir()

    def _migrate_legacy_storage_names(self) -> None:
        for session_path in list(self.sessions_dir.glob("*.json")):
            try:
                data = json.loads(session_path.read_text())
            except json.JSONDecodeError:
                continue

            session_id = data.get("session_id", session_path.stem)
            created_at = data.get("created_at") or data.get("updated_at") or utc_now_iso()
            storage_name = data.get("storage_name") or self._storage_name(created_at, session_id)
            desired_session_path = self.sessions_dir / f"{storage_name}.json"
            desired_trace_path = self.traces_dir / f"{storage_name}.jsonl"

            original_trace_path = Path(
                data.get("trace_path", str(self.traces_dir / f"{session_id}.jsonl"))
            )
            if original_trace_path.exists() and original_trace_path != desired_trace_path:
                desired_trace_path.parent.mkdir(parents=True, exist_ok=True)
                if not desired_trace_path.exists():
                    original_trace_path.rename(desired_trace_path)
                else:
                    original_trace_path.unlink()

            data["storage_name"] = storage_name
            data["trace_path"] = str(desired_trace_path)
            desired_session_path.write_text(json.dumps(data, indent=2))
            if session_path != desired_session_path:
                session_path.unlink()

    def _storage_name(self, timestamp_iso: str, session_id: str) -> str:
        return f"{self._timestamp_slug(timestamp_iso)}_{session_id}"

    def _timestamp_slug(self, timestamp_iso: str) -> str:
        try:
            parsed = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        except ValueError:
            return "unknown-time"
        return parsed.strftime("%Y-%m-%d_%H-%M-%S")
