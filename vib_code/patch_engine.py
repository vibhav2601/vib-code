from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path


_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


class PatchError(ValueError):
    pass


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


class PatchEngine:
    def preview(self, path: Path, patch: str) -> str:
        original_text = path.read_text(encoding="utf-8", errors="replace")
        updated_text = self.apply_to_text(original_text, patch)
        if updated_text == original_text:
            raise PatchError("Patch would not change the file.")
        original_lines = original_text.splitlines()
        updated_lines = updated_text.splitlines()
        diff = difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
        )
        return "\n".join(diff)

    def apply(self, path: Path, patch: str) -> None:
        original_text = path.read_text(encoding="utf-8", errors="replace")
        updated_text = self.apply_to_text(original_text, patch)
        path.write_text(updated_text, encoding="utf-8")

    def apply_to_text(self, original_text: str, patch: str) -> str:
        original_lines = original_text.splitlines()
        hunks = self._parse_hunks(patch)
        cursor = 0
        output: list[str] = []
        for hunk in hunks:
            target_index = hunk.old_start - 1
            if target_index < cursor or target_index > len(original_lines):
                raise PatchError("Patch hunk is out of range for the current file.")
            output.extend(original_lines[cursor:target_index])
            cursor = target_index
            for raw_line in hunk.lines:
                if not raw_line:
                    raise PatchError("Patch contains an empty diff line.")
                prefix = raw_line[0]
                content = raw_line[1:]
                if prefix == " ":
                    self._expect_line(original_lines, cursor, content)
                    output.append(original_lines[cursor])
                    cursor += 1
                elif prefix == "-":
                    self._expect_line(original_lines, cursor, content)
                    cursor += 1
                elif prefix == "+":
                    output.append(content)
                else:
                    raise PatchError(f"Unsupported patch line prefix: {prefix!r}")
            output.extend(original_lines[cursor : hunk.old_start - 1 + hunk.old_count])
            cursor = hunk.old_start - 1 + hunk.old_count
        output.extend(original_lines[cursor:])
        result = "\n".join(output)
        if original_text.endswith("\n"):
            result += "\n"
        return result

    def _parse_hunks(self, patch: str) -> list[Hunk]:
        raw_lines = patch.splitlines()
        lines: list[str] = []
        for line in raw_lines:
            if line.startswith("--- ") or line.startswith("+++ "):
                continue
            if line == r"\ No newline at end of file":
                continue
            lines.append(line)

        hunks: list[Hunk] = []
        current: Hunk | None = None
        for line in lines:
            match = _HUNK_RE.match(line)
            if match:
                if current is not None:
                    hunks.append(current)
                current = Hunk(
                    old_start=int(match.group("old_start")),
                    old_count=int(match.group("old_count") or "1"),
                    new_start=int(match.group("new_start")),
                    new_count=int(match.group("new_count") or "1"),
                    lines=[],
                )
                continue
            if current is None:
                raise PatchError("Patch must start with a unified diff hunk header.")
            current.lines.append(line)

        if current is not None:
            hunks.append(current)
        if not hunks:
            raise PatchError("Patch did not contain any hunks.")
        return hunks

    def _expect_line(self, original_lines: list[str], index: int, expected: str) -> None:
        if index >= len(original_lines):
            raise PatchError("Patch expected more source lines than the file contains.")
        actual = original_lines[index]
        if actual != expected:
            raise PatchError(
                f"Patch context mismatch at line {index + 1}: expected {expected!r}, found {actual!r}"
            )
