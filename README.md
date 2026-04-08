# vib-code

`vib-code` is a local terminal coding harness that drives a model through structured JSON actions instead of free-form shell access.

It is designed for repository inspection and file editing workflows, with a small built-in toolset for listing files, reading files, searching text, applying exact replacements, writing unified diff patches, and running approved commands. The harness supports both Ollama and OpenAI backends, keeps per-session traces, and uses runtime state plus strict action schemas to keep model behavior deterministic.

## Highlights

- Structured tool calls instead of ad hoc text parsing
- Local terminal loop with session persistence and trace logging
- File-aware edit workflow with approval gates for mutating actions
- Support for both Ollama and OpenAI chat backends
- Lightweight Python package with a `vib-code` CLI entrypoint

## Run

From the repository root:

```bash
python3 -m vib_code.cli chat
```

To target the package directory as the working workspace:

```bash
python3 -m vib_code.cli --cwd vib_code chat
```
