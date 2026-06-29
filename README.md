# vib-code

A custom, lightweight terminal coding harness with switchable models, modeled around Vib's own workflow: plan first, track work as todos, then implement. Built for long-running tasks on low-intelligence, free models — keeping a small model on rails through structured JSON actions instead of free-form output.

## Highlights

- **Switchable models** — local Ollama or OpenAI-compatible backends, swapped per session via config or env vars.
- **Structured actions** — every model turn is one JSON object matching a strict schema (`list_files`, `read_file`, `search_text`, `create_file`, `replace_in_file`, `write_patch`, `run_command`).
- **Plan-first workflow** — lightweight planning and todos drive long-running tasks so weaker models stay on track.
- **Approval gates** — edits, patches, and commands pass an approval layer before touching the workspace.
- **Sessions & traces** — sessions are saved, resumable, and replayable per turn.

## Demos

| Adding a tool to vib-code | Building a calculator app |
| --- | --- |
| [<img src="https://cdn.loom.com/sessions/thumbnails/18fcfc53e9fa4437be3a781a68bf37e3-b4879187c607e18f.gif" width="360">](https://www.loom.com/share/18fcfc53e9fa4437be3a781a68bf37e3) | [<img src="https://cdn.loom.com/sessions/thumbnails/fc23349556024513a1b019dad4b8445e-5c6859e9ff18ed4f.gif" width="360">](https://www.loom.com/share/fc23349556024513a1b019dad4b8445e) |

## Run

```bash
python3 -m vib_code.cli chat                  # interactive session
python3 -m vib_code.cli --cwd path/to chat    # target a workspace
python3 -m vib_code.cli resume <session_id>   # resume a session
python3 -m vib_code.cli trace <session_id>    # inspect a trace
```

## Configuration

Set via `.vib-code-config.json` or env vars:

| Setting | Env var | Default |
| --- | --- | --- |
| Provider | `VIB_CODE_PROVIDER` | `ollama` |
| Model | `VIB_CODE_MODEL` | `qwen2.5:3b` |
| Ollama host | `VIB_CODE_OLLAMA_HOST` | `http://127.0.0.1:11434` |
| OpenAI base URL | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| OpenAI API key | `OPENAI_API_KEY` | — |
