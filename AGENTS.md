# AGENTS.md

## Purpose

This repository is a local Python FastMCP service built and maintained with uv-first workflows. Keep changes direct, small, and easy to reason about.

## Stack

- Python `3.12`
- `uv` for environment, dependency, and command execution
- `FastMCP` for the MCP server surface
- `pytest`, `ruff`, and `mypy` as the baseline validation path

## Working Rules

- Use `uv run ...` for Python commands instead of calling `python` directly.
- Prefer direct FastMCP tools, resources, and prompts over extra wrapper layers.
- Be cautious about adding new abstractions, managers, services, or dependencies; this repo benefits from staying small and explicit.
- Keep MCP-facing behavior and docs aligned. If tool signatures or behavior change, update `README.md` in the same pass.
- Leave unrelated user changes alone. Do not revert work you did not make.

## Validation

Run these checks before committing:

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```

- Never run tests in parallel in this repository when they may load the TTS model or other large runtime state into memory. Gale's machine has 24GB of RAM, and overlapping model loads can exhaust memory and destabilize the system.

## FastMCP Guidance

- Use the `fastmcp_docs` MCP server for up-to-date FastMCP behavior and APIs.
- Prefer protocol-native FastMCP features when they fit, especially for tasks, progress reporting, prompts, and resources.
- Keep the MCP surface small and predictable unless the user explicitly asks to expand it.

## Project Notes

- `generate_audio` is the explicit file-producing tool.
- `speak_text` is the playback-oriented tool and should not expose persistent-output controls unless the user explicitly wants that behavior.
- Text chunking helpers live in `app/text_chunking.py`; MCP/runtime adapters live in `app/tools.py`.
