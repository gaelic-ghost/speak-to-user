# Roadmap

This roadmap tracks likely next directions for `speak-to-user`. It is intentionally forward-looking and should not be read as a promise that every item will ship in this exact form.

## Near-Term Priorities

- Add persistent file storage for retained outputs and batch-generation workflows.
- Expose agent-facing MCP Prompts and Resources with guidance for chunking, voice usage, and reliable server interaction.
- Improve time-to-first-audio for the 1.7B voice-design path.
- Evaluate smaller or alternate TTS backends that reduce latency on Apple Silicon.
- Expand reusable profile support beyond clone prompts, including named voice-description presets for the voice-design path.

## Product Surface Expansion

- Add FastMCP Apps and UI support.
- Create a related agent skill that teaches consistent and correct usage patterns for this server.
- Add support for additional TTS runtimes beyond the current Qwen voice-design model.
- Publish the package on PyPI.
- Support composition with a FastAPI service.
- Explore packaging and distribution as a macOS `MenuBarExtra` app.
- Add saved-audio workflows when Gale explicitly wants retained outputs instead of immediate playback.

## Notes

- New layers, wrappers, managers, or service abstractions should be treated with caution. This project benefits from staying direct and understandable unless a stronger structure is clearly justified.
- When the roadmap starts turning into implementation work, the README should be updated alongside the code so the public project description stays accurate.
