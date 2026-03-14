# Roadmap

This roadmap tracks likely next directions for `speak-to-user`. It is intentionally forward-looking and should not be read as a promise that every item will ship in this exact form.

## Near-Term Priorities

- Add a speech-generation queue so concurrent `speak_text` and `generate_audio` requests can be handled more predictably.
- Add persistent file storage for retained outputs and batch-generation workflows.
- Expose agent-facing MCP Prompts and Resources with guidance for chunking, voice usage, and reliable server interaction.
- Add voice description profile support so repeated usage patterns can be named and reused instead of rewritten each time.
- Support voice profile selection through MCP user elicitation.
- Support automatic voice profile switching with FastMCP Context.

## Product Surface Expansion

- Add FastMCP Apps and UI support.
- Create a related agent skill that teaches consistent and correct usage patterns for this server.
- Add support for additional TTS runtimes beyond the current Qwen voice-design model.
- Publish the package on PyPI.
- Support composition with a FastAPI service.
- Explore packaging and distribution as a macOS `MenuBarExtra` app.

## Notes

- New layers, wrappers, managers, or service abstractions should be treated with caution. This project benefits from staying direct and understandable unless a stronger structure is clearly justified.
- When the roadmap starts turning into implementation work, the README should be updated alongside the code so the public project description stays accurate.
