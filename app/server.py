from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext
from fastmcp.server.lifespan import lifespan

from app.resources import speak_text_process_resource
from app.runtime import TTSRuntime
from app.tools import (
    generate_audio as generate_audio_tool,
    health_payload,
    load_model as load_model_tool,
    set_idle_unload_timeout as set_idle_unload_timeout_tool,
    speak_text as speak_text_tool,
    tts_status as tts_status_tool,
    unload_model as unload_model_tool,
)

@lifespan
async def app_lifespan(_server: FastMCP):
    runtime = TTSRuntime.from_env()
    runtime.start_background_preload()
    try:
        yield {"runtime": runtime}
    finally:
        runtime.shutdown()


current_context = CurrentContext()
mcp = FastMCP(
    "speak-to-user",
    lifespan=app_lifespan,
)


@mcp.tool
def health() -> dict[str, str]:
    """Return a lightweight health payload for smoke testing."""
    return health_payload()


@mcp.resource(
    "info://speak-text-process",
    name="speak_text process",
    description="Agent-facing explanation of the detached local playback path used by speak_text.",
    mime_type="text/markdown",
)
def speak_text_process() -> str:
    return speak_text_process_resource()


@mcp.tool
def tts_status(ctx: Context = current_context) -> dict[str, object]:
    """Return current TTS runtime state."""
    return tts_status_tool(ctx)


@mcp.tool
def load_model(ctx: Context = current_context) -> dict[str, object]:
    """Load the TTS model into memory."""
    return load_model_tool(ctx)


@mcp.tool
def unload_model(ctx: Context = current_context) -> dict[str, object]:
    """Unload the TTS model from memory."""
    return unload_model_tool(ctx)


@mcp.tool
def set_idle_unload_timeout(
    seconds: int,
    ctx: Context = current_context,
) -> dict[str, object]:
    """Update the idle-unload timeout in seconds."""
    return set_idle_unload_timeout_tool(ctx, seconds)


@mcp.tool
def generate_audio(
    text: str,
    voice_description: str,
    language: str = "en",
    output_format: str = "wav",
    filename_stem: str | None = None,
    ctx: Context = current_context,
) -> dict[str, object]:
    """Generate one local audio file with the voice-design model."""
    return generate_audio_tool(
        ctx,
        text=text,
        voice_description=voice_description,
        language=language,
        output_format=output_format,
        filename_stem=filename_stem,
    )


@mcp.tool
def speak_text(
    text: str,
    voice_description: str,
    language: str = "en",
    ctx: Context = current_context,
) -> dict[str, object]:
    """Queue one detached speech job for local playback on this machine.

    The MCP call returns after handing the request to a detached local playback process.
    That helper process performs one model batch call for the full request and keeps
    host playback alive even after the stdio MCP session exits.
    """
    return speak_text_tool(
        ctx,
        text=text,
        voice_description=voice_description,
        language=language,
    )


if __name__ == "__main__":
    mcp.run()
