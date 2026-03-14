from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext, Progress
from fastmcp.server.lifespan import lifespan
from fastmcp.server.tasks import TaskConfig
from key_value.aio.stores.memory import MemoryStore

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


cache_store = MemoryStore()
current_context = CurrentContext()
task_progress = Progress()
mcp = FastMCP(
    "speak-to-user",
    lifespan=app_lifespan,
    session_state_store=cache_store,
)


@mcp.tool
def health() -> dict[str, str]:
    """Return a lightweight health payload for smoke testing."""
    return health_payload()


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


@mcp.tool(task=TaskConfig(mode="optional"))
async def speak_text(
    text: str,
    voice_description: str,
    language: str = "en",
    ctx: Context = current_context,
    progress: Progress = task_progress,
) -> dict[str, object]:
    """Queue one speech job for local playback on this machine.

    One queue slot equals one full `speak_text` request, including its full chunk list.
    If too many requests are already pending, the tool rejects the new job and the client
    should try again later. Playback then proceeds chunk by chunk on the background worker,
    and clients can poll `tts_status` for active chunk progress.
    """
    return await speak_text_tool(
        ctx,
        progress,
        text=text,
        voice_description=voice_description,
        language=language,
    )


if __name__ == "__main__":
    mcp.run()
