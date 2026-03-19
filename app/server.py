from dataclasses import dataclass
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext
from fastmcp.server.lifespan import lifespan

from app.runtime import TTSRuntime
from app.tools import (
    choose_speak_to_user_workflow_prompt as choose_speak_to_user_workflow_prompt_tool,
    delete_speech_profile as delete_speech_profile_tool,
    generate_speech_profile as generate_speech_profile_tool,
    generate_speech_profile_from_voice_design as generate_speech_profile_from_voice_design_tool,
    guide_speech_profile_workflow_prompt as guide_speech_profile_workflow_prompt_tool,
    health_payload,
    load_model as load_model_tool,
    list_speech_profiles as list_speech_profiles_tool,
    speak_text as speak_text_tool,
    speak_text_as_clone as speak_text_as_clone_tool,
    speech_profiles_resource as speech_profiles_resource_tool,
    speak_with_profile as speak_with_profile_tool,
    status_resource as status_resource_tool,
    set_startup_model as set_startup_model_tool,
    tts_status as tts_status_tool,
    unload_model as unload_model_tool,
    usage_guide_resource as usage_guide_resource_tool,
)


# MARK: Server Configuration

DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8765
DEFAULT_HTTP_PATH = "/mcp"


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str
    port: int
    path: str


def _normalize_http_host(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("SPEAK_TO_USER_HOST must not be empty")
    return normalized


def _normalize_http_port(value: str | None) -> int:
    if value is None:
        return DEFAULT_HTTP_PORT

    normalized = value.strip()
    if not normalized:
        raise ValueError("SPEAK_TO_USER_PORT must not be empty")

    try:
        port = int(normalized)
    except ValueError as exc:
        raise ValueError("SPEAK_TO_USER_PORT must be an integer") from exc

    if not 1 <= port <= 65535:
        raise ValueError("SPEAK_TO_USER_PORT must be between 1 and 65535")
    return port


def _normalize_http_path(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("SPEAK_TO_USER_MCP_PATH must not be empty")
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized


def server_config_from_env() -> ServerConfig:
    return ServerConfig(
        host=_normalize_http_host(os.getenv("SPEAK_TO_USER_HOST", DEFAULT_HTTP_HOST)),
        port=_normalize_http_port(os.getenv("SPEAK_TO_USER_PORT")),
        path=_normalize_http_path(os.getenv("SPEAK_TO_USER_MCP_PATH", DEFAULT_HTTP_PATH)),
    )


@lifespan
async def app_lifespan(_server: FastMCP):
    runtime = TTSRuntime.from_env()
    await runtime.preload(state_store=_server._state_store)
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


@mcp.prompt
def choose_speak_to_user_workflow() -> str:
    """Guide an agent toward the right speak-to-user tool for the job."""
    return choose_speak_to_user_workflow_prompt_tool()


@mcp.prompt
def guide_speech_profile_workflow() -> str:
    """Guide an agent through profile creation and reuse decisions."""
    return guide_speech_profile_workflow_prompt_tool()


@mcp.resource("guide://speak-to-user/usage")
def usage_guide() -> str:
    """Provide a compact usage guide for speak-to-user."""
    return usage_guide_resource_tool()


@mcp.resource("state://speak-to-user/status", mime_type="application/json")
def status_resource(ctx: Context = current_context) -> str:
    """Provide a read-only runtime status snapshot for agents."""
    return status_resource_tool(ctx)


@mcp.resource("state://speak-to-user/profiles", mime_type="application/json")
async def speech_profiles_resource(ctx: Context = current_context) -> str:
    """Provide a read-only saved-profile summary for agents."""
    return await speech_profiles_resource_tool(ctx)


@mcp.tool
async def generate_speech_profile(
    name: str,
    reference_audio_path: str,
    reference_text: str | None = None,
    ctx: Context = current_context,
) -> dict[str, object]:
    """Create a named reusable speech profile from clone reference audio."""
    return await generate_speech_profile_tool(
        ctx,
        name=name,
        reference_audio_path=reference_audio_path,
        reference_text=reference_text,
    )


@mcp.tool
async def generate_speech_profile_from_voice_design(
    name: str,
    text: str,
    voice_description: str,
    language: str = "en",
    ctx: Context = current_context,
) -> dict[str, object]:
    """Create a named reusable speech profile by synthesizing a voice-designed seed clip."""
    return await generate_speech_profile_from_voice_design_tool(
        ctx,
        name=name,
        text=text,
        voice_description=voice_description,
        language=language,
    )


@mcp.tool
async def list_speech_profiles(ctx: Context = current_context) -> dict[str, object]:
    """List saved reusable speech profiles."""
    return await list_speech_profiles_tool(ctx)


@mcp.tool
async def delete_speech_profile(
    name: str,
    ctx: Context = current_context,
) -> dict[str, object]:
    """Delete a saved reusable speech profile by name."""
    return await delete_speech_profile_tool(
        ctx,
        name=name,
    )


@mcp.tool
def tts_status(ctx: Context = current_context) -> dict[str, object]:
    """Return current TTS runtime state."""
    return tts_status_tool(ctx)


@mcp.tool
def load_model(
    model_id: str,
    ctx: Context = current_context,
) -> dict[str, object]:
    """Load one resident voice model by model id."""
    return load_model_tool(
        ctx,
        model_id=model_id,
    )


@mcp.tool
def unload_model(
    model_id: str,
    ctx: Context = current_context,
) -> dict[str, object]:
    """Unload one resident voice model by model id."""
    return unload_model_tool(
        ctx,
        model_id=model_id,
    )


@mcp.tool
async def set_startup_model(
    option: str,
    ctx: Context = current_context,
) -> dict[str, object]:
    """Persist which model or models preload on server startup."""
    return await set_startup_model_tool(
        ctx,
        option=option,
    )


@mcp.tool
async def speak_text(
    text: str,
    voice_description: str,
    language: str = "en",
    ctx: Context = current_context,
) -> dict[str, object]:
    """Queue one full text job for local audio playback on this machine."""
    return await speak_text_tool(
        ctx,
        text=text,
        voice_description=voice_description,
        language=language,
    )


@mcp.tool
async def speak_text_as_clone(
    text: str,
    reference_audio_path: str,
    reference_text: str | None = None,
    language: str = "en",
    ctx: Context = current_context,
) -> dict[str, object]:
    """Queue one full text job for local playback using the clone voice model."""
    return await speak_text_as_clone_tool(
        ctx,
        text=text,
        reference_audio_path=reference_audio_path,
        reference_text=reference_text,
        language=language,
    )


@mcp.tool
async def speak_with_profile(
    name: str,
    text: str,
    language: str = "en",
    ctx: Context = current_context,
) -> dict[str, object]:
    """Queue one full text job for local playback using a saved speech profile."""
    return await speak_with_profile_tool(
        ctx,
        name=name,
        text=text,
        language=language,
    )


def main() -> None:
    config = server_config_from_env()
    mcp.run(
        transport="http",
        host=config.host,
        port=config.port,
        path=config.path,
    )


if __name__ == "__main__":
    main()
