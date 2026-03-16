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
    health_payload,
    speak_text as speak_text_tool,
    speak_text_as_clone as speak_text_as_clone_tool,
    tts_status as tts_status_tool,
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
    runtime.preload()
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


@mcp.tool
def tts_status(ctx: Context = current_context) -> dict[str, object]:
    """Return current TTS runtime state."""
    return tts_status_tool(ctx)


@mcp.tool
def speak_text(
    text: str,
    voice_description: str,
    language: str = "en",
    ctx: Context = current_context,
) -> dict[str, object]:
    """Queue one full text job for local audio playback on this machine."""
    return speak_text_tool(
        ctx,
        text=text,
        voice_description=voice_description,
        language=language,
    )


@mcp.tool
def speak_text_as_clone(
    text: str,
    reference_audio_path: str,
    reference_text: str | None = None,
    language: str = "en",
    ctx: Context = current_context,
) -> dict[str, object]:
    """Queue one full text job for local playback using the clone voice model."""
    return speak_text_as_clone_tool(
        ctx,
        text=text,
        reference_audio_path=reference_audio_path,
        reference_text=reference_text,
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
