from __future__ import annotations

# MARK: Imports

import json

from fastmcp import Context

from app.prompts import USAGE_GUIDE_TEXT
from app.tools import list_speech_profiles, tts_status


# MARK: Resource Implementations

def usage_guide_resource() -> str:
    return USAGE_GUIDE_TEXT


def status_resource(ctx: Context) -> str:
    return json.dumps(tts_status(ctx), indent=2, sort_keys=True)


async def speech_profiles_resource(ctx: Context) -> str:
    profiles = await list_speech_profiles(ctx)
    return json.dumps(profiles, indent=2, sort_keys=True)
