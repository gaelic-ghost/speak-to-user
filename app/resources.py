from __future__ import annotations

# MARK: Imports

import json

from fastmcp import Context

from app.prompts import USAGE_GUIDE_TEXT
from app.tools import _runtime_from_context, _state_store_from_context


# MARK: Resource Implementations

def usage_guide_resource() -> str:
    return USAGE_GUIDE_TEXT


def status_resource(ctx: Context) -> str:
    return json.dumps(_runtime_from_context(ctx).status(), indent=2, sort_keys=True)


async def speech_profiles_resource(ctx: Context) -> str:
    profiles = await _runtime_from_context(ctx).list_speech_profiles(
        state_store=_state_store_from_context(ctx),
    )
    return json.dumps(profiles, indent=2, sort_keys=True)
