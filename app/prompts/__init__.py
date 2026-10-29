"""Versioned prompt definitions and snapshots."""

from app.prompts.v1 import (
    V1_PROMPT_NAME,
    V1_PROMPT_VERSION,
    v1_prompt_snapshot,
)
from app.prompts.v2 import (
    V2_PROMPT_NAME,
    V2_PROMPT_VERSION,
    v2_prompt_snapshot,
)

__all__ = [
    "V1_PROMPT_NAME",
    "V1_PROMPT_VERSION",
    "v1_prompt_snapshot",
    "V2_PROMPT_NAME",
    "V2_PROMPT_VERSION",
    "v2_prompt_snapshot",
]
