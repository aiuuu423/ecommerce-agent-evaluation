from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel

from app.tools.schemas import PriorToolExecution

if TYPE_CHECKING:
    from app.data.database import Catalog

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class UnknownToolError(LookupError):
    """Registry could not resolve a requested tool name."""


class ToolInputValidationError(ValueError):
    """Registry input boundary rejected tool arguments."""


class ToolOutputValidationError(RuntimeError):
    """Registry output boundary rejected a handler result."""


@dataclass(frozen=True)
class ToolDefinition(Generic[InputT, OutputT]):
    name: str
    description: str
    input_model: type[InputT]
    output_model: type[OutputT]
    handler: Callable[[InputT, ToolContext], OutputT]
    output_validator: Callable[[InputT, OutputT], None] | None = None


@dataclass(frozen=True)
class ToolContext:
    catalog: Catalog
    prior_executions: tuple[PriorToolExecution, ...]
    next_result_id: Callable[[], str]
