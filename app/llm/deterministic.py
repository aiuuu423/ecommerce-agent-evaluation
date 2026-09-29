from collections.abc import Callable

from app.llm.schemas import (
    AdapterRequest,
    AdapterResponse,
    AssistantAction,
)

DeterministicPolicy = Callable[[AdapterRequest], AssistantAction]


class DeterministicAdapter:
    def __init__(self, policy: DeterministicPolicy) -> None:
        if not callable(policy):
            raise TypeError("policy must be callable")
        self._policy = policy

    @property
    def adapter_name(self) -> str:
        return "deterministic"

    def start_run(self) -> None:
        pass

    def complete(self, request: AdapterRequest) -> AdapterResponse:
        action = self._policy(request)
        return AdapterResponse(action=action, raw_response=None, usage=None)
