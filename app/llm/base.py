from typing import Protocol

from app.llm.schemas import AdapterRequest, AdapterResponse


class LLMAdapter(Protocol):
    @property
    def adapter_name(self) -> str:
        raise NotImplementedError

    def start_run(self) -> None:
        raise NotImplementedError

    def complete(self, request: AdapterRequest) -> AdapterResponse:
        raise NotImplementedError
