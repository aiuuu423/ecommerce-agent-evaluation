from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_TIMEOUT_SECONDS = 30.0


class ProviderConfigError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ProviderSettings:
    api_key: str
    model: str
    base_url: str
    temperature: float
    timeout_seconds: float


def base_url_host(base_url: str) -> str:
    try:
        return urlsplit(base_url).hostname or ""
    except ValueError:
        return ""


def load_provider_settings(environ: Mapping[str, str]) -> ProviderSettings:
    api_key = environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise ProviderConfigError("missing_api_key")

    model = environ.get("LLM_MODEL", "").strip()
    if not model:
        raise ProviderConfigError("missing_model")

    raw_base_url = environ.get("LLM_BASE_URL")
    if raw_base_url is None or not raw_base_url.strip():
        base_url = DEFAULT_BASE_URL
    else:
        base_url = raw_base_url.strip().rstrip("/")
    try:
        parsed_base_url = urlsplit(base_url)
        if (
            parsed_base_url.scheme not in {"http", "https"}
            or not parsed_base_url.hostname
            or parsed_base_url.username is not None
            or parsed_base_url.password is not None
        ):
            raise ProviderConfigError("invalid_base_url")
    except ValueError:
        raise ProviderConfigError("invalid_base_url") from None

    try:
        temperature = float(environ.get("LLM_TEMPERATURE", "0"))
    except ValueError:
        raise ProviderConfigError("invalid_temperature") from None
    if not isfinite(temperature) or temperature != 0.0:
        raise ProviderConfigError("invalid_temperature")

    try:
        timeout_seconds = float(
            environ.get("LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        )
    except ValueError:
        raise ProviderConfigError("invalid_timeout") from None
    if not isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ProviderConfigError("invalid_timeout")

    return ProviderSettings(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )
