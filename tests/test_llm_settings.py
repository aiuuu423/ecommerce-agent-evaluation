from dataclasses import FrozenInstanceError

import pytest

from app.experiments.llm_settings import (
    DEFAULT_BASE_URL,
    ProviderConfigError,
    ProviderSettings,
    base_url_host,
    load_provider_settings,
)


def valid_environ(**overrides: str) -> dict[str, str]:
    environ = {
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "ep-2026xxx",
    }
    environ.update(overrides)
    return environ


def test_load_provider_settings_uses_defaults_and_returns_frozen_settings() -> None:
    settings = load_provider_settings(valid_environ())

    assert settings == ProviderSettings(
        api_key="test-key",
        model="ep-2026xxx",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        temperature=0.0,
        timeout_seconds=30.0,
    )
    with pytest.raises(FrozenInstanceError):
        settings.model = "another-model"


def test_base_url_host_returns_only_hostname() -> None:
    assert base_url_host(DEFAULT_BASE_URL) == "ark.cn-beijing.volces.com"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"LLM_API_KEY": ""}, "missing_api_key"),
        ({"LLM_API_KEY": " \t "}, "missing_api_key"),
        ({"LLM_MODEL": ""}, "missing_model"),
        ({"LLM_MODEL": " \n "}, "missing_model"),
        ({"LLM_TEMPERATURE": "0.1"}, "invalid_temperature"),
        ({"LLM_TEMPERATURE": "1"}, "invalid_temperature"),
        ({"LLM_TEMPERATURE": "nan"}, "invalid_temperature"),
        ({"LLM_TEMPERATURE": "inf"}, "invalid_temperature"),
        ({"LLM_TEMPERATURE": "-inf"}, "invalid_temperature"),
        ({"LLM_TIMEOUT_SECONDS": "0"}, "invalid_timeout"),
        ({"LLM_TIMEOUT_SECONDS": "-1"}, "invalid_timeout"),
        ({"LLM_TIMEOUT_SECONDS": "abc"}, "invalid_timeout"),
        ({"LLM_TIMEOUT_SECONDS": "nan"}, "invalid_timeout"),
        ({"LLM_TIMEOUT_SECONDS": "inf"}, "invalid_timeout"),
        ({"LLM_TIMEOUT_SECONDS": "-inf"}, "invalid_timeout"),
        ({"LLM_BASE_URL": "not-a-url"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "///"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "file:///tmp/x"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://user@example.com/v1"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://user:password@example.com/v1"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "http://["}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://example.test:abc/v1"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://example.test:99999/v1"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://example.test/v1?x=1"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://example.test/v1#frag"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://example.test/v1?"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://example.test/v1#"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://exam ple.test/v1"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://exam\tple.test/v1"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://exam\nple.test/v1"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://exam\rple.test/v1"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://example.test\\evil/v1"}, "invalid_base_url"),
        ({"LLM_BASE_URL": "https://example\x7f.test/v1"}, "invalid_base_url"),
    ],
)
def test_load_provider_settings_rejects_invalid_values(
    overrides: dict[str, str],
    code: str,
) -> None:
    with pytest.raises(ProviderConfigError) as caught:
        load_provider_settings(valid_environ(**overrides))

    assert caught.value.code == code
    assert str(caught.value) == code
    assert "test-key" not in repr(caught.value)
    if "LLM_BASE_URL" in overrides:
        assert overrides["LLM_BASE_URL"] not in repr(caught.value)


def test_load_provider_settings_strips_values_and_normalizes_base_url() -> None:
    settings = load_provider_settings(
        {
            "LLM_API_KEY": "  test-key  ",
            "LLM_MODEL": "  ep-2026xxx  ",
            "LLM_BASE_URL": "  http://localhost:8080/v1///  ",
            "LLM_TEMPERATURE": " 0 ",
            "LLM_TIMEOUT_SECONDS": " 15.5 ",
        }
    )

    assert settings == ProviderSettings(
        api_key="test-key",
        model="ep-2026xxx",
        base_url="http://localhost:8080/v1",
        temperature=0.0,
        timeout_seconds=15.5,
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8080/v1",
        "https://example.test/v1",
        "http://127.0.0.1:8080/v1",
        "http://[::1]:8080/v1",
    ],
)
def test_load_provider_settings_accepts_supported_host_forms(base_url: str) -> None:
    assert (
        load_provider_settings(valid_environ(LLM_BASE_URL=base_url)).base_url
        == base_url
    )


@pytest.mark.parametrize("base_url", [None, "", " \t "])
def test_load_provider_settings_uses_default_for_missing_or_blank_base_url(
    base_url: str | None,
) -> None:
    environ = valid_environ()
    if base_url is not None:
        environ["LLM_BASE_URL"] = base_url

    assert load_provider_settings(environ).base_url == DEFAULT_BASE_URL


def test_base_url_host_returns_empty_string_for_unparseable_url() -> None:
    assert base_url_host("http://[") == ""
