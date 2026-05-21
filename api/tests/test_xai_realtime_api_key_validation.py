from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from api.services.configuration.check_validity import UserConfigurationValidator
from api.services.configuration.registry import ServiceProviders


def _openai_error(exc_cls, status_code: int):
    request = httpx.Request("GET", "https://api.x.ai/v1/models")
    response = httpx.Response(status_code, request=request)
    return exc_cls("boom", response=response, body=None)


def test_xai_realtime_validator_is_registered():
    validator = UserConfigurationValidator()
    # Registered under the string value, and resolvable via the str-enum member
    # (which is how the provider arrives from the config object).
    assert (
        validator._validator_map.get(ServiceProviders.XAI_REALTIME.value) is not None
    )
    assert validator._validator_map.get(ServiceProviders.XAI_REALTIME) is not None


@patch("api.services.configuration.check_validity.openai.OpenAI")
def test_xai_realtime_valid_key_returns_true(mock_openai):
    mock_openai.return_value.models.list.return_value = MagicMock()
    validator = UserConfigurationValidator()

    assert validator._check_xai_realtime_api_key("m", "good-key") is True
    # Uses the xAI OpenAI-compatible endpoint.
    _, kwargs = mock_openai.call_args
    assert kwargs["base_url"] == "https://api.x.ai/v1"


@pytest.mark.parametrize(
    "exc_cls,status",
    [
        (openai.BadRequestError, 400),  # xAI's signal for an incorrect key
        (openai.AuthenticationError, 401),
        (openai.PermissionDeniedError, 403),
    ],
)
@patch("api.services.configuration.check_validity.openai.OpenAI")
def test_xai_realtime_bad_key_returns_false(mock_openai, exc_cls, status):
    mock_openai.return_value.models.list.side_effect = _openai_error(exc_cls, status)
    validator = UserConfigurationValidator()

    assert validator._check_xai_realtime_api_key("m", "bad-key") is False
