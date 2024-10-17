import pytest
from openai import BadRequestError
from openai.types import ChatCompletionResponse

from api.api_management import call_gpt_api, error_handle, NoMessageError

from api._proto.auto_chunker_pb2 import ChunkResponse


@pytest.fixture
def mock_error():
    return BadRequestError("Bad request")


@pytest.fixture
def mock_custom_error():
    class CustomError(Exception):
        pass

    return CustomError("Custom error")


@pytest.fixture
def fake_client():
    class FakeClient:
        def call_api(self, messages):
            return ChatCompletionResponse(
                choices=[{"message": {"content": "Generated content"}}]
            )

    return FakeClient()


@pytest.fixture
def fake_client_without_response():
    class FakeClient:
        def call_api(self, messages):
            return ChatCompletionResponse(
                choices=[{"message": {"content": ""}}]
            )

    return FakeClient()


class TestErrorHandle:
    def test_unresolvable_error_handling(self, mocker, mock_error):
        result = error_handle(mock_error)
        assert isinstance(result, ChunkResponse)
        assert "A critical error has occurred" in result.status_message

    def test_resolvable_error_handling(self, mocker, mock_error):
        class CustomError(Exception):
            pass

        mock_error = CustomError("Custom error without response")
        with mocker.patch("api.api_management.time.sleep"):
            result = error_handle(mock_error)
        assert isinstance(result, int)
        assert result == 1

    def test_logs_error_details(self, mocker, mock_error):
        result = error_handle(mock_error)
        assert "A critical error has occurred" in result.status_message

    def test_error_handle_unknown_status(self, mocker, mock_error):
        result = error_handle(mock_error)
        assert isinstance(result, ChunkResponse)
        assert "A critical error has occurred" in result.status_message

    def test_manage_401_error(self, mocker):
        class Custom401Error(Exception):
            status_code = 401

        mock_401_error = Custom401Error("401 Error")

        result = error_handle(mock_401_error, retry_count=3)
        assert isinstance(result, ChunkResponse)
        assert "A critical error has occurred" in result.status_message

    def test_quota_exceeded_error_message(self, mocker):
        class ExceededQuotaError(Exception):
            pass

        mock_quota_error = ExceededQuotaError(
            "You have exceeded your current quota"
        )
        result = error_handle(mock_quota_error)
        assert "A critical error has occurred" in result.status_message

    def test_error_image_for_unresolvable_errors(self, mocker, mock_error):
        result = error_handle(mock_error)
        assert (
            '<img src="/static/alert-light.png" alt="error icon" id="endError">'  # noqa E501
            in result.status_message
        )

    def test_replicate_increment_retry_count(self, mocker, mock_custom_error):
        with mocker.patch("api.api_management.time.sleep"):
            result = error_handle(mock_custom_error, retry_count=2)
        assert result == 3

    def test_exponential_backoff_in_retry(self, mocker, mock_custom_error):
        with mocker.patch("api.api_management.time.sleep") as mock_sleep:
            error_handle(mock_custom_error, retry_count=3)

        mock_sleep.assert_called_once_with(11)


class TestCallGptApi:
    def test_successful_api_call_with_valid_prompt(self, fake_client):
        prompt = "Test prompt"
        result = call_gpt_api(prompt, client=fake_client)
        assert result == "Generated content"

    def test_handles_no_message_error(
        self, mocker, fake_client_without_response
    ):
        prompt = "Test prompt"
        with mocker.patch(
            "error_handle", return_value="Second Attempt"
        ) as mock_error_handle:
            call_gpt_api(prompt, client=fake_client_without_response)
        mock_error_handle.assert_called_once()
        assert isinstance(mock_error_handle.call_args.args[0], NoMessageError)

    def test_default_retry_count(self, mocker, fake_client_without_response):
        prompt = "Test prompt"
        with mocker.patch(
            "error_handle", return_value="Second Attempt"
        ) as mock_error_handle:
            call_gpt_api(prompt, client=fake_client_without_response)
        assert mock_error_handle.call_args.args[1] == 1

    def test_call_gpt_api_retry_count_incremented(
        self, mocker, fake_client_without_response
    ):
        prompt = "Test prompt"
        with mocker.patch(
            "error_handle", return_value="Second Attempt"
        ) as mock_error_handle:
            call_gpt_api(
                prompt, client=fake_client_without_response, retry_count=1
            )
        assert mock_error_handle.call_args.args[1] == 2
