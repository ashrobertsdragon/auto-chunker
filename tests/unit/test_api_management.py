import httpx
import pytest
from openai import BadRequestError
from openai.types.chat.chat_completion import ChatCompletion, Choice

from auto_chunker.outgoing.openai_management import call_gpt_api
from auto_chunker.errors.error_handling import error_handle
from auto_chunker.errors._exceptions import NoMessageError


@pytest.fixture
def mock_error():
    return BadRequestError(
        "Bad request",
        response=httpx.Response(
            400,
            request=httpx.Request(
                "GET", "https://openai.com/v1/chat/completions"
            ),
        ),
        body={"error": {"message": "A critical error has occurred"}},
    )


@pytest.fixture
def mock_custom_error():
    class CustomError(Exception):
        pass

    return CustomError("Custom error")


@pytest.fixture
def fake_client():
    class FakeClient:
        def call_api(self, messages):
            return ChatCompletion(
                id="chatcmpl-123456",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message={
                            "role": "assistant",
                            "content": "Generated content",
                        },
                    )
                ],
                created=1728933352,
                model="gpt-4o-mini",
                object="chat.completion",
                _request_id="123456",
            )

    return FakeClient()


@pytest.fixture
def fake_client_without_response():
    class FakeClient:
        def call_api(self, messages):
            return ChatCompletion(
                id="chatcmpl-123456",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message={
                            "role": "assistant",
                        },
                    )
                ],
                created=1728933352,
                model="gpt-4o-mini",
                object="chat.completion",
                _request_id="123456",
            )

    return FakeClient()


class TestErrorHandle:
    def test_unresolvable_error_handling(self, mocker, mock_error):
        # sourcery skip: class-extract-method
        result = error_handle(mock_error)
        assert result == 500

    def test_resolvable_error_handling(self, mocker, mock_error):
        class CustomError(Exception):
            pass

        mock_error = CustomError("Custom error without response")
        mocker.patch("auto_chunker.errors.error_handling.asyncio..sleep")
        result = error_handle(mock_error)
        assert isinstance(result, int)
        assert result == 1

    def test_logs_error_details(self, mocker, mock_error):
        result = error_handle(mock_error)
        assert result == 500

    def test_error_handle_unknown_status(self, mocker, mock_error):
        result = error_handle(mock_error)
        assert result == 500

    def test_manage_401_error(self, mocker):
        class Custom401Error(Exception):
            status_code = 401

        mock_401_error = Custom401Error("401 Error")

        result = error_handle(mock_401_error, retry_count=3)
        assert result == 500

    def test_quota_exceeded_error_message(self, mocker):
        class ExceededQuotaError(Exception):
            pass

        mock_quota_error = ExceededQuotaError(
            "You have exceeded your current quota"
        )
        result = error_handle(mock_quota_error)
        assert result == 500

    def test_replicate_increment_retry_count(self, mocker, mock_custom_error):
        mocker.patch("auto_chunker.errors.error_handling.asyncio.sleep")
        result = error_handle(mock_custom_error, retry_count=2)
        assert result == 3

    def test_exponential_backoff_in_retry(self, mocker, mock_custom_error):
        mock_sleep = mocker.patch(
            "auto_chunker.errors.error_handling.asyncio.sleep"
        )

        error_handle(mock_custom_error, retry_count=3)

        mock_sleep.assert_called_once_with(17)

    def test_max_retry_count_exceeded(self, mocker, mock_custom_error):
        mocker.patch("auto_chunker.errors.error_handling.asyncio.sleep")
        result = error_handle(mock_custom_error, retry_count=5)
        assert result == 500


class TestCallGptApi:
    def test_successful_api_call_with_valid_prompt(self, fake_client):
        prompt = "Test prompt"
        result = call_gpt_api(prompt, client=fake_client)
        assert result == "Generated content"

    def test_handles_no_message_error(
        self, mocker, fake_client_without_response
    ):
        prompt = "Test prompt"
        mock_error_handle = mocker.patch(
            "auto_chunker.errors.error_handling.error_handle", return_value=500
        )

        call_gpt_api(prompt, client=fake_client_without_response)

        assert isinstance(
            mock_error_handle.call_args.kwargs["e"], NoMessageError
        )

    def test_default_retry_count(self, mocker, fake_client_without_response):
        prompt = "Test prompt"
        mocker.patch("auto_chunker.errors.error_handling.asyncio.sleep")
        mock_error_handle = mocker.patch(
            "auto_chunker.errors.error_handling.error_handle", return_value=500
        )

        call_gpt_api(prompt, client=fake_client_without_response)

        assert mock_error_handle.call_args.kwargs["retry_count"] == 0

    def test_call_gpt_api_returns_chunk_response_after_5_retries(
        self, mocker, fake_client_without_response
    ):
        prompt = "Test prompt"
        mocker.patch(
            "auto_chunker.errors.error_handling.error_handle", return_value=500
        )

        result = call_gpt_api(
            prompt, client=fake_client_without_response, retry_count=5
        )

        assert result == 500
