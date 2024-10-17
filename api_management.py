import time
from typing import Any

import openai
from loguru import logger

from _proto.auto_chunker_pb2 import ChunkResponse
from openai_client import OpenAIAPI

CLIENT = OpenAIAPI()


class NoMessageError(Exception):
    pass


def unresolvable_errors() -> list:
    return [
        openai.BadRequestError,
        openai.AuthenticationError,
        openai.NotFoundError,
        openai.PermissionDeniedError,
        openai.UnprocessableEntityError,
    ]


def check_json_response(response: Any) -> dict:
    """Attempt to parse JSON safely. Return None if parsing fails."""

    try:
        return response.json()
    except ValueError:
        return {}


def error_handle(e: Any, retry_count: int = 0) -> int:
    """
    Determines whether error is unresolvable or should be retried. If
    unresolvable, error is logged and administrator is emailed before exit.
    Otherwise, exponential backoff is used for up to 5 retries.

    Args:
        e: an Exception body
        retry_count: the number of attempts so far

    Returns:
        retry_count: the number of attempts so far
    """

    error_image = (
        '<img src="/static/alert-light.png" alt="error icon" id="endError">'
    )

    error_code = getattr(e, "status_code", None)
    error_message = "Unknown error occurred"

    if hasattr(e, "response"):
        json_data = check_json_response(e.response)
        if json_data:
            error_message = json_data.get("error", {}).get(
                "message", "Unknown error"
            )

    logger.error(
        f"{e}. Error code: {error_code}. Error message: {error_message}"
    )

    if (
        isinstance(e, tuple(unresolvable_errors()))
        or error_code == 401
        or "exceeded your current quota" in error_message
    ):
        error_message = (
            f"{error_image} A critical error has occurred. "
            "The administrator has been contacted. "
            "Sorry for the inconvenience"
        )
        return ChunkResponse(jsonl_content="", status_message=error_message)

    retry_count += 1
    if retry_count > 5:
        logger.error("Retry count exceeded")
        error_message: str = (
            "A critical error has occurred. Administrator has been contacted."
        )
        return ChunkResponse(
            jsonl_content="",
            status_message=error_message,
        )
    else:
        sleep_time = (5 - retry_count) + (retry_count**2)
        time.sleep(sleep_time)
    return retry_count


def call_gpt_api(prompt: str, retry_count: int = 0, client=CLIENT) -> str:
    """
    Makes API calls to the OpenAI ChatCompletions engine.

    Args:.
        prompt (str): The user's prompt.
        retry_count (int): The number of retry attempts. Defaults to 0.

    Returns:
        str: The generated content from the OpenAI GPT-3 model.
    """

    role_script = "You are an expert developmental editor who specializes in "
    "writing scene beats that are clear and concise. For the following "
    "chapter, please reverse engineer the scene beats for the author. Provide "
    "only the beats and not any commentary the beginning or end."
    messages = [
        {"role": "system", "content": role_script},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.call_api(messages)
        if response.choices and response.choices[0].message.content:
            answer = response.choices[0].message.content.strip()
        else:
            raise NoMessageError("No message content found")

    except tuple([NoMessageError] + unresolvable_errors()) as e:
        retry_count = error_handle(e, retry_count)
        answer = call_gpt_api(prompt, retry_count)
    return answer
