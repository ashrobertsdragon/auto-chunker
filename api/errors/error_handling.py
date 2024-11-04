import time
from typing import Any

from loguru import logger

from errors._exceptions import unresolvable_errors, APIError
from errors.email_admin import EmailAdmin


MAX_RETRY_COUNT = 2


def email_admin(e: Exception):
    EmailAdmin(e).send_email()


def check_json_response(response: Any) -> dict:
    """Attempt to parse JSON safely. Return None if parsing fails."""

    try:
        return response.json()
    except (AttributeError, ValueError):
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
        int: the number of attempts so far

    Raises:
        APIError: if the error is unresolvable or the retry count is exceeded.
    """

    error_code = getattr(e, "status_code", None)
    error_message = "Unknown error occurred"

    if hasattr(e, "response"):
        if json_data := check_json_response(e.response):
            error_message = json_data.get("error", {}).get(
                "message", "Unknown error"
            )
    else:
        error_message = str(e)

    logger.error(
        f"{e}. Error code: {error_code}. Error message: {error_message}"
    )

    if (
        isinstance(e, tuple(unresolvable_errors()))
        or error_code == 401
        or "exceeded your current quota" in error_message
    ):
        email_admin(e)
        raise APIError

    retry_count += 1
    if retry_count > MAX_RETRY_COUNT:
        logger.error("Retry count exceeded")
        email_admin(e)
        raise APIError
    else:
        sleep_time = (MAX_RETRY_COUNT - retry_count) + (retry_count**2)
        time.sleep(sleep_time)

    return retry_count
