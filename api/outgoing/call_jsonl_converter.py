import requests
from decouple import config
from requests.exceptions import RequestException

from api.errors._exceptions import APIError, BadRequestError
from api.errors.error_handling import error_handle


def get_jsonl_api_config() -> tuple[str, str]:
    """
    Get the API key and URL from environment variables.

    Returns:
        tuple[str, str]: A tuple containing the API key and URL.

    Raises:
        APIError: If the API key or URL is not found in the environment
            variables.
    """
    api_key: str = config("JSONL_API_KEY", default="", cast=str)
    api_url: str = config("JSONL_API_URL", default="", cast=str)
    if not api_key:
        raise APIError("JSONL_API_KEY not found in environment variables")
    if not api_url:
        raise APIError("JSONL_API_URL not found in environment variables")
    return api_key, api_url


def get_jsonl(csv_str: str, retry_count: int = 0) -> str:
    """
    Get the JSONL content from the API.

    Args:
        csv_str (str): The CSV content to send to the API.
        retry_count (int, optional): The number of retries. Defaults to 0.

    Returns:
        str: The JSONL content from the API.
    """
    jsonl_api_key, jsonl_api_url = get_jsonl_api_config()
    try:
        response = requests.get(
            url=jsonl_api_url,
            params={"csv_str": csv_str},
            headers={"X-API-KEY": jsonl_api_key},
        )
        if response.status_code == 200:
            return response.content
        else:
            raise BadRequestError(response.status_code)
    except (RequestException, BadRequestError) as e:
        retry_count = error_handle(
            e, retry_count
        )  # raises APIError after 5 retries
        return get_jsonl(csv_str, retry_count + 1)
