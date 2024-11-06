import aiohttp
from decouple import config
from aiohttp.client_exceptions import ClientResponseError

from auto_chunker.errors._exceptions import APIError, BadRequestError
from auto_chunker.errors.error_handling import error_handle


def get_jsonl_api_config() -> tuple[str, str]:
    """
    Get the API key and URL from environment variables.

    Returns:
        tuple[str, str]: A tuple containing the API key and URL.

    Raises:
        APIError: If the API key or URL is not found in the environment
            variables.
    """
    api_key: str = config("JSONL_API_KEY", default="")
    api_url: str = config("JSONL_API_URL", default="")
    if not api_key:
        raise APIError("JSONL_API_KEY not found in environment variables")
    if not api_url:
        raise APIError("JSONL_API_URL not found in environment variables")
    if not api_url.startswith(("http://", "https://")):
        api_url = f"http://{api_url}"

    return api_key, api_url


async def get_jsonl(csv_str: str, retry_count: int = 0) -> str:
    """
    Get the JSONL content from the

    Args:
        csv_str (str): The CSV content to send to the
        retry_count (int, optional): The number of retries. Defaults to 0.

    Returns:
        str: The JSONL content from the
    """
    jsonl_api_key, jsonl_api_url = get_jsonl_api_config()
    headers = {"X-API-KEY": jsonl_api_key, "Content-Type": "text/csv"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=jsonl_api_url, data=csv_str, headers=headers
            ) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise BadRequestError(response.status)
    except (ClientResponseError, BadRequestError) as e:
        retry_count = error_handle(
            e, retry_count
        )  # raises APIError after 2 retries
        return get_jsonl(csv_str, retry_count + 1)
