import requests
from decouple import config
from requests.exceptions import RequestException

from api.errors._exceptions import BadRequestError
from api.errors.error_handling import error_handle


def get_jsonl(csv_str: str, retry_count: int = 0) -> str:
    """
    Get the JSONL content from the API.

    Args:
        csv_str (str): The CSV content to send to the API.
        retry_count (int, optional): The number of retries. Defaults to 0.

    Returns:
        str: The JSONL content from the API.
        str: The word 'Error' if there is an error.
    """
    try:
        response = requests.get(
            f"{config('API_URL')}/api/create_jsonl",
            params={"csv_str": csv_str},
        )
        if response.status_code == 200:
            return response.content
        else:
            raise BadRequestError(response.status_code)
    except (RequestException, BadRequestError) as e:
        retry_count = error_handle(e, retry_count)
        return get_jsonl(csv_str, retry_count + 1)
