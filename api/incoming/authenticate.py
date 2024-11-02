import hashlib

from decouple import config
from fastapi import Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from api.errors.error_handling import email_admin

api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Depends(api_key_header)):
    """
    Verify the API key provided in the request header.

    Args:
        api_key (str): The API key provided in the request header.

    Raises:
        HTTPException: If the API key is invalid.
    """
    HASHED_API_KEY = config("HASHED_API_KEY")
    hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
    try:
        if hashed_key != HASHED_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
            )
    except HTTPException as e:
        email_admin(e)
        raise e