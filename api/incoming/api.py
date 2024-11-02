from fastapi import Depends, FastAPI, HTTPException, status

from api.incoming.authenticate import verify_api_key
from api.application.chunking import chunk
from api.application.chunking_method import ChunkingMethod
from api.errors._exceptions import APIError
from api.outgoing.call_jsonl_converter import get_jsonl


app = FastAPI()


@app.get("/api/chunk_text")
async def chunk_text_api(
    book: str,
    chunk_type: ChunkingMethod,
    role: str,
    api_key: str = Depends(verify_api_key),
):
    """
    API endpoint for chunking text.

    Args:
        book (str): The book text to chunk.
        chunk_type (ChunkingMethod): The type of chunking to use.
        role (str): The role of the user.

    Returns:
        bytes: JSONL content of the chunked text.
    """
    try:
        csv_str = chunk(book, chunk_type, role)
        return get_jsonl(csv_str)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        ) from e
