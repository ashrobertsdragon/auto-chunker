from fastapi import Depends, FastAPI, HTTPException, status

from incoming.dependencies.authenticate import verify_api_key
from incoming.schema.auto_chunk_request_schema import AutoChunkRequest
from application.chunking import initiate_auto_chunker
from application.chunking_method import ChunkingMethod
from errors._exceptions import APIError
from outgoing.call_jsonl_converter import get_jsonl


app = FastAPI()


@app.post("/api/generate-auto-chunk-jsonl")
async def post_generate_auto_chunk_jsonl(
    request: AutoChunkRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    API endpoint for chunking text.

    Args:
        request (AutoChunkRequest): The request body containing book text, chunk type, and role.
        book (str): The book text to chunk.
        chunk_type (int): The enum value of the type of chunking to use.
        role (str): The role of the user.

    Returns:
        bytes: JSONL content of the chunked text.
    """
    try:
        csv_str = await initiate_auto_chunker(
            request.book, ChunkingMethod(request.chunk_type), request.role
        )
        return await get_jsonl(csv_str)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        ) from e
