from pydantic import BaseModel


class AutoChunkRequest(BaseModel):
    """
    Request body for chunking text.

    Attrs:
        book (str): The book text to chunk.
        chunk_type (int): The enum value of the type of chunking to use.
        role (str): The role of the user.
    """

    book: str
    chunk_type: int
    role: str

    class Config:
        json_schema_extra = {
            "example": {
                "book": "Once upon a time...",
                "chunk_type": 0,
                "role": "You are an author",
            }
        }
