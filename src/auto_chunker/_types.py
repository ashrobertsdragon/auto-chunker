from collections.abc import Callable, Awaitable
from typing import TypeAlias

ChunkResult: TypeAlias = tuple[list[str], list[str]]
ChunkingFunction: TypeAlias = (
    Callable[[list[str]], ChunkResult]
    | Callable[[list[str]], Awaitable[ChunkResult]]
)
