from enum import Enum


class ChunkingMethod(Enum):
    SLIDING_WINDOW = 0
    DIALOGUE_PROSE = 1
    GENERATE_BEATS = 2
