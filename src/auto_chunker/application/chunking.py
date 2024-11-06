import re
from inspect import iscoroutinefunction

from decouple import config
from loguru import logger

from _types import ChunkingFunction
from errors._exceptions import APIError
from outgoing.openai_management import call_gpt_api
from application.chunking_method import ChunkingMethod
from application.data_preparation import (
    count_tokens,
    get_end_paragraph_tokens,
    separate_into_chapters,
    TOKENIZER,
)
from application.write_csv import create_csv_str

PUNCTUATION: list[str] = [".", "?", "!"]


def extract_dialogue(paragraph: str) -> tuple[str, str]:
    """
    Extract dialogue and prose from a paragraph.

    Args:
        paragraph (str): The paragraph text.

    Returns:
        tuple[str, str]: A tuple of dialogue and prose.
    """
    dialogue: list[str] = []
    prose: list[str] = []
    current_text: list[str] = []
    in_dialogue: bool = False

    for char in paragraph:
        if char == '"':
            if in_dialogue:
                dialogue.append("".join(current_text).strip())
            else:
                prose.append("".join(current_text).strip())
            in_dialogue = not in_dialogue
            current_text = []
        elif char == ",":
            current_text.append(", ")
        else:
            current_text.append(char)

    if in_dialogue:
        dialogue.append("".join(current_text).strip())
    else:
        prose.append("".join(current_text).strip())

    clean_prose: str = " ".join(prose).replace("  ", " ").strip().rstrip(",")
    clean_dialogue: str = (
        " ".join(dialogue).replace("  ", " ").strip().rstrip(",")
    )

    return clean_dialogue, clean_prose


def count_sentence_endings(text: str) -> int:
    """
    Count the number of sentence endings in the text.
    The regex

    Args:
        text (str): The text to search for sentences.

    Returns:
        int: The number of sentences in the text or 1 if there are none.
    """
    safe_punctuation = "".join(re.escape(mark) for mark in PUNCTUATION)
    sentence_endings = re.findall(rf"[{safe_punctuation}]+(?=\s|$)", text)
    return max(len(sentence_endings), 1)


def count_sentences(dialogue: str, prose: str) -> tuple[int, str, int, str]:
    """
    Count the number of sentences in the dialogue and prose.

    Args:
        dialogue (str): The dialogue text.
        prose (str): The prose text.

    Returns:
        tuple[int, str, int, str]: A tuple of dialogue and prose sentence
            counts and whether to use singular or plural version of the word
            'sentence'.
    """
    dialogue_sentences = count_sentence_endings(dialogue)
    prose_sentences = count_sentence_endings(prose)

    d_sentence: str = "sentence" if dialogue_sentences == 1 else "sentences"
    p_sentence: str = "sentence" if prose_sentences == 1 else "sentences"

    return dialogue_sentences, d_sentence, prose_sentences, p_sentence


def dialogue_prose(chapters: list[str]) -> tuple[list[str], list[str]]:
    """
    Split the book into prose and dialogue chunks.

    Args:
        chapters (list[str]): The list of chapter texts.

    Returns:
        tuple[list[str], list[str]]: A tuple of formatted chunks and user
            messages.
    """
    chunks = []
    user_messages = []

    for chapter in chapters:
        for paragraph in chapter.split("\n"):
            dialogue, prose = extract_dialogue(paragraph)
            dialogue_sentences, d_sentence, prose_sentences, p_sentence = (
                count_sentences(dialogue, prose)
            )
            if dialogue:
                chunks.append(dialogue)
                user_messages.append(
                    f"Write {dialogue_sentences} {d_sentence} of dialogue"
                )
            if prose:
                chunks.append(prose)
                user_messages.append(
                    f"Write {prose_sentences} {p_sentence} "
                    "of description and action"
                )
    return chunks, user_messages


async def generate_beats(
    chapters: list[str],
) -> tuple[list[str], list[str]]:
    """
    Generate beats for each chapter in the book using an LLM.

    Args:
        chapters (list[str]): The list of chapter texts.

    Returns:
        tuple[list[str], list[str]]: A tuple of formatted chunks and user
            messages.
    """
    user_message_list = []
    chunk_list = []

    for i, chapter in enumerate(chapters, start=1):
        words = len(chapter.split(" "))
        chapter_prompt = f"Chapter: {chapter}"
        logger.info(
            f"Sending {words} to GPT-4o Mini for chapter {i} of {len(chapters)}"
        )
        chapter_beats = await call_gpt_api(chapter_prompt)
        user_message_list.append(
            f"Write {words} words for a chapter with the following scene "
            f"beats:\n{chapter_beats}"
        )
        chunk_list.append(chapter)
    return chunk_list, user_message_list


def sliding_window(chapters: list[str]) -> tuple[list[str], list[str]]:
    """
    Split the book into chunks using a sliding window.

    Args:
        chapters (list[str]): The list of chapter texts.

    Returns:
        tuple[list[str], list[str]]: A tuple of formatted chunks and user
            messages.
    """
    all_chunks: list[str] = []
    max_chunk_size = config("MAX_TOKENS", cast=int, default=4096)
    end_paragraph_tokens = get_end_paragraph_tokens()
    for chapter in chapters:
        tokens, num_tokens = count_tokens(chapter)
        for start_index in range(0, num_tokens, max_chunk_size):
            end_index = min(start_index + max_chunk_size, num_tokens)

            if end_index < num_tokens:
                end_index = next(
                    (
                        i
                        for i in range(end_index, start_index, -1)
                        if tokens[i] in end_paragraph_tokens
                    ),
                    end_index,
                )

            chunk = TOKENIZER.decode(tokens[start_index:end_index])
            all_chunks.append(chunk)
    return all_chunks[1:], all_chunks[:-1]


async def chunk_text(
    book: str, chunk_type: ChunkingMethod
) -> tuple[list[str], list[str]]:
    """
    Split the book into chunks of the specified type.

    Args:
        chapters (list[str]): The list of chapter texts.
        chunk_type (ChunkingMethod): The type of chunking to use.

    Returns:
        tuple[list, list]: A tuple of formatted chunks and user messages.

    Raises:
        ValueError: If the book text does not contain at least one chapter.
        ValueError: If the chunking method is not supported.
    """
    chapters: list[str] = separate_into_chapters(book)

    if not chapters:
        raise ValueError("Book text must contain at least one chapter")
    chunk_map: dict[ChunkingMethod, ChunkingFunction] = {
        ChunkingMethod.DIALOGUE_PROSE: dialogue_prose,
        ChunkingMethod.GENERATE_BEATS: generate_beats,
        ChunkingMethod.SLIDING_WINDOW: sliding_window,
    }
    if chunk_type not in chunk_map:
        raise ValueError(f"Chunk method {chunk_type} not supported")
    chunking_func = chunk_map[chunk_type]
    if iscoroutinefunction(chunking_func):
        return await chunking_func(chapters)
    return chunking_func(chapters)


async def initiate_auto_chunker(
    text_content: str, chunking_method: ChunkingMethod, role: str
) -> str:
    """
    Split the book into prose and dialogue chunks.

    Args:
        text_content (str): The book text to chunk.
        chunking_method (ChunkingMethod): The type of chunking to use.
        role (str): The role of the user.

    Returns:
        str: CSV content of the chunked text.

    Raises:
        APIError: If there is an error during the chunking process.
    """
    try:
        chunks, user_messages = await chunk_text(text_content, chunking_method)
        return create_csv_str(chunks, user_messages, role)
    except ValueError as e:
        raise APIError from e
