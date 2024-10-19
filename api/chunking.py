from itertools import islice

from decouple import config

from api_management import call_gpt_api
from chunking_method import ChunkMethod
from data_preparation import (
    count_tokens,
    separate_into_chapters,
    TOKENIZER,
)


def generate_beats(chapters: list[str]) -> tuple[list[str], list[str]]:
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

    for chapter in chapters:
        words = len(chapter.split(" "))
        prompt = f"Chapter: {chapter}"
        chapter_beats = call_gpt_api(prompt)
        user_message_list.append(
            f"Write {words} words for a chapter with the following scene "
            f"beats:\n{chapter_beats}"
        )
        chunk_list.append(chapter)
    return chunk_list, user_message_list


def extract_dialogue(paragraph: str) -> tuple[str, str]:
    """
    Extract dialogue and prose from a paragraph.

    Args:
        paragraph (str): The paragraph text.

    Returns:
        tuple[str, str]: A tuple of dialogue and prose.
    """
    dialogue = str()
    prose = str()
    sentence = str()
    check_next_char = False
    end_sentence = False
    in_dialogue = False
    punctuation: list[str] = [".", "?", "!"]
    quote_count: int = 0

    for i, char in enumerate(paragraph):
        sentence += char
        if char in ['"', "'"]:
            quote_count += 1
            in_dialogue: bool = quote_count // 2 == 1
            end_sentence: bool = check_next_char is True
            check_next_char = False
        if char in punctuation:
            if i + 1 < len(paragraph):
                check_next_char = True
                continue
            end_sentence = True
        if end_sentence is True:
            if in_dialogue is False:
                prose += sentence
            elif in_dialogue is True:
                dialogue += sentence.strip("\"'")
            sentence = ""
            end_sentence = False
    return prose.strip(), dialogue.strip()


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
    punctuation = [".", "?", "!"]

    for chapter in chapters:
        paragraphs = chapter.split("\n")
        for paragraph in paragraphs:
            prose_sentences = 0
            dialogue_sentences = 0
            prose, dialogue = extract_dialogue(paragraph)
            for mark in punctuation:
                prose_sentences += prose.count(mark)
                p_sentence = (
                    "sentence" if prose_sentences == 1 else "sentences"
                )
                dialogue_sentences += dialogue.count(mark)
                d_sentence = (
                    "sentence" if dialogue_sentences == 1 else "sentences"
                )
            if prose:
                chunks.append(prose)
                user_messages.append(
                    f"Write {prose_sentences} {p_sentence} "
                    "of description and action"
                )
            if dialogue:
                chunks.append(dialogue)
                user_messages.append(
                    f"Write {dialogue_sentences} {d_sentence} of dialogue"
                )
    return chunks, user_messages


def sliding_window(chapters: list[str]) -> tuple[list[str], list[str]]:
    """
    Split the book into chunks using a sliding window.

    Args:
        chapters (list[str]): The list of chapter texts.

    Returns:
        tuple[list[str], list[str]]: A tuple of formatted chunks and user
            messages.
    """
    user_messages: list[str] = []
    chunks: list[str] = []
    max_chunk_size = config("MAX_CHUNK_SIZE", cast=int, default=4096)
    for chapter in chapters:
        tokens, num_tokens = count_tokens(chapter)
        start_index = 0

        while start_index < num_tokens:
            end_index = min(start_index + max_chunk_size, num_tokens)

            if end_index < num_tokens:
                end_index = next(
                    (
                        i
                        for i in range(end_index, start_index, -1)
                        if tokens[i] == 10
                    ),
                    end_index,
                )

            chunk = TOKENIZER.decode(
                list(islice(tokens, start_index, end_index))
            )
            chunks.append(chunk)
            user_messages.append(chunk)

            start_index = end_index

    return chunks, user_messages


def chunk_text(
    book: str, chunk_type: ChunkMethod
) -> tuple[list[str], list[str]]:
    """
    Split the book into chunks of the specified type.

    Args:
        chapters (list[str]): The list of chapter texts.
        chunk_type (ChunkMethod): The type of chunking to use.

    Returns:
        tuple[list, list]: A tuple of formatted chunks and user messages.
    """
    chunks = []
    user_messages = []
    chapters = separate_into_chapters(book)
    if chunk_type == ChunkMethod.SLIDING_WINDOW:
        chunks, user_messages = sliding_window(chapters)
    if chunk_type == ChunkMethod.DIALOGUE_PROSE:
        chunks, user_messages = dialogue_prose(chapters)
    if chunk_type == ChunkMethod.GENERATE_BEATS:
        chunks, user_messages = generate_beats(chapters)
    return chunks, user_messages
