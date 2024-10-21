from decouple import config

from api_management import call_gpt_api
from chunking_method import ChunkingMethod
from data_preparation import (
    count_tokens,
    get_end_paragraph_tokens,
    separate_into_chapters,
    TOKENIZER,
)


PUNCTUATION: list[str] = [".", "?", "!"]


def extract_dialogue(paragraph: str) -> tuple[str, str]:
    """
    Extract dialogue and prose from a paragraph.

    Args:
        paragraph (str): The paragraph text.

    Returns:
        tuple[str, str]: A tuple of dialogue and prose.
    """
    dialogue, prose = [], []
    current_text = []
    in_dialogue = False

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

    clean_prose = " ".join(prose).replace("  ", " ").strip()
    clean_dialogue = " ".join(dialogue).replace("  ", " ").strip()

    return clean_dialogue, clean_prose


def count_punctuation(dialogue: str, prose: str) -> tuple[int, str, int, str]:
    """
    Count the number of punctuation marks in the dialogue and prose.

    Args:
        dialogue (str): The dialogue text.
        prose (str): The prose text.

    Returns:
        tuple[int, str, int, str]: A tuple of dialogue and prose sentence
            counts and whether to use singular or plural version of the word
            'sentence'.
    """
    dialogue_sentences: int = 0
    prose_sentences: int = 0
    for mark in PUNCTUATION:
        dialogue_sentences += dialogue.count(mark)
        prose_sentences += prose.count(mark)

    dialogue_sentences = 1 if dialogue_sentences == 0 else dialogue_sentences

    prose_sentences = 1 if prose_sentences == 0 else prose_sentences
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
                count_punctuation(dialogue, prose)
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


def chunk_text(
    book: str, chunk_type: ChunkingMethod
) -> tuple[list[str], list[str]]:
    """
    Split the book into chunks of the specified type.

    Args:
        chapters (list[str]): The list of chapter texts.
        chunk_type (ChunkingMethod): The type of chunking to use.

    Returns:
        tuple[list, list]: A tuple of formatted chunks and user messages.
    """
    chapters: list[str] = separate_into_chapters(book)
    chunk_map = {
        ChunkingMethod.DIALOGUE_PROSE: dialogue_prose,
        ChunkingMethod.GENERATE_BEATS: generate_beats,
        ChunkingMethod.SLIDING_WINDOW: sliding_window,
    }
    if chunk_type not in chunk_map:
        raise ValueError(f"Chunk method {chunk_type} not supported")
    chunking_func = chunk_map[chunk_type]
    return chunking_func(chapters)
