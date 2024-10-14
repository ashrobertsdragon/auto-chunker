from api_management import call_gpt_api
from chunking_method import ChunkMethod
from data_preparation import (
    adjust_to_newline,
    count_tokens,
    format_for_finetuning,
    separate_into_chapters,
    sliding_window_format,
    TOKENIZER,
)


def generate_beats(book: str) -> list:
    """
    Generate beats for each chapter in the book using GPT-3.5.
    """
    user_message_list = []
    chunk_list = []
    chapters = separate_into_chapters(book)

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
    """
    dialogue = ""
    prose = ""
    sentence = ""
    check_next_char = False
    end_sentence = False
    in_dialogue = False
    punctuation = [".", "?", "!"]
    quote_count = 0

    for i, char in enumerate(paragraph):
        sentence += char
        if char == '"':
            quote_count += 1
            in_dialogue = True if (quote_count // 2 == 1) else False
            end_sentence = True if check_next_char is True else False
            check_next_char = False
        if char in punctuation:
            if i + 1 < len(paragraph):
                check_next_char = True
                continue
            end_sentence = True
        if end_sentence is True:
            if in_dialogue is False:
                prose += sentence.strip()
            elif in_dialogue is True:
                dialogue += sentence.strip()
            sentence = ""
            end_sentence = False
    return prose, dialogue


def dialogue_prose(book: str) -> list:
    """
    Split the book into prose and dialogue chunks.
    """
    chunk_list = []
    user_message_list = []
    punctuation = [".", "?", "!"]
    chapters = separate_into_chapters(book)

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
                chunk_list.append(prose)
                user_message_list.append(
                    f"Write {prose_sentences} {p_sentence} "
                    "of description and action"
                )
            if dialogue:
                chunk_list.append(dialogue)
                user_message_list.append(
                    f"Write {dialogue_sentences} {d_sentence} of dialogue"
                )
    return chunk_list, user_message_list


def sliding_window(book: str) -> list:
    """
    Split the book into chunks of 4096 tokens.
    """
    chunk_list = []
    chunk_size = 4096
    start_index = 0
    tokens, num_tokens = count_tokens(book)

    while start_index < num_tokens:
        end_index = min(start_index + chunk_size, num_tokens)
        # Adjust end_index to the last newline token in the chunk
        if end_index < num_tokens:
            end_index = adjust_to_newline(tokens, end_index)
        chunk_tokens = tokens[start_index:end_index]
        chunk_list.append(TOKENIZER.decode(chunk_tokens))
        start_index = end_index
    return chunk_list


def format_messages(
    chunks: list, user_messages: list, role: str, chunk_type: ChunkMethod
) -> list:
    """
    Formats chunked data for finetuning.
    """
    return (
        sliding_window_format(chunks, role)
        if chunk_type == ChunkMethod.sliding_window
        else format_for_finetuning(chunks, user_messages, role)
    )


def chunk_text(book: str, chunk_type: ChunkMethod) -> tuple[list, list]:
    """
    Split the book into chunks of the specified type.
    """
    chunks = []
    user_messages = []
    if chunk_type == ChunkMethod.SLIDING_WINDOW:
        chunks = sliding_window(book)
    if chunk_type == ChunkMethod.DIALOGUE_PROSE:
        chunks, user_messages = dialogue_prose(book)
    if chunk_type == ChunkMethod.GENERATE_BEATS:
        chunks, user_messages = generate_beats(book)
    return chunks, user_messages


def chunk_book(book: str, role: str, chunk_type: str) -> list:
    """
    Split the book into chunks of the specified type.
    """
    chunks, user_messages = chunk_text(book, chunk_type)

    return format_messages(chunks, user_messages, role, chunk_type)
