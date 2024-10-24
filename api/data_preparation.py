import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> tuple[list[int], int]:
    """
    Uses Tiktoken tokenizer to tokenize text and count the number of tokens
    """
    tokens = TOKENIZER.encode(text)
    num_tokens = len(tokens)
    return tokens, num_tokens


def get_end_paragraph_tokens() -> list[int]:
    """
    Returns the list of token ids indicating the end of a paragraph
    """
    return [
        198,
        627,
        4999,
        5380,
        702,
        10246,
        25765,
        48469,
        34184,
        1270,
        7058,
        7233,
        11192,
    ]


def adjust_to_newline(tokens: list[int], end_index: int) -> int:
    """
    Adjusts the end index to the end of the last paragraph, based on token ids
    indicating the end of a paragraph
    """
    end_paragraph_tokens = get_end_paragraph_tokens()
    while end_index > 0 and tokens[end_index - 1] not in end_paragraph_tokens:
        end_index -= 1
    return end_index


def separate_into_chapters(text: str) -> list:
    """
    Separates the text into chapters
    """
    return text.split("***")
