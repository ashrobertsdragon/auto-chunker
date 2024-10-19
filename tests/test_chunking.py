import pytest

from api.chunking import (
    chunk_text,
    generate_beats,
    extract_dialogue,
    dialogue_prose,
    sliding_window,
)
from chunking_method import ChunkMethod
from data_preparation import separate_into_chapters, TOKENIZER


class TestGenerateBeats:
    def test_generates_beats_for_multiple_chapters(self, mocker):
        mocker.patch(
            "api.chunking.call_gpt_api", return_value="Generated beat"
        )
        chapters = ["Chapter 1 text", "Chapter 2 text"]

        chunk_list, user_message_list = generate_beats(chapters)

        assert len(chunk_list) == 2
        assert len(user_message_list) == 2
        assert chunk_list == chapters
        assert "Generated beat" in user_message_list[0]
        assert "Write" in user_message_list[0]
        assert "words for a chapter" in user_message_list[0]
        assert "Generated beat" in user_message_list[1]
        assert "Write" in user_message_list[1]
        assert "words for a chapter" in user_message_list[1]

    def test_handles_empty_chapter_list(self, mocker):
        mocker.patch(
            "api.chunking.call_gpt_api", return_value="Generated beat"
        )
        chapters = []

        chunk_list, user_message_list = generate_beats(chapters)

        assert chunk_list == []
        assert user_message_list == []

    def test_constructs_correct_user_message_format(self, mocker):
        mocker.patch(
            "api.chunking.call_gpt_api", return_value="Generated beat"
        )
        chapter = "This is a test chapter with five words"

        _, user_message_list = generate_beats([chapter])

        expected_word_count = len(chapter.split())
        message = user_message_list[0]
        assert f"Write {expected_word_count} words" in message
        assert "scene beats" in message
        assert "Generated beat" in message

    def test_preserves_original_chapters_in_chunk_list(self, mocker):
        mocker.patch(
            "api.chunking.call_gpt_api", return_value="Generated beat"
        )
        chapters = ["Chapter 1: A beginning", "Chapter 2: The middle part"]

        chunk_list, _ = generate_beats(chapters)

        assert chunk_list == chapters
        assert isinstance(chunk_list[0], str)
        assert isinstance(chunk_list[1], str)

    def test_handles_single_word_chapter(self, mocker):
        mocker.patch(
            "api.chunking.call_gpt_api", return_value="Generated beat"
        )
        chapter = "OneWordChapter"

        chunk_list, user_message_list = generate_beats([chapter])

        assert len(chunk_list) == 1
        assert len(user_message_list) == 1
        assert "Write 1 words" in user_message_list[0]

    def test_returns_correct_tuple_structure(self, mocker):
        mocker.patch(
            "api.chunking.call_gpt_api", return_value="Generated beat"
        )
        chapters = ["Test chapter"]

        result = generate_beats(chapters)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)


class TestExtractDialogue:
    def test_basic_double_quotes(self):  # sourcery skip: class-extract-method
        paragraph = 'He said, "Hello there!" and waved.'
        prose, dialogue = extract_dialogue(paragraph)
        assert prose == "He said, and waved."
        assert dialogue == "Hello there!"

    def test_dialogue_with_interruption(self):
        paragraph = '"Stop," he said, "you\'re going too fast!"'
        prose, dialogue = extract_dialogue(paragraph)
        assert prose == "he said,"
        assert dialogue == "Stop, you're going too fast!"

    def test_unbalanced_quotes(self):
        paragraph = 'He said, "This is a test.'
        prose, dialogue = extract_dialogue(paragraph)
        assert prose == "He said,"
        assert dialogue == "This is a test."

    def test_empty_string(self):
        prose, dialogue = extract_dialogue("")
        assert prose == ""
        assert dialogue == ""

    def test_single_quotes(self):
        paragraph = "She whispered, 'Don't be afraid,' and smiled."
        prose, dialogue = extract_dialogue(paragraph)
        assert prose == "She whispered, and smiled."
        assert dialogue == "Don't be afraid,"

    def test_nested_quotes(self):
        paragraph = """She said, "He told me 'Hello there!'" and smiled."""
        prose, dialogue = extract_dialogue(paragraph)
        assert prose == "She said, and smiled."
        assert dialogue == "He told me 'Hello there!'"

    def test_consecutive_punctuation(self):
        paragraph = 'She exclaimed, "Oh my...!" and ran.'
        prose, dialogue = extract_dialogue(paragraph)
        assert prose == "She exclaimed, and ran."
        assert dialogue == "Oh my...!"

    def test_only_prose(self):
        paragraph = "The sun was setting beautifully."
        prose, dialogue = extract_dialogue(paragraph)
        assert prose == "The sun was setting beautifully."
        assert dialogue == ""

    def test_only_dialogue(self):
        paragraph = '"This is all dialogue!"'
        prose, dialogue = extract_dialogue(paragraph)
        assert prose == ""
        assert dialogue == "This is all dialogue!"

    def test_mixed_quote_types(self):
        paragraph = (
            """'Single quotes first.' He paused. "Then double quotes." """
        )
        prose, dialogue = extract_dialogue(paragraph)
        assert prose == "He paused."
        assert dialogue == "Single quotes first. Then double quotes."


class TestDialogueProse:
    def test_correct_split_prose_and_dialogue(self):
        chapters = [
            'He said, "Hello!"\nShe replied, "Hi!"',
            'The sun was setting.\n"Beautiful evening," he remarked.',
        ]
        expected_chunks = [
            "He said, ",
            '"Hello!"',
            "She replied, ",
            '"Hi!"',
            "The sun was setting.",
            '"Beautiful evening,"',
        ]
        expected_messages = [
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_accurate_sentence_counts(self):
        chapters = [
            '"Hello!" she said. "How are you?"\nHe nodded. "I am fine."'
        ]
        expected_chunks = [
            '"Hello!"',
            '"How are you?"',
            "He nodded.",
            '"I am fine."',
        ]
        expected_messages = [
            "Write 2 sentences of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_empty_chapters_handling(self):
        chapters = [""]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == []
        assert user_messages == []

    def test_no_dialogue_or_prose_paragraphs(self):
        chapters = ["\n\n", "Just some text without dialogue.\n\n"]
        expected_chunks = ["Just some text without dialogue."]
        expected_messages = ["Write 1 sentence of description and action"]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_handles_mixed_prose_and_dialogue_effectively(self):
        chapters = [
            'The storm raged. "Take cover!" he shouted.\nShe ran inside.'
        ]
        expected_chunks = [
            "The storm raged.",
            '"Take cover!"',
            "She ran inside.",
        ]
        expected_messages = [
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_generates_user_messages(self):
        chapters = ['She whispered, "Careful."\nThe door creaked open.']
        expected_chunks = [
            "She whispered, ",
            '"Careful."',
            "The door creaked open.",
        ]
        expected_messages = [
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_processes_multiple_chapters(self):
        chapters = [
            'Chapter 1\nShe smiled.\n"Nice weather," he said.',
            'Chapter 2\nThe rain fell.\n"Indeed," she replied.',
        ]
        expected_chunks = [
            "She smiled.",
            '"Nice weather,"',
            "The rain fell.",
            '"Indeed,"',
        ]
        expected_messages = [
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_handles_chapters_no_punctuation(self):
        chapters = ["He spoke softly", '"Hello" she said', "The wind blew"]
        expected_chunks = ["He spoke softly", '"Hello"', "The wind blew"]
        expected_messages = [
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_supports_varying_paragraph_lengths(self):
        chapters = [
            'Short line.\n"Brief."\nLong paragraph with multiple sentences. More text.',  # noqa E501
            "Another chapter.",
        ]
        expected_chunks = [
            "Short line.",
            '"Brief."',
            "Long paragraph with multiple sentences. More text.",
            "Another chapter.",
        ]
        expected_messages = [
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 2 sentences of description and action",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_maintains_order_of_chunks_and_messages(self):
        chapters = [
            'First action.\n"First dialogue."\nSecond action.',
            '"Second dialogue."\nThird action.',
        ]
        expected_chunks = [
            "First action.",
            '"First dialogue."',
            "Second action.",
            '"Second dialogue."',
            "Third action.",
        ]
        expected_messages = [
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_handles_non_standard_punctuation(self):
        chapters = ['He said...! "Well...?"\nShe replied?! "Hmm..."']
        expected_chunks = [
            "He said...!",
            '"Well...?"',
            "She replied?!",
            '"Hmm..."',
        ]
        expected_messages = [
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_manage_chapters_with_special_characters(self):
        chapters = ['He yelled, "Stop!!"\nShe screamed, "No!!!"']
        expected_chunks = [
            "He yelled, ",
            '"Stop!!"',
            "She screamed, ",
            '"No!!!"',
        ]
        expected_messages = [
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_sentence_count_grammar(self):
        chapters = [
            '"First!" he said. "Second!"\nHe waved.\nShe smiled. She laughed.',
            "The end.",
        ]
        expected_chunks = [
            '"First!"',
            '"Second!"',
            "He waved.",
            "She smiled. She laughed.",
            "The end.",
        ]
        expected_messages = [
            "Write 2 sentences of dialogue",
            "Write 1 sentence of description and action",
            "Write 2 sentences of description and action",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages


@pytest.fixture
def mock_config(mocker):
    return mocker.patch("decouple.config", return_value=10)


class TestSlidingWindow:
    def test_processes_chapters_into_chunks_and_user_messages(
        self, mock_config
    ):
        """Split text into chunks while respecting maximum token size."""
        chapter = """In the bustling city of New York, where dreams take flight and ambitions soar high above the towering skyscrapers, Sarah found herself at a crossroads. The decision she faced would not only shape her career but potentially alter the course of her life. As she sat in her favorite coffee shop on 5th Avenue, watching the endless stream of pedestrians hurrying past the window, she contemplated her choices. The startup she had poured her heart into for the past three years had finally received an acquisition offer from a major tech company. The numbers were impressive - life-changing, even. But accepting would mean giving up control of her vision, the very thing that had driven her to create this company in the first place."""  # noqa: E501

        chunks, user_messages = sliding_window([chapter])

        tokens = TOKENIZER.encode(chapter)
        expected_num_chunks = len(tokens) // 10 + (
            1 if len(tokens) % 10 else 0
        )
        assert len(chunks) == expected_num_chunks
        for chunk in chunks:
            assert len(TOKENIZER.encode(chunk)) <= 10
        reassembled = "".join(chunks)
        assert reassembled == chapter
        assert user_messages == chunks

    def test_handles_chapters_with_varying_lengths(self, mock_config):
        """Handle multiple chapters of different lengths correctly."""
        chapters = ["Short", "Much longer chapter text"]
        chunks, _ = sliding_window(chapters)
        assert "Short" in chunks
        assert len(chunks) > 1

    def test_handles_no_line_breaks(self, mock_config):
        """Process text without natural break points."""
        chapters = ["ThisIsAVeryLongWordWithoutBreaks"]
        chunks, _ = sliding_window(chapters)
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_chapters_shorter_than_chunk_size(self):
        """Keep short chapters intact without splitting."""
        chapters = ["Short chapter."]
        chunks, user_messages = sliding_window(chapters)
        assert chunks == ["Short chapter."]
        assert user_messages == ["Short chapter."]

    def test_handles_non_ascii_characters(self, mock_config):
        """Process text containing non-ASCII characters correctly."""
        chapters = ["Hello 你好 नमस्ते"]
        chunks, _ = sliding_window(chapters)
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_performance_with_large_input(self, mock_config):
        """Handle large number of chapters efficiently."""
        chapters = ["Chapter text"] * 1000
        chunks, user_messages = sliding_window(chapters)
        assert len(chunks) == len(user_messages)
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_special_characters(self, mock_config):
        """Process text with special characters properly."""
        chapters = ["Text with !@#$%^&*()"]
        chunks, _ = sliding_window(chapters)
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_exact_chunk_size(self, mock_config):
        """Handle text exactly matching chunk size."""
        chapters = ["1234567890"]
        chunks, _ = sliding_window(chapters)
        assert len(chunks[0]) == 10
        assert len(chunks) == 1


class TestChunkText:
    def test_correctly_splits_book_into_chapters(self, mocker):
        book = "Chapter 1 text *** Chapter 2 text"
        expected_chapters = ["Chapter 1 text", "Chapter 2 text"]

        mocker.patch(
            "api.data_preparation.separate_into_chapters",
            return_value=expected_chapters,
        )

        chapters, _ = chunk_text(book, ChunkMethod.DIALOGUE_PROSE)

        separate_into_chapters.assert_called_once_with(book)
        assert chapters == expected_chapters

    def test_maps_chunk_type_to_correct_function(self, mocker):
        book = "Chapter 1 text *** Chapter 2 text"
        mock_func = mocker.patch(
            "api.chunking.dialogue_prose", return_value=([], [])
        )

        chunk_text(book, ChunkMethod.DIALOGUE_PROSE)

        mock_func.assert_called_once()

    def test_raises_value_error_for_unsupported_chunk_type(self):
        book = "Chapter 1 text *** Chapter 2 text"

        with pytest.raises(ValueError, match="Chunk method .* not supported"):
            chunk_text(book, "UNSUPPORTED_METHOD")

    def test_handles_empty_book_input_gracefully(self, mocker):
        book = ""
        mock_func = mocker.patch(
            "api.chunking.dialogue_prose", return_value=([], [])
        )

        chapters, user_messages = chunk_text(book, ChunkMethod.DIALOGUE_PROSE)

        assert chapters == []
        assert user_messages == []
        mock_func.assert_called_once_with([])

    def test_correct_function_called_for_chunk_type(self, mocker):
        book = "Chapter 1 text *** Chapter 2 text"
        expected_chapters = ["Chapter 1 text", "Chapter 2 text"]

        mocker.patch(
            "api.data_preparation.separate_into_chapters",
            return_value=expected_chapters,
        )

        mock_dialogue_prose = mocker.patch(
            "api.chunking.dialogue_prose", return_value=([], [])
        )
        mock_generate_beats = mocker.patch(
            "api.chunking.generate_beats", return_value=([], [])
        )
        mock_sliding_window = mocker.patch(
            "api.chunking.sliding_window", return_value=([], [])
        )

        chunk_text(book, ChunkMethod.DIALOGUE_PROSE)
        mock_dialogue_prose.assert_called_once_with(expected_chapters)

        chunk_text(book, ChunkMethod.GENERATE_BEATS)
        mock_generate_beats.assert_called_once_with(expected_chapters)

        chunk_text(book, ChunkMethod.SLIDING_WINDOW)
        mock_sliding_window.assert_called_once_with(expected_chapters)

    def test_returns_two_lists(self, mocker):
        book = "Chapter 1 text *** Chapter 2 text"
        expected_chapters = ["Chapter 1 text", "Chapter 2 text"]
        mocker.patch(
            "api.data_preparation.separate_into_chapters",
            return_value=expected_chapters,
        )

        result = chunk_text(book, ChunkMethod.DIALOGUE_PROSE)

        separate_into_chapters.assert_called_once_with(book)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # Formatted text chunks
        assert isinstance(result[1], list)  # User messages

    def test_first_list_contains_formatted_chunks(self, mocker):
        book = "Chapter 1 text *** Chapter 2 text"
        expected_chapters = ["Chapter 1 text", "Chapter 2 text"]

        mocker.patch(
            "api.data_preparation.separate_into_chapters",
            return_value=expected_chapters,
        )

        chapters, _ = chunk_text(book, ChunkMethod.DIALOGUE_PROSE)

        separate_into_chapters.assert_called_once_with(book)
        assert chapters == expected_chapters

    def test_user_messages_relevant_to_chunking(self, mocker):
        book = "Chapter 1 text *** Chapter 2 text"
        expected_chapters = ["Chapter 1 text", "Chapter 2 text"]

        mocker.patch(
            "api.data_preparation.separate_into_chapters",
            return_value=expected_chapters,
        )
        mocker.patch(
            "api.chunking.dialogue_prose",
            return_value=(expected_chapters, ["Some user message"]),
        )

        _, user_messages = chunk_text(book, ChunkMethod.DIALOGUE_PROSE)

        assert user_messages == ["Some user message"]
