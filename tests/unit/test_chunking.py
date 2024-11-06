import pytest

from auto_chunker.application import chunking
from auto_chunker.application.chunking import (
    chunk_text,
    generate_beats,
    extract_dialogue,
    dialogue_prose,
    sliding_window,
)
from auto_chunker.application.chunking_method import ChunkingMethod
from auto_chunker.application.data_preparation import TOKENIZER


@pytest.fixture
def mock_config(mocker):
    return mocker.patch(
        "auto_chunker.application.chunking.config",
        side_effect=lambda key, **kwargs: {
            "OPENAI_API_KEY": "test_key",
            "OPENAI_ORG_ID": "test_org",
            "OPENAI_PROJECT_ID": "test_project",
            "OPENAI_MODEL": "test_model",
            "MAX_TOKENS": 10,
            "TEMPERATURE": 0.5,
        }.get(key, kwargs.get("default")),
    )


class TestGenerateBeats:
    @pytest.mark.asyncio
    async def test_generates_beats_for_multiple_chapters(self, mocker):
        mocker.patch(
            "auto_chunker.application.chunking.call_gpt_api",
            return_value="Generated beat",
        )
        chapters = ["Chapter 1 text", "Chapter 2 text"]

        chunk_list, user_message_list = await generate_beats(chapters)

        assert len(chunk_list) == 2
        assert len(user_message_list) == 2
        assert chunk_list == chapters
        assert "Generated beat" in user_message_list[0]
        assert "Write" in user_message_list[0]
        assert "words for a chapter" in user_message_list[0]
        assert "Generated beat" in user_message_list[1]
        assert "Write" in user_message_list[1]
        assert "words for a chapter" in user_message_list[1]

    @pytest.mark.asyncio
    async def test_constructs_correct_user_message_format(self, mocker):
        mocker.patch(
            "auto_chunker.application.chunking.call_gpt_api",
            return_value="Generated beat",
        )
        chapter = "This is a test chapter with five words"

        _, user_message_list = await generate_beats([chapter])

        expected_word_count = len(chapter.split())
        message = user_message_list[0]
        assert f"Write {expected_word_count} words" in message
        assert "scene beats" in message
        assert "Generated beat" in message

    @pytest.mark.asyncio
    async def test_preserves_original_chapters_in_chunk_list(self, mocker):
        mocker.patch(
            "auto_chunker.application.chunking.call_gpt_api",
            return_value="Generated beat",
        )
        chapters = ["Chapter 1: A beginning", "Chapter 2: The middle part"]

        chunk_list, _ = await generate_beats(chapters)

        assert chunk_list == chapters
        assert isinstance(chunk_list[0], str)
        assert isinstance(chunk_list[1], str)

    @pytest.mark.asyncio
    async def test_handles_single_word_chapter(self, mocker):
        mocker.patch(
            "auto_chunker.application.chunking.call_gpt_api",
            return_value="Generated beat",
        )
        chapter = "OneWordChapter"

        chunk_list, user_message_list = await generate_beats([chapter])

        assert len(chunk_list) == 1
        assert len(user_message_list) == 1
        assert "Write 1 words" in user_message_list[0]

    @pytest.mark.asyncio
    async def test_returns_correct_tuple_structure(self, mocker):
        mocker.patch(
            "auto_chunker.application.chunking.call_gpt_api",
            return_value="Generated beat",
        )
        chapters = ["Test chapter"]

        result = await generate_beats(chapters)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)


class TestExtractDialogue:
    def test_basic_double_quotes(self):  # sourcery skip: class-extract-method
        paragraph = 'He said, "Hello there!" and waved.'
        dialogue, prose = extract_dialogue(paragraph)
        assert prose == "He said, and waved."
        assert dialogue == "Hello there!"

    def test_dialogue_with_interruption(self):
        paragraph = '"Stop," he said, "you\'re going too fast!"'
        dialogue, prose = extract_dialogue(paragraph)
        assert prose == "he said"
        assert dialogue == "Stop, you're going too fast!"

    def test_unbalanced_quotes(self):
        paragraph = 'He said, "This is a test.'
        dialogue, prose = extract_dialogue(paragraph)
        assert prose == "He said"
        assert dialogue == "This is a test."

    def test_empty_string(self):
        dialogue, prose = extract_dialogue("")
        assert prose == ""
        assert dialogue == ""

    def test_nested_quotes(self):
        paragraph = """She said, "He told me 'Hello there!'" and smiled."""
        dialogue, prose = extract_dialogue(paragraph)
        assert prose == "She said, and smiled."
        assert dialogue == "He told me 'Hello there!'"

    def test_consecutive_punctuation(self):
        paragraph = 'She exclaimed, "Oh my...!" and ran.'
        dialogue, prose = extract_dialogue(paragraph)
        assert prose == "She exclaimed, and ran."
        assert dialogue == "Oh my...!"

    def test_only_prose(self):
        paragraph = "The sun was setting beautifully."
        dialogue, prose = extract_dialogue(paragraph)
        assert prose == "The sun was setting beautifully."
        assert dialogue == ""

    def test_only_dialogue(self):
        paragraph = '"This is all dialogue!"'
        dialogue, prose = extract_dialogue(paragraph)
        assert prose == ""
        assert dialogue == "This is all dialogue!"


class TestDialogueProse:
    def test_correct_split_prose_and_dialogue(self):
        chapters = [
            'He said, "Hello!"\nShe replied, "Hi!"',
            'The sun was setting.\n"Beautiful evening," he remarked.',
        ]
        expected_chunks = [
            "Hello!",
            "He said",
            "Hi!",
            "She replied",
            "The sun was setting.",
            "Beautiful evening",
            "he remarked.",
        ]
        expected_messages = [
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_accurate_sentence_counts(self):
        chapters = [
            '"Hello!" she said. "How are you?"\nHe nodded. "I am fine."'
        ]
        expected_chunks = [
            "Hello! How are you?",
            "she said.",
            "I am fine.",
            "He nodded.",
        ]
        expected_messages = [
            "Write 2 sentences of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
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
            "Take cover!",
            "The storm raged. he shouted.",
            "She ran inside.",
        ]
        expected_messages = [
            "Write 1 sentence of dialogue",
            "Write 2 sentences of description and action",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_generates_user_messages(self):
        chapters = ['She whispered, "Careful."\nThe door creaked open.']
        expected_chunks = [
            "Careful.",
            "She whispered",
            "The door creaked open.",
        ]
        expected_messages = [
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_processes_multiple_chapters(self):
        chapters = [
            'She smiled.\n"Nice weather," he said.',
            'The rain fell.\n"Indeed," she replied.',
        ]
        expected_chunks = [
            "She smiled.",
            "Nice weather",
            "he said.",
            "The rain fell.",
            "Indeed",
            "she replied.",
        ]
        expected_messages = [
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_handles_chapters_no_punctuation(self):
        chapters = ["He spoke softly", '"Hello," she said', "The wind blew"]
        expected_chunks = [
            "He spoke softly",
            "Hello",
            "she said",
            "The wind blew",
        ]
        expected_messages = [
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_supports_varying_paragraph_lengths(self):
        chapters = [
            'Short line.\n"Brief."\nLong paragraph with multiple sentences. More text.',
            "Another chapter.",
        ]
        expected_chunks = [
            "Short line.",
            "Brief.",
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
            "First dialogue.",
            "Second action.",
            "Second dialogue.",
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
        chapters = ['He said...!\n"Well...?"\nShe replied?!\n"Hmm..."']
        expected_chunks = [
            "He said...!",
            "Well...?",
            "She replied?!",
            "Hmm...",
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
        chapters = [
            'He yelled, "Stop!!"\nShe screamed, "No!!!"',
            '"My email address is fake@fake.com"',
            '"What the $%*&," she swore."',
        ]
        expected_chunks = [
            "Stop!!",
            "He yelled",
            "No!!!",
            "She screamed",
            "My email address is fake@fake.com",
            "What the $%*&",
            "she swore.",
        ]
        expected_messages = [
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages

    def test_sentence_count_grammar(self):
        chapters = [
            '"First," he said.\nShe smiled. "You are too much. You are so funny." She laughed.'
        ]
        expected_chunks = [
            "First",
            "he said.",
            "You are too much. You are so funny.",
            "She smiled. She laughed.",
        ]
        expected_messages = [
            "Write 1 sentence of dialogue",
            "Write 1 sentence of description and action",
            "Write 2 sentences of dialogue",
            "Write 2 sentences of description and action",
        ]
        chunks, user_messages = dialogue_prose(chapters)
        assert chunks == expected_chunks
        assert user_messages == expected_messages


class TestSlidingWindow:
    def test_processes_chapters_into_chunks_and_user_messages(
        self, mock_config
    ):
        """Split text into chunks while respecting maximum token size."""
        chapter = """In the bustling city of New York, where dreams take flight and ambitions soar high above the towering skyscrapers, Sarah found herself at a crossroads. The decision she faced would not only shape her career but potentially alter the course of her life. As she sat in her favorite coffee shop on 5th Avenue, watching the endless stream of pedestrians hurrying past the window, she contemplated her choices. The startup she had poured her heart into for the past three years had finally received an acquisition offer from a major tech company. The numbers were impressive - life-changing, even. But accepting would mean giving up control of her vision, the very thing that had driven her to create this company in the first place."""  # noqa: E501
        expected_num_chunks = 14
        expected_num_user_messages = 14

        chunks, user_messages = sliding_window([chapter])

        assert len(chunks) == expected_num_chunks
        assert len(user_messages) == expected_num_user_messages

    def test_handles_chapters_with_varying_lengths(self):
        """Handle multiple chapters of different lengths correctly."""
        chapters = [
            "Short",
            "Four score and seven years ago, our fathers brought forth, on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.",
            "Another short chapter",
            "Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.",
            "Third short chapter",
            "Fourth short chapter",
            "But, in a larger sense, we can not dedicate—we can not consecrate—we can not hallow—this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us—that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion—that we here highly resolve that these dead shall not have died in vain—that this nation, under God, shall have a new birth of freedom—and that government of the people, by the people, for the people, shall not perish from the earth.",
            "Last chapter",
        ]
        chunks, user_messages = sliding_window(chapters)
        assert "Short" in user_messages
        assert "Last chapter" in chunks
        assert len(chunks) == 7
        assert len(user_messages) == 7

    def test_handles_no_line_breaks(self, mock_config):
        """Process text without natural break points."""
        chapters = ["ThisIsAVeryLongWordWithoutBreaks"]
        chunks, _ = sliding_window(chapters)
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_chapters_shorter_than_chunk_size(self):
        """Keep short chapters intact without splitting."""
        chapters = ["Short chapter.", "Another short chapter."]
        chunks, user_messages = sliding_window(chapters)
        assert chunks == ["Another short chapter."]
        assert user_messages == ["Short chapter."]

    def test_handles_non_ascii_characters(self, mock_config):
        """Process text containing non-ASCII characters correctly."""
        chapters = ["Hello 你好 नमस्ते"]
        chunks, _ = sliding_window(chapters)
        assert all(len(TOKENIZER.encode(chunk)) <= 10 for chunk in chunks)

    def test_performance_with_large_input(self, mock_config):
        """Handle large number of chapters efficiently."""
        chapters = ["Chapter text"] * 1000
        chunks, user_messages = sliding_window(chapters)
        assert len(chunks) == len(user_messages)
        assert all(len(TOKENIZER.encode(chunk)) <= 10 for chunk in chunks)

    def test_special_characters(self, mock_config):
        """Process text with special characters properly."""
        chapters = ["Text with !@#$%^&*()"]
        chunks, _ = sliding_window(chapters)
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_exact_chunk_size(self, mock_config):
        """Handle text exactly matching chunk size."""
        chapters = ["ABCD", "1234567890"]
        chunks, _ = sliding_window(chapters)
        assert len(chunks[0]) == 10
        assert len(chunks) == 1


class TestChunkText:
    def test_calls_separate_into_chapters(self, mocker):
        book = "Chapter 1 text *** Chapter 2 text"
        mocker.patch(
            "auto_chunker.application.chunking.dialogue_prose",
            return_value=([], []),
        )
        separate_spy = mocker.spy(chunking, "separate_into_chapters")

        chunk_text(book, ChunkingMethod.DIALOGUE_PROSE)

        separate_spy.assert_called_once_with(book)

    def test_maps_chunk_type_to_correct_function(self, mocker):
        book = "Chapter 1 text *** Chapter 2 text"
        mock_func = mocker.patch(
            "auto_chunker.application.chunking.dialogue_prose",
            return_value=([], []),
        )

        chunk_text(book, ChunkingMethod.DIALOGUE_PROSE)

        mock_func.assert_called_once()

    def test_raises_value_error_for_unsupported_chunk_type(self):
        book = "Chapter 1 text *** Chapter 2 text"

        with pytest.raises(ValueError, match="Chunk method .* not supported"):
            chunk_text(book, "UNSUPPORTED_METHOD")

    def test_handles_empty_book_input_gracefully(self, mocker):
        book = ""
        mock_func = mocker.patch(
            "auto_chunker.application.chunking.dialogue_prose",
            return_value=([], []),
        )

        chapters, user_messages = chunk_text(
            book, ChunkingMethod.DIALOGUE_PROSE
        )

        assert chapters == []
        assert user_messages == []
        mock_func.assert_called_once_with([""])

    def test_correct_function_called_for_chunk_type(self, mocker):
        book = "Chapter 1 text *** Chapter 2 text"
        expected_chapters = ["Chapter 1 text", "Chapter 2 text"]

        mocker.patch(
            "auto_chunker.application.chunking.separate_into_chapters",
            return_value=expected_chapters,
        )

        mock_dialogue_prose = mocker.patch(
            "auto_chunker.application.chunking.dialogue_prose",
            return_value=([], []),
        )
        mock_generate_beats = mocker.patch(
            "auto_chunker.application.chunking.generate_beats",
            return_value=([], []),
        )
        mock_sliding_window = mocker.patch(
            "auto_chunker.application.chunking.sliding_window",
            return_value=([], []),
        )

        chunk_text(book, ChunkingMethod.DIALOGUE_PROSE)
        mock_dialogue_prose.assert_called_once_with(expected_chapters)

        chunk_text(book, ChunkingMethod.GENERATE_BEATS)
        mock_generate_beats.assert_called_once_with(expected_chapters)

        chunk_text(book, ChunkingMethod.SLIDING_WINDOW)
        mock_sliding_window.assert_called_once_with(expected_chapters)

    def test_returns_two_lists(self, mocker):
        book = "Chapter 1 text *** Chapter 2 text"
        expected_chapters = ["Chapter 1 text", "Chapter 2 text"]
        mock_separate = mocker.patch(
            "auto_chunker.application.chunking.separate_into_chapters",
            return_value=expected_chapters,
        )

        result = chunk_text(book, ChunkingMethod.DIALOGUE_PROSE)

        mock_separate.assert_called_once_with(book)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # Formatted text chunks
        assert isinstance(result[1], list)  # User messages

    def test_user_messages_relevant_to_chunking(self, mocker):
        book = "Chapter 1 text *** Chapter 2 text"
        expected_chapters = ["Chapter 1 text", "Chapter 2 text"]

        mocker.patch(
            "auto_chunker.application.chunking.separate_into_chapters",
            return_value=expected_chapters,
        )
        mocker.patch(
            "chunking.dialogue_prose",
            return_value=(expected_chapters, ["Some user message"]),
        )

        _, user_messages = chunk_text(book, ChunkingMethod.DIALOGUE_PROSE)

        assert user_messages == ["Some user message"]
