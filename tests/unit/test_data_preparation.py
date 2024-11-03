import pytest
import time


from application.data_preparation import (
    separate_into_chapters,
    count_tokens,
)


class TestCountTokens:
    def test_tokenizes_simple_sentence(self):
        text = "Hello world"
        expected_tokens = [9906, 1917]
        expected_num_tokens = 2

        tokens, num_tokens = count_tokens(text)

        assert tokens == expected_tokens
        assert num_tokens == expected_num_tokens

    def test_returns_correct_number_of_tokens(self):
        text = "This is a test sentence."
        expected_num_tokens = 6

        _, num_tokens = count_tokens(text)

        assert num_tokens == expected_num_tokens

    def test_handles_very_long_text(self):
        text = "word " * 10000
        expected_num_tokens = 10001  # "word" + " word" * 9999 + " "

        _, num_tokens = count_tokens(text)

        assert num_tokens == expected_num_tokens

    def test_manages_whitespace_only_text(self):
        text = "    "
        expected_tokens = [257]
        expected_num_tokens = 1

        tokens, num_tokens = count_tokens(text)

        assert tokens == expected_tokens
        assert num_tokens == expected_num_tokens

    def test_processes_text_with_special_characters(self):
        text = "Hello@world!"
        expected_tokens = [9906, 31, 14957, 0]
        expected_num_tokens = 4

        tokens, num_tokens = count_tokens(text)

        assert tokens == expected_tokens
        assert num_tokens == expected_num_tokens

    def test_handles_empty_string_input(self):
        text = ""
        expected_tokens = []
        expected_num_tokens = 0

        tokens, num_tokens = count_tokens(text)

        assert tokens == expected_tokens
        assert num_tokens == expected_num_tokens

    @pytest.mark.parametrize(
        "text, expected_tokens",
        [
            (
                "This is a\nmultiline\ntext",
                [2028, 374, 264, 198, 93860, 198, 1342],
            ),
            ("aaa", [33746]),
            ("Bonjour le monde", [82681, 514, 38900]),
        ],
    )
    def test_various_text_inputs(self, text, expected_tokens):
        tokens, num_tokens = count_tokens(text)

        assert tokens == expected_tokens
        assert num_tokens == len(expected_tokens)

    def test_tokenizer_called_correctly(self, mocker):
        text = "Test text"

        mock_tokenizer = mocker.patch(
            "data_preparation.TOKENIZER.encode", return_value=[1, 2, 3]
        )
        count_tokens(text)

        mock_tokenizer.assert_called_once_with(text)

    def test_performance_with_large_text(self):
        large_text = "word " * 100000

        start_time = time.time()
        _, num_tokens = count_tokens(large_text)
        end_time = time.time()

        assert num_tokens == 100001
        assert end_time - start_time < 1  # 1 second


class TestSeparateIntoChapters:
    def test_split_with_asterisks(self):  # sourcery skip: class-extract-method
        text = "Chapter 1 *** Chapter 2 *** Chapter 3"
        expected = ["Chapter 1", "Chapter 2", "Chapter 3"]
        result = separate_into_chapters(text)
        assert result == expected

    def test_valid_text_input(self):
        text = "Introduction *** Body *** Conclusion"
        expected = ["Introduction", "Body", "Conclusion"]
        result = separate_into_chapters(text)
        assert result == expected

    def test_no_separators(self):
        text = "This is a single chapter text"
        expected = ["This is a single chapter text"]
        result = separate_into_chapters(text)
        assert result == expected

    def test_whitespace_around_separators(self):
        text = "Chapter 1  ***  Chapter 2  ***  Chapter 3"
        expected = ["Chapter 1 ", "Chapter 2 ", "Chapter 3"]
        result = separate_into_chapters(text)
        assert result == expected

    def test_manages_empty_string_input(self):
        text = ""
        expected = [""]
        result = separate_into_chapters(text)
        assert result == expected

    def test_processes_large_text_inputs_efficiently(self):
        text = "Chapter 1 *** " + " *** ".join(
            f"Chapter {i}" for i in range(2, 100000)
        )

        start = time.time()
        separate_into_chapters(text)
        end = time.time()
        assert end - start < 1

    # Handles non-standard whitespace characters around separators
    def test_handles_non_standard_whitespace(self):
        text = "Chapter 1  *** Chapter 2 *** Chapter 3"
        expected = ["Chapter 1 ", "Chapter 2", "Chapter 3"]
        result = separate_into_chapters(text)
        assert result == expected

    def test_maintains_text_integrity_when_no_separators_present(self):
        text = "Chapter 1 Chapter 2 Chapter 3"
        expected = ["Chapter 1 Chapter 2 Chapter 3"]
        result = separate_into_chapters(text)
        assert result == expected

    def test_no_data_loss_or_modification(self):
        text = "Chapter 1 *** Chapter 2 *** Chapter 3"
        expected = ["Chapter 1", "Chapter 2", "Chapter 3"]
        result = separate_into_chapters(text)
        assert result == expected
