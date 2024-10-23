import time

import pytest  # noqa F401

from api.write_csv import create_csv_str


class TestCreateCsvStr:
    def test_generate_csv_from_valid_data(self):
        # sourcery skip: class-extract-method
        chunks = ["response1", "response2"]
        user_messages = ["message1", "message2"]
        role = "system"
        expected_csv = '"system","message1","response1"\n"system","message2","response2"\n'  # noqa E501
        result = create_csv_str(chunks, user_messages, role)
        assert result == expected_csv

    def test_role_inclusion_in_csv(self):
        chunks = ["response1"]
        user_messages = ["message1"]
        role = "admin"
        expected_csv = '"admin","message1","response1"\n'
        result = create_csv_str(chunks, user_messages, role)
        assert result == expected_csv

    def test_empty_lists_handling(self):
        chunks = []
        user_messages = []
        role = "system"
        expected_csv = ""
        result = create_csv_str(chunks, user_messages, role)
        assert result == expected_csv

    def test_returns_csv_with_quoted_fields(self):
        chunks = ["response1", "response2"]
        user_messages = ["message1", "message2"]
        role = "system"
        expected_csv = '"system","message1","response1"\n"system","message2","response2"\n'  # noqa E501
        result = create_csv_str(chunks, user_messages, role)
        assert result == expected_csv

    def test_handles_special_characters(self):
        chunks = ["response1,", "response2"]
        user_messages = ["message1", "message2"]
        role = "system"
        expected_csv = '"system","message1","response1,"\n"system","message2","response2"\n'  # noqa E501
        result = create_csv_str(chunks, user_messages, role)
        assert result == expected_csv

    def test_role_included_in_every_row(self):
        chunks = ["response1", "response2"]
        user_messages = ["message1", "message2"]
        role = "system"
        result = create_csv_str(chunks, user_messages, role)
        assert all(row.startswith('"system"') for row in result.splitlines())

    def test_does_not_modify_input_lists(self):
        chunks = ["response1", "response2"]
        chunks_copy = chunks.copy()
        user_messages = ["message1", "message2"]
        user_messages_copy = user_messages.copy()
        role = "system"

        create_csv_str(chunks, user_messages, role)

        assert chunks == chunks_copy
        assert user_messages == user_messages_copy

    def test_multiple_calls_no_side_effects(self):
        chunks1 = ["response1", "response2"]
        chunks2 = ["answer1", "answer2", "answer3"]
        user_messages1 = ["message1", "message2"]
        user_messages2 = ["question1", "question2", "question3"]
        role1 = "system"
        role2 = "user"

        expected1 = '"system","message1","response1"\n"system","message2","response2"\n'  # noqa E501
        expected2 = '"user","question1","answer1"\n"user","question2","answer2"\n"user","question3","answer3"\n'  # noqa E501

        result1 = create_csv_str(chunks1, user_messages1, role1)
        result2 = create_csv_str(chunks2, user_messages2, role2)

        assert result1 == expected1
        assert result2 == expected2

        assert chunks1 == ["response1", "response2"]
        assert user_messages1 == ["message1", "message2"]
        assert result1 != result2
        assert result1 == expected1
        assert result2 == expected2

    def test_performance_large_data_sets(self):
        chunks = ["response" + str(i) for i in range(10000)]
        user_messages = ["message" + str(i) for i in range(10000)]
        role = "system"

        start_time = time.time()
        result = create_csv_str(chunks, user_messages, role)
        end_time = time.time()

        execution_time = end_time - start_time
        assert len(result.splitlines()) == 10000
        assert all(
            line.startswith('"system","message')
            for line in result.splitlines()
        )
        assert execution_time < 1.0
