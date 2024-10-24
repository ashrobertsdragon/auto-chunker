import pytest  # noqa F401

import csv

from api.chunking import chunk_text
from api.chunking_method import ChunkingMethod
from api.write_csv import create_csv_str


def test_all(book):
    csv_contents: dict[str, str] = {}
    role = "system"
    chapters = len(book.split("***"))
    for method in ChunkingMethod:
        chunks, user_messages = chunk_text(book, method)
        csv_contents[method.value] = create_csv_str(
            chunks, user_messages, role
        )

    assert len(csv_contents) == len(ChunkingMethod)
    for method_name, csv_str in csv_contents.items():
        reader = csv.reader(csv_str.splitlines())
        rows = list(reader)
        for i, row in enumerate(rows, start=1):
            assert (
                len(row) == 3
            ), f"Row {i + 1} in CSV for {method_name} has {len(row)} columns, expected 3"
        assert (
            len(rows) == chapters
        ), f"CSV for {method_name} has {len(rows)} rows, expected {chapters}"
