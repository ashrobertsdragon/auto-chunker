import csv

import pytest  # noqa F401

from auto_chunker.application.chunking import chunk_text
from auto_chunker.application.chunking_method import ChunkingMethod
from auto_chunker.application.write_csv import create_csv_str


def test_dialogue_prose(book):
    role = "system"
    chapters = len(book.split("***"))
    chunks, user_messages = chunk_text(book, ChunkingMethod.DIALOGUE_PROSE)
    csv_str = create_csv_str(chunks, user_messages, role)
    reader = csv.reader(csv_str.splitlines())
    rows = list(reader)
    for i, row in enumerate(rows, start=1):
        assert len(row) == 3, f"Row {i + 1} has {len(row)} columns, expected 3"


def test_sliding_window(book):
    role = "system"
    chapters = len(book.split("***"))
    chunks, user_messages = chunk_text(book, ChunkingMethod.SLIDING_WINDOW)
    csv_str = create_csv_str(chunks, user_messages, role)
    reader = csv.reader(csv_str.splitlines())
    rows = list(reader)
    for i, row in enumerate(rows, start=1):
        assert len(row) == 3, f"Row {i + 1} has {len(row)} columns, expected 3"
    assert (
        len(rows) == chapters - 1
    ), f"CSV has {len(rows)} rows, expected {chapters - 1}"


def test_generate_beats(book):
    role = "system"
    chapters = len(book.split("***"))
    chunks, user_messages = chunk_text(book, ChunkingMethod.GENERATE_BEATS)
    csv_str = create_csv_str(chunks, user_messages, role)
    reader = csv.reader(csv_str.splitlines())
    rows = list(reader)
    for i, row in enumerate(rows, start=1):
        assert len(row) == 3, f"Row {i + 1} has {len(row)} columns, expected 3"
    assert (
        len(rows) == chapters
    ), f"CSV has {len(rows)} rows, expected {chapters}"
