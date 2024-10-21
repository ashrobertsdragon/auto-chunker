import csv
import io


def create_csv_str(
    chunks: list[str], user_messages: list[str], role: str
) -> str:
    """
    Writes the chunks and user messages to a CSV file.

    Args:
        chunks (list[str]): The list of proposed model responses.
        user_messages (list[str]): The list of user messages.
        role (str): The system message.

    Returns:
        str: The CSV file.
    """
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(
        csv_buffer, quoting=csv.QUOTE_ALL, lineterminator="\n"
    )

    for message, chunk in zip(user_messages, chunks):
        csv_writer.writerow([role, message, chunk])
    csv_buffer.seek(0)

    return csv_buffer.getvalue()
