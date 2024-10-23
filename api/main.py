import grpc
from concurrent import futures
import api._proto.auto_chunker_pb2 as auto_chunker
import api._proto.auto_chunker_pb2_grpc as auto_chunker_grpc
import api._proto.jsonl_file_creator_pb2 as jsonl_file_creator
import api._proto.jsonl_file_creator_pb2_grpc as jsonl_file_creator_grpc

import api.write_csv as write_csv
from api.chunking import chunk_text


def chunk(request: auto_chunker.ChunkRequest) -> auto_chunker.ChunkResponse:
    work: tuple[list[str], list[str]] | auto_chunker.ChunkResponse = (
        chunk_text(request.text_content, request.chunking_method)
    )
    if work.status_message:
        return work
    chunks, user_messages = work
    csv_content = write_csv.create_csv_str(chunks, user_messages, request.role)
    with grpc.insecure_channel("jsonl-file-creator:50053") as channel:
        stub = jsonl_file_creator_grpc.JsonlFileCreatorStub(channel)
        response = stub.CreateJsonl(
            jsonl_file_creator.CreateJsonlRequest(csv_content=csv_content)
        )
        jsonl_content = response.jsonl_content
    return auto_chunker.ChunkResponse(
        jsonl_content=jsonl_content, status_message=""
    )


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    auto_chunker_grpc.add_AutoChunkerServicer_to_server(
        auto_chunker_grpc.AutoChunkerServicer(Chunk=chunk), server
    )
    server.add_listen_addr("[::]:50052", grpc.insecure_port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
