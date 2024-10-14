import grpc
from concurrent import futures
import _proto.auto_chunker_pb2 as auto_chunker
import _proto.auto_chunker_pb2_grpc as auto_chunker_grpc
import _proto.jsonl_file_creator_pb2 as jsonl_file_creator
import _proto.jsonl_file_creator_pb2_grpc as jsonl_file_creator_grpc

import write_csv
from chunking import chunk_book


def chunk(request: auto_chunker.ChunkRequest) -> auto_chunker.ChunkResponse:
    chunks, user_messages = chunk_book(
        request.text_content, request.role, request.chunking_method
    )
    csv_content = write_csv.create_csv_str(chunks, user_messages, request.role)
    with grpc.insecure_channel("jsonl-file-creator:50053") as channel:
        stub = jsonl_file_creator_grpc.JsonlFileCreatorStub(channel)
        response = stub.CreateJsonl(
            jsonl_file_creator.CreateJsonlRequest(csv_content=csv_content)
        )
        jsonl_content = response.jsonl_content
    return auto_chunker.ChunkResponse(jsonl_content=jsonl_content)


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
