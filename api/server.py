import grpc
from concurrent import futures
from decouple import config

import api._proto.auto_chunker_pb2 as auto_chunker
import api._proto.auto_chunker_pb2_grpc as auto_chunker_grpc
import api._proto.jsonl_file_creator_pb2 as jsonl_file_creator
import api._proto.jsonl_file_creator_pb2_grpc as jsonl_file_creator_grpc
import api.write_csv as write_csv
from api.chunking import chunk_text


class Authentication:
    def __init__(self):
        self.service_token: str = config("SERVICE_TOKEN")
        self.authentication_status: bool = False

    def authenticate(self, context) -> bool:
        metadata = dict(context.invocation_metadata())
        auth_token = metadata.get("authorization", "").replace("Bearer ", "")
        if auth_token == self.service_token:
            self.authentication_status = True
        else:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("Invalid authentication credentials")
        return self.authentication_status


def chunk(
    request: auto_chunker.ChunkRequest, context: grpc.ServicerContext
) -> auto_chunker.ChunkResponse:
    auth = Authentication().authenticate(context)
    if not auth:
        return auto_chunker.ChunkResponse(
            jsonl_content="", status_message=context.details()
        )
    work: tuple[list[str], list[str]] | auto_chunker.ChunkResponse = (
        chunk_text(request.text_content, request.chunking_method)
    )
    if hasattr(work, "status_message"):
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


def get_server() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    auto_chunker_grpc.add_AutoChunkerServicer_to_server(
        auto_chunker_grpc.AutoChunkerServicer(Chunk=chunk), server
    )
    server.add_secure_port("[::]:50052", grpc.local_server_credentials())
    return server


if __name__ == "__main__":
    server = get_server()
    server.start()
    server.wait_for_termination()
