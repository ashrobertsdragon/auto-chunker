# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: auto-chunker.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'auto-chunker.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12\x61uto-chunker.proto\x12\x0c\x61uto_chunker\"\xb6\x01\n\x0c\x43hunkRequest\x12\x14\n\x0ctext_content\x18\x01 \x01(\t\x12\x42\n\x0f\x63hunking_method\x18\x02 \x01(\x0e\x32).auto_chunker.ChunkRequest.ChunkingMethod\"L\n\x0e\x43hunkingMethod\x12\x12\n\x0eSLIDING_WINDOW\x10\x00\x12\x12\n\x0e\x44IALOGUE_PROSE\x10\x01\x12\x12\n\x0eGENERATE_BEATS\x10\x02\"&\n\rChunkResponse\x12\x15\n\rjsonl_content\x18\x01 \x01(\t2Q\n\x0b\x41utoChunker\x12\x42\n\x05\x43hunk\x12\x1a.auto_chunker.ChunkRequest\x1a\x1b.auto_chunker.ChunkResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'auto_chunker_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_CHUNKREQUEST']._serialized_start=37
  _globals['_CHUNKREQUEST']._serialized_end=219
  _globals['_CHUNKREQUEST_CHUNKINGMETHOD']._serialized_start=143
  _globals['_CHUNKREQUEST_CHUNKINGMETHOD']._serialized_end=219
  _globals['_CHUNKRESPONSE']._serialized_start=221
  _globals['_CHUNKRESPONSE']._serialized_end=259
  _globals['_AUTOCHUNKER']._serialized_start=261
  _globals['_AUTOCHUNKER']._serialized_end=342
# @@protoc_insertion_point(module_scope)