# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: jsonl-file-creator.proto
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
    'jsonl-file-creator.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18jsonl-file-creator.proto\x12\x12jsonl_file_creator\")\n\x12\x43reateJsonlRequest\x12\x13\n\x0b\x63sv_content\x18\x01 \x01(\t\",\n\x13\x43reateJsonlResponse\x12\x15\n\rjsonl_content\x18\x01 \x01(\t2t\n\x10JsonlFileCreator\x12`\n\x0b\x43reateJsonl\x12&.jsonl_file_creator.CreateJsonlRequest\x1a\'.jsonl_file_creator.CreateJsonlResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'jsonl_file_creator_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_CREATEJSONLREQUEST']._serialized_start=48
  _globals['_CREATEJSONLREQUEST']._serialized_end=89
  _globals['_CREATEJSONLRESPONSE']._serialized_start=91
  _globals['_CREATEJSONLRESPONSE']._serialized_end=135
  _globals['_JSONLFILECREATOR']._serialized_start=137
  _globals['_JSONLFILECREATOR']._serialized_end=253
# @@protoc_insertion_point(module_scope)
