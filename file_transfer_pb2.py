# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: file_transfer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13\x66ile_transfer.proto\x12\x0c\x66iletransfer\"\x1b\n\x0b\x46ileRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\"\x1f\n\x0c\x46ileResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\"+\n\tFileChunk\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x10\n\x08\x66ilename\x18\x02 \x01(\t2S\n\x0c\x46ileTransfer\x12\x43\n\nUploadFile\x12\x17.filetransfer.FileChunk\x1a\x1a.filetransfer.FileResponse(\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'file_transfer_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_FILEREQUEST']._serialized_start=37
  _globals['_FILEREQUEST']._serialized_end=64
  _globals['_FILERESPONSE']._serialized_start=66
  _globals['_FILERESPONSE']._serialized_end=97
  _globals['_FILECHUNK']._serialized_start=99
  _globals['_FILECHUNK']._serialized_end=142
  _globals['_FILETRANSFER']._serialized_start=144
  _globals['_FILETRANSFER']._serialized_end=227
# @@protoc_insertion_point(module_scope)
