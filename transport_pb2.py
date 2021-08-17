# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: transport.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='transport.proto',
  package='transport',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0ftransport.proto\x12\ttransport\"\xbb\x01\n\x08ReadyReq\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\r\n\x05\x63name\x18\x02 \x01(\t\x12\x1f\n\x05state\x18\x03 \x01(\x0e\x32\x10.transport.State\x12/\n\x06\x63onfig\x18\x04 \x03(\x0b\x32\x1f.transport.ReadyReq.ConfigEntry\x1a@\n\x0b\x43onfigEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.transport.Scalar:\x02\x38\x01\"\xb1\x01\n\tUpdateReq\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x19\n\x0c\x62uffer_chunk\x18\x02 \x01(\x0cH\x00\x88\x01\x01\x12\x12\n\x05title\x18\x03 \x01(\tH\x01\x88\x01\x01\x12$\n\x05state\x18\x04 \x01(\x0e\x32\x10.transport.StateH\x02\x88\x01\x01\x12\x12\n\x05\x63name\x18\x05 \x01(\tH\x03\x88\x01\x01\x42\x0f\n\r_buffer_chunkB\x08\n\x06_titleB\x08\n\x06_stateB\x08\n\x06_cname\"y\n\x10transportRequest\x12(\n\tready_req\x18\x01 \x01(\x0b\x32\x13.transport.ReadyReqH\x00\x12*\n\nupdate_req\x18\x02 \x01(\x0b\x32\x14.transport.UpdateReqH\x00\x42\x0f\n\rrequest_oneof\"}\n\x08ReadyRep\x12/\n\x06\x63onfig\x18\x01 \x03(\x0b\x32\x1f.transport.ReadyRep.ConfigEntry\x1a@\n\x0b\x43onfigEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.transport.Scalar:\x02\x38\x01\"c\n\tUpdateRep\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x19\n\x0c\x62uffer_chunk\x18\x02 \x01(\x0cH\x00\x88\x01\x01\x12\x12\n\x05title\x18\x03 \x01(\tH\x01\x88\x01\x01\x42\x0f\n\r_buffer_chunkB\x08\n\x06_title\"{\n\x11transportResponse\x12(\n\tready_rep\x18\x01 \x01(\x0b\x32\x13.transport.ReadyRepH\x00\x12*\n\nupdate_rep\x18\x02 \x01(\x0b\x32\x14.transport.UpdateRepH\x00\x42\x10\n\x0eresponse_oneof\"r\n\x06Scalar\x12\x12\n\x08scdouble\x18\x01 \x01(\x01H\x00\x12\x11\n\x07scfloat\x18\x02 \x01(\x02H\x00\x12\x11\n\x07scint32\x18\x03 \x01(\x05H\x00\x12\x12\n\x08scstring\x18\x04 \x01(\tH\x00\x12\x10\n\x06scbool\x18\x05 \x01(\x08H\x00\x42\x08\n\x06scalar*@\n\x05State\x12\x08\n\x04NONE\x10\x00\x12\x06\n\x02ON\x10\x01\x12\x07\n\x03OFF\x10\x02\x12\x0c\n\x08TRAINING\x10\x03\x12\x0e\n\nTRAIN_DONE\x10\x04\x32`\n\x10TransportService\x12L\n\ttransport\x12\x1b.transport.transportRequest\x1a\x1c.transport.transportResponse\"\x00(\x01\x30\x01\x62\x06proto3'
)

_STATE = _descriptor.EnumDescriptor(
  name='State',
  full_name='transport.State',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ON', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='OFF', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TRAINING', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TRAIN_DONE', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=992,
  serialized_end=1056,
)
_sym_db.RegisterEnumDescriptor(_STATE)

State = enum_type_wrapper.EnumTypeWrapper(_STATE)
NONE = 0
ON = 1
OFF = 2
TRAINING = 3
TRAIN_DONE = 4



_READYREQ_CONFIGENTRY = _descriptor.Descriptor(
  name='ConfigEntry',
  full_name='transport.ReadyReq.ConfigEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='transport.ReadyReq.ConfigEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='transport.ReadyReq.ConfigEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=154,
  serialized_end=218,
)

_READYREQ = _descriptor.Descriptor(
  name='ReadyReq',
  full_name='transport.ReadyReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='transport.ReadyReq.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cname', full_name='transport.ReadyReq.cname', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='state', full_name='transport.ReadyReq.state', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='config', full_name='transport.ReadyReq.config', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_READYREQ_CONFIGENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=218,
)


_UPDATEREQ = _descriptor.Descriptor(
  name='UpdateReq',
  full_name='transport.UpdateReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='transport.UpdateReq.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='buffer_chunk', full_name='transport.UpdateReq.buffer_chunk', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='title', full_name='transport.UpdateReq.title', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='state', full_name='transport.UpdateReq.state', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cname', full_name='transport.UpdateReq.cname', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='_buffer_chunk', full_name='transport.UpdateReq._buffer_chunk',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
    _descriptor.OneofDescriptor(
      name='_title', full_name='transport.UpdateReq._title',
      index=1, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
    _descriptor.OneofDescriptor(
      name='_state', full_name='transport.UpdateReq._state',
      index=2, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
    _descriptor.OneofDescriptor(
      name='_cname', full_name='transport.UpdateReq._cname',
      index=3, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=221,
  serialized_end=398,
)


_TRANSPORTREQUEST = _descriptor.Descriptor(
  name='transportRequest',
  full_name='transport.transportRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ready_req', full_name='transport.transportRequest.ready_req', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='update_req', full_name='transport.transportRequest.update_req', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='request_oneof', full_name='transport.transportRequest.request_oneof',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=400,
  serialized_end=521,
)


_READYREP_CONFIGENTRY = _descriptor.Descriptor(
  name='ConfigEntry',
  full_name='transport.ReadyRep.ConfigEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='transport.ReadyRep.ConfigEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='transport.ReadyRep.ConfigEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=154,
  serialized_end=218,
)

_READYREP = _descriptor.Descriptor(
  name='ReadyRep',
  full_name='transport.ReadyRep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='config', full_name='transport.ReadyRep.config', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_READYREP_CONFIGENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=523,
  serialized_end=648,
)


_UPDATEREP = _descriptor.Descriptor(
  name='UpdateRep',
  full_name='transport.UpdateRep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='transport.UpdateRep.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='buffer_chunk', full_name='transport.UpdateRep.buffer_chunk', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='title', full_name='transport.UpdateRep.title', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='_buffer_chunk', full_name='transport.UpdateRep._buffer_chunk',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
    _descriptor.OneofDescriptor(
      name='_title', full_name='transport.UpdateRep._title',
      index=1, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=650,
  serialized_end=749,
)


_TRANSPORTRESPONSE = _descriptor.Descriptor(
  name='transportResponse',
  full_name='transport.transportResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ready_rep', full_name='transport.transportResponse.ready_rep', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='update_rep', full_name='transport.transportResponse.update_rep', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='response_oneof', full_name='transport.transportResponse.response_oneof',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=751,
  serialized_end=874,
)


_SCALAR = _descriptor.Descriptor(
  name='Scalar',
  full_name='transport.Scalar',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='scdouble', full_name='transport.Scalar.scdouble', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scfloat', full_name='transport.Scalar.scfloat', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scint32', full_name='transport.Scalar.scint32', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scstring', full_name='transport.Scalar.scstring', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scbool', full_name='transport.Scalar.scbool', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='scalar', full_name='transport.Scalar.scalar',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=876,
  serialized_end=990,
)

_READYREQ_CONFIGENTRY.fields_by_name['value'].message_type = _SCALAR
_READYREQ_CONFIGENTRY.containing_type = _READYREQ
_READYREQ.fields_by_name['state'].enum_type = _STATE
_READYREQ.fields_by_name['config'].message_type = _READYREQ_CONFIGENTRY
_UPDATEREQ.fields_by_name['state'].enum_type = _STATE
_UPDATEREQ.oneofs_by_name['_buffer_chunk'].fields.append(
  _UPDATEREQ.fields_by_name['buffer_chunk'])
_UPDATEREQ.fields_by_name['buffer_chunk'].containing_oneof = _UPDATEREQ.oneofs_by_name['_buffer_chunk']
_UPDATEREQ.oneofs_by_name['_title'].fields.append(
  _UPDATEREQ.fields_by_name['title'])
_UPDATEREQ.fields_by_name['title'].containing_oneof = _UPDATEREQ.oneofs_by_name['_title']
_UPDATEREQ.oneofs_by_name['_state'].fields.append(
  _UPDATEREQ.fields_by_name['state'])
_UPDATEREQ.fields_by_name['state'].containing_oneof = _UPDATEREQ.oneofs_by_name['_state']
_UPDATEREQ.oneofs_by_name['_cname'].fields.append(
  _UPDATEREQ.fields_by_name['cname'])
_UPDATEREQ.fields_by_name['cname'].containing_oneof = _UPDATEREQ.oneofs_by_name['_cname']
_TRANSPORTREQUEST.fields_by_name['ready_req'].message_type = _READYREQ
_TRANSPORTREQUEST.fields_by_name['update_req'].message_type = _UPDATEREQ
_TRANSPORTREQUEST.oneofs_by_name['request_oneof'].fields.append(
  _TRANSPORTREQUEST.fields_by_name['ready_req'])
_TRANSPORTREQUEST.fields_by_name['ready_req'].containing_oneof = _TRANSPORTREQUEST.oneofs_by_name['request_oneof']
_TRANSPORTREQUEST.oneofs_by_name['request_oneof'].fields.append(
  _TRANSPORTREQUEST.fields_by_name['update_req'])
_TRANSPORTREQUEST.fields_by_name['update_req'].containing_oneof = _TRANSPORTREQUEST.oneofs_by_name['request_oneof']
_READYREP_CONFIGENTRY.fields_by_name['value'].message_type = _SCALAR
_READYREP_CONFIGENTRY.containing_type = _READYREP
_READYREP.fields_by_name['config'].message_type = _READYREP_CONFIGENTRY
_UPDATEREP.oneofs_by_name['_buffer_chunk'].fields.append(
  _UPDATEREP.fields_by_name['buffer_chunk'])
_UPDATEREP.fields_by_name['buffer_chunk'].containing_oneof = _UPDATEREP.oneofs_by_name['_buffer_chunk']
_UPDATEREP.oneofs_by_name['_title'].fields.append(
  _UPDATEREP.fields_by_name['title'])
_UPDATEREP.fields_by_name['title'].containing_oneof = _UPDATEREP.oneofs_by_name['_title']
_TRANSPORTRESPONSE.fields_by_name['ready_rep'].message_type = _READYREP
_TRANSPORTRESPONSE.fields_by_name['update_rep'].message_type = _UPDATEREP
_TRANSPORTRESPONSE.oneofs_by_name['response_oneof'].fields.append(
  _TRANSPORTRESPONSE.fields_by_name['ready_rep'])
_TRANSPORTRESPONSE.fields_by_name['ready_rep'].containing_oneof = _TRANSPORTRESPONSE.oneofs_by_name['response_oneof']
_TRANSPORTRESPONSE.oneofs_by_name['response_oneof'].fields.append(
  _TRANSPORTRESPONSE.fields_by_name['update_rep'])
_TRANSPORTRESPONSE.fields_by_name['update_rep'].containing_oneof = _TRANSPORTRESPONSE.oneofs_by_name['response_oneof']
_SCALAR.oneofs_by_name['scalar'].fields.append(
  _SCALAR.fields_by_name['scdouble'])
_SCALAR.fields_by_name['scdouble'].containing_oneof = _SCALAR.oneofs_by_name['scalar']
_SCALAR.oneofs_by_name['scalar'].fields.append(
  _SCALAR.fields_by_name['scfloat'])
_SCALAR.fields_by_name['scfloat'].containing_oneof = _SCALAR.oneofs_by_name['scalar']
_SCALAR.oneofs_by_name['scalar'].fields.append(
  _SCALAR.fields_by_name['scint32'])
_SCALAR.fields_by_name['scint32'].containing_oneof = _SCALAR.oneofs_by_name['scalar']
_SCALAR.oneofs_by_name['scalar'].fields.append(
  _SCALAR.fields_by_name['scstring'])
_SCALAR.fields_by_name['scstring'].containing_oneof = _SCALAR.oneofs_by_name['scalar']
_SCALAR.oneofs_by_name['scalar'].fields.append(
  _SCALAR.fields_by_name['scbool'])
_SCALAR.fields_by_name['scbool'].containing_oneof = _SCALAR.oneofs_by_name['scalar']
DESCRIPTOR.message_types_by_name['ReadyReq'] = _READYREQ
DESCRIPTOR.message_types_by_name['UpdateReq'] = _UPDATEREQ
DESCRIPTOR.message_types_by_name['transportRequest'] = _TRANSPORTREQUEST
DESCRIPTOR.message_types_by_name['ReadyRep'] = _READYREP
DESCRIPTOR.message_types_by_name['UpdateRep'] = _UPDATEREP
DESCRIPTOR.message_types_by_name['transportResponse'] = _TRANSPORTRESPONSE
DESCRIPTOR.message_types_by_name['Scalar'] = _SCALAR
DESCRIPTOR.enum_types_by_name['State'] = _STATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ReadyReq = _reflection.GeneratedProtocolMessageType('ReadyReq', (_message.Message,), {

  'ConfigEntry' : _reflection.GeneratedProtocolMessageType('ConfigEntry', (_message.Message,), {
    'DESCRIPTOR' : _READYREQ_CONFIGENTRY,
    '__module__' : 'transport_pb2'
    # @@protoc_insertion_point(class_scope:transport.ReadyReq.ConfigEntry)
    })
  ,
  'DESCRIPTOR' : _READYREQ,
  '__module__' : 'transport_pb2'
  # @@protoc_insertion_point(class_scope:transport.ReadyReq)
  })
_sym_db.RegisterMessage(ReadyReq)
_sym_db.RegisterMessage(ReadyReq.ConfigEntry)

UpdateReq = _reflection.GeneratedProtocolMessageType('UpdateReq', (_message.Message,), {
  'DESCRIPTOR' : _UPDATEREQ,
  '__module__' : 'transport_pb2'
  # @@protoc_insertion_point(class_scope:transport.UpdateReq)
  })
_sym_db.RegisterMessage(UpdateReq)

transportRequest = _reflection.GeneratedProtocolMessageType('transportRequest', (_message.Message,), {
  'DESCRIPTOR' : _TRANSPORTREQUEST,
  '__module__' : 'transport_pb2'
  # @@protoc_insertion_point(class_scope:transport.transportRequest)
  })
_sym_db.RegisterMessage(transportRequest)

ReadyRep = _reflection.GeneratedProtocolMessageType('ReadyRep', (_message.Message,), {

  'ConfigEntry' : _reflection.GeneratedProtocolMessageType('ConfigEntry', (_message.Message,), {
    'DESCRIPTOR' : _READYREP_CONFIGENTRY,
    '__module__' : 'transport_pb2'
    # @@protoc_insertion_point(class_scope:transport.ReadyRep.ConfigEntry)
    })
  ,
  'DESCRIPTOR' : _READYREP,
  '__module__' : 'transport_pb2'
  # @@protoc_insertion_point(class_scope:transport.ReadyRep)
  })
_sym_db.RegisterMessage(ReadyRep)
_sym_db.RegisterMessage(ReadyRep.ConfigEntry)

UpdateRep = _reflection.GeneratedProtocolMessageType('UpdateRep', (_message.Message,), {
  'DESCRIPTOR' : _UPDATEREP,
  '__module__' : 'transport_pb2'
  # @@protoc_insertion_point(class_scope:transport.UpdateRep)
  })
_sym_db.RegisterMessage(UpdateRep)

transportResponse = _reflection.GeneratedProtocolMessageType('transportResponse', (_message.Message,), {
  'DESCRIPTOR' : _TRANSPORTRESPONSE,
  '__module__' : 'transport_pb2'
  # @@protoc_insertion_point(class_scope:transport.transportResponse)
  })
_sym_db.RegisterMessage(transportResponse)

Scalar = _reflection.GeneratedProtocolMessageType('Scalar', (_message.Message,), {
  'DESCRIPTOR' : _SCALAR,
  '__module__' : 'transport_pb2'
  # @@protoc_insertion_point(class_scope:transport.Scalar)
  })
_sym_db.RegisterMessage(Scalar)


_READYREQ_CONFIGENTRY._options = None
_READYREP_CONFIGENTRY._options = None

_TRANSPORTSERVICE = _descriptor.ServiceDescriptor(
  name='TransportService',
  full_name='transport.TransportService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1058,
  serialized_end=1154,
  methods=[
  _descriptor.MethodDescriptor(
    name='transport',
    full_name='transport.TransportService.transport',
    index=0,
    containing_service=None,
    input_type=_TRANSPORTREQUEST,
    output_type=_TRANSPORTRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_TRANSPORTSERVICE)

DESCRIPTOR.services_by_name['TransportService'] = _TRANSPORTSERVICE

# @@protoc_insertion_point(module_scope)
