#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import os
import socket
import struct
import time
from pathlib import Path
from typing import BinaryIO


_CRC32C_POLYNOMIAL = 0x82F63B78
_MASK_DELTA = 0xA282EAD8


def _varint(value: int, /) -> bytes:
    value_ = int(value)
    if value_ < 0:
        raise ValueError("protobuf varints must be nonnegative.")
    output = bytearray()
    while value_ >= 0x80:
        output.append((value_ & 0x7F) | 0x80)
        value_ >>= 7
    output.append(value_)
    return bytes(output)


def _key(field_number: int, wire_type: int, /) -> bytes:
    return _varint((int(field_number) << 3) | int(wire_type))


def _length_delimited(field_number: int, payload: bytes, /) -> bytes:
    return _key(field_number, 2) + _varint(len(payload)) + payload


def _event_file_version(wall_time: float, /) -> bytes:
    return (
        _key(1, 1)
        + struct.pack("<d", float(wall_time))
        + _length_delimited(3, b"brain.Event:2")
    )


def _scalar_event(tag: str, value: float, step: int, wall_time: float, /) -> bytes:
    encoded_tag = tag.encode("utf-8")
    summary_value = (
        _length_delimited(1, encoded_tag) + _key(2, 5) + struct.pack("<f", float(value))
    )
    summary = _length_delimited(1, summary_value)
    return (
        _key(1, 1)
        + struct.pack("<d", float(wall_time))
        + _key(2, 0)
        + _varint(step)
        + _length_delimited(5, summary)
    )


def _crc32c(payload: bytes, /) -> int:
    crc = 0xFFFFFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (_CRC32C_POLYNOMIAL if crc & 1 else 0)
    return (~crc) & 0xFFFFFFFF


def _masked_crc32c(payload: bytes, /) -> int:
    crc = _crc32c(payload)
    return (((crc >> 15) | (crc << 17)) + _MASK_DELTA) & 0xFFFFFFFF


def _write_record(stream: BinaryIO, payload: bytes, /) -> None:
    length = struct.pack("<Q", len(payload))
    stream.write(length)
    stream.write(struct.pack("<I", _masked_crc32c(length)))
    stream.write(payload)
    stream.write(struct.pack("<I", _masked_crc32c(payload)))


class ScalarEventWriter:
    """Append TensorBoard-compatible scalar events without TensorBoard runtime code."""

    def __init__(self, log_dir: str | Path, /):
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        identity = f"{socket.gethostname()}.{os.getpid()}.{time.time_ns()}"
        self.path = directory / f"events.out.tfevents.{timestamp}.{identity}"
        self._stream = self.path.open("xb")
        self._closed = False
        _write_record(self._stream, _event_file_version(time.time()))

    def scalar(self, tag: str, value: float, step: int, /) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed scalar event writer.")
        if not isinstance(tag, str) or not tag or "\x00" in tag:
            raise ValueError("TensorBoard scalar tags must be nonempty text without NUL.")
        step_ = int(step)
        if step_ < 0:
            raise ValueError("TensorBoard scalar steps must be nonnegative.")
        value_ = float(value)
        if not (-float("inf") < value_ < float("inf")):
            raise ValueError("TensorBoard scalar values must be finite.")
        _write_record(self._stream, _scalar_event(tag, value_, step_, time.time()))

    def flush(self) -> None:
        if not self._closed:
            self._stream.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._stream.flush()
        self._stream.close()
        self._closed = True


__all__ = ["ScalarEventWriter"]
