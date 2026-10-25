#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Isolated extraction of fairchem UMA inference checkpoints (standard library only).

This file is never executed inside Phydrax. ``_uma_checkpoint`` stages its exact
bytes into a private directory and runs it with a caller-pinned interpreter
through ``run_pinned_command`` (``python -I -S``, empty environment)::

    worker.py request.json result.zip

No source framework is imported and nothing is unpickled. fairchem saves an
``MLIPInferenceCheckpoint`` dataclass with ``torch.save`` (a stored ZIP of
``data.pkl`` plus raw ``data/<key>`` storages). Its pickle opcodes are decoded
with ``pickletools`` and interpreted symbolically against an exact allowlist:
plain containers and scalars, the checkpoint record (``NEWOBJ`` + ``BUILD``),
``collections.OrderedDict``, ``torch._utils._rebuild_tensor_v2`` and
``_rebuild_parameter`` and typed torch storages. Any other global, opcode or
object refuses. The ZIP end records and every central-directory entry are
bounded before ``zipfile`` parses the directory; configuration metadata is
converted to plain data under explicit depth, node and byte budgets that
charge every expanded memo reference and refuse cycles. Tensors are read only
as contiguous byte ranges of their admitted storages, after the container,
every record size, every storage and every view has been bounded by the
request's explicit limits.

The result is the canonical Phydrax array archive: stored NPY members (EMA
parameters ``ema/<name>``, configuration tensors ``config/<n>``) and
``manifest.json`` with the plain configuration, the EMA averaging count and
the full name/dtype/shape inventory of the non-EMA training weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickletools
import struct
import sys
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


_REFUSED_EXIT = 3
_RESULT_FORMAT = "phydrax-uma-source-result"
_RECORD = ("fairchem.core.units.mlip_unit.api.inference", "MLIPInferenceCheckpoint")
_RECORD_FIELDS = frozenset(
    {"model_config", "model_state_dict", "ema_state_dict", "tasks_config"}
)
_ORDERED_DICT = ("collections", "OrderedDict")
_REBUILD_TENSOR = ("torch._utils", "_rebuild_tensor_v2")
_REBUILD_PARAMETER = ("torch._utils", "_rebuild_parameter")
_STORAGES = {
    "DoubleStorage": ("<f8", 8),
    "FloatStorage": ("<f4", 4),
    "LongStorage": ("<i8", 8),
    "IntStorage": ("<i4", 4),
    "ShortStorage": ("<i2", 2),
    "CharStorage": ("|i1", 1),
    "ByteStorage": ("|u1", 1),
    "BoolStorage": ("|b1", 1),
}
_GLOBALS = frozenset(
    {
        _RECORD,
        _ORDERED_DICT,
        _REBUILD_TENSOR,
        _REBUILD_PARAMETER,
        *(("torch", name) for name in _STORAGES),
    }
)
_TORCH_METADATA = frozenset(
    {
        "byteorder",
        "version",
        ".data/version",
        ".format_version",
        ".storage_alignment",
        ".data/serialization_id",
    }
)
_CHUNK = 16_777_216
# Python-recursion ceiling for the depth budget of configuration conversion.
_MAX_CONFIG_DEPTH = 512
_ZIP_END = struct.Struct("<4s4H2LH")
_ZIP64_LOCATOR = struct.Struct("<4sLQL")
_ZIP64_END = struct.Struct("<4sQ2H2L4Q")
_ZIP64_RECORD_SIZE = _ZIP64_END.size - 12
_ZIP_CENTRAL_SIZE = 46


class ProviderRefusal(ValueError):
    """A source outside the admitted extraction contract."""


def _refuse(message: str, /) -> ProviderRefusal:
    return ProviderRefusal(message)


@dataclass(frozen=True, slots=True)
class _Limits:
    max_source_bytes: int
    max_result_bytes: int
    max_tensor_bytes: int
    max_tensor_elements: int
    max_members: int
    max_manifest_bytes: int
    max_array_rank: int
    max_total_elements: int
    max_config_depth: int
    max_config_nodes: int


def _limits(request: Mapping[str, Any], /) -> _Limits:
    declared = request.get("limits")
    names = [item.name for item in fields(_Limits)]
    if (
        not isinstance(declared, Mapping)
        or set(declared) != set(names)
        or any(type(declared[name]) is not int or declared[name] <= 0 for name in names)
    ):
        raise _refuse("The request must declare explicit positive resource limits.")
    if declared["max_config_depth"] > _MAX_CONFIG_DEPTH:
        raise _refuse(f"max_config_depth must not exceed {_MAX_CONFIG_DEPTH}.")
    return _Limits(**{name: declared[name] for name in names})


@dataclass(frozen=True, slots=True)
class _Global:
    module: str
    name: str


@dataclass(frozen=True, slots=True)
class _Storage:
    key: str
    descr: str
    itemsize: int
    numel: int


@dataclass(frozen=True, slots=True)
class _Tensor:
    storage: _Storage
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


class _Ordered(dict):
    """A pickled ``collections.OrderedDict`` (plain mapping)."""


class _Record:
    """The pickled ``MLIPInferenceCheckpoint`` instance, as its field mapping."""

    def __init__(self) -> None:
        self.state: dict[str, Any] | None = None


def _global(module: Any, name: Any, /) -> _Global:
    if (module, name) not in _GLOBALS:
        raise _refuse(
            f"The checkpoint pickle names {module}.{name}, outside the admitted format."
        )
    return _Global(module, name)


def _extents(value: Any, /) -> bool:
    return isinstance(value, tuple) and all(
        type(item) is int and item >= 0 for item in value
    )


def _storage(identifier: Any, storages: dict[str, _Storage], /) -> _Storage:
    if not (
        isinstance(identifier, tuple)
        and len(identifier) == 5
        and identifier[0] == "storage"
        and isinstance(identifier[1], _Global)
        and identifier[1].module == "torch"
        and isinstance(identifier[2], str)
        and identifier[2]
        and "/" not in identifier[2]
        and isinstance(identifier[3], str)
        and type(identifier[4]) is int
        and identifier[4] >= 0
    ):
        raise _refuse("A storage declaration is malformed.")
    descr, itemsize = _STORAGES[identifier[1].name]
    storage = _Storage(identifier[2], descr, itemsize, identifier[4])
    if storages.setdefault(storage.key, storage) != storage:
        raise _refuse(f"Storage {storage.key!r} is declared inconsistently.")
    return storage


def _call(function: Any, arguments: Any, /) -> Any:
    if not isinstance(function, _Global) or not isinstance(arguments, tuple):
        raise _refuse("The checkpoint pickle reduces a non-global.")
    match (function.module, function.name):
        case ("collections", "OrderedDict") if not arguments:
            return _Ordered()
        case ("torch._utils", "_rebuild_tensor_v2") if len(arguments) in (6, 7):
            storage, offset, shape, stride, requires_grad, hooks = arguments[:6]
            if (
                not isinstance(storage, _Storage)
                or type(offset) is not int
                or offset < 0
                or not _extents(shape)
                or not _extents(stride)
                or len(shape) != len(stride)
                or type(requires_grad) is not bool
                or not isinstance(hooks, dict)
                or hooks
                or (len(arguments) == 7 and arguments[6] not in (None, {}))
            ):
                raise _refuse("A tensor declaration is malformed.")
            return _Tensor(storage, offset, shape, stride)
        case ("torch._utils", "_rebuild_parameter") if len(arguments) == 3:
            tensor, requires_grad, hooks = arguments
            if (
                not isinstance(tensor, _Tensor)
                or type(requires_grad) is not bool
                or (not isinstance(hooks, dict) or hooks)
            ):
                raise _refuse("A parameter declaration is malformed.")
            return tensor
    raise _refuse(f"The checkpoint pickle calls {function.module}.{function.name}.")


# Keep this symbolic opcode grammar together: the stack/memo/storage transitions
# and refusal of executable globals are one security boundary, not provider calls.
def _decode(payload: bytes, /) -> tuple[Any, dict[str, _Storage]]:
    """Interpret ``data.pkl`` symbolically; nothing is unpickled or called."""
    stack: list[Any] = []
    frames: list[list[Any]] = []
    memo: dict[int, Any] = {}
    storages: dict[str, _Storage] = {}
    try:
        for opcode, argument, _ in pickletools.genops(payload):
            name = opcode.name
            match name:
                case "PROTO" | "FRAME":
                    pass
                case "MARK":
                    frames.append(stack)
                    stack = []
                case "BINPUT" | "LONG_BINPUT":
                    if not isinstance(argument, int):
                        raise ValueError(name)
                    memo[argument] = stack[-1]
                case "MEMOIZE":
                    memo[len(memo)] = stack[-1]
                case "BINGET" | "LONG_BINGET":
                    if not isinstance(argument, int):
                        raise ValueError(name)
                    stack.append(memo[argument])
                case "GLOBAL":
                    module, attribute = str(argument).split(" ")
                    stack.append(_global(module, attribute))
                case "STACK_GLOBAL":
                    attribute = stack.pop()
                    stack.append(_global(stack.pop(), attribute))
                case (
                    "BINUNICODE"
                    | "SHORT_BINUNICODE"
                    | "BINUNICODE8"
                    | "BININT"
                    | "BININT1"
                    | "BININT2"
                    | "LONG1"
                    | "BINFLOAT"
                ):
                    stack.append(argument)
                case "NONE":
                    stack.append(None)
                case "NEWTRUE" | "NEWFALSE":
                    stack.append(name == "NEWTRUE")
                case "EMPTY_TUPLE":
                    stack.append(())
                case "TUPLE":
                    items = tuple(stack)
                    stack = frames.pop()
                    stack.append(items)
                case "TUPLE1" | "TUPLE2" | "TUPLE3":
                    count = int(name[-1])
                    if len(stack) < count:
                        raise IndexError(name)
                    items = tuple(stack[-count:])
                    del stack[-count:]
                    stack.append(items)
                case "EMPTY_LIST":
                    stack.append([])
                case "LIST":
                    items = list(stack)
                    stack = frames.pop()
                    stack.append(items)
                case "APPEND":
                    value = stack.pop()
                    if not isinstance(stack[-1], list):
                        raise ValueError(name)
                    stack[-1].append(value)
                case "APPENDS":
                    items = stack
                    stack = frames.pop()
                    if not isinstance(stack[-1], list):
                        raise ValueError(name)
                    stack[-1].extend(items)
                case "EMPTY_DICT":
                    stack.append({})
                case "DICT":
                    items = stack
                    stack = frames.pop()
                    if len(items) % 2:
                        raise ValueError(name)
                    stack.append(dict(zip(items[::2], items[1::2], strict=True)))
                case "SETITEM":
                    value = stack.pop()
                    key = stack.pop()
                    if not isinstance(stack[-1], dict):
                        raise ValueError(name)
                    stack[-1][key] = value
                case "SETITEMS":
                    items = stack
                    stack = frames.pop()
                    if len(items) % 2 or not isinstance(stack[-1], dict):
                        raise ValueError(name)
                    for key, value in zip(items[::2], items[1::2], strict=True):
                        stack[-1][key] = value
                case "BINPERSID":
                    stack.append(_storage(stack.pop(), storages))
                case "REDUCE":
                    arguments = stack.pop()
                    stack.append(_call(stack.pop(), arguments))
                case "NEWOBJ":
                    arguments = stack.pop()
                    cls = stack.pop()
                    if not (
                        isinstance(cls, _Global)
                        and (cls.module, cls.name) == _RECORD
                        and arguments == ()
                    ):
                        raise _refuse(
                            "Only the inference checkpoint record is constructed."
                        )
                    stack.append(_Record())
                case "BUILD":
                    state = stack.pop()
                    target = stack[-1]
                    if isinstance(target, _Ordered):
                        if not isinstance(state, dict) or set(state) - {"_metadata"}:
                            raise ValueError(name)
                    elif isinstance(target, _Record) and target.state is None:
                        if not isinstance(state, dict) or set(state) != _RECORD_FIELDS:
                            raise _refuse("The checkpoint record has unexpected fields.")
                        target.state = state
                    else:
                        raise ValueError(name)
                case "STOP":
                    result = stack.pop()
                    if stack or frames:
                        raise ValueError(name)
                    return result, storages
                case other:
                    raise _refuse(
                        f"Pickle opcode {other} is outside the admitted format."
                    )
    except ProviderRefusal:
        raise
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise _refuse("The checkpoint pickle is malformed.") from error
    raise _refuse("The checkpoint pickle has no STOP.")


def _directory(data: bytes, limits: _Limits, /) -> int:
    """Bound the end records and walk the central directory; returns its entry count.

    Runs before ``zipfile`` reads anything: the directory must sit exactly
    before its (ZIP64) end record, fit ``max_manifest_bytes`` and hold at most
    ``max_members`` well-formed entries whose lengths stay inside it.
    """
    end = len(data) - _ZIP_END.size
    if (
        not data.startswith(b"PK\x03\x04")
        or end < 0
        or data[end : end + 4] != b"PK\x05\x06"
    ):
        raise _refuse("The source is not the ZIP container torch.save writes.")
    _, disk, directory_disk, disk_entries, entries, size, offset, comment = (
        _ZIP_END.unpack_from(data, end)
    )
    if disk or directory_disk or disk_entries != entries or comment:
        raise _refuse("The source ZIP end record is not canonical.")
    directory_end = end
    locator = end - _ZIP64_LOCATOR.size
    if locator >= _ZIP64_END.size and data[locator : locator + 4] == b"PK\x06\x07":
        _, locator_disk, record, disks = _ZIP64_LOCATOR.unpack_from(data, locator)
        if locator_disk or disks != 1 or record != locator - _ZIP64_END.size:
            raise _refuse("The source ZIP64 locator is not canonical.")
        (
            signature,
            record_size,
            _,
            _,
            disk64,
            directory64,
            disk_entries64,
            entries64,
            size64,
            offset64,
        ) = _ZIP64_END.unpack_from(data, record)
        if (
            signature != b"PK\x06\x06"
            or record_size != _ZIP64_RECORD_SIZE
            or disk64
            or directory64
            or disk_entries64 != entries64
            or entries not in (entries64, 0xFFFF)
            or size not in (size64, 0xFFFFFFFF)
            or offset not in (offset64, 0xFFFFFFFF)
        ):
            raise _refuse("The source ZIP64 end record is inconsistent.")
        entries, size, offset, directory_end = entries64, size64, offset64, record
    elif entries == 0xFFFF or 0xFFFFFFFF in (size, offset):
        raise _refuse("The source ZIP64 end record is missing.")
    if entries > limits.max_members or size > limits.max_manifest_bytes:
        raise _refuse("The source ZIP directory exceeds its member or byte limits.")
    if offset + size != directory_end:
        raise _refuse("The source ZIP directory is not where its end record says.")
    position, count = offset, 0
    while position < directory_end:
        if (
            count == entries
            or position + _ZIP_CENTRAL_SIZE > directory_end
            or data[position : position + 4] != b"PK\x01\x02"
        ):
            raise _refuse("The source ZIP directory entries are malformed.")
        name, extra, remark = struct.unpack_from("<3H", data, position + 28)
        position += _ZIP_CENTRAL_SIZE + name + extra + remark
        count += 1
    if position != directory_end or count != entries:
        raise _refuse("The source ZIP directory entries are malformed.")
    return entries


def _container(data: bytes, limits: _Limits, /) -> tuple[zipfile.ZipFile, str]:
    """Admit the stored torch.save ZIP container and its record sizes."""
    import io

    entries = _directory(data, limits)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise _refuse("The source ZIP directory is malformed.") from error
    try:
        return _records(archive, entries, limits)
    except BaseException:
        archive.close()
        raise


def _records(
    archive: zipfile.ZipFile, entries: int, limits: _Limits, /
) -> tuple[zipfile.ZipFile, str]:
    members = archive.infolist()
    if len(members) != entries:
        raise _refuse("The source ZIP directory differs from its preflight.")
    names = [member.filename for member in members]
    prefix = names[0].split("/", 1)[0] + "/" if names and "/" in names[0] else ""
    if (
        not prefix
        or len(set(names)) != len(names)
        or any(not name.startswith(prefix) for name in names)
        or prefix + "data.pkl" not in names
    ):
        raise _refuse("The source records are not one torch.save archive.")
    expanded = 0
    for member in members:
        record = member.filename.removeprefix(prefix)
        if (
            member.compress_type != zipfile.ZIP_STORED
            or member.compress_size != member.file_size
            or member.flag_bits & 0x1
        ):
            raise _refuse("Source records must be stored and unencrypted.")
        key = record.removeprefix("data/")
        if key != record and key and "/" not in key:
            maximum = limits.max_tensor_bytes
        elif record == "data.pkl" or record in _TORCH_METADATA:
            maximum = limits.max_manifest_bytes
        else:
            raise _refuse(f"Source record {record!r} is outside the torch.save layout.")
        if member.file_size > maximum:
            raise _refuse(f"Source record {record!r} exceeds its byte limit.")
        expanded += member.file_size
    if expanded > limits.max_source_bytes:
        raise _refuse("The source records exceed the source byte limit.")
    if prefix + "byteorder" in names and archive.read(prefix + "byteorder") != b"little":
        raise _refuse("Only little-endian torch storages are admitted.")
    return archive, prefix


def _contiguous(tensor: _Tensor, /) -> bool:
    expected = 1
    for extent, step in zip(reversed(tensor.shape), reversed(tensor.stride)):
        if extent > 1 and step != expected:
            return False
        expected *= extent
    return True


def _admit_tensor(name: str, tensor: _Tensor, limits: _Limits, /) -> int:
    elements = math.prod(tensor.shape)
    if (
        len(tensor.shape) > limits.max_array_rank
        or any(extent > limits.max_tensor_elements for extent in tensor.shape)
        or elements > limits.max_tensor_elements
        or elements * tensor.storage.itemsize > limits.max_tensor_bytes
    ):
        raise _refuse(f"Tensor {name} exceeds the tensor limits.")
    if not _contiguous(tensor):
        raise _refuse(f"Tensor {name} is a non-contiguous view; refused.")
    if tensor.offset + elements > tensor.storage.numel:
        raise _refuse(f"Tensor {name} views beyond its storage.")
    return elements


@dataclass(slots=True)
class _Plain:
    """Bounded, cycle-aware conversion of decoded metadata into JSON values.

    Memo references keep the decoded graph small, but its plain form repeats
    every shared node. Each expanded node is charged against ``max_config_nodes``
    and its JSON encoding against the ``max_manifest_bytes`` budget *before* any
    output is built; nesting beyond ``max_config_depth`` and containers that
    contain themselves refuse. A shared tensor object gets one array reference.
    """

    nodes: int
    size: int
    depth: int
    arrays: dict[str, _Tensor] = field(default_factory=dict)
    references: dict[int, str] = field(default_factory=dict)
    active: set[int] = field(default_factory=set)

    @classmethod
    def of(cls, limits: _Limits, /) -> _Plain:
        return cls(
            limits.max_config_nodes, limits.max_manifest_bytes, limits.max_config_depth
        )

    def _charge(self, path: str, size: int, nodes: int = 1, /) -> None:
        self.nodes -= nodes
        self.size -= size
        if self.nodes < 0:
            raise _refuse(
                f"Configuration metadata expands beyond max_config_nodes at {path}."
            )
        if self.size < 0:
            raise _refuse(
                f"Configuration metadata expands beyond max_manifest_bytes at {path}."
            )

    def convert(self, value: Any, path: str, depth: int = 0, /) -> Any:
        """JSON value of configuration data; tensors become array references."""
        if depth > self.depth:
            raise _refuse(
                f"Configuration value {path} is nested beyond max_config_depth."
            )
        if value is None or isinstance(value, bool):
            self._charge(path, 5)
            return value
        if isinstance(value, int):
            self._charge(path, len(str(value)))
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise _refuse(f"Configuration value {path} is not finite.")
            self._charge(path, len(repr(value)))
            return value
        if isinstance(value, str):
            self._charge(path, len(json.dumps(value)))
            return value
        if isinstance(value, _Tensor):
            name = self.references.get(id(value), f"config/{len(self.arrays)}")
            reference = {"__array__": name}
            self._charge(path, len(json.dumps(reference)))
            if name not in self.arrays:
                self.arrays[name] = value
                self.references[id(value)] = name
            return reference
        if not isinstance(value, (list, tuple, dict)):
            raise _refuse(
                f"Configuration value {path} is not plain data ({type(value).__name__})."
            )
        identity = id(value)
        if identity in self.active:
            raise _refuse(
                f"Configuration value {path} contains itself (cyclic metadata)."
            )
        self._charge(path, 2 + 2 * max(len(value) - 1, 0))
        self.active.add(identity)
        try:
            if isinstance(value, dict):
                mapping: dict[str, Any] = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise _refuse(
                            f"Configuration mapping {path} has non-string keys."
                        )
                    self._charge(path, len(json.dumps(key)) + 2, 0)
                    mapping[key] = self.convert(item, f"{path}.{key}", depth + 1)
                return mapping
            items: list[Any] = []
            for index, item in enumerate(value):
                items.append(self.convert(item, f"{path}[{index}]", depth + 1))
            return items
        finally:
            self.active.discard(identity)


def _state(value: Any, name: str, /) -> dict[str, _Tensor]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, _Tensor)
        for key, item in value.items()
    ):
        raise _refuse(f"{name} must map names to tensors.")
    return dict(value)


def _npy_header(descr: str, shape: tuple[int, ...], /) -> bytes:
    shape_text = "(" + "".join(f"{extent}, " for extent in shape).rstrip(" ")
    if len(shape) == 1:
        shape_text = f"({shape[0]},"
    shape_text += ")"
    text = (
        "{'descr': '"
        + descr
        + "', 'fortran_order': False, 'shape': "
        + shape_text
        + ", }"
    )
    padding = 64 - (10 + len(text) + 1) % 64
    text = text + " " * (padding % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(text)) + text.encode("latin1")


def _stored(name: str, size: int, /) -> zipfile.ZipInfo:
    information = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    information.compress_type = zipfile.ZIP_STORED
    information.create_system = 3
    information.external_attr = 0o600 << 16
    information.file_size = size
    return information


def _chunks(data: bytes, start: int, size: int, /) -> Iterator[bytes]:
    for offset in range(start, start + size, _CHUNK):
        yield data[offset : min(offset + _CHUNK, start + size)]


def _write(
    path: Path,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, _Tensor],
    archive: zipfile.ZipFile,
    prefix: str,
    limits: _Limits,
    /,
) -> None:
    storages: dict[str, bytes] = {}
    planned = []
    total_elements = 0
    total_bytes = 0
    for index, name in enumerate(sorted(arrays)):
        tensor = arrays[name]
        elements = _admit_tensor(name, tensor, limits)
        header = _npy_header(tensor.storage.descr, tensor.shape)
        size = len(header) + elements * tensor.storage.itemsize
        total_elements += elements
        total_bytes += size
        planned.append((name, f"arrays/{index:06d}.npy", header, tensor, elements, size))
    if (
        total_elements > limits.max_total_elements
        or total_bytes > limits.max_result_bytes
    ):
        raise _refuse("The extraction result exceeds the result limits.")
    if len(planned) + 1 > limits.max_members:
        raise _refuse("The extraction result has too many members.")
    inventory: dict[str, Any] = {}
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as result:
        for name, member, header, tensor, elements, size in planned:
            key = tensor.storage.key
            if key not in storages:
                data = archive.read(prefix + "data/" + key)
                if len(data) != tensor.storage.numel * tensor.storage.itemsize:
                    raise _refuse(f"Storage {key!r} holds a different byte count.")
                storages[key] = data
            data = storages[key]
            digest = hashlib.sha256()
            with result.open(_stored(member, size), mode="w") as stream:
                digest.update(header)
                stream.write(header)
                for chunk in _chunks(
                    data,
                    tensor.offset * tensor.storage.itemsize,
                    elements * tensor.storage.itemsize,
                ):
                    digest.update(chunk)
                    stream.write(chunk)
            inventory[name] = {
                "member": member,
                "shape": list(tensor.shape),
                "dtype": tensor.storage.descr,
                "sha256": digest.hexdigest(),
                "order": "C",
            }
        payload = json.dumps(
            {**manifest, "arrays": inventory}, allow_nan=False, indent=2, sort_keys=True
        ).encode("utf-8")
        if len(payload) > limits.max_manifest_bytes:
            raise _refuse("The extraction manifest exceeds the manifest byte limit.")
        result.writestr(_stored("manifest.json", len(payload)), payload)


def _execute(request_path: Path, output: Path, /) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, Mapping) or request.get("operation") != "extract":
        raise _refuse("Only the extract operation exists.")
    limits = _limits(request)
    source = request.get("source")
    if not isinstance(source, Mapping) or set(source) != {"sha256", "byte_size"}:
        raise _refuse("The request must describe its source digest and size.")
    data = (request_path.parent / "source.bin").read_bytes()
    if (
        len(data) != source["byte_size"]
        or hashlib.sha256(data).hexdigest() != source["sha256"]
    ):
        raise _refuse("The staged source differs from the admitted digest.")
    if len(data) > limits.max_source_bytes:
        raise _refuse("The source exceeds the source byte limit.")
    archive, prefix = _container(data, limits)
    with archive:
        root, storages = _decode(archive.read(prefix + "data.pkl"))
        if not isinstance(root, _Record) or root.state is None:
            raise _refuse("The source is not an MLIPInferenceCheckpoint record.")
        records = {
            member.filename.removeprefix(prefix + "data/")
            for member in archive.infolist()
            if member.filename.startswith(prefix + "data/")
        }
        if records != set(storages):
            raise _refuse("Storage records differ from the declared storages.")
        state = root.state
        ema = _state(state["ema_state_dict"], "ema_state_dict")
        model = _state(state["model_state_dict"], "model_state_dict")
        plain = _Plain.of(limits)
        configuration = {
            "model_config": plain.convert(state["model_config"], "model_config"),
            "tasks_config": plain.convert(state["tasks_config"], "tasks_config"),
        }
        arrays: dict[str, _Tensor] = plain.arrays
        averaged = ema.pop("n_averaged", None)
        if averaged is None or averaged.shape != () or averaged.storage.descr != "<i8":
            raise _refuse("The EMA state must carry its scalar int64 n_averaged count.")
        count_bytes = archive.read(prefix + "data/" + averaged.storage.key)
        n_averaged = struct.unpack_from("<q", count_bytes, averaged.offset * 8)[0]
        for name, tensor in ema.items():
            arrays[f"ema/{name}"] = tensor
        for name, tensor in model.items():
            _admit_tensor(name, tensor, limits)
        manifest = {
            "format": _RESULT_FORMAT,
            "operation": "extract",
            "source": {"sha256": source["sha256"], "byte_size": source["byte_size"]},
            "checkpoint": {
                **configuration,
                "ema_n_averaged": int(n_averaged),
                "model_state_inventory": [
                    [name, tensor.storage.descr, list(tensor.shape)]
                    for name, tensor in sorted(model.items())
                ],
            },
        }
        _write(output, manifest, arrays, archive, prefix, limits)


def main(arguments: Sequence[str], /) -> int:
    if len(arguments) != 3:
        sys.stderr.write("usage: worker.py request.json result.zip\n")
        return 2
    try:
        _execute(Path(arguments[1]), Path(arguments[2]))
    except ProviderRefusal as refusal:
        sys.stderr.write(f"REFUSED: {refusal}\n")
        return _REFUSED_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
