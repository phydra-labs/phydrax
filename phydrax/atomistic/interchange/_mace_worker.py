#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-side MACE extraction program executed by a pinned interpreter.

This file is never executed inside Phydrax. ``_mace_checkpoint`` stages its exact
bytes into a private directory and runs it with the caller's pinned provider
interpreter through ``run_pinned_command``. A fixed isolated bootstrap
(``python -I -S -c``) adds only the declared provider ``site-packages`` and then
runs this file as ``__main__`` with the arguments::

    worker.py <site-packages> request.json result.zip

torch, e3nn, mace-torch and ASE come from that ``site-packages`` after their
installed releases and the mace-torch/e3nn RECORD file digests are verified. A
subprocess is not a security sandbox: full-object torch deserialization runs
only when the host has admitted the exact source digest under an explicit
trusted-source capability, and the source bytes are re-hashed here before any
deserialization.

Nor is it a memory sandbox: the host bounds its wall time, log and output file
bytes, not its allocations. The request therefore carries the host's explicit
``MACEConversionLimits`` and they are enforced here before allocation. Source
bytes must be the stored ZIP container ``torch.save`` writes (other torch
serializations allocate pickled storage sizes before reading them and have no
bounded admission path); its directory, record sizes and, for a safe state
dict, the pickle metadata of every storage and tensor view are admitted with
``pickletools`` (no object is unpickled) before ``torch.load``. Every
size-determining dimension of a safe source's declared architecture is then
pinned to those admitted tensors, and every provider model construction
(source, rebuild or fixture) first bounds the declared harmonic degree and
irreps entry counts and is then planned from irreps metadata: each
parameter, buffer, coupling-basis recursion depth, traced example input and
einsum intermediate it allocates is charged against the limits before it
runs. Every neighbor list the provider discovers is first counted here in
bounded pair blocks and its topology (with an evaluation's whole result)
admitted before discovery. A trusted full-object pickle executes, and
allocates, outside these bounds once its container is admitted.

The result is the canonical Phydrax array archive (stored ZIP, pickle-free NPY
members plus ``manifest.json``) so the host admits it through
``read_array_archive`` with explicit byte, member and shape bounds. Every array
is admitted against the same bounds from its metadata before it is converted,
and members are streamed into the archive one at a time.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import math
import os
import pickletools
import struct
import sys
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


_REFUSED_EXIT = 3
_FIBONACCI_DIRECTIONS = 64
_REPRODUCTION_ULPS = 64

# Exact provider classes admitted for standard MACE conversion. Generated
# e3nn/opt_einsum code modules are admitted only under their generated names.
_ADMITTED_MODEL_CLASSES = {
    ("mace.modules.models", "MACE"): "MACE",
    ("mace.modules.models", "ScaleShiftMACE"): "ScaleShiftMACE",
}
_ADMITTED_INTERACTIONS = (
    "RealAgnosticInteractionBlock",
    "RealAgnosticResidualInteractionBlock",
    "RealAgnosticDensityInteractionBlock",
    "RealAgnosticDensityResidualInteractionBlock",
)
_ADMITTED_MODULE_TYPES = frozenset(
    {
        ("mace.modules.models", "MACE"),
        ("mace.modules.models", "ScaleShiftMACE"),
        ("mace.modules.blocks", "LinearNodeEmbeddingBlock"),
        ("mace.modules.blocks", "RadialEmbeddingBlock"),
        ("mace.modules.blocks", "AtomicEnergiesBlock"),
        ("mace.modules.blocks", "EquivariantProductBasisBlock"),
        ("mace.modules.blocks", "LinearReadoutBlock"),
        ("mace.modules.blocks", "NonLinearReadoutBlock"),
        ("mace.modules.blocks", "ScaleShiftBlock"),
        *(("mace.modules.blocks", name) for name in _ADMITTED_INTERACTIONS),
        ("mace.modules.irreps_tools", "reshape_irreps"),
        ("mace.modules.radial", "BesselBasis"),
        ("mace.modules.radial", "PolynomialCutoff"),
        ("mace.modules.radial", "AgnesiTransform"),
        ("mace.modules.radial", "ZBLBasis"),
        ("mace.modules.symmetric_contraction", "SymmetricContraction"),
        ("mace.modules.symmetric_contraction", "Contraction"),
        ("e3nn.math._normalize_activation", "normalize2mom"),
        ("e3nn.nn._activation", "Activation"),
        ("e3nn.nn._fc", "FullyConnectedNet"),
        ("e3nn.nn._fc", "_Layer"),
        ("e3nn.o3._linear", "Linear"),
        ("e3nn.o3._spherical_harmonics", "SphericalHarmonics"),
        ("e3nn.o3._tensor_product._sub", "FullyConnectedTensorProduct"),
        ("e3nn.o3._tensor_product._tensor_product", "TensorProduct"),
        ("torch.nn.modules.container", "ModuleList"),
        ("torch.nn.modules.container", "ParameterList"),
    }
)
_GENERATED_MODULE_TYPES = frozenset(
    {
        ("torch.fx.graph_module", "GraphModule.__new__.<locals>.GraphModuleImpl"),
        ("torch.jit._script", "RecursiveScriptModule"),
    }
)
_GENERATED_MODULE_NAMES = frozenset(
    {
        "_compiled_main",
        "_compiled_main_left_right",
        "_compiled_main_right",
        "graph_opt_main",
        "contractions_weighting",
        "contractions_features",
    }
)
_VERIFIED_DISTRIBUTIONS = ("mace-torch", "e3nn")
# cuequivariance computes reduced generalized-CG U bases; mace-torch builds
# them only when cuequivariance-torch is importable as well.
_REDUCED_CG_DISTRIBUTIONS = ("cuequivariance", "cuequivariance-torch")
# Distributions whose installed files define provider numerics; every file of
# each is verified against its RECORD digest.
_GENERATING_DISTRIBUTIONS = (*_VERIFIED_DISTRIBUTIONS, *_REDUCED_CG_DISTRIBUTIONS)
_ZBL_TYPE = ("mace.modules.radial", "ZBLBasis")
_CUTOFF_TYPE = ("mace.modules.radial", "PolynomialCutoff")
_CONTRACTION_TYPE = ("mace.modules.symmetric_contraction", "Contraction")
# Releases before multihead readouts store no ``heads``; the admitted
# provider's own calculator then evaluates their single head under this name.
_LEGACY_HEAD = "Default"
# State of earlier ZBLBasis releases (a fixed-radius ``PolynomialCutoff`` child
# and an ``r_max`` buffer) that the admitted ZBLBasis.forward never reads: it
# envelopes each pair at the sum of its covalent radii with ``p``.
_LEGACY_ZBL_STATE = ("r_max", "cutoff.r_max", "cutoff.p")
# Directory records of the ZIP container torch.save writes (it always adds the
# ZIP64 end record and its locator).
_ZIP_END = struct.Struct("<4s4H2LH")
_ZIP64_LOCATOR = struct.Struct("<4sLQL")
_ZIP64_END = struct.Struct("<4sQ2H2L4Q")
# Records torch.save writes besides ``data.pkl`` and the ``data/<key>`` storages.
_TORCH_METADATA_RECORDS = frozenset(
    {
        "byteorder",
        "version",
        ".data/version",
        ".format_version",
        ".storage_alignment",
        ".data/serialization_id",
    }
)
# Typed storages of a safe state dict: element dtype and byte size. Reduced
# precision and complex storages are refused like reduced-precision results.
_STATE_STORAGES = {
    "DoubleStorage": ("float64", 8),
    "FloatStorage": ("float32", 4),
    "LongStorage": ("int64", 8),
    "IntStorage": ("int32", 4),
    "ShortStorage": ("int16", 2),
    "CharStorage": ("int8", 1),
    "ByteStorage": ("uint8", 1),
    "BoolStorage": ("bool", 1),
}
_ORDERED_DICT = ("collections", "OrderedDict")
_REBUILD_TENSOR = ("torch._utils", "_rebuild_tensor_v2")
_STATE_GLOBALS = frozenset(
    {_ORDERED_DICT, _REBUILD_TENSOR, *(("torch", name) for name in _STATE_STORAGES)}
)
_NPY_CHUNK_BYTES = 16_777_216
# Neighbor-count preflight: pair distances held per block, rows per block,
# and the relative slack of its cutoff test over provider rounding.
_PAIR_BLOCK = 1_048_576
_PAIR_ROWS = 256
_PAIR_SLACK = 1.0e-9


class ProviderRefusal(ValueError):
    """A source or request outside the admitted provider contract."""


def _refuse(message: str, /) -> ProviderRefusal:
    return ProviderRefusal(message)


@dataclass(frozen=True, slots=True)
class _Limits:
    """The host's ``MACEConversionLimits`` as enforced inside the provider.

    Source: at most ``max_members`` records expanding to ``max_source_bytes``;
    each storage record and each tensor's logical view within
    ``max_tensor_bytes`` and ``max_tensor_elements``; ``data.pkl``, metadata
    records and the ZIP directory within ``max_manifest_bytes``. Result: the
    same per-array bounds, ``max_array_rank``, ``max_total_elements``, at most
    ``max_members`` members and ``max_result_bytes`` of archive.
    """

    max_source_bytes: int
    max_result_bytes: int
    max_tensor_bytes: int
    max_tensor_elements: int
    max_members: int
    max_manifest_bytes: int
    max_array_rank: int
    max_total_elements: int


def _limits(request: Mapping[str, Any], /) -> _Limits:
    declared = request.get("limits")
    names = [field.name for field in dataclasses.fields(_Limits)]
    if (
        not isinstance(declared, Mapping)
        or set(declared) != set(names)
        or any(type(declared[name]) is not int or declared[name] <= 0 for name in names)
    ):
        raise _refuse("Provider requests must declare explicit positive resource limits.")
    return _Limits(**{name: declared[name] for name in names})


def _normalized_name(name: str, /) -> str:
    return name.lower().replace("_", "-")


def _distribution(name: str, site_packages: str, /) -> metadata.Distribution:
    matches = [
        distribution
        for distribution in metadata.distributions(path=[site_packages])
        if _normalized_name(distribution.metadata["Name"]) == _normalized_name(name)
    ]
    if len(matches) != 1:
        raise _refuse(f"Provider distribution {name!r} is not uniquely installed.")
    return matches[0]


def _record_digest(distribution: metadata.Distribution, /) -> str:
    record = distribution.read_text("RECORD")
    if record is None:
        raise _refuse("Provider distribution has no RECORD manifest.")
    return hashlib.sha256(record.encode("utf-8")).hexdigest()


def _verify_record_files(distribution: metadata.Distribution, /) -> int:
    files = distribution.files
    if files is None:
        raise _refuse("Provider distribution has no file inventory.")
    verified = 0
    for package_path in files:
        recorded = package_path.hash
        if recorded is None:
            continue
        if recorded.mode != "sha256":
            raise _refuse("Provider RECORD uses an unsupported digest.")
        location = Path(str(distribution.locate_file(package_path)))
        digest = hashlib.sha256(location.read_bytes()).digest()
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        if encoded != recorded.value:
            raise _refuse(f"Provider file {package_path} differs from its RECORD.")
        verified += 1
    return verified


def _provider_identity(
    site_packages: str, required: Mapping[str, str], /
) -> dict[str, Any]:
    distributions: dict[str, Any] = {}
    for name, version in sorted(required.items()):
        distribution = _distribution(name, site_packages)
        installed = distribution.version
        if installed != version:
            raise _refuse(
                f"Provider {name} {installed} differs from the declared {version}."
            )
        verified_files = (
            _verify_record_files(distribution)
            if _normalized_name(name) in _GENERATING_DISTRIBUTIONS
            else 0
        )
        distributions[name] = {
            "version": installed,
            "record_sha256": _record_digest(distribution),
            "verified_files": verified_files,
        }
    for name in _VERIFIED_DISTRIBUTIONS:
        if not any(_normalized_name(key) == name for key in required):
            raise _refuse(f"The provider request must declare the {name} release.")
    # cuequivariance changes which U basis mace-torch constructs; an installed
    # but undeclared release would make rebuilt bases depend on the host.
    installed_names = {
        _normalized_name(item.metadata["Name"])
        for item in metadata.distributions(path=[site_packages])
    }
    declared_names = {_normalized_name(key) for key in required}
    for name in _REDUCED_CG_DISTRIBUTIONS:
        if name in installed_names and name not in declared_names:
            raise _refuse(f"The provider has {name} installed; declare its release.")
    return {
        "distributions": distributions,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "site_packages": site_packages,
    }


def _array(value: Any, /) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        if value.dtype in (torch.bfloat16, torch.float16, torch.complex32):
            raise _refuse("Provider tensors with reduced-precision dtypes are refused.")
        # np.ascontiguousarray promotes 0-d arrays to 1-d; keep exact shapes.
        return np.array(value.detach().cpu().numpy(), order="C", copy=True)
    return np.array(value, order="C", copy=True)


def _result_layout(name: str, value: Any, /) -> tuple[np.dtype[Any], tuple[int, ...]]:
    """Dtype and shape of one result value from its metadata alone."""

    import torch

    if isinstance(value, torch.Tensor):
        if value.dtype in (torch.bfloat16, torch.float16, torch.complex32):
            raise _refuse("Provider tensors with reduced-precision dtypes are refused.")
        dtype = torch.empty(0, dtype=value.dtype).numpy().dtype
        shape = tuple(int(extent) for extent in value.shape)
    elif isinstance(value, np.ndarray):
        dtype, shape = value.dtype, value.shape
    else:
        raise _refuse(f"Provider array {name!r} is not an array.")
    if dtype.hasobject or dtype.kind not in "biufc":
        raise _refuse(f"Provider array {name!r} has an unsupported dtype.")
    return dtype, shape


def _result_view(value: Any, /) -> np.ndarray:
    """A NumPy view of an admitted result value; no data is copied."""

    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().resolve_conj().resolve_neg().numpy()
    return value


def _npy_header(dtype: np.dtype[Any], shape: tuple[int, ...], /) -> bytes:
    """The exact version-1.0 C-order header ``np.save`` writes."""

    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return header.getvalue()


def _npy_chunks(header: bytes, array: np.ndarray, /) -> Iterator[bytes]:
    """``np.save``'s C-order payload in bounded chunks, never one full copy.

    C order is the concatenation of consecutive leading-axis slices, so each
    chunk copies only its own slices (at most one chunk, or one slice when a
    slice alone is larger; either is within the admitted tensor byte limit).
    """

    yield header
    if array.ndim == 0:
        yield array.tobytes("C")
        return
    slice_bytes = array.itemsize * math.prod(array.shape[1:])
    step = max(_NPY_CHUNK_BYTES // max(slice_bytes, 1), 1)
    for start in range(0, array.shape[0], step):
        yield array[start : start + step].tobytes("C")


def _stored_information(name: str, size: int, /) -> zipfile.ZipInfo:
    information = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    information.compress_type = zipfile.ZIP_STORED
    information.create_system = 3
    information.external_attr = 0o600 << 16
    information.file_size = size
    return information


def _stored_member_bytes(name: str, size: int, /) -> int:
    # Local header, payload and central-directory record of one stored member.
    encoded = len(name.encode("utf-8"))
    return 30 + encoded + size + 46 + encoded


def _manifest_payload(
    manifest: Mapping[str, Any], inventory: Mapping[str, Any], /
) -> bytes:
    complete = {**manifest, "arrays": inventory}
    return json.dumps(complete, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")


def _admit_result_plan(members: int, payload_bytes: int, limits: _Limits, /) -> None:
    """Refuse a result whose member count or payload is known to exceed the limits."""

    if members > limits.max_members:
        raise _refuse(
            f"The provider result needs {members} members, above the member limit "
            f"{limits.max_members}."
        )
    if payload_bytes > limits.max_result_bytes:
        raise _refuse(
            f"The provider result needs {payload_bytes} bytes, above the result byte "
            f"limit {limits.max_result_bytes}."
        )


# Result array name -> (dtype, shape), and one planned stored NPY member.
_Layouts = Mapping[str, tuple[np.dtype[Any], tuple[int, ...]]]
_Member = tuple[str, str, bytes, np.dtype[Any], tuple[int, ...], int]


def _planned_members(layouts: _Layouts, limits: _Limits, /) -> list[_Member]:
    """Admit result arrays from dtype and shape alone; return their stored members.

    Every array must fit ``max_array_rank``, ``max_tensor_elements`` and, as an
    NPY member, ``max_tensor_bytes``; all arrays together ``max_total_elements``.
    """

    planned = []
    elements_total = 0
    for index, name in enumerate(sorted(layouts)):
        dtype, shape = layouts[name]
        elements = math.prod(shape)
        if (
            len(shape) > limits.max_array_rank
            or any(extent > limits.max_tensor_elements for extent in shape)
            or elements > limits.max_tensor_elements
        ):
            raise _refuse(
                f"Provider array {name!r} of shape {shape} exceeds the rank or tensor "
                f"element limit {limits.max_tensor_elements}."
            )
        header = _npy_header(dtype, shape)
        size = len(header) + elements * dtype.itemsize
        if size > limits.max_tensor_bytes:
            raise _refuse(
                f"Provider array {name!r} needs {size} bytes, above the tensor byte "
                f"limit {limits.max_tensor_bytes}."
            )
        elements_total += elements
        planned.append((name, f"arrays/{index:06d}.npy", header, dtype, shape, size))
    if elements_total > limits.max_total_elements:
        raise _refuse("Provider arrays exceed the aggregate element limit.")
    return planned


def _member_bytes(planned: Sequence[_Member], /) -> int:
    """Container bytes of the planned members and the ZIP end record."""

    return _ZIP_END.size + sum(
        _stored_member_bytes(member, size) for _, member, _, _, _, size in planned
    )


def _write_archive(
    path: Path,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, Any],
    limits: _Limits,
    /,
) -> None:
    """Admit every array from its metadata, then stream the archive.

    All shapes, dtypes, member sizes, the manifest and the container size are
    checked before any array is converted or encoded; members are then written
    one at a time from views and the manifest, with their digests, last.
    """

    planned = _planned_members(
        {name: _result_layout(name, value) for name, value in arrays.items()}, limits
    )
    inventory = {
        name: {
            "member": member,
            "shape": list(shape),
            "dtype": dtype.str,
            # Digests are fixed-width; the final manifest has this exact size.
            "sha256": "0" * 64,
            "order": "C",
        }
        for name, member, _, dtype, shape, _ in planned
    }
    manifest_size = len(_manifest_payload(manifest, inventory))
    if manifest_size > limits.max_manifest_bytes:
        raise _refuse("The provider result manifest exceeds the manifest byte limit.")
    _admit_result_plan(
        len(planned) + 1,
        _member_bytes(planned) + _stored_member_bytes("manifest.json", manifest_size),
        limits,
    )
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, member, header, _, _, size in planned:
            digest = hashlib.sha256()
            with archive.open(_stored_information(member, size), mode="w") as stream:
                for chunk in _npy_chunks(header, _result_view(arrays[name])):
                    digest.update(chunk)
                    stream.write(chunk)
            inventory[name]["sha256"] = digest.hexdigest()
        payload = _manifest_payload(manifest, inventory)
        archive.writestr(_stored_information("manifest.json", len(payload)), payload)


def _torch_records(members: Sequence[zipfile.ZipInfo], limits: _Limits, /) -> str:
    """Admit the records of one torch.save archive; return its record prefix."""

    names = [member.filename for member in members]
    prefix = names[0].split("/", 1)[0] + "/" if names and "/" in names[0] else ""
    if (
        not prefix
        or len(set(names)) != len(names)
        or any(not name.startswith(prefix) for name in names)
        or prefix + "data.pkl" not in names
    ):
        raise _refuse("The torch source records are not one torch.save archive.")
    expanded = 0
    for member in members:
        record = member.filename.removeprefix(prefix)
        if (
            member.compress_type != zipfile.ZIP_STORED
            or member.compress_size != member.file_size
            or member.flag_bits & 0x1
        ):
            raise _refuse(
                "Torch source records must be stored uncompressed and unencrypted, "
                "as torch.save writes them."
            )
        key = record.removeprefix("data/")
        if key != record and key and "/" not in key:
            maximum = limits.max_tensor_bytes
        elif record == "data.pkl" or record in _TORCH_METADATA_RECORDS:
            maximum = limits.max_manifest_bytes
        else:
            raise _refuse(
                f"Torch source record {record!r} is outside the torch.save layout."
            )
        if member.file_size > maximum:
            raise _refuse(
                f"Torch source record {record!r} has {member.file_size} bytes, above its "
                f"limit {maximum}."
            )
        expanded += member.file_size
    if expanded > limits.max_source_bytes:
        raise _refuse("The torch source records expand beyond the source byte limit.")
    return prefix


def _torch_container(data: bytes, limits: _Limits, /) -> tuple[zipfile.ZipFile, str]:
    """Admit the ``torch.save`` ZIP container before any record is read.

    The end records are decoded first, so the member count and directory size
    are bounded before the directory is parsed. Every record must be stored as
    torch.save writes it; its declared size is exactly what torch allocates to
    read it, and is bounded here.
    """

    end = len(data) - _ZIP_END.size
    if (
        not data.startswith(b"PK\x03\x04")
        or end < 0
        or data[end : end + 4] != b"PK\x05\x06"
    ):
        raise _refuse(
            "The torch source is not the ZIP container torch.save writes; other "
            "torch serializations have no bounded admission path."
        )
    _, disk, directory_disk, disk_entries, entries, size, offset, comment = (
        _ZIP_END.unpack_from(data, end)
    )
    if disk or directory_disk or disk_entries != entries or comment:
        raise _refuse("The torch source ZIP directory is not canonical.")
    directory_end = end
    locator = end - _ZIP64_LOCATOR.size
    if locator >= _ZIP64_END.size and data[locator : locator + 4] == b"PK\x06\x07":
        _, locator_disk, record, disks = _ZIP64_LOCATOR.unpack_from(data, locator)
        # zipfile reads the ZIP64 end record just before the locator; torch's
        # reader follows the locator. Both must name the same record.
        if locator_disk or disks != 1 or record != locator - _ZIP64_END.size:
            raise _refuse("The torch source ZIP64 locator is not canonical.")
        (
            signature,
            _,
            _,
            _,
            disk64,
            directory_disk64,
            disk_entries64,
            entries64,
            size64,
            offset64,
        ) = _ZIP64_END.unpack_from(data, record)
        if (
            signature != b"PK\x06\x06"
            or disk64
            or directory_disk64
            or disk_entries64 != entries64
            or entries not in (entries64, 0xFFFF)
            or size not in (size64, 0xFFFFFFFF)
            or offset not in (offset64, 0xFFFFFFFF)
        ):
            raise _refuse("The torch source ZIP64 directory is inconsistent.")
        entries, size, offset, directory_end = entries64, size64, offset64, record
    elif entries == 0xFFFF or 0xFFFFFFFF in (size, offset):
        raise _refuse("The torch source ZIP64 directory is missing.")
    if entries > limits.max_members:
        raise _refuse(
            f"The torch source container has {entries} records, above the member "
            f"limit {limits.max_members}."
        )
    if size > limits.max_manifest_bytes:
        raise _refuse("The torch source ZIP directory exceeds the manifest byte limit.")
    if offset + size != directory_end:
        raise _refuse("The torch source ZIP directory is not where its end record says.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise _refuse("The torch source ZIP directory is malformed.") from error
    try:
        if len(archive.infolist()) != entries:
            raise _refuse("The torch source ZIP directory is inconsistent.")
        prefix = _torch_records(archive.infolist(), limits)
    except ProviderRefusal:
        archive.close()
        raise
    return archive, prefix


@dataclass(frozen=True, slots=True)
class _PickledGlobal:
    module: str
    name: str


class _PickledMapping(dict):
    """A pickled ``collections.OrderedDict``, reconstructed only as metadata."""


@dataclass(frozen=True, slots=True)
class _StorageDeclaration:
    key: str
    dtype: str
    itemsize: int
    numel: int


@dataclass(frozen=True, slots=True)
class _TensorDeclaration:
    storage: _StorageDeclaration
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def _extents(value: Any, /) -> bool:
    return isinstance(value, tuple) and all(
        type(extent) is int and extent >= 0 for extent in value
    )


def _pickled_global(module: Any, name: Any, /) -> _PickledGlobal:
    if (module, name) not in _STATE_GLOBALS:
        raise _refuse(
            f"The state-dict pickle names {module}.{name}, outside the safe "
            "state-dict format."
        )
    return _PickledGlobal(module, name)


def _storage_declaration(
    identifier: Any, storages: dict[str, _StorageDeclaration], /
) -> _StorageDeclaration:
    if not (
        isinstance(identifier, tuple)
        and len(identifier) == 5
        and identifier[0] == "storage"
        and isinstance(identifier[1], _PickledGlobal)
        and identifier[1].module == "torch"
        and isinstance(identifier[2], str)
        and identifier[2]
        and "/" not in identifier[2]
        and isinstance(identifier[3], str)
        and type(identifier[4]) is int
        and identifier[4] >= 0
    ):
        raise _refuse("A state-dict storage declaration is malformed.")
    dtype, itemsize = _STATE_STORAGES[identifier[1].name]
    storage = _StorageDeclaration(identifier[2], dtype, itemsize, identifier[4])
    if storages.setdefault(storage.key, storage) != storage:
        raise _refuse(f"State-dict storage {storage.key!r} is declared inconsistently.")
    return storage


def _pickled_call(function: Any, arguments: Any, /) -> Any:
    if not isinstance(function, _PickledGlobal) or not isinstance(arguments, tuple):
        raise _refuse("The state-dict pickle reduces a non-global.")
    match (function.module, function.name):
        case ("collections", "OrderedDict") if not arguments:
            return _PickledMapping()
        case ("torch._utils", "_rebuild_tensor_v2") if len(arguments) == 6:
            storage, offset, shape, stride, requires_grad, hooks = arguments
            if (
                not isinstance(storage, _StorageDeclaration)
                or type(offset) is not int
                or offset < 0
                or not _extents(shape)
                or not _extents(stride)
                or len(shape) != len(stride)
                or type(requires_grad) is not bool
                or not isinstance(hooks, _PickledMapping)
                or hooks
            ):
                raise _refuse("A state-dict tensor declaration is malformed.")
            return _TensorDeclaration(storage, offset, shape, stride)
    raise _refuse(
        f"The state-dict pickle calls {function.module}.{function.name} outside the "
        "safe state-dict format."
    )


def _state_declarations(
    payload: bytes, /
) -> tuple[dict[str, _TensorDeclaration], dict[str, _StorageDeclaration]]:
    """Read a safe state dict's ``data.pkl`` as metadata without unpickling it.

    ``pickletools`` decodes the opcodes; only the opcodes and globals
    ``torch.save`` emits for a (possibly ordered) dict of tensors are interpreted,
    symbolically: persistent storage identifiers become declarations and
    ``_rebuild_tensor_v2`` calls become tensor views. Nothing is called and
    nothing is allocated from a declared size.
    """

    stack: list[Any] = []
    frames: list[list[Any]] = []
    memo: dict[int, Any] = {}
    storages: dict[str, _StorageDeclaration] = {}
    result: Any = None
    try:
        for opcode, argument, _ in pickletools.genops(payload):
            match opcode.name:
                case "PROTO" | "FRAME":
                    pass
                case "MARK":
                    frames.append(stack)
                    stack = []
                case "BINPUT" | "LONG_BINPUT":
                    if not isinstance(argument, int):
                        raise ValueError(opcode.name)
                    memo[argument] = stack[-1]
                case "MEMOIZE":
                    memo[len(memo)] = stack[-1]
                case "BINGET" | "LONG_BINGET":
                    if not isinstance(argument, int):
                        raise ValueError(opcode.name)
                    stack.append(memo[argument])
                case "GLOBAL":
                    if not isinstance(argument, str):
                        raise ValueError(opcode.name)
                    module, name = argument.split(" ")
                    stack.append(_pickled_global(module, name))
                case "STACK_GLOBAL":
                    name = stack.pop()
                    stack.append(_pickled_global(stack.pop(), name))
                case (
                    "BINUNICODE"
                    | "SHORT_BINUNICODE"
                    | "BINUNICODE8"
                    | "BININT"
                    | "BININT1"
                    | "BININT2"
                    | "LONG1"
                ):
                    stack.append(argument)
                case "NONE":
                    stack.append(None)
                case "NEWTRUE" | "NEWFALSE":
                    stack.append(opcode.name == "NEWTRUE")
                case "EMPTY_TUPLE":
                    stack.append(())
                case "TUPLE":
                    items = tuple(stack)
                    stack = frames.pop()
                    stack.append(items)
                case "TUPLE1" | "TUPLE2" | "TUPLE3":
                    count = int(opcode.name[-1])
                    if len(stack) < count:
                        raise IndexError(opcode.name)
                    items = tuple(stack[-count:])
                    del stack[-count:]
                    stack.append(items)
                case "EMPTY_DICT":
                    stack.append({})
                case "SETITEM":
                    value = stack.pop()
                    key = stack.pop()
                    stack[-1][key] = value
                case "SETITEMS":
                    items = stack
                    stack = frames.pop()
                    if len(items) % 2 or not isinstance(stack[-1], dict):
                        raise ValueError(opcode.name)
                    for key, value in zip(items[::2], items[1::2], strict=True):
                        stack[-1][key] = value
                case "BINPERSID":
                    stack.append(_storage_declaration(stack.pop(), storages))
                case "REDUCE":
                    arguments = stack.pop()
                    stack.append(_pickled_call(stack.pop(), arguments))
                case "BUILD":
                    # torch.save keeps the state dict's ``_metadata`` (module
                    # versions) as the OrderedDict's instance state.
                    state = stack.pop()
                    if (
                        not isinstance(stack[-1], _PickledMapping)
                        or not isinstance(state, dict)
                        or set(state) - {"_metadata"}
                    ):
                        raise ValueError(opcode.name)
                case "STOP":
                    result = stack.pop()
                    if stack or frames:
                        raise ValueError(opcode.name)
                case other:
                    raise _refuse(
                        f"Pickle opcode {other} is outside the safe state-dict format."
                    )
    except ProviderRefusal:
        raise
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise _refuse("The state-dict pickle metadata is malformed.") from error
    # torch.save writes an OrderedDict (module state dicts) or a plain dict.
    if not isinstance(result, dict) or any(
        not isinstance(name, str) or not isinstance(value, _TensorDeclaration)
        for name, value in result.items()
    ):
        raise _refuse("A safe state-dict source must map names to tensors.")
    return dict(result), storages


def _state_inventory(
    archive: zipfile.ZipFile, prefix: str, limits: _Limits, /
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """Admit every storage and tensor view of a safe state dict before loading.

    Storage records must hold exactly their declared bytes; each tensor's
    logical rank, elements and bytes (what any later copy of it allocates) and
    their aggregate are bounded, and every view must lie inside its storage.
    """

    tensors, storages = _state_declarations(archive.read(prefix + "data.pkl"))
    records = {
        member.filename.removeprefix(prefix + "data/"): member.file_size
        for member in archive.infolist()
        if member.filename.startswith(prefix + "data/")
    }
    if set(records) != set(storages):
        raise _refuse("The state-dict storage records differ from its declared storages.")
    for key, storage in storages.items():
        if records[key] != storage.numel * storage.itemsize:
            raise _refuse(
                f"State-dict storage {key!r} declares {storage.numel} {storage.dtype} "
                f"elements but its record holds {records[key]} bytes."
            )
    logical = 0
    inventory = []
    for name, tensor in tensors.items():
        elements = math.prod(tensor.shape)
        if (
            len(tensor.shape) > limits.max_array_rank
            or any(extent > limits.max_tensor_elements for extent in tensor.shape)
            or elements > limits.max_tensor_elements
        ):
            raise _refuse(
                f"Source tensor {name} of shape {tensor.shape} exceeds the rank or tensor"
                f" element limit {limits.max_tensor_elements}."
            )
        size = elements * tensor.storage.itemsize
        if size > limits.max_tensor_bytes:
            raise _refuse(
                f"Source tensor {name} views {size} logical bytes, above the tensor byte "
                f"limit {limits.max_tensor_bytes}."
            )
        span = tensor.offset + (
            sum((extent - 1) * step for extent, step in zip(tensor.shape, tensor.stride))
            + 1
            if elements
            else 0
        )
        if span > tensor.storage.numel:
            raise _refuse(f"Source tensor {name} views beyond its storage.")
        logical += size
        inventory.append((name, tensor.storage.dtype, tensor.shape))
    if logical > limits.max_source_bytes:
        raise _refuse(
            "Source tensors view more logical bytes than the source byte limit."
        )
    return tuple(inventory)


def _state_shape(state: Mapping[str, Any], name: str, /) -> tuple[int, ...]:
    if name not in state:
        raise _refuse(f"The admitted state dict lacks the declared tensor {name}.")
    return tuple(int(extent) for extent in state[name].shape)


def _indices(state: Mapping[str, Any], prefix: str, /) -> set[int]:
    """The integer children ``prefix.<i>.`` named in the state dict."""

    return {
        int(child)
        for child in (
            name.removeprefix(prefix).split(".", 1)[0]
            for name in state
            if name.startswith(prefix)
        )
        if child.isdigit()
    }


def _admit_declared_dimensions(
    declaration: Mapping[str, Any], state: Mapping[str, Any], /
) -> None:
    """Pin every size-determining declared field to the admitted state dict.

    The provider constructor allocates the declared model, including its
    generalized-CG U bases, before its layout can be compared with the state.
    Elements, interactions, Bessel and radial widths, spherical-harmonic
    degree, hidden irreps and channels, correlation orders, heads and readout
    MLP widths are each read off admitted tensors here (whose sizes are within
    the source limits), so construction realizes exactly the admitted
    dimensions; the exact name/shape/dtype comparison follows construction.
    """

    from e3nn import o3

    numbers = declaration["atomic_numbers"]
    count = declaration["num_interactions"]
    widths = [declaration["num_bessel"], *declaration["radial_MLP"]]
    heads = declaration["heads"]
    harmonics = (declaration["max_ell"] + 1) ** 2
    try:
        hidden = o3.Irreps(declaration["hidden_irreps"])
        mlp = None if count == 1 else o3.Irreps(declaration["MLP_irreps"])
    except (AssertionError, TypeError, ValueError) as error:
        raise _refuse("The declared irreps are not e3nn irreps.") from error

    def require(condition: bool, field: str, /) -> None:
        if not condition:
            raise _refuse(f"The declared {field} differs from the admitted state dict.")

    require(
        _state_shape(state, "atomic_numbers") == (len(numbers),)
        and state["atomic_numbers"].tolist() == numbers,
        "atomic_numbers",
    )
    require(
        _state_shape(state, "num_interactions") == ()
        and int(state["num_interactions"]) == count
        and _indices(state, "interactions.") == set(range(count))
        and _indices(state, "products.") == set(range(count)),
        "num_interactions",
    )
    require(
        _state_shape(state, "radial_embedding.bessel_fn.bessel_weights") == (widths[0],),
        "num_bessel",
    )
    require(
        _state_shape(state, "atomic_energies_fn.atomic_energies")
        == np.shape(declaration["atomic_energies"])
        and _state_shape(state, "readouts.0.linear.output_mask") == (len(heads),),
        "heads",
    )
    if mlp is not None:
        require(
            _state_shape(state, f"readouts.{count - 1}.linear_1.output_mask")
            == (len(heads) * mlp.dim,),
            "MLP_irreps",
        )
    for index in range(count):
        radial = f"interactions.{index}.conv_tp_weights.layer"
        require(
            all(
                _state_shape(state, f"{radial}{layer}.weight")[0] == width
                for layer, width in enumerate(widths)
            )
            and f"{radial}{len(widths)}.weight" not in state,
            "radial_MLP",
        )
        # The last product keeps only the first (scalar) hidden irrep.
        target = hidden[:1] if index == count - 1 else hidden
        contractions = f"products.{index}.symmetric_contractions.contractions."
        require(_indices(state, contractions) == set(range(len(target))), "hidden_irreps")
        for order, (multiplicity, irrep) in enumerate(target):
            prefix = f"{contractions}{order}."
            output = () if irrep.l == 0 else (2 * irrep.l + 1,)
            require(
                _state_shape(state, prefix + "weights_max")[-1:] == (multiplicity,),
                "hidden_irreps",
            )
            degree = declaration["correlation"][index]
            require(
                all(
                    _state_shape(state, f"{prefix}U_matrix_{nu}")[:-1]
                    == output + (harmonics,) * nu
                    for nu in range(1, degree + 1)
                )
                and f"{prefix}U_matrix_{degree + 1}" not in state,
                "max_ell, hidden_irreps or correlation",
            )


def _load_source(
    request: Mapping[str, Any], root: Path, limits: _Limits, /
) -> tuple[Any, str]:
    import torch

    source = request["source"]
    data = (root / "source.bin").read_bytes()
    if (
        len(data) != source["byte_size"]
        or hashlib.sha256(data).hexdigest() != source["sha256"]
    ):
        raise _refuse("Staged source bytes differ from the admitted source digest.")
    kind = source["kind"]
    if kind not in ("torch-full-model", "torch-state-dict"):
        raise _refuse(f"Unsupported source kind {kind!r}.")
    if kind == "torch-full-model" and source.get("trusted_full_object") is not True:
        raise _refuse(
            "Full-object torch checkpoints require the explicit trusted-source "
            "capability."
        )
    archive, prefix = _torch_container(data, limits)
    with archive:
        inventory = (
            None
            if kind == "torch-full-model"
            else _state_inventory(archive, prefix, limits)
        )
    if kind == "torch-full-model":
        # Executable deserialization of the admitted digest, explicitly trusted
        # by the caller; never reached by an absent or failed trust decision.
        model = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise _refuse("The trusted torch checkpoint did not contain a module.")
        return model, kind
    try:
        state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as error:
        # Never retried with executable deserialization.
        raise _refuse(
            "The weights-only loader refused the state-dict source; "
            f"there is no unsafe fallback ({type(error).__name__})."
        ) from error
    if not isinstance(state, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise _refuse("A safe state-dict source must map names to tensors.")
    loaded = tuple(
        (name, str(value.dtype).removeprefix("torch."), tuple(value.shape))
        for name, value in state.items()
    )
    if loaded != inventory:
        raise _refuse("The loaded state differs from its admitted pickle metadata.")
    declaration = request.get("architecture")
    if not isinstance(declaration, Mapping):
        raise _refuse("A state-dict source requires an exact declared architecture.")
    # Construction allocates the declared model; its dimensions must first be
    # those of the admitted state.
    try:
        _admit_declared_dimensions(declaration, state)
    except ProviderRefusal:
        raise
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise _refuse("The declared architecture is malformed.") from error
    dtype = _state_dtype(state)
    model = _build(declaration, dtype, limits)
    expected = model.state_dict()
    if list(expected) != list(state) or any(
        expected[name].shape != state[name].shape
        or expected[name].dtype != state[name].dtype
        for name in state
    ):
        raise _refuse("State-dict tensors differ from the declared architecture layout.")
    model.load_state_dict(dict(state), strict=True)
    return model, kind


def _state_dtype(state: Mapping[str, Any], /) -> Any:
    import torch

    dtypes = {value.dtype for value in state.values() if value.is_floating_point()}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.float32, torch.float64):
        raise _refuse("Source floating tensors must share float32 or float64.")
    return next(iter(dtypes))


def _type_key(value: Any, /) -> tuple[str, str]:
    kind = type(value)
    return kind.__module__, kind.__qualname__


def _admit_module_tree(model: Any, /) -> str:
    model_class = _ADMITTED_MODEL_CLASSES.get(_type_key(model))
    if model_class is None:
        raise _refuse(f"Model class {_type_key(model)} is not an admitted standard MACE.")
    for name, module in model.named_modules():
        key = _type_key(module)
        if key in _ADMITTED_MODULE_TYPES:
            continue
        generated = any(part in _GENERATED_MODULE_NAMES for part in name.split("."))
        if key in _GENERATED_MODULE_TYPES and generated:
            continue
        raise _refuse(f"Module {name or '<root>'} of type {key} is not admitted.")
    for refused in ("joint_embedding", "embedding_readout", "lora"):
        if hasattr(model, refused):
            raise _refuse(f"Model extension {refused!r} is outside standard MACE.")
    return model_class


def _activation_name(function: Any, /) -> str:
    import torch

    if function is torch.nn.functional.silu:
        return "silu"
    if function is torch.tanh:
        return "tanh"
    raise _refuse("Provider activation is not an admitted source function.")


def _normalized_activation(module: Any, /) -> dict[str, Any]:
    if _type_key(module) != ("e3nn.math._normalize_activation", "normalize2mom"):
        raise _refuse("Provider activation is not an e3nn normalize2mom function.")
    return {
        "function": _activation_name(module.f),
        "constant": float(module.cst),
        "identity": bool(module._is_id),
    }


def _fully_connected(module: Any, /) -> dict[str, Any]:
    layers = []
    for index in range(len(module.hs) - 1):
        layer = getattr(module, f"layer{index}")
        layers.append(
            {
                "h_in": int(layer.h_in),
                "h_out": int(layer.h_out),
                "var_in": float(layer.var_in),
                "var_out": float(layer.var_out),
                "activation": None
                if layer.act is None
                else _normalized_activation(layer.act),
            }
        )
    return {"hs": [int(value) for value in module.hs], "layers": layers}


def _linear(module: Any, /) -> dict[str, Any]:
    if _type_key(module) != ("e3nn.o3._linear", "Linear"):
        raise _refuse("Provider linear map is not an e3nn Linear.")
    if module.bias.numel() != 0:
        raise _refuse("Standard MACE linear maps carry no bias.")
    return {
        "irreps_in": str(module.irreps_in),
        "irreps_out": str(module.irreps_out),
        "instructions": [
            {
                "i_in": int(instruction.i_in),
                "i_out": int(instruction.i_out),
                "path_shape": [int(value) for value in instruction.path_shape],
                "path_weight": float(instruction.path_weight),
            }
            for instruction in module.instructions
        ],
    }


def _tensor_product(module: Any, /) -> dict[str, Any]:
    return {
        "irreps_in1": str(module.irreps_in1),
        "irreps_in2": str(module.irreps_in2),
        "irreps_out": str(module.irreps_out),
        "internal_weights": bool(module.internal_weights),
        "shared_weights": bool(module.shared_weights),
        "instructions": [
            {
                "i_in1": int(instruction.i_in1),
                "i_in2": int(instruction.i_in2),
                "i_out": int(instruction.i_out),
                "connection_mode": str(instruction.connection_mode),
                "has_weight": bool(instruction.has_weight),
                "path_weight": float(instruction.path_weight),
                "path_shape": [int(value) for value in instruction.path_shape],
            }
            for instruction in module.instructions
        ],
    }


def _flag(module: Any, name: str, default: bool, /) -> bool:
    # Provider forwards read these attributes through ``hasattr``; an absent
    # attribute selects the forward's documented default, not a guess.
    return bool(getattr(module, name)) if hasattr(module, name) else default


def _interaction_record(interaction: Any, /) -> dict[str, Any]:
    kind = type(interaction).__qualname__
    record: dict[str, Any] = {
        "class": kind,
        "avg_num_neighbors": float(interaction.avg_num_neighbors),
        "radial_MLP": [int(value) for value in interaction.radial_MLP],
        "node_feats_irreps": str(interaction.node_feats_irreps),
        # Earlier releases store no ``edge_irreps``; the admitted constructor's
        # default for an absent value is ``node_feats_irreps``.
        "edge_irreps": str(
            interaction.edge_irreps
            if hasattr(interaction, "edge_irreps")
            else interaction.node_feats_irreps
        ),
        "target_irreps": str(interaction.target_irreps),
        "hidden_irreps": str(interaction.hidden_irreps),
        "linear_up": _linear(interaction.linear_up),
        "conv_tp": _tensor_product(interaction.conv_tp),
        "conv_tp_weights": _fully_connected(interaction.conv_tp_weights),
        "linear": _linear(interaction.linear),
        "skip_tp": _tensor_product(interaction.skip_tp),
        "density_fn": _fully_connected(interaction.density_fn)
        if hasattr(interaction, "density_fn")
        else None,
    }
    if hasattr(interaction, "conv_fusion"):
        raise _refuse("Fused provider convolutions are outside the admitted layout.")
    if (
        getattr(interaction, "cueq_config", None) is not None
        or getattr(interaction, "oeq_config", None) is not None
    ):
        raise _refuse(
            "cuEquivariance/OpenEquivariance module layouts are refused; convert the "
            "source to the e3nn layout with the provider's mace.cli.convert_cueq_e3nn "
            "(or convert_oeq_e3nn) and admit the converted bytes."
        )
    return record


def _product_record(product: Any, /) -> dict[str, Any]:
    if getattr(product, "cueq_config", None) is not None:
        raise _refuse(
            "cuEquivariance symmetric-contraction modules are refused; convert the "
            "source with the provider's mace.cli.convert_cueq_e3nn first."
        )
    contraction = product.symmetric_contractions
    if _type_key(contraction) != (
        "mace.modules.symmetric_contraction",
        "SymmetricContraction",
    ):
        raise _refuse("Product basis is not the e3nn-layout symmetric contraction.")
    return {
        "use_sc": bool(product.use_sc),
        "use_agnostic_product": _flag(product, "use_agnostic_product", False),
        "irreps_in": str(contraction.irreps_in),
        "irreps_out": str(contraction.irreps_out),
        "correlations": [int(item.correlation) for item in contraction.contractions],
        "linear": _linear(product.linear),
    }


def _readout_record(readout: Any, /) -> dict[str, Any]:
    kind = type(readout).__qualname__
    if kind == "LinearReadoutBlock":
        return {"class": kind, "linear": _linear(readout.linear)}
    activation = readout.non_linearity
    if len(activation.acts) != 1:
        raise _refuse("Nonlinear readouts must use one scalar activation.")
    return {
        "class": kind,
        "num_heads": int(readout.num_heads) if hasattr(readout, "num_heads") else 1,
        "linear_1": _linear(readout.linear_1),
        "linear_2": _linear(readout.linear_2),
        "activation": _normalized_activation(activation.acts[0]),
    }


def _scalar_count(irreps: Any, /) -> int:
    from e3nn import o3

    return o3.Irreps(irreps).count(o3.Irrep(0, 1))


def _declaration(model: Any, model_class: str, /) -> dict[str, Any]:
    from e3nn import o3

    interactions = list(model.interactions)
    count = len(interactions)
    if count < 1 or int(model.num_interactions) != count:
        raise _refuse("Interaction count is inconsistent.")
    classes = [type(item).__qualname__ for item in interactions]
    if any(name not in _ADMITTED_INTERACTIONS for name in classes):
        raise _refuse("An interaction class is outside the admitted standard set.")
    if len(set(classes[1:])) > 1:
        raise _refuse("Interactions after the first must share one source class.")
    averages = {float(item.avg_num_neighbors) for item in interactions}
    radial = {tuple(int(v) for v in item.radial_MLP) for item in interactions}
    if len(averages) != 1 or len(radial) != 1:
        raise _refuse("Interactions disagree on neighbor normalization or radial MLP.")
    radial_embedding = model.radial_embedding
    if _type_key(radial_embedding.bessel_fn) != ("mace.modules.radial", "BesselBasis"):
        raise _refuse("Only the Bessel radial basis is admitted.")
    if hasattr(radial_embedding, "distance_transform"):
        transform = radial_embedding.distance_transform
        if _type_key(transform) != ("mace.modules.radial", "AgnesiTransform"):
            raise _refuse("Only the Agnesi distance transform is admitted.")
        distance_transform = "Agnesi"
    else:
        distance_transform = "None"
    harmonics = model.spherical_harmonics
    max_ell = int(harmonics._lmax)
    if (
        not bool(harmonics.normalize)
        or harmonics.normalization != "component"
        or o3.Irreps(harmonics.irreps_out) != o3.Irreps.spherical_harmonics(max_ell)
    ):
        raise _refuse("Edge harmonics must be normalized O(3) component harmonics.")
    hidden = (
        o3.Irreps(str(interactions[1].node_feats_irreps))
        if count > 1
        else o3.Irreps(str(model.products[0].linear.irreps_out))
    )
    last_target = o3.Irreps(str(model.products[-1].linear.irreps_out))
    if last_target != o3.Irreps(str(hidden[0])):
        raise _refuse("The last product must output only hidden scalars.")
    readouts = list(model.readouts)
    readout_kinds = [type(item).__qualname__ for item in readouts]
    last_only = len(readouts) == 1 and count > 1
    expected_kinds = (
        ["NonLinearReadoutBlock"]
        if last_only
        else ["LinearReadoutBlock"] * (count - 1)
        + (["NonLinearReadoutBlock"] if count > 1 else ["LinearReadoutBlock"])
    )
    if readout_kinds != expected_kinds:
        raise _refuse("Readout sequence is not the standard MACE readout layout.")
    heads = _heads(model)
    gate = None
    mlp_irreps = None
    if count > 1:
        final = readouts[-1]
        hidden_scalars = _scalar_count(final.linear_1.irreps_out)
        if hidden_scalars % len(heads):
            raise _refuse("Nonlinear readout width is not divisible by the head count.")
        mlp_irreps = f"{hidden_scalars // len(heads)}x0e"
        gate = _activation_name(final.non_linearity.acts[0].f)
    atomic_numbers = [int(value) for value in model.atomic_numbers.tolist()]
    declaration: dict[str, Any] = {
        "model_class": model_class,
        "r_max": float(model.r_max),
        "num_bessel": int(radial_embedding.out_dim),
        "num_polynomial_cutoff": int(radial_embedding.cutoff_fn.p),
        "max_ell": max_ell,
        "interaction_classes": classes,
        "num_interactions": count,
        "atomic_numbers": atomic_numbers,
        "hidden_irreps": str(hidden),
        "MLP_irreps": mlp_irreps,
        "avg_num_neighbors": averages.pop(),
        "correlation": [
            int(product.symmetric_contractions.contractions[0].correlation)
            for product in model.products
        ],
        "gate": gate,
        "pair_repulsion": hasattr(model, "pair_repulsion"),
        "distance_transform": distance_transform,
        "radial_MLP": list(radial.pop()),
        "radial_type": "bessel",
        "heads": heads,
        "apply_cutoff": _flag(model, "apply_cutoff", True),
        "use_reduced_cg": _reduced_cg_basis(model),
        "use_agnostic_product": _flag(model, "use_agnostic_product", False),
        "use_last_readout_only": last_only,
        "atomic_energies": model.atomic_energies_fn.atomic_energies.double().tolist(),
    }
    if model_class == "ScaleShiftMACE":
        declaration["atomic_inter_scale"] = model.scale_shift.scale.double().tolist()
        declaration["atomic_inter_shift"] = model.scale_shift.shift.double().tolist()
    if _flag(model, "use_so3", False) or getattr(model, "edge_irreps", None) is not None:
        raise _refuse("SO(3)-only or custom edge-irreps variants are not admitted.")
    return declaration


def _cuequivariance_available() -> bool:
    from mace.modules import wrapper_ops
    from mace.tools import cg

    return bool(cg.CUET_AVAILABLE) and bool(wrapper_ops.CUET_AVAILABLE)


def _u_relations(model: Any, /) -> dict[str, dict[bool, str]]:
    """Relation of every stored ``U_matrix_nu`` to each provider basis.

    ``"equal"``: the provider construction at the stored precision.
    ``"span"``: the same shape and coupling subspace in another orthonormal
    parameterization (earlier releases ordered or orthogonalized some paths
    differently), proven by an exact least-squares change of path basis.
    """

    import torch
    from e3nn import o3
    from mace.tools import cg

    bases = (False, True) if _cuequivariance_available() else (False,)
    relations: dict[str, dict[bool, str]] = {}
    for index, product in enumerate(model.products):
        contractions = product.symmetric_contractions
        for position, (contraction, irrep) in enumerate(
            zip(contractions.contractions, contractions.irreps_out, strict=True)
        ):
            buffers = dict(contraction.named_buffers())
            for order in range(1, int(contraction.correlation) + 1):
                name = (
                    f"products.{index}.symmetric_contractions.contractions."
                    f"{position}.U_matrix_{order}"
                )
                stored = buffers[f"U_matrix_{order}"].double()
                tolerance = 64.0 * float(
                    torch.finfo(buffers[f"U_matrix_{order}"].dtype).eps
                )
                relation = {}
                for reduced in bases:
                    reference = cg.U_matrix_real(
                        contraction.coupling_irreps,
                        o3.Irreps(str(irrep.ir)),
                        order,
                        use_cueq_cg=reduced,
                        dtype=torch.float64,
                    )[-1]
                    if reference.shape != stored.shape:
                        continue
                    scale = max(1.0, float(reference.abs().max()))
                    if float((reference - stored).abs().max()) <= tolerance * scale:
                        relation[reduced] = "equal"
                        continue
                    paths = stored.shape[-1]
                    left = reference.reshape(-1, paths)
                    right = stored.reshape(-1, paths)
                    change = torch.linalg.lstsq(left, right).solution
                    inverse = torch.linalg.lstsq(right, left).solution
                    if (
                        float((left @ change - right).abs().max()) <= tolerance * scale
                        and float((right @ inverse - left).abs().max())
                        <= tolerance * scale
                    ):
                        relation[reduced] = "span"
                relations[name] = relation
    return relations


def _reduced_cg_basis(model: Any, /) -> bool:
    """Identify the generalized-CG U basis from the stored contraction buffers.

    Every stored ``U_matrix_nu`` must be the provider's original e3nn basis or
    its reduced (cuequivariance) basis, or an exact reparameterization of the
    same coupling subspace, consistently across all contractions; exact
    equality is preferred. The ``use_reduced_cg`` attribute only selected a
    construction; earlier releases omit it and a provider without
    cuequivariance silently builds the original basis, so it is not evidence.
    The stored U buffers, not a reconstruction, are what the source executes.
    """

    relations = _u_relations(model)
    bases = (False, True) if _cuequivariance_available() else (False,)
    exact = {
        basis
        for basis in bases
        if all(relation.get(basis) == "equal" for relation in relations.values())
    }
    spanning = {
        basis
        for basis in bases
        if all(basis in relation for relation in relations.values())
    }
    candidates = exact or spanning
    if not candidates:
        if not _cuequivariance_available():
            raise _refuse(
                "Stored U bases are not the original e3nn basis; identifying a "
                "reduced-CG source requires the declared cuequivariance package."
            )
        raise _refuse("Stored U bases are neither one provider generalized-CG basis.")
    if len(candidates) == 1:
        return candidates.pop()
    # Both constructions coincide for every stored order: the flag is inert.
    return _flag(model, "use_reduced_cg", False)


def _reparameterized_u(model: Any, reduced: bool, /) -> list[str]:
    """Stored U buffers that span the declared basis in another parameterization."""

    return sorted(
        name
        for name, relation in _u_relations(model).items()
        if relation.get(reduced) == "span"
    )


# Batch size of the example inputs mace-torch traces symmetric contractions
# with, and of e3nn's own einsum-optimization examples.
_CONTRACTION_EXAMPLE_BATCH = 10
_E3NN_EXAMPLE_BATCH = 4
# Fixed buffers: Agnesi transform (q, p, a, 119 covalent radii) and ZBL
# repulsion (4 coefficients, p, 119 covalent radii, exponent, prefactor).
_AGNESI_ELEMENTS = (1, 1, 1, 119)
_ZBL_ELEMENTS = (4, 1, 119, 1, 1)


@dataclass(slots=True)
class _ConstructionPlan:
    """Element counts of every tensor a provider construction allocates.

    ``persistent`` lists each parameter and buffer of the constructed model
    (its state dict); ``transient`` the construction-time tensors: coupling
    recursion depths, traced example inputs and einsum intermediates.
    """

    limits: _Limits
    itemsize: int
    persistent: list[int]
    transient: list[int]

    def charge(self, elements: int, /, *, kept: bool = True) -> None:
        if (
            elements > self.limits.max_tensor_elements
            or elements * self.itemsize > self.limits.max_tensor_bytes
        ):
            raise _refuse(
                f"Provider construction needs a {elements}-element tensor, above the "
                "tensor limits."
            )
        (self.persistent if kept else self.transient).append(elements)


def _linear_numel(irreps_in: Any, irreps_out: Any, /) -> int:
    """Weights of an e3nn ``Linear``: every equal-irrep input/output pair."""

    return sum(
        mul_in * mul_out
        for mul_in, ir_in in irreps_in
        for mul_out, ir_out in irreps_out
        if ir_in == ir_out
    )


def _full_product_numel(first: Any, second: Any, output: Any, /) -> int:
    """Weights of an e3nn ``FullyConnectedTensorProduct`` ("uvw" paths)."""

    return sum(
        mul_1 * mul_2 * mul_out
        for mul_1, ir_1 in first
        for mul_2, ir_2 in second
        for mul_out, ir_out in output
        if ir_out in ir_1 * ir_2
    )


def _coupling_paths(coupling: Any, order: int, target: Any, reduced: bool, /) -> int:
    """Columns of mace-torch's U basis for ``target`` in ``coupling^(x order)``.

    The e3nn recursion keeps every coupling path tree ending in ``target``
    (intermediate irreps below l = 12 at order 4); the reduced basis is the
    multiplicity of ``target`` in the symmetric power, counted from weights.
    An empty basis is stored as one zero column.
    """

    if reduced:
        return max(_symmetric_multiplicity(coupling, order, target), 1)
    irreps = [ir for _, ir in coupling]
    allowed = None if order != 4 else {(l, 1 if l % 2 == 0 else -1) for l in range(12)}
    counts = {ir: 1 for ir in irreps}
    for _ in range(order - 1):
        following: dict[Any, int] = {}
        for left, count in counts.items():
            for right in irreps:
                for product in left * right:
                    if allowed is None or (product.l, product.p) in allowed:
                        following[product] = following.get(product, 0) + count
        counts = following
    return max(counts.get(target, 0), 1)


def _symmetric_multiplicity(coupling: Any, order: int, target: Any, /) -> int:
    """Multiplicity of ``target`` in ``Sym^order`` of the coupling irreps."""

    # (size, total m, parity) -> number of multisets of basis vectors.
    weights: dict[tuple[int, int, int], int] = {(0, 0, 1): 1}
    for _, ir in coupling:
        for m in range(-ir.l, ir.l + 1):
            grown: dict[tuple[int, int, int], int] = {}
            for (size, total, parity), count in weights.items():
                for repeat in range(order - size + 1):
                    key = (size + repeat, total + repeat * m, parity * ir.p**repeat)
                    grown[key] = grown.get(key, 0) + count
            weights = grown
    return weights.get((order, target.l, target.p), 0) - weights.get(
        (order, target.l + 1, target.p), 0
    )


def _einsum_peak(
    operands: Sequence[str], output: str, sizes: Mapping[str, int], /
) -> int:
    """Largest intermediate of the path opt_einsum chooses from shapes alone.

    mace-torch optimizes its traced contractions with ``opt_einsum_fx``, whose
    shape propagation does not execute einsums; the optimized graph's
    intermediates are those of ``opt_einsum.contract_path`` with default
    options, computed here from shapes without any tensor.
    """

    # Pinned external-provider allocation planning must reproduce the provider's
    # own path; importing native JAX execution here would change that boundary.
    import opt_einsum  # noqa: TID251

    equation = ",".join(operands) + "->" + output
    shapes = [tuple(sizes[index] for index in operand) for operand in operands]
    _, information = opt_einsum.contract_path(equation, *shapes, shapes=True)
    return int(information.largest_intermediate)


def _plan_contraction(
    plan: _ConstructionPlan,
    coupling: Any,
    target: Any,
    order: int,
    elements: int,
    channels: int,
    reduced: bool,
    /,
) -> None:
    """One mace-torch ``Contraction``: U bases, weights, flags and tracing."""

    width = coupling.dim
    workspace = 2
    for _ in range(2 * order if width > 1 else 0):
        workspace *= width
        # The e3nn recursion holds depth k: d**k paths' output dimensions
        # spanning d**k entries each, plus the previous depth.
        plan.charge(workspace, kept=False)
    letters = "fghjlmnopqrstuvwxyz"
    equivariant = "a" if target.l > 0 else ""
    batch = _CONTRACTION_EXAMPLE_BATCH
    for nu in range(1, order + 1):
        paths = _coupling_paths(coupling, nu, target, reduced)
        ells = letters[:nu]
        sizes = {"a": target.dim, "k": paths, "e": elements, "c": channels, "b": batch}
        sizes.update(dict.fromkeys(ells, width))
        basis = equivariant + ells + "k"
        plan.charge(math.prod(sizes[index] for index in basis))
        plan.charge(elements * paths * channels)
        plan.charge(1)
        plan.charge(batch * elements, kept=False)
        if nu == order:
            operands = (basis, "ekc", "bc" + ells[-1], "be")
            output = "bc" + equivariant + ells[:-1]
        else:
            operands = (basis, "ekc", "be")
            output = "bc" + equivariant + ells
            features = "bc" + equivariant + ells
            plan.charge(math.prod(sizes[index] for index in features), kept=False)
            plan.charge(
                _einsum_peak(
                    (features, "bc" + ells[-1]), "bc" + equivariant + ells[:-1], sizes
                ),
                kept=False,
            )
        plan.charge(batch * channels * width, kept=False)
        plan.charge(_einsum_peak(operands, output, sizes), kept=False)


def _plan_interaction(
    plan: _ConstructionPlan,
    declaration: Mapping[str, Any],
    index: int,
    irreps: Mapping[str, Any],
    /,
) -> Any:
    """One interaction block; returns its target irreps (the product input)."""

    from mace.modules.irreps_tools import tp_out_irreps_with_instructions

    node = irreps["node"] if index == 0 else irreps["hidden"]
    target = irreps["first"] if index == 0 else irreps["interaction"]
    output = irreps["outputs"][index]
    attributes = irreps["attributes"]
    harmonics = irreps["harmonics"]
    kind = declaration["interaction_classes"][index]
    plan.charge(_linear_numel(node, node))
    plan.charge(0)
    plan.charge(node.dim)
    middle, instructions = tp_out_irreps_with_instructions(node, harmonics, target)
    weights = sum(node[i].mul * harmonics[j].mul for i, j, _, _, _ in instructions)
    plan.charge(0)
    plan.charge(middle.dim)
    for l1, l2, l3 in sorted(
        {
            (node[i].ir.l, harmonics[j].ir.l, middle[k].ir.l)
            for i, j, k, _, _ in instructions
        }
    ):
        if min(l1, l2, l3) > 0:
            plan.charge((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1))
    plan.charge(_E3NN_EXAMPLE_BATCH * node.dim * harmonics.dim, kept=False)
    widths = [declaration["num_bessel"], *declaration["radial_MLP"], weights]
    for fan_in, fan_out in zip(widths, widths[1:]):
        plan.charge(fan_in * fan_out)
    plan.charge(_linear_numel(middle, target))
    plan.charge(0)
    plan.charge(target.dim)
    skip_in, skip_out = (node, output) if "Residual" in kind else (target, target)
    plan.charge(_full_product_numel(skip_in, attributes, skip_out))
    plan.charge(skip_out.dim)
    plan.charge(_E3NN_EXAMPLE_BATCH * skip_in.dim * attributes.dim, kept=False)
    if "Density" in kind:
        plan.charge(declaration["num_bessel"])
    return target


def _plan_readouts(
    plan: _ConstructionPlan,
    declaration: Mapping[str, Any],
    irreps: Mapping[str, Any],
    /,
) -> None:
    from e3nn import o3

    count = declaration["num_interactions"]
    heads = o3.Irreps(f"{len(declaration['heads'])}x0e")
    for index, output in enumerate(irreps["outputs"]):
        if index > 0 and index == count - 1:
            hidden = (len(declaration["heads"]) * irreps["mlp"]).simplify()
            stages = ((output, hidden), (hidden, heads))
        elif declaration["use_last_readout_only"]:
            continue
        else:
            stages = ((output, heads),)
        for irreps_in, irreps_out in stages:
            plan.charge(_linear_numel(irreps_in, irreps_out))
            plan.charge(0)
            plan.charge(irreps_out.dim)


def _admit_declared_extents(
    declaration: Mapping[str, Any], itemsize: int, limits: _Limits, /
) -> tuple[Any, Any]:
    """Bound the declared extents the planner expands, before expanding them.

    The planner's e3nn collections grow with the harmonic degree and the
    irreps entry counts, so before any of them: the ``(max_ell + 1)**2``
    edge-harmonic components (the leading extent of every first-layer
    coupling basis and traced convolution input the plan charges) must fit
    the per-tensor limits; the hidden irreps hold at most ``max_members``
    entries (each one symmetric contraction with separate basis and weight
    tensors, like each correlation order); and a nonlinear readout's MLP
    irreps simplify to the one entry the provider's single gate activates.
    Returns the hidden irreps and the simplified MLP irreps (``None`` for one
    interaction).
    """

    from e3nn import o3

    degree = declaration["max_ell"]
    if type(degree) is not int or degree < 0:
        raise _refuse("The declared max_ell must be a nonnegative integer.")
    components = (degree + 1) ** 2
    if (
        components > limits.max_tensor_elements
        or components * itemsize > limits.max_tensor_bytes
    ):
        raise _refuse(
            f"The declared max_ell {degree} spans {components} spherical-harmonic "
            "components, above the tensor limits."
        )
    hidden = o3.Irreps(declaration["hidden_irreps"])
    if len(hidden) > limits.max_members:
        raise _refuse(
            f"The declared hidden irreps hold {len(hidden)} entries, above the member "
            f"limit {limits.max_members}."
        )
    if declaration["num_interactions"] == 1:
        return hidden, None
    mlp = o3.Irreps(declaration["MLP_irreps"]).simplify()
    if len(mlp) != 1:
        raise _refuse(
            f"The declared MLP_irreps simplify to {len(mlp)} entries; the provider's "
            "nonlinear readout gates exactly one."
        )
    return hidden, mlp


def _channelwise(irreps: Any, channels: int, /) -> Any:
    """mace-torch's ``(irreps * channels).sort()[0].simplify()`` for distinct entries.

    Equal for irreps whose entries are distinct (spherical harmonics and their
    parity doubling), without the ``channels``-fold entry list the expression
    builds before simplifying.
    """

    from e3nn import o3

    return o3.Irreps([(channels, irrep) for _, irrep in irreps]).sort()[0].simplify()


def _admit_construction(
    declaration: Mapping[str, Any], itemsize: int, limits: _Limits, /
) -> _ConstructionPlan:
    """Plan and charge a provider construction before anything is allocated.

    Mirrors the admitted mace-torch 0.3.16 constructors from irreps metadata
    alone (mace-torch's own ``tp_out_irreps_with_instructions`` gives the
    convolution paths), after the declared degree and irreps entry counts are
    bounded (``_admit_declared_extents``): every parameter and buffer, every
    coupling-basis recursion depth, traced example input and worst-order
    einsum intermediate must fit the per-tensor limits, and the model's
    parameters and buffers together within ``max_source_bytes``.
    """

    from e3nn import o3

    count = declaration["num_interactions"]
    if sum(declaration["correlation"]) > limits.max_members:
        raise _refuse("The declared correlation orders exceed the member limit.")
    hidden, mlp = _admit_declared_extents(declaration, itemsize, limits)
    numbers = len(declaration["atomic_numbers"])
    channels = hidden.count(o3.Irrep(0, 1))
    harmonics = o3.Irreps.spherical_harmonics(declaration["max_ell"])
    coupled = (
        o3.Irreps("+".join(f"1x{l}e+1x{l}o" for l in range(declaration["max_ell"] + 1)))
        if hidden.count(o3.Irrep(0, -1)) > 0
        else harmonics
    )
    irreps = {
        "hidden": hidden,
        "harmonics": harmonics,
        "attributes": o3.Irreps([(numbers, (0, 1))]),
        "node": o3.Irreps([(channels, (0, 1))]),
        "first": _channelwise(harmonics, channels),
        "interaction": _channelwise(coupled, channels),
        "outputs": [
            o3.Irreps(str(hidden[0])) if index == count - 1 else hidden
            for index in range(count)
        ],
        "mlp": mlp,
    }
    plan = _ConstructionPlan(limits, itemsize, [], [])
    for elements in (
        numbers,
        1,
        1,
        numbers * channels,
        0,
        channels,
        declaration["num_bessel"],
        1,
        1,
        *(_AGNESI_ELEMENTS if declaration["distance_transform"] == "Agnesi" else ()),
        1,
        1,
        *(_ZBL_ELEMENTS if declaration["pair_repulsion"] else ()),
        int(np.size(declaration["atomic_energies"])),
        *((1, 1) if declaration["model_class"] == "ScaleShiftMACE" else ()),
    ):
        plan.charge(elements)
    elements = 1 if declaration["use_agnostic_product"] else numbers
    for index in range(count):
        target = _plan_interaction(plan, declaration, index, irreps)
        coupling = o3.Irreps([ir for _, ir in target])
        output = irreps["outputs"][index]
        for _, irrep in output:
            _plan_contraction(
                plan,
                coupling,
                irrep,
                declaration["correlation"][index],
                elements,
                target.count(o3.Irrep(0, 1)),
                declaration["use_reduced_cg"],
            )
        plan.charge(_linear_numel(output, output))
        plan.charge(0)
        plan.charge(output.dim)
    _plan_readouts(plan, declaration, irreps)
    if sum(plan.persistent) * itemsize > limits.max_source_bytes:
        raise _refuse(
            f"The declared model holds {sum(plan.persistent)} parameter and buffer "
            "elements, above the source byte limit."
        )
    return plan


def _build(declaration: Mapping[str, Any], dtype: Any, limits: _Limits, /) -> Any:
    import torch

    _admit_construction(declaration, torch.finfo(dtype).bits // 8, limits)
    from e3nn import o3
    from mace.modules import blocks, models

    classes = [getattr(blocks, name) for name in declaration["interaction_classes"]]
    if any(
        name not in _ADMITTED_INTERACTIONS for name in declaration["interaction_classes"]
    ):
        raise _refuse("Declared interaction class is not admitted.")
    gates = {"silu": torch.nn.functional.silu, "tanh": torch.tanh, None: None}
    if declaration["gate"] not in gates:
        raise _refuse("Declared readout gate is not admitted.")
    if declaration["radial_type"] != "bessel" or declaration[
        "distance_transform"
    ] not in (
        "None",
        "Agnesi",
    ):
        raise _refuse("Declared radial family is not admitted.")
    if declaration["use_reduced_cg"] and not _cuequivariance_available():
        # mace-torch would silently build the original basis instead.
        raise _refuse(
            "The reduced-CG U basis requires the declared cuequivariance provider package."
        )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        keywords: dict[str, Any] = {
            "r_max": declaration["r_max"],
            "num_bessel": declaration["num_bessel"],
            "num_polynomial_cutoff": declaration["num_polynomial_cutoff"],
            "max_ell": declaration["max_ell"],
            "interaction_cls": classes[1] if len(classes) > 1 else classes[0],
            "interaction_cls_first": classes[0],
            "num_interactions": declaration["num_interactions"],
            "num_elements": len(declaration["atomic_numbers"]),
            "hidden_irreps": o3.Irreps(declaration["hidden_irreps"]),
            # One-interaction models construct no nonlinear readout, so the
            # MLP irreps argument is unused by the provider constructor.
            "MLP_irreps": o3.Irreps(declaration["MLP_irreps"] or "0x0e"),
            "atomic_energies": np.asarray(
                declaration["atomic_energies"], dtype=np.float64
            ),
            "avg_num_neighbors": declaration["avg_num_neighbors"],
            "atomic_numbers": list(declaration["atomic_numbers"]),
            "correlation": list(declaration["correlation"]),
            "gate": gates[declaration["gate"]],
            "pair_repulsion": declaration["pair_repulsion"],
            "distance_transform": declaration["distance_transform"],
            "radial_MLP": list(declaration["radial_MLP"]),
            "radial_type": "bessel",
            "heads": list(declaration["heads"]),
            "apply_cutoff": declaration["apply_cutoff"],
            "use_reduced_cg": declaration["use_reduced_cg"],
            "use_agnostic_product": declaration["use_agnostic_product"],
            "use_last_readout_only": declaration["use_last_readout_only"],
        }
        match declaration["model_class"]:
            case "MACE":
                return models.MACE(**keywords)
            case "ScaleShiftMACE":
                return models.ScaleShiftMACE(
                    atomic_inter_scale=declaration["atomic_inter_scale"],
                    atomic_inter_shift=declaration["atomic_inter_shift"],
                    **keywords,
                )
            case other:
                raise _refuse(f"Declared model class {other!r} is not admitted.")
    finally:
        torch.set_default_dtype(previous)


def _probe_configurations(declaration: Mapping[str, Any], /) -> list[dict[str, Any]]:
    numbers = list(declaration["atomic_numbers"])
    species = [numbers[index % len(numbers)] for index in range(4)]
    r_max = float(declaration["r_max"])
    finite = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.31, 0.27, 0.05],
                [-0.12, 0.36, 0.29],
                [0.22, -0.18, 0.41],
            ],
            dtype=np.float64,
        )
        * r_max
    )
    cell = (
        np.array(
            [[0.9, 0.0, 0.0], [0.17, 0.85, 0.0], [0.11, 0.13, 0.95]], dtype=np.float64
        )
        * r_max
    )
    periodic = (
        np.array(
            [[0.03, 0.02, 0.01], [0.41, 0.37, 0.12], [0.22, 0.61, 0.44]], dtype=np.float64
        )
        @ cell
    )
    # A close pair inside every covalent-radius envelope exercises ZBL.
    close = np.array(
        [[0.0, 0.0, 0.0], [0.55, 0.0, 0.0], [0.1, 0.85, 0.2]], dtype=np.float64
    )
    return [
        {
            "numbers": species,
            "positions": finite,
            "cell": np.zeros((3, 3)),
            "pbc": [False] * 3,
        },
        {"numbers": species[:3], "positions": periodic, "cell": cell, "pbc": [True] * 3},
        {
            "numbers": [numbers[0], numbers[-1], numbers[0]],
            "positions": close,
            "cell": np.zeros((3, 3)),
            "pbc": [False] * 3,
        },
    ]


def _heads(model: Any, /) -> list[str]:
    if not hasattr(model, "heads"):
        return [_LEGACY_HEAD]
    return [str(head) for head in model.heads]


def _admitted_edges(
    configuration: Mapping[str, Any], cutoff: float, limits: _Limits, label: str, /
) -> int:
    """Bound the provider neighbor list of one configuration before discovery.

    mace-torch's ``get_neighborhood`` keeps every matscipy ``neighbour_list``
    triple ``(i, j, S)`` with ``|p_j - p_i + S @ cell| < cutoff`` except
    ``(i, i, 0)`` and holds whole ``[E, 3]`` float64 shifts (the largest of its
    ``[2, E]``/``[E, 3]`` topology tensors), so at most
    ``min(max_tensor_elements // 3, max_tensor_bytes // 24)`` edges are
    admitted. The same triples are counted here without materializing any:
    pair distances in bounded blocks, windowed along sorted abscissae, with a
    relative rounding slack on the cutoff so the count bounds the discovered
    ``E`` from above. Periodic triples range over the image shifts that the
    cell heights admit around the wrapped positions; that ``[T, 3]`` int64
    shift table is charged against the same bound. Counting stops at the
    first block past the bound.
    """

    capacity = min(limits.max_tensor_elements // 3, limits.max_tensor_bytes // 24)
    positions = np.asarray(configuration["positions"], dtype=np.float64)
    atoms = len(positions)
    if atoms == 0:
        return 0
    scale = cutoff + float(np.max(np.abs(positions)))
    if all(configuration["pbc"]):
        cell = np.asarray(configuration["cell"], dtype=np.float64)
        if not cell.any():
            # get_neighborhood's replacement of an all-zero cell.
            cell = np.identity(3)
        try:
            inverse = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            raise _refuse(f"Provider {label} has a singular periodic cell.") from None
        lengths = np.linalg.norm(cell, axis=1)
        # Reciprocal row norms: the inverse heights between lattice planes.
        reciprocal = np.linalg.norm(inverse, axis=0)
        slack = _PAIR_SLACK * (
            scale + float(np.sum(lengths * (cutoff * reciprocal + 2.0)))
        )
        reach = cutoff + slack
        extents = reach * reciprocal
        if not np.all(np.isfinite(extents)) or not math.isfinite(slack):
            raise _refuse(f"Provider {label} has a degenerate periodic cell.")
        # Wrapped fractional coordinates differ by less than one cell per axis.
        bounds = [math.ceil(extent) + 1 for extent in extents.tolist()]
        images = math.prod(2 * bound + 1 for bound in bounds)
        if images > capacity:
            raise _refuse(
                f"Provider {label} needs {images} periodic image shifts within the "
                f"cutoff, above the {capacity} the tensor limits admit."
            )
        points = positions - np.floor(positions @ inverse) @ cell
        grid = np.meshgrid(
            *(np.arange(-bound, bound + 1) for bound in bounds), indexing="ij"
        )
        vectors = np.stack([axis.reshape(-1) for axis in grid], axis=1) @ cell
        del grid
        vectors = vectors[
            np.linalg.norm(vectors, axis=1) <= reach + float(np.sum(lengths)) + slack
        ]
    else:
        reach = cutoff + _PAIR_SLACK * scale
        points = positions
        vectors = np.zeros((1, 3))
    points = points[np.argsort(points[:, 0], kind="stable")]
    squared_reach = reach * reach
    width = max(1, _PAIR_BLOCK // _PAIR_ROWS)
    # Every (i, i, 0) lies within the cutoff and is not an edge.
    edges = -atoms
    for vector in vectors:
        targets = points + vector
        abscissae = targets[:, 0]
        for start in range(0, atoms, _PAIR_ROWS):
            block = points[start : start + _PAIR_ROWS]
            low = int(np.searchsorted(abscissae, block[0, 0] - reach, side="left"))
            high = int(np.searchsorted(abscissae, block[-1, 0] + reach, side="right"))
            for first in range(low, high, width):
                window = targets[first : min(high, first + width)]
                squared = np.zeros((len(block), len(window)))
                for axis in range(3):
                    squared += np.subtract.outer(block[:, axis], window[:, axis]) ** 2
                edges += int(np.count_nonzero(squared <= squared_reach))
            if edges > capacity:
                raise _refuse(
                    f"Provider {label} has at least {edges} neighbor-list edges within "
                    f"the cutoff {cutoff}, above the {capacity} whose [E, 3] float64 "
                    "shifts fit the tensor limits."
                )
    return edges


def _batch(
    model: Any, configuration: Mapping[str, Any], head: str, edges: int, /
) -> dict[str, Any]:
    """The provider's own graph of an admitted configuration of at most ``edges``."""

    import ase
    from mace import data as mace_data
    from mace.tools import AtomicNumberTable, torch_geometric

    heads = _heads(model)
    if head not in heads:
        raise _refuse(f"Head {head!r} is not one of the source heads {heads}.")
    atoms = ase.Atoms(
        numbers=np.asarray(configuration["numbers"], dtype=np.int64),
        positions=np.asarray(configuration["positions"], dtype=np.float64),
        cell=np.asarray(configuration["cell"], dtype=np.float64),
        pbc=[bool(value) for value in configuration["pbc"]],
    )
    config = mace_data.config_from_atoms(atoms, head_name=head)
    table = AtomicNumberTable([int(value) for value in model.atomic_numbers.tolist()])
    item = mace_data.AtomicData.from_config(
        config, z_table=table, cutoff=float(model.r_max), heads=heads
    )
    discovered = int(item.edge_index.shape[1])
    if discovered > edges:
        raise _refuse(
            f"The provider neighbor list holds {discovered} edges, above the {edges} "
            "admitted before discovery."
        )
    return torch_geometric.batch.Batch.from_data_list([item]).to_dict()


def _evaluate_case(
    model: Any, configuration: Mapping[str, Any], head: str, edges: int, /
) -> dict[str, np.ndarray]:
    periodic = all(bool(value) for value in configuration["pbc"])
    batch = _batch(model, configuration, head, edges)
    output = model(batch, training=False, compute_force=True, compute_stress=periodic)
    result = {
        "energy": _array(output["energy"]).reshape(()),
        "forces": _array(output["forces"]),
        "node_energy": _array(output["node_energy"]),
        "edge_index": _array(batch["edge_index"]).astype(np.int64),
        "unit_shifts": _array(batch["unit_shifts"]),
    }
    if periodic:
        result["stress"] = _array(output["stress"]).reshape(3, 3)
    return result


def _parameter_dtype(model: Any, /) -> Any:
    import torch

    dtypes = {parameter.dtype for parameter in model.parameters()}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.float32, torch.float64):
        raise _refuse("Source parameters must share float32 or float64.")
    return next(iter(dtypes))


def _legacy_zbl_owners(
    source: Mapping[str, Any], rebuilt: Mapping[str, Any], /
) -> tuple[str, ...]:
    """ZBL modules of an earlier release that carry the unread cutoff child."""

    owners = []
    for name, module in source.items():
        child = f"{name}.cutoff" if name else "cutoff"
        if (
            _type_key(module) == _ZBL_TYPE
            and _type_key(rebuilt.get(name)) == _ZBL_TYPE
            and child in source
            and child not in rebuilt
        ):
            if _type_key(source[child]) != _CUTOFF_TYPE or any(
                True for _ in source[child].children()
            ):
                raise _refuse(f"Source ZBL child {child} is not the legacy cutoff.")
            owners.append(name)
    return tuple(owners)


def _zero_flag_target(name: str, /) -> str:
    owner, flag = name.rsplit(".", 1)
    if flag == "weights_max_zeroed":
        return f"{owner}.weights_max"
    return f"{owner}.weights.{flag.removeprefix('weights_').removesuffix('_zeroed')}"


def _canonical_state(
    model: Any, rebuilt: Any, /
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Express a loaded source in the exact layout of the admitted provider.

    Earlier mace-torch releases pickle the same standard modules with a few
    layout differences the admitted forward pass does not read. Exactly these
    are normalized, each checked against the source and recorded:

    - an absent ``heads`` attribute is the provider's single legacy head;
    - legacy ZBL ``cutoff``/``r_max`` state, consistent with ``p``, is dropped;
    - the integral cutoff power ``p`` stored as a float becomes the provider's
      integer buffer of the same value;
    - absent symmetric-contraction ``*_zeroed`` flags are the source's own
      truth: a weight is zeroed only when it is the provider's ``EmptyParam``.

    Every other module path and type, tensor name, shape and dtype must match
    exactly. The caller then proves source equivalence by reproducing source
    outputs.
    """

    import torch
    from mace.modules.symmetric_contraction import EmptyParam

    source_modules = dict(model.named_modules())
    rebuilt_modules = dict(rebuilt.named_modules())
    owners = _legacy_zbl_owners(source_modules, rebuilt_modules)
    inert_modules = {f"{owner}.cutoff" if owner else "cutoff" for owner in owners}
    # Paths and exact types must agree; registration order is incidental
    # (releases register the same submodules in different orders).
    source_tree = {
        name: _type_key(module)
        for name, module in source_modules.items()
        if name not in inert_modules
    }
    rebuilt_tree = {name: _type_key(module) for name, module in rebuilt_modules.items()}
    if source_tree != rebuilt_tree:
        difference = sorted(
            (
                (name, source_tree.get(name), rebuilt_tree.get(name))
                for name in set(source_tree) | set(rebuilt_tree)
                if source_tree.get(name) != rebuilt_tree.get(name)
            ),
            key=str,
        )[:4]
        raise _refuse(
            f"The source module tree differs from its declared architecture: {difference}."
        )
    state = model.state_dict()
    expected = rebuilt.state_dict()
    inert_state = set()
    for owner in owners:
        prefix = f"{owner}." if owner else ""
        present = [name for name in _LEGACY_ZBL_STATE if prefix + name in state]
        if {"cutoff.r_max", "cutoff.p"} - set(present):
            raise _refuse(f"Legacy ZBL cutoff state of {owner} is incomplete.")
        if not torch.equal(
            state[prefix + "cutoff.p"].double(), state[prefix + "p"].double()
        ) or (
            "r_max" in present
            and not torch.equal(
                state[prefix + "r_max"].double(), state[prefix + "cutoff.r_max"].double()
            )
        ):
            raise _refuse(f"Legacy ZBL cutoff state of {owner} is inconsistent.")
        inert_state.update(prefix + name for name in present)
    extra = set(state) - set(expected) - inert_state
    if extra:
        raise _refuse(
            f"Source tensors outside the declared architecture: {sorted(extra)}."
        )
    canonical: dict[str, Any] = {}
    integral, derived = [], []
    for name, reference in expected.items():
        value = state.get(name)
        owner_name, leaf = name.rsplit(".", 1) if "." in name else ("", name)
        owner = rebuilt_modules.get(owner_name)
        if value is None:
            if _type_key(owner) != _CONTRACTION_TYPE or not leaf.endswith("_zeroed"):
                raise _refuse(f"Source lacks the declared tensor {name}.")
            target = model.get_parameter(_zero_flag_target(name))
            canonical[name] = torch.tensor(
                isinstance(target, EmptyParam), dtype=torch.bool
            )
            derived.append(name)
            continue
        if value.shape != reference.shape:
            raise _refuse(f"Source tensor {name} has a different shape.")
        if value.dtype != reference.dtype:
            integral_value = value.is_floating_point() and torch.equal(
                value, torch.round(value)
            )
            if (
                leaf != "p"
                or _type_key(owner) not in (_ZBL_TYPE, _CUTOFF_TYPE)
                or reference.is_floating_point()
                or not integral_value
            ):
                raise _refuse(f"Source tensor {name} has a different dtype.")
            value = value.to(reference.dtype)
            integral.append(name)
        canonical[name] = value
    record = {
        "legacy_head": not hasattr(model, "heads"),
        # The stored U basis, not the construction attribute, is declared.
        "relabelled_reduced_cg": hasattr(model, "use_reduced_cg")
        and bool(model.use_reduced_cg) != bool(rebuilt.use_reduced_cg),
        "dropped_inert_state": sorted(inert_state),
        "integral_buffers": sorted(integral),
        "derived_zero_flags": sorted(derived),
        # Executed as stored; the native factory consumes these exact buffers.
        "reparameterized_u_buffers": _reparameterized_u(
            model, bool(rebuilt.use_reduced_cg)
        ),
    }
    return canonical, record


def _exact_precision_buffers(model: Any, dtype: Any, /) -> list[str]:
    """Floating buffers stored at another precision than the parameters.

    The provider evaluates a source at one precision (its calculator casts
    the whole module). Such buffers are admitted only when that cast is exact.
    """

    import torch

    names = []
    for name, buffer in model.named_buffers():
        if not buffer.is_floating_point() or buffer.dtype == dtype:
            continue
        if not torch.equal(buffer.to(dtype).to(buffer.dtype), buffer):
            raise _refuse(f"Source buffer {name} is not exact at the source precision.")
        names.append(name)
    return sorted(names)


def _verify_declaration(
    model: Any, declaration: Mapping[str, Any], limits: _Limits, /
) -> dict[str, Any]:
    import torch

    dtype = _parameter_dtype(model)
    precision_buffers = _exact_precision_buffers(model, dtype)
    model.to(dtype)
    rebuilt = _build(declaration, dtype, limits)
    state, layout = _canonical_state(model, rebuilt)
    layout["precision_buffers"] = precision_buffers
    if _state_dtype(state) != dtype:
        raise _refuse("Source buffers and parameters disagree on precision.")
    rebuilt.load_state_dict(state, strict=True)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    # Provider autograd scatter accumulation is not bitwise repeatable (the same
    # model differs from itself at the last ulps); topology must match exactly
    # and values within a fixed multiple of the source precision.
    epsilon = float(torch.finfo(dtype).eps)
    deviation = 0.0
    try:
        head = _heads(model)[0]
        cutoff = float(model.r_max)
        for index, configuration in enumerate(_probe_configurations(declaration)):
            edges = _admitted_edges(
                configuration, cutoff, limits, f"probe configuration {index}"
            )
            expected = _evaluate_case(model, configuration, head, edges)
            actual = _evaluate_case(rebuilt, configuration, head, edges)
            for name, value in expected.items():
                if value.dtype.kind in "iub":
                    if not np.array_equal(value, actual[name]):
                        raise _refuse(f"Declared architecture changes source {name}.")
                    continue
                scale = max(1.0, float(np.max(np.abs(value), initial=0.0)))
                difference = float(np.max(np.abs(value - actual[name]), initial=0.0))
                deviation = max(deviation, difference / scale)
                if difference > _REPRODUCTION_ULPS * epsilon * scale:
                    raise _refuse(
                        f"Declared architecture does not reproduce source {name}."
                    )
    finally:
        torch.set_default_dtype(previous)
    return {
        "source_dtype": str(dtype).removeprefix("torch."),
        "state": state,
        "layout": layout,
        "reproduction_deviation": deviation,
    }


def _fibonacci_directions(count: int, /) -> np.ndarray:
    index = np.arange(count, dtype=np.float64) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = math.pi * (1.0 + math.sqrt(5.0)) * index
    return np.stack(
        (
            np.cos(azimuth) * np.sin(polar),
            np.sin(azimuth) * np.sin(polar),
            np.cos(polar),
        ),
        axis=1,
    )


def _basis_evidence(
    model: Any, declaration: Mapping[str, Any], limits: _Limits, /
) -> dict[str, Any]:
    import torch
    from e3nn import o3

    hidden_degree = o3.Irreps(declaration["hidden_irreps"]).lmax
    degree = max(int(declaration["max_ell"]), hidden_degree)
    paths: set[tuple[int, int, int]] = set()
    for interaction in model.interactions:
        tensor_product = interaction.conv_tp
        for instruction in tensor_product.instructions:
            paths.add(
                (
                    tensor_product.irreps_in1[instruction.i_in1].ir.l,
                    tensor_product.irreps_in2[instruction.i_in2].ir.l,
                    tensor_product.irreps_out[instruction.i_out].ir.l,
                )
            )
    # Degree-proportional evidence is admitted from its shapes before it exists.
    real = np.dtype(np.float64)
    _planned_members(
        {
            "basis/directions": (real, (_FIBONACCI_DIRECTIONS, 3)),
            "basis/spherical_harmonics": (
                real,
                (_FIBONACCI_DIRECTIONS, (degree + 1) ** 2),
            ),
            **{
                f"basis/wigner_3j/{l1}_{l2}_{l3}": (
                    real,
                    (2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1),
                )
                for l1, l2, l3 in paths
            },
        },
        limits,
    )
    directions = _fibonacci_directions(_FIBONACCI_DIRECTIONS)
    harmonics = o3.spherical_harmonics(
        list(range(degree + 1)),
        torch.as_tensor(directions, dtype=torch.float64),
        normalize=True,
        normalization="component",
    )
    arrays: dict[str, Any] = {
        "basis/directions": directions,
        "basis/spherical_harmonics": harmonics,
    }
    for l1, l2, l3 in sorted(paths):
        arrays[f"basis/wigner_3j/{l1}_{l2}_{l3}"] = o3.wigner_3j(
            l1, l2, l3, dtype=torch.float64
        )
    return arrays


def _module_records(model: Any, /) -> dict[str, Any]:
    return {
        "node_embedding": _linear(model.node_embedding.linear),
        "interactions": [_interaction_record(item) for item in model.interactions],
        "products": [_product_record(item) for item in model.products],
        "readouts": [_readout_record(item) for item in model.readouts],
    }


def _configurations(request: Mapping[str, Any], /) -> list[dict[str, Any]]:
    configurations = request.get("configurations")
    if not isinstance(configurations, Sequence) or not configurations:
        raise _refuse("Provider evaluation requires at least one configuration.")
    parsed = []
    for item in configurations:
        positions = np.asarray(item["positions"], dtype=np.float64)
        numbers = [int(value) for value in item["numbers"]]
        cell = np.asarray(item["cell"], dtype=np.float64)
        pbc = [bool(value) for value in item["pbc"]]
        if positions.shape != (len(numbers), 3) or cell.shape != (3, 3) or len(pbc) != 3:
            raise _refuse("Provider configuration shapes are inconsistent.")
        if any(pbc) and not all(pbc):
            raise _refuse("Partially periodic provider configurations are not admitted.")
        parsed.append(
            {"numbers": numbers, "positions": positions, "cell": cell, "pbc": pbc}
        )
    return parsed


def _evaluation_dtype(request: Mapping[str, Any], /) -> Any:
    import torch

    match request.get("evaluation_dtype"):
        case "float32":
            return torch.float32
        case "float64":
            return torch.float64
        case other:
            raise _refuse(f"Unsupported evaluation dtype {other!r}.")


def _run_evaluate(
    model: Any, request: Mapping[str, Any], limits: _Limits, /
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    dtype = _evaluation_dtype(request)
    head = str(request["head"])
    configurations = _configurations(request)
    cutoff = float(model.r_max)
    # Energy, forces, node energies, neighbor topology and (periodic) stress of
    # every case, admitted with each neighbor list's bound before discovery.
    real = torch.empty(0, dtype=dtype).numpy().dtype
    edges = []
    layouts: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {}
    for index, item in enumerate(configurations):
        count = _admitted_edges(item, cutoff, limits, f"configuration {index}")
        atoms = len(item["numbers"])
        prefix = f"cases/{index:04d}/"
        layouts.update(
            {
                prefix + "energy": (real, ()),
                prefix + "forces": (real, (atoms, 3)),
                prefix + "node_energy": (real, (atoms,)),
                prefix + "edge_index": (np.dtype(np.int64), (2, count)),
                prefix + "unit_shifts": (real, (count, 3)),
            }
        )
        if all(item["pbc"]):
            layouts[prefix + "stress"] = (real, (3, 3))
        edges.append(count)
    planned = _planned_members(layouts, limits)
    _admit_result_plan(len(planned) + 1, _member_bytes(planned), limits)
    model = model.to(dtype)
    torch.set_default_dtype(dtype)
    arrays: dict[str, Any] = {}
    for index, configuration in enumerate(configurations):
        evaluated = _evaluate_case(model, configuration, head, edges[index])
        for name, value in evaluated.items():
            arrays[f"cases/{index:04d}/{name}"] = value
    return {"head": head, "evaluation_dtype": str(dtype).removeprefix("torch.")}, arrays


def _run_gradients(
    model: Any, request: Mapping[str, Any], limits: _Limits, /
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    dtype = _evaluation_dtype(request)
    head = str(request["head"])
    model = model.to(dtype)
    torch.set_default_dtype(dtype)
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named]
    weights = request.get("force_weights")
    configurations = _configurations(request)
    if not isinstance(weights, Sequence) or len(weights) != len(configurations):
        raise _refuse("Every gradient configuration requires force weights.")
    # Two parameter gradients per case are retained until the archive is written.
    _admit_result_plan(
        1 + len(configurations) * (2 * len(named) + 2),
        2
        * len(configurations)
        * dtype.itemsize
        * sum(item.numel() for item in parameters),
        limits,
    )
    # Each graph the gradients differentiate through is admitted before discovery.
    cutoff = float(model.r_max)
    edges = [
        _admitted_edges(item, cutoff, limits, f"configuration {index}")
        for index, item in enumerate(configurations)
    ]
    arrays: dict[str, Any] = {}
    for index, configuration in enumerate(configurations):
        batch = _batch(model, configuration, head, edges[index])
        output = model(batch, training=True, compute_force=True, compute_stress=False)
        force_weight = torch.as_tensor(np.asarray(weights[index]), dtype=dtype)
        if force_weight.shape != output["forces"].shape:
            raise _refuse("Force weights must match the configuration forces.")
        energy_gradient = torch.autograd.grad(
            output["energy"].sum(), parameters, retain_graph=True, allow_unused=True
        )
        force_gradient = torch.autograd.grad(
            (output["forces"] * force_weight).sum(), parameters, allow_unused=True
        )
        for (name, parameter), energy, force in zip(
            named, energy_gradient, force_gradient, strict=True
        ):
            zero = torch.zeros_like(parameter)
            arrays[f"cases/{index:04d}/energy_gradient/{name}"] = (
                zero if energy is None else energy
            )
            arrays[f"cases/{index:04d}/force_gradient/{name}"] = (
                zero if force is None else force
            )
        arrays[f"cases/{index:04d}/energy"] = _array(output["energy"]).reshape(())
        arrays[f"cases/{index:04d}/forces"] = output["forces"]
    return {
        "head": head,
        "evaluation_dtype": str(dtype).removeprefix("torch."),
        "parameters": [name for name, _ in named],
    }, arrays


def _run_fixture(
    request: Mapping[str, Any], output: Path, limits: _Limits, /
) -> dict[str, Any]:
    import torch

    declaration = request.get("architecture")
    seed = request.get("seed")
    kind = request.get("source_kind")
    if not isinstance(declaration, Mapping) or type(seed) is not int:
        raise _refuse("A provider fixture requires a declaration and integer seed.")
    match request.get("source_dtype"):
        case "float32":
            dtype = torch.float32
        case "float64":
            dtype = torch.float64
        case other:
            raise _refuse(f"Unsupported fixture source dtype {other!r}.")
    torch.manual_seed(seed)
    model = _build(declaration, dtype, limits)
    buffer = io.BytesIO()
    match kind:
        case "torch-state-dict":
            torch.save(model.state_dict(), buffer)
        case "torch-full-model":
            torch.save(model, buffer)
        case other:
            raise _refuse(f"Unsupported fixture source kind {other!r}.")
    payload = buffer.getvalue()
    (output.parent / "fixture.pt").write_bytes(payload)
    return {
        "seed": seed,
        "source_kind": kind,
        "source_dtype": str(dtype).removeprefix("torch."),
        # Exactly what extraction declares for these bytes (reference
        # energies and scales at the source precision).
        "declaration": _declaration(model, _admit_module_tree(model)),
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "fixture_byte_size": len(payload),
    }


def _execute(site_packages: str, request_path: Path, output: Path, /) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, Mapping):
        raise _refuse("Provider request must be a JSON object.")
    limits = _limits(request)
    provider = _provider_identity(site_packages, request["provider"]["distributions"])
    # e3nn 0.4.4 loads its own RECORD-verified constants without an explicit
    # weights_only argument (mace-torch sets this same variable on import).
    # Every source load below passes weights_only explicitly, which this
    # variable cannot override.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    import torch

    torch.set_grad_enabled(True)
    operation = request["operation"]
    manifest: dict[str, Any] = {
        "format": "phydrax-mace-provider-result",
        "operation": operation,
        "provider": provider,
    }
    if operation == "fixture":
        manifest["fixture"] = _run_fixture(request, output, limits)
        _write_archive(output, manifest, {}, limits)
        return
    model, source_kind = _load_source(request, request_path.parent, limits)
    model_class = _admit_module_tree(model)
    declaration = _declaration(model, model_class)
    declared = request.get("architecture")
    if declared is not None and dict(declared) != declaration:
        raise _refuse("The source does not match the declared architecture exactly.")
    verification = _verify_declaration(model, declaration, limits)
    manifest["source"] = {
        "kind": source_kind,
        "sha256": request["source"]["sha256"],
        "byte_size": request["source"]["byte_size"],
        "dtype": verification["source_dtype"],
        "reproduction_deviation": verification["reproduction_deviation"],
        "layout_normalization": verification["layout"],
    }
    manifest["declaration"] = declaration
    match operation:
        case "extract":
            manifest["modules"] = _module_records(model)
            arrays = {
                f"state/{name}": value for name, value in verification["state"].items()
            }
            arrays.update(_basis_evidence(model, declaration, limits))
        case "evaluate":
            manifest["evaluation"], arrays = _run_evaluate(model, request, limits)
        case "gradients":
            manifest["evaluation"], arrays = _run_gradients(model, request, limits)
        case other:
            raise _refuse(f"Unsupported provider operation {other!r}.")
    _write_archive(output, manifest, arrays, limits)


def main(arguments: Sequence[str], /) -> int:
    if len(arguments) != 3:
        sys.stderr.write("usage: worker.py <site-packages> request.json result.zip\n")
        return 2
    site_packages, request_path, output = arguments
    try:
        _execute(site_packages, Path(request_path), Path(output))
    except ProviderRefusal as refusal:
        sys.stderr.write(f"REFUSED: {refusal}\n")
        return _REFUSED_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
