#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded native gauge archives and explicit NERSC/MILC/ILDG-like envelopes."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from math import prod
from pathlib import Path
from typing import Literal, TypeAlias

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike

from ..._array_archive import (
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint, canonical_json
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._lattice_distribution import LatticeDecompositionPlan


GaugeInterchangeKind: TypeAlias = Literal["nersc", "milc", "ildg"]
GaugeByteOrder: TypeAlias = Literal["big", "little"]
GaugeComponentPrecision: TypeAlias = Literal["float32", "float64"]

_MAGIC = {
    "nersc": b"NERQCD01",
    "milc": b"MILQCD01",
    "ildg": b"ILDQCD01",
}


def _file_sha256(path: Path, maximum_bytes: int, /) -> tuple[str, int]:
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ValueError("Gauge archive exceeds its container byte limit.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            chunk = stream.read(min(1_048_576, remaining))
            if not chunk:
                raise ValueError("Gauge archive was truncated while hashing.")
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise ValueError("Gauge archive grew while hashing.")
    return digest.hexdigest(), size


def _unique_header(pairs: list[tuple[str, object]], /) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate gauge interchange header member {name!r}.")
        result[name] = value
    return result


def _reject_json_constant(value: str, /) -> object:
    raise ValueError(f"Non-finite gauge interchange JSON value {value!r}.")


class GaugeIOPlan(StrictModule, NonTrainableState):
    """Fail-before-allocation limits for gauge configuration I/O."""

    archive_limits: ArrayArchiveLimits = eqx.field(static=True)
    maximum_sites: int = eqx.field(static=True)
    maximum_dimension: int = eqx.field(static=True)
    maximum_colors: int = eqx.field(static=True)
    maximum_header_bytes: int = eqx.field(static=True)
    maximum_payload_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        archive_limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
        maximum_sites: int = 16_777_216,
        maximum_dimension: int = 8,
        maximum_colors: int = 16,
        maximum_header_bytes: int = 1_048_576,
        maximum_payload_bytes: int = 4_294_967_296,
    ) -> None:
        if not isinstance(archive_limits, ArrayArchiveLimits):
            raise TypeError("archive_limits must be ArrayArchiveLimits.")
        values = tuple(
            (
                maximum_sites,
                maximum_dimension,
                maximum_colors,
                maximum_header_bytes,
                maximum_payload_bytes,
            )
        )
        if any(value <= 0 for value in values):
            raise ValueError("Gauge I/O limits must be positive.")
        (
            self.maximum_sites,
            self.maximum_dimension,
            self.maximum_colors,
            self.maximum_header_bytes,
            self.maximum_payload_bytes,
        ) = values
        self.archive_limits = archive_limits
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gauge-io-resource-plan",
                "maximum_sites": values[0],
                "maximum_dimension": values[1],
                "maximum_colors": values[2],
                "maximum_header_bytes": values[3],
                "maximum_payload_bytes": values[4],
                "archive_limits": {
                    "container": archive_limits.max_container_bytes,
                    "aggregate": archive_limits.max_aggregate_bytes,
                    "member": archive_limits.max_member_bytes,
                    "members": archive_limits.max_members,
                    "elements": archive_limits.max_total_array_elements,
                },
            }
        )


class GaugeFieldRecord(StrictModule, NonTrainableState):
    """One canonical C-order global gauge field and its exact semantics."""

    links: np.ndarray
    global_shape: tuple[int, ...] = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    boundary_phases: tuple[tuple[float, float], ...] = eqx.field(static=True)
    source_format: str = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    color_count: int = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        links: ArrayLike,
        global_shape: tuple[int, ...],
        representation_id: str,
        /,
        *,
        boundary_phases: tuple[complex, ...] | None = None,
        source_format: str = "native",
        policy: GaugeIOPlan | None = None,
    ) -> None:
        shape = tuple(global_shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("global_shape must contain positive extents.")
        array = np.asarray(links)
        site_count = prod(shape)
        dimension = len(shape)
        if (
            array.ndim != 4
            or array.shape[0] != site_count
            or array.shape[1] != dimension
            or array.shape[2] != array.shape[3]
            or array.shape[2] < 1
        ):
            raise ValueError(
                "links must have shape (prod(global_shape), dimension, colors, colors)."
            )
        if array.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise TypeError("Gauge links must use complex64 or complex128 precision.")
        if np.any(~np.isfinite(array)):
            raise ValueError("Gauge links must be finite.")
        representation = str(representation_id).strip()
        source = str(source_format).strip()
        if not representation or not source:
            raise ValueError("Gauge representation and source format must be non-empty.")
        phases = (
            (1.0 + 0.0j,) * dimension
            if boundary_phases is None
            else tuple(complex(value) for value in boundary_phases)
        )
        if len(phases) != dimension or any(
            not np.isfinite(value.real)
            or not np.isfinite(value.imag)
            or not np.isclose(abs(value), 1.0, rtol=0.0, atol=1.0e-12)
            for value in phases
        ):
            raise ValueError(
                "boundary_phases must contain one finite unit phase per axis."
            )
        plan = GaugeIOPlan() if policy is None else policy
        if not isinstance(plan, GaugeIOPlan):
            raise TypeError("policy must be GaugeIOPlan or None.")
        if (
            site_count > plan.maximum_sites
            or dimension > plan.maximum_dimension
            or array.shape[2] > plan.maximum_colors
            or array.nbytes > plan.maximum_payload_bytes
        ):
            raise ValueError("Gauge field exceeds its I/O resource plan.")
        phase_record = tuple((float(value.real), float(value.imag)) for value in phases)
        contiguous = np.ascontiguousarray(array)
        contiguous.setflags(write=False)
        self.links = contiguous
        self.global_shape = shape
        self.representation_id = representation
        self.boundary_phases = phase_record
        self.source_format = source
        self.site_count = site_count
        self.dimension = dimension
        self.color_count = array.shape[2]
        self.field_id = canonical_fingerprint(
            {
                "kind": "canonical-global-gauge-field",
                "global_shape": shape,
                "representation": representation,
                "boundary_phases": phase_record,
                "source_format": source,
                "links": array_tree_fingerprint(contiguous),
            }
        )

    @property
    def complex_boundary_phases(self) -> tuple[complex, ...]:
        return tuple(complex(real, imag) for real, imag in self.boundary_phases)


class GaugeArchiveEvidence(StrictModule, NonTrainableState):
    field: GaugeFieldRecord
    path: str = eqx.field(static=True)
    container_kind: str = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    byte_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: GaugeFieldRecord,
        path: str | Path,
        container_kind: str,
        checksum: str,
        byte_count: int,
        /,
    ) -> None:
        if not isinstance(field, GaugeFieldRecord):
            raise TypeError("field must be GaugeFieldRecord.")
        path_ = str(Path(path))
        kind = str(container_kind).strip()
        digest = str(checksum).strip()
        size = int(byte_count)
        if not path_ or not kind or len(digest) != 64 or size < 0:
            raise ValueError("Gauge archive evidence is invalid.")
        self.field = field
        self.path = path_
        self.container_kind = kind
        self.checksum = digest
        self.byte_count = size
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "gauge-archive-evidence",
                "field": field.field_id,
                "path": path_,
                "container_kind": kind,
                "checksum": digest,
                "byte_count": size,
            }
        )


def gauge_field_from_owned_shards(
    decomposition: LatticeDecompositionPlan,
    owned_links: ArrayLike,
    representation_id: str,
    /,
    *,
    boundary_phases: tuple[complex, ...] | None = None,
    policy: GaugeIOPlan | None = None,
) -> GaugeFieldRecord:
    """Assemble exactly-once site-owned shards into a partition-independent record."""

    if not isinstance(decomposition, LatticeDecompositionPlan):
        raise TypeError("decomposition must be LatticeDecompositionPlan.")
    values = np.asarray(owned_links)
    expected = (decomposition.partition_count, decomposition.owned_capacity)
    if values.shape[:2] != expected:
        raise ValueError(f"owned_links must begin with shape {expected}.")
    ids = np.asarray(decomposition.owned_global_ids)
    valid = np.asarray(decomposition.owned_valid)
    links = np.zeros(
        (decomposition.site_count,) + values.shape[2:],
        dtype=values.dtype,
    )
    for part in range(decomposition.partition_count):
        links[ids[part, valid[part]]] = values[part, valid[part]]
    return GaugeFieldRecord(
        links,
        decomposition.global_shape,
        representation_id,
        boundary_phases=boundary_phases,
        source_format="native",
        policy=policy,
    )


def write_native_gauge_archive(
    path: str | Path,
    field: GaugeFieldRecord,
    /,
    *,
    policy: GaugeIOPlan | None = None,
) -> GaugeArchiveEvidence:
    if not isinstance(field, GaugeFieldRecord):
        raise TypeError("field must be GaugeFieldRecord.")
    plan = GaugeIOPlan() if policy is None else policy
    if not isinstance(plan, GaugeIOPlan):
        raise TypeError("policy must be GaugeIOPlan or None.")
    array_bytes = field.links.size * field.links.dtype.itemsize
    estimated_member = array_bytes + plan.archive_limits.max_npy_header_bytes
    estimated_container = (
        estimated_member
        + plan.archive_limits.max_manifest_bytes
        + plan.archive_limits.max_central_directory_bytes
        + 65_536
    )
    if (
        array_bytes > plan.maximum_payload_bytes
        or estimated_member > plan.archive_limits.max_member_bytes
        or estimated_container > plan.archive_limits.max_container_bytes
    ):
        raise ValueError("Native gauge archive exceeds its write resource plan.")
    destination = write_array_archive(
        path,
        manifest={
            "kind": "native-global-gauge-field",
            "global_shape": list(field.global_shape),
            "representation_id": field.representation_id,
            "boundary_phases": [list(value) for value in field.boundary_phases],
            "source_format": field.source_format,
            "field_id": field.field_id,
        },
        arrays={"links": field.links},
    )
    checksum, byte_count = _file_sha256(
        destination,
        plan.archive_limits.max_container_bytes,
    )
    return GaugeArchiveEvidence(
        field,
        destination,
        "native",
        checksum,
        byte_count,
    )


def read_native_gauge_archive(
    path: str | Path,
    /,
    *,
    policy: GaugeIOPlan | None = None,
) -> GaugeArchiveEvidence:
    plan = GaugeIOPlan() if policy is None else policy
    if not isinstance(plan, GaugeIOPlan):
        raise TypeError("policy must be GaugeIOPlan or None.")
    manifest, arrays = read_array_archive(path, limits=plan.archive_limits)
    if (
        set(manifest)
        != {
            "kind",
            "global_shape",
            "representation_id",
            "boundary_phases",
            "source_format",
            "field_id",
            "arrays",
        }
        or manifest["kind"] != "native-global-gauge-field"
    ):
        raise ValueError("Archive is not a canonical native global gauge field.")
    if set(arrays) != {"links"}:
        raise ValueError("Native gauge archive must contain exactly one link array.")
    raw_shape = manifest["global_shape"]
    raw_phases = manifest["boundary_phases"]
    if (
        not isinstance(raw_shape, list)
        or any(type(value) is not int for value in raw_shape)
        or not isinstance(raw_phases, list)
        or any(
            not isinstance(value, list)
            or len(value) != 2
            or any(type(component) not in (int, float) for component in value)
            for value in raw_phases
        )
        or not isinstance(manifest["representation_id"], str)
        or not isinstance(manifest["source_format"], str)
        or not isinstance(manifest["field_id"], str)
    ):
        raise ValueError("Native gauge metadata is malformed.")
    field = GaugeFieldRecord(
        arrays["links"],
        tuple(raw_shape),
        manifest["representation_id"],
        boundary_phases=tuple(complex(*value) for value in raw_phases),
        source_format=manifest["source_format"],
        policy=plan,
    )
    if field.field_id != manifest["field_id"]:
        raise ValueError("Native gauge field identity does not match its payload.")
    source = Path(path)
    checksum, byte_count = _file_sha256(
        source,
        plan.archive_limits.max_container_bytes,
    )
    return GaugeArchiveEvidence(
        field,
        source,
        "native",
        checksum,
        byte_count,
    )


def _interchange_field(
    field: GaugeFieldRecord,
    kind: GaugeInterchangeKind,
    precision: GaugeComponentPrecision,
    policy: GaugeIOPlan,
    /,
) -> GaugeFieldRecord:
    complex_dtype = np.complex64 if precision == "float32" else np.complex128
    return GaugeFieldRecord(
        np.asarray(field.links, dtype=complex_dtype),
        field.global_shape,
        field.representation_id,
        boundary_phases=field.complex_boundary_phases,
        source_format=f"{kind}-like",
        policy=policy,
    )


def write_gauge_interchange(
    path: str | Path,
    field: GaugeFieldRecord,
    kind: GaugeInterchangeKind,
    /,
    *,
    byte_order: GaugeByteOrder = "big",
    precision: GaugeComponentPrecision = "float64",
    policy: GaugeIOPlan | None = None,
) -> GaugeArchiveEvidence:
    """Write a self-describing standards-like envelope without claiming conformance."""

    if not isinstance(field, GaugeFieldRecord):
        raise TypeError("field must be GaugeFieldRecord.")
    if kind not in _MAGIC:
        raise ValueError("kind must be nersc, milc, or ildg.")
    if byte_order not in ("big", "little"):
        raise ValueError("byte_order must be big or little.")
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be float32 or float64.")
    plan = GaugeIOPlan() if policy is None else policy
    if not isinstance(plan, GaugeIOPlan):
        raise TypeError("policy must be GaugeIOPlan or None.")
    payload_field = _interchange_field(field, kind, precision, plan)
    component_dtype = np.dtype(
        (">" if byte_order == "big" else "<") + ("f4" if precision == "float32" else "f8")
    )
    links = np.asarray(payload_field.links)
    expected_payload_bytes = links.size * 2 * component_dtype.itemsize
    if expected_payload_bytes > plan.maximum_payload_bytes:
        raise ValueError("Gauge interchange payload exceeds maximum_payload_bytes.")
    components = np.stack((links.real, links.imag), axis=-1).astype(
        component_dtype, copy=False
    )
    payload = np.ascontiguousarray(components).tobytes(order="C")
    if len(payload) != expected_payload_bytes:
        raise ValueError("Gauge interchange payload byte count changed during encoding.")
    checksum = hashlib.sha256(payload).hexdigest()
    header_record = {
        "kind": f"{kind}-like-global-gauge-field",
        "global_shape": list(payload_field.global_shape),
        "representation_id": payload_field.representation_id,
        "boundary_phases": [list(value) for value in payload_field.boundary_phases],
        "byte_order": byte_order,
        "component_precision": precision,
        "color_count": payload_field.color_count,
        "payload_bytes": len(payload),
        "payload_sha256": checksum,
        "payload_field_id": payload_field.field_id,
        "source_field_id": field.field_id,
    }
    header = canonical_json(header_record).encode("utf-8")
    if len(header) > plan.maximum_header_bytes:
        raise ValueError("Gauge interchange header exceeds maximum_header_bytes.")
    total_size = 12 + len(header) + len(payload)
    if total_size > plan.archive_limits.max_container_bytes:
        raise ValueError("Gauge interchange exceeds the container byte limit.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as stream:
        stream.write(_MAGIC[kind])
        stream.write(len(header).to_bytes(4, "big"))
        stream.write(header)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(destination)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    os.fsync(directory_descriptor)
    os.close(directory_descriptor)
    total_size = destination.stat().st_size
    return GaugeArchiveEvidence(
        payload_field,
        destination,
        f"{kind}-like",
        checksum,
        total_size,
    )


def read_gauge_interchange(
    path: str | Path,
    /,
    *,
    expected_kind: GaugeInterchangeKind | None = None,
    expected_byte_order: GaugeByteOrder | None = None,
    expected_precision: GaugeComponentPrecision | None = None,
    policy: GaugeIOPlan | None = None,
) -> GaugeArchiveEvidence:
    plan = GaugeIOPlan() if policy is None else policy
    if not isinstance(plan, GaugeIOPlan):
        raise TypeError("policy must be GaugeIOPlan or None.")
    if expected_kind is not None and expected_kind not in _MAGIC:
        raise ValueError("expected_kind must be nersc, milc, ildg, or None.")
    if expected_byte_order is not None and expected_byte_order not in ("big", "little"):
        raise ValueError("expected_byte_order must be big, little, or None.")
    if expected_precision is not None and expected_precision not in (
        "float32",
        "float64",
    ):
        raise ValueError("expected_precision must be float32, float64, or None.")
    source = Path(path)
    container_size = source.stat().st_size
    maximum_container = min(
        plan.archive_limits.max_container_bytes,
        plan.maximum_header_bytes + plan.maximum_payload_bytes + 12,
    )
    if container_size < 12 or container_size > maximum_container:
        raise ValueError("Gauge interchange container size is outside policy.")
    with source.open("rb") as stream:
        magic = stream.read(8)
        kinds = tuple(kind for kind, value in _MAGIC.items() if value == magic)
        if len(kinds) != 1:
            raise ValueError("Gauge interchange magic is not recognized.")
        kind = kinds[0]
        if expected_kind is not None and kind != expected_kind:
            raise ValueError("Gauge interchange kind differs from the required kind.")
        header_size_bytes = stream.read(4)
        if len(header_size_bytes) != 4:
            raise ValueError("Gauge interchange header length is truncated.")
        header_size = int.from_bytes(header_size_bytes, "big")
        if header_size <= 0 or header_size > plan.maximum_header_bytes:
            raise ValueError("Gauge interchange header length is outside policy.")
        header_bytes = stream.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError("Gauge interchange header is truncated.")
        payload = stream.read(plan.maximum_payload_bytes + 1)
    if len(payload) > plan.maximum_payload_bytes:
        raise ValueError("Gauge interchange payload exceeds maximum_payload_bytes.")
    header = json.loads(
        header_bytes.decode("utf-8"),
        object_pairs_hook=_unique_header,
        parse_constant=_reject_json_constant,
    )
    if canonical_json(header).encode("utf-8") != header_bytes:
        raise ValueError("Gauge interchange header is not canonical JSON.")
    expected_fields = {
        "kind",
        "global_shape",
        "representation_id",
        "boundary_phases",
        "byte_order",
        "component_precision",
        "color_count",
        "payload_bytes",
        "payload_sha256",
        "payload_field_id",
        "source_field_id",
    }
    if not isinstance(header, dict) or set(header) != expected_fields:
        raise ValueError("Gauge interchange header is not canonical.")
    if header["kind"] != f"{kind}-like-global-gauge-field":
        raise ValueError("Gauge interchange header kind and magic disagree.")
    byte_order = header["byte_order"]
    precision = header["component_precision"]
    if byte_order not in ("big", "little") or precision not in ("float32", "float64"):
        raise ValueError("Gauge interchange endian or precision metadata is invalid.")
    if expected_byte_order is not None and byte_order != expected_byte_order:
        raise ValueError("Gauge interchange byte order differs from the required order.")
    if expected_precision is not None and precision != expected_precision:
        raise ValueError(
            "Gauge interchange precision differs from the required precision."
        )
    raw_shape = header["global_shape"]
    raw_phases = header["boundary_phases"]
    color_count = header["color_count"]
    payload_bytes = header["payload_bytes"]
    if (
        not isinstance(raw_shape, list)
        or not raw_shape
        or any(type(value) is not int or value <= 0 for value in raw_shape)
        or not isinstance(color_count, int)
        or isinstance(color_count, bool)
        or color_count <= 0
        or not isinstance(payload_bytes, int)
        or isinstance(payload_bytes, bool)
        or payload_bytes < 0
        or not isinstance(header["representation_id"], str)
        or not isinstance(header["payload_sha256"], str)
        or not isinstance(header["payload_field_id"], str)
        or not isinstance(header["source_field_id"], str)
        or not isinstance(raw_phases, list)
        or any(
            not isinstance(value, list)
            or len(value) != 2
            or any(type(component) not in (int, float) for component in value)
            for value in raw_phases
        )
    ):
        raise ValueError("Gauge interchange shape or identity metadata is invalid.")
    site_count = prod(raw_shape)
    dimension = len(raw_shape)
    component_bytes = 4 if precision == "float32" else 8
    expected_payload_bytes = (
        site_count * dimension * color_count * color_count * 2 * component_bytes
    )
    if (
        site_count > plan.maximum_sites
        or dimension > plan.maximum_dimension
        or color_count > plan.maximum_colors
        or payload_bytes != expected_payload_bytes
        or len(payload) != expected_payload_bytes
    ):
        raise ValueError("Gauge interchange payload shape or byte size is inconsistent.")
    observed_checksum = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(observed_checksum, header["payload_sha256"]):
        raise ValueError("Gauge interchange payload checksum failed.")
    component_dtype = np.dtype(
        (">" if byte_order == "big" else "<") + ("f4" if precision == "float32" else "f8")
    )
    components = np.frombuffer(payload, dtype=component_dtype).reshape(
        (site_count, dimension, color_count, color_count, 2)
    )
    links = components[..., 0] + 1j * components[..., 1]
    field = GaugeFieldRecord(
        links,
        tuple(raw_shape),
        header["representation_id"],
        boundary_phases=tuple(complex(*value) for value in raw_phases),
        source_format=f"{kind}-like",
        policy=plan,
    )
    if field.field_id != header["payload_field_id"]:
        raise ValueError("Gauge interchange field identity does not match its payload.")
    return GaugeArchiveEvidence(
        field,
        source,
        f"{kind}-like",
        observed_checksum,
        container_size,
    )


__all__ = [
    "GaugeArchiveEvidence",
    "GaugeByteOrder",
    "GaugeComponentPrecision",
    "GaugeFieldRecord",
    "GaugeInterchangeKind",
    "GaugeIOPlan",
    "gauge_field_from_owned_shards",
    "read_gauge_interchange",
    "read_native_gauge_archive",
    "write_gauge_interchange",
    "write_native_gauge_archive",
]
