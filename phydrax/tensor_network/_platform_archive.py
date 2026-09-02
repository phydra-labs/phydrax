#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import os
import zipfile
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._array_archive import (
    array_collection_digest,
    array_payload_byte_count,
    ArrayArchiveCorruptionError,
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._fingerprint import canonical_fingerprint, canonical_json
from .._strict import StrictModule
from .._tensor_network_precision import TensorNetworkPrecisionPolicy
from .._trainable import NonTrainableState
from ._core import LocallyPurifiedDensity, MatrixProductOperator, MatrixProductState
from ._platform_support import _identifier, _positive_integer, TensorNetworkFailure


class TensorNetworkArchiveKind(StrEnum):
    MPS = "mps"
    MPO = "mpo"
    LPDO = "lpdo"
    ARRAY_PYTREE = "array-pytree"


class TensorNetworkArchiveLimits(StrictModule, NonTrainableState):
    maximum_archive_bytes: int = eqx.field(static=True)
    maximum_manifest_bytes: int = eqx.field(static=True)
    maximum_members: int = eqx.field(static=True)
    maximum_array_bytes: int = eqx.field(static=True)
    maximum_total_array_bytes: int = eqx.field(static=True)
    maximum_array_rank: int = eqx.field(static=True)
    maximum_array_elements: int = eqx.field(static=True)
    limits_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_archive_bytes: int = 4 * 1024**3,
        maximum_manifest_bytes: int = 16 * 1024**2,
        maximum_members: int = 100_001,
        maximum_array_bytes: int = 2 * 1024**3,
        maximum_total_array_bytes: int = 4 * 1024**3,
        maximum_array_rank: int = 16,
        maximum_array_elements: int = 1_000_000_000,
    ):
        values = tuple(
            _positive_integer(value, name)
            for value, name in (
                (maximum_archive_bytes, "maximum_archive_bytes"),
                (maximum_manifest_bytes, "maximum_manifest_bytes"),
                (maximum_members, "maximum_members"),
                (maximum_array_bytes, "maximum_array_bytes"),
                (maximum_total_array_bytes, "maximum_total_array_bytes"),
                (maximum_array_rank, "maximum_array_rank"),
                (maximum_array_elements, "maximum_array_elements"),
            )
        )
        (
            self.maximum_archive_bytes,
            self.maximum_manifest_bytes,
            self.maximum_members,
            self.maximum_array_bytes,
            self.maximum_total_array_bytes,
            self.maximum_array_rank,
            self.maximum_array_elements,
        ) = values
        self.limits_id = canonical_fingerprint(
            {"kind": "tensor-network-archive-limits", "values": values}
        )


class TensorNetworkArchiveError(RuntimeError):
    failure: TensorNetworkFailure

    def __init__(self, failure: TensorNetworkFailure, detail: str, /):
        failure_ = TensorNetworkFailure(failure)
        detail_ = _identifier(detail, "archive failure detail")
        if failure_ == TensorNetworkFailure.NONE:
            raise ValueError("Archive errors require a failure category.")
        self.failure = failure_
        self.detail = detail_
        super().__init__(f"{failure_.value}: {detail_}")


class TensorNetworkArchiveCorruptionError(TensorNetworkArchiveError):
    def __init__(self, detail: str, /):
        super().__init__(TensorNetworkFailure.ARCHIVE_CORRUPTION, detail)


class TensorNetworkArchiveMismatchError(TensorNetworkArchiveError):
    def __init__(self, detail: str, /):
        super().__init__(TensorNetworkFailure.ARCHIVE_MISMATCH, detail)


class TensorNetworkArchiveSecurityError(TensorNetworkArchiveError):
    def __init__(self, detail: str, /):
        super().__init__(TensorNetworkFailure.SECURITY_LIMIT, detail)


class TensorNetworkArchiveRecord(StrictModule, NonTrainableState):
    artifact_kind: TensorNetworkArchiveKind = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    leaf_count: int = eqx.field(static=True)
    payload_bytes: int = eqx.field(static=True)
    array_digest: str = eqx.field(static=True)

    def __init__(
        self,
        artifact_kind: TensorNetworkArchiveKind,
        artifact_id: str,
        structure_id: str,
        precision_policy_id: str,
        source_id: str,
        leaf_count: int,
        payload_bytes: int,
        array_digest: str,
        /,
    ):
        kind = TensorNetworkArchiveKind(artifact_kind)
        identifiers = tuple(
            _identifier(value, name)
            for value, name in (
                (artifact_id, "artifact_id"),
                (structure_id, "structure_id"),
                (precision_policy_id, "precision_policy_id"),
                (source_id, "source_id"),
                (array_digest, "array_digest"),
            )
        )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (identifiers[0], identifiers[4])
        ):
            raise ValueError("Archive content addresses must be lowercase SHA-256.")
        leaves = _positive_integer(leaf_count, "leaf_count")
        payload = _positive_integer(payload_bytes, "payload_bytes")
        self.artifact_kind = kind
        self.artifact_id = identifiers[0]
        self.structure_id = identifiers[1]
        self.precision_policy_id = identifiers[2]
        self.source_id = identifiers[3]
        self.leaf_count = leaves
        self.payload_bytes = payload
        self.array_digest = identifiers[4]


class TensorNetworkArchiveValidation(StrictModule, NonTrainableState):
    record: TensorNetworkArchiveRecord
    limits_id: str = eqx.field(static=True)
    archive_bytes: int = eqx.field(static=True)
    member_count: int = eqx.field(static=True)
    valid: bool = eqx.field(static=True)
    validation_id: str = eqx.field(static=True)

    def __init__(
        self,
        record: TensorNetworkArchiveRecord,
        limits_id: str,
        archive_bytes: int,
        member_count: int,
        valid: bool,
        /,
    ):
        if not isinstance(record, TensorNetworkArchiveRecord):
            raise TypeError("record must be TensorNetworkArchiveRecord.")
        limits = _identifier(limits_id, "limits_id")
        archive_bytes_ = _positive_integer(archive_bytes, "archive_bytes")
        members = _positive_integer(member_count, "member_count")
        if not bool(valid):
            raise ValueError(
                "Archive validation records represent successful validation."
            )
        self.record = record
        self.limits_id = limits
        self.archive_bytes = archive_bytes_
        self.member_count = members
        self.valid = True
        self.validation_id = canonical_fingerprint(
            {
                "kind": "tensor-network-archive-validation",
                "artifact": record.artifact_id,
                "limits": limits,
                "archive_bytes": archive_bytes_,
                "member_count": members,
                "valid": True,
            }
        )


class TensorNetworkArchivedArtifact(StrictModule):
    value: Any
    record: TensorNetworkArchiveRecord
    validation: TensorNetworkArchiveValidation

    def __init__(
        self,
        value: Any,
        record: TensorNetworkArchiveRecord,
        validation: TensorNetworkArchiveValidation,
        /,
    ):
        if not isinstance(record, TensorNetworkArchiveRecord) or not isinstance(
            validation, TensorNetworkArchiveValidation
        ):
            raise TypeError("Archived artifacts require record and validation.")
        if validation.record.artifact_id != record.artifact_id:
            raise ValueError("Artifact record and validation identities differ.")
        self.value = value
        self.record = record
        self.validation = validation


def _limits(value: TensorNetworkArchiveLimits | None, /) -> TensorNetworkArchiveLimits:
    if value is None:
        return TensorNetworkArchiveLimits()
    if not isinstance(value, TensorNetworkArchiveLimits):
        raise TypeError("limits must be TensorNetworkArchiveLimits or None.")
    return value


def _precision_record(policy: TensorNetworkPrecisionPolicy, /) -> dict[str, str | None]:
    return {
        "storage_dtype": policy.storage_dtype,
        "contraction_dtype": policy.contraction_dtype,
        "factorization_dtype": policy.factorization_dtype,
        "accumulation_dtype": policy.accumulation_dtype,
        "decision_dtype": policy.decision_dtype,
        "output_dtype": policy.output_dtype,
    }


def _precision_from_record(
    record: object, policy_id: str, /
) -> TensorNetworkPrecisionPolicy:
    fields = {
        "storage_dtype",
        "contraction_dtype",
        "factorization_dtype",
        "accumulation_dtype",
        "decision_dtype",
        "output_dtype",
    }
    if not isinstance(record, Mapping) or set(record) != fields:
        raise TensorNetworkArchiveCorruptionError(
            "archive precision policy record is not canonical"
        )
    try:
        policy = TensorNetworkPrecisionPolicy(**dict(record))
    except (TypeError, ValueError) as error:
        raise TensorNetworkArchiveCorruptionError(
            "archive precision policy values are invalid"
        ) from error
    if policy.policy_id != policy_id:
        raise TensorNetworkArchiveCorruptionError(
            "archive precision policy content address is invalid"
        )
    return policy


def _array_only_tree(value: Any, /) -> tuple[Any, tuple[np.ndarray, ...]]:
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(value)
    if not path_leaves:
        raise TypeError("Archived array PyTrees must be nonempty.")
    arrays: list[np.ndarray] = []
    for path, leaf in path_leaves:
        array = np.asarray(leaf)
        if array.dtype.hasobject or array.dtype.kind not in "biufc":
            location = jax.tree_util.keystr(path) or "<root>"
            raise TypeError(f"Archive leaf {location} must be a numerical array.")
        arrays.append(array)
    return value, tuple(arrays)


def _artifact_tree(
    value: Any, kind: TensorNetworkArchiveKind, /
) -> tuple[Any, TensorNetworkPrecisionPolicy | None, str | None]:
    if kind == TensorNetworkArchiveKind.MPS:
        if not isinstance(value, MatrixProductState):
            raise TypeError("MPS archives require MatrixProductState values.")
        return value.tensors, value.precision, value.structure_id
    if kind == TensorNetworkArchiveKind.MPO:
        if not isinstance(value, MatrixProductOperator):
            raise TypeError("MPO archives require MatrixProductOperator values.")
        return value.tensors, value.precision, value.structure_id
    if kind == TensorNetworkArchiveKind.LPDO:
        if not isinstance(value, LocallyPurifiedDensity):
            raise TypeError("LPDO archives require LocallyPurifiedDensity values.")
        return value.tensors, value.precision, value.structure_id
    tree, _ = _array_only_tree(value)
    return tree, None, None


def _check_array_bounds(
    arrays: Mapping[str, Any], limits: TensorNetworkArchiveLimits, /
) -> tuple[int, int]:
    if not arrays:
        raise TensorNetworkArchiveCorruptionError("archive contains no numerical arrays")
    if len(arrays) + 1 > limits.maximum_members:
        raise TensorNetworkArchiveSecurityError("archive member count exceeds capacity")
    total_payload = 0
    total_elements = 0
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise TensorNetworkArchiveCorruptionError("archive array name is invalid")
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind not in "biufc":
            raise TensorNetworkArchiveCorruptionError(
                f"archive array {name!r} is not numerical"
            )
        if array.ndim > limits.maximum_array_rank:
            raise TensorNetworkArchiveSecurityError(
                f"archive array {name!r} rank exceeds capacity"
            )
        elements = int(array.size)
        payload_bytes = array_payload_byte_count(array)
        if elements > limits.maximum_array_elements:
            raise TensorNetworkArchiveSecurityError(
                f"archive array {name!r} element count exceeds capacity"
            )
        if payload_bytes > limits.maximum_array_bytes:
            raise TensorNetworkArchiveSecurityError(
                f"archive array {name!r} byte count exceeds capacity"
            )
        total_elements += elements
        total_payload += payload_bytes
        if total_elements > limits.maximum_array_elements:
            raise TensorNetworkArchiveSecurityError(
                "archive aggregate element count exceeds capacity"
            )
        if total_payload > limits.maximum_total_array_bytes:
            raise TensorNetworkArchiveSecurityError(
                "archive aggregate array bytes exceed capacity"
            )
    return total_elements, total_payload


def _canonical_member_name(name: str, /) -> bool:
    path = PurePosixPath(name)
    return bool(
        name
        and not name.startswith("/")
        and "\\" not in name
        and all(part not in ("", ".", "..") for part in path.parts)
        and path.as_posix() == name
    )


def _preflight_archive(
    path: str | os.PathLike[str], limits: TensorNetworkArchiveLimits, /
) -> tuple[int, int]:
    source = Path(path)
    try:
        archive_bytes = source.stat().st_size
        if archive_bytes > limits.maximum_archive_bytes:
            raise TensorNetworkArchiveSecurityError("archive bytes exceed capacity")
        with zipfile.ZipFile(source, mode="r") as archive:
            members = archive.infolist()
            if len(members) > limits.maximum_members:
                raise TensorNetworkArchiveSecurityError(
                    "archive member count exceeds capacity"
                )
            names = tuple(member.filename for member in members)
            if len(names) != len(set(names)):
                raise TensorNetworkArchiveCorruptionError(
                    "archive contains duplicate members"
                )
            if any(not _canonical_member_name(name) for name in names):
                raise TensorNetworkArchiveSecurityError(
                    "archive contains a noncanonical or unsafe member path"
                )
            for member in members:
                mode = member.external_attr >> 16
                if mode & 0o170000 == 0o120000:
                    raise TensorNetworkArchiveSecurityError(
                        "archive symbolic-link members are prohibited"
                    )
                if member.flag_bits & 0x1:
                    raise TensorNetworkArchiveSecurityError(
                        "encrypted archive members are prohibited"
                    )
                if member.compress_type != zipfile.ZIP_STORED:
                    raise TensorNetworkArchiveSecurityError(
                        "compressed archive members are prohibited"
                    )
                if member.file_size != member.compress_size:
                    raise TensorNetworkArchiveCorruptionError(
                        "stored archive member sizes are inconsistent"
                    )
            manifests = tuple(
                member for member in members if member.filename == "manifest.json"
            )
            if len(manifests) != 1:
                raise TensorNetworkArchiveCorruptionError(
                    "archive requires exactly one manifest"
                )
            if manifests[0].file_size > limits.maximum_manifest_bytes:
                raise TensorNetworkArchiveSecurityError(
                    "archive manifest bytes exceed capacity"
                )
            array_members = tuple(
                member for member in members if member.filename != "manifest.json"
            )
            total_elements = 0
            for member in array_members:
                if member.file_size > limits.maximum_array_bytes:
                    raise TensorNetworkArchiveSecurityError(
                        "archive array member bytes exceed capacity"
                    )
                try:
                    with archive.open(member, mode="r") as stream:
                        version = np.lib.format.read_magic(stream)
                        if version != (1, 0):
                            raise TensorNetworkArchiveCorruptionError(
                                "archive array header is not canonical NumPy format"
                            )
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                        header_bytes = stream.tell()
                except TensorNetworkArchiveError:
                    raise
                except (EOFError, ValueError) as error:
                    raise TensorNetworkArchiveCorruptionError(
                        "archive array header is invalid"
                    ) from error
                dtype_ = np.dtype(dtype)
                if dtype_.hasobject or dtype_.kind not in "biufc":
                    raise TensorNetworkArchiveSecurityError(
                        "archive array dtype is not numerical"
                    )
                if len(shape) > limits.maximum_array_rank:
                    raise TensorNetworkArchiveSecurityError(
                        "archive array rank exceeds capacity"
                    )
                if any(dimension < 0 for dimension in shape):
                    raise TensorNetworkArchiveCorruptionError(
                        "archive array shape contains a negative dimension"
                    )
                elements = math.prod(shape)
                if elements > limits.maximum_array_elements:
                    raise TensorNetworkArchiveSecurityError(
                        "archive array elements exceed capacity"
                    )
                total_elements += elements
                if total_elements > limits.maximum_array_elements:
                    raise TensorNetworkArchiveSecurityError(
                        "archive aggregate elements exceed capacity"
                    )
                if header_bytes + elements * dtype_.itemsize != member.file_size:
                    raise TensorNetworkArchiveCorruptionError(
                        "archive array header and payload sizes differ"
                    )
            total_member_bytes = sum(member.file_size for member in members)
            if (
                total_member_bytes
                > limits.maximum_total_array_bytes + limits.maximum_manifest_bytes
            ):
                raise TensorNetworkArchiveSecurityError(
                    "archive expanded bytes exceed capacity"
                )
            return archive_bytes, len(members)
    except TensorNetworkArchiveError:
        raise
    except (FileNotFoundError, PermissionError, zipfile.BadZipFile, OSError) as error:
        raise TensorNetworkArchiveCorruptionError("archive cannot be opened") from error


def _manifest_record(
    manifest: Mapping[str, Any], arrays: Mapping[str, Any], /
) -> TensorNetworkArchiveRecord:
    expected = {
        "kind",
        "artifact_kind",
        "artifact_id",
        "structure_id",
        "precision_policy_id",
        "source_id",
        "precision",
        "tree",
        "leaf_count",
        "payload_bytes",
        "array_digest",
        "arrays",
    }
    if set(manifest) != expected or manifest.get("kind") != "tensor-network-artifact":
        raise TensorNetworkArchiveCorruptionError(
            "archive manifest fields are not canonical"
        )
    try:
        kind = TensorNetworkArchiveKind(str(manifest["artifact_kind"]))
    except ValueError as error:
        raise TensorNetworkArchiveCorruptionError(
            "archive artifact kind is unsupported"
        ) from error
    identifier_names = (
        "artifact_id",
        "structure_id",
        "precision_policy_id",
        "source_id",
    )
    if any(not isinstance(manifest[name], str) for name in identifier_names):
        raise TensorNetworkArchiveCorruptionError(
            "archive identity fields must be strings"
        )
    identifiers = tuple(_identifier(manifest[name], name) for name in identifier_names)
    if (
        isinstance(manifest["leaf_count"], bool)
        or not isinstance(manifest["leaf_count"], int)
        or isinstance(manifest["payload_bytes"], bool)
        or not isinstance(manifest["payload_bytes"], int)
    ):
        raise TensorNetworkArchiveCorruptionError(
            "archive leaf count and payload bytes must be integers"
        )
    leaf_count = manifest["leaf_count"]
    payload_bytes = manifest["payload_bytes"]
    if leaf_count <= 0 or payload_bytes <= 0 or leaf_count != len(arrays):
        raise TensorNetworkArchiveCorruptionError(
            "archive leaf count or payload byte count is invalid"
        )
    tree = manifest["tree"]
    if not isinstance(tree, Mapping):
        raise TensorNetworkArchiveCorruptionError("archive tree record is invalid")
    names = tree.get("arrays")
    if (
        not isinstance(names, list)
        or len(names) != leaf_count
        or len(set(names)) != leaf_count
        or set(names) != set(arrays)
        or tree.get("num_leaves") != leaf_count
    ):
        raise TensorNetworkArchiveCorruptionError(
            "archive tree and array inventory differ"
        )
    digest = array_collection_digest(arrays)
    actual_payload_bytes = sum(
        array_payload_byte_count(value) for value in arrays.values()
    )
    if digest != manifest["array_digest"] or actual_payload_bytes != payload_bytes:
        raise TensorNetworkArchiveCorruptionError(
            "archive array collection content address is invalid"
        )
    content = {
        "kind": "tensor-network-artifact",
        "artifact_kind": kind.value,
        "structure_id": identifiers[1],
        "precision_policy_id": identifiers[2],
        "source_id": identifiers[3],
        "precision": manifest["precision"],
        "tree": tree,
        "leaf_count": leaf_count,
        "payload_bytes": payload_bytes,
        "array_digest": digest,
    }
    if canonical_fingerprint(content) != identifiers[0]:
        raise TensorNetworkArchiveCorruptionError(
            "archive artifact content address is invalid"
        )
    return TensorNetworkArchiveRecord(
        kind,
        identifiers[0],
        identifiers[1],
        identifiers[2],
        identifiers[3],
        leaf_count,
        payload_bytes,
        digest,
    )


def _load_validated_archive(
    path: str | os.PathLike[str],
    limits: TensorNetworkArchiveLimits,
    /,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    TensorNetworkArchiveValidation,
]:
    archive_bytes, member_count = _preflight_archive(path, limits)
    try:
        manifest, arrays = read_array_archive(path)
    except ArrayArchiveCorruptionError as error:
        raise TensorNetworkArchiveCorruptionError(str(error)) from error
    _check_array_bounds(arrays, limits)
    record = _manifest_record(manifest, arrays)
    validation = TensorNetworkArchiveValidation(
        record,
        limits.limits_id,
        archive_bytes,
        member_count,
        True,
    )
    return manifest, arrays, validation


def validate_tensor_network_archive(
    path: str | os.PathLike[str],
    /,
    *,
    limits: TensorNetworkArchiveLimits | None = None,
) -> TensorNetworkArchiveValidation:
    """Bound and checksum-validate a canonical pickle-free artifact archive."""

    limits_ = _limits(limits)
    _, _, validation = _load_validated_archive(path, limits_)
    return validation


def write_tensor_network_archive(
    path: str | os.PathLike[str],
    value: Any,
    /,
    *,
    kind: TensorNetworkArchiveKind,
    source_id: str,
    structure_id: str | None = None,
    precision_policy_id: str | None = None,
    limits: TensorNetworkArchiveLimits | None = None,
) -> TensorNetworkArchiveValidation:
    """Atomically publish MPS/MPO/LPDO or one explicit array-only PyTree kind."""

    kind_ = TensorNetworkArchiveKind(kind)
    limits_ = _limits(limits)
    source = _identifier(source_id, "source_id")
    tree, precision, inherent_structure = _artifact_tree(value, kind_)
    _, raw_arrays = _array_only_tree(tree)
    if kind_ == TensorNetworkArchiveKind.ARRAY_PYTREE:
        structure = _identifier(structure_id, "structure_id")
        policy_id = _identifier(precision_policy_id, "precision_policy_id")
        precision_record: dict[str, str | None] | None = None
    else:
        if precision is None or inherent_structure is None:
            raise TypeError("Tensor-network value lacks precision or structure evidence.")
        structure = inherent_structure
        policy_id = precision.policy_id
        precision_record = _precision_record(precision)
        if (
            structure_id is not None
            and _identifier(structure_id, "structure_id") != structure
        ):
            raise TensorNetworkArchiveMismatchError(
                "supplied structure_id differs from the tensor value"
            )
        if (
            precision_policy_id is not None
            and _identifier(precision_policy_id, "precision_policy_id") != policy_id
        ):
            raise TensorNetworkArchiveMismatchError(
                "supplied precision_policy_id differs from the tensor value"
            )
    arrays: dict[str, object] = {}
    tree_record = pack_array_tree("artifact", tree, arrays)
    _check_array_bounds(arrays, limits_)
    payload_bytes = sum(array_payload_byte_count(value_) for value_ in arrays.values())
    array_digest = array_collection_digest(arrays)
    content: dict[str, Any] = {
        "kind": "tensor-network-artifact",
        "artifact_kind": kind_.value,
        "structure_id": structure,
        "precision_policy_id": policy_id,
        "source_id": source,
        "precision": precision_record,
        "tree": tree_record,
        "leaf_count": len(raw_arrays),
        "payload_bytes": payload_bytes,
        "array_digest": array_digest,
    }
    artifact_id = canonical_fingerprint(content)
    manifest = {**content, "artifact_id": artifact_id}
    if len(canonical_json(manifest).encode("utf-8")) > limits_.maximum_manifest_bytes:
        raise TensorNetworkArchiveSecurityError("archive manifest bytes exceed capacity")
    write_array_archive(path, manifest=manifest, arrays=arrays)
    _, _, validation = _load_validated_archive(path, limits_)
    return validation


def _read_arrays(
    path: str | os.PathLike[str], limits: TensorNetworkArchiveLimits, /
) -> tuple[dict[str, Any], dict[str, np.ndarray], TensorNetworkArchiveValidation]:
    return _load_validated_archive(path, limits)


def read_tensor_network_archive(
    path: str | os.PathLike[str],
    /,
    *,
    kind: TensorNetworkArchiveKind,
    template: Any | None = None,
    expected_structure_id: str | None = None,
    expected_precision_policy_id: str | None = None,
    expected_source_id: str | None = None,
    limits: TensorNetworkArchiveLimits | None = None,
) -> TensorNetworkArchivedArtifact:
    """Read an explicitly selected kind and enforce every supplied identity."""

    kind_ = TensorNetworkArchiveKind(kind)
    limits_ = _limits(limits)
    manifest, arrays, validation = _read_arrays(path, limits_)
    record = validation.record
    if record.artifact_kind != kind_:
        raise TensorNetworkArchiveMismatchError(
            "requested archive kind differs from the stored explicit kind"
        )
    comparisons = (
        (expected_structure_id, record.structure_id, "structure_id"),
        (
            expected_precision_policy_id,
            record.precision_policy_id,
            "precision_policy_id",
        ),
        (expected_source_id, record.source_id, "source_id"),
    )
    for expected, observed, name in comparisons:
        if expected is not None and _identifier(expected, name) != observed:
            raise TensorNetworkArchiveMismatchError(
                f"archive {name} differs from the required identity"
            )
    tree_record = manifest["tree"]
    names = tree_record["arrays"]
    if kind_ == TensorNetworkArchiveKind.ARRAY_PYTREE:
        if template is None:
            raise TensorNetworkArchiveMismatchError(
                "array-pytree reads require an exact runtime template"
            )
        try:
            value = unpack_array_tree(tree_record, arrays, template)
        except ValueError as error:
            raise TensorNetworkArchiveMismatchError(str(error)) from error
        return TensorNetworkArchivedArtifact(value, record, validation)
    if template is not None:
        raise TensorNetworkArchiveMismatchError(
            "MPS/MPO/LPDO archives reconstruct themselves and reject templates"
        )
    paths = tree_record.get("paths")
    expected_paths = [f"[{index}]" for index in range(len(names))]
    if paths != expected_paths:
        raise TensorNetworkArchiveCorruptionError(
            "tensor archive tree paths are not canonical"
        )
    tensors = tuple(jnp.asarray(arrays[name]) for name in names)
    policy = _precision_from_record(manifest["precision"], record.precision_policy_id)
    try:
        if kind_ == TensorNetworkArchiveKind.MPS:
            value = MatrixProductState(tensors, precision=policy)
        elif kind_ == TensorNetworkArchiveKind.MPO:
            value = MatrixProductOperator(tensors, precision=policy)
        else:
            value = LocallyPurifiedDensity(tensors, precision=policy)
    except (TypeError, ValueError) as error:
        raise TensorNetworkArchiveCorruptionError(
            "stored tensor arrays violate their declared representation"
        ) from error
    if value.structure_id != record.structure_id:
        raise TensorNetworkArchiveCorruptionError(
            "stored tensor structure content address is invalid"
        )
    return TensorNetworkArchivedArtifact(value, record, validation)


__all__ = [
    "TensorNetworkArchiveCorruptionError",
    "TensorNetworkArchiveError",
    "TensorNetworkArchiveKind",
    "TensorNetworkArchiveLimits",
    "TensorNetworkArchiveMismatchError",
    "TensorNetworkArchiveRecord",
    "TensorNetworkArchiveSecurityError",
    "TensorNetworkArchiveValidation",
    "TensorNetworkArchivedArtifact",
    "read_tensor_network_archive",
    "validate_tensor_network_archive",
    "write_tensor_network_archive",
]
