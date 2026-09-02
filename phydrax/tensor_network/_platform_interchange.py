#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint, canonical_json
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._platform_archive import TensorNetworkArchiveKind
from ._platform_support import (
    _identifier,
    _positive_integer,
    TensorNetworkFailure,
)


InterchangeAttribute: TypeAlias = str | int | float | bool
_HEX = frozenset("0123456789abcdef")


def _sha256(value: object, name: str, /) -> str:
    digest = _identifier(value, name)
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return digest


def _dataset_name(value: object, /) -> str:
    name = _identifier(value, "dataset name")
    parts = name.split("/")
    if name.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise ValueError("Interchange dataset names must be canonical relative paths.")
    return name


def _canonical_array(value: Any, /) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biufc":
        raise TypeError("Interchange datasets must be numerical arrays.")
    dtype = array.dtype
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
    return np.ascontiguousarray(array, dtype=dtype)


def _payload_digest(value: np.ndarray, /) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _attribute(value: object, /) -> InterchangeAttribute:
    if type(value) not in (str, int, float, bool):
        raise TypeError("Interchange attributes must be JSON scalar values.")
    if isinstance(value, str) and not value:
        raise ValueError("Interchange string attributes must be nonempty.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Interchange numeric attributes must be finite.")
    return value


class TensorNetworkInterchangeLimits(StrictModule, NonTrainableState):
    maximum_datasets: int = eqx.field(static=True)
    maximum_rank: int = eqx.field(static=True)
    maximum_elements_per_dataset: int = eqx.field(static=True)
    maximum_total_elements: int = eqx.field(static=True)
    maximum_total_bytes: int = eqx.field(static=True)
    maximum_metadata_bytes: int = eqx.field(static=True)
    limits_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_datasets: int = 100_000,
        maximum_rank: int = 16,
        maximum_elements_per_dataset: int = 1_000_000_000,
        maximum_total_elements: int = 1_000_000_000,
        maximum_total_bytes: int = 4 * 1024**3,
        maximum_metadata_bytes: int = 16 * 1024**2,
    ):
        values = tuple(
            _positive_integer(value, name)
            for value, name in (
                (maximum_datasets, "maximum_datasets"),
                (maximum_rank, "maximum_rank"),
                (
                    maximum_elements_per_dataset,
                    "maximum_elements_per_dataset",
                ),
                (maximum_total_elements, "maximum_total_elements"),
                (maximum_total_bytes, "maximum_total_bytes"),
                (maximum_metadata_bytes, "maximum_metadata_bytes"),
            )
        )
        (
            self.maximum_datasets,
            self.maximum_rank,
            self.maximum_elements_per_dataset,
            self.maximum_total_elements,
            self.maximum_total_bytes,
            self.maximum_metadata_bytes,
        ) = values
        self.limits_id = canonical_fingerprint(
            {"kind": "tensor-network-interchange-limits", "values": values}
        )


class TensorNetworkInterchangeSecurityError(RuntimeError):
    failure = TensorNetworkFailure.SECURITY_LIMIT

    def __init__(self, detail: str, /):
        self.detail = _identifier(detail, "interchange security detail")
        super().__init__(f"security-limit: {self.detail}")


class TensorNetworkInterchangeDataset(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    axis_labels: tuple[str, ...] = eqx.field(static=True)
    role: str = eqx.field(static=True)
    payload_bytes: int = eqx.field(static=True)
    payload_sha256: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        shape: Sequence[int],
        dtype: str,
        /,
        *,
        axis_labels: Sequence[str],
        role: str,
        payload_bytes: int,
        payload_sha256: str,
    ):
        name_ = _dataset_name(name)
        shape_ = tuple(int(value) for value in shape)
        if any(value < 0 for value in shape_):
            raise ValueError("Interchange dataset dimensions must be nonnegative.")
        dtype_ = np.dtype(_identifier(dtype, "dataset dtype"))
        if dtype_.hasobject or dtype_.kind not in "biufc":
            raise ValueError("Interchange dataset dtype must be numerical.")
        if dtype_.itemsize > 1:
            dtype_ = dtype_.newbyteorder("<")
        labels = tuple(_identifier(value, "axis label") for value in axis_labels)
        if len(labels) != len(shape_) or len(set(labels)) != len(labels):
            raise ValueError("Interchange axis labels must uniquely name every axis.")
        role_ = _identifier(role, "dataset role")
        bytes_ = int(payload_bytes)
        expected_bytes = math.prod(shape_) * dtype_.itemsize
        if bytes_ < 0 or bytes_ != expected_bytes:
            raise ValueError("Interchange payload byte count is inconsistent.")
        digest = _sha256(payload_sha256, "payload_sha256")
        self.name = name_
        self.shape = shape_
        self.dtype = dtype_.str
        self.axis_labels = labels
        self.role = role_
        self.payload_bytes = bytes_
        self.payload_sha256 = digest
        self.dataset_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "axis_labels": list(self.axis_labels),
            "role": self.role,
            "payload_bytes": self.payload_bytes,
            "payload_sha256": self.payload_sha256,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "dataset_id": self.dataset_id}

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], /
    ) -> TensorNetworkInterchangeDataset:
        expected = {
            "name",
            "shape",
            "dtype",
            "axis_labels",
            "role",
            "payload_bytes",
            "payload_sha256",
            "dataset_id",
        }
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError("Interchange dataset record fields are not canonical.")
        shape = record["shape"]
        labels = record["axis_labels"]
        if (
            not isinstance(shape, Sequence)
            or isinstance(shape, str)
            or not isinstance(labels, Sequence)
            or isinstance(labels, str)
        ):
            raise TypeError("Interchange shape and axis labels must be sequences.")
        text_fields = ("name", "dtype", "role", "payload_sha256", "dataset_id")
        if any(not isinstance(record[field], str) for field in text_fields):
            raise TypeError("Interchange dataset identity fields must be strings.")
        if any(
            isinstance(item, bool) or not isinstance(item, int) for item in shape
        ) or any(not isinstance(item, str) for item in labels):
            raise TypeError(
                "Interchange shape entries must be integers and labels must be strings."
            )
        if isinstance(record["payload_bytes"], bool) or not isinstance(
            record["payload_bytes"], int
        ):
            raise TypeError("Interchange payload_bytes must be an integer.")
        value = cls(
            record["name"],
            tuple(shape),
            record["dtype"],
            axis_labels=tuple(labels),
            role=record["role"],
            payload_bytes=record["payload_bytes"],
            payload_sha256=record["payload_sha256"],
        )
        if value.dataset_id != record["dataset_id"]:
            raise ValueError("Interchange dataset content address is invalid.")
        return value


class TensorNetworkLicenseProvenance(StrictModule, NonTrainableState):
    license_id: str = eqx.field(static=True)
    name: str = eqx.field(static=True)
    source_uri: str = eqx.field(static=True)
    text_sha256: str = eqx.field(static=True)
    notice_required: bool = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        license_id: str,
        /,
        *,
        name: str,
        source_uri: str,
        text_sha256: str,
        notice_required: bool,
    ):
        values = tuple(
            _identifier(value, field)
            for value, field in (
                (license_id, "license_id"),
                (name, "license name"),
                (source_uri, "license source_uri"),
            )
        )
        digest = _sha256(text_sha256, "license text_sha256")
        self.license_id, self.name, self.source_uri = values
        self.text_sha256 = digest
        self.notice_required = bool(notice_required)
        self.record_id = canonical_fingerprint(
            {
                "kind": "tensor-network-license-provenance",
                "license_id": values[0],
                "name": values[1],
                "source_uri": values[2],
                "text_sha256": digest,
                "notice_required": self.notice_required,
            }
        )


class TensorNetworkDependencyProvenance(StrictModule, NonTrainableState):
    package: str = eqx.field(static=True)
    installed_version: str = eqx.field(static=True)
    source_uri: str = eqx.field(static=True)
    artifact_sha256: str = eqx.field(static=True)
    license_ids: tuple[str, ...] = eqx.field(static=True)
    direct: bool = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        package: str,
        /,
        *,
        installed_version: str,
        source_uri: str,
        artifact_sha256: str,
        license_ids: Sequence[str],
        direct: bool,
    ):
        values = tuple(
            _identifier(value, field)
            for value, field in (
                (package, "package"),
                (installed_version, "installed_version"),
                (source_uri, "dependency source_uri"),
            )
        )
        digest = _sha256(artifact_sha256, "dependency artifact_sha256")
        licenses = tuple(
            sorted(_identifier(value, "license_id") for value in license_ids)
        )
        if not licenses or len(set(licenses)) != len(licenses):
            raise ValueError("Dependency license IDs must be nonempty and unique.")
        self.package, self.installed_version, self.source_uri = values
        self.artifact_sha256 = digest
        self.license_ids = licenses
        self.direct = bool(direct)
        self.record_id = canonical_fingerprint(
            {
                "kind": "tensor-network-dependency-provenance",
                "package": values[0],
                "installed_version": values[1],
                "source_uri": values[2],
                "artifact_sha256": digest,
                "license_ids": licenses,
                "direct": self.direct,
            }
        )


class TensorNetworkProvenance(StrictModule, NonTrainableState):
    source_commit: str = eqx.field(static=True)
    source_tree_sha256: str = eqx.field(static=True)
    dependency_lock_sha256: str = eqx.field(static=True)
    dependencies: tuple[TensorNetworkDependencyProvenance, ...]
    licenses: tuple[TensorNetworkLicenseProvenance, ...]
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_commit: str,
        source_tree_sha256: str,
        dependency_lock_sha256: str,
        dependencies: Sequence[TensorNetworkDependencyProvenance],
        licenses: Sequence[TensorNetworkLicenseProvenance],
    ):
        commit = _identifier(source_commit, "source_commit")
        source_digest = _sha256(source_tree_sha256, "source_tree_sha256")
        lock_digest = _sha256(dependency_lock_sha256, "dependency_lock_sha256")
        dependencies_ = tuple(dependencies)
        licenses_ = tuple(licenses)
        if not dependencies_ or any(
            not isinstance(value, TensorNetworkDependencyProvenance)
            for value in dependencies_
        ):
            raise TypeError("Provenance requires typed dependencies.")
        if not licenses_ or any(
            not isinstance(value, TensorNetworkLicenseProvenance) for value in licenses_
        ):
            raise TypeError("Provenance requires typed license records.")
        packages = tuple(value.package for value in dependencies_)
        license_ids = tuple(value.license_id for value in licenses_)
        if len(set(packages)) != len(packages) or len(set(license_ids)) != len(
            license_ids
        ):
            raise ValueError("Provenance packages and licenses must be unique.")
        known = set(license_ids)
        if any(
            license_id not in known
            for dependency in dependencies_
            for license_id in dependency.license_ids
        ):
            raise ValueError("Dependency provenance cites an unknown license.")
        dependencies_ = tuple(sorted(dependencies_, key=lambda value: value.package))
        licenses_ = tuple(sorted(licenses_, key=lambda value: value.license_id))
        self.source_commit = commit
        self.source_tree_sha256 = source_digest
        self.dependency_lock_sha256 = lock_digest
        self.dependencies = dependencies_
        self.licenses = licenses_
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "tensor-network-provenance",
                "source_commit": commit,
                "source_tree_sha256": source_digest,
                "dependency_lock_sha256": lock_digest,
                "dependencies": [value.record_id for value in dependencies_],
                "licenses": [value.record_id for value in licenses_],
            }
        )


class TensorNetworkInterchangeManifest(StrictModule, NonTrainableState):
    artifact_kind: TensorNetworkArchiveKind = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    datasets: tuple[TensorNetworkInterchangeDataset, ...]
    attributes: tuple[tuple[str, InterchangeAttribute], ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        artifact_kind: TensorNetworkArchiveKind,
        structure_id: str,
        precision_policy_id: str,
        source_id: str,
        provenance_id: str,
        datasets: Sequence[TensorNetworkInterchangeDataset],
        attributes: Mapping[str, InterchangeAttribute] | None = None,
    ):
        kind = TensorNetworkArchiveKind(artifact_kind)
        identifiers = tuple(
            _identifier(value, name)
            for value, name in (
                (structure_id, "structure_id"),
                (precision_policy_id, "precision_policy_id"),
                (source_id, "source_id"),
                (provenance_id, "provenance_id"),
            )
        )
        datasets_ = tuple(datasets)
        if not datasets_ or any(
            not isinstance(value, TensorNetworkInterchangeDataset) for value in datasets_
        ):
            raise TypeError("Interchange manifests require typed datasets.")
        names = tuple(value.name for value in datasets_)
        if len(set(names)) != len(names):
            raise ValueError("Interchange dataset names must be unique.")
        datasets_ = tuple(sorted(datasets_, key=lambda value: value.name))
        attributes_map = {} if attributes is None else attributes
        if not isinstance(attributes_map, Mapping):
            raise TypeError("Interchange attributes must be a mapping.")
        attributes_ = tuple(
            sorted(
                (
                    _identifier(name, "attribute name"),
                    _attribute(value),
                )
                for name, value in attributes_map.items()
            )
        )
        self.artifact_kind = kind
        (
            self.structure_id,
            self.precision_policy_id,
            self.source_id,
            self.provenance_id,
        ) = identifiers
        self.datasets = datasets_
        self.attributes = attributes_
        self.manifest_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "tensor-network-hdf5-neutral",
            "artifact_kind": self.artifact_kind.value,
            "structure_id": self.structure_id,
            "precision_policy_id": self.precision_policy_id,
            "source_id": self.source_id,
            "provenance_id": self.provenance_id,
            "layout": "row-major",
            "datasets": [value.to_record() for value in self.datasets],
            "attributes": dict(self.attributes),
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "manifest_id": self.manifest_id}

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], /
    ) -> TensorNetworkInterchangeManifest:
        expected = {
            "kind",
            "artifact_kind",
            "structure_id",
            "precision_policy_id",
            "source_id",
            "provenance_id",
            "layout",
            "datasets",
            "attributes",
            "manifest_id",
        }
        if (
            not isinstance(record, Mapping)
            or set(record) != expected
            or record.get("kind") != "tensor-network-hdf5-neutral"
            or record.get("layout") != "row-major"
        ):
            raise ValueError("Interchange manifest fields are not canonical.")
        datasets = record["datasets"]
        attributes = record["attributes"]
        if (
            not isinstance(datasets, Sequence)
            or isinstance(datasets, str)
            or not isinstance(attributes, Mapping)
        ):
            raise TypeError("Interchange datasets or attributes are invalid.")
        text_fields = (
            "artifact_kind",
            "structure_id",
            "precision_policy_id",
            "source_id",
            "provenance_id",
            "manifest_id",
        )
        if any(not isinstance(record[field], str) for field in text_fields):
            raise TypeError("Interchange manifest identity fields must be strings.")
        if any(not isinstance(name, str) for name in attributes):
            raise TypeError("Interchange attribute names must be strings.")
        value = cls(
            artifact_kind=TensorNetworkArchiveKind(record["artifact_kind"]),
            structure_id=record["structure_id"],
            precision_policy_id=record["precision_policy_id"],
            source_id=record["source_id"],
            provenance_id=record["provenance_id"],
            datasets=tuple(
                TensorNetworkInterchangeDataset.from_record(item) for item in datasets
            ),
            attributes={name: _attribute(item) for name, item in attributes.items()},
        )
        if value.manifest_id != record["manifest_id"]:
            raise ValueError("Interchange manifest content address is invalid.")
        return value


class TensorNetworkInterchangeValidation(StrictModule, NonTrainableState):
    manifest_id: str = eqx.field(static=True)
    limits_id: str = eqx.field(static=True)
    valid: bool = eqx.field(static=True)
    failure: TensorNetworkFailure = eqx.field(static=True)
    mismatches: tuple[str, ...] = eqx.field(static=True)
    validation_id: str = eqx.field(static=True)

    def __init__(
        self,
        manifest_id: str,
        limits_id: str,
        valid: bool,
        failure: TensorNetworkFailure,
        mismatches: Sequence[str],
        /,
    ):
        manifest = _identifier(manifest_id, "manifest_id")
        limits = _identifier(limits_id, "limits_id")
        valid_ = bool(valid)
        failure_ = TensorNetworkFailure(failure)
        mismatches_ = tuple(
            _identifier(value, "interchange mismatch") for value in mismatches
        )
        if valid_:
            if failure_ != TensorNetworkFailure.NONE or mismatches_:
                raise ValueError("Valid interchange records cannot contain mismatches.")
        elif failure_ != TensorNetworkFailure.ARCHIVE_MISMATCH or not mismatches_:
            raise ValueError("Invalid interchange records require typed mismatches.")
        self.manifest_id = manifest
        self.limits_id = limits
        self.valid = valid_
        self.failure = failure_
        self.mismatches = mismatches_
        self.validation_id = canonical_fingerprint(
            {
                "kind": "tensor-network-interchange-validation",
                "manifest": manifest,
                "limits": limits,
                "valid": valid_,
                "failure": failure_.value,
                "mismatches": mismatches_,
            }
        )


def _limits(
    value: TensorNetworkInterchangeLimits | None, /
) -> TensorNetworkInterchangeLimits:
    if value is None:
        return TensorNetworkInterchangeLimits()
    if not isinstance(value, TensorNetworkInterchangeLimits):
        raise TypeError("limits must be TensorNetworkInterchangeLimits or None.")
    return value


def _validate_manifest_bounds(
    manifest: TensorNetworkInterchangeManifest,
    limits: TensorNetworkInterchangeLimits,
    /,
) -> None:
    if len(manifest.datasets) > limits.maximum_datasets:
        raise TensorNetworkInterchangeSecurityError(
            "interchange dataset count exceeds capacity"
        )
    total_elements = 0
    total_bytes = 0
    for dataset in manifest.datasets:
        elements = math.prod(dataset.shape)
        if len(dataset.shape) > limits.maximum_rank:
            raise TensorNetworkInterchangeSecurityError(
                "interchange dataset rank exceeds capacity"
            )
        if elements > limits.maximum_elements_per_dataset:
            raise TensorNetworkInterchangeSecurityError(
                "interchange dataset elements exceed capacity"
            )
        total_elements += elements
        total_bytes += dataset.payload_bytes
        if total_elements > limits.maximum_total_elements:
            raise TensorNetworkInterchangeSecurityError(
                "interchange total elements exceed capacity"
            )
        if total_bytes > limits.maximum_total_bytes:
            raise TensorNetworkInterchangeSecurityError(
                "interchange total bytes exceed capacity"
            )
    metadata_bytes = len(canonical_json(manifest.to_record()).encode("utf-8"))
    if metadata_bytes > limits.maximum_metadata_bytes:
        raise TensorNetworkInterchangeSecurityError(
            "interchange metadata exceeds capacity"
        )


def _preflight_arrays(
    arrays: Mapping[str, Any],
    limits: TensorNetworkInterchangeLimits,
    /,
) -> dict[str, np.ndarray]:
    if not arrays or len(arrays) > limits.maximum_datasets:
        raise TensorNetworkInterchangeSecurityError(
            "interchange dataset count exceeds capacity"
        )
    values: dict[str, np.ndarray] = {}
    total_elements = 0
    total_bytes = 0
    for original_name, original_value in arrays.items():
        name = _dataset_name(original_name)
        value = np.asarray(original_value)
        if value.dtype.hasobject or value.dtype.kind not in "biufc":
            raise TypeError("Interchange datasets must be numerical arrays.")
        if value.ndim > limits.maximum_rank:
            raise TensorNetworkInterchangeSecurityError(
                f"interchange dataset {name!r} rank exceeds capacity"
            )
        if value.size > limits.maximum_elements_per_dataset:
            raise TensorNetworkInterchangeSecurityError(
                f"interchange dataset {name!r} elements exceed capacity"
            )
        total_elements += int(value.size)
        total_bytes += int(value.nbytes)
        if total_elements > limits.maximum_total_elements:
            raise TensorNetworkInterchangeSecurityError(
                "interchange total elements exceed capacity"
            )
        if total_bytes > limits.maximum_total_bytes:
            raise TensorNetworkInterchangeSecurityError(
                "interchange total bytes exceed capacity"
            )
        values[name] = value
    if len(values) != len(arrays):
        raise ValueError("Interchange dataset names are not canonically unique.")
    return values


def make_tensor_network_interchange_manifest(
    arrays: Mapping[str, Any],
    axis_labels: Mapping[str, Sequence[str]],
    roles: Mapping[str, str],
    /,
    *,
    artifact_kind: TensorNetworkArchiveKind,
    structure_id: str,
    precision_policy_id: str,
    source_id: str,
    provenance_id: str,
    attributes: Mapping[str, InterchangeAttribute] | None = None,
    limits: TensorNetworkInterchangeLimits | None = None,
) -> TensorNetworkInterchangeManifest:
    """Describe concrete arrays in a canonical backend-neutral dataset manifest."""

    if not isinstance(arrays, Mapping) or not arrays:
        raise TypeError("Interchange arrays must be a nonempty mapping.")
    if set(arrays) != set(axis_labels) or set(arrays) != set(roles):
        raise ValueError("Every interchange array needs exact axes and role metadata.")
    limits_ = _limits(limits)
    preflight = _preflight_arrays(arrays, limits_)
    datasets = []
    for name in sorted(preflight):
        value = _canonical_array(preflight[name])
        datasets.append(
            TensorNetworkInterchangeDataset(
                name,
                value.shape,
                value.dtype.str,
                axis_labels=axis_labels[name],
                role=roles[name],
                payload_bytes=int(value.nbytes),
                payload_sha256=_payload_digest(value),
            )
        )
    manifest = TensorNetworkInterchangeManifest(
        artifact_kind=artifact_kind,
        structure_id=structure_id,
        precision_policy_id=precision_policy_id,
        source_id=source_id,
        provenance_id=provenance_id,
        datasets=datasets,
        attributes=attributes,
    )
    _validate_manifest_bounds(manifest, limits_)
    return manifest


def validate_tensor_network_interchange(
    manifest: TensorNetworkInterchangeManifest,
    arrays: Mapping[str, Any],
    /,
    *,
    limits: TensorNetworkInterchangeLimits | None = None,
) -> TensorNetworkInterchangeValidation:
    """Validate bounded array payloads against every canonical dataset record."""

    if not isinstance(manifest, TensorNetworkInterchangeManifest):
        raise TypeError("manifest must be TensorNetworkInterchangeManifest.")
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping.")
    limits_ = _limits(limits)
    _validate_manifest_bounds(manifest, limits_)
    preflight = _preflight_arrays(arrays, limits_)
    by_name = {value.name: value for value in manifest.datasets}
    mismatches: list[str] = []
    if set(preflight) != set(by_name):
        mismatches.append("dataset names")
    for name in sorted(set(preflight).intersection(by_name)):
        value = _canonical_array(preflight[name])
        dataset = by_name[name]
        if tuple(value.shape) != dataset.shape:
            mismatches.append(f"{name}:shape")
        if value.dtype.str != dataset.dtype:
            mismatches.append(f"{name}:dtype")
        if int(value.nbytes) != dataset.payload_bytes:
            mismatches.append(f"{name}:bytes")
        if _payload_digest(value) != dataset.payload_sha256:
            mismatches.append(f"{name}:digest")
    valid = not mismatches
    failure = (
        TensorNetworkFailure.NONE if valid else TensorNetworkFailure.ARCHIVE_MISMATCH
    )
    return TensorNetworkInterchangeValidation(
        manifest.manifest_id,
        limits_.limits_id,
        valid,
        failure,
        tuple(mismatches),
    )


__all__ = [
    "TensorNetworkDependencyProvenance",
    "TensorNetworkInterchangeDataset",
    "TensorNetworkInterchangeLimits",
    "TensorNetworkInterchangeManifest",
    "TensorNetworkInterchangeSecurityError",
    "TensorNetworkInterchangeValidation",
    "TensorNetworkLicenseProvenance",
    "TensorNetworkProvenance",
    "make_tensor_network_interchange_manifest",
    "validate_tensor_network_interchange",
]
