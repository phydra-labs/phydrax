#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned, artifact-import-only Moqui NPZ and embedded-MHA score profiles."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np

from ...._fingerprint import canonical_fingerprint
from ....interchange import (
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    AdapterWaiver,
    BoundedResource,
)
from ....qualification import ReferenceArtifactManifest
from .._scores import (
    _identifier,
    _positive_integer,
    _UNCERTAINTY_KINDS,
    ExternalRadiationRunIdentity,
    ExternalRadiationScoreResult,
    radiation_score_content_id,
    RadiationEstimatorEvidence,
    RadiationScoreDefinition,
    require_import_rights,
    require_profile_semantics,
)


_BASE_SEMANTICS = (
    "build-identity",
    "calibration-identity",
    "configuration-identity",
    "grid-affine",
    "score-dtype",
    "score-normalization",
    "score-quantity",
    "score-shape",
    "seed-lineage",
    "table-identity",
    "units",
)
_MHA_TYPES = {"MET_FLOAT": np.dtype("f4"), "MET_DOUBLE": np.dtype("f8")}
_MHA_FIELDS = frozenset(
    {
        "ObjectType",
        "NDims",
        "BinaryData",
        "BinaryDataByteOrderMSB",
        "CompressedData",
        "TransformMatrix",
        "Offset",
        "Position",
        "Origin",
        "CenterOfRotation",
        "AnatomicalOrientation",
        "ElementSpacing",
        "DimSize",
        "ElementNumberOfChannels",
        "ElementType",
        "ElementDataFile",
    }
)


def _array_name(value: str | None, name: str, /) -> str | None:
    if value is None:
        return None
    result = _identifier(value, name)
    if result.endswith(".npy") or "/" in result or "\\" in result:
        raise ValueError(f"{name} must be one bare NPZ array name without .npy.")
    return result


def _permutation(values: tuple[int, ...], rank: int, /) -> tuple[int, ...]:
    result = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError("storage_to_score_axes must contain integers.")
    if result != tuple(sorted(result)) and set(result) != set(range(rank)):
        raise ValueError("storage_to_score_axes must be a complete axis permutation.")
    if len(result) != rank or set(result) != set(range(rank)):
        raise ValueError("storage_to_score_axes must be a complete axis permutation.")
    return result


def _decode_npz(
    resource: BoundedResource, profile: MoquiArrayProfile, /
) -> tuple[np.ndarray, np.ndarray | None]:
    stream = BytesIO(resource.data)
    if not zipfile.is_zipfile(stream):
        raise ValueError("Moqui NPZ artifact is not a ZIP-based NPZ archive.")
    stream.seek(0)
    expected = {f"{profile.score_member}.npy", f"{profile.affine_member}.npy"}
    if profile.uncertainty_member is not None:
        expected.add(f"{profile.uncertainty_member}.npy")
    with zipfile.ZipFile(stream) as archive:
        entries = archive.infolist()
        names = tuple(entry.filename for entry in entries)
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Moqui NPZ members differ from the exact array profile.")
        if any(entry.is_dir() or entry.flag_bits & 0x1 for entry in entries):
            raise ValueError("Encrypted or directory NPZ members are forbidden.")
        if sum(entry.file_size for entry in entries) > resource.manifest.limits.max_bytes:
            raise ValueError(
                "Moqui NPZ uncompressed arrays exceed the resource byte limit."
            )
    stream.seek(0)
    with np.load(stream, allow_pickle=False, max_header_size=64_000) as arrays:
        values = np.asarray(arrays[profile.score_member])
        affine = np.asarray(arrays[profile.affine_member])
        uncertainty = (
            None
            if profile.uncertainty_member is None
            else np.asarray(arrays[profile.uncertainty_member])
        )
    if affine.shape != (4, 4) or not np.issubdtype(affine.dtype, np.number):
        raise ValueError("Moqui NPZ affine must be one numeric 4 x 4 matrix.")
    if not np.array_equal(
        affine.astype(profile.score.grid_affine.matrix.dtype, copy=False),
        profile.score.grid_affine.matrix,
    ):
        raise ValueError("Moqui NPZ grid affine differs from the pinned profile.")
    return values, uncertainty


def _mha_header_and_payload(data: bytes, /) -> tuple[dict[str, str], bytes]:
    marker = b"\r\n\r\n"
    boundary = data.find(marker)
    if boundary < 0:
        marker = b"\n\n"
        boundary = data.find(marker)
    if boundary < 0:
        raise ValueError("Embedded MHA requires a blank line before local binary data.")
    header_text = data[:boundary].decode("ascii", errors="strict")
    fields: dict[str, str] = {}
    for line in header_text.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value:
            raise ValueError("MHA header lines must be nonempty key = value pairs.")
        if key not in _MHA_FIELDS:
            raise ValueError(f"Unsupported MHA header field {key!r}.")
        if key in fields:
            raise ValueError(f"Duplicate MHA header field {key!r}.")
        fields[key] = value
    return fields, data[boundary + len(marker) :]


def _mha_boolean(
    fields: dict[str, str], name: str, /, *, default: bool | None = None
) -> bool:
    if name not in fields:
        if default is None:
            raise ValueError(f"MHA header is missing {name}.")
        return default
    value = fields[name].lower()
    if value not in ("true", "false"):
        raise ValueError(f"MHA {name} must be True or False.")
    return value == "true"


def _mha_numbers(fields: dict[str, str], name: str, count: int, /) -> np.ndarray:
    if name not in fields:
        raise ValueError(f"MHA header is missing {name}.")
    tokens = fields[name].split()
    if len(tokens) != count:
        raise ValueError(f"MHA {name} must contain exactly {count} values.")
    values = np.asarray(tuple(float(token) for token in tokens), dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"MHA {name} values must be finite.")
    return values


def _decode_mha(resource: BoundedResource, profile: MoquiArrayProfile, /) -> np.ndarray:
    fields, payload = _mha_header_and_payload(resource.data)
    if fields.get("ObjectType") != "Image":
        raise ValueError("Moqui MHA ObjectType must be Image.")
    if fields.get("ElementDataFile") != "LOCAL":
        raise ValueError("Only self-contained MHA ElementDataFile = LOCAL is supported.")
    if fields.get("NDims") != "3" or len(profile.storage_shape) != 3:
        raise ValueError("Moqui embedded-MHA profile requires exactly three dimensions.")
    if not _mha_boolean(fields, "BinaryData"):
        raise ValueError("Moqui MHA requires binary element data.")
    if _mha_boolean(fields, "CompressedData", default=False):
        raise ValueError("Compressed MHA element data is unsupported.")
    if fields.get("ElementNumberOfChannels", "1") != "1":
        raise ValueError("Moqui MHA score arrays must have one scalar channel.")
    element_type = fields.get("ElementType")
    if element_type not in _MHA_TYPES:
        raise ValueError("Moqui MHA ElementType must be MET_FLOAT or MET_DOUBLE.")
    base_dtype = _MHA_TYPES[element_type]
    big_endian = _mha_boolean(fields, "BinaryDataByteOrderMSB", default=False)
    dtype = base_dtype.newbyteorder(">" if big_endian else "<")
    dim_size = _mha_numbers(fields, "DimSize", 3)
    if np.any(dim_size != np.floor(dim_size)) or np.any(dim_size < 1.0):
        raise ValueError("MHA DimSize must contain positive integers.")
    dimensions = tuple(int(value) for value in dim_size)
    if tuple(reversed(dimensions)) != profile.storage_shape:
        raise ValueError("MHA DimSize differs from the profile storage shape.")
    expected_bytes = int(np.prod(profile.storage_shape)) * dtype.itemsize
    if len(payload) != expected_bytes:
        raise ValueError("MHA local element payload size differs from the exact profile.")
    storage = np.frombuffer(payload, dtype=dtype).reshape(profile.storage_shape)
    values = np.transpose(storage, profile.storage_to_score_axes)
    transform = _mha_numbers(fields, "TransformMatrix", 9).reshape((3, 3))
    spacing = _mha_numbers(fields, "ElementSpacing", 3)
    if np.any(spacing <= 0.0):
        raise ValueError("MHA ElementSpacing must be positive.")
    origins = [name for name in ("Offset", "Position", "Origin") if name in fields]
    if len(origins) != 1:
        raise ValueError("MHA requires exactly one of Offset, Position, or Origin.")
    origin = _mha_numbers(fields, origins[0], 3)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = transform @ np.diag(spacing)
    affine[:3, 3] = origin
    if not np.array_equal(affine, profile.score.grid_affine.matrix.astype(np.float64)):
        raise ValueError(
            "MHA transform, spacing, or origin differs from the pinned affine."
        )
    return values


@dataclass(frozen=True, slots=True)
class MoquiArrayProfile:
    """One exact Moqui array container, storage order, and score meaning."""

    container: str
    run_identity: ExternalRadiationRunIdentity
    score: RadiationScoreDefinition
    storage_shape: tuple[int, ...]
    storage_to_score_axes: tuple[int, ...]
    score_member: str | None
    affine_member: str | None
    uncertainty_member: str | None
    uncertainty_kind: str
    estimator: str
    history_count: int
    batch_count: int
    correlation_model: str
    declared_losses: tuple[AdapterLoss, ...] = ()
    profile_id: str = field(init=False)
    available_semantics: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        container = _identifier(self.container, "container").lower()
        if container not in ("npz", "mha"):
            raise ValueError("Moqui container must be npz or mha.")
        if not isinstance(self.run_identity, ExternalRadiationRunIdentity):
            raise TypeError("run_identity must be ExternalRadiationRunIdentity.")
        if self.run_identity.engine != "Moqui":
            raise ValueError("Moqui array profiles require a Moqui run identity.")
        if not isinstance(self.score, RadiationScoreDefinition):
            raise TypeError("score must be RadiationScoreDefinition.")
        storage_shape = tuple(
            _positive_integer(value, "storage shape entry")
            for value in self.storage_shape
        )
        if len(storage_shape) != len(self.score.shape):
            raise ValueError("Storage rank must match the pinned score rank.")
        permutation = _permutation(self.storage_to_score_axes, len(storage_shape))
        transposed_shape = tuple(storage_shape[index] for index in permutation)
        if transposed_shape != self.score.shape:
            raise ValueError(
                "Storage shape and axis permutation do not produce score shape."
            )
        score_member = _array_name(self.score_member, "score_member")
        affine_member = _array_name(self.affine_member, "affine_member")
        uncertainty_member = _array_name(self.uncertainty_member, "uncertainty_member")
        if container == "npz":
            if score_member is None or affine_member is None:
                raise ValueError(
                    "Moqui NPZ profiles require score and affine member names."
                )
            names = tuple(
                value
                for value in (score_member, affine_member, uncertainty_member)
                if value is not None
            )
            if len(set(names)) != len(names):
                raise ValueError("Moqui NPZ member names must be distinct.")
        elif any(
            value is not None
            for value in (score_member, affine_member, uncertainty_member)
        ):
            raise ValueError("Embedded-MHA profiles do not use NPZ member names.")
        if container == "mha" and self.score.representation != "voxel-grid":
            raise ValueError("Embedded MHA supports voxel-grid scores only.")
        uncertainty_kind = _identifier(self.uncertainty_kind, "uncertainty_kind")
        if uncertainty_kind not in _UNCERTAINTY_KINDS:
            raise ValueError("Unknown Moqui estimator uncertainty kind.")
        correlation = _identifier(self.correlation_model, "correlation_model")
        if (uncertainty_member is None) != (uncertainty_kind == "unreported"):
            raise ValueError("Moqui uncertainty member and uncertainty kind must agree.")
        if uncertainty_member is None and correlation != "unreported":
            raise ValueError(
                "Absent Moqui uncertainty cannot claim correlation evidence."
            )
        if uncertainty_member is not None and correlation in (
            "unreported",
            "assumed-independent",
        ):
            raise ValueError("Moqui uncertainty requires upstream correlation evidence.")
        histories = _positive_integer(self.history_count, "history_count")
        batches = _positive_integer(self.batch_count, "batch_count")
        losses = tuple(self.declared_losses)
        if any(not isinstance(loss, AdapterLoss) for loss in losses):
            raise TypeError("declared_losses must contain AdapterLoss values.")
        available = _BASE_SEMANTICS + ("estimator",)
        if uncertainty_member is not None:
            available += ("uncertainty", "correlation")
        object.__setattr__(self, "container", container)
        object.__setattr__(self, "storage_shape", storage_shape)
        object.__setattr__(self, "storage_to_score_axes", permutation)
        object.__setattr__(self, "score_member", score_member)
        object.__setattr__(self, "affine_member", affine_member)
        object.__setattr__(self, "uncertainty_member", uncertainty_member)
        object.__setattr__(self, "uncertainty_kind", uncertainty_kind)
        object.__setattr__(self, "estimator", _identifier(self.estimator, "estimator"))
        object.__setattr__(self, "history_count", histories)
        object.__setattr__(self, "batch_count", batches)
        object.__setattr__(self, "correlation_model", correlation)
        object.__setattr__(self, "declared_losses", losses)
        object.__setattr__(self, "available_semantics", tuple(sorted(set(available))))
        object.__setattr__(
            self,
            "profile_id",
            canonical_fingerprint(
                {
                    "kind": "moqui-array-score-profile",
                    "container": container,
                    "run": self.run_identity.identity_id,
                    "score": self.score.definition_id,
                    "storage_shape": list(storage_shape),
                    "storage_to_score_axes": list(permutation),
                    "score_member": score_member,
                    "affine_member": affine_member,
                    "uncertainty_member": uncertainty_member,
                    "uncertainty_kind": uncertainty_kind,
                    "estimator": self.estimator,
                    "history_count": histories,
                    "batch_count": batches,
                    "correlation_model": correlation,
                    "declared_losses": [loss.loss_id for loss in losses],
                }
            ),
        )


def import_moqui_arrays(
    resource: BoundedResource,
    reference: ReferenceArtifactManifest,
    profile: MoquiArrayProfile,
    /,
    *,
    required_semantics: tuple[str, ...] = (),
    commercial_use: bool = False,
    redistribution: bool = False,
    training_use: bool = False,
    export: bool = False,
) -> ExternalRadiationScoreResult:
    """Import a pinned NPZ or embedded MHA artifact without provider execution."""

    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be BoundedResource.")
    if not isinstance(reference, ReferenceArtifactManifest):
        raise TypeError("reference must be ReferenceArtifactManifest.")
    if not isinstance(profile, MoquiArrayProfile):
        raise TypeError("profile must be MoquiArrayProfile.")
    require_profile_semantics(required_semantics, profile.available_semantics)
    reference.verify_bytes(resource.data)
    require_import_rights(
        (reference,),
        profile.run_identity,
        commercial_use=commercial_use,
        redistribution=redistribution,
        training_use=training_use,
        export=export,
    )
    if profile.container == "npz":
        storage, uncertainty_storage = _decode_npz(resource, profile)
        values = np.transpose(storage, profile.storage_to_score_axes)
        if uncertainty_storage is None:
            uncertainty = None
        elif profile.uncertainty_kind == "covariance":
            uncertainty = uncertainty_storage
        else:
            uncertainty = np.transpose(uncertainty_storage, profile.storage_to_score_axes)
    else:
        values = _decode_mha(resource, profile)
        uncertainty = None
    if values.dtype.str != profile.score.dtype or values.shape != profile.score.shape:
        raise ValueError("Moqui score shape or dtype differs from the exact profile.")
    if uncertainty is not None:
        expected_uncertainty_shape = (
            (int(np.prod(profile.score.shape)),) * 2
            if profile.uncertainty_kind == "covariance"
            else profile.score.shape
        )
        if uncertainty.shape != expected_uncertainty_shape or not np.issubdtype(
            uncertainty.dtype, np.floating
        ):
            raise ValueError(
                "Moqui uncertainty array differs from the exact estimator profile."
            )
    evidence = RadiationEstimatorEvidence(
        profile.estimator,
        profile.history_count,
        profile.batch_count,
        profile.uncertainty_kind,
        uncertainty,
        profile.correlation_model,
        (reference.manifest_id,),
    )
    score_id = radiation_score_content_id(
        values,
        profile.score,
        profile.run_identity,
        evidence,
        (reference,),
        profile.profile_id,
    )
    status = (
        AdapterStatus.DECLARED_LOSS if profile.declared_losses else AdapterStatus.LOSSLESS
    )
    report = AdapterReport(
        status,
        f"Moqui-{profile.container.upper()}",
        "ExternalRadiationScoreResult",
        source_id=reference.manifest_id,
        target_id=score_id,
        source_profile=AdapterFormatProfile(
            f"Moqui-{profile.container.upper()}",
            qualifiers={
                "external_profile_id": profile.profile_id,
                "provider_revision": profile.run_identity.engine_revision,
                "score_quantity": profile.score.quantity_kind.value,
            },
        ),
        coordinate_mapping=(
            f"storage-to-score axis permutation {profile.storage_to_score_axes}",
            f"grid affine {profile.score.grid_affine.affine_id}",
        ),
        preserved_fields=(
            "score values",
            "score dtype",
            "score normalization",
            "quantity and unit",
            "grid affine",
            "build/configuration/table/calibration/seed identities",
            "native estimator and correlation declaration",
        ),
        assumptions=(
            "caller-supplied profile metadata is authoritative for semantics absent from the array container",
            "artifact import does not qualify or execute Moqui",
            "result scope is research-only",
        ),
        losses=profile.declared_losses,
        waivers=tuple(
            AdapterWaiver(
                loss,
                "The exact research-only profile explicitly retains this declared source loss.",
            )
            for loss in profile.declared_losses
            if loss.changes_interpretation
        ),
    )
    return ExternalRadiationScoreResult(
        values,
        profile.score,
        profile.run_identity,
        evidence,
        (reference,),
        profile.profile_id,
        report,
    )


__all__ = ["MoquiArrayProfile", "import_moqui_arrays"]
