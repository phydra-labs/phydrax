#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


HostMetadataValue: TypeAlias = str | int | float | bool


class ImageCoordinateFrame(StrEnum):
    """Supported medical-image patient coordinate frames."""

    LPS = "LPS"
    RAS = "RAS"


class ImageLengthUnit(StrEnum):
    """Explicit affine translation unit at the host image boundary."""

    MILLIMETER = "mm"
    CENTIMETER = "cm"
    METER = "m"
    MICROMETER = "um"

    @property
    def millimeters_per_unit(self) -> float:
        return {
            ImageLengthUnit.MILLIMETER: 1.0,
            ImageLengthUnit.CENTIMETER: 10.0,
            ImageLengthUnit.METER: 1000.0,
            ImageLengthUnit.MICROMETER: 0.001,
        }[self]


class ImageAcquisitionIdentity(StrictModule, NonTrainableState):
    """De-identified acquisition identity, without patient or date fields."""

    acquisition_id: str = eqx.field(static=True)
    series_id: str = eqx.field(static=True)
    modality: str = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)

    def __init__(
        self,
        acquisition_id: str,
        series_id: str,
        modality: str,
        protocol_id: str,
        /,
        *,
        identity_id: str | None = None,
    ):
        values = tuple(
            str(value) for value in (acquisition_id, series_id, modality, protocol_id)
        )
        if any(not value for value in values):
            raise ValueError("Acquisition identity fields must be non-empty.")
        self.acquisition_id = values[0]
        self.series_id = values[1]
        self.modality = values[2]
        self.protocol_id = values[3]
        payload = {
            "kind": "cardiac-image-acquisition-identity",
            "acquisition": values[0],
            "series": values[1],
            "modality": values[2],
            "protocol": values[3],
        }
        self.identity_id = _resolved_id("identity_id", identity_id, payload)


class ImageDeidentificationIdentity(StrictModule, NonTrainableState):
    """Identity of the de-identification process and its attestation."""

    method_id: str = eqx.field(static=True)
    processing_id: str = eqx.field(static=True)
    attestation_id: str = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)

    def __init__(
        self,
        method_id: str,
        processing_id: str,
        attestation_id: str,
        /,
        *,
        identity_id: str | None = None,
    ):
        values = tuple(str(value) for value in (method_id, processing_id, attestation_id))
        if any(not value for value in values):
            raise ValueError("De-identification identity fields must be non-empty.")
        self.method_id = values[0]
        self.processing_id = values[1]
        self.attestation_id = values[2]
        payload = {
            "kind": "cardiac-image-deidentification-identity",
            "method": values[0],
            "processing": values[1],
            "attestation": values[2],
        }
        self.identity_id = _resolved_id("identity_id", identity_id, payload)


class ImageDataRightsIdentity(StrictModule, NonTrainableState):
    """Stable data-rights identity and explicitly permitted uses."""

    rights_id: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    controller_id: str = eqx.field(static=True)
    permitted_use_ids: tuple[str, ...] = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)

    def __init__(
        self,
        rights_id: str,
        license_id: str,
        controller_id: str,
        /,
        *,
        permitted_use_ids: Sequence[str],
        identity_id: str | None = None,
    ):
        values = tuple(str(value) for value in (rights_id, license_id, controller_id))
        uses = tuple(str(value) for value in permitted_use_ids)
        if any(not value for value in values):
            raise ValueError("Data-rights identity fields must be non-empty.")
        if not uses or any(not value for value in uses) or len(set(uses)) != len(uses):
            raise ValueError("permitted_use_ids must be unique non-empty identifiers.")
        self.rights_id = values[0]
        self.license_id = values[1]
        self.controller_id = values[2]
        self.permitted_use_ids = uses
        payload = {
            "kind": "cardiac-image-data-rights-identity",
            "rights": values[0],
            "license": values[1],
            "controller": values[2],
            "permitted_uses": list(uses),
        }
        self.identity_id = _resolved_id("identity_id", identity_id, payload)


class MedicalImageAffine(StrictModule, NonTrainableState):
    """Voxel-index to LPS or RAS world affine with an explicit length unit."""

    voxel_to_world: Array
    coordinate_frame: ImageCoordinateFrame = eqx.field(static=True)
    length_unit: ImageLengthUnit = eqx.field(static=True)
    affine_id: str = eqx.field(static=True)

    def __init__(
        self,
        voxel_to_world: ArrayLike,
        coordinate_frame: ImageCoordinateFrame,
        length_unit: ImageLengthUnit,
        /,
        *,
        affine_id: str | None = None,
    ):
        if not isinstance(coordinate_frame, ImageCoordinateFrame):
            raise TypeError("coordinate_frame must be an ImageCoordinateFrame.")
        if not isinstance(length_unit, ImageLengthUnit):
            raise TypeError("length_unit must be an ImageLengthUnit.")
        matrix = np.asarray(voxel_to_world)
        if matrix.shape != (4, 4):
            raise ValueError("voxel_to_world must have shape (4, 4).")
        if not np.issubdtype(matrix.dtype, np.inexact):
            matrix = matrix.astype(float)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("voxel_to_world must be finite.")
        if not np.array_equal(matrix[3], np.asarray((0.0, 0.0, 0.0, 1.0))):
            raise ValueError("voxel_to_world must be a homogeneous affine matrix.")
        linear = matrix[:3, :3]
        determinant = (
            linear[0, 0] * (linear[1, 1] * linear[2, 2] - linear[1, 2] * linear[2, 1])
            - linear[0, 1] * (linear[1, 0] * linear[2, 2] - linear[1, 2] * linear[2, 0])
            + linear[0, 2] * (linear[1, 0] * linear[2, 1] - linear[1, 1] * linear[2, 0])
        )
        if not np.isfinite(determinant) or determinant == 0.0:
            raise ValueError("voxel_to_world must have an invertible spatial block.")
        matrix_array = jnp.asarray(matrix)
        self.voxel_to_world = matrix_array
        self.coordinate_frame = coordinate_frame
        self.length_unit = length_unit
        payload = {
            "kind": "medical-image-affine",
            "matrix": array_tree_fingerprint(matrix_array),
            "coordinate_frame": str(coordinate_frame),
            "length_unit": str(length_unit),
        }
        self.affine_id = _resolved_id("affine_id", affine_id, payload)

    def reframe(self, coordinate_frame: ImageCoordinateFrame, /) -> MedicalImageAffine:
        if not isinstance(coordinate_frame, ImageCoordinateFrame):
            raise TypeError("coordinate_frame must be an ImageCoordinateFrame.")
        if coordinate_frame is self.coordinate_frame:
            return self
        frame_flip = jnp.diag(
            jnp.asarray((-1.0, -1.0, 1.0, 1.0), dtype=self.voxel_to_world.dtype)
        )
        matrix = oe.contract("ij,jk->ik", frame_flip, self.voxel_to_world)
        return MedicalImageAffine(matrix, coordinate_frame, self.length_unit)

    def in_millimeters(self) -> MedicalImageAffine:
        if self.length_unit is ImageLengthUnit.MILLIMETER:
            return self
        scale = jnp.diag(
            jnp.asarray(
                (
                    self.length_unit.millimeters_per_unit,
                    self.length_unit.millimeters_per_unit,
                    self.length_unit.millimeters_per_unit,
                    1.0,
                ),
                dtype=self.voxel_to_world.dtype,
            )
        )
        matrix = oe.contract("ij,jk->ik", scale, self.voxel_to_world)
        return MedicalImageAffine(
            matrix,
            self.coordinate_frame,
            ImageLengthUnit.MILLIMETER,
        )


class CardiacImageBoundaryMetadata(StrictModule, NonTrainableState):
    """Non-PHI image metadata admitted to the cardiovascular geometry boundary."""

    affine: MedicalImageAffine
    acquisition: ImageAcquisitionIdentity
    deidentification: ImageDeidentificationIdentity
    data_rights: ImageDataRightsIdentity
    host_fields: tuple[tuple[str, HostMetadataValue], ...] = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)

    def __init__(
        self,
        affine: MedicalImageAffine,
        acquisition: ImageAcquisitionIdentity,
        deidentification: ImageDeidentificationIdentity,
        data_rights: ImageDataRightsIdentity,
        /,
        *,
        coordinate_frame: ImageCoordinateFrame,
        host_fields: Mapping[str, HostMetadataValue] | None = None,
        metadata_id: str | None = None,
    ):
        if not isinstance(affine, MedicalImageAffine):
            raise TypeError("affine must be a MedicalImageAffine.")
        if not isinstance(acquisition, ImageAcquisitionIdentity):
            raise TypeError("acquisition must be an ImageAcquisitionIdentity.")
        if not isinstance(deidentification, ImageDeidentificationIdentity):
            raise TypeError("deidentification must be an ImageDeidentificationIdentity.")
        if not isinstance(data_rights, ImageDataRightsIdentity):
            raise TypeError("data_rights must be an ImageDataRightsIdentity.")
        if not isinstance(coordinate_frame, ImageCoordinateFrame):
            raise TypeError("coordinate_frame must be an ImageCoordinateFrame.")
        if coordinate_frame is not affine.coordinate_frame:
            raise ValueError("The declared image frame conflicts with the affine frame.")
        fields = _validated_host_fields({} if host_fields is None else host_fields)
        self.affine = affine
        self.acquisition = acquisition
        self.deidentification = deidentification
        self.data_rights = data_rights
        self.host_fields = fields
        payload = {
            "kind": "cardiac-image-boundary-metadata",
            "affine": affine.affine_id,
            "acquisition": acquisition.identity_id,
            "deidentification": deidentification.identity_id,
            "data_rights": data_rights.identity_id,
            "host_fields": [list(value) for value in fields],
        }
        self.metadata_id = _resolved_id("metadata_id", metadata_id, payload)

    def reframe(
        self, coordinate_frame: ImageCoordinateFrame, /
    ) -> CardiacImageBoundaryMetadata:
        reframed = self.affine.reframe(coordinate_frame)
        return CardiacImageBoundaryMetadata(
            reframed,
            self.acquisition,
            self.deidentification,
            self.data_rights,
            coordinate_frame=coordinate_frame,
            host_fields=dict(self.host_fields),
        )

    def in_kernel_units(self) -> CardiacImageBoundaryMetadata:
        affine = self.affine.in_millimeters()
        return CardiacImageBoundaryMetadata(
            affine,
            self.acquisition,
            self.deidentification,
            self.data_rights,
            coordinate_frame=affine.coordinate_frame,
            host_fields=dict(self.host_fields),
        )


_ALLOWED_HOST_FIELDS = frozenset(
    {
        "acquisition_plane",
        "body_part",
        "contrast_agent_class",
        "echo_time_ms",
        "field_strength_t",
        "flip_angle_degree",
        "reconstruction_kernel",
        "repetition_time_ms",
        "sequence_id",
        "slice_thickness_mm",
        "spatial_resolution_mm",
        "temporal_resolution_ms",
    }
)


def _resolved_id(name: str, value: str | None, payload: dict[str, object], /) -> str:
    if value is None:
        return canonical_fingerprint(payload)
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _validated_host_fields(
    values: Mapping[str, HostMetadataValue], /
) -> tuple[tuple[str, HostMetadataValue], ...]:
    if not isinstance(values, Mapping):
        raise TypeError("host_fields must be a mapping.")
    fields: list[tuple[str, HostMetadataValue]] = []
    for raw_name, value in values.items():
        name = str(raw_name).strip().lower()
        if name not in _ALLOWED_HOST_FIELDS:
            raise ValueError(
                f"Host image field {raw_name!r} is not on the non-PHI allowlist."
            )
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError("Host image field values must be scalar metadata values.")
        if isinstance(value, str) and not value:
            raise ValueError("Host image string values must be non-empty.")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Host image numeric values must be finite.")
        fields.append((name, value))
    if len({name for name, _ in fields}) != len(fields):
        raise ValueError("Host image field names must be unique after normalization.")
    return tuple(sorted(fields))


__all__ = [
    "CardiacImageBoundaryMetadata",
    "HostMetadataValue",
    "ImageAcquisitionIdentity",
    "ImageCoordinateFrame",
    "ImageDataRightsIdentity",
    "ImageDeidentificationIdentity",
    "ImageLengthUnit",
    "MedicalImageAffine",
]
