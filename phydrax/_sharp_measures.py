#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import Enum, IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


class SharpMeasureFidelity(str, Enum):
    """Fidelity of absolute fluid measures, not merely of their source field."""

    EXACT = "exact"
    CERTIFIED_BOUNDED_ERROR = "certified_bounded_error"
    UNQUALIFIED = "unqualified"


class SharpGeometryStatus(IntFlag):
    """Fail-closed status bits for a fixed-capacity sharp realization."""

    SUCCESS = 0
    UNQUALIFIED_SOURCE = 1
    INVALID_BOUNDS = 2
    UNRESOLVED_TOPOLOGY = 4
    CAPACITY_EXHAUSTED = 8
    GCL_FAILED = 16
    NONFINITE = 32
    REFRESH_REQUIRED = 64
    INTERFACE_MOMENTS_UNQUALIFIED = 128


class SharpGeometryEvidence(StrictModule):
    """JAX-safe numerical evidence attached to one sharp realization."""

    cell_bound_width: Array
    face_bound_width: tuple[Array, ...]
    gcl_residual_lower: Array
    gcl_residual_upper: Array
    source_qualified: Array
    bounds_valid: Array
    topology_resolved: Array
    gcl_satisfied: Array
    interface_moments_qualified: Array
    finite: Array
    accepted: Array
    refresh_required: Array
    status: Array
    evidence_id: str = eqx.field(static=True)


class QualifiedSharpGeometry(StrictModule, NonTrainableState):
    """One source-bound carrier for absolute sharp cell and face measures.

    Fractions are deliberately derived conveniences. Consumers bind the absolute
    measures and their support/operator identities and must check ``accepted``.
    """

    cell_fluid_measure: Array
    cell_fluid_measure_lower: Array
    cell_fluid_measure_upper: Array
    cell_full_measure: Array
    face_open_measure: tuple[Array, ...]
    face_open_measure_lower: tuple[Array, ...]
    face_open_measure_upper: tuple[Array, ...]
    face_full_measure: tuple[Array, ...]
    interface_measure: Array
    interface_measure_lower: Array
    interface_measure_upper: Array
    interface_centroid: Array
    interface_normal: Array
    body_id: Array
    wall_velocity: tuple[Array, ...]
    swept_cell_measure_rate: Array
    cell_active: Array
    face_active: tuple[Array, ...]
    small_cell_mask: Array
    epoch: Array
    evidence: SharpGeometryEvidence
    measure_fidelity: SharpMeasureFidelity = eqx.field(static=True)
    source_fidelity: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    cell_field_id: str = eqx.field(static=True)
    face_field_ids: tuple[str, ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    pairing_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    @property
    def cell_fluid_fraction(self) -> Array:
        return self.cell_fluid_measure / self.cell_full_measure

    @property
    def cell_fluid_fraction_lower(self) -> Array:
        return self.cell_fluid_measure_lower / self.cell_full_measure

    @property
    def cell_fluid_fraction_upper(self) -> Array:
        return self.cell_fluid_measure_upper / self.cell_full_measure

    @property
    def face_open_fraction(self) -> tuple[Array, ...]:
        return tuple(
            value / full
            for value, full in zip(
                self.face_open_measure, self.face_full_measure, strict=True
            )
        )

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted

    @property
    def qualified(self) -> Array:
        return self.evidence.source_qualified & (
            self.measure_fidelity is not SharpMeasureFidelity.UNQUALIFIED
        )


def exact_sharp_geometry(
    cell_fluid_measure: ArrayLike,
    cell_full_measure: ArrayLike,
    face_open_measure: tuple[ArrayLike, ...],
    face_full_measure: tuple[ArrayLike, ...],
    /,
    *,
    measure_evidence_id: str,
    source_id: str,
    source_fidelity: str,
    support_id: str,
    cell_field_id: str,
    face_field_ids: tuple[str, ...],
    operator_id: str,
    pairing_id: str,
    interface_measure: ArrayLike | None = None,
    interface_centroid: ArrayLike | None = None,
    interface_normal: ArrayLike | None = None,
    body_id: ArrayLike | None = None,
    wall_velocity: tuple[ArrayLike, ...] | None = None,
    swept_cell_measure_rate: ArrayLike | None = None,
    small_cell_fraction: float = 0.0,
    epoch: int = 0,
    interface_moments_qualified: bool = False,
) -> QualifiedSharpGeometry:
    """Build an exact, host-validated realization from absolute measures.

    This constructor is for a producer that already owns an exact clipping
    theorem named by ``measure_evidence_id``. It performs no sampling, clipping,
    or implicit conversion.
    """

    fluid = np.asarray(cell_fluid_measure)
    full = np.asarray(cell_full_measure)
    opened = tuple(np.asarray(value) for value in face_open_measure)
    face_full = tuple(np.asarray(value) for value in face_full_measure)
    dimension = fluid.ndim
    if (
        fluid.shape != full.shape
        or dimension not in (2, 3)
        or len(opened) != dimension
        or len(face_full) != dimension
        or len(face_field_ids) != dimension
    ):
        raise ValueError("Exact sharp measures have incompatible dimensions or shapes.")
    if (
        np.any(~np.isfinite(fluid))
        or np.any(~np.isfinite(full))
        or np.any(full <= 0.0)
        or np.any((fluid < 0.0) | (fluid > full))
        or any(
            value.shape != total.shape
            or np.any(~np.isfinite(value))
            or np.any(~np.isfinite(total))
            or np.any(total <= 0.0)
            or np.any((value < 0.0) | (value > total))
            for value, total in zip(opened, face_full, strict=True)
        )
    ):
        raise ValueError("Exact sharp measures must be finite and physically bounded.")
    identifiers = (
        str(measure_evidence_id),
        str(source_id),
        str(source_fidelity),
        str(support_id),
        str(cell_field_id),
        *(str(value) for value in face_field_ids),
        str(operator_id),
        str(pairing_id),
    )
    if any(not value for value in identifiers):
        raise ValueError("Sharp geometry provenance identifiers must be nonempty.")
    small = float(small_cell_fraction)
    if not 0.0 <= small < 1.0:
        raise ValueError("small_cell_fraction must lie in [0, 1).")

    area = (
        np.zeros_like(fluid)
        if interface_measure is None
        else np.asarray(interface_measure)
    )
    centroid = (
        np.zeros(fluid.shape + (dimension,), dtype=fluid.dtype)
        if interface_centroid is None
        else np.asarray(interface_centroid)
    )
    normal = (
        np.zeros_like(centroid)
        if interface_normal is None
        else np.asarray(interface_normal)
    )
    bodies = (
        np.full(fluid.shape, -1, dtype=np.int32)
        if body_id is None
        else np.asarray(body_id, dtype=np.int32)
    )
    if (
        area.shape != fluid.shape
        or centroid.shape != fluid.shape + (dimension,)
        or normal.shape != centroid.shape
        or bodies.shape != fluid.shape
        or np.any(~np.isfinite(area))
        or np.any(area < 0.0)
        or np.any(~np.isfinite(centroid))
        or np.any(~np.isfinite(normal))
    ):
        raise ValueError("Exact sharp interface moments have incompatible values.")
    cut = (fluid > 0.0) & (fluid < full) & (area > 0.0)
    norm = np.linalg.norm(normal, axis=-1)
    if interface_moments_qualified and np.any(np.abs(norm[cut] - 1.0) > 1.0e-8):
        raise ValueError("Qualified cut-interface normals must be unit length.")

    walls = (
        tuple(np.zeros_like(value) for value in opened)
        if wall_velocity is None
        else tuple(np.asarray(value) for value in wall_velocity)
    )
    if len(walls) != dimension or any(
        value.shape != opened[axis].shape or np.any(~np.isfinite(value))
        for axis, value in enumerate(walls)
    ):
        raise ValueError("Sharp wall velocity must match the directional face layouts.")
    swept = (
        np.zeros_like(fluid)
        if swept_cell_measure_rate is None
        else np.asarray(swept_cell_measure_rate)
    )
    if swept.shape != fluid.shape or np.any(~np.isfinite(swept)):
        raise ValueError("Swept cell measure rate must be finite and cell shaped.")

    arrays_id = array_tree_fingerprint(
        (fluid, full, opened, face_full, area, centroid, normal, bodies, walls, swept)
    )
    realization_id = canonical_fingerprint(
        {
            "kind": "qualified-exact-sharp-geometry",
            "source": str(source_id),
            "support": str(support_id),
            "operator": str(operator_id),
            "pairing": str(pairing_id),
            "measure_evidence": str(measure_evidence_id),
            "arrays": arrays_id,
            "epoch": int(epoch),
        }
    )
    moment_ok = jnp.asarray(bool(interface_moments_qualified))
    status = jnp.asarray(
        int(SharpGeometryStatus.SUCCESS)
        if interface_moments_qualified
        else int(SharpGeometryStatus.INTERFACE_MOMENTS_UNQUALIFIED),
        dtype=jnp.int32,
    )
    zero = jnp.zeros_like(jnp.asarray(fluid))
    evidence = SharpGeometryEvidence(
        zero,
        tuple(jnp.zeros_like(jnp.asarray(value)) for value in opened),
        zero,
        zero,
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(True),
        moment_ok,
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(False),
        status,
        canonical_fingerprint(
            {
                "kind": "exact-sharp-geometry-evidence",
                "realization": realization_id,
                "external_measure_evidence": str(measure_evidence_id),
            }
        ),
    )
    fluid_ = jnp.asarray(fluid)
    full_ = jnp.asarray(full)
    opened_ = tuple(jnp.asarray(value) for value in opened)
    face_full_ = tuple(jnp.asarray(value) for value in face_full)
    active = fluid_ > 0.0
    return QualifiedSharpGeometry(
        fluid_,
        fluid_,
        fluid_,
        full_,
        opened_,
        opened_,
        opened_,
        face_full_,
        jnp.asarray(area),
        jnp.asarray(area),
        jnp.asarray(area),
        jnp.asarray(centroid),
        jnp.asarray(normal),
        jnp.asarray(bodies),
        tuple(jnp.asarray(value) for value in walls),
        jnp.asarray(swept),
        active,
        tuple(value > 0.0 for value in opened_),
        active & (fluid_ < small * full_) if small > 0.0 else jnp.zeros_like(active),
        jnp.asarray(int(epoch), dtype=jnp.int32),
        evidence,
        SharpMeasureFidelity.EXACT,
        str(source_fidelity),
        str(source_id),
        str(support_id),
        str(cell_field_id),
        tuple(str(value) for value in face_field_ids),
        str(operator_id),
        str(pairing_id),
        realization_id,
    )


__all__ = [
    "QualifiedSharpGeometry",
    "SharpGeometryEvidence",
    "SharpGeometryStatus",
    "SharpMeasureFidelity",
    "exact_sharp_geometry",
]
