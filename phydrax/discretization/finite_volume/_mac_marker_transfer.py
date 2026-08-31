#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._incompressible import FaceVelocity, PreparedMACOperators


class MACMarkerRelation(StrictModule, NonTrainableState):
    """Fixed-capacity componentwise marker-to-MAC-face relation."""

    face_indices: tuple[Array, ...]
    weights: tuple[Array, ...]
    valid: tuple[Array, ...]
    support_count: tuple[Array, ...]
    partition_residual: tuple[Array, ...]
    capacity_overflow: tuple[Array, ...]
    support_truncated: Array
    periodic_image_used: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class MACMarkerTransferDiagnostics(StrictModule):
    """Conservation, adjointness, capacity, and support evidence."""

    marker_resultant: Array
    face_resultant: Array
    force_residual: Array
    interpolation_work: Array
    spreading_work: Array
    work_adjoint_residual: Array
    maximum_partition_residual: Array
    capacity_overflow: Array
    support_truncated: Array
    periodic_image_used: Array
    finite: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class MACMarkerTransferPlan(StrictModule, NonTrainableState):
    """Plan a compact componentwise marker coupling on a prepared MAC grid."""

    operators: PreparedMACOperators
    support_radius: float = eqx.field(static=True)
    maximum_faces_per_marker: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        support_radius: float,
        maximum_faces_per_marker: int,
        /,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        radius = float(support_radius)
        capacity = int(maximum_faces_per_marker)
        face_counts = tuple(
            int(np.prod(layout.shape)) for layout in operators.discretization.face_layouts
        )
        if (
            not np.isfinite(radius)
            or radius <= 0.0
            or capacity <= 0
            or capacity > min(face_counts)
        ):
            raise ValueError("MAC marker support radius or face capacity is invalid.")
        for axis in operators.discretization.grid.structured_axes:
            if axis.periodic:
                period = float(axis.bounds[1] - axis.bounds[0])
                if radius > 0.5 * period:
                    raise ValueError(
                        "MAC marker support radius cannot exceed half a periodic span."
                    )
        self.operators = operators
        self.support_radius = radius
        self.maximum_faces_per_marker = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-marker-transfer-plan",
                "operators": operators.prepared_id,
                "support_radius": radius,
                "maximum_faces_per_marker": capacity,
            }
        )

    def prepare(self, /) -> PreparedMACMarkerTransfer:
        return PreparedMACMarkerTransfer(self)


class PreparedMACMarkerTransfer(StrictModule, NonTrainableState):
    """Prepared dual-measure-adjoint MAC face/marker transfer."""

    operators: PreparedMACOperators
    support_radius: float = eqx.field(static=True)
    maximum_faces_per_marker: int = eqx.field(static=True)
    flattened_face_centers: tuple[Array, ...]
    flattened_dual_measures: tuple[Array, ...]
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MACMarkerTransferPlan, /):
        if not isinstance(plan, MACMarkerTransferPlan):
            raise TypeError("plan must be MACMarkerTransferPlan.")
        dimension = len(plan.operators.discretization.cell_shape)
        centers = tuple(
            value.reshape((-1, dimension))
            for value in plan.operators.discretization.face_centers
        )
        measures = tuple(
            value.reshape((-1,)) for value in plan.operators.face_dual_measures
        )
        self.operators = plan.operators
        self.support_radius = plan.support_radius
        self.maximum_faces_per_marker = plan.maximum_faces_per_marker
        self.flattened_face_centers = centers
        self.flattened_dual_measures = measures
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-marker-transfer",
                "plan": plan.plan_id,
                "face_shapes": [list(value.shape) for value in centers],
                "dual_measure_shapes": [list(value.shape) for value in measures],
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.flattened_face_centers)

    def relation(
        self,
        marker_positions: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> MACMarkerRelation:
        position = jnp.asarray(
            marker_positions, dtype=self.operators.pressure_space.dtype
        )
        if (
            position.ndim != 2
            or position.shape[0] == 0
            or position.shape[1] != self.dimension
        ):
            raise ValueError("marker_positions must have shape (markers,dimension).")
        active = (
            jnp.ones((position.shape[0],), dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        if active.shape != (position.shape[0],):
            raise ValueError("active_mask must have one entry per marker.")

        truncated = jnp.zeros((position.shape[0],), dtype=bool)
        for axis_index, axis in enumerate(
            self.operators.discretization.grid.structured_axes
        ):
            if not axis.periodic:
                lower = jnp.asarray(axis.bounds[0], dtype=position.dtype)
                upper = jnp.asarray(axis.bounds[1], dtype=position.dtype)
                truncated = truncated | (
                    (position[:, axis_index] - self.support_radius < lower)
                    | (position[:, axis_index] + self.support_radius > upper)
                )
        truncated = active & truncated
        finite_position = jnp.all(jnp.isfinite(position), axis=-1)

        all_indices = []
        all_weights = []
        all_valid = []
        all_counts = []
        all_partition = []
        all_overflow = []
        periodic_used = jnp.zeros((position.shape[0],), dtype=bool)
        for centers in self.flattened_face_centers:
            displacement = position[:, None, :] - centers[None, :, :]
            periodic_faces = jnp.zeros(displacement.shape[:2], dtype=bool)
            for axis_index, axis in enumerate(
                self.operators.discretization.grid.structured_axes
            ):
                if axis.periodic:
                    period = jnp.asarray(
                        axis.bounds[1] - axis.bounds[0], dtype=position.dtype
                    )
                    original = displacement[..., axis_index]
                    wrapped = original - period * jnp.floor(original / period + 0.5)
                    periodic_faces = periodic_faces | (wrapped != original)
                    displacement = displacement.at[..., axis_index].set(wrapped)
            distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
            q = distance / self.support_radius
            raw = jnp.where(q < 1.0, (1.0 - q) ** 4 * (1.0 + 4.0 * q), 0.0)
            positive_count = jnp.sum(raw > 0.0, axis=-1, dtype=jnp.int32)
            values, indices = jax.lax.top_k(raw, self.maximum_faces_per_marker)
            valid = (values > 0.0) & active[:, None]
            component_periodic = jnp.any(
                jnp.take_along_axis(periodic_faces, indices, axis=1) & valid,
                axis=1,
            )
            weight_sum = jnp.sum(jnp.where(valid, values, 0.0), axis=-1)
            normalizable = weight_sum > 0.0
            weights = jnp.where(
                valid,
                values / jnp.where(normalizable, weight_sum, 1.0)[:, None],
                0.0,
            )
            partition = jnp.sum(weights, axis=-1)
            residual = jnp.where(active, jnp.abs(partition - 1.0), 0.0)
            overflow = active & (positive_count > self.maximum_faces_per_marker)
            all_indices.append(indices.astype(jnp.int32))
            all_weights.append(weights)
            all_valid.append(valid)
            all_counts.append(jnp.where(active, positive_count, 0))
            all_partition.append(residual)
            all_overflow.append(overflow)
            periodic_used = periodic_used | (active & component_periodic)

        maximum_partition = jnp.max(jnp.stack(tuple(all_partition)))
        any_overflow = jnp.any(jnp.stack(tuple(all_overflow)))
        all_normalizable = jnp.all(
            jnp.stack(
                tuple(
                    (~active) | (jnp.sum(weights, axis=-1) > 0.0)
                    for weights in all_weights
                )
            )
        )
        all_finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in all_weights))
        )
        epsilon = jnp.finfo(position.dtype).eps
        successful = (
            jnp.all((~active) | finite_position)
            & ~jnp.any(truncated)
            & ~any_overflow
            & all_normalizable
            & all_finite
            & (maximum_partition <= 64.0 * epsilon)
        )
        return MACMarkerRelation(
            tuple(all_indices),
            tuple(all_weights),
            tuple(all_valid),
            tuple(all_counts),
            tuple(all_partition),
            tuple(all_overflow),
            truncated,
            periodic_used,
            successful,
            self.prepared_id,
        )

    def gather(
        self,
        relation: MACMarkerRelation,
        face_velocity: FaceVelocity,
        /,
    ) -> Array:
        self._validate_relation(relation)
        velocity = self.operators.validate_velocity(face_velocity)
        components = []
        for axis, value in enumerate(velocity):
            flat = value.reshape((-1,))
            sampled = flat[relation.face_indices[axis]]
            components.append(jnp.sum(relation.weights[axis] * sampled, axis=1))
        return jnp.stack(tuple(components), axis=-1)

    def spread(
        self,
        relation: MACMarkerRelation,
        marker_values: ArrayLike,
        /,
    ) -> FaceVelocity:
        """Apply the marker-to-face adjoint, returning unit-density rate units."""

        self._validate_relation(relation)
        values = jnp.asarray(marker_values, dtype=self.operators.pressure_space.dtype)
        marker_count = relation.weights[0].shape[0]
        if values.shape != (marker_count, self.dimension):
            raise ValueError("marker_values must have shape (markers,dimension).")
        output = []
        for axis, layout in enumerate(self.operators.discretization.face_layouts):
            weights = relation.weights[axis]
            payload = jnp.where(
                relation.valid[axis], weights * values[:, axis, None], 0.0
            )
            integrated = (
                jnp.zeros(self.flattened_dual_measures[axis].shape, dtype=values.dtype)
                .at[relation.face_indices[axis].reshape((-1,))]
                .add(payload.reshape((-1,)))
            )
            output.append(
                (integrated / self.flattened_dual_measures[axis]).reshape(layout.shape)
            )
        return tuple(output)

    def diagnostics(
        self,
        relation: MACMarkerRelation,
        face_velocity: FaceVelocity,
        marker_values: ArrayLike,
        /,
    ) -> MACMarkerTransferDiagnostics:
        self._validate_relation(relation)
        velocity = self.operators.validate_velocity(face_velocity)
        values = jnp.asarray(marker_values, dtype=self.operators.pressure_space.dtype)
        marker_count = relation.weights[0].shape[0]
        if values.shape != (marker_count, self.dimension):
            raise ValueError("marker_values must have shape (markers,dimension).")
        sampled = self.gather(relation, velocity)
        spread = self.spread(relation, values)
        interpolation_work = jnp.sum(sampled * values)
        spreading_work = jnp.real(self.operators.velocity_space.inner(velocity, spread))
        work_residual = spreading_work - interpolation_work
        marker_resultant = jnp.sum(values, axis=0)
        face_resultant = jnp.stack(
            tuple(
                jnp.sum(measure * component)
                for measure, component in zip(
                    self.operators.face_dual_measures, spread, strict=True
                )
            )
        )
        force_residual = face_resultant - marker_resultant
        partition = jnp.stack(relation.partition_residual)
        overflow = jnp.stack(relation.capacity_overflow)
        finite = (
            jnp.all(jnp.isfinite(sampled))
            & jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in spread)))
            & jnp.all(jnp.isfinite(force_residual))
            & jnp.isfinite(work_residual)
        )
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.abs(interpolation_work),
                jnp.maximum(
                    jnp.max(jnp.abs(marker_resultant)),
                    jnp.max(jnp.abs(face_resultant)),
                ),
            ),
        )
        tolerance = 1024.0 * jnp.finfo(values.dtype).eps * scale
        successful = (
            relation.successful
            & finite
            & (jnp.abs(work_residual) <= tolerance)
            & (jnp.max(jnp.abs(force_residual)) <= tolerance)
        )
        return MACMarkerTransferDiagnostics(
            marker_resultant,
            face_resultant,
            force_residual,
            interpolation_work,
            spreading_work,
            work_residual,
            jnp.max(partition),
            overflow,
            relation.support_truncated,
            relation.periodic_image_used,
            finite,
            successful,
            self.prepared_id,
        )

    def _validate_relation(self, relation: MACMarkerRelation, /) -> None:
        if not isinstance(relation, MACMarkerRelation):
            raise TypeError("relation must be MACMarkerRelation.")
        if relation.transfer_id != self.prepared_id:
            raise ValueError("MAC marker relation belongs to another transfer.")


__all__ = [
    "MACMarkerRelation",
    "MACMarkerTransferDiagnostics",
    "MACMarkerTransferPlan",
    "PreparedMACMarkerTransfer",
]
