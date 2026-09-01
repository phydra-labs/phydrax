#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._lagrangian_marker import LagrangianMarkerDiscretization
from ._incompressible import FaceVelocity
from ._mac_marker_transfer import MACMarkerAccumulation


CompositeFaceVelocity = tuple[FaceVelocity, ...]


class CompositeMACMarkerRelation(StrictModule):
    route_level: tuple[Array, ...]
    route_index: tuple[Array, ...]
    route_valid: tuple[Array, ...]
    weights: tuple[Array, ...]
    marker_position: Array
    partition_residual: tuple[Array, ...]
    first_moment_residual: tuple[Array, ...]
    condition_number: tuple[Array, ...]
    finite: Array
    successful: Array
    relation_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)


class CompositeMACMarkerTransferDiagnostics(StrictModule):
    interpolation_work: Array
    spreading_work: Array
    work_adjoint_residual: Array
    maximum_partition_residual: Array
    maximum_first_moment_residual: Array
    maximum_condition_number: Array
    finite: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class CompositeMarkerImpulseLedger(StrictModule):
    marker_impulse: Array
    fluid_impulse: Array
    start_time: Array
    end_time: Array
    accepted_substeps: Array
    conservation_residual: Array
    finite: Array
    transfer_id: str = eqx.field(static=True)


class CompositeMACMarkerTransferPlan(StrictModule, NonTrainableState):
    """Finest-owner conservative marker transfer on a fixed AMR hierarchy epoch."""

    markers: LagrangianMarkerDiscretization
    face_shapes: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    face_measures: tuple[FaceVelocity, ...]
    accumulation: MACMarkerAccumulation = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        markers: LagrangianMarkerDiscretization,
        face_measures: Sequence[Sequence[ArrayLike]],
        /,
        *,
        accumulation: MACMarkerAccumulation = "deterministic",
        condition_limit: float = 1.0e10,
    ):
        if not isinstance(markers, LagrangianMarkerDiscretization):
            raise TypeError("markers must be LagrangianMarkerDiscretization.")
        measures = tuple(
            tuple(jnp.asarray(value) for value in level) for level in face_measures
        )
        if not measures or any(
            len(level) != markers.ambient_dimension for level in measures
        ):
            raise ValueError("Each AMR level needs one face measure per dimension.")
        host = tuple(
            tuple(np.asarray(value) for value in level) for level in face_measures
        )
        if any(
            np.any(~np.isfinite(value)) or np.any(value <= 0.0)
            for level in host
            for value in level
        ):
            raise ValueError("Composite face measures must be positive and finite.")
        if accumulation not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown marker accumulation policy.")
        limit = float(condition_limit)
        if not np.isfinite(limit) or limit <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one.")
        self.markers = markers
        self.face_shapes = tuple(tuple(value.shape for value in level) for level in host)
        self.face_measures = measures
        self.accumulation = accumulation
        self.condition_limit = limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "composite-mac-marker-transfer",
                "markers": markers.prepared_id,
                "face_measures": array_tree_fingerprint(host),
                "accumulation": accumulation,
                "condition_limit": limit,
            }
        )

    @property
    def level_count(self) -> int:
        return len(self.face_shapes)

    def relation(
        self,
        marker_position: ArrayLike,
        candidate_level: Sequence[ArrayLike],
        candidate_index: Sequence[ArrayLike],
        candidate_center: Sequence[ArrayLike],
        candidate_valid: Sequence[ArrayLike],
        /,
    ) -> CompositeMACMarkerRelation:
        position_full = jnp.asarray(marker_position)
        expected = (self.markers.capacity, self.markers.ambient_dimension)
        if position_full.shape != expected:
            raise ValueError(f"marker_position must have shape {expected}.")
        position = position_full[self.markers.active_indices]
        if not all(
            len(values) == self.markers.ambient_dimension
            for values in (
                candidate_level,
                candidate_index,
                candidate_center,
                candidate_valid,
            )
        ):
            raise ValueError("Composite candidates need one route block per component.")
        levels = []
        indices = []
        validity = []
        weights = []
        partitions = []
        moments = []
        conditions = []
        successful = jnp.asarray(True)
        for component in range(self.markers.ambient_dimension):
            level = jnp.asarray(candidate_level[component], dtype=jnp.int32)
            index = jnp.asarray(candidate_index[component], dtype=jnp.int32)
            center = jnp.asarray(candidate_center[component], dtype=position.dtype)
            valid = jnp.asarray(candidate_valid[component], dtype=bool)
            if (
                level.shape != index.shape
                or valid.shape != level.shape
                or center.shape != level.shape + (self.markers.ambient_dimension,)
            ):
                raise ValueError("Composite marker candidate shapes are incompatible.")
            if level.shape[0] != self.markers.active_count:
                raise ValueError(
                    "Composite candidates must use active-marker coordinates."
                )
            safe_level = jnp.where(valid, level, -1)
            finest = jnp.max(safe_level, axis=-1, keepdims=True)
            owned = valid & (level == finest)
            offsets = center - position[:, None, :]
            distance_squared = jnp.sum(offsets * offsets, axis=-1)
            scale = jnp.maximum(
                jnp.max(jnp.where(owned, distance_squared, 0.0), axis=-1, keepdims=True),
                jnp.finfo(position.dtype).eps,
            )
            radial = jnp.where(owned, jnp.exp(-4.0 * distance_squared / scale), 0.0)
            basis = jnp.concatenate(
                (jnp.ones(level.shape + (1,), dtype=position.dtype), offsets),
                axis=-1,
            )
            gram = contract("mri,mr,mrj->mij", basis, radial, basis)
            eigenvalues = jnp.linalg.eigvalsh(gram)
            condition = eigenvalues[..., -1] / jnp.maximum(
                eigenvalues[..., 0], jnp.finfo(position.dtype).tiny
            )
            target = (
                jnp.zeros(
                    (position.shape[0], self.markers.ambient_dimension + 1),
                    dtype=position.dtype,
                )
                .at[:, 0]
                .set(1.0)
            )
            coefficients = jnp.linalg.solve(gram, target[..., None])[..., 0]
            component_weight = radial * contract("mri,mi->mr", basis, coefficients)
            captured = jnp.sum(component_weight, axis=-1)
            first = jnp.sum(component_weight[..., None] * offsets, axis=1)
            finite = (
                jnp.all(jnp.isfinite(component_weight), axis=-1)
                & jnp.isfinite(condition)
                & (finest[:, 0] >= 0)
            )
            successful = successful & jnp.all(
                finite & (condition <= self.condition_limit)
            )
            levels.append(jax.lax.stop_gradient(level))
            indices.append(jax.lax.stop_gradient(index))
            validity.append(jax.lax.stop_gradient(owned))
            weights.append(jnp.where(owned, component_weight, 0.0))
            partitions.append(jnp.abs(captured - 1.0))
            moments.append(jnp.abs(first))
            conditions.append(condition)
        finite = jnp.all(jnp.isfinite(position)) & jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in weights))
        )
        return CompositeMACMarkerRelation(
            tuple(levels),
            tuple(indices),
            tuple(validity),
            tuple(weights),
            position,
            tuple(partitions),
            tuple(moments),
            tuple(conditions),
            finite,
            finite & successful,
            canonical_fingerprint(
                {
                    "kind": "composite-mac-marker-relation",
                    "transfer": self.plan_id,
                    "route_width": int(levels[0].shape[1]),
                }
            ),
            self.plan_id,
        )

    def gather(
        self, relation: CompositeMACMarkerRelation, velocity: CompositeFaceVelocity, /
    ) -> Array:
        self._validate_relation(relation)
        self._validate_velocity(velocity)
        components = []
        for component in range(self.markers.ambient_dimension):
            gathered = jnp.zeros(
                (self.markers.active_count,), dtype=relation.marker_position.dtype
            )
            for level in range(self.level_count):
                flattened = velocity[level][component].reshape((-1,))
                selected = flattened[relation.route_index[component]]
                gathered = gathered + jnp.sum(
                    jnp.where(
                        relation.route_valid[component]
                        & (relation.route_level[component] == level),
                        relation.weights[component] * selected,
                        0.0,
                    ),
                    axis=-1,
                )
            components.append(gathered)
        return jnp.stack(tuple(components), axis=-1)

    def spread(
        self,
        relation: CompositeMACMarkerRelation,
        marker_force: ArrayLike,
        /,
    ) -> CompositeFaceVelocity:
        self._validate_relation(relation)
        raw = jnp.asarray(marker_force)
        force = (
            self.markers.active_values(raw)
            if raw.shape == (self.markers.capacity, self.markers.ambient_dimension)
            else self.markers.active_velocity_space.validate(raw)
        )
        quadrature = self.markers.plan.quadrature_weight[
            self.markers.active_indices
        ].astype(force.dtype)
        level_output = [
            [jnp.zeros(shape, dtype=force.dtype) for shape in shapes]
            for shapes in self.face_shapes
        ]
        order = self.markers.stable_active_order
        for component in range(self.markers.ambient_dimension):
            contribution = (
                relation.weights[component]
                * force[:, component, None]
                * quadrature[:, None]
            )
            for level in range(self.level_count):
                flat = level_output[level][component].reshape((-1,))
                valid = relation.route_valid[component] & (
                    relation.route_level[component] == level
                )
                if self.accumulation == "fast":
                    flat = flat.at[relation.route_index[component]].add(
                        jnp.where(valid, contribution, 0.0)
                    )
                else:

                    def add_marker(
                        marker_order,
                        values,
                        component_=component,
                        valid_=valid,
                        contribution_=contribution,
                    ):
                        marker = order[marker_order]
                        return values.at[relation.route_index[component_][marker]].add(
                            jnp.where(valid_[marker], contribution_[marker], 0.0)
                        )

                    flat = jax.lax.fori_loop(0, order.size, add_marker, flat)
                level_output[level][component] = flat.reshape(
                    self.face_shapes[level][component]
                ) / self.face_measures[level][component].astype(force.dtype)
        return tuple(tuple(level) for level in level_output)

    def diagnostics(
        self,
        relation: CompositeMACMarkerRelation,
        velocity: CompositeFaceVelocity,
        marker_force: ArrayLike,
        /,
    ) -> CompositeMACMarkerTransferDiagnostics:
        gathered = self.gather(relation, velocity)
        raw = jnp.asarray(marker_force)
        force = (
            self.markers.active_values(raw)
            if raw.shape == (self.markers.capacity, self.markers.ambient_dimension)
            else self.markers.active_velocity_space.validate(raw)
        )
        spread = self.spread(relation, force)
        interpolation_work = jnp.real(
            self.markers.active_velocity_space.inner(gathered, force)
        )
        spreading_work = sum(
            jnp.sum(
                self.face_measures[level][component]
                * velocity[level][component]
                * spread[level][component]
            )
            for level in range(self.level_count)
            for component in range(self.markers.ambient_dimension)
        )
        residual = interpolation_work - spreading_work
        finite = relation.finite & jnp.isfinite(residual)
        tolerance = (
            512.0
            * jnp.finfo(force.dtype).eps
            * jnp.maximum(1.0, jnp.abs(interpolation_work) + jnp.abs(spreading_work))
        )
        return CompositeMACMarkerTransferDiagnostics(
            interpolation_work,
            spreading_work,
            residual,
            jnp.max(jnp.stack(relation.partition_residual)),
            jnp.max(jnp.stack(relation.first_moment_residual)),
            jnp.max(jnp.stack(relation.condition_number)),
            finite,
            relation.successful & finite & (jnp.abs(residual) <= tolerance),
            self.plan_id,
        )

    def impulse_ledger(
        self,
        relation: CompositeMACMarkerRelation,
        marker_force: ArrayLike,
        start_time: ArrayLike,
        end_time: ArrayLike,
        /,
        *,
        accepted_substeps: ArrayLike = 1,
    ) -> CompositeMarkerImpulseLedger:
        start = jnp.asarray(start_time)
        end = jnp.asarray(end_time)
        step = end - start
        raw = jnp.asarray(marker_force)
        force = (
            self.markers.active_values(raw)
            if raw.shape == (self.markers.capacity, self.markers.ambient_dimension)
            else self.markers.active_velocity_space.validate(raw)
        )
        quadrature = self.markers.plan.quadrature_weight[
            self.markers.active_indices
        ].astype(force.dtype)
        marker_impulse = step * jnp.sum(quadrature[:, None] * force, axis=0)
        spread = self.spread(relation, force)
        fluid_impulse = jnp.stack(
            tuple(
                step
                * sum(
                    jnp.sum(
                        self.face_measures[level][component] * spread[level][component]
                    )
                    for level in range(self.level_count)
                )
                for component in range(self.markers.ambient_dimension)
            )
        )
        residual = fluid_impulse - marker_impulse
        finite = jnp.all(jnp.isfinite(residual)) & (step > 0.0)
        return CompositeMarkerImpulseLedger(
            marker_impulse,
            fluid_impulse,
            start,
            end,
            jnp.asarray(accepted_substeps, dtype=jnp.int32),
            residual,
            finite,
            self.plan_id,
        )

    def _validate_relation(self, relation: CompositeMACMarkerRelation, /) -> None:
        if not isinstance(relation, CompositeMACMarkerRelation):
            raise TypeError("relation must be CompositeMACMarkerRelation.")
        if relation.transfer_id != self.plan_id:
            raise ValueError("Composite marker relation belongs to another transfer.")

    def _validate_velocity(self, velocity: CompositeFaceVelocity, /) -> None:
        if len(velocity) != self.level_count:
            raise ValueError("Composite velocity level count differs from the plan.")
        for level, values in enumerate(velocity):
            if len(values) != self.markers.ambient_dimension or any(
                value.shape != self.face_shapes[level][component]
                for component, value in enumerate(values)
            ):
                raise ValueError("Composite velocity has an incompatible face layout.")


class CompositeMarkerImpulseReflux(StrictModule):
    coarse_fluid_impulse: Array
    fine_fluid_impulse: Array
    correction_impulse: Array
    marker_impulse: Array
    conservation_residual: Array
    start_time: Array
    end_time: Array
    fine_substeps: Array
    finite: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


def reflux_composite_marker_impulse(
    coarse: CompositeMarkerImpulseLedger,
    fine: tuple[CompositeMarkerImpulseLedger, ...],
    /,
    *,
    tolerance: float = 1.0e-10,
) -> CompositeMarkerImpulseReflux:
    if not isinstance(coarse, CompositeMarkerImpulseLedger) or not fine:
        raise TypeError(
            "Marker impulse reflux needs one coarse and at least one fine ledger."
        )
    if any(
        not isinstance(ledger, CompositeMarkerImpulseLedger)
        or ledger.transfer_id != coarse.transfer_id
        for ledger in fine
    ):
        raise ValueError("Marker impulse reflux ledgers have incompatible identities.")
    tolerance_ = float(tolerance)
    if tolerance_ <= 0.0:
        raise ValueError("Marker impulse reflux tolerance must be positive.")
    continuity = jnp.asarray(True)
    for left, right in zip(fine[:-1], fine[1:], strict=True):
        continuity = continuity & (left.end_time == right.start_time)
    interval = (fine[0].start_time == coarse.start_time) & (
        fine[-1].end_time == coarse.end_time
    )
    fine_fluid = sum(
        (ledger.fluid_impulse for ledger in fine),
        jnp.zeros_like(coarse.fluid_impulse),
    )
    fine_marker = sum(
        (ledger.marker_impulse for ledger in fine),
        jnp.zeros_like(coarse.marker_impulse),
    )
    correction = fine_fluid - coarse.fluid_impulse
    residual = fine_fluid - fine_marker
    finite = (
        coarse.finite
        & jnp.all(jnp.stack(tuple(ledger.finite for ledger in fine)))
        & jnp.all(jnp.isfinite(correction))
        & jnp.all(jnp.isfinite(residual))
    )
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(fine_marker)))
    successful = (
        finite
        & continuity
        & interval
        & (jnp.max(jnp.abs(residual)) <= tolerance_ * scale)
    )
    return CompositeMarkerImpulseReflux(
        coarse.fluid_impulse,
        fine_fluid,
        correction,
        fine_marker,
        residual,
        coarse.start_time,
        coarse.end_time,
        sum(ledger.accepted_substeps for ledger in fine),
        finite,
        successful,
        coarse.transfer_id,
    )


__all__ = [
    "CompositeFaceVelocity",
    "CompositeMACMarkerRelation",
    "CompositeMACMarkerTransferDiagnostics",
    "CompositeMACMarkerTransferPlan",
    "CompositeMarkerImpulseReflux",
    "CompositeMarkerImpulseLedger",
    "reflux_composite_marker_impulse",
]
