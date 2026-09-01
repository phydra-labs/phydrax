#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


MarkerTopologyDifferentiationPolicy = Literal["frozen-schedule", "event-map"]


class MarkerEpochPlan(StrictModule, NonTrainableState):
    marker_ids: Array
    active_mask: Array
    quadrature_weight: Array
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        marker_ids: ArrayLike,
        quadrature_weight: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        epoch_id: str | None = None,
    ):
        ids = np.asarray(marker_ids)
        weights = np.asarray(quadrature_weight)
        active = (
            np.ones(ids.shape, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("marker_ids must be a rank-one integer array.")
        if ids.size == 0 or np.unique(ids).size != ids.size:
            raise ValueError("marker_ids must be nonempty and unique.")
        if weights.shape != ids.shape or active.shape != ids.shape:
            raise ValueError("Marker epoch arrays must share one capacity shape.")
        if np.any(~np.isfinite(weights[active])) or np.any(weights[active] <= 0.0):
            raise ValueError("Active marker weights must be positive and finite.")
        generated = canonical_fingerprint(
            {
                "kind": "marker-topology-epoch",
                "arrays": array_tree_fingerprint((ids, weights, active)),
            }
        )
        identifier = generated if epoch_id is None else str(epoch_id)
        if not identifier:
            raise ValueError("epoch_id must be nonempty.")
        self.marker_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.active_mask = jnp.asarray(active)
        self.quadrature_weight = jnp.asarray(weights)
        self.epoch_id = identifier

    @property
    def capacity(self) -> int:
        return int(self.marker_ids.size)


class MarkerEpochState(StrictModule):
    value: Array
    time: Array
    accepted_steps: Array
    epoch_index: Array
    epoch_id: str = eqx.field(static=True)


class MarkerEpochTransitionRecord(StrictModule):
    source_epoch_id: str = eqx.field(static=True)
    target_epoch_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    differentiation_policy: MarkerTopologyDifferentiationPolicy = eqx.field(static=True)
    event_parameter: Array
    conservation_residual: Array
    finite: Array
    accepted: Array


class MarkerEpochTransitionResult(StrictModule):
    previous: MarkerEpochState
    candidate: MarkerEpochState
    record: MarkerEpochTransitionRecord
    successful: Array


class MarkerEpochTransferPlan(StrictModule, NonTrainableState):
    """Conservative primal/dual transfer across a marker topology event."""

    source: MarkerEpochPlan
    target: MarkerEpochPlan
    primal: Array
    dual: Array
    differentiation_policy: MarkerTopologyDifferentiationPolicy = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: MarkerEpochPlan,
        target: MarkerEpochPlan,
        primal: ArrayLike,
        /,
        *,
        differentiation_policy: MarkerTopologyDifferentiationPolicy = "frozen-schedule",
        tolerance: float = 1.0e-10,
    ):
        matrix = np.asarray(primal)
        expected = (target.capacity, source.capacity)
        tolerance_ = float(tolerance)
        if matrix.shape != expected or np.any(~np.isfinite(matrix)):
            raise ValueError(f"primal must be finite with shape {expected}.")
        if differentiation_policy not in ("frozen-schedule", "event-map"):
            raise ValueError("Unknown marker topology differentiation policy.")
        if tolerance_ <= 0.0 or not np.isfinite(tolerance_):
            raise ValueError("Marker epoch tolerance must be positive and finite.")
        source_weight = np.asarray(source.quadrature_weight)
        target_weight = np.asarray(target.quadrature_weight)
        source_active = np.asarray(source.active_mask)
        target_active = np.asarray(target.active_mask)
        active_matrix = matrix * target_active[:, None] * source_active[None, :]
        constant = active_matrix @ source_active.astype(matrix.dtype)
        if not np.allclose(
            constant[target_active], 1.0, atol=tolerance_, rtol=tolerance_
        ):
            raise ValueError("Marker epoch primal transfer must preserve constants.")
        conservation = target_weight @ active_matrix
        if not np.allclose(
            conservation[source_active],
            source_weight[source_active],
            atol=tolerance_,
            rtol=tolerance_,
        ):
            raise ValueError(
                "Marker epoch primal transfer must preserve weighted integrals."
            )
        dual = (
            active_matrix.T
            * target_weight[None, :]
            / np.where(source_active, source_weight, 1.0)[:, None]
        )
        self.source = source
        self.target = target
        self.primal = jnp.asarray(active_matrix)
        self.dual = jnp.asarray(dual)
        self.differentiation_policy = differentiation_policy
        self.tolerance = tolerance_
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "marker-epoch-transfer",
                "source": source.epoch_id,
                "target": target.epoch_id,
                "primal": array_tree_fingerprint(matrix),
                "differentiation_policy": differentiation_policy,
                "tolerance": tolerance_,
            }
        )

    def apply_primal(self, value: ArrayLike, /) -> Array:
        source_value = jnp.asarray(value)
        if source_value.shape[0] != self.source.capacity:
            raise ValueError("Marker epoch source value has the wrong capacity.")
        transferred = contract("ts,s...->t...", self.primal, source_value)
        mask = self.target.active_mask.reshape(
            (self.target.capacity,) + (1,) * (transferred.ndim - 1)
        )
        return jnp.where(mask, transferred, 0.0)

    def apply_dual(self, value: ArrayLike, /) -> Array:
        target_value = jnp.asarray(value)
        if target_value.shape[0] != self.target.capacity:
            raise ValueError("Marker epoch target dual has the wrong capacity.")
        transferred = contract("st,t...->s...", self.dual, target_value)
        mask = self.source.active_mask.reshape(
            (self.source.capacity,) + (1,) * (transferred.ndim - 1)
        )
        return jnp.where(mask, transferred, 0.0)

    def transition(
        self,
        state: MarkerEpochState,
        event_parameter: ArrayLike,
        /,
    ) -> MarkerEpochTransitionResult:
        if state.epoch_id != self.source.epoch_id:
            raise ValueError("Marker state belongs to another source epoch.")
        candidate_value = self.apply_primal(state.value)
        source_integral = jnp.sum(
            self.source.quadrature_weight.reshape(
                (self.source.capacity,) + (1,) * (state.value.ndim - 1)
            )
            * state.value,
            axis=0,
        )
        target_integral = jnp.sum(
            self.target.quadrature_weight.reshape(
                (self.target.capacity,) + (1,) * (candidate_value.ndim - 1)
            )
            * candidate_value,
            axis=0,
        )
        residual = target_integral - source_integral
        finite = jnp.all(jnp.isfinite(candidate_value)) & jnp.all(jnp.isfinite(residual))
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(source_integral)))
        successful = finite & (jnp.max(jnp.abs(residual)) <= self.tolerance * scale)
        candidate = MarkerEpochState(
            candidate_value,
            state.time,
            state.accepted_steps,
            state.epoch_index + 1,
            self.target.epoch_id,
        )
        record = MarkerEpochTransitionRecord(
            self.source.epoch_id,
            self.target.epoch_id,
            self.transfer_id,
            self.differentiation_policy,
            jnp.asarray(event_parameter),
            residual,
            finite,
            successful,
        )
        return MarkerEpochTransitionResult(state, candidate, record, record.accepted)


class MarkerMechanicsMigrationResult(StrictModule):
    position: Array
    velocity: Array
    force_density: Array
    source_force: Array
    target_force: Array
    force_residual: Array
    source_torque: Array
    target_torque: Array
    torque_residual: Array
    finite: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class MarkerMechanicsMigrationPlan(StrictModule, NonTrainableState):
    """Conservative marker kinematics/load migration at a mechanics remesh epoch."""

    transfer: MarkerEpochTransferPlan
    ambient_dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: MarkerEpochTransferPlan,
        ambient_dimension: int,
        /,
        *,
        tolerance: float = 1.0e-10,
    ):
        if not isinstance(transfer, MarkerEpochTransferPlan):
            raise TypeError("transfer must be MarkerEpochTransferPlan.")
        dimension = int(ambient_dimension)
        tolerance_ = float(tolerance)
        if dimension not in (2, 3) or tolerance_ <= 0.0:
            raise ValueError("Mechanics migration dimension/tolerance is invalid.")
        self.transfer = transfer
        self.ambient_dimension = dimension
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "marker-mechanics-migration",
                "transfer": transfer.transfer_id,
                "ambient_dimension": dimension,
                "tolerance": tolerance_,
            }
        )

    def migrate(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        force_density: ArrayLike,
        /,
        *,
        torque_origin: ArrayLike | None = None,
    ) -> MarkerMechanicsMigrationResult:
        source_shape = (self.transfer.source.capacity, self.ambient_dimension)
        position_ = jnp.asarray(position)
        velocity_ = jnp.asarray(velocity, dtype=position_.dtype)
        force_ = jnp.asarray(force_density, dtype=position_.dtype)
        if (
            position_.shape != source_shape
            or velocity_.shape != source_shape
            or force_.shape != source_shape
        ):
            raise ValueError(
                "Mechanics migration arrays must have source-capacity vector shape."
            )
        target_position = self.transfer.apply_primal(position_)
        target_velocity = self.transfer.apply_primal(velocity_)
        target_force_density = self.transfer.apply_primal(force_)
        source_weight = self.transfer.source.quadrature_weight.astype(position_.dtype)
        target_weight = self.transfer.target.quadrature_weight.astype(position_.dtype)
        source_force = jnp.sum(source_weight[:, None] * force_, axis=0)
        target_force = jnp.sum(target_weight[:, None] * target_force_density, axis=0)
        origin = (
            jnp.zeros((self.ambient_dimension,), dtype=position_.dtype)
            if torque_origin is None
            else jnp.asarray(torque_origin, dtype=position_.dtype)
        )
        if origin.shape != (self.ambient_dimension,):
            raise ValueError("torque_origin must have ambient-dimension shape.")
        source_arm = position_ - origin
        target_arm = target_position - origin
        if self.ambient_dimension == 2:
            source_torque = jnp.sum(
                source_weight
                * (source_arm[:, 0] * force_[:, 1] - source_arm[:, 1] * force_[:, 0])
            ).reshape((1,))
            target_torque = jnp.sum(
                target_weight
                * (
                    target_arm[:, 0] * target_force_density[:, 1]
                    - target_arm[:, 1] * target_force_density[:, 0]
                )
            ).reshape((1,))
        else:
            source_torque = jnp.sum(
                source_weight[:, None] * jnp.cross(source_arm, force_), axis=0
            )
            target_torque = jnp.sum(
                target_weight[:, None] * jnp.cross(target_arm, target_force_density),
                axis=0,
            )
        force_residual = target_force - source_force
        torque_residual = target_torque - source_torque
        finite = (
            jnp.all(jnp.isfinite(target_position))
            & jnp.all(jnp.isfinite(target_velocity))
            & jnp.all(jnp.isfinite(target_force_density))
            & jnp.all(jnp.isfinite(force_residual))
            & jnp.all(jnp.isfinite(torque_residual))
        )
        scale = jnp.maximum(
            1.0,
            jnp.max(jnp.abs(source_force)) + jnp.max(jnp.abs(source_torque)),
        )
        successful = (
            finite
            & (jnp.max(jnp.abs(force_residual)) <= self.tolerance * scale)
            & (jnp.max(jnp.abs(torque_residual)) <= self.tolerance * scale)
        )
        return MarkerMechanicsMigrationResult(
            target_position,
            target_velocity,
            target_force_density,
            source_force,
            target_force,
            force_residual,
            source_torque,
            target_torque,
            torque_residual,
            finite,
            successful,
            self.transfer.transfer_id,
        )


__all__ = [
    "MarkerMechanicsMigrationPlan",
    "MarkerMechanicsMigrationResult",
    "MarkerEpochPlan",
    "MarkerEpochState",
    "MarkerEpochTransferPlan",
    "MarkerEpochTransitionRecord",
    "MarkerEpochTransitionResult",
    "MarkerTopologyDifferentiationPolicy",
]
