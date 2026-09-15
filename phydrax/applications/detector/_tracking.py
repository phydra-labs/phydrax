#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


class TrackMeasurementBank(StrictModule, NonTrainableState):
    event_ids: Array
    positions: Array
    times: Array
    variances: Array
    surface_ids: Array
    active: Array
    valid: Array
    association_id: str = eqx.field(static=True)
    conditions_id: str = eqx.field(static=True)
    track_capacity: int = eqx.field(static=True)
    measurement_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        event_ids: ArrayLike,
        positions: ArrayLike,
        times: ArrayLike,
        variances: ArrayLike,
        surface_ids: ArrayLike,
        active: ArrayLike,
        association_id: str,
        conditions_id: str,
    ):
        event_ids_ = jnp.asarray(event_ids)
        positions_ = jnp.asarray(positions)
        times_ = jnp.asarray(times, dtype=positions_.dtype)
        variances_ = jnp.asarray(variances, dtype=positions_.dtype)
        surfaces = jnp.asarray(surface_ids, dtype=jnp.int32)
        active_ = jnp.asarray(active, dtype=bool)
        if positions_.ndim != 4 or positions_.shape[-1] != 3:
            raise ValueError("positions must have shape (event, track, measurement, 3).")
        expected = positions_.shape[:-1]
        if (
            times_.shape != expected
            or surfaces.shape != expected
            or active_.shape != expected
            or variances_.shape != positions_.shape
            or event_ids_.shape != (expected[0],)
        ):
            raise ValueError("Track measurement fields must align.")
        association = str(association_id).strip()
        conditions = str(conditions_id).strip()
        if not association or not conditions:
            raise ValueError("association_id and conditions_id are required.")
        valid = (
            jnp.all(jnp.isfinite(positions_), axis=-1)
            & jnp.isfinite(times_)
            & jnp.all(jnp.isfinite(variances_) & (variances_ > 0.0), axis=-1)
            & (surfaces >= 0)
        )
        self.event_ids = event_ids_
        self.positions = positions_
        self.times = times_
        self.variances = variances_
        self.surface_ids = surfaces
        self.active = active_
        self.valid = jnp.where(active_, valid, True)
        self.association_id = association
        self.conditions_id = conditions
        self.track_capacity = int(expected[1])
        self.measurement_capacity = int(expected[2])


class TrackFitPlan(StrictModule, NonTrainableState):
    regularization: float = eqx.field(static=True)
    minimum_measurements: int = eqx.field(static=True)
    association_id: str = eqx.field(static=True)
    conditions_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        association_id: str,
        conditions_id: str,
        minimum_measurements: int = 2,
        regularization: float = 1.0e-12,
    ):
        association = str(association_id).strip()
        conditions = str(conditions_id).strip()
        minimum = int(minimum_measurements)
        regularization_ = float(regularization)
        if not association or not conditions or minimum < 2 or regularization_ < 0.0:
            raise ValueError(
                "Track-fit association, conditions, and numerical policy are invalid."
            )
        self.regularization = regularization_
        self.minimum_measurements = minimum
        self.association_id = association
        self.conditions_id = conditions
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-association-linear-track-fit",
                "association": association,
                "conditions": conditions,
                "minimum_measurements": minimum,
                "regularization": regularization_,
            }
        )


class ReconstructedTrackBank(StrictModule, NonTrainableState):
    event_ids: Array
    parameters: Array
    covariance: Array
    chi_square: Array
    degrees_of_freedom: Array
    active: Array
    valid: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


def fit_associated_tracks(
    plan: TrackFitPlan,
    measurements: TrackMeasurementBank,
    /,
) -> ReconstructedTrackBank:
    """Fit fixed associations to straight six-parameter trajectories."""
    if not isinstance(plan, TrackFitPlan) or not isinstance(
        measurements, TrackMeasurementBank
    ):
        raise TypeError("plan and measurements must use detector tracking types.")
    if (
        plan.association_id != measurements.association_id
        or plan.conditions_id != measurements.conditions_id
    ):
        raise ValueError("Track fit and measurement association/conditions differ.")
    _, _, measurement_count, _ = measurements.positions.shape
    identity = jnp.eye(3, dtype=measurements.positions.dtype)

    def one(positions, times, variances, active, valid):
        admitted = active & valid
        h = jnp.concatenate(
            (
                jnp.broadcast_to(identity, (measurement_count, 3, 3)),
                times[:, None, None] * identity[None, :, :],
            ),
            axis=-1,
        )
        weights = jnp.where(admitted[:, None], 1.0 / variances, 0.0)
        normal = ein.contract("mdi,md,mdj->ij", h, weights, h)
        normal = normal + plan.regularization * jnp.eye(6, dtype=normal.dtype)
        rhs = ein.contract("mdi,md,md->i", h, weights, positions)
        solution = solve(
            LinearSystem(DenseLinearOperator(normal)),
            rhs,
            policy=LinearSolvePolicy(DenseLU()),
        )
        covariance_result = solve(
            LinearSystem(DenseLinearOperator(normal)),
            jnp.eye(6, dtype=normal.dtype),
            policy=LinearSolvePolicy(DenseLU()),
        )
        predicted = ein.contract("mdi,i->md", h, solution.value)
        residual = positions - predicted
        chi_square = jnp.sum(
            jnp.where(admitted[:, None], residual * residual / variances, 0.0)
        )
        count = jnp.sum(admitted, dtype=jnp.int32)
        degrees = jnp.maximum(3 * count - 6, 0)
        successful = (
            (count >= plan.minimum_measurements)
            & jnp.all(solution.status == 0)
            & jnp.all(covariance_result.status == 0)
            & jnp.all(jnp.isfinite(solution.value))
            & jnp.all(jnp.isfinite(covariance_result.value))
        )
        return solution.value, covariance_result.value, chi_square, degrees, successful

    parameters, covariance, chi_square, degrees, successful = jax.vmap(jax.vmap(one))(
        measurements.positions,
        measurements.times,
        measurements.variances,
        measurements.active,
        measurements.valid,
    )
    return ReconstructedTrackBank(
        measurements.event_ids,
        parameters,
        covariance,
        chi_square,
        degrees,
        successful,
        successful,
        successful,
        plan.plan_id,
    )


__all__ = [
    "ReconstructedTrackBank",
    "TrackFitPlan",
    "TrackMeasurementBank",
    "fit_associated_tracks",
]
