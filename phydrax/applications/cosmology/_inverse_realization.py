#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.splatting import PreparedParticleGridSplat
from ...solver._multiphysics_inference import (
    FieldObservationPlan,
    SimulationSensitivityReport,
)


ParticleTargetKind = Literal["density", "extensive-content"]


class ParticleFieldRealizationEvaluation(StrictModule):
    positions: Array
    predicted_observation: Array
    residual: Array
    predicted_content: Array
    predicted_density: Array
    log_likelihood: Array
    objective: Array
    mass_balance_defect: Array
    captured_fraction_minimum: Array
    support_complete: Array
    seam_distance: Array
    finite: Array
    successful: Array
    particle_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class ParticleFieldRealizationPlan(StrictModule, NonTrainableState):
    """Covariance-aware inverse realization over existing particle-grid splatting."""

    transfer: PreparedParticleGridSplat
    observation: FieldObservationPlan
    target_kind: ParticleTargetKind = eqx.field(static=True)
    lower_bounds: tuple[float, ...] = eqx.field(static=True)
    box_lengths: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedParticleGridSplat,
        observation: FieldObservationPlan,
        /,
        *,
        target_kind: ParticleTargetKind = "density",
        plan_id: str,
    ):
        if not isinstance(transfer, PreparedParticleGridSplat):
            raise TypeError("transfer must be PreparedParticleGridSplat.")
        if not isinstance(observation, FieldObservationPlan):
            raise TypeError("observation must be FieldObservationPlan.")
        if transfer.plan.execution.geometry_ad != "piecewise":
            raise ValueError(
                "Particle position inference requires piecewise geometry AD."
            )
        if target_kind not in ("density", "extensive-content"):
            raise ValueError("Unknown particle-field target kind.")
        axes = transfer.plan.target.axes
        if any(not axis.periodic or axis.bounds is None for axis in axes):
            raise ValueError(
                "Wrapped particle realization requires finite periodic axes."
            )
        lower = tuple(float(np.asarray(axis.bounds)[0]) for axis in axes)
        lengths = tuple(
            float(np.asarray(axis.bounds)[1] - np.asarray(axis.bounds)[0])
            for axis in axes
        )
        if not plan_id:
            raise ValueError("Particle realization plan_id must be non-empty.")
        self.transfer = transfer
        self.observation = observation
        self.target_kind = target_kind
        self.lower_bounds = lower
        self.box_lengths = lengths
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-field-realization",
                "declared_id": plan_id,
                "particles": transfer.particles.prepared_id,
                "transfer": transfer.prepared_id,
                "observation": observation.observation_id,
                "target_kind": target_kind,
                "lower_bounds": list(lower),
                "box_lengths": list(lengths),
            }
        )

    def positions(self, raw_positions: ArrayLike, /) -> Array:
        raw = jnp.asarray(raw_positions)
        expected = (
            self.transfer.particles.capacity,
            self.transfer.particles.ambient_dimension,
        )
        if raw.shape != expected:
            raise ValueError(f"Particle position parameters must have shape {expected}.")
        lower = jnp.asarray(self.lower_bounds, dtype=raw.dtype)
        lengths = jnp.asarray(self.box_lengths, dtype=raw.dtype)
        return lower + jnp.mod(raw - lower, lengths)

    def evaluate(
        self,
        raw_positions: ArrayLike,
        args: Any = None,
        /,
    ) -> ParticleFieldRealizationEvaluation:
        positions = self.positions(raw_positions)
        routes = self.transfer.build(positions)
        deposited = self.transfer.deposit_content(routes, self.transfer.particles.masses)
        content = deposited.content
        density = deposited.density
        predicted = density if self.target_kind == "density" else content
        log_likelihood = self.observation.log_likelihood(predicted, args)
        lower = jnp.asarray(self.lower_bounds, dtype=positions.dtype)
        upper = lower + jnp.asarray(self.box_lengths, dtype=positions.dtype)
        active = self.transfer.particles.active_mask[:, None]
        seam = jnp.minimum(positions - lower, upper - positions)
        seam_distance = jnp.min(jnp.where(active, seam, jnp.inf))
        predicted_observation = jnp.asarray(
            self.observation.operator(predicted, args)
        ).reshape((-1,))
        residual = predicted_observation - self.observation.observed
        support_complete = deposited.successful
        finite = (
            jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(content))
            & jnp.all(jnp.isfinite(density))
            & jnp.all(jnp.isfinite(predicted_observation))
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(log_likelihood)
        )
        successful = finite & support_complete
        return ParticleFieldRealizationEvaluation(
            positions,
            predicted_observation,
            residual,
            content,
            density,
            log_likelihood,
            -log_likelihood,
            deposited.balance.maximum_absolute_balance_defect,
            jnp.min(
                deposited.balance.supported_source_total
                / jnp.maximum(
                    deposited.balance.active_source_total,
                    jnp.finfo(content.dtype).tiny,
                )
            ),
            support_complete,
            seam_distance,
            finite,
            successful,
            self.transfer.particles.prepared_id,
            self.transfer.prepared_id,
            self.observation.observation_id,
            self.plan_id,
        )

    def objective(self, raw_positions: ArrayLike, args: Any = None, /) -> Array:
        return self.evaluate(raw_positions, args).objective

    def value_and_gradient(
        self, raw_positions: ArrayLike, args: Any = None, /
    ) -> tuple[Array, Array]:
        return jax.value_and_grad(lambda value: self.objective(value, args))(
            jnp.asarray(raw_positions)
        )

    def sensitivity(
        self,
        raw_positions: ArrayLike,
        direction: ArrayLike,
        args: Any = None,
        /,
        *,
        epsilon: float = 1.0e-5,
    ) -> SimulationSensitivityReport:
        return SimulationSensitivityReport.evaluate(
            lambda value: self.objective(value, args),
            raw_positions,
            direction,
            epsilon=epsilon,
        )


__all__ = [
    "ParticleFieldRealizationEvaluation",
    "ParticleFieldRealizationPlan",
    "ParticleTargetKind",
]
