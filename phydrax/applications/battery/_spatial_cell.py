#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dfn import DFNParameters, DFNState, IsothermalDFNPlan


class SpatialBatteryCellState(StrictModule):
    electrolyte_concentration_mol_m3: Array
    negative_particle_concentration_mol_m3: Array
    positive_particle_concentration_mol_m3: Array
    time_s: Array


class SpatialBatteryCellEvidence(StrictModule):
    local_accepted: Array
    current_balance_defect_a_m2: Array
    voltage_spread_v: Array
    finite: Array
    successful: Array


class SpatialBatteryCellStepResult(StrictModule):
    candidate_state: SpatialBatteryCellState
    accepted_state: SpatialBatteryCellState
    local_voltage_v: Array
    terminal_voltage_v: Array
    local_residual_norm: Array
    evidence: SpatialBatteryCellEvidence


class SpatialBatteryCellPlan(StrictModule, NonTrainableState):
    """Homogeneous local 1D DFN lanes distributed over fixed 2D/3D macro sites."""

    local_plan: IsothermalDFNPlan
    site_coordinates: Array
    current_weights: Array
    parameters: DFNParameters = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_plan: IsothermalDFNPlan,
        parameters: DFNParameters,
        site_coordinates: ArrayLike,
        /,
        *,
        current_weights: ArrayLike | None = None,
    ):
        if not isinstance(local_plan, IsothermalDFNPlan):
            raise TypeError("local_plan must be an IsothermalDFNPlan.")
        if not isinstance(parameters, DFNParameters):
            raise TypeError("parameters must be DFNParameters.")
        coordinates = np.asarray(site_coordinates, dtype=np.float64)
        if (
            coordinates.ndim != 2
            or coordinates.shape[0] == 0
            or coordinates.shape[1] not in (2, 3)
            or np.any(~np.isfinite(coordinates))
        ):
            raise ValueError(
                "site_coordinates must have finite shape (site_count, 2 or 3)."
            )
        count = coordinates.shape[0]
        if current_weights is None:
            weights = np.full((count,), 1.0 / count, dtype=np.float64)
        else:
            weights = np.asarray(current_weights, dtype=np.float64)
        if (
            weights.shape != (count,)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1e-14)
        ):
            raise ValueError(
                "current_weights must be finite, nonnegative, and sum exactly to one."
            )
        self.local_plan = local_plan
        self.parameters = parameters
        self.site_coordinates = jnp.asarray(coordinates)
        self.current_weights = jnp.asarray(weights)
        self.spatial_dimension = coordinates.shape[1]
        self.site_count = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spatial-battery-cell-plan",
                "local_plan": local_plan.plan_id,
                "coordinates": array_tree_fingerprint(coordinates),
                "current_weights": weights.tolist(),
            }
        )

    def initialize(
        self,
        *,
        electrolyte_concentration_mol_m3: float,
        negative_stoichiometry: float,
        positive_stoichiometry: float,
    ) -> SpatialBatteryCellState:
        local = self.local_plan.initial_state(
            self.parameters,
            electrolyte_concentration_mol_m3=electrolyte_concentration_mol_m3,
            negative_stoichiometry=negative_stoichiometry,
            positive_stoichiometry=positive_stoichiometry,
        )
        return SpatialBatteryCellState(
            jnp.broadcast_to(
                local.electrolyte_concentration_mol_m3,
                (self.site_count, *local.electrolyte_concentration_mol_m3.shape),
            ),
            jnp.broadcast_to(
                local.negative_particle_concentration_mol_m3,
                (
                    self.site_count,
                    *local.negative_particle_concentration_mol_m3.shape,
                ),
            ),
            jnp.broadcast_to(
                local.positive_particle_concentration_mol_m3,
                (
                    self.site_count,
                    *local.positive_particle_concentration_mol_m3.shape,
                ),
            ),
            jnp.zeros((self.site_count,), dtype=local.time_s.dtype),
        )

    def step(
        self,
        state: SpatialBatteryCellState,
        total_current_density_a_m2: ArrayLike,
        step_size_s: ArrayLike,
        /,
    ) -> SpatialBatteryCellStepResult:
        if not isinstance(state, SpatialBatteryCellState):
            raise TypeError("state must be SpatialBatteryCellState.")
        total_current = jnp.asarray(total_current_density_a_m2).reshape(())
        step_size = jnp.asarray(step_size_s).reshape(())
        local_currents = total_current * self.site_count * self.current_weights

        def lane(electrolyte, negative, positive, time, current):
            result = self.local_plan.step(
                DFNState(electrolyte, negative, positive, time),
                self.parameters,
                current,
                step_size,
            )
            return (
                result.candidate_state.electrolyte_concentration_mol_m3,
                result.candidate_state.negative_particle_concentration_mol_m3,
                result.candidate_state.positive_particle_concentration_mol_m3,
                result.candidate_state.time_s,
                result.evaluation.voltage_v,
                result.evaluation.residual_norm,
                result.accepted,
            )

        candidate_values = jax.vmap(lane)(
            state.electrolyte_concentration_mol_m3,
            state.negative_particle_concentration_mol_m3,
            state.positive_particle_concentration_mol_m3,
            state.time_s,
            local_currents,
        )
        candidate = SpatialBatteryCellState(*candidate_values[:4])
        voltage = candidate_values[4]
        residual = candidate_values[5]
        local_accepted = candidate_values[6]
        current_defect = jnp.mean(local_currents) - total_current
        voltage_spread = jnp.max(voltage) - jnp.min(voltage)
        finite = (
            jnp.all(jnp.isfinite(voltage))
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(current_defect)
            & jnp.isfinite(voltage_spread)
        )
        successful = jnp.all(local_accepted) & finite
        accepted = jax.tree.map(
            lambda proposed, previous: jnp.where(successful, proposed, previous),
            candidate,
            state,
        )
        evidence = SpatialBatteryCellEvidence(
            local_accepted,
            current_defect,
            voltage_spread,
            finite,
            successful,
        )
        return SpatialBatteryCellStepResult(
            candidate,
            accepted,
            voltage,
            jnp.sum(self.current_weights * voltage),
            residual,
            evidence,
        )


__all__ = [
    "SpatialBatteryCellEvidence",
    "SpatialBatteryCellPlan",
    "SpatialBatteryCellState",
    "SpatialBatteryCellStepResult",
]
