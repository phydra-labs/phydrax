#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleNeighborhoodState
from ._dynamics import PreparedAtomisticDynamics
from ._hydrodynamic_mobility import (
    AbstractHydrodynamicMobilityPlan,
    AbstractPreparedHydrodynamicMobility,
)


HydrodynamicDifferentiationPolicy: TypeAlias = Literal["pathwise", "weak"]


class _HydrodynamicMobilityProvider(StrictModule, NonTrainableState):
    mobility: AbstractPreparedHydrodynamicMobility

    def __call__(self, positions: ArrayLike, /):
        return self.mobility.operator(positions)


class HydrodynamicBrownianPlan(StrictModule, NonTrainableState):
    step_size: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    drift_epsilon: float = eqx.field(static=True)
    differentiation: HydrodynamicDifferentiationPolicy = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    maximum_displacement: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        step_size: float,
        temperature: float,
        /,
        *,
        drift_epsilon: float = 1.0e-6,
        differentiation: HydrodynamicDifferentiationPolicy = "pathwise",
        realization_id: int = 0,
        maximum_displacement: float | None = None,
    ):
        step = float(step_size)
        thermal = float(temperature)
        epsilon = float(drift_epsilon)
        realization = int(realization_id)
        maximum = None if maximum_displacement is None else float(maximum_displacement)
        if (
            not math.isfinite(step)
            or step <= 0.0
            or not math.isfinite(thermal)
            or thermal < 0.0
            or not math.isfinite(epsilon)
            or epsilon <= 0.0
            or differentiation not in ("pathwise", "weak")
            or realization < 0
            or realization > np.iinfo(np.uint32).max
            or (maximum is not None and (not math.isfinite(maximum) or maximum <= 0.0))
        ):
            raise ValueError("Hydrodynamic Brownian controls are invalid.")
        self.step_size = step
        self.temperature = thermal
        self.drift_epsilon = epsilon
        self.differentiation = differentiation
        self.realization_id = realization
        self.maximum_displacement = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hydrodynamic-brownian-plan",
                "step_size": step,
                "temperature": thermal,
                "drift_epsilon": epsilon,
                "differentiation": differentiation,
                "realization_id": realization,
                "maximum_displacement": maximum,
                "stochastic_convention": "ito",
            }
        )

    def prepare(
        self,
        dynamics: PreparedAtomisticDynamics,
        mobility: AbstractHydrodynamicMobilityPlan,
        /,
    ) -> PreparedHydrodynamicBrownian:
        return PreparedHydrodynamicBrownian(self, dynamics, mobility)


class HydrodynamicBrownianState(StrictModule):
    positions: Array
    image_counts: Array
    cell_vectors: Array
    neighborhood: ParticleNeighborhoodState
    forces: Array
    potential_energy: Array
    step_index: Array
    key_data: Array
    successful: Array
    mobility_route_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class HydrodynamicBrownianStepResult(StrictModule):
    candidate_state: HydrodynamicBrownianState
    accepted_state: HydrodynamicBrownianState
    fib: Any
    background_increment: Array
    displacement_norm: Array
    mobility_configuration_valid: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PreparedHydrodynamicBrownian(StrictModule, NonTrainableState):
    plan: HydrodynamicBrownianPlan
    dynamics: PreparedAtomisticDynamics
    mobility: AbstractPreparedHydrodynamicMobility
    fib: Any
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: HydrodynamicBrownianPlan,
        dynamics: PreparedAtomisticDynamics,
        mobility: AbstractHydrodynamicMobilityPlan,
        /,
    ):
        if not isinstance(plan, HydrodynamicBrownianPlan):
            raise TypeError("plan must be HydrodynamicBrownianPlan.")
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        if not isinstance(mobility, AbstractHydrodynamicMobilityPlan):
            raise TypeError("mobility must be AbstractHydrodynamicMobilityPlan.")
        if dynamics.constraints is not None:
            raise ValueError("Hydrodynamic Brownian dynamics does not admit constraints.")
        active = np.asarray(dynamics.system.active_mask, dtype=np.bool_)
        active_slots = np.flatnonzero(active)
        prepared_mobility = mobility.prepare(dynamics.system, active_slots)
        from ..solver._mac_stochastic_immersed import FIBOverdampedPlan

        fib = FIBOverdampedPlan(
            prepared_mobility.coordinate_space,
            _HydrodynamicMobilityProvider(prepared_mobility),
            temperature=plan.temperature,
            boltzmann_constant=dynamics.system.plan.units.boltzmann_constant,
            drift_epsilon=plan.drift_epsilon,
            differentiation=plan.differentiation,
        )
        self.plan = plan
        self.dynamics = dynamics
        self.mobility = prepared_mobility
        self.fib = fib
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-hydrodynamic-brownian",
                "plan": plan.plan_id,
                "dynamics": dynamics.prepared_id,
                "mobility": prepared_mobility.prepared_id,
                "fib": fib.plan_id,
            }
        )

    def _initial_cell_vectors(self, dtype) -> Array:
        cell = self.dynamics.system.cell
        return (
            jnp.zeros((0, 3), dtype=dtype) if cell is None else cell.vectors.astype(dtype)
        )

    def _wrap(self, unwrapped: Array, cell_vectors: Array, /) -> tuple[Array, Array]:
        cell = self.dynamics.system.cell
        if cell is None:
            return unwrapped, jnp.zeros((unwrapped.shape[0], 0), dtype=jnp.int32)
        return cell.wrap_with_vectors(unwrapped, cell_vectors)

    def _unwrapped(
        self, positions: Array, image_counts: Array, cell_vectors: Array, /
    ) -> Array:
        if self.dynamics.system.cell is None:
            return positions
        return positions + contract(
            "ni,ij->nj", image_counts.astype(positions.dtype), cell_vectors
        )

    def _evaluate(
        self,
        positions: Array,
        image_counts: Array,
        cell_vectors: Array,
        /,
    ):
        neighborhood = self.dynamics.neighborhood.build(positions)
        unwrapped = self._unwrapped(positions, image_counts, cell_vectors)
        kwargs = {
            "unwrapped_positions": unwrapped,
            "species": self.dynamics.system.plan.atom_type_ids,
            "cell": self.dynamics.system.cell,
        }
        if self.dynamics.system.cell is not None:
            cell = self.dynamics.system.cell
            kwargs["fractional_positions"] = cell.fractional_with_vectors(
                positions, cell_vectors
            )
            kwargs["cell_vectors"] = cell_vectors
        evaluation = self.dynamics.potential.evaluate(positions, neighborhood, **kwargs)
        return neighborhood, evaluation

    def initialize(
        self,
        positions: ArrayLike,
        /,
        *,
        key: Key[Array, ""],
    ) -> HydrodynamicBrownianState:
        value = jnp.asarray(positions, dtype=self.dynamics.system.plan.coordinate_dtype)
        expected = (self.dynamics.system.capacity, 3)
        if value.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        key_data = jr.key_data(key)
        if key_data.shape != (2,):
            raise ValueError("key must contain two uint32 words.")
        vectors = self._initial_cell_vectors(value.dtype)
        wrapped, images = self._wrap(value, vectors)
        neighborhood, evaluation = self._evaluate(wrapped, images, vectors)
        unwrapped = self._unwrapped(wrapped, images, vectors)
        mobility_valid = self.mobility.configuration_valid(
            unwrapped[self.mobility.active_slots]
        )
        successful = evaluation.successful & mobility_valid & jnp.all(jnp.isfinite(value))
        return HydrodynamicBrownianState(
            wrapped,
            images,
            vectors,
            neighborhood,
            evaluation.forces,
            evaluation.energy,
            jnp.zeros((), dtype=jnp.int32),
            key_data,
            successful,
            self.mobility.route_id,
            self.prepared_id,
        )

    def step(
        self,
        state: HydrodynamicBrownianState,
        /,
        *,
        background_velocity: ArrayLike | None = None,
    ) -> HydrodynamicBrownianStepResult:
        if not isinstance(state, HydrodynamicBrownianState):
            raise TypeError("state must be HydrodynamicBrownianState.")
        if (
            state.prepared_id != self.prepared_id
            or state.mobility_route_id != self.mobility.route_id
        ):
            raise ValueError("Hydrodynamic Brownian state belongs to another route.")
        unwrapped = self._unwrapped(
            state.positions, state.image_counts, state.cell_vectors
        )
        active_positions = unwrapped[self.mobility.active_slots]
        active_forces = state.forces[self.mobility.active_slots]
        mobility_configuration_valid = self.mobility.configuration_valid(active_positions)
        from ..solver._mac_stochastic_immersed import StochasticReplayKey

        replay_key = StochasticReplayKey(
            state.key_data[0],
            state.step_index,
            jnp.asarray(self.plan.realization_id, dtype=jnp.uint32),
            state.key_data[1],
        )
        fib = self.fib.step(
            active_positions,
            active_forces,
            self.plan.step_size,
            replay_key,
        )
        background = (
            jnp.zeros_like(active_positions)
            if background_velocity is None
            else jnp.asarray(background_velocity, dtype=active_positions.dtype)
        )
        if background.shape != active_positions.shape:
            raise ValueError(
                "background_velocity must match active particle coordinates."
            )
        background_increment = self.plan.step_size * background
        proposed_active = fib.position + background_increment
        proposed_unwrapped = unwrapped.at[self.mobility.active_slots].set(proposed_active)
        positions, images = self._wrap(proposed_unwrapped, state.cell_vectors)
        neighborhood, evaluation = self._evaluate(positions, images, state.cell_vectors)
        displacement = proposed_active - active_positions
        displacement_norm = jnp.sqrt(jnp.sum(displacement * displacement))
        displacement_valid = (
            jnp.asarray(True)
            if self.plan.maximum_displacement is None
            else displacement_norm <= self.plan.maximum_displacement
        )
        successful = (
            state.successful
            & mobility_configuration_valid
            & fib.accepted
            & evaluation.successful
            & displacement_valid
            & jnp.all(jnp.isfinite(background))
            & jnp.all(jnp.isfinite(positions))
        )
        candidate = HydrodynamicBrownianState(
            positions,
            images,
            state.cell_vectors,
            neighborhood,
            evaluation.forces,
            evaluation.energy,
            state.step_index + 1,
            state.key_data,
            successful,
            self.mobility.route_id,
            self.prepared_id,
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        return HydrodynamicBrownianStepResult(
            candidate,
            accepted,
            fib,
            background_increment,
            displacement_norm,
            mobility_configuration_valid,
            successful,
            self.prepared_id,
        )


__all__ = [
    "HydrodynamicBrownianPlan",
    "HydrodynamicBrownianState",
    "HydrodynamicBrownianStepResult",
    "HydrodynamicDifferentiationPolicy",
    "PreparedHydrodynamicBrownian",
]
