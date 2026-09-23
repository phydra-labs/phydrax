#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._compressible_contracts import (
    CompressibleKineticConservationEvidence,
    CompressibleKineticPopulationState,
    CompressibleKineticStepResult,
)
from ._compressible_rules import CompressibleVelocityRule, d3q33_filtered_rule
from ._positive_kinetic import PositiveCompressibleKineticPlan


class FilteredD3Q33Evidence(StrictModule):
    filtered_moment_norm: Array
    conserved_moment_defect: Array
    minimum_population: Array
    filter_strength: float = eqx.field(static=True)
    successful: Array


class FilteredD3Q33Plan(StrictModule, NonTrainableState):
    """Monatomic thermal D3Q33 MRT collision with bounded spatial filtering."""

    model: PositiveCompressibleKineticPlan
    transform: Array
    inverse_transform: Array
    filter_indices: Array
    filter_strength: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rule: CompressibleVelocityRule | None = None,
        /,
        *,
        gas_constant: float = 1.0,
        filter_strength: float = 1.0 / 16.0,
    ):
        selected_rule = d3q33_filtered_rule() if rule is None else rule
        if not isinstance(selected_rule, CompressibleVelocityRule):
            raise TypeError("rule must be a CompressibleVelocityRule.")
        if selected_rule.model_kind != "filtered-d3q33":
            raise ValueError("FilteredD3Q33Plan requires the D3Q33 rule.")
        strength = float(filter_strength)
        if not np.isfinite(strength) or strength < 0.0 or strength > 1.0 / 6.0:
            raise ValueError("filter_strength must lie in [0, 1/6].")
        model = PositiveCompressibleKineticPlan(
            selected_rule,
            gamma=5.0 / 3.0,
            gas_constant=gas_constant,
            collision_kind="bgk",
        )
        velocities = np.asarray(selected_rule.velocities, dtype=np.float64)
        q = selected_rule.population_count
        rows: list[np.ndarray] = [
            np.ones(q),
            velocities[:, 0],
            velocities[:, 1],
            velocities[:, 2],
            np.sum(velocities * velocities, axis=1),
        ]
        degrees = [0, 1, 1, 1, 2]
        for degree in range(2, 9):
            for exponent in itertools.product(range(degree + 1), repeat=3):
                if sum(exponent) != degree:
                    continue
                candidate = np.prod(velocities ** np.asarray(exponent)[None, :], axis=1)
                trial = np.vstack((*rows, candidate))
                if np.linalg.matrix_rank(trial) > len(rows):
                    rows.append(candidate)
                    degrees.append(degree)
                if len(rows) == q:
                    break
            if len(rows) == q:
                break
        if len(rows) != q:
            raise ValueError("D3Q33 moment basis construction was rank deficient.")
        transform = np.vstack(rows)
        inverse_result = solve(
            LinearSystem(DenseLinearOperator(transform)),
            np.eye(q),
            policy=LinearSolvePolicy(DenseLU()),
        )
        if not bool(jnp.all(inverse_result.successful)):
            raise ValueError("D3Q33 moment transform factorization failed.")
        inverse = np.asarray(inverse_result.value)
        filter_indices = np.asarray(
            [
                index
                for index, degree in enumerate(degrees)
                if index >= 5 and degree in (2, 3)
            ],
            dtype=np.int32,
        )
        if filter_indices.size != 15:
            raise ValueError("D3Q33 filter must own exactly 15 second/third-order modes.")
        self.model = model
        self.transform = jnp.asarray(transform)
        self.inverse_transform = jnp.asarray(inverse)
        self.filter_indices = jnp.asarray(filter_indices)
        self.filter_strength = strength
        self.plan_id = canonical_fingerprint(
            {
                "kind": "filtered-d3q33",
                "model": model.model_id,
                "transform": transform.tolist(),
                "filter_indices": filter_indices.tolist(),
                "filter_strength": strength,
            }
        )

    def initialize(
        self,
        density: ArrayLike,
        velocity: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> CompressibleKineticPopulationState:
        return self.model.initialize(density, velocity, temperature)

    def collide(
        self,
        state: CompressibleKineticPopulationState,
        relaxation_rate: ArrayLike,
        /,
        *,
        apply_filter: bool = True,
    ) -> tuple[CompressibleKineticStepResult, FilteredD3Q33Evidence]:
        old = self.model.moments(state)
        populations = state.population("particle")
        rate = jnp.broadcast_to(
            jnp.asarray(relaxation_rate, dtype=populations.dtype), old.density.shape
        )
        rate = eqx.error_if(
            rate,
            jnp.any(~jnp.isfinite(rate) | (rate <= 0.0) | (rate >= 2.0)),
            "D3Q33 relaxation_rate must lie in (0, 2).",
        )
        equilibrium, dual, _ = self.model.equilibrium(
            old.density,
            old.velocity,
            old.temperature,
            initial_dual=state.equilibrium_dual,
        )
        transform = self.transform.astype(populations.dtype)
        inverse = self.inverse_transform.astype(populations.dtype)
        moments = populations @ jnp.swapaxes(transform, -1, -2)
        equilibrium_moments = equilibrium[0] @ jnp.swapaxes(transform, -1, -2)
        collision_increment = -rate[..., None] * (moments - equilibrium_moments)
        collision_increment = collision_increment.at[..., :5].set(0.0)
        filtered_norm = jnp.zeros(old.density.shape, dtype=populations.dtype)
        if apply_filter and populations.ndim == 4 and self.filter_strength > 0.0:
            selected = collision_increment[..., self.filter_indices]
            neighbor_sum = sum(
                jnp.roll(selected, shift=offset, axis=axis)
                for axis in range(3)
                for offset in (-1, 1)
            )
            filtered = selected + self.filter_strength * (neighbor_sum - 6.0 * selected)
            collision_increment = collision_increment.at[..., self.filter_indices].set(
                filtered
            )
            filtered_norm = jnp.linalg.norm(filtered - selected, axis=-1)
        candidate_populations = populations + collision_increment @ jnp.swapaxes(
            inverse, -1, -2
        )
        candidate = CompressibleKineticPopulationState(
            (candidate_populations,),
            dual,
            jnp.ones(old.density.shape, dtype=populations.dtype),
            state.frame_velocity,
            state.frame_temperature_scale,
            self.model.layout,
            self.model.model_id,
            self.model.rule.rule_id,
        )
        new = self.model.moments(candidate)
        mass_defect = new.density - old.density
        momentum_defect = jnp.max(jnp.abs(new.momentum - old.momentum), axis=-1)
        energy_defect = new.total_energy - old.total_energy
        minimum = jnp.min(candidate_populations, axis=-1)
        conserved_defect = jnp.maximum(
            jnp.abs(mass_defect), jnp.maximum(momentum_defect, jnp.abs(energy_defect))
        )
        tolerance = (
            512.0
            * jnp.finfo(populations.dtype).eps
            * jnp.maximum(jnp.abs(old.total_energy), 1.0)
        )
        successful = (
            old.admissible
            & new.admissible
            & new.finite
            & (minimum > 0.0)
            & (conserved_defect <= tolerance)
        )
        accepted = CompressibleKineticPopulationState(
            (jnp.where(successful[..., None], candidate_populations, populations),),
            jnp.where(successful[..., None], dual, state.equilibrium_dual),
            state.stabilizer,
            state.frame_velocity,
            state.frame_temperature_scale,
            self.model.layout,
            self.model.model_id,
            self.model.rule.rule_id,
        )
        conservation = CompressibleKineticConservationEvidence(
            mass_defect=mass_defect,
            momentum_defect=momentum_defect,
            energy_defect=energy_defect,
            entropy_change=jnp.zeros_like(mass_defect),
            minimum_population=minimum,
            finite=new.finite,
            successful=successful,
        )
        result = CompressibleKineticStepResult(
            candidate=candidate,
            accepted=accepted,
            macroscopic=new,
            conservation=conservation,
            status=jnp.where(successful, 0, 1).astype(jnp.int32),
            successful=successful,
            model_id=self.plan_id,
        )
        evidence = FilteredD3Q33Evidence(
            filtered_moment_norm=filtered_norm,
            conserved_moment_defect=conserved_defect,
            minimum_population=minimum,
            filter_strength=self.filter_strength,
            successful=successful,
        )
        return result, evidence


__all__ = ["FilteredD3Q33Evidence", "FilteredD3Q33Plan"]
