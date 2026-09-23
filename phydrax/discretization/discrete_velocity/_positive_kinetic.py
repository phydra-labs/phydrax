#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._exponential_family import (
    FiniteSupportExponentialFamily,
    FiniteSupportNaturalSolvePlan,
    solve_finite_support_mean,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._kinetic_entropy import KineticEntropyRootPlan, solve_kinetic_entropy_root
from ._compressible_contracts import (
    CompressibleKineticConservationEvidence,
    CompressibleKineticMacroscopicState,
    CompressibleKineticPopulationState,
    CompressibleKineticStepResult,
    KineticPopulationFieldSpec,
    KineticPopulationLayout,
)
from ._compressible_rules import (
    CompressibleVelocityRule,
    d3q39_guided_rule,
    d3q343_entropic_rule,
)


PositiveKineticCollisionKind = Literal["bgk", "entropic"]


class PositiveCompressibleKineticPlan(StrictModule, NonTrainableState):
    """Positive compressible kinetic model on one prepared velocity rule."""

    rule: CompressibleVelocityRule
    family: FiniteSupportExponentialFamily
    layout: KineticPopulationLayout
    entropy_root: KineticEntropyRootPlan
    gamma: float = eqx.field(static=True)
    gas_constant: float = eqx.field(static=True)
    heat_capacity_cv: float = eqx.field(static=True)
    internal_heat_capacity: float = eqx.field(static=True)
    collision_kind: PositiveKineticCollisionKind = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        rule: CompressibleVelocityRule,
        /,
        *,
        gamma: float = 1.4,
        gas_constant: float = 1.0,
        collision_kind: PositiveKineticCollisionKind | None = None,
        equilibrium_solve: FiniteSupportNaturalSolvePlan | None = None,
        entropy_root: KineticEntropyRootPlan | None = None,
    ):
        if not isinstance(rule, CompressibleVelocityRule):
            raise TypeError("rule must be a CompressibleVelocityRule.")
        gamma_value = float(gamma)
        gas_value = float(gas_constant)
        if not np.isfinite(gamma_value) or gamma_value <= 1.0 or gamma_value > 5.0 / 3.0:
            raise ValueError("gamma must lie in (1, 5/3].")
        if not np.isfinite(gas_value) or gas_value <= 0.0:
            raise ValueError("gas_constant must be finite and positive.")
        selected_collision = (
            ("entropic" if rule.model_kind == "entropic-d3q343" else "bgk")
            if collision_kind is None
            else collision_kind
        )
        if selected_collision not in ("bgk", "entropic"):
            raise ValueError(
                f"Unknown positive kinetic collision {selected_collision!r}."
            )
        solve = (
            FiniteSupportNaturalSolvePlan()
            if equilibrium_solve is None
            else equilibrium_solve
        )
        root = KineticEntropyRootPlan() if entropy_root is None else entropy_root
        if not isinstance(solve, FiniteSupportNaturalSolvePlan):
            raise TypeError("equilibrium_solve must be FiniteSupportNaturalSolvePlan.")
        if not isinstance(root, KineticEntropyRootPlan):
            raise TypeError("entropy_root must be KineticEntropyRootPlan.")
        heat_capacity = gas_value / (gamma_value - 1.0)
        internal = heat_capacity - 1.5 * gas_value
        fields = [
            KineticPopulationFieldSpec("particle", "particle", rule.population_count)
        ]
        if internal > 64.0 * np.finfo(np.float64).eps * heat_capacity:
            fields.append(
                KineticPopulationFieldSpec(
                    "internal_energy",
                    "internal-energy",
                    rule.population_count,
                )
            )
        layout = KineticPopulationLayout(tuple(fields))
        family = FiniteSupportExponentialFamily(
            rule.guided_features,
            rule.base_probabilities,
            family_id=f"kinetic-equilibrium:{rule.name}",
            support_id=rule.rule_id,
            solve_plan=solve,
        )
        self.rule = rule
        self.family = family
        self.layout = layout
        self.entropy_root = root
        self.gamma = gamma_value
        self.gas_constant = gas_value
        self.heat_capacity_cv = heat_capacity
        self.internal_heat_capacity = max(internal, 0.0)
        self.collision_kind = selected_collision
        self.model_id = canonical_fingerprint(
            {
                "kind": "positive-compressible-kinetic-plan",
                "rule": rule.rule_id,
                "layout": layout.layout_id,
                "gamma": gamma_value,
                "gas_constant": gas_value,
                "collision": selected_collision,
                "equilibrium_solve": solve.plan_id,
                "entropy_root": root.plan_id,
            }
        )

    def _target_features(
        self,
        velocity: Array,
        temperature: Array,
        /,
    ) -> Array:
        thermal = self.gas_constant * temperature
        speed_squared = jnp.sum(velocity * velocity, axis=-1)
        if self.rule.model_kind == "guided-d3q39":
            pressure = (
                thermal[..., None, None]
                * jnp.eye(self.rule.dimension, dtype=velocity.dtype)
                + velocity[..., :, None] * velocity[..., None, :]
            )
            heat = velocity * (speed_squared + 5.0 * thermal)[..., None]
            return jnp.concatenate(
                (
                    velocity,
                    pressure[..., (0, 1, 2), (0, 1, 2)],
                    jnp.stack(
                        (pressure[..., 0, 1], pressure[..., 0, 2], pressure[..., 1, 2]),
                        axis=-1,
                    ),
                    heat,
                ),
                axis=-1,
            )
        return jnp.concatenate(
            (velocity, (speed_squared + 3.0 * thermal)[..., None]), axis=-1
        )

    def equilibrium(
        self,
        density: ArrayLike,
        velocity: ArrayLike,
        temperature: ArrayLike,
        /,
        *,
        initial_dual: ArrayLike | None = None,
    ) -> tuple[tuple[Array, ...], Array, Array]:
        rho = jnp.asarray(density)
        flow_velocity = jnp.asarray(velocity, dtype=rho.dtype)
        thermal = jnp.asarray(temperature, dtype=rho.dtype)
        if flow_velocity.shape != rho.shape + (self.rule.dimension,):
            raise ValueError("velocity must have shape density.shape + (dimension,).")
        if thermal.shape != rho.shape:
            raise ValueError("temperature must have the density shape.")
        target = self._target_features(flow_velocity, thermal)
        result = solve_finite_support_mean(
            self.family,
            target,
            initial=initial_dual,
        )
        valid = (
            result.evidence.successful
            & jnp.isfinite(rho)
            & (rho > 0.0)
            & jnp.isfinite(thermal)
            & (thermal > 0.0)
        )
        checked = eqx.error_if(
            result.probabilities,
            jnp.any(~valid),
            "Compressible kinetic equilibrium is outside the positive support.",
        )
        particle = rho[..., None] * checked
        populations: list[Array] = [particle]
        if len(self.layout.fields) == 2:
            internal = self.internal_heat_capacity * thermal[..., None] * particle
            populations.append(internal)
        return (
            tuple(populations),
            result.conversion.natural.values,
            result.evidence.residual,
        )

    def initialize(
        self,
        density: ArrayLike,
        velocity: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> CompressibleKineticPopulationState:
        populations, dual, _ = self.equilibrium(density, velocity, temperature)
        rho = jnp.asarray(density)
        frame = jnp.broadcast_to(
            jnp.asarray(self.rule.frame_shift, dtype=rho.dtype),
            rho.shape + (self.rule.dimension,),
        )
        return CompressibleKineticPopulationState(
            populations,
            dual,
            jnp.full(rho.shape, 2.0, dtype=rho.dtype),
            frame,
            jnp.ones(rho.shape, dtype=rho.dtype),
            self.layout,
        )

    def moments(
        self,
        state: CompressibleKineticPopulationState,
        /,
    ) -> CompressibleKineticMacroscopicState:
        if state.layout.layout_id != self.layout.layout_id:
            raise ValueError("State population layout does not match this kinetic plan.")
        particle = state.population("particle")
        velocities = self.rule.velocities.astype(particle.dtype)
        density = jnp.sum(particle, axis=-1)
        momentum = particle @ velocities
        safe_density = jnp.where(density > 0.0, density, 1.0)
        velocity = momentum / safe_density[..., None]
        speed_squared = jnp.sum(velocities * velocities, axis=-1)
        translational = 0.5 * jnp.sum(particle * speed_squared, axis=-1)
        internal = (
            jnp.sum(state.population("internal-energy"), axis=-1)
            if len(self.layout.fields) == 2
            else jnp.zeros_like(density)
        )
        total = translational + internal
        bulk_kinetic = 0.5 * jnp.sum(momentum * velocity, axis=-1)
        temperature = (total - bulk_kinetic) / (safe_density * self.heat_capacity_cv)
        pressure = density * self.gas_constant * temperature
        peculiar = velocities - velocity[..., None, :]
        stress = (
            jnp.swapaxes(
                particle[..., :, None] * peculiar,
                -1,
                -2,
            )
            @ peculiar
        )
        heat_flux = 0.5 * jnp.sum(
            particle[..., :, None]
            * peculiar
            * jnp.sum(peculiar * peculiar, axis=-1)[..., :, None],
            axis=-2,
        )
        finite = (
            jnp.isfinite(density)
            & jnp.all(jnp.isfinite(momentum), axis=-1)
            & jnp.isfinite(total)
            & jnp.isfinite(temperature)
        )
        admissible = finite & (density > 0.0) & (temperature > 0.0)
        return CompressibleKineticMacroscopicState(
            density=density,
            momentum=momentum,
            velocity=velocity,
            translational_energy=translational,
            internal_energy=internal,
            total_energy=total,
            temperature=temperature,
            pressure=pressure,
            stress=stress,
            heat_flux=heat_flux,
            finite=finite,
            admissible=admissible,
        )

    def collide(
        self,
        state: CompressibleKineticPopulationState,
        relaxation_rate: ArrayLike,
        /,
    ) -> CompressibleKineticStepResult:
        old = self.moments(state)
        rate = jnp.broadcast_to(
            jnp.asarray(relaxation_rate, dtype=old.density.dtype), old.density.shape
        )
        rate = eqx.error_if(
            rate,
            jnp.any(~jnp.isfinite(rate) | (rate <= 0.0) | (rate >= 2.0)),
            "Kinetic relaxation_rate must lie in (0, 2).",
        )
        equilibrium, dual, _ = self.equilibrium(
            old.density,
            old.velocity,
            old.temperature,
            initial_dual=state.equilibrium_dual,
        )
        old_particle = state.population("particle")
        if self.collision_kind == "entropic":
            root = solve_kinetic_entropy_root(
                self.entropy_root,
                old_particle,
                equilibrium[0] - old_particle,
                base_measure=self.rule.base_probabilities,
                initial=state.stabilizer,
            )
            factor = 0.5 * rate * root.evidence.alpha
            root_success = root.evidence.successful
            stabilizer = root.evidence.alpha
        else:
            factor = rate
            root_success = jnp.ones(old.density.shape, dtype=jnp.bool_)
            stabilizer = jnp.ones(old.density.shape, dtype=old.density.dtype)
        candidates = tuple(
            value + factor[..., None] * (target - value)
            for value, target in zip(state.populations, equilibrium, strict=True)
        )
        candidate = CompressibleKineticPopulationState(
            candidates,
            dual,
            stabilizer,
            state.frame_velocity,
            state.frame_temperature_scale,
            self.layout,
        )
        new = self.moments(candidate)
        mass_defect = new.density - old.density
        momentum_defect = jnp.max(jnp.abs(new.momentum - old.momentum), axis=-1)
        energy_defect = new.total_energy - old.total_energy
        entropy_before = jnp.sum(
            old_particle * jnp.log(old_particle / self.rule.base_probabilities),
            axis=-1,
        )
        particle_candidate = candidates[0]
        entropy_after = jnp.sum(
            particle_candidate
            * jnp.log(particle_candidate / self.rule.base_probabilities),
            axis=-1,
        )
        minimum_population = jnp.min(
            jnp.stack(tuple(jnp.min(value, axis=-1) for value in candidates), axis=0),
            axis=0,
        )
        finite = new.finite & jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value), axis=-1) for value in candidates),
                axis=0,
            ),
            axis=0,
        )
        numerical_tolerance = jnp.maximum(
            512.0 * jnp.finfo(old.density.dtype).eps,
            8.0 * self.family.solve_plan.residual_tolerance,
        )
        density_scale = jnp.maximum(old.density, 1.0)
        energy_scale = jnp.maximum(jnp.abs(old.total_energy), 1.0)
        successful = (
            old.admissible
            & new.admissible
            & finite
            & root_success
            & (minimum_population > 0.0)
            & (jnp.abs(mass_defect) <= numerical_tolerance * density_scale)
            & (jnp.abs(energy_defect) <= numerical_tolerance * energy_scale)
        )
        accepted_populations = tuple(
            jnp.where(successful[..., None], new_value, old_value)
            for new_value, old_value in zip(candidates, state.populations, strict=True)
        )
        accepted = CompressibleKineticPopulationState(
            accepted_populations,
            jnp.where(successful[..., None], dual, state.equilibrium_dual),
            jnp.where(successful, stabilizer, state.stabilizer),
            state.frame_velocity,
            state.frame_temperature_scale,
            self.layout,
        )
        conservation = CompressibleKineticConservationEvidence(
            mass_defect=mass_defect,
            momentum_defect=momentum_defect,
            energy_defect=energy_defect,
            entropy_change=entropy_after - entropy_before,
            minimum_population=minimum_population,
            finite=finite,
            successful=successful,
        )
        return CompressibleKineticStepResult(
            candidate=candidate,
            accepted=accepted,
            macroscopic=new,
            conservation=conservation,
            status=jnp.where(successful, 0, 1).astype(jnp.int32),
            successful=successful,
            model_id=self.model_id,
        )


def guided_d3q39_plan(
    *,
    gamma: float = 1.4,
    gas_constant: float = 1.0,
    collision_kind: PositiveKineticCollisionKind = "bgk",
    dtype: np.dtype | str = np.float64,
) -> PositiveCompressibleKineticPlan:
    return PositiveCompressibleKineticPlan(
        d3q39_guided_rule(dtype=dtype),
        gamma=gamma,
        gas_constant=gas_constant,
        collision_kind=collision_kind,
    )


def entropic_d3q343_plan(
    *,
    gamma: float = 1.4,
    gas_constant: float = 1.0,
    dtype: np.dtype | str = np.float64,
) -> PositiveCompressibleKineticPlan:
    return PositiveCompressibleKineticPlan(
        d3q343_entropic_rule(dtype=dtype),
        gamma=gamma,
        gas_constant=gas_constant,
        collision_kind="entropic",
    )


__all__ = [
    "PositiveCompressibleKineticPlan",
    "PositiveKineticCollisionKind",
    "entropic_d3q343_plan",
    "guided_d3q39_plan",
]
