#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PreparedFiniteVolumeDynamics
from ...solver import ParticleMeshGravityPlan
from ._background import FLRWBackground
from ._particles import CosmologicalKDKPlan, CosmologicalParticleState


class ComovingEulerState(StrictModule):
    cell_average: Array
    scale_factor: Array


class ComovingEulerDiagnostics(StrictModule):
    stable_step: Array
    density_positive: Array
    pressure_positive: Array
    finite: Array
    successful: Array


class ComovingEulerPlan(StrictModule, NonTrainableState):
    """Fixed-substep ideal Euler transport in scale factor with expansion/gravity."""

    dynamics: PreparedFiniteVolumeDynamics
    adiabatic_index: float = eqx.field(static=True)
    expansion_dimension: int = eqx.field(static=True)
    cfl: float = eqx.field(static=True)
    substeps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedFiniteVolumeDynamics,
        /,
        *,
        adiabatic_index: float = 5.0 / 3.0,
        expansion_dimension: int = 3,
        cfl: float = 0.3,
        substeps: int = 4,
    ):
        if not isinstance(dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError("dynamics must be PreparedFiniteVolumeDynamics.")
        gamma = float(adiabatic_index)
        expansion_dimension_ = int(expansion_dimension)
        cfl_ = float(cfl)
        substeps_ = int(substeps)
        if (
            not np.isfinite(gamma)
            or gamma <= 1.0
            or expansion_dimension_ not in (1, 2, 3)
            or not np.isfinite(cfl_)
            or cfl_ <= 0.0
            or substeps_ <= 0
            or dynamics.system.dimension + 2 != dynamics.discretization.component_count
        ):
            raise ValueError("Comoving Euler configuration is invalid.")
        self.dynamics = dynamics
        self.adiabatic_index = gamma
        self.expansion_dimension = expansion_dimension_
        self.cfl = cfl_
        self.substeps = substeps_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "comoving-euler-scale-factor-transport",
                "dynamics": dynamics.dynamics_id,
                "adiabatic_index": gamma,
                "expansion_dimension": expansion_dimension_,
                "cfl": cfl_,
                "substeps": substeps_,
            }
        )

    @property
    def dimension(self) -> int:
        return self.dynamics.system.dimension

    def initialize(
        self, cell_average: ArrayLike, scale_factor: ArrayLike, /
    ) -> ComovingEulerState:
        values = jnp.asarray(cell_average)
        expected = self.dynamics.discretization.cell_shape + (self.dimension + 2,)
        if values.shape != expected:
            raise ValueError(f"Comoving Euler state must have shape {expected}.")
        scale = jnp.asarray(scale_factor, dtype=values.dtype)
        if scale.shape != ():
            raise ValueError("Comoving Euler scale factor must be scalar.")
        density = values[..., 0]
        momentum = values[..., 1 : 1 + self.dimension]
        energy = values[..., -1]
        kinetic = jnp.sum(momentum**2, axis=-1) / (2.0 * density)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values))
            | ~jnp.isfinite(scale)
            | (scale <= 0.0)
            | jnp.any(density <= 0.0)
            | jnp.any(energy - kinetic <= 0.0),
            "Comoving Euler initial state is inadmissible.",
        )
        return ComovingEulerState(values, scale)

    def _rate(
        self,
        background: FLRWBackground,
        scale_factor: Array,
        cell_average: Array,
        gravity: Array,
        args: Any,
        /,
    ) -> Array:
        transport, _ = self.dynamics.residual_with_diagnostics(
            scale_factor, cell_average, args
        )
        hubble = background.hubble(scale_factor)
        rate = transport / (scale_factor**2 * hubble)
        density = cell_average[..., 0]
        momentum = cell_average[..., 1 : 1 + self.dimension]
        energy = cell_average[..., -1]
        kinetic = jnp.sum(momentum**2, axis=-1) / (2.0 * density)
        pressure = (self.adiabatic_index - 1.0) * (energy - kinetic)
        momentum_source = -momentum / scale_factor + density[..., None] * gravity / (
            scale_factor**3 * hubble
        )
        energy_source = -(
            2.0 * kinetic + self.expansion_dimension * pressure
        ) / scale_factor + contract("...d,...d->...", momentum, gravity) / (
            scale_factor**3 * hubble
        )
        rate = rate.at[..., 1 : 1 + self.dimension].add(momentum_source)
        return rate.at[..., -1].add(energy_source)

    def advance(
        self,
        background: FLRWBackground,
        state: ComovingEulerState,
        end_scale_factor: ArrayLike,
        gravity_start: ArrayLike,
        gravity_end: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[ComovingEulerState, ComovingEulerDiagnostics]:
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        if not isinstance(state, ComovingEulerState):
            raise TypeError("state must be ComovingEulerState.")
        end = jnp.asarray(end_scale_factor, dtype=state.scale_factor.dtype)
        if end.shape != ():
            raise ValueError("Comoving Euler end scale factor must be scalar.")
        end = background.require_flat(end)
        gravity_0 = jnp.asarray(gravity_start, dtype=state.cell_average.dtype)
        gravity_1 = jnp.asarray(gravity_end, dtype=state.cell_average.dtype)
        expected_gravity = state.cell_average.shape[:-1] + (self.dimension,)
        if gravity_0.shape != expected_gravity or gravity_1.shape != expected_gravity:
            raise ValueError(
                "Comoving Euler gravity must match grid cells and dimension."
            )
        interval = end - state.scale_factor
        interval = eqx.error_if(
            interval,
            ~jnp.isfinite(interval) | (interval <= 0.0),
            "Comoving Euler interval must be finite and positive.",
        )
        step = interval / self.substeps

        def substep(index, carry):
            values, successful, stable_min = carry
            fraction_0 = index / self.substeps
            fraction_mid = (index + 0.5) / self.substeps
            start = state.scale_factor + fraction_0 * interval
            midpoint = state.scale_factor + fraction_mid * interval
            gravity_a = gravity_0 + fraction_0 * (gravity_1 - gravity_0)
            gravity_mid = gravity_0 + fraction_mid * (gravity_1 - gravity_0)
            stable_physical = self.dynamics.stable_step(values, args, cfl=self.cfl)
            stable_scale = stable_physical * start**2 * background.hubble(start)
            first_rate = self._rate(background, start, values, gravity_a, args)
            midpoint_values = values + 0.5 * step * first_rate
            midpoint_rate = self._rate(
                background, midpoint, midpoint_values, gravity_mid, args
            )
            candidate = values + step * midpoint_rate
            density = candidate[..., 0]
            momentum = candidate[..., 1 : 1 + self.dimension]
            energy = candidate[..., -1]
            kinetic = jnp.sum(momentum**2, axis=-1) / (2.0 * density)
            admissible = (
                successful
                & (step <= stable_scale)
                & jnp.all(jnp.isfinite(candidate))
                & jnp.all(density > 0.0)
                & jnp.all(energy - kinetic > 0.0)
            )
            return (
                jnp.where(admissible, candidate, values),
                admissible,
                jnp.minimum(stable_min, stable_scale),
            )

        initial = (
            state.cell_average,
            jnp.asarray(True),
            jnp.asarray(jnp.inf, dtype=state.cell_average.dtype),
        )
        values, successful, stable_min = jax.lax.fori_loop(
            0, self.substeps, substep, initial
        )
        density = values[..., 0]
        momentum = values[..., 1 : 1 + self.dimension]
        energy = values[..., -1]
        kinetic = jnp.sum(momentum**2, axis=-1) / (2.0 * density)
        pressure_positive = jnp.all(energy - kinetic > 0.0)
        result = ComovingEulerState(
            jnp.where(successful, values, state.cell_average),
            jnp.where(successful, end, state.scale_factor),
        )
        diagnostics = ComovingEulerDiagnostics(
            stable_step=stable_min,
            density_positive=jnp.all(density > 0.0),
            pressure_positive=pressure_positive,
            finite=jnp.all(jnp.isfinite(values)),
            successful=successful,
        )
        return result, diagnostics


class SharedGasParticleGravityResult(StrictModule):
    potential: Array
    cell_acceleration: Array
    particle_acceleration: Array
    mass_balance_defect: Array
    net_force: Array
    successful: Array


class CosmologicalGasParticleState(StrictModule):
    gas: ComovingEulerState
    particles: CosmologicalParticleState


class CosmologicalGasParticleDiagnostics(StrictModule):
    accepted: Array
    gas_successful: Array
    gravity_successful: Array
    mass_balance_defect: Array
    net_force: Array
    accepted_steps: Array
    completed: Array
    first_failed_step: Array


class CosmologicalGasParticleResult(StrictModule):
    state: CosmologicalGasParticleState
    diagnostics: CosmologicalGasParticleDiagnostics
    successful: Array


class CosmologicalGasParticleGravityPlan(StrictModule, NonTrainableState):
    """Transactional adiabatic gas + collisionless-DM epoch with shared PM gravity."""

    gas: ComovingEulerPlan
    particles: CosmologicalKDKPlan
    gravity: ParticleMeshGravityPlan
    scale_factors: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gas: ComovingEulerPlan,
        particles: CosmologicalKDKPlan,
        gravity: ParticleMeshGravityPlan,
        scale_factors: ArrayLike,
        /,
    ):
        if not isinstance(gas, ComovingEulerPlan):
            raise TypeError("gas must be ComovingEulerPlan.")
        if not isinstance(particles, CosmologicalKDKPlan):
            raise TypeError("particles must be CosmologicalKDKPlan.")
        if not isinstance(gravity, ParticleMeshGravityPlan):
            raise TypeError("gravity must be ParticleMeshGravityPlan.")
        if particles.particles.prepared_id != gravity.transfer.particles.prepared_id:
            raise ValueError("Gas-particle coupling must share one particle support.")
        if (
            gas.dynamics.discretization.grid.prepared_id
            != gravity.transfer.plan.target.prepared_id
        ):
            raise ValueError("Gas and particle gravity must share one prepared grid.")
        schedule = np.asarray(scale_factors, dtype=float).reshape((-1,))
        if (
            schedule.size < 2
            or np.any(~np.isfinite(schedule))
            or np.any(schedule <= 0.0)
            or np.any(np.diff(schedule) <= 0.0)
        ):
            raise ValueError("Coupled scale-factor schedule is invalid.")
        self.gas = gas
        self.particles = particles
        self.gravity = gravity
        self.scale_factors = jnp.asarray(schedule)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cosmological-adiabatic-gas-particle-gravity",
                "gas": gas.plan_id,
                "particles": particles.plan_id,
                "gravity": gravity.plan_id,
                "scale_factors": schedule.tolist(),
            }
        )

    def shared_gravity(
        self,
        state: CosmologicalGasParticleState,
        args: Any = None,
        /,
    ) -> SharedGasParticleGravityResult:
        deposited, routes = self.gravity.density(state.particles.positions)
        gas_density = state.gas.cell_average[..., 0]
        if gas_density.shape != deposited.density.shape:
            raise ValueError("Gas density and particle gravity grid disagree.")
        total_density = gas_density + deposited.density
        potential, _, cell_acceleration, solved = self.gravity.gravity.solve_density(
            total_density, args
        )
        gathered = self.gravity.transfer.gather(routes, cell_acceleration)
        active_mask = self.particles.particles.active_mask
        particle_acceleration = jnp.where(active_mask[:, None], gathered.values, 0.0)
        cell_volumes = self.gas.dynamics.discretization.cell_volumes
        gas_force = contract(
            "...,...d->d",
            gas_density * cell_volumes,
            cell_acceleration,
        )
        particle_force = jnp.sum(
            self.particles.particles.masses[:, None] * particle_acceleration,
            axis=0,
        )
        support_complete = jnp.all(gathered.support | ~active_mask)
        successful = (
            deposited.successful
            & solved.converged
            & support_complete
            & jnp.all(jnp.isfinite(cell_acceleration))
            & jnp.all(jnp.isfinite(particle_acceleration))
        )
        return SharedGasParticleGravityResult(
            potential,
            cell_acceleration,
            particle_acceleration,
            deposited.balance.maximum_absolute_balance_defect,
            gas_force + particle_force,
            successful,
        )

    def rollout(
        self,
        background: FLRWBackground,
        state: CosmologicalGasParticleState,
        args: Any = None,
        /,
    ) -> CosmologicalGasParticleResult:
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        if not isinstance(state, CosmologicalGasParticleState):
            raise TypeError("state must be CosmologicalGasParticleState.")
        initial_scale = background.require_flat(
            self.scale_factors[0].astype(state.gas.scale_factor.dtype)
        )
        initial_scale = eqx.error_if(
            initial_scale,
            (state.gas.scale_factor != initial_scale)
            | (state.particles.scale_factor != initial_scale),
            "Coupled gas and particles must start at the first scheduled scale.",
        )
        state = CosmologicalGasParticleState(
            ComovingEulerState(state.gas.cell_average, initial_scale),
            CosmologicalParticleState(
                state.particles.positions,
                state.particles.canonical_momenta,
                initial_scale,
            ),
        )

        def step(carry, end):
            current, active, count = carry

            def attempt(_):
                force_0 = self.shared_gravity(current, args)
                proposal = self.particles.propose(
                    background,
                    current.particles,
                    end,
                    force_0.particle_acceleration,
                )
                predicted_gas, predicted_evidence = self.gas.advance(
                    background,
                    current.gas,
                    end,
                    force_0.cell_acceleration,
                    force_0.cell_acceleration,
                    args,
                )
                predicted_state = CosmologicalGasParticleState(
                    predicted_gas,
                    CosmologicalParticleState(
                        proposal.positions,
                        proposal.half_momenta,
                        proposal.end_scale_factor,
                    ),
                )
                predicted_force = self.shared_gravity(predicted_state, args)
                corrected_gas, gas_evidence = self.gas.advance(
                    background,
                    current.gas,
                    end,
                    force_0.cell_acceleration,
                    predicted_force.cell_acceleration,
                    args,
                )
                endpoint_state = CosmologicalGasParticleState(
                    corrected_gas,
                    predicted_state.particles,
                )
                endpoint_force = self.shared_gravity(endpoint_state, args)
                completed_particles, particle_evidence = self.particles.complete(
                    current.particles,
                    proposal,
                    endpoint_force.particle_acceleration,
                )
                successful = (
                    active
                    & force_0.successful
                    & predicted_evidence.successful
                    & predicted_force.successful
                    & gas_evidence.successful
                    & endpoint_force.successful
                    & particle_evidence.successful
                )
                accepted = CosmologicalGasParticleState(
                    ComovingEulerState(
                        jnp.where(
                            successful,
                            corrected_gas.cell_average,
                            current.gas.cell_average,
                        ),
                        jnp.where(
                            successful,
                            corrected_gas.scale_factor,
                            current.gas.scale_factor,
                        ),
                    ),
                    CosmologicalParticleState(
                        jnp.where(
                            successful,
                            completed_particles.positions,
                            current.particles.positions,
                        ),
                        jnp.where(
                            successful,
                            completed_particles.canonical_momenta,
                            current.particles.canonical_momenta,
                        ),
                        jnp.where(
                            successful,
                            completed_particles.scale_factor,
                            current.particles.scale_factor,
                        ),
                    ),
                )
                diagnostics = (
                    successful,
                    gas_evidence.successful,
                    endpoint_force.successful,
                    jnp.maximum(
                        force_0.mass_balance_defect,
                        endpoint_force.mass_balance_defect,
                    ),
                    endpoint_force.net_force,
                )
                return (
                    accepted,
                    successful,
                    count + successful.astype(jnp.int32),
                ), diagnostics

            def stopped(_):
                zero = jnp.asarray(0.0, dtype=current.gas.cell_average.dtype)
                diagnostics = (
                    jnp.asarray(False),
                    jnp.asarray(False),
                    jnp.asarray(False),
                    zero,
                    jnp.zeros((self.gas.dimension,), dtype=zero.dtype),
                )
                return carry, diagnostics

            return jax.lax.cond(active, attempt, stopped, operand=None)

        initial_carry = (state, jnp.asarray(True), jnp.asarray(0, dtype=jnp.int32))
        final_carry, recorded = jax.lax.scan(
            step,
            initial_carry,
            self.scale_factors[1:].astype(state.gas.scale_factor.dtype),
        )
        final_state, completed, accepted_steps = final_carry
        accepted, gas_success, gravity_success, mass_defect, net_force = recorded
        failed = ~accepted
        first_failed = jnp.where(
            jnp.any(failed),
            jnp.argmax(failed).astype(jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
        )
        diagnostics = CosmologicalGasParticleDiagnostics(
            accepted,
            gas_success,
            gravity_success,
            mass_defect,
            net_force,
            accepted_steps,
            completed,
            first_failed,
        )
        return CosmologicalGasParticleResult(final_state, diagnostics, completed)


__all__ = [
    "ComovingEulerDiagnostics",
    "ComovingEulerPlan",
    "ComovingEulerState",
    "CosmologicalGasParticleDiagnostics",
    "CosmologicalGasParticleGravityPlan",
    "CosmologicalGasParticleResult",
    "CosmologicalGasParticleState",
    "SharedGasParticleGravityResult",
]
