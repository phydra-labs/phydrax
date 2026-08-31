#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.pic import (
    ChargeConservingCurrentPlan,
    PICEnergyLedger,
    PICMaxwellCurrentArguments,
    PICMaxwellCurrentSource,
    PICParticleState,
    PICRejectionReason,
    PICRunStatus,
    PreparedPICParticleCochainTransfer,
    RelativisticBorisPlan,
)
from ._cochain_electrostatic import CochainElectrostaticPlan
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ._maxwell import CompatibleMaxwellState, PreparedCompatibleMaxwell


class ElectromagneticPICState(StrictModule):
    particles: tuple[PICParticleState, ...]
    maxwell: CompatibleMaxwellState
    time: Array
    accepted_step: Array
    status: Array


class ElectromagneticPICDiagnostics(StrictModule):
    continuity_defect: Array
    particle_maxwell_charge_defect: Array
    electric_constraint: Array
    magnetic_constraint: Array
    maximum_displacement_fraction: Array
    energy: PICEnergyLedger
    field: Any
    transfer_successful: Array
    current_successful: Array
    pusher_successful: Array
    finite: Array
    successful: Array
    rejection_reason: Array


class ElectromagneticPICStepResult(StrictModule):
    candidate_state: ElectromagneticPICState
    accepted_state: ElectromagneticPICState
    diagnostics: ElectromagneticPICDiagnostics
    current: Array
    successful: Array


class ElectromagneticPICPlan(StrictModule, NonTrainableState):
    """Three-dimensional periodic fixed-population electromagnetic PIC."""

    maxwell: PreparedCompatibleMaxwell
    electrostatic: CochainElectrostaticPlan
    transfers: tuple[PreparedPICParticleCochainTransfer, ...]
    currents: tuple[ChargeConservingCurrentPlan, ...]
    pusher: RelativisticBorisPlan
    maximum_displacement_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maxwell: PreparedCompatibleMaxwell,
        electrostatic: CochainElectrostaticPlan,
        transfers: Sequence[PreparedPICParticleCochainTransfer],
        currents: Sequence[ChargeConservingCurrentPlan],
        /,
        *,
        pusher: RelativisticBorisPlan | None = None,
        maximum_displacement_fraction: float = 0.5,
    ):
        if not isinstance(maxwell, PreparedCompatibleMaxwell):
            raise TypeError("maxwell must be PreparedCompatibleMaxwell.")
        if not isinstance(electrostatic, CochainElectrostaticPlan):
            raise TypeError("electrostatic must be CochainElectrostaticPlan.")
        transfer_values = tuple(transfers)
        current_values = tuple(currents)
        if not transfer_values or len(transfer_values) != len(current_values):
            raise ValueError("One transfer and current plan is required per PIC species.")
        if any(
            not isinstance(value, PreparedPICParticleCochainTransfer)
            for value in transfer_values
        ) or any(not isinstance(value, ChargeConservingCurrentPlan) for value in current_values):
            raise TypeError("PIC transfers and current plans have incompatible types.")
        bridge = maxwell.plan.bridge
        if bridge.dimension != 3 or any(not axis.periodic for axis in bridge.grid.structured_axes):
            raise ValueError("Electromagnetic PIC currently requires a periodic 3-D grid.")
        if electrostatic.bridge.bridge_id != bridge.bridge_id or any(
            value.bridge.bridge_id != bridge.bridge_id for value in transfer_values
        ):
            raise ValueError("PIC electrostatic, transfer, and Maxwell plans must share one bridge.")
        if any(
            current.transfer.prepared_id != transfer.prepared_id
            for current, transfer in zip(current_values, transfer_values, strict=True)
        ):
            raise ValueError("Every current plan must use its matching PIC transfer.")
        if not isinstance(maxwell.plan.current_source, PICMaxwellCurrentSource):
            raise ValueError(
                "Maxwell PIC requires CompatibleMaxwellPlan(current_source=PICMaxwellCurrentSource())."
            )
        if maxwell.pml is not None or maxwell.boundaries:
            raise ValueError("Initial electromagnetic PIC supports periodic fields without PML.")
        if not maxwell.capabilities.lossless or maxwell.capabilities.dispersive:
            raise ValueError("Initial electromagnetic PIC requires lossless instantaneous Maxwell material.")
        pusher_ = RelativisticBorisPlan() if pusher is None else pusher
        maximum = float(maximum_displacement_fraction)
        if not isinstance(pusher_, RelativisticBorisPlan):
            raise TypeError("pusher must be RelativisticBorisPlan or None.")
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_displacement_fraction must be positive and finite.")
        self.maxwell = maxwell
        self.electrostatic = electrostatic
        self.transfers = transfer_values
        self.currents = current_values
        self.pusher = pusher_
        self.maximum_displacement_fraction = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electromagnetic-pic-plan",
                "maxwell": maxwell.prepared_id,
                "electrostatic": electrostatic.plan_id,
                "transfers": [value.prepared_id for value in transfer_values],
                "currents": [value.plan_id for value in current_values],
                "pusher": pusher_.plan_id,
            }
        )

    def _proper_velocity(self, velocity, transfer):
        value = jnp.asarray(velocity, dtype=transfer.species.particles.safe_masses.dtype)
        expected = (transfer.species.capacity, 3)
        if value.shape != expected:
            raise ValueError(f"Electromagnetic PIC velocity must have shape {expected}.")
        speed2 = jnp.sum(value * value, axis=-1)
        c2 = self.pusher.speed_of_light**2
        value = eqx.error_if(
            value,
            jnp.any(transfer.species.particles.active_mask & (~jnp.isfinite(speed2) | (speed2 >= c2))),
            "Initial electromagnetic PIC velocity must be finite and subluminal.",
        )
        gamma = 1.0 / jnp.sqrt(1.0 - speed2 / c2)
        return jnp.where(transfer.species.particles.active_mask[:, None], gamma[:, None] * value, 0.0)

    def _charge(self, particles):
        routes = tuple(
            transfer.build(state.position)
            for transfer, state in zip(self.transfers, particles, strict=True)
        )
        deposits = tuple(
            transfer.deposit_charge(route)
            for transfer, route in zip(self.transfers, routes, strict=True)
        )
        charge = sum(
            (value.cochain for value in deposits),
            jnp.zeros((self.maxwell.primary_counts[2],), dtype=particles[0].position.dtype),
        )
        successful = jnp.all(jnp.stack(tuple(value.successful for value in deposits)))
        return charge, routes, successful

    def initialize(
        self,
        positions: Sequence[ArrayLike],
        velocities: Sequence[ArrayLike],
        step_size: ArrayLike,
        /,
        *,
        magnetic_flux: ArrayLike | None = None,
        time: ArrayLike = 0.0,
    ) -> ElectromagneticPICState:
        if len(tuple(positions)) != len(self.transfers) or len(tuple(velocities)) != len(
            self.transfers
        ):
            raise ValueError("One position and velocity array is required per species.")
        particle_states = []
        for transfer, position, velocity in zip(
            self.transfers, tuple(positions), tuple(velocities), strict=True
        ):
            position_ = jnp.asarray(position, dtype=transfer.species.particles.safe_masses.dtype)
            expected = (transfer.species.capacity, 3)
            if position_.shape != expected:
                raise ValueError(f"Electromagnetic PIC position must have shape {expected}.")
            particle_states.append(
                PICParticleState(
                    jnp.where(transfer.species.particles.active_mask[:, None], position_, 0.0),
                    self._proper_velocity(velocity, transfer),
                )
            )
        particles = tuple(particle_states)
        charge, routes, charge_success = self._charge(particles)
        electrostatic = self.electrostatic.solve(charge)
        displacement = self.maxwell.constitutive.electric_displacement(
            electrostatic.electric,
            self.maxwell.constitutive.initialize_state(),
        )
        magnetic = (
            jnp.zeros((self.maxwell.primary_counts[1],), dtype=displacement.dtype)
            if magnetic_flux is None
            else jnp.asarray(magnetic_flux, dtype=displacement.dtype)
        )
        maxwell_state = self.maxwell.pack(
            displacement,
            magnetic,
            charge,
            material_state=self.maxwell.constitutive.initialize_state(),
        )
        electric = self.maxwell.electric_field(maxwell_state)
        magnetic_field = self.maxwell.magnetic_field(maxwell_state)
        dt = jnp.asarray(step_size, dtype=displacement.dtype).reshape(())
        bootstrapped = []
        for transfer, route, particle in zip(
            self.transfers, routes, particles, strict=True
        ):
            e = transfer.gather_electric(route, electric)
            b = transfer.gather_magnetic(route, magnetic_field)
            backward = self.pusher.push(
                particle.proper_velocity,
                e.values,
                b.values,
                transfer.species.specific_charge,
                transfer.species.particles.active_mask,
                -0.5 * dt,
            )
            bootstrapped.append(PICParticleState(particle.position, backward.proper_velocity))
            charge_success = charge_success & e.successful & b.successful & backward.successful
        state = ElectromagneticPICState(
            tuple(bootstrapped),
            maxwell_state,
            jnp.asarray(time, dtype=dt.dtype).reshape(()),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(PICRunStatus.SUCCESS), dtype=jnp.int32),
        )
        return jax.tree.map(
            lambda value: eqx.error_if(
                value,
                ~charge_success | ~electrostatic.successful,
                "Electromagnetic PIC initialization failed.",
            )
            if eqx.is_array(value)
            else value,
            state,
        )

    def _kinetic(self, particles):
        total = jnp.asarray(0.0, dtype=particles[0].position.dtype)
        c2 = self.pusher.speed_of_light**2
        for transfer, particle in zip(self.transfers, particles, strict=True):
            gamma = jnp.sqrt(1.0 + jnp.sum(particle.proper_velocity**2, axis=-1) / c2)
            total = total + jnp.sum(
                jnp.where(
                    transfer.species.particles.active_mask,
                    transfer.species.particles.masses.astype(gamma.dtype) * c2 * (gamma - 1.0),
                    0.0,
                )
            )
        return total

    def step_detailed(
        self, state: ElectromagneticPICState, step_size: ArrayLike, /
    ) -> ElectromagneticPICStepResult:
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        electric_cochain = self.maxwell.electric_field(state.maxwell)
        magnetic_cochain = self.maxwell.magnetic_field(state.maxwell)
        next_particles = []
        current_results = []
        pusher_success = jnp.asarray(True)
        maximum_fraction = jnp.asarray(0.0, dtype=dt.dtype)
        widths = jnp.asarray(
            [jnp.min(axis.interval_widths) for axis in self.maxwell.plan.bridge.grid.structured_axes],
            dtype=dt.dtype,
        )
        for transfer, current_plan, particle in zip(
            self.transfers, self.currents, state.particles, strict=True
        ):
            routes = transfer.build(particle.position)
            electric = transfer.gather_electric(routes, electric_cochain)
            magnetic = transfer.gather_magnetic(routes, magnetic_cochain)
            pushed = self.pusher.push(
                particle.proper_velocity,
                electric.values,
                magnetic.values,
                transfer.species.specific_charge,
                transfer.species.particles.active_mask,
                dt,
            )
            displacement = dt * pushed.velocity
            position = particle.position + displacement
            position = jnp.where(transfer.species.particles.active_mask[:, None], position, 0.0)
            fraction = jnp.max(
                jnp.where(
                    transfer.species.particles.active_mask[:, None],
                    jnp.abs(displacement) / widths,
                    0.0,
                ),
                initial=0.0,
            )
            maximum_fraction = jnp.maximum(maximum_fraction, fraction)
            current = current_plan.deposit(particle.position, position, dt)
            current_results.append(current)
            next_particles.append(PICParticleState(position, pushed.proper_velocity))
            pusher_success = pusher_success & electric.successful & magnetic.successful & pushed.successful
        total_current = sum(
            (value.current for value in current_results),
            jnp.zeros((self.maxwell.primary_counts[0],), dtype=dt.dtype),
        )
        args = PICMaxwellCurrentArguments(total_current, None)
        candidate_maxwell = self.maxwell.leapfrog_step(
            state.time, state.maxwell, dt, args
        )
        field_diagnostics = self.maxwell.diagnostics(
            state.time + dt, candidate_maxwell, args
        )
        end_charge = sum(
            (value.end_charge.cochain for value in current_results),
            jnp.zeros_like(candidate_maxwell.primary.charge),
        )
        charge_defect = jnp.max(
            jnp.abs(candidate_maxwell.primary.charge - end_charge), initial=0.0
        )
        continuity = jnp.max(
            jnp.stack(tuple(value.maximum_continuity_defect for value in current_results)),
            initial=0.0,
        )
        current_success = jnp.all(
            jnp.stack(tuple(value.successful for value in current_results))
        )
        previous_kinetic = self._kinetic(state.particles)
        next_tuple = tuple(next_particles)
        next_kinetic = self._kinetic(next_tuple)
        previous_field = self.maxwell.energy(state.maxwell)
        next_field = self.maxwell.energy(candidate_maxwell)
        previous_total = previous_kinetic + previous_field
        total = next_kinetic + next_field
        energy = PICEnergyLedger(
            next_kinetic,
            next_field,
            jnp.asarray(0.0, dtype=total.dtype),
            total,
            previous_total,
            total - previous_total,
        )
        stable = (
            dt <= self.maxwell.stable_dt
        ) & (maximum_fraction <= self.maximum_displacement_fraction)
        finite = (
            jnp.isfinite(dt)
            & (dt > 0.0)
            & jnp.all(jnp.isfinite(total_current))
            & jnp.isfinite(total)
        )
        constraints = (
            (continuity <= 1.0e-9)
            & (charge_defect <= 1.0e-9)
            & (field_diagnostics.electric_constraint_linf <= 1.0e-8)
            & (field_diagnostics.magnetic_constraint_linf <= 1.0e-8)
        )
        successful = pusher_success & current_success & stable & finite & constraints
        reason = jnp.asarray(int(PICRejectionReason.NONE), dtype=jnp.int32)
        reason = jnp.where(
            current_success, reason, reason | int(PICRejectionReason.CONTINUITY)
        )
        reason = jnp.where(
            stable, reason, reason | int(PICRejectionReason.DISPLACEMENT)
        )
        reason = jnp.where(
            field_diagnostics.electric_constraint_linf <= 1.0e-8,
            reason,
            reason | int(PICRejectionReason.GAUSS),
        )
        reason = jnp.where(
            field_diagnostics.magnetic_constraint_linf <= 1.0e-8,
            reason,
            reason | int(PICRejectionReason.MAGNETIC),
        )
        reason = jnp.where(finite, reason, reason | int(PICRejectionReason.NONFINITE))
        candidate = ElectromagneticPICState(
            next_tuple,
            candidate_maxwell,
            state.time + dt,
            state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
            jnp.where(successful, int(PICRunStatus.SUCCESS), int(PICRunStatus.INVALID_STATE)).astype(jnp.int32),
        )
        accepted = jax.tree.map(
            lambda proposed, current: jnp.where(successful, proposed, current),
            candidate,
            state,
        )
        diagnostics = ElectromagneticPICDiagnostics(
            continuity,
            charge_defect,
            field_diagnostics.electric_constraint_linf,
            field_diagnostics.magnetic_constraint_linf,
            maximum_fraction,
            energy,
            field_diagnostics,
            pusher_success,
            current_success,
            pusher_success,
            finite,
            successful,
            reason,
        )
        return ElectromagneticPICStepResult(
            candidate, accepted, diagnostics, total_current, successful
        )


class ElectromagneticPICFixedStepMethod(AbstractFixedStepMethod, NonTrainableState):
    plan: ElectromagneticPICPlan
    method_id: str = eqx.field(static=True)

    def __init__(self, plan: ElectromagneticPICPlan, /):
        self.plan = plan
        self.method_id = canonical_fingerprint(
            {"kind": "electromagnetic-pic-fixed-step", "plan": plan.plan_id}
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: ElectromagneticPICState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index, time, args
        result = self.plan.step_detailed(state, step_size)
        residual = jnp.max(
            jnp.stack(
                (
                    result.diagnostics.continuity_defect,
                    result.diagnostics.particle_maxwell_charge_defect,
                    result.diagnostics.electric_constraint,
                    result.diagnostics.magnetic_constraint,
                )
            )
        )
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            residual,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=state.time.dtype),
        )


__all__ = [
    "ElectromagneticPICDiagnostics",
    "ElectromagneticPICFixedStepMethod",
    "ElectromagneticPICPlan",
    "ElectromagneticPICState",
    "ElectromagneticPICStepResult",
]
