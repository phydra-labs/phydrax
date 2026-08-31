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
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.pic import (
    PICEnergyLedger,
    PICParticleState,
    PICRejectionReason,
    PICRunStatus,
    PreparedPICParticleCochainTransfer,
    RelativisticBorisPlan,
)
from ._cochain_electrostatic import CochainElectrostaticPlan, CochainElectrostaticResult
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult


class ElectrostaticPICState(StrictModule):
    particles: tuple[PICParticleState, ...]
    charge: Array
    potential: Array
    electric: Array
    time: Array
    accepted_step: Array
    status: Array


class ElectrostaticPICDiagnostics(StrictModule):
    charge_balance_defect: Array
    poisson_residual: Array
    gauss_defect: Array
    maximum_displacement_fraction: Array
    energy: PICEnergyLedger
    transfer_successful: Array
    pusher_successful: Array
    field_successful: Array
    finite: Array
    successful: Array
    rejection_reason: Array


class ElectrostaticPICStepResult(StrictModule):
    candidate_state: ElectrostaticPICState
    accepted_state: ElectrostaticPICState
    diagnostics: ElectrostaticPICDiagnostics
    field: CochainElectrostaticResult
    successful: Array


class ElectrostaticPICPlan(StrictModule, NonTrainableState):
    """Fixed-population compatible electrostatic PIC with kick-drift-kick stepping."""

    field: CochainElectrostaticPlan
    transfers: tuple[PreparedPICParticleCochainTransfer, ...]
    pusher: RelativisticBorisPlan
    background_charge: Array
    maximum_displacement_fraction: float = eqx.field(static=True)
    discretization_bundle: DiscretizationBundle
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: CochainElectrostaticPlan,
        transfers: Sequence[PreparedPICParticleCochainTransfer],
        /,
        *,
        pusher: RelativisticBorisPlan | None = None,
        background_charge: ArrayLike | None = None,
        maximum_displacement_fraction: float = 0.5,
    ):
        if not isinstance(field, CochainElectrostaticPlan):
            raise TypeError("field must be CochainElectrostaticPlan.")
        values = tuple(transfers)
        if not values or any(
            not isinstance(value, PreparedPICParticleCochainTransfer) for value in values
        ):
            raise TypeError("transfers must contain prepared PIC cochain transfers.")
        if any(value.bridge.bridge_id != field.bridge.bridge_id for value in values):
            raise ValueError("PIC transfers and electrostatic field must share one bridge.")
        support_ids = [value.species.particles.support.support_id for value in values]
        if len(set(support_ids)) != len(support_ids):
            raise ValueError("Each electrostatic PIC species requires a distinct particle support.")
        pusher_ = RelativisticBorisPlan() if pusher is None else pusher
        if not isinstance(pusher_, RelativisticBorisPlan):
            raise TypeError("pusher must be RelativisticBorisPlan or None.")
        maximum = float(maximum_displacement_fraction)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_displacement_fraction must be positive and finite.")
        n0 = field.bridge.cochain.cell_counts[0]
        background = (
            jnp.zeros((n0,), dtype=field.bridge.cochain.hodge_stars[0].dtype)
            if background_charge is None
            else jnp.asarray(background_charge, dtype=field.bridge.cochain.hodge_stars[0].dtype)
        )
        if background.shape != (n0,):
            raise ValueError("background_charge must be a degree-zero cochain.")
        background = eqx.error_if(
            background,
            jnp.any(~jnp.isfinite(background)),
            "background_charge must be finite.",
        )
        records = []
        species_keys = []
        for index, value in enumerate(values):
            particles = value.species.particles
            species_key = DiscretizationKey(
                f"pic_species_{index}_{value.species.plan.species_id}",
                DiscretizationRole.PHYSICAL,
                domain_labels=field.bridge.grid.axis_names,
            )
            species_keys.append(species_key)
            records.append(
                DiscretizationRecord(
                    species_key,
                    "prepared-charged-particles",
                    value.species.prepared_id,
                    numeric_version=particles.numeric_version,
                    precision_evidence_id=particles.precision_evidence_id,
                    resource_evidence_id=particles.resource_evidence_id,
                )
            )
        field_key = DiscretizationKey(
            "electrostatic_pic",
            DiscretizationRole.RESIDUAL,
            domain_labels=field.bridge.grid.axis_names,
        )
        records.append(
            DiscretizationRecord(
                field_key,
                "cochain-electrostatic-pic",
                field.plan_id,
                dependency_key_ids=tuple(value.key_id for value in species_keys),
            )
        )
        self.field = field
        self.transfers = values
        self.pusher = pusher_
        self.background_charge = background
        self.maximum_displacement_fraction = maximum
        self.discretization_bundle = DiscretizationBundle(tuple(records))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electrostatic-pic-plan",
                "field": field.plan_id,
                "transfers": [value.prepared_id for value in values],
                "pusher": pusher_.plan_id,
                "maximum_displacement_fraction": maximum,
            }
        )

    def _proper_velocity(self, velocity: ArrayLike, transfer, /) -> Array:
        value = jnp.asarray(velocity, dtype=transfer.species.particles.safe_masses.dtype)
        capacity = transfer.species.capacity
        dimension = transfer.species.spatial_dimension
        if value.shape not in ((capacity, dimension), (capacity, 3)):
            raise ValueError("velocity must have particle-capacity by d or 3 shape.")
        if value.shape[-1] < 3:
            value = jnp.pad(value, ((0, 0), (0, 3 - value.shape[-1])))
        speed2 = jnp.sum(value * value, axis=-1)
        c2 = self.pusher.speed_of_light**2
        value = eqx.error_if(
            value,
            jnp.any(transfer.species.particles.active_mask & (~jnp.isfinite(speed2) | (speed2 >= c2))),
            "Initial PIC velocity must be finite and subluminal.",
        )
        gamma = 1.0 / jnp.sqrt(1.0 - speed2 / c2)
        return jnp.where(
            transfer.species.particles.active_mask[:, None], gamma[:, None] * value, 0.0
        )

    def _charge_and_field(
        self,
        particles: tuple[PICParticleState, ...],
        /,
        *,
        initial_potential: ArrayLike | None = None,
    ):
        route_states = tuple(
            transfer.build(state.position)
            for transfer, state in zip(self.transfers, particles, strict=True)
        )
        deposits = tuple(
            transfer.deposit_charge(routes)
            for transfer, routes in zip(self.transfers, route_states, strict=True)
        )
        charge = self.background_charge + sum(
            (value.cochain for value in deposits),
            jnp.zeros_like(self.background_charge),
        )
        field = self.field.solve(charge, initial_potential=initial_potential)
        gathered = tuple(
            transfer.gather_electric(routes, field.electric)
            for transfer, routes in zip(self.transfers, route_states, strict=True)
        )
        transfer_successful = jnp.all(
            jnp.stack(
                tuple(value.successful for value in deposits)
                + tuple(value.successful for value in gathered)
            )
        )
        charge_defect = jnp.max(
            jnp.stack(tuple(value.balance.maximum_absolute_balance_defect for value in deposits)),
            initial=0.0,
        )
        return charge, field, gathered, transfer_successful, charge_defect

    def initialize(
        self,
        positions: Sequence[ArrayLike],
        velocities: Sequence[ArrayLike],
        /,
        *,
        time: ArrayLike = 0.0,
    ) -> ElectrostaticPICState:
        position_values = tuple(positions)
        velocity_values = tuple(velocities)
        if len(position_values) != len(self.transfers) or len(velocity_values) != len(
            self.transfers
        ):
            raise ValueError("One position and velocity array is required per PIC species.")
        particles = []
        for transfer, position, velocity in zip(
            self.transfers, position_values, velocity_values, strict=True
        ):
            position_ = jnp.asarray(position, dtype=transfer.species.particles.safe_masses.dtype)
            expected = (transfer.species.capacity, transfer.species.spatial_dimension)
            if position_.shape != expected:
                raise ValueError(f"PIC position must have shape {expected}.")
            position_ = jnp.where(
                transfer.species.particles.active_mask[:, None], position_, 0.0
            )
            particles.append(PICParticleState(position_, self._proper_velocity(velocity, transfer)))
        particle_tuple = tuple(particles)
        charge, field, _, transfer_successful, _ = self._charge_and_field(particle_tuple)
        state = ElectrostaticPICState(
            particle_tuple,
            charge,
            field.potential,
            field.electric,
            jnp.asarray(time, dtype=field.potential.dtype).reshape(()),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(PICRunStatus.SUCCESS), dtype=jnp.int32),
        )
        return jax.tree.map(
            lambda value: eqx.error_if(
                value, ~transfer_successful | ~field.successful, "PIC initialization failed."
            )
            if eqx.is_array(value)
            else value,
            state,
        )

    def _kinetic_energy(self, particles: tuple[PICParticleState, ...], /) -> Array:
        total = jnp.asarray(0.0, dtype=particles[0].position.dtype)
        c2 = self.pusher.speed_of_light**2
        for transfer, state in zip(self.transfers, particles, strict=True):
            gamma = jnp.sqrt(1.0 + jnp.sum(state.proper_velocity**2, axis=-1) / c2)
            mass = transfer.species.particles.masses.astype(gamma.dtype)
            total = total + jnp.sum(
                jnp.where(
                    transfer.species.particles.active_mask,
                    mass * c2 * (gamma - 1.0),
                    0.0,
                )
            )
        return total

    def step_detailed(
        self,
        state: ElectrostaticPICState,
        step_size: ArrayLike,
        /,
    ) -> ElectrostaticPICStepResult:
        if not isinstance(state, ElectrostaticPICState):
            raise TypeError("state must be ElectrostaticPICState.")
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        current_routes = tuple(
            transfer.build(value.position)
            for transfer, value in zip(self.transfers, state.particles, strict=True)
        )
        current_electric = tuple(
            transfer.gather_electric(routes, state.electric)
            for transfer, routes in zip(self.transfers, current_routes, strict=True)
        )
        half_states = []
        drifted = []
        pusher_success = jnp.asarray(True)
        maximum_fraction = jnp.asarray(0.0, dtype=dt.dtype)
        widths = jnp.asarray(
            [jnp.min(axis.interval_widths) for axis in self.field.bridge.grid.structured_axes],
            dtype=dt.dtype,
        )
        for transfer, particle, electric in zip(
            self.transfers, state.particles, current_electric, strict=True
        ):
            zeros = jnp.zeros_like(electric.values)
            half = self.pusher.push(
                particle.proper_velocity,
                electric.values,
                zeros,
                transfer.species.specific_charge,
                transfer.species.particles.active_mask,
                0.5 * dt,
            )
            displacement = dt * half.velocity[:, : transfer.species.spatial_dimension]
            position = particle.position + displacement
            active = transfer.species.particles.active_mask
            position = jnp.where(active[:, None], position, 0.0)
            fraction = jnp.max(
                jnp.where(active[:, None], jnp.abs(displacement) / widths, 0.0),
                initial=0.0,
            )
            maximum_fraction = jnp.maximum(maximum_fraction, fraction)
            pusher_success = pusher_success & half.successful
            half_states.append(half)
            drifted.append(PICParticleState(position, half.proper_velocity))
        drifted_tuple = tuple(drifted)
        charge, field, next_electric, transfer_success, charge_defect = self._charge_and_field(
            drifted_tuple, initial_potential=state.potential
        )
        final_particles = []
        for transfer, particle, electric in zip(
            self.transfers, drifted_tuple, next_electric, strict=True
        ):
            zeros = jnp.zeros_like(electric.values)
            final = self.pusher.push(
                particle.proper_velocity,
                electric.values,
                zeros,
                transfer.species.specific_charge,
                transfer.species.particles.active_mask,
                0.5 * dt,
            )
            pusher_success = pusher_success & final.successful
            final_particles.append(PICParticleState(particle.position, final.proper_velocity))
        final_tuple = tuple(final_particles)
        previous_total = self._kinetic_energy(state.particles) + 0.5 * jnp.real(
            self.field.bridge.cochain.space(1).vector_space.inner(
                state.electric, self.field.permittivity * state.electric
            )
        )
        total = self._kinetic_energy(final_tuple) + field.field_energy
        energy = PICEnergyLedger(
            self._kinetic_energy(final_tuple),
            field.field_energy,
            jnp.asarray(0.0, dtype=total.dtype),
            total,
            previous_total,
            total - previous_total,
        )
        particle_finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value.position))
                    & jnp.all(jnp.isfinite(value.proper_velocity))
                    for value in final_tuple
                )
            )
        )
        finite = (
            jnp.isfinite(dt)
            & (dt > 0.0)
            & particle_finite
            & jnp.isfinite(total)
        )
        stable = maximum_fraction <= self.maximum_displacement_fraction
        successful = transfer_success & field.successful & pusher_success & finite & stable
        reason = jnp.asarray(int(PICRejectionReason.NONE), dtype=jnp.int32)
        reason = jnp.where(transfer_success, reason, reason | int(PICRejectionReason.ROUTE))
        reason = jnp.where(field.successful, reason, reason | int(PICRejectionReason.FIELD))
        reason = jnp.where(pusher_success, reason, reason | int(PICRejectionReason.PUSHER))
        reason = jnp.where(stable, reason, reason | int(PICRejectionReason.DISPLACEMENT))
        reason = jnp.where(finite, reason, reason | int(PICRejectionReason.NONFINITE))
        candidate = ElectrostaticPICState(
            final_tuple,
            charge,
            field.potential,
            field.electric,
            state.time + dt,
            state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
            jnp.where(successful, int(PICRunStatus.SUCCESS), int(PICRunStatus.INVALID_STATE)).astype(jnp.int32),
        )
        accepted = jax.tree.map(
            lambda proposed, current: jnp.where(successful, proposed, current),
            candidate,
            state,
        )
        diagnostics = ElectrostaticPICDiagnostics(
            charge_defect,
            field.residual_norm,
            field.residual_norm,
            maximum_fraction,
            energy,
            transfer_success,
            pusher_success,
            field.successful,
            finite,
            successful,
            reason,
        )
        return ElectrostaticPICStepResult(candidate, accepted, diagnostics, field, successful)


class ElectrostaticPICFixedStepMethod(AbstractFixedStepMethod, NonTrainableState):
    plan: ElectrostaticPICPlan
    method_id: str = eqx.field(static=True)

    def __init__(self, plan: ElectrostaticPICPlan, /):
        if not isinstance(plan, ElectrostaticPICPlan):
            raise TypeError("plan must be ElectrostaticPICPlan.")
        self.plan = plan
        self.method_id = canonical_fingerprint(
            {"kind": "electrostatic-pic-fixed-step", "plan": plan.plan_id}
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: ElectrostaticPICState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index, time, args
        result = self.plan.step_detailed(state, step_size)
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            jnp.maximum(result.diagnostics.poisson_residual, result.diagnostics.gauss_defect),
            result.field.linear.diagnostics.iterations,
            result.field.linear.diagnostics.iterations,
            jnp.asarray(False),
            jnp.zeros((), dtype=state.time.dtype),
        )


__all__ = [
    "ElectrostaticPICDiagnostics",
    "ElectrostaticPICFixedStepMethod",
    "ElectrostaticPICPlan",
    "ElectrostaticPICState",
    "ElectrostaticPICStepResult",
]
