#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._relativistic_neutrino import (
    GRNeutrinoInteractionPlan,
    GRNeutrinoM1System,
    GRNeutrinoMatterExchange,
)
from ..metrix._adm_exchange import StressEnergyProjection
from ._gr_multigroup_radiation import (
    FixedGridGRMultigroupM1SSPRK3Plan,
    GRMultigroupM1State,
    GRMultigroupM1StepResult,
)
from ._relativistic_finite_volume import ValenciaFiniteVolumeStageGeometry


class GRNeutrinoM1State(StrictModule):
    densitized_moments: Array
    matter_internal_energy: Array
    matter_momentum_covector: Array
    electron_fraction: Array
    time: Array
    accepted_steps: Array


class GRNeutrinoLeptonLedger(StrictModule):
    radiation_energy_change: Array
    matter_energy_change: Array
    radiation_momentum_change: Array
    matter_momentum_change: Array
    radiation_lepton_number_change: Array
    matter_lepton_number_change: Array
    energy_balance_residual: Array
    momentum_balance_residual: Array
    lepton_balance_residual: Array
    accepted: Array
    finite: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class GRNeutrinoM1StepResult(StrictModule):
    candidate: GRNeutrinoM1State
    state: GRNeutrinoM1State
    species_transport: tuple[GRMultigroupM1StepResult, ...]
    exchange: GRNeutrinoMatterExchange
    stress_energy: StressEnergyProjection
    ledger: GRNeutrinoLeptonLedger
    accepted: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class FixedGridGRNeutrinoM1Plan(StrictModule, NonTrainableState):
    """Multispecies transport followed by an atomic conservative lepton source step."""

    system: GRNeutrinoM1System
    species_transport: tuple[FixedGridGRMultigroupM1SSPRK3Plan, ...]
    interaction: GRNeutrinoInteractionPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: GRNeutrinoM1System,
        species_transport: tuple[FixedGridGRMultigroupM1SSPRK3Plan, ...],
        interaction: GRNeutrinoInteractionPlan,
        /,
    ) -> None:
        if not isinstance(system, GRNeutrinoM1System):
            raise TypeError("system must be GRNeutrinoM1System.")
        plans = tuple(species_transport)
        if len(plans) != system.species_count or any(
            not isinstance(value, FixedGridGRMultigroupM1SSPRK3Plan) for value in plans
        ):
            raise TypeError("One multigroup transport is required per neutrino species.")
        if not isinstance(interaction, GRNeutrinoInteractionPlan):
            raise TypeError("interaction must be GRNeutrinoInteractionPlan.")
        for species_system, transport in zip(system.species_systems, plans, strict=True):
            if transport.system.system_id != species_system.system_id:
                raise ValueError("Neutrino species and transport systems differ.")
        if interaction.neutrinos.system_id != system.system_id:
            raise ValueError("Neutrino transport and interaction systems differ.")
        first = plans[0]
        if any(
            value.groups[0].discretization.prepared_id
            != first.groups[0].discretization.prepared_id
            for value in plans[1:]
        ):
            raise ValueError("Neutrino species transports must share one grid.")
        self.system = system
        self.species_transport = plans
        self.interaction = interaction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-grid-gr-neutrino-m1",
                "system": system.system_id,
                "transport": [value.plan_id for value in plans],
                "interaction": interaction.plan_id,
            }
        )

    @property
    def cell_shape(self) -> tuple[int, ...]:
        return self.species_transport[0].cell_shape

    def initialize(
        self,
        moments: ArrayLike,
        matter_internal_energy: ArrayLike,
        electron_fraction: ArrayLike,
        geometry: ValenciaFiniteVolumeStageGeometry,
        /,
        *,
        matter_momentum_covector: ArrayLike | None = None,
        time: ArrayLike = 0.0,
    ) -> GRNeutrinoM1State:
        grouped = self.system.species_group_moments(moments)
        species_states = tuple(
            transport.initialize(
                species_system.flatten_groups(grouped[..., index, :, :]),
                geometry,
                time=time,
            )
            for index, (species_system, transport) in enumerate(
                zip(
                    self.system.species_systems,
                    self.species_transport,
                    strict=True,
                )
            )
        )
        material = jnp.asarray(matter_internal_energy, dtype=grouped.dtype)
        fraction = jnp.asarray(electron_fraction, dtype=grouped.dtype)
        if material.shape != self.cell_shape or fraction.shape != self.cell_shape:
            raise ValueError("Neutrino material fields must match the transport grid.")
        momentum = (
            jnp.zeros(self.cell_shape + (3,), dtype=grouped.dtype)
            if matter_momentum_covector is None
            else jnp.asarray(matter_momentum_covector, dtype=grouped.dtype)
        )
        if momentum.shape != self.cell_shape + (3,):
            raise ValueError("Matter momentum must end in three covector components.")
        physical = (
            jnp.all(jnp.isfinite(material))
            & jnp.all(material >= 0.0)
            & jnp.all(jnp.isfinite(momentum))
            & jnp.all(jnp.isfinite(fraction))
            & jnp.all((fraction >= 0.0) & (fraction <= 1.0))
        )
        material = eqx.error_if(
            material, ~physical, "Initial neutrino-coupled material state is invalid."
        )
        return GRNeutrinoM1State(
            jnp.stack(
                tuple(value.densitized_moments for value in species_states), axis=-3
            ),
            material,
            momentum,
            fraction,
            species_states[0].time,
            jnp.zeros((), dtype=jnp.int32),
        )

    def advance(
        self,
        state: GRNeutrinoM1State,
        start_time: ArrayLike,
        end_time: ArrayLike,
        stage_geometries: tuple[
            ValenciaFiniteVolumeStageGeometry,
            ValenciaFiniteVolumeStageGeometry,
            ValenciaFiniteVolumeStageGeometry,
        ],
        rest_mass_density: ArrayLike,
        fluid_velocity: ArrayLike,
        matter_temperature: ArrayLike,
        baryon_number_density: ArrayLike,
        /,
        *,
        magnetic_squared: ArrayLike = 0.0,
        transport_extinction: ArrayLike = 0.0,
    ) -> GRNeutrinoM1StepResult:
        if not isinstance(state, GRNeutrinoM1State):
            raise TypeError("state must be GRNeutrinoM1State.")
        expected = self.cell_shape + (
            self.system.species_count,
            self.system.group_count,
            4,
        )
        if state.densitized_moments.shape != expected:
            raise ValueError(f"Neutrino densitized moments must have shape {expected}.")
        start = jnp.asarray(start_time, dtype=state.time.dtype).reshape(())
        end = jnp.asarray(end_time, dtype=state.time.dtype).reshape(())
        step = end - start
        extinction = jnp.asarray(transport_extinction)
        target = self.cell_shape + (
            self.system.species_count,
            self.system.group_count,
        )
        if extinction.shape in (
            (),
            (self.system.group_count,),
            (self.system.species_count, self.system.group_count),
        ):
            extinction = jnp.broadcast_to(extinction, target)
        elif extinction.shape != target:
            raise ValueError("Neutrino transport extinction has invalid shape.")
        transported = []
        for index, transport in enumerate(self.species_transport):
            species_state = GRMultigroupM1State(
                state.densitized_moments[..., index, :, :],
                state.time,
                state.accepted_steps,
            )
            transported.append(
                transport.advance(
                    species_state,
                    start,
                    end,
                    stage_geometries,
                    transport_extinction=extinction[..., index, :],
                )
            )
        transport_results = tuple(transported)
        transported_densitized = jnp.stack(
            tuple(value.state.densitized_moments for value in transport_results),
            axis=-3,
        )
        final_geometry = stage_geometries[-1].cell
        determinant = final_geometry.sqrt_det_spatial_metric[..., None, None, None]
        local_before_source = transported_densitized / determinant
        exchange = self.interaction.matter_exchange(
            self.system.flatten_moments(local_before_source),
            rest_mass_density,
            fluid_velocity,
            matter_temperature,
            baryon_number_density,
            state.electron_fraction,
            final_geometry,
            magnetic_squared=magnetic_squared,
        )
        source = jnp.concatenate(
            (
                exchange.radiation_energy_source[..., None],
                exchange.radiation_flux_source,
            ),
            axis=-1,
        )
        local_candidate = local_before_source + step * source
        densitized_candidate = determinant * local_candidate
        matter_energy_candidate = (
            state.matter_internal_energy + step * exchange.matter_energy_source
        )
        matter_momentum_candidate = (
            state.matter_momentum_covector + step * exchange.matter_momentum_source
        )
        fraction_candidate = (
            state.electron_fraction + step * exchange.electron_fraction_source
        )
        closure = self.system.closure(
            self.system.flatten_moments(local_candidate), final_geometry
        )
        transport_accepted = jnp.all(
            jnp.stack(tuple(value.accepted for value in transport_results))
        )
        finite = (
            jnp.all(jnp.stack(tuple(value.finite for value in transport_results)))
            & jnp.all(exchange.finite)
            & jnp.all(closure.finite | ~final_geometry.active)
            & jnp.all(jnp.isfinite(matter_energy_candidate))
            & jnp.all(jnp.isfinite(matter_momentum_candidate))
            & jnp.all(jnp.isfinite(fraction_candidate))
        )
        physically_valid = (
            jnp.all(
                jnp.stack(tuple(value.physically_valid for value in transport_results))
            )
            & jnp.all(exchange.physically_valid)
            & jnp.all(closure.physically_valid | ~final_geometry.active)
            & jnp.all(matter_energy_candidate >= 0.0)
            & jnp.all((fraction_candidate >= 0.0) & (fraction_candidate <= 1.0))
        )
        qualified = (
            transport_accepted
            & finite
            & physically_valid
            & jnp.all(exchange.qualified)
            & jnp.all(closure.qualified | ~final_geometry.active)
        )
        derivative = (
            qualified
            & jnp.all(
                jnp.stack(tuple(value.derivative_valid for value in transport_results))
            )
            & jnp.all(exchange.derivative_valid)
            & jnp.all(closure.derivative_valid | ~final_geometry.active)
        )
        candidate = GRNeutrinoM1State(
            densitized_candidate,
            matter_energy_candidate,
            matter_momentum_candidate,
            fraction_candidate,
            end,
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
        )
        selected = lambda new, old: jnp.where(qualified, new, old)
        accepted_state = GRNeutrinoM1State(
            selected(candidate.densitized_moments, state.densitized_moments),
            selected(candidate.matter_internal_energy, state.matter_internal_energy),
            selected(candidate.matter_momentum_covector, state.matter_momentum_covector),
            selected(candidate.electron_fraction, state.electron_fraction),
            selected(candidate.time, state.time),
            selected(candidate.accepted_steps, state.accepted_steps),
        )
        radiation_energy_change = jnp.sum(
            local_candidate[..., 0] - local_before_source[..., 0], axis=(-2, -1)
        )
        matter_energy_change = matter_energy_candidate - state.matter_internal_energy
        light_speed = jnp.asarray(
            self.system.species_systems[0].groups[0].physical_light_speed,
            dtype=local_candidate.dtype,
        )
        radiation_momentum_change = jnp.sum(
            (local_candidate[..., 1:] - local_before_source[..., 1:]) / light_speed,
            axis=(-3, -2),
        )
        matter_momentum_change = (
            matter_momentum_candidate - state.matter_momentum_covector
        )
        radiation_lepton_change = self.system.radiation_lepton_number(
            self.system.flatten_moments(local_candidate)
        ) - self.system.radiation_lepton_number(
            self.system.flatten_moments(local_before_source)
        )
        baryon_density = jnp.asarray(baryon_number_density, dtype=local_candidate.dtype)
        matter_lepton_change = baryon_density * (
            fraction_candidate - state.electron_fraction
        )
        energy_residual = radiation_energy_change + matter_energy_change
        momentum_residual = radiation_momentum_change + matter_momentum_change
        lepton_residual = radiation_lepton_change + matter_lepton_change
        zero_if_rejected = lambda value: jnp.where(
            qualified, value, jnp.zeros_like(value)
        )
        ledger = GRNeutrinoLeptonLedger(
            zero_if_rejected(radiation_energy_change),
            zero_if_rejected(matter_energy_change),
            zero_if_rejected(radiation_momentum_change),
            zero_if_rejected(matter_momentum_change),
            zero_if_rejected(radiation_lepton_change),
            zero_if_rejected(matter_lepton_change),
            zero_if_rejected(energy_residual),
            zero_if_rejected(momentum_residual),
            zero_if_rejected(lepton_residual),
            qualified,
            finite,
            qualified,
            self.plan_id,
        )
        accepted_local = accepted_state.densitized_moments / determinant
        projection = self.system.stress_energy_projection(
            self.system.flatten_moments(accepted_local), final_geometry
        )
        return GRNeutrinoM1StepResult(
            candidate,
            accepted_state,
            transport_results,
            exchange,
            projection,
            ledger,
            qualified,
            finite,
            physically_valid,
            qualified,
            derivative,
            self.plan_id,
        )


__all__ = [
    "FixedGridGRNeutrinoM1Plan",
    "GRNeutrinoLeptonLedger",
    "GRNeutrinoM1State",
    "GRNeutrinoM1StepResult",
]
