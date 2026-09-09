#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...nonlinear import (
    implicit_root_result,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._coupled import CoupledWaterHeatPlan
from ._richards import _finish_root
from ._state import WaterHeatState
from ._surface_coupled import _complementarity
from ._surface_exchange import OrthogonalDiffusiveWaveSurfacePlan, SurfaceWaterState


class SurfaceWaterHeatResult(StrictModule):
    state: WaterHeatState
    surface: SurfaceWaterState
    candidate: WaterHeatState
    candidate_surface: SurfaceWaterState
    dry_pressure_head: Array
    exchange_mass_rate: Array
    exchange_energy_rate: Array
    lateral_volume_rate: Array
    residual: Array
    combined_mass_balance: Array
    combined_energy_balance: Array
    root: NonlinearResult
    successful: Array
    derivative_available: Array


class SurfaceWaterHeatPlan(StrictModule):
    """Monolithic pressure-temperature surface/subsurface conservation.

    Wet cells impose pressure/head and temperature continuity. Dry cells retain
    the declared subsurface thermal boundary while their physically absent
    surface-temperature coordinate is fixed to the common enthalpy reference.
    A Fischer--Burmeister equation selects wet depth or nonnegative suction.
    The transition is semismooth; derivatives are admitted only away from it.

    Surface liquid is incompressible with the same constant density, heat
    capacity, and enthalpy reference as the porous liquid. Temperature may still
    alter viscosity, so pressure and heat remain one coupled nonlinear root.
    """

    coupled: CoupledWaterHeatPlan
    surface: OrthogonalDiffusiveWaveSurfacePlan
    pressure_per_head: Array
    interface_faces: Array
    external_faces: Array
    unknown_scale: Array
    termination: NonlinearTermination
    active_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        coupled: CoupledWaterHeatPlan,
        surface: OrthogonalDiffusiveWaveSurfacePlan,
        /,
        *,
        gravity_m_s2: float = 9.80665,
        active_tolerance: float = 1.0e-8,
        termination: NonlinearTermination | None = None,
    ):
        if not isinstance(coupled, CoupledWaterHeatPlan) or not isinstance(
            surface, OrthogonalDiffusiveWaveSurfacePlan
        ):
            raise TypeError(
                "Surface water-heat coupling requires prepared coupled and surface plans."
            )
        surface.trace.require_geometry(coupled.discretization)
        if not np.isfinite(gravity_m_s2) or gravity_m_s2 <= 0:
            raise ValueError("Surface gravity magnitude must be positive and finite.")
        if not np.isfinite(active_tolerance) or active_tolerance <= 0:
            raise ValueError("Wet/dry active tolerance must be positive and finite.")
        material = coupled.water.material
        density = np.asarray(material.density_kg_m3)
        compressibility = np.asarray(material.fluid_compressibility_Pa_inverse)
        expansion = np.asarray(material.thermal_expansion_K_inverse)
        heat_capacity = np.asarray(coupled.thermal.liquid_heat_capacity_J_kg_K)
        reference = np.asarray(coupled.thermal.reference_temperature_K)
        if (
            density.ndim > 1
            or not np.allclose(density, surface.density)
            or np.any(compressibility != 0)
            or np.any(expansion != 0)
            or heat_capacity.shape != ()
            or not np.allclose(heat_capacity, surface.heat_capacity)
            or reference.shape != ()
            or not np.allclose(reference, surface.reference_temperature)
        ):
            raise ValueError(
                "Surface storage requires matching constant liquid density, heat capacity, and enthalpy reference."
            )
        interfaces = np.asarray(surface.trace.parent_faces, dtype=np.int32)
        boundary = np.flatnonzero(np.asarray(coupled.discretization.neighbour_cells) < 0)
        external = np.setdiff1d(boundary, interfaces, assume_unique=False)
        ns = interfaces.size
        self.coupled = coupled
        self.surface = surface
        self.pressure_per_head = surface.density * gravity_m_s2
        self.interface_faces = jnp.asarray(interfaces)
        self.external_faces = jnp.asarray(external)
        self.unknown_scale = jnp.concatenate(
            (
                coupled.unknown_scale,
                jnp.ones(ns),
                jnp.ones(ns),
                jnp.full(ns, coupled.temperature_scale_K),
            )
        )
        self.termination = (
            coupled.water.termination if termination is None else termination
        )
        self.active_tolerance = float(active_tolerance)

    def _dynamic_coupled(self, surface_temperature: Array) -> CoupledWaterHeatPlan:
        inflow = self.coupled.inflow_temperature_K.at[self.interface_faces].set(
            surface_temperature
        )
        qualified = self.coupled.inflow_qualified.at[self.interface_faces].set(True)
        result = eqx.tree_at(lambda plan: plan.inflow_temperature_K, self.coupled, inflow)
        return eqx.tree_at(lambda plan: plan.inflow_qualified, result, qualified)

    def _surface_temperature(self, state: SurfaceWaterState) -> Array:
        return self.surface.temperature(state)

    def step(
        self,
        previous: WaterHeatState,
        surface_state: SurfaceWaterState,
        dt_s: ArrayLike,
        /,
        *,
        rainfall_m_s: ArrayLike = 0.0,
        rain_temperature_K: ArrayLike = 293.15,
        source_kg_s: ArrayLike = 0.0,
        source_W: ArrayLike = 0.0,
    ) -> SurfaceWaterHeatResult:
        dt = jnp.asarray(dt_s)
        if dt.ndim != 0:
            raise ValueError("Surface water-heat time step must be scalar.")
        dt = eqx.error_if(
            dt, ~jnp.isfinite(dt) | (dt <= 0), "Time step must be positive and finite."
        )
        nc = self.coupled.water.diffusion.cell_count
        nf = self.coupled.water.diffusion.face_count
        base_size = nc + nf
        ns = self.interface_faces.size
        if surface_state.volume.shape != (ns,) or surface_state.energy.shape != (ns,):
            raise ValueError("Surface state does not match the coupled interface trace.")
        head0 = surface_state.volume / self.surface.projected_areas
        dry0 = jnp.maximum(
            -previous.face_pressure_Pa[self.interface_faces] / self.pressure_per_head, 0.0
        )
        temperature0 = self._surface_temperature(surface_state)
        initial = jnp.concatenate((previous.unknown, head0, dry0, temperature0))
        rain = self.surface._vector(rainfall_m_s, "Rainfall")
        rain_temperature = self.surface._vector(rain_temperature_K, "Rain temperature")
        initial = eqx.error_if(
            initial,
            jnp.any(~jnp.isfinite(initial))
            | jnp.any(head0 < 0)
            | jnp.any(rain < 0)
            | jnp.any(rain_temperature <= 0),
            "Surface water-heat initial state and forcing must be physically admissible.",
        )
        rain_rate = rain * self.surface.projected_areas
        rho = self.surface.density
        cp = self.surface.heat_capacity
        reference_temperature = self.surface.reference_temperature
        rain_energy_rate = (
            rho * cp * rain_rate * (rain_temperature - reference_temperature)
        )
        mass_scale = self.coupled.water.mass_rate_scale_kg_s
        energy_scale = self.coupled.energy_rate_scale_W
        temperature_scale = self.coupled.temperature_scale_K
        head_scale = jnp.asarray(1.0)

        def physical_residual(unknown):
            base = unknown[: 2 * base_size]
            head = unknown[2 * base_size : 2 * base_size + ns]
            dry = unknown[2 * base_size + ns : 2 * base_size + 2 * ns]
            surface_temperature = unknown[2 * base_size + 2 * ns :]
            dynamic = self._dynamic_coupled(surface_temperature)
            raw = dynamic.residual(base, previous, dt, source_kg_s, source_W)
            scaled = raw / dynamic.residual_scales()
            pressure = base[nc : nc + nf]
            pressure_relation = (
                pressure[self.interface_faces] / self.pressure_per_head - head + dry
            ) / head_scale
            scaled = scaled.at[nc + self.interface_faces].set(pressure_relation)
            state = dynamic.state_from_unknown(base)
            water_flux, heat_flux = dynamic.fluxes(base)
            volume = self.surface.projected_areas * head
            energy = rho * cp * volume * (surface_temperature - reference_temperature)
            lateral = self.surface.lateral_rates(volume)
            donor = jnp.where(
                lateral >= 0, self.surface.edge_owner, self.surface.edge_neighbour
            )
            lateral_energy = (
                rho * cp * lateral * (surface_temperature[donor] - reference_temperature)
            )
            mass_balance = (
                rho * (volume - surface_state.volume) / dt
                + rho * self.surface.divergence(lateral)
                - rho * rain_rate
                - water_flux.mass_face_rates[self.interface_faces]
            )
            energy_balance = (
                (energy - surface_state.energy) / dt
                + self.surface.divergence(lateral_energy)
                - rain_energy_rate
                - heat_flux.total_face_rates_W[self.interface_faces]
            )
            wet_weight = head
            dry_weight = dry
            denominator = wet_weight + dry_weight
            at_transition = denominator <= self.active_tolerance
            denominator = jnp.where(at_transition, 1.0, denominator)
            original_heat_face = scaled[base_size + nc + self.interface_faces]
            heat_interface = (
                wet_weight
                * (state.face_temperature_K[self.interface_faces] - surface_temperature)
                / temperature_scale
                + dry_weight * original_heat_face
            ) / denominator
            heat_interface = jnp.where(at_transition, original_heat_face, heat_interface)
            surface_energy = (
                wet_weight * energy_balance / energy_scale
                + dry_weight
                * (surface_temperature - reference_temperature)
                / temperature_scale
            ) / denominator
            surface_energy = jnp.where(
                at_transition,
                (surface_temperature - reference_temperature) / temperature_scale,
                surface_energy,
            )
            scaled = scaled.at[base_size + nc + self.interface_faces].set(heat_interface)
            return jnp.concatenate(
                (
                    scaled,
                    mass_balance / mass_scale,
                    surface_energy,
                    _complementarity(head, dry) / head_scale,
                )
            )

        def scaled_residual(scaled, args):
            del args
            return physical_residual(scaled * self.unknown_scale)

        def valid(scaled, residual, auxiliary, args):
            del residual, auxiliary, args
            unknown = scaled * self.unknown_scale
            base = unknown[: 2 * base_size]
            head = unknown[2 * base_size : 2 * base_size + ns]
            dry = unknown[2 * base_size + ns : 2 * base_size + 2 * ns]
            temperature = unknown[2 * base_size + 2 * ns :]
            dynamic = self._dynamic_coupled(temperature)
            return (
                dynamic.admissible(base)
                & jnp.all(jnp.isfinite(head) & (head >= 0))
                & jnp.all(jnp.isfinite(dry) & (dry >= 0))
                & jnp.all(jnp.isfinite(temperature) & (temperature > 0))
            )

        root = implicit_root_result(
            NonlinearSystemProblem(
                scaled_residual,
                validity=valid,
                problem_id="surface-water-heat-complementarity",
            ),
            initial / self.unknown_scale,
            method=self.coupled.water.method,
            termination=self.termination,
            derivative_policy=self.coupled.water.derivative_policy,
        )
        successful = root.successful & valid(root.state, root.residual, None, None)
        root = _finish_root(root, successful)
        unknown = root.state * self.unknown_scale
        base = unknown[: 2 * base_size]
        head = unknown[2 * base_size : 2 * base_size + ns]
        dry = unknown[2 * base_size + ns : 2 * base_size + 2 * ns]
        surface_temperature = unknown[2 * base_size + 2 * ns :]
        dynamic = self._dynamic_coupled(surface_temperature)
        candidate = dynamic.state_from_unknown(base, time_s=previous.time_s + dt)
        volume = self.surface.projected_areas * head
        energy = rho * cp * volume * (surface_temperature - reference_temperature)
        candidate_surface = SurfaceWaterState(volume, energy)
        water_flux, heat_flux = dynamic.fluxes(base)
        lateral = self.surface.lateral_rates(volume)
        wet = head > self.active_tolerance
        mass_balance = (
            jnp.sum(candidate.water_mass_kg - previous.water_mass_kg)
            + rho * jnp.sum(volume - surface_state.volume)
            + dt
            * (
                jnp.sum(water_flux.mass_face_rates[self.external_faces])
                - rho * jnp.sum(rain_rate)
                - jnp.sum(jnp.broadcast_to(jnp.asarray(source_kg_s), (nc,)))
            )
        )
        energy_balance = (
            jnp.sum(candidate.energy_J - previous.energy_J)
            + jnp.sum(energy - surface_state.energy)
            + dt
            * (
                jnp.sum(heat_flux.total_face_rates_W[self.external_faces])
                + jnp.sum(
                    jnp.where(
                        wet,
                        0.0,
                        heat_flux.total_face_rates_W[self.interface_faces],
                    )
                )
                - jnp.sum(rain_energy_rate)
                - jnp.sum(jnp.broadcast_to(jnp.asarray(source_W), (nc,)))
            )
        )
        committed = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, previous
        )
        committed_surface = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old),
            candidate_surface,
            surface_state,
        )
        derivative_available = successful & jnp.all(head + dry > self.active_tolerance)
        return SurfaceWaterHeatResult(
            committed,
            committed_surface,
            candidate,
            candidate_surface,
            dry,
            water_flux.mass_face_rates[self.interface_faces],
            heat_flux.total_face_rates_W[self.interface_faces],
            lateral,
            physical_residual(unknown),
            mass_balance,
            energy_balance,
            root,
            successful,
            derivative_available,
        )


__all__ = ["SurfaceWaterHeatPlan", "SurfaceWaterHeatResult"]
