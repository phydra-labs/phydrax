#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Monolithic pressure-temperature backward Euler with exact shared liquid mass flux."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...discretization.finite_volume._diffusion_boundary import HybridDiffusionBoundary
from ...ein import contract
from ...nonlinear import implicit_root_result, NonlinearSystemProblem
from ._materials import _finite
from ._richards import _cell_array, _finish_root, RichardsPlan
from ._state import PorousStepResult, WaterHeatState
from ._thermal import PorousHeatFluxes, PorousThermalMaterial


class CoupledWaterHeatPlan(StrictModule):
    """One global root [cell p, face p, cell T, face T], all physical feedback live.

    Boundary thermal laws prescribe *conductive* W (Neumann), K (Dirichlet), or
    W/K and external K (Robin). Advected liquid enthalpy is added separately from
    the exact same mass rate as the water residual, never from a second Darcy
    evaluation. Inflow temperatures must be explicitly supplied, or a thermal
    Dirichlet boundary must supply them; unqualified exterior inflow fails.
    """

    water: RichardsPlan
    thermal: PorousThermalMaterial
    thermal_boundaries: HybridDiffusionBoundary
    dry_matrices: Array
    saturated_matrices: Array
    inflow_temperature_K: Array
    inflow_qualified: Array
    temperature_scale_K: float = eqx.field(static=True)
    energy_rate_scale_W: float = eqx.field(static=True)

    def __init__(
        self,
        water,
        thermal,
        thermal_boundaries,
        /,
        *,
        inflow_temperature_K=None,
        temperature_scale_K=300.0,
        energy_rate_scale_W=1000.0,
    ):
        if not isinstance(water, RichardsPlan) or not isinstance(
            thermal, PorousThermalMaterial
        ):
            raise TypeError(
                "Coupled water-heat requires RichardsPlan and PorousThermalMaterial."
            )
        if not isinstance(thermal_boundaries, HybridDiffusionBoundary):
            raise TypeError("Thermal boundaries must be HybridDiffusionBoundary.")
        if thermal_boundaries.geometry_id != water.discretization.geometry_id:
            raise ValueError("Thermal boundaries must share the water geometry.")
        self.water, self.thermal, self.thermal_boundaries = (
            water,
            thermal,
            thermal_boundaries,
        )
        diffusion = water.diffusion
        self.dry_matrices = diffusion.local_matrices(thermal.dry_conductivity_W_m_K)
        self.saturated_matrices = diffusion.local_matrices(
            thermal.saturated_conductivity_W_m_K
        )
        _cell_array(
            thermal.solid_heat_capacity_J_m3_K,
            diffusion.cell_count,
            "solid heat capacity",
        )
        if inflow_temperature_K is None:
            self.inflow_temperature_K = jnp.where(
                thermal_boundaries.kind == 1,
                thermal_boundaries.value,
                thermal.reference_temperature_K,
            )
            self.inflow_qualified = thermal_boundaries.kind == 1
        else:
            self.inflow_temperature_K = _cell_array(
                _finite(inflow_temperature_K, "inflow temperature", positive=True),
                diffusion.face_count,
                "inflow_temperature_K",
            )
            self.inflow_qualified = jnp.ones(diffusion.face_count, dtype=bool)
        for value in (temperature_scale_K, energy_rate_scale_W):
            if not float(value) > 0 or not jnp.isfinite(value):
                raise ValueError(
                    "Thermal nonlinear reference scales must be finite and positive."
                )
        self.temperature_scale_K, self.energy_rate_scale_W = (
            float(temperature_scale_K),
            float(energy_rate_scale_W),
        )

    @property
    def discretization(self):
        return self.water.discretization

    @property
    def unknown_scale(self):
        size = self.water.diffusion.cell_count + self.water.diffusion.face_count
        return jnp.concatenate(
            (
                jnp.full(size, self.water.pressure_scale_Pa),
                jnp.full(size, self.temperature_scale_K),
            )
        )

    def residual_scales(self):
        diffusion = self.water.diffusion
        face = jnp.where(
            self.thermal_boundaries.kind == 1,
            self.temperature_scale_K,
            self.energy_rate_scale_W,
        )
        return jnp.concatenate(
            (
                self.water.residual_scales(),
                jnp.full(diffusion.cell_count, self.energy_rate_scale_W),
                face,
            )
        )

    def state_from_unknown(self, unknown, *, time_s=0.0):
        count, faces = self.water.diffusion.cell_count, self.water.diffusion.face_count
        size = count + faces
        unknown = jnp.asarray(unknown)
        if unknown.shape != (2 * size,):
            raise ValueError(
                "Coupled unknown must contain cell p, face p, cell T, face T."
            )
        water = self.water.state_from_unknown(
            unknown[:size],
            time_s=time_s,
            temperature_K=unknown[size : size + count],
            face_temperature_K=unknown[size + count :],
        )
        return WaterHeatState(
            water, self.thermal.energy(water, self.discretization, self.water.material)
        )

    def initialize(
        self,
        pressure_Pa,
        temperature_K,
        *,
        face_pressure_Pa=None,
        face_temperature_K=None,
        time_s=0.0,
    ):
        temperature = _cell_array(
            _finite(temperature_K, "temperature", positive=True),
            self.water.diffusion.cell_count,
            "temperature_K",
        )
        face_temperature = self.thermal_boundaries.impose_dirichlet(
            self.water._face_temperature(temperature, face_temperature_K)
        )
        face_temperature = _finite(face_temperature, "face temperature", positive=True)
        water = self.water.initialize(
            pressure_Pa,
            face_pressure_Pa,
            time_s=time_s,
            temperature_K=temperature,
            face_temperature_K=face_temperature,
        )
        return WaterHeatState(
            water, self.thermal.energy(water, self.discretization, self.water.material)
        )

    def heat_fluxes(self, state: WaterHeatState, water_fluxes=None):
        diffusion = self.water.diffusion
        if water_fluxes is None:
            water_fluxes = self.water.fluxes(
                state.pressure_Pa,
                state.face_pressure_Pa,
                state.temperature_K,
                state.face_temperature_K,
            )
        saturation = self.water.retention.saturation(state.pressure_Pa)
        matrices = self.dry_matrices + saturation[:, None, None] * (
            self.saturated_matrices - self.dry_matrices
        )
        difference = (
            state.face_temperature_K[diffusion.cell_faces] - state.temperature_K[:, None]
        )
        conduction = -contract("cfg,cg->cf", matrices, difference)
        conduction = jnp.where(diffusion.valid, conduction, 0.0)
        face = diffusion.cell_faces
        owner, neighbour = (
            self.discretization.owner_cells,
            self.discretization.neighbour_cells,
        )
        cells = jnp.arange(diffusion.cell_count)[:, None]
        opposite = jnp.where(cells == owner[face], neighbour[face], owner[face])
        enthalpy = self.thermal.enthalpy(state.temperature_K)
        incoming = jnp.where(
            opposite >= 0,
            enthalpy[jnp.maximum(opposite, 0)],
            self.thermal.enthalpy(self.inflow_temperature_K)[face],
        )
        local_advection = water_fluxes.local_mass_rates * jnp.where(
            water_fluxes.local_mass_rates >= 0, enthalpy[:, None], incoming
        )
        conductive_face = diffusion.owner_rates(conduction)
        advective_face = diffusion.owner_rates(local_advection)
        return PorousHeatFluxes(
            conductive_face,
            advective_face,
            conductive_face + advective_face,
            conduction,
            local_advection,
        )

    def fluxes(self, unknown):
        state = self.state_from_unknown(unknown)
        water = self.water.fluxes(
            state.pressure_Pa,
            state.face_pressure_Pa,
            state.temperature_K,
            state.face_temperature_K,
        )
        return water, self.heat_fluxes(state, water)

    def residual(
        self, unknown, previous: WaterHeatState, dt_s, source_kg_s=0.0, source_W=0.0
    ):
        state = self.state_from_unknown(unknown)
        diffusion = self.water.diffusion
        water_flux = self.water.fluxes(
            state.pressure_Pa,
            state.face_pressure_Pa,
            state.temperature_K,
            state.face_temperature_K,
        )
        heat_flux = self.heat_fluxes(state, water_flux)
        mass_cells = (
            (state.water_mass_kg - previous.water_mass_kg) / dt_s
            + jnp.sum(water_flux.local_mass_rates, axis=1)
            - source_kg_s
        )
        mass_faces = self.water.boundaries.face_residual(
            state.face_pressure_Pa,
            diffusion.continuity_residual(water_flux.local_mass_rates),
        )
        heat_cells = (
            (state.energy_J - previous.energy_J) / dt_s
            + jnp.sum(
                heat_flux.local_conductive_rates_W + heat_flux.local_advective_rates_W,
                axis=1,
            )
            - source_W
        )
        heat_faces = self.thermal_boundaries.face_residual(
            state.face_temperature_K,
            diffusion.continuity_residual(heat_flux.local_conductive_rates_W),
        )
        return jnp.concatenate((mass_cells, mass_faces, heat_cells, heat_faces))

    def admissible(self, unknown):
        state = self.state_from_unknown(unknown)
        water_flux = self.water.fluxes(
            state.pressure_Pa,
            state.face_pressure_Pa,
            state.temperature_K,
            state.face_temperature_K,
        )
        exterior_inflow = (self.discretization.neighbour_cells < 0) & (
            water_flux.mass_face_rates < 0
        )
        return (
            self.water.admissible(state.pressure_Pa, state.temperature_K)
            & self.water.well_posed(state.pressure_Pa, state.temperature_K)
            & jnp.all(jnp.isfinite(state.energy_J))
            & jnp.all(
                jnp.isfinite(state.face_temperature_K) & (state.face_temperature_K > 0)
            )
            & jnp.all(~exterior_inflow | self.inflow_qualified)
        )

    def step(
        self,
        previous: WaterHeatState,
        dt_s,
        *,
        source_kg_s=0.0,
        source_W=0.0,
        initial_unknown=None,
    ):
        """Advance all water/heat unknowns in one implicitly differentiated root.

        ``source_W`` is total supplied sensible heat, including the enthalpy of
        volumetric mass sources when present; it is not silently inferred.
        """
        dt = _finite(dt_s, "time step", positive=True)
        if dt.shape != ():
            raise ValueError("dt_s must be scalar.")
        count = self.water.diffusion.cell_count
        mass_source = _cell_array(
            _finite(source_kg_s, "mass source"), count, "source_kg_s"
        )
        heat_source = _cell_array(_finite(source_W, "heat source"), count, "source_W")
        initial = (
            previous.unknown if initial_unknown is None else jnp.asarray(initial_unknown)
        )
        unknown_scale, residual_scale = self.unknown_scale, self.residual_scales()

        def residual(scaled, args):
            return (
                self.residual(
                    scaled * unknown_scale, previous, dt, mass_source, heat_source
                )
                / residual_scale
            )

        def valid(scaled, residual_value, auxiliary, args):
            return self.admissible(scaled * unknown_scale)

        root = implicit_root_result(
            NonlinearSystemProblem(
                residual,
                validity=valid,
                problem_id="monolithic-richards-heat-backward-euler",
            ),
            initial / unknown_scale,
            method=self.water.method,
            termination=self.water.termination,
            derivative_policy=self.water.derivative_policy,
        )
        successful = root.successful & self.admissible(root.state * unknown_scale)
        root = _finish_root(root, successful)
        unknown = root.state * unknown_scale
        candidate = self.state_from_unknown(unknown, time_s=previous.time_s + dt)
        state = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, previous
        )
        water_flux = self.water.fluxes(
            candidate.pressure_Pa,
            candidate.face_pressure_Pa,
            candidate.temperature_K,
            candidate.face_temperature_K,
        )
        return PorousStepResult(
            state,
            candidate,
            water_flux,
            root,
            self.residual(unknown, previous, dt, mass_source, heat_source),
            successful,
        )


__all__ = ["CoupledWaterHeatPlan"]
