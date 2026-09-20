#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Mass-conserving transient elastohydrodynamic lubrication."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract


@dataclass(frozen=True, slots=True)
class EHLState:
    fluid_content_m: Array
    pressure_pa: Array


@dataclass(frozen=True, slots=True)
class EHLStep:
    state: EHLState
    film_thickness_m: Array
    saturation: Array
    face_volume_flux_m2_s: Array
    mass_balance_residual_m2: Array
    complementarity_residual_pa: Array
    coupling_residual_pa: Array
    hydrodynamic_load_n_m: Array
    friction_force_n_m: Array
    coupling_iterations: int
    successful: Array


@dataclass(frozen=True, slots=True)
class MassConservingEHLSolver:
    """Finite-volume Elrod content transport with elastic film deformation.

    `fluid_content_m = h θ`. Full-film cells use a linear bulk-modulus
    compressibility; cavitated cells remain at cavitation pressure with θ < 1.
    The conservative variable is never reconstructed from pressure.
    """

    cell_widths_m: Array
    undeformed_gap_m: Array
    elastic_compliance_m_pa: Array
    viscosity_pa_s: float
    sliding_velocity_m_s: float
    cavitation_pressure_pa: float
    bulk_modulus_pa: float

    @classmethod
    def create(
        cls,
        cell_widths_m: ArrayLike,
        undeformed_gap_m: ArrayLike,
        elastic_compliance_m_pa: ArrayLike,
        viscosity_pa_s: float,
        sliding_velocity_m_s: float,
        /,
        *,
        cavitation_pressure_pa: float = 0.0,
        bulk_modulus_pa: float = 1.0e9,
        symmetry_tolerance: float = 1e-10,
    ) -> MassConservingEHLSolver:
        widths = np.asarray(cell_widths_m, dtype=float)
        gap = np.asarray(undeformed_gap_m, dtype=float)
        compliance = np.asarray(elastic_compliance_m_pa, dtype=float)
        if widths.ndim != 1 or widths.size < 2 or np.any(widths <= 0):
            raise ValueError("EHL cell widths must be a positive vector.")
        if gap.shape != widths.shape or np.any(gap <= 0):
            raise ValueError("EHL undeformed gap must be positive and cell aligned.")
        if compliance.shape != (widths.size, widths.size):
            raise ValueError("EHL elastic compliance has incompatible shape.")
        if not np.allclose(compliance, compliance.T, atol=symmetry_tolerance, rtol=0):
            raise ValueError("EHL elastic compliance must be symmetric.")
        if np.min(np.linalg.eigvalsh(compliance)) < -symmetry_tolerance:
            raise ValueError("EHL elastic compliance must be positive semidefinite.")
        if viscosity_pa_s <= 0 or bulk_modulus_pa <= 0:
            raise ValueError("EHL viscosity and bulk modulus must be positive.")
        return cls(
            jnp.asarray(widths),
            jnp.asarray(gap),
            jnp.asarray(compliance),
            float(viscosity_pa_s),
            float(sliding_velocity_m_s),
            float(cavitation_pressure_pa),
            float(bulk_modulus_pa),
        )

    def film_and_saturation(
        self,
        fluid_content_m: ArrayLike,
        pressure_guess_pa: ArrayLike,
        /,
        *,
        iterations: int = 64,
        relaxation: float = 0.5,
    ) -> tuple[Array, Array, Array, Array, Array]:
        content = jnp.asarray(fluid_content_m)
        pressure = jnp.asarray(pressure_guess_pa)
        if content.shape != self.cell_widths_m.shape or pressure.shape != content.shape:
            raise ValueError("EHL state does not match solver cells.")
        if iterations <= 0 or not 0 < relaxation <= 1:
            raise ValueError("EHL coupling iteration controls are invalid.")
        for _ in range(iterations):
            film = self.undeformed_gap_m + self.elastic_compliance_m_pa @ (
                pressure - self.cavitation_pressure_pa
            )
            if bool(jnp.any(film <= 0)):
                raise ValueError("EHL elastic deformation closed the lubricating film.")
            ratio = content / film
            target = self.cavitation_pressure_pa + self.bulk_modulus_pa * jnp.maximum(
                ratio - 1.0, 0
            )
            pressure = (1 - relaxation) * pressure + relaxation * target
        film = self.undeformed_gap_m + self.elastic_compliance_m_pa @ (
            pressure - self.cavitation_pressure_pa
        )
        saturation = jnp.minimum(content / film, 1.0)
        target = self.cavitation_pressure_pa + self.bulk_modulus_pa * jnp.maximum(
            content / film - 1.0, 0
        )
        coupling_residual = jnp.max(jnp.abs(pressure - target))
        complementarity = jnp.max(
            jnp.abs((pressure - self.cavitation_pressure_pa) * (1 - saturation))
        )
        return film, saturation, pressure, complementarity, coupling_residual

    def face_flux(
        self,
        pressure_pa: ArrayLike,
        film_thickness_m: ArrayLike,
        saturation: ArrayLike,
        /,
        *,
        inlet_flux_m2_s: float,
        outlet_flux_m2_s: float,
    ) -> Array:
        pressure = jnp.asarray(pressure_pa)
        film = jnp.asarray(film_thickness_m)
        theta = jnp.asarray(saturation)
        center_distance = 0.5 * (self.cell_widths_m[:-1] + self.cell_widths_m[1:])
        mobility = 0.5 * (film[:-1] ** 3 + film[1:] ** 3) / (12 * self.viscosity_pa_s)
        pressure_flux = -mobility * (pressure[1:] - pressure[:-1]) / center_distance
        couette_content = jnp.where(
            self.sliding_velocity_m_s >= 0,
            film[:-1] * theta[:-1],
            film[1:] * theta[1:],
        )
        interior = pressure_flux + 0.5 * self.sliding_velocity_m_s * couette_content
        return jnp.concatenate(
            (
                jnp.asarray((inlet_flux_m2_s,)),
                interior,
                jnp.asarray((outlet_flux_m2_s,)),
            )
        )

    def advance(
        self,
        state: EHLState,
        step_size_s: float,
        /,
        *,
        inlet_flux_m2_s: float = 0.0,
        outlet_flux_m2_s: float = 0.0,
        coupling_iterations: int = 64,
    ) -> EHLStep:
        if step_size_s <= 0:
            raise ValueError("EHL step size must be positive.")
        film, saturation, pressure, _, _ = self.film_and_saturation(
            state.fluid_content_m,
            state.pressure_pa,
            iterations=coupling_iterations,
        )
        flux = self.face_flux(
            pressure,
            film,
            saturation,
            inlet_flux_m2_s=inlet_flux_m2_s,
            outlet_flux_m2_s=outlet_flux_m2_s,
        )
        rate = -(flux[1:] - flux[:-1]) / self.cell_widths_m
        content = jnp.asarray(state.fluid_content_m) + float(step_size_s) * rate
        if bool(jnp.any(content < -1e-14)):
            raise ValueError("EHL transport CFL condition violated non-negative content.")
        content = jnp.maximum(content, 0)
        film, saturation, pressure, complementarity, coupling_residual = (
            self.film_and_saturation(
                content,
                pressure,
                iterations=coupling_iterations,
            )
        )
        initial_inventory = contract("q,q->", self.cell_widths_m, state.fluid_content_m)
        final_inventory = contract("q,q->", self.cell_widths_m, content)
        expected_change = float(step_size_s) * (inlet_flux_m2_s - outlet_flux_m2_s)
        balance = final_inventory - initial_inventory - expected_change
        gauge_pressure = pressure - self.cavitation_pressure_pa
        load = contract("q,q->", self.cell_widths_m, gauge_pressure)
        centers = jnp.cumsum(self.cell_widths_m) - 0.5 * self.cell_widths_m
        pressure_gradient = jnp.gradient(pressure, centers)
        friction_density = (
            self.viscosity_pa_s * abs(self.sliding_velocity_m_s) / film
            + 0.5 * jnp.abs(pressure_gradient) * film
        )
        friction = contract("q,q->", self.cell_widths_m, friction_density)
        successful = (
            jnp.all(jnp.isfinite(pressure))
            & jnp.all(content >= 0)
            & jnp.all((saturation >= 0) & (saturation <= 1))
        )
        return EHLStep(
            EHLState(content, pressure),
            film,
            saturation,
            flux,
            balance,
            complementarity,
            coupling_residual,
            load,
            friction,
            coupling_iterations,
            successful,
        )


__all__ = ["EHLState", "EHLStep", "MassConservingEHLSolver"]
