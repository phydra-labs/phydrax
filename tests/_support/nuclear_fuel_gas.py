"""Explicit pellet-scale fission gas setups shared by the nuclear fuel gas tests.

The constants describe one LWR UO2 pellet site: an 8.2 mm diameter, 10 mm tall
pellet whose cracks triple the geometric surface, irradiated at a nominal fission
rate density with the stable Xe + Kr yield of 0.3 atoms per fission under a
compressive hydrostatic stress.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike

from phydrax.applications.nuclear_fuel import (
    AthermalRelease,
    FissionGasModel,
    FissionGasSiteState,
    FissionGasStepResult,
)


DAY = 86400.0  # [s]
FISSION_RATE = 1.0e19  # nominal fission rate density [fissions/m^3/s]
XE_KR_YIELD = 0.3  # stable Xe + Kr atoms per fission
PRODUCTION = XE_KR_YIELD * FISSION_RATE  # [atoms/m^3/s]
PELLET_RADIUS = 4.1e-3  # [m]
PELLET_HEIGHT = 10.0e-3  # [m]
CRACK_SURFACE_FACTOR = 3.0  # cracked over geometric pellet surface
NOMINAL_HYDROSTATIC_STRESS = -2.0e7  # compressive pellet stress [Pa]
GEOMETRIC_SURFACE_TO_VOLUME = 2.0 / PELLET_RADIUS + 2.0 / PELLET_HEIGHT  # [1/m]


def fission_gas_model(*, mode_count: int) -> FissionGasModel:
    """Literature-default gas model of the cracked reference pellet."""
    return FissionGasModel(
        athermal=AthermalRelease(
            geometric_surface_to_volume=GEOMETRIC_SURFACE_TO_VOLUME,
            total_surface_to_volume=CRACK_SURFACE_FACTOR * GEOMETRIC_SURFACE_TO_VOLUME,
        ),
        mode_count=mode_count,
    )


def advance_gas(
    model: FissionGasModel,
    state: FissionGasSiteState,
    *,
    temperature: ArrayLike,
    previous_temperature: ArrayLike,
    burnup: ArrayLike,
    time_step: ArrayLike,
    fission_rate: ArrayLike = FISSION_RATE,
    production: ArrayLike = PRODUCTION,
    stress: ArrayLike = NOMINAL_HYDROSTATIC_STRESS,
) -> FissionGasStepResult:
    """One public gas step with the model's own grain growth to the endpoint.

    ``burnup`` is the endpoint burnup [MWd/kgHM] and ``time_step`` is in seconds.
    """
    temperature_ = jnp.asarray(temperature)
    burnup_ = jnp.asarray(burnup)
    dt = jnp.asarray(time_step)
    radius = model.grain_growth.advance_radius(
        state.grain_radius, temperature_, dt, burnup_mwd_per_kghm=burnup_
    )
    return model.advance(
        state,
        temperature=temperature_,
        grain_radius=radius,
        fission_rate_density=jnp.asarray(fission_rate),
        gas_production_rate=jnp.asarray(production),
        burnup_mwd_per_kghm=burnup_,
        hydrostatic_stress=jnp.asarray(stress),
        time_step=dt,
        previous_temperature=jnp.asarray(previous_temperature),
    )
