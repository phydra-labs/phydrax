"""Vortex-sheet air on a volume-constrained quadrupole soap bubble."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.applications.foams import (
    FoamDynamicsPlan,
    FoamDynamicsState,
    FoamMaterialPlan,
    PreparedFoamDynamics,
    RegionPressureAirPlan,
    VortexSheetAirPlan,
)
from phydrax.geometry.multiregion_surface import PreparedMultiRegionSurface, seed_sphere


SIGMA = 0.025
AIR_DENSITY = 1.2
RADIUS = 0.0237
MODE = 2


def _mode_amplitude(positions: Array, /) -> float:
    radius = jnp.linalg.norm(positions, axis=1)
    cosine = jnp.where(radius > 0.0, positions[:, 2] / radius, 0.0)
    quadrupole = 0.5 * (3.0 * cosine * cosine - 1.0)
    mean_radius = jnp.mean(radius)
    return float(
        jnp.sum((radius / mean_radius - 1.0) * quadrupole) / jnp.sum(quadrupole**2)
    )


def run() -> dict[str, float | int | bool | str]:
    seed = seed_sphere(RADIUS, subdivisions=0)
    topology = seed.topology(seed.capacity_plan(resource_id="vortex-bubble-example"))
    unscaled = seed.state(topology)
    finite = jnp.asarray(topology.finite_region_indices, dtype=jnp.int32)
    unscaled_surface = PreparedMultiRegionSurface(topology, unscaled)
    unscaled_volume = unscaled_surface.region_volumes(unscaled.positions)[finite]
    physical_volume = 4.0 * jnp.pi * RADIUS**3 / 3.0
    base_scale = (physical_volume / unscaled_volume[0]) ** (1.0 / 3.0)
    base = unscaled.with_positions(base_scale * unscaled.positions)
    base_surface = PreparedMultiRegionSurface(topology, base)
    target = base_surface.region_volumes(base.positions)[finite]
    radius = jnp.linalg.norm(base.positions, axis=1)
    cosine = jnp.where(radius > 0.0, base.positions[:, 2] / radius, 0.0)
    quadrupole = 0.5 * (3.0 * cosine * cosine - 1.0)
    raw_positions = base.positions * (1.0 + 0.015 * quadrupole)[:, None]
    raw_volume = base_surface.region_volumes(raw_positions)[finite]
    perturbation_scale = (target[0] / raw_volume[0]) ** (1.0 / 3.0)
    positions = perturbation_scale * raw_positions
    surface_state = base.with_positions(positions)
    surface = PreparedMultiRegionSurface(topology, surface_state)
    reference_radius = (3.0 * target[0] / (4.0 * jnp.pi)) ** (1.0 / 3.0)
    dynamics_state = FoamDynamicsState(surface_state)
    dynamics = PreparedFoamDynamics(
        FoamDynamicsPlan(
            route="film-inertia",
            time_step=1.0e-4,
            # Constraint-projection metric only; vortex air owns the physical inertia.
            areal_mass=AIR_DENSITY * RADIUS,
            volume_tolerance=1.0e-15,
        ),
        surface,
        FoamMaterialPlan.soap_film(topology.region_ids, SIGMA),
        RegionPressureAirPlan.incompressible(target),
        dynamics_state,
    )
    prepared = VortexSheetAirPlan(
        air_density=AIR_DENSITY,
        time_step=1.0e-4,
        steps=1,
        core_radius_fraction=0.4,
        fmm_depth=2,
        fmm_leaf_capacity=8,
        maximum_fmm_relative_error=0.15,
    ).prepare(dynamics, dynamics_state)
    initial = prepared.initialize(dynamics_state)
    result = prepared.advance(initial)
    # Kornek equation (2): gamma_film=2 sigma and rho_i=rho_o=AIR_DENSITY.
    omega_squared = (
        2.0
        * SIGMA
        * (MODE - 1)
        * MODE
        * (MODE + 1)
        * (MODE + 2)
        / (AIR_DENSITY * reference_radius**3 * (2 * MODE + 1))
    )
    return {
        "status": int(result.evidence.status),
        "successful": result.successful,
        "vertices": topology.vertex_count,
        "elapsed_time_s": float(result.state.dynamics.time),
        "kornek_l2_frequency_hz": float(np.sqrt(omega_squared) / (2.0 * np.pi)),
        "kornek_reference_radius_m": float(reference_radius),
        "initial_mode_amplitude": _mode_amplitude(initial.surface.positions),
        "final_mode_amplitude": _mode_amplitude(result.state.surface.positions),
        "maximum_gauge_residual": float(result.evidence.maximum_gauge_residual),
        "fmm_relative_l2_error": float(result.evidence.fmm_relative_l2_error),
        "volume_residual": float(result.evidence.volume_residual),
        "core_policy": result.evidence.core_policy,
        "no_circulation_smoothing": result.evidence.no_circulation_smoothing,
    }


if __name__ == "__main__":
    print(run())
