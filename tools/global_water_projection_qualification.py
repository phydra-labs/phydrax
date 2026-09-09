# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Qualify closest-feasible global water projection against global contraction."""

from __future__ import annotations

import json

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.applications.atmosphere import (
    GlobalAtmosphereProcesses,
    GlobalPrimitiveEquationPlan,
    MoistThermodynamicPlan,
)
from phydrax.applications.geophysics import HybridPressureCoordinate
from phydrax.discretization import SphericalSpectralPlan


def _reconstruct_water_phases(model, water):
    coefficients = jnp.stack(water, axis=-1)
    flattened = coefficients.reshape(coefficients.shape[:2] + (-1,))
    return model.reconstruct(flattened).reshape(
        model.work_space.sample_shape + (model.levels, 3)
    )


def main():
    space = SphericalSpectralPlan(4, sampling="gl").prepare(radius=6.371e6)
    thermodynamics = MoistThermodynamicPlan()
    model = GlobalPrimitiveEquationPlan(
        space,
        HybridPressureCoordinate([0.1, 0.05, 0.0], [0.0, 0.5, 1.0]),
        dt=20.0,
        processes=GlobalAtmosphereProcesses(thermodynamics=thermodynamics),
        water_limiter="conservative",
        maximum_water_phase_repartition_fraction=1e-2,
    ).prepare()
    initial = model.initialize(temperature=280.0, vapor=0.005, liquid=1e-6)
    view = model.view(initial.state)
    theta = model.work_space.transform.theta[:, None, None]
    phi = model.work_space.transform.phi[None, :, None]
    signed_liquid = view.layer_mass * 1e-6 * (1.0 + 2.0 * jnp.sin(theta) * jnp.cos(phi))
    candidate = eqx.tree_at(
        lambda state: state.water,
        initial.state,
        (
            initial.state.water[0],
            model.project(signed_liquid),
            initial.state.water[2],
        ),
    )
    (
        limited,
        active,
        total_moved,
        total_relative,
        phase_moved,
        phase_relative,
        energy_residual,
        successful,
    ) = model.limit_water_inventories(candidate)
    original = _reconstruct_water_phases(model, candidate.water)
    constrained = _reconstruct_water_phases(model, limited.water)
    area = 4 * jnp.pi * model.plan.space.radius**2
    means = model.work_space.integral(original) / area
    minima = jnp.min(original, axis=(0, 1))
    factors = jnp.where(minima < 0, means / (means - minima), 1.0)
    contracted = means + factors * (original - means)
    contracted = contracted.at[..., 0].add(
        jnp.sum(original, axis=-1) - jnp.sum(contracted, axis=-1)
    )
    constrained_difference = constrained - original
    contraction_difference = contracted - original
    constrained_l2 = jnp.sqrt(
        jnp.sum(model.work_space.integral(constrained_difference**2))
    )
    contraction_l2 = jnp.sqrt(
        jnp.sum(model.work_space.integral(contraction_difference**2))
    )
    contraction_moved = 0.5 * jnp.sum(
        model.work_space.integral(jnp.abs(contraction_difference))
    )
    moved_ratio = phase_moved / contraction_moved
    l2_ratio = constrained_l2 / contraction_l2
    total_before = sum(candidate.water)
    total_after = sum(limited.water)
    total_field_error = jnp.max(
        jnp.abs(model.reconstruct(total_after) - model.reconstruct(total_before))
    )
    total_scale = jnp.maximum(jnp.max(jnp.abs(model.reconstruct(total_before))), 1.0)
    total_relative_error = total_field_error / total_scale
    phase_mean_change = model.work_space.integral(constrained - original) / area
    energy_w_per_m2 = jnp.abs(energy_residual) / area
    passed = (
        successful
        & active
        & model.admissible(limited)
        & (moved_ratio < 0.8)
        & (l2_ratio < 0.8)
        & (total_relative_error < 5e-14)
        & (total_relative < model.plan.water_projection_tolerance)
        & (energy_w_per_m2 < 1e-5)
    )
    report = {
        "method": "joint-phase-Dykstra-simplex-fixed-total-water",
        "iterations": model.plan.water_projection_iterations,
        "feasibility_tolerance": model.plan.water_projection_tolerance,
        "successful": bool(successful),
        "active": bool(active),
        "minimum_before_kg_m2": float(jnp.min(original)),
        "minimum_after_kg_m2": float(jnp.min(constrained)),
        "phase_repartition_mass_kg": float(phase_moved),
        "total_water_redistribution_mass_kg": float(total_moved),
        "former_phasewise_contraction_moved_mass_kg": float(contraction_moved),
        "phase_repartition_mass_ratio": float(moved_ratio),
        "weighted_l2_correction": float(constrained_l2),
        "former_phasewise_contraction_weighted_l2": float(contraction_l2),
        "weighted_l2_ratio": float(l2_ratio),
        "maximum_total_water_relative_error": float(total_relative_error),
        "phase_mean_change_kg_m2": np.asarray(phase_mean_change).tolist(),
        "energy_roundoff_j": float(energy_residual),
        "energy_roundoff_j_m2": float(energy_w_per_m2),
        "total_redistribution_fraction_of_atmospheric_water": float(total_relative),
        "phase_repartition_fraction_of_atmospheric_water": float(phase_relative),
        "passed": bool(passed),
        "claim_boundary": (
            "Closest-feasible joint phase composition at this represented field; "
            "total water is unchanged, but this is not monotone transport or a "
            "bound on cumulative climate bias."
        ),
    }
    print(json.dumps(report, indent=2))
    if not bool(passed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
