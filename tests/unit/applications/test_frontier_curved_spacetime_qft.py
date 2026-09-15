#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.curved_spacetime_qft import (
    adiabatic_hadamard_subtraction,
    adiabatic_initial_state,
    bogoliubov_particle_production,
    BogoliubovEvidence,
    evolve_flrw_modes,
    FLRWModePlan,
    ModeEvolutionEvidence,
    prepare_flrw_modes,
    prepare_semiclassical_einstein,
    semiclassical_einstein_backreaction,
    SemiclassicalEinsteinPlan,
)


jax.config.update("jax_enable_x64", True)


def _minkowski_preparation():
    times = jnp.linspace(0.0, 1.0, 41)
    plan = FLRWModePlan(
        times,
        jnp.ones_like(times),
        jnp.zeros_like(times),
        jnp.zeros_like(times),
        jnp.asarray((0.8, 1.3, 2.1)),
        jnp.asarray((0.2, 0.3, 0.4)),
        mass=0.4,
    )
    return prepare_flrw_modes(plan)


def test_frontier_bogoliubov_normalization_is_preserved_by_mode_evolution():
    prepared = _minkowski_preparation()
    initial = adiabatic_initial_state(prepared)
    evolution = jax.jit(evolve_flrw_modes)(prepared, initial)
    evidence: BogoliubovEvidence = bogoliubov_particle_production(
        prepared, evolution, tolerance=2e-12
    )
    assert bool(evolution.finite)
    assert float(evolution.maximum_wronskian_drift) < 2e-12
    assert bool(evidence.normalized)
    assert float(evidence.maximum_normalization_residual) < 2e-12
    assert jnp.all(evidence.occupation_numbers >= 0.0)
    assert "finite-mode" in evidence.claim


def test_frontier_hadamard_subtraction_conserves_minkowski_vacuum_stress():
    prepared = _minkowski_preparation()
    times = prepared.plan.conformal_times[:, None]
    frequency = prepared.frequencies
    modes = jnp.exp(-1.0j * times * frequency) / jnp.sqrt(2.0 * frequency)
    derivatives = -1.0j * frequency * modes
    wronskians = jnp.ones_like(modes.real)
    evolution = ModeEvolutionEvidence(
        modes=modes,
        derivatives=derivatives,
        wronskians=wronskians,
        initial_wronskian_residual=jnp.asarray(0.0),
        maximum_wronskian_drift=jnp.asarray(0.0),
        finite=jnp.asarray(True),
        prepared_id=prepared.prepared_id,
        claim="analytic-minkowski-reference",
    )
    stress = adiabatic_hadamard_subtraction(
        prepared, evolution, order=4, conservation_tolerance=1e-11
    )
    np.testing.assert_allclose(stress.renormalized_energy_density, 0.0, atol=2e-13)
    np.testing.assert_allclose(stress.renormalized_pressure, 0.0, atol=2e-13)
    assert bool(stress.conserved)
    assert float(stress.maximum_relative_continuity_residual) < 1e-11
    assert "no-covariant-renormalization" in stress.claim

    backreaction = semiclassical_einstein_backreaction(
        prepare_semiclassical_einstein(
            SemiclassicalEinsteinPlan(
                prepared.plan.conformal_times,
                initial_scale_factor=1.0,
                initial_hubble=0.0,
                newton_constant=1.0,
                residual_tolerance=1e-11,
            )
        ),
        stress,
    )
    np.testing.assert_allclose(backreaction.scale_factors, 1.0, atol=2e-13)
    np.testing.assert_allclose(backreaction.hubble_parameters, 0.0, atol=2e-13)
    assert bool(backreaction.self_consistent)
    assert "research-reference-only" in backreaction.claim
