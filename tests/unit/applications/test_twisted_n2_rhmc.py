#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.supersymmetric_lattice import (
    assess_twisted_n2_chain,
    PfaffianControlPlan,
    prepare_twisted_n2_rhmc,
    sample_twisted_n2_rhmc,
    TwistedN2SYMPlan,
    TwistedSYMCoordinateLayout,
    WardIdentityPlan,
)
from phydrax.operators.path_integral import (
    evaluate_pseudofermion_action,
    pseudofermion_force,
    refresh_pseudofermion,
)
from phydrax.sampling import RHMCResourcePolicy


jax.config.update("jax_enable_x64", True)


def _fixture():
    theory = TwistedN2SYMPlan(
        (1, 1),
        matrix_rank=1,
        coupling=1.0,
        fermion_mass=1.0,
        coordinate_bound=0.25,
        temporal_axis=0,
        fermion_boundary_phase=-1.0,
    )
    layout = TwistedSYMCoordinateLayout(theory.prepare_bosonic())
    coordinates = jnp.zeros(layout.coordinate_shape, dtype=jnp.float64)
    prepared = prepare_twisted_n2_rhmc(
        theory,
        coordinates,
        step_size=1e-4,
        trajectory_steps=1,
        rational_poles=4,
        rational_verification_points=1025,
        bosonic_substeps=1,
        resources=RHMCResourcePolicy(
            maximum_terms=8,
            maximum_force_evaluations=64,
            maximum_retained_bytes=20_000_000,
            maximum_output_bytes=20_000_000,
            maximum_draws=4,
        ),
    )
    return prepared, coordinates


def test_twisted_phase_quenched_pseudofermion_force_matches_directional_difference():
    prepared, coordinates = _fixture()
    refresh = refresh_pseudofermion(
        prepared.pseudofermion, jax.random.key(3), links=coordinates
    )
    assert bool(refresh.successful)
    direction = jnp.linspace(-0.2, 0.2, coordinates.size).reshape(coordinates.shape)
    force = pseudofermion_force(prepared.pseudofermion, refresh.field, coordinates)
    epsilon = 1e-5
    forward = evaluate_pseudofermion_action(
        prepared.pseudofermion,
        refresh.field,
        role="force",
        links=coordinates + epsilon * direction,
    ).value
    backward = evaluate_pseudofermion_action(
        prepared.pseudofermion,
        refresh.field,
        role="force",
        links=coordinates - epsilon * direction,
    ).value
    finite_difference = (forward - backward) / (2.0 * epsilon)
    force_derivative = -jnp.vdot(force.force, direction).real
    np.testing.assert_allclose(force_derivative, finite_difference, rtol=3e-4, atol=3e-5)
    assert bool(force.successful)


def test_twisted_rhmc_runs_bounded_phase_quenched_transition():
    prepared, coordinates = _fixture()
    run = sample_twisted_n2_rhmc(
        prepared,
        coordinates,
        jax.random.key(17),
        num_draws=1,
    )
    assert run.samples.configurations.shape == (1,) + coordinates.shape
    assert bool(run.evidence.finite)
    assert int(run.evidence.nonfinite_count) == 0
    assert int(run.evidence.membership_failure_count) == 0
    assert prepared.phase_quenched_power == 0.25
    assert "no-continuum" in run.evidence.claim


def test_twisted_chain_retains_pfaffian_phase_and_overlap_evidence():
    prepared, coordinates = _fixture()
    run = sample_twisted_n2_rhmc(
        prepared,
        coordinates,
        jax.random.key(23),
        num_draws=1,
    )
    evidence = assess_twisted_n2_chain(
        prepared,
        run,
        WardIdentityPlan(
            float(run.samples.bosonic_action[0]),
            absolute_tolerance=1e-12,
            maximum_samples=1,
        ),
        PfaffianControlPlan(
            maximum_dimension=4,
            minimum_magnitude=0.0,
        ),
        minimum_phase_effective_samples=1.0,
        maximum_dense_elements=16,
    )
    assert evidence.pfaffian_phases.shape == (1,)
    np.testing.assert_allclose(evidence.phase_effective_samples, 1.0)
    assert bool(evidence.phase_overlap_sufficient)
    assert "no-continuum" in evidence.claim
