#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from phydrax.applications.semiclassical_qed import (
    FiniteSpatialSpinorQEDPlan,
    FiniteSpatialSpinorQEDVectorField,
    HomogeneousSpinorQED3DPlan,
    HomogeneousSpinorQEDPlan,
    HomogeneousSpinorQEDState,
    HomogeneousSpinorQEDTangentState,
    HomogeneousSpinorQEDTangentVectorField,
    HomogeneousSpinorQEDVectorField,
    negative_energy_spinor_modes,
    negative_energy_spinor_modes_3d,
    RetardedVolterraResponsePlan,
    solve_homogeneous_spinor_qed,
    tangent_finite_difference_evidence,
)


def test_qft_spinor_vacuum_modes_are_normalized_and_subtracted_at_finite_cutoff():
    momenta = jnp.asarray((-2.0, -0.25, 0.0, 0.7, 3.0))
    modes = negative_energy_spinor_modes(
        momenta, mass=0.4, charge=0.8, vector_potential=0.3
    )
    plan = HomogeneousSpinorQEDPlan(
        momenta,
        jnp.asarray((0.1, 0.2, 0.3, 0.2, 0.1)),
        charge=0.8,
        mass=0.4,
    )

    np.testing.assert_allclose(jnp.sum(jnp.abs(modes) ** 2, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        plan.raw_current(modes), plan.adiabatic_current(0.3), atol=1e-12
    )
    np.testing.assert_allclose(plan.renormalized_current(0.3, modes), 0.0, atol=1e-12)


def test_qft_massless_spinor_modes_remain_finite_at_zero_kinetic_momentum():
    momenta = jnp.asarray((-1.0, 0.0, 1.0))
    modes = negative_energy_spinor_modes(momenta, mass=0.0)
    plan = HomogeneousSpinorQEDPlan(momenta, jnp.ones((3,)), charge=1.0, mass=0.0)

    assert bool(jnp.all(jnp.isfinite(modes)))
    np.testing.assert_allclose(jnp.sum(jnp.abs(modes) ** 2, axis=-1), 1.0)
    assert bool(jnp.isfinite(plan.adiabatic_current(0.0)))


def test_qft_homogeneous_backreaction_reports_energy_and_ward_evidence():
    plan = HomogeneousSpinorQEDPlan(
        [-1.0, -0.2, 0.4, 1.3],
        [0.2, 0.3, 0.3, 0.2],
        charge=0.25,
        mass=0.6,
        energy_tolerance=2e-6,
        ward_tolerance=2e-7,
    )
    prepared = plan.prepare(
        t0=0.0,
        t1=0.04,
        vector_potential=0.15,
        electric_field=0.2,
    )
    result = solve_homogeneous_spinor_qed(
        prepared,
        save_times=jnp.linspace(0.0, 0.04, 9),
        rtol=1e-9,
        atol=1e-11,
    )

    assert bool(result.solution.successful)
    assert float(result.evidence.maximum_mode_norm_residual) < 2e-7
    assert float(result.evidence.maximum_energy_balance_residual) < 2e-6
    assert float(result.evidence.maximum_ward_residual) < 2e-7
    assert bool(result.evidence.successful)


def test_qft_tangent_rhs_matches_central_finite_difference():
    plan = HomogeneousSpinorQEDPlan(
        [-0.8, 0.3, 1.1], [0.2, 0.5, 0.3], charge=0.4, mass=0.7
    )
    modes = negative_energy_spinor_modes(plan.momenta, mass=plan.mass)
    base = HomogeneousSpinorQEDState(0.2, -0.1, modes)
    mode_tangent = jnp.asarray(
        ((0.01 + 0.02j, -0.03j), (0.02, -0.01 + 0.01j), (-0.02j, 0.03))
    )
    tangent = HomogeneousSpinorQEDTangentState(base, 0.13, -0.07, mode_tangent)
    tangent_field = HomogeneousSpinorQEDTangentVectorField(
        plan, lambda time: jnp.asarray(0.0), "zero-perturbation"
    )
    tangent_rate = tangent_field(jnp.asarray(0.1), tangent, None)
    base_field = HomogeneousSpinorQEDVectorField(plan)
    epsilon = 2e-5
    plus = HomogeneousSpinorQEDState(
        base.vector_potential + epsilon * tangent.vector_potential_tangent,
        base.electric_field + epsilon * tangent.electric_field_tangent,
        base.mode_spinors + epsilon * tangent.mode_spinor_tangent,
    )
    minus = HomogeneousSpinorQEDState(
        base.vector_potential - epsilon * tangent.vector_potential_tangent,
        base.electric_field - epsilon * tangent.electric_field_tangent,
        base.mode_spinors - epsilon * tangent.mode_spinor_tangent,
    )
    plus_rate = base_field(jnp.asarray(0.1), plus, None)
    minus_rate = base_field(jnp.asarray(0.1), minus, None)

    for positive, negative, derivative in (
        (
            plus_rate.vector_potential,
            minus_rate.vector_potential,
            tangent_rate.vector_potential_tangent,
        ),
        (
            plus_rate.electric_field,
            minus_rate.electric_field,
            tangent_rate.electric_field_tangent,
        ),
        (
            plus_rate.mode_spinors,
            minus_rate.mode_spinors,
            tangent_rate.mode_spinor_tangent,
        ),
    ):
        evidence = tangent_finite_difference_evidence(
            positive,
            negative,
            derivative,
            epsilon,
            relative_tolerance=3e-3,
        )
        assert bool(evidence.successful)


def test_qft_retarded_volterra_response_has_no_future_support():
    times = jnp.asarray((0.0, 0.2, 0.5, 1.0))
    kernel = jnp.tril(jnp.ones((4, 4)))
    plan = RetardedVolterraResponsePlan(times, kernel)
    source = jnp.asarray((0.0, 0.0, 0.0, 2.0))
    result = plan.apply(source)

    np.testing.assert_allclose(result.response[:3], 0.0)
    assert bool(result.evidence.causal)
    assert bool(result.evidence.successful)


def test_qft_three_dimensional_negative_energy_modes_span_two_normalized_states():
    momenta = jnp.asarray(((0.0, 0.0, 0.0), (0.2, -0.4, 0.7)))
    modes = negative_energy_spinor_modes_3d(momenta, mass=0.0)
    gram = jnp.einsum("ksi,kti->kst", jnp.conj(modes), modes)
    plan = HomogeneousSpinorQED3DPlan(momenta, [0.4, 0.6], charge=0.3, mass=0.0)

    np.testing.assert_allclose(gram, jnp.broadcast_to(jnp.eye(2), gram.shape), atol=1e-10)
    np.testing.assert_allclose(
        plan.raw_current(modes), plan.adiabatic_current(jnp.zeros((3,))), atol=1e-10
    )


def test_qft_finite_spatial_modes_use_weighted_normalization_and_gauss_evidence():
    derivative = jnp.asarray(((0.0, 0.5, -0.5), (-0.5, 0.0, 0.5), (0.5, -0.5, 0.0)))
    plan = FiniteSpatialSpinorQEDPlan(
        derivative,
        jnp.ones((3,)),
        [0.0],
        [1.0],
        charge=0.3,
        mass=0.5,
    )
    modes = jnp.broadcast_to(jnp.asarray((0.0, 1.0), dtype=complex), (1, 3, 2))
    prepared = plan.prepare(
        modes,
        t0=0.0,
        t1=0.1,
        vector_potential=jnp.zeros((3,)),
        electric_field=jnp.zeros((3,)),
    )
    normalized = prepared.initial_state.mode_spinors
    norms = jnp.sum(
        plan.spatial_weights[None, :, None] * jnp.abs(normalized) ** 2,
        axis=(1, 2),
    )
    rate = FiniteSpatialSpinorQEDVectorField(plan)(
        jnp.asarray(0.0), prepared.initial_state, None
    )

    np.testing.assert_allclose(norms, 1.0, atol=1e-12)
    np.testing.assert_allclose(
        prepared.gauss_residual(prepared.initial_state, reference_charge=0.1),
        0.0,
        atol=1e-12,
    )
    assert bool(jnp.all(jnp.isfinite(rate.mode_spinors)))
