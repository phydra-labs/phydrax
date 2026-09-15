#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.supersymmetric_lattice import (
    assess_complexified_gauge_invariance,
    assess_ward_pfaffian,
    BFSSPlan,
    ComplexifiedPFormField,
    finite_pfaffian,
    PfaffianControlPlan,
    PFormLatticePlan,
    prepare_bfss,
    prepare_twisted_sym,
    transform_bfss_configuration,
    transform_twisted_configuration,
    TwistedSYMPlan,
    WardIdentityPlan,
)


jax.config.update("jax_enable_x64", True)


def _site_gauge(shape):
    diagonal = np.empty(shape + (2, 2), dtype=np.complex128)
    inverse = np.empty_like(diagonal)
    for site in np.ndindex(shape):
        first = 1.1 + 0.07 * sum(site)
        second = 0.8 + 0.05j * (1 + sum(site))
        diagonal[site] = np.diag((first, second))
        inverse[site] = np.diag((1.0 / first, 1.0 / second))
    return jnp.asarray(diagonal), jnp.asarray(inverse)


def test_frontier_p_form_placement_and_complexified_gauge_invariance():
    two_form = PFormLatticePlan((2, 3, 4), 2)
    values = jnp.zeros(two_form.configuration_shape(2), dtype=jnp.complex128)
    field = ComplexifiedPFormField(two_form, values)
    assert field.plan.orientations == ((0, 1), (0, 2), (1, 2))
    np.testing.assert_array_equal(
        field.plan.endpoint_offsets,
        np.asarray(((1, 1, 0), (1, 0, 1), (0, 1, 1))),
    )

    prepared = prepare_twisted_sym(TwistedSYMPlan((2, 2), matrix_rank=2, coupling=1.7))
    key = jax.random.key(17)
    real, imaginary = jax.random.split(key)
    links = 0.4 * jax.random.normal(
        real, prepared.link_plan.configuration_shape(2)
    ) + 0.3j * jax.random.normal(imaginary, prepared.link_plan.configuration_shape(2))
    reverse = jnp.swapaxes(jnp.conj(links), -1, -2)
    configuration = prepared.configuration(links, reverse)
    np.testing.assert_allclose(
        jax.jit(lambda action, value: action.action(value))(prepared, configuration),
        prepared.action(configuration),
        rtol=2e-12,
        atol=2e-12,
    )
    gauge, gauge_inverse = _site_gauge((2, 2))
    transformed = transform_twisted_configuration(
        configuration, gauge, inverse_gauge=gauge_inverse
    )
    np.testing.assert_allclose(
        prepared.action(transformed),
        prepared.action(configuration),
        rtol=2e-11,
        atol=2e-11,
    )
    gauge_evidence = assess_complexified_gauge_invariance(
        prepared,
        configuration,
        gauge,
        inverse_gauge=gauge_inverse,
        tolerance=2e-11,
    )
    assert bool(gauge_evidence.invariant)
    assert "reference-only" in gauge_evidence.claim


def test_frontier_bfss_action_and_ward_pfaffian_controls_are_explicit():
    prepared = prepare_bfss(BFSSPlan(4, 3, 2, coupling=1.2, time_spacing=0.1, mass=0.2))
    key = jax.random.key(91)
    keys = jax.random.split(key, 4)
    matrices = jax.random.normal(keys[0], (4, 3, 2, 2)).astype(jnp.complex128)
    matrices = matrices + 0.2j * jax.random.normal(keys[1], matrices.shape)
    dual = jnp.swapaxes(jnp.conj(matrices), -1, -2)
    links = jnp.broadcast_to(jnp.eye(2, dtype=jnp.complex128), (4, 2, 2))
    reverse = links
    configuration = prepared.configuration(matrices, dual, links, reverse)
    gauge, gauge_inverse = _site_gauge((4,))
    transformed = transform_bfss_configuration(
        configuration, gauge, inverse_gauge=gauge_inverse
    )
    np.testing.assert_allclose(
        prepared.action(transformed),
        prepared.action(configuration),
        rtol=2e-11,
        atol=2e-11,
    )

    fermion = jnp.asarray(
        (
            (0.0, 2.0, -1.0, 0.5),
            (-2.0, 0.0, 3.0, 1.5),
            (1.0, -3.0, 0.0, 4.0),
            (-0.5, -1.5, -4.0, 0.0),
        ),
        dtype=jnp.complex128,
    )
    pfaffian_plan = PfaffianControlPlan(
        maximum_dimension=6, maximum_phase_magnitude=1e-12
    )
    expected_pfaffian = 2.0 * 4.0 - (-1.0) * 1.5 + 0.5 * 3.0
    np.testing.assert_allclose(finite_pfaffian(fermion, pfaffian_plan), expected_pfaffian)
    expected_action = float(prepared.action(configuration))
    evidence = assess_ward_pfaffian(
        prepared,
        (configuration, configuration),
        (fermion, fermion),
        WardIdentityPlan(expected_action, absolute_tolerance=1e-12),
        pfaffian_plan,
    )
    assert bool(evidence.ward_satisfied)
    assert bool(evidence.pfaffians_accepted)
    assert bool(evidence.accepted)
    assert "research-only" in evidence.claim
