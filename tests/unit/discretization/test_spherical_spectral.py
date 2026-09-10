#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import sph_harm_y

import phydrax as phx
from phydrax.discretization import spectral as spectral_api


def _harmonic(space, degree=2, order=1):
    theta, phi = np.meshgrid(
        np.asarray(space.transform.theta),
        np.asarray(space.transform.phi),
        indexing="ij",
    )
    return jnp.asarray(np.real(sph_harm_y(degree, order, theta, phi)))


def _scipy_harmonic_on_vectors(degree, order, directions):
    vectors = np.asarray(directions, dtype=float)
    unit = vectors / np.linalg.norm(vectors, axis=-1)[..., None]
    theta = np.arccos(np.clip(unit[..., 2], -1.0, 1.0))
    phi = np.arctan2(unit[..., 1], unit[..., 0])
    return sph_harm_y(degree, order, theta, phi)


def test_spherical_mode_layout_tracks_valid_storage_and_real_conjugacy():
    layout = phx.discretization.SphericalModeLayout(5)
    assert layout.coefficient_shape == (5, 9)
    assert layout.logical_mode_count == 25
    assert layout.level_multiplicities == (1, 3, 5, 7, 9)
    assert int(jnp.sum(layout.valid_mask)) == 25
    assert int(jnp.sum(layout.independent_mask)) == 15
    assert jnp.array_equal(
        layout.conjugate_indices[layout.conjugate_indices],
        jnp.arange(9),
    )

    coefficients = jnp.zeros(layout.coefficient_shape, dtype=complex)
    coefficients = coefficients.at[3, 5].set(0.4 - 0.7j)
    canonical = layout.canonicalize_reality(coefficients)
    assert jnp.allclose(canonical[3, 3], -(0.4 + 0.7j))
    assert jnp.allclose(layout.conjugacy_defect(canonical), 0.0)

    contaminated = canonical.at[0, 0].set(jnp.nan + 1j * jnp.inf)
    masked = layout.mask_invalid(contaminated)
    assert jnp.all(jnp.isfinite(masked))
    assert jnp.allclose(masked, canonical)


def test_spherical_discretization_roundtrips_integrates_and_applies_laplacian():
    radius = 1.7
    space = phx.discretization.SphericalSpectralPlan(5).prepare(radius=radius)
    values = _harmonic(space, degree=2, order=1)
    coefficients = space.project(values)
    reconstructed = space.reconstruct(coefficients)
    laplacian = space.laplacian(values)

    assert space.state_shape == space.transform.sample_shape
    assert space.layout.logical_mode_count == 25
    assert space.physical_space.representation == "point_value"
    assert jnp.allclose(
        jnp.linalg.norm(space.points, axis=-1), radius, rtol=1e-12, atol=1e-12
    )
    assert jnp.allclose(
        jnp.sum(space.quadrature_weights),
        4.0 * jnp.pi * radius**2,
        rtol=1e-12,
        atol=1e-12,
    )
    assert jnp.allclose(reconstructed, values, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(
        laplacian,
        -6.0 / radius**2 * values,
        rtol=2e-10,
        atol=2e-10,
    )
    assert jnp.allclose(
        space.integral(jnp.ones(space.state_shape)), 4 * jnp.pi * radius**2
    )
    assert float(space.conjugacy_defect(coefficients)) < 1e-12
    assert dict(space.preparation.resource_counts)["dense_transform_entries"] == 0


def test_spherical_invalid_coefficient_capacity_is_numerically_inert():
    space = phx.discretization.SphericalSpectralPlan(4).prepare()
    values = _harmonic(space, degree=2, order=1)
    coefficients = space.project(values)
    contaminated = coefficients.at[0, 0].set(jnp.nan + 1j * jnp.inf)

    actual = space.reconstruct(contaminated)
    expected = space.reconstruct(coefficients)

    assert jnp.all(jnp.isfinite(actual))
    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert not jnp.isfinite(space.invalid_storage_defect(contaminated))


def test_spherical_real_eigenpairs_are_complete_degree_weighted_modes():
    space = phx.discretization.SphericalSpectralPlan(4).prepare(radius=2.0)
    eigenvalues, modes = space.eigenpairs(rank=4)
    flattened = modes.reshape((-1, 4))
    weights = space.quadrature_weights.reshape((-1,))
    gram = flattened.T @ (weights[:, None] * flattened)

    assert jnp.allclose(eigenvalues, jnp.asarray([0.0, 0.5, 0.5, 0.5]))
    assert jnp.allclose(gram, jnp.eye(4), rtol=1e-10, atol=1e-10)
    assert space.eigenmode_ids(rank=4) == (
        "sphere-real:ell:0:m:0",
        "sphere-real:ell:1:m:0",
        "sphere-real:ell:1:m:1:cos",
        "sphere-real:ell:1:m:1:sin",
    )
    with pytest.raises(ValueError, match="complete-degree square"):
        space.eigenpairs(rank=2)


def test_spherical_laplacian_operator_is_pairing_self_adjoint():
    space = phx.discretization.SphericalSpectralPlan(4).prepare()
    operator = phx.discretization.spherical_laplacian_operator(space)
    left = _harmonic(space, degree=1, order=0)
    right = _harmonic(space, degree=2, order=1)
    pairing = space.physical_space.vector_space

    assert operator.properties.self_adjoint
    assert jnp.allclose(
        pairing.inner(left, operator.mv(right)),
        pairing.inner(operator.mv(left), right),
        rtol=1e-10,
        atol=1e-10,
    )
    assert jnp.real(pairing.inner(left, operator.mv(left))) <= 1e-12


def test_spherical_discretization_is_jittable_and_rejects_unsupported_contracts():
    space = phx.discretization.SphericalSpectralPlan(4).prepare()
    values = _harmonic(space, degree=2, order=1)
    actual = eqx.filter_jit(lambda prepared, field: prepared.laplacian(field))(
        space, values
    )
    gradient = jax.grad(lambda field: jnp.sum(space.laplacian(field) ** 2))(values)

    assert jnp.all(jnp.isfinite(actual))
    assert jnp.all(jnp.isfinite(gradient))
    with pytest.raises(NotImplementedError, match="coordinate derivative frame"):
        space.partial_derivative(values, axis=0)
    with pytest.raises(ValueError, match="both intrinsic axes"):
        space.laplacian(values, axes=(0,))
    with pytest.raises(ValueError, match="finite and positive"):
        phx.discretization.SphericalSpectralPlan(4).prepare(radius=0.0)

    complex_precision = phx.discretization.SpectralPrecisionPolicy(jnp.complex128)
    spin_space = phx.discretization.SphericalSpectralPlan(
        4,
        spin=1,
        reality=False,
        precision=complex_precision,
    ).prepare()
    with pytest.raises(ValueError, match="require spin zero"):
        spin_space.negative_laplacian_levels()
    with pytest.raises(ValueError, match="real spin-zero"):
        spin_space.eigenpairs(rank=4)


def test_spherical_modal_integral_spin_ladders_and_rotation():
    radius = 1.3
    space = phx.discretization.SphericalSpectralPlan(4).prepare(radius=radius)
    coefficients = space.project(jnp.ones(space.sample_shape))
    np.testing.assert_allclose(
        space.modal_integral(coefficients),
        4.0 * np.pi * radius**2,
        rtol=1e-11,
    )
    raised = spectral_api.SphericalSpinOperatorPlan(
        "raise", physical_units=False
    ).prepare(space)
    np.testing.assert_allclose(raised.apply(coefficients), 0.0, atol=1e-11)
    rotation = spectral_api.SphericalRotationPlan(space).prepare()
    rotated = rotation.apply(coefficients, jnp.asarray([0.3, 0.4, -0.2]))
    np.testing.assert_allclose(rotated, coefficients, atol=1e-11)


def test_spherical_scattered_fit_healpix_and_inactive_nan_safety():
    space = phx.discretization.SphericalSpectralPlan(3).prepare()
    sample_plan = spectral_api.SphericalSamplePlan.healpix(2, ordering="nested")
    prepared = sample_plan.prepare(space)
    coefficients = space.project(_harmonic(space, degree=1, order=1))
    values = prepared.evaluate(coefficients)
    fitted = prepared.fit(values)
    np.testing.assert_allclose(fitted.coefficients, coefficients, atol=5e-9)
    assert prepared.report.sample_capacity == 48
    assert np.isclose(prepared.report.active_weight_sum, 4.0 * np.pi)
    masked = spectral_api.SphericalSamplePlan(
        sample_plan.points.at[-1].set(jnp.asarray([jnp.nan] * 3)),
        weights=sample_plan.weights,
        active_mask=jnp.arange(48) < 47,
        tikhonov=1e-8,
    ).prepare(space)
    assert jnp.all(jnp.isfinite(masked.evaluate(coefficients)))


def test_spherical_clebsch_gordan_constant_identity_and_modal_transfer():
    coarse = phx.discretization.SphericalSpectralPlan(3).prepare()
    fine = phx.discretization.SphericalSpectralPlan(5).prepare()
    constant = coarse.project(jnp.ones(coarse.sample_shape))
    harmonic = coarse.project(_harmonic(coarse, degree=1, order=1))
    coupling = spectral_api.SphericalClebschGordanPlan(
        coarse, coarse, output_bandlimit=5
    ).prepare()
    product = coupling.apply(constant, harmonic)
    transferred = spectral_api.prepare_spectral_modal_transfer(coarse, fine)(harmonic)
    assert coupling.report.recurrence_residual < 1e-10
    np.testing.assert_allclose(product, transferred, atol=2e-10)
    restricted = spectral_api.prepare_spectral_modal_transfer(fine, coarse)
    evidence = restricted.apply_with_evidence(product)
    assert evidence.removed_coefficient_energy >= 0.0


def test_spherical_dynamic_evaluation_matches_grid_and_prepared_samples():
    space = phx.discretization.SphericalSpectralPlan(4).prepare(radius=1.7)
    values = _harmonic(space, degree=3, order=2) + 0.35 * _harmonic(
        space, degree=2, order=1
    )
    coefficients = space.project(values)
    directions = space.points.reshape(space.sample_shape + (3,))

    # Negative orders are derived from the independent half, and invalid padded
    # storage cannot leak even when it contains nonfinite values.
    contaminated = coefficients.at[3, 1].set(9.0 - 4.0j)
    contaminated = contaminated.at[0, 0].set(jnp.nan + 1j * jnp.inf)
    expected = space.reconstruct(coefficients)
    actual = space.evaluate(contaminated, directions)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)

    payload_scale = jnp.asarray([[1.0, -0.5, 0.25], [1.5, 0.75, -2.0]])
    payload_coefficients = contaminated[..., None, None] * payload_scale
    payload = space.evaluate(payload_coefficients, directions)
    assert payload.shape == space.sample_shape + payload_scale.shape
    np.testing.assert_allclose(
        payload,
        expected[..., None, None] * payload_scale,
        rtol=2e-10,
        atol=2e-10,
    )

    sample_plan = spectral_api.SphericalSamplePlan.healpix(2, ordering="nested")
    prepared = sample_plan.prepare(space)
    np.testing.assert_allclose(
        space.evaluate(coefficients, sample_plan.points),
        prepared.evaluate(coefficients),
        rtol=2e-10,
        atol=2e-10,
    )


def test_spherical_complex_dynamic_evaluation_is_linear_jitted_and_lane_local():
    precision = phx.discretization.SpectralPrecisionPolicy(jnp.complex128)
    space = phx.discretization.SphericalSpectralPlan(
        4, reality=False, precision=precision
    ).prepare()
    center = space.layout.bandlimit - 1
    first_amplitude = 0.7 - 0.4j
    second_amplitude = -0.25 + 0.6j
    first = jnp.zeros(space.coefficient_shape, dtype=jnp.complex128)
    first = first.at[3, center - 2].set(first_amplitude)
    second = jnp.zeros_like(first)
    second = second.at[2, center + 1].set(second_amplitude)
    directions = jnp.asarray(
        [
            [[1.2, -0.4, 0.8], [-0.7, 1.1, 0.3]],
            [[0.25, 0.8, -1.3], [1.4, 0.6, -0.5]],
        ]
    )

    first_reference = first_amplitude * _scipy_harmonic_on_vectors(3, -2, directions)
    second_reference = second_amplitude * _scipy_harmonic_on_vectors(2, 1, directions)
    np.testing.assert_allclose(
        space.evaluate(first, directions),
        first_reference,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        space.evaluate(second, directions),
        second_reference,
        rtol=2e-11,
        atol=2e-11,
    )

    alpha, beta = 0.4 + 0.2j, -0.3 + 0.5j
    combined = alpha * first + beta * second
    evaluated = eqx.filter_jit(
        lambda prepared, modal, query: prepared.evaluate(modal, query)
    )(space, combined, directions)
    expected = alpha * first_reference + beta * second_reference
    np.testing.assert_allclose(evaluated, expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        evaluated,
        alpha * space.evaluate(first, directions)
        + beta * space.evaluate(second, directions),
        rtol=2e-11,
        atol=2e-11,
    )

    valid = directions[0, 0]
    lane_directions = jnp.stack(
        (
            valid,
            jnp.zeros(3),
            jnp.full((3,), jnp.nan),
            jnp.asarray([jnp.inf, 1.0, 0.0]),
            7.0 * valid,
        )
    )
    lanes = space.evaluate(combined, lane_directions)
    assert jnp.all(jnp.isfinite(lanes[jnp.asarray([0, 4])]))
    assert jnp.all(jnp.isnan(jnp.real(lanes[1:4])))
    assert jnp.all(jnp.isnan(jnp.imag(lanes[1:4])))
    np.testing.assert_allclose(lanes[0], lanes[4], rtol=2e-12, atol=2e-12)


def test_spherical_dynamic_evaluation_direction_ad_and_spin_contract():
    space = phx.discretization.SphericalSpectralPlan(3).prepare()
    center = space.layout.bandlimit - 1
    coefficients = jnp.zeros(space.coefficient_shape, dtype=jnp.complex128)
    coefficients = coefficients.at[1, center].set(np.sqrt(4.0 * np.pi / 3.0))
    direction = jnp.asarray([1.2, -0.7, 2.1])

    value, derivative = jax.value_and_grad(
        lambda query: space.evaluate(coefficients, query)
    )(direction)
    radius = jnp.linalg.norm(direction)
    expected_value = direction[2] / radius
    expected_derivative = (
        jnp.asarray([0.0, 0.0, 1.0]) / radius - direction[2] * direction / radius**3
    )
    np.testing.assert_allclose(value, expected_value, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=2e-11, atol=2e-11)

    precision = phx.discretization.SpectralPrecisionPolicy(jnp.complex128)
    spin_space = phx.discretization.SphericalSpectralPlan(
        2, spin=1, reality=False, precision=precision
    ).prepare()
    spin_coefficients = jnp.zeros(spin_space.coefficient_shape, dtype=jnp.complex128)
    spin_coefficients = spin_coefficients.at[1, 2].set(1.0)
    with pytest.raises(ValueError, match="spin zero"):
        spin_space.evaluate(spin_coefficients, jnp.asarray([1.0, 0.0, 0.0]))
