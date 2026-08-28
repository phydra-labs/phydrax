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


def _harmonic(space, degree=2, order=1):
    theta, phi = np.meshgrid(
        np.asarray(space.transform.theta),
        np.asarray(space.transform.phi),
        indexing="ij",
    )
    return jnp.asarray(np.real(sph_harm_y(degree, order, theta, phi)))


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
    assert jnp.allclose(space.integral(jnp.ones(space.state_shape)), 4 * jnp.pi * radius**2)
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
    actual = eqx.filter_jit(
        lambda prepared, field: prepared.laplacian(field)
    )(space, values)
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
