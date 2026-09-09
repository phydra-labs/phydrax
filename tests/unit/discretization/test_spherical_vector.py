#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import SpectralPrecisionPolicy, SphericalSpectralPlan
from phydrax.discretization.spectral._spherical_vector import (
    PreparedSphericalVectorOperators,
)


@pytest.mark.parametrize(
    ("sampling", "execution"), [("mwss", "recursive"), ("gl", "precomputed")]
)
def test_tangent_gradient_and_hodge_identities_including_poles(sampling, execution):
    radius = 2.3
    space = SphericalSpectralPlan(5, sampling=sampling, execution=execution).prepare(
        radius=radius
    )
    vector = PreparedSphericalVectorOperators(space)
    theta, phi = jnp.meshgrid(space.transform.theta, space.transform.phi, indexing="ij")
    scalar = jnp.sin(theta) * jnp.cos(phi)
    coefficients = space.project(scalar)
    east, north = vector.gradient(coefficients)

    np.testing.assert_allclose(east, -jnp.sin(phi) / radius, atol=2e-11)
    np.testing.assert_allclose(north, -jnp.cos(theta) * jnp.cos(phi) / radius, atol=2e-11)
    np.testing.assert_allclose(
        vector.divergence(east, north), space.modal_laplacian(coefficients), atol=2e-11
    )
    np.testing.assert_allclose(vector.curl(east, north), 0.0, atol=2e-11)
    np.testing.assert_allclose(vector.divergence(-north, east), 0.0, atol=2e-11)
    np.testing.assert_allclose(
        vector.curl(-north, east), space.modal_laplacian(coefficients), atol=2e-11
    )
    if sampling == "mwss":
        np.testing.assert_allclose(theta[jnp.array([0, -1]), 0], [0.0, np.pi], atol=1e-15)
        # Longitude-labelled frame components vary at each pole; the Cartesian
        # vector is single valued. No division by sin(theta) is allowed here.
        cartesian_x = -east * jnp.sin(phi) - north * jnp.cos(theta) * jnp.cos(phi)
        cartesian_y = east * jnp.cos(phi) - north * jnp.cos(theta) * jnp.sin(phi)
        cartesian_z = north * jnp.sin(theta)
        np.testing.assert_allclose(
            cartesian_x[jnp.array([0, -1])], 1.0 / radius, atol=2e-11
        )
        np.testing.assert_allclose(cartesian_y[jnp.array([0, -1])], 0.0, atol=2e-11)
        np.testing.assert_allclose(cartesian_z[jnp.array([0, -1])], 0.0, atol=2e-11)


def test_oblique_solid_rotation_curl_and_helmholtz_inversion():
    radius, omega = 1.7, 0.23
    space = SphericalSpectralPlan(5, sampling="mwss").prepare(radius=radius)
    vector = PreparedSphericalVectorOperators(space)
    theta, phi = jnp.meshgrid(space.transform.theta, space.transform.phi, indexing="ij")
    east = omega * radius * (0.3 * jnp.sin(theta) - jnp.cos(theta) * jnp.cos(phi))
    north = omega * radius * jnp.sin(phi)
    vorticity = space.project(
        2.0 * omega * (jnp.sin(theta) * jnp.cos(phi) + 0.3 * jnp.cos(theta))
    )
    zero = jnp.zeros(space.coefficient_shape, dtype=complex)

    np.testing.assert_allclose(vector.curl(east, north), vorticity, atol=2e-11)
    np.testing.assert_allclose(vector.divergence(east, north), zero, atol=2e-11)
    reconstructed = vector.wind(vorticity, zero)
    np.testing.assert_allclose(reconstructed[0], east, atol=2e-11)
    np.testing.assert_allclose(reconstructed[1], north, atol=2e-11)


def test_batched_channel_last_helmholtz_roundtrip_is_jittable():
    space = SphericalSpectralPlan(5).prepare(radius=3.0)
    vector = PreparedSphericalVectorOperators(space)
    theta, phi = jnp.meshgrid(space.transform.theta, space.transform.phi, indexing="ij")
    potential = space.project(jnp.sin(theta) ** 2 * jnp.cos(2.0 * phi))
    stream = space.project(jnp.cos(theta))
    scales = jnp.asarray([[1.0, -0.5, 2.0], [0.3, 0.8, -0.2]])
    chi = potential[None, ..., None] * scales[:, None, None, :]
    psi = stream[None, ..., None] * scales[:, None, None, :]
    vorticity = space.modal_laplacian(psi)
    divergence = space.modal_laplacian(chi)
    wind = eqx.filter_jit(vector.wind)(vorticity, divergence)
    expected_chi = vector.gradient(chi)
    expected_psi = vector.gradient(psi)
    np.testing.assert_allclose(wind[0], expected_chi[0] - expected_psi[1], atol=2e-11)
    np.testing.assert_allclose(wind[1], expected_chi[1] + expected_psi[0], atol=2e-11)
    np.testing.assert_allclose(vector.divergence(*wind), divergence, atol=2e-11)
    np.testing.assert_allclose(vector.curl(*wind), vorticity, atol=2e-11)


def test_constant_gradient_and_explicit_wind_null_mode_policy_under_jit():
    space = SphericalSpectralPlan(4).prepare()
    vector = PreparedSphericalVectorOperators(space)
    project = PreparedSphericalVectorOperators(space, mean_policy="project")
    constant = jnp.zeros(space.coefficient_shape, dtype=complex).at[0, 3].set(2.0)
    zero = jnp.zeros_like(constant)
    np.testing.assert_allclose(vector.gradient(constant), 0.0, atol=0.0)
    assert float(eqx.filter_jit(vector.null_mode_defect)(constant)) == 2.0
    compiled = eqx.filter_jit(vector.wind)
    with pytest.raises(eqx.EquinoxRuntimeError, match="zero-mean"):
        jax.block_until_ready(compiled(constant, zero))
    with pytest.raises(eqx.EquinoxRuntimeError, match="zero-mean"):
        jax.block_until_ready(compiled(zero, constant))
    np.testing.assert_allclose(eqx.filter_jit(project.wind)(constant, constant), 0.0)
    assert project.operator_id != vector.operator_id


def test_reality_conjugacy_and_layout_are_enforced_not_silently_repaired():
    space = SphericalSpectralPlan(4).prepare()
    vector = PreparedSphericalVectorOperators(space)
    zero = jnp.zeros(space.coefficient_shape, dtype=complex)
    compiled = eqx.filter_jit(vector.gradient)
    # A non-real zonal mode is missed by negative-order-only conjugacy checks.
    for malformed in (zero.at[1, 3].set(1j), zero.at[1, 4].set(1.0)):
        with pytest.raises(eqx.EquinoxRuntimeError, match="conjugacy"):
            jax.block_until_ready(compiled(malformed))
    with pytest.raises(ValueError, match="mode shape"):
        vector.gradient(jnp.zeros((4, 4)))
    physical = jnp.zeros(space.sample_shape)
    with pytest.raises(TypeError, match="real arrays"):
        vector.divergence(physical.astype(complex), physical)
    with pytest.raises(ValueError, match="identical shapes"):
        vector.curl(physical, physical[..., None])
    precision = SpectralPrecisionPolicy(jnp.complex128)
    complex_space = SphericalSpectralPlan(4, reality=False, precision=precision).prepare()
    spin_space = SphericalSpectralPlan(
        4, spin=1, reality=False, precision=precision
    ).prepare()
    for incompatible in (complex_space, spin_space):
        with pytest.raises(ValueError, match="real spin-zero"):
            PreparedSphericalVectorOperators(incompatible)


def test_invalid_padded_capacity_is_inert_for_vector_calculus():
    space = SphericalSpectralPlan(4).prepare()
    vector = PreparedSphericalVectorOperators(space)
    coefficients = jnp.zeros(space.coefficient_shape, dtype=complex).at[1, 3].set(0.7)
    contaminated = coefficients.at[0, 0].set(jnp.nan + 1j * jnp.inf)
    np.testing.assert_allclose(
        vector.gradient(contaminated), vector.gradient(coefficients)
    )
    np.testing.assert_allclose(
        vector.wind(contaminated, coefficients), vector.wind(coefficients, coefficients)
    )
