from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _complex_space(bandlimit: Any, spin: Any) -> Any:
    return phx.discretization.SphericalSpectralPlan(
        bandlimit,
        spin=spin,
        reality=False,
        precision=phx.discretization.SpectralPrecisionPolicy(jnp.complex128),
    ).prepare()


def _modal(space: Any) -> Any:
    center = space.layout.bandlimit - 1
    coefficients = jnp.zeros(space.coefficient_shape, dtype=jnp.complex128)
    for degree in range(abs(space.layout.spin), space.layout.bandlimit):
        coefficients = coefficients.at[degree, center + min(1, degree)].set(
            0.2 * (degree + 1) - 0.1j
        )
        coefficients = coefficients.at[degree, center - min(1, degree)].set(
            -0.07 + 0.03j * degree
        )
    return coefficients


def test_spherical_spin_evaluation_scenario_1() -> None:
    for spin in [-2, -1, 1, 2]:
        space = _complex_space(5, spin)
        coefficients = _modal(space)
        theta, phi = jnp.meshgrid(
            space.transform.theta,
            space.transform.phi,
            indexing="ij",
        )
        actual = eqx.filter_jit(space.evaluate_angles)(coefficients, theta, phi)
        expected = space.reconstruct(coefficients)
        np.testing.assert_allclose(actual, expected, rtol=4e-11, atol=4e-12)
    spin = 2
    space = _complex_space(4, spin)
    coefficients = _modal(space)
    theta, phi, angle = 0.8, -0.4, 0.3
    baseline = space.evaluate_angles(coefficients, theta, phi)
    rotated = space.evaluate_angles(
        coefficients,
        theta,
        phi,
        frame_angle=angle,
    )
    np.testing.assert_allclose(
        rotated,
        baseline * jnp.exp(-1j * spin * angle),
        rtol=3e-12,
        atol=3e-13,
    )

    north = jnp.asarray([0.0, 0.0, 1.0])
    east = jnp.asarray([0.0, 1.0, 0.0])
    local_north = jnp.asarray([-1.0, 0.0, 0.0])
    framed = space.evaluate(
        coefficients,
        north,
        tangent_frame=(east, local_north),
    )
    angular = space.evaluate_angles(coefficients, 0.0, 0.0)
    np.testing.assert_allclose(framed, angular, rtol=3e-12, atol=3e-13)
    _, tangent = jax.jvp(
        lambda polar: space.evaluate_angles(coefficients, polar, phi).real,
        (jnp.asarray(theta),),
        (jnp.asarray(1.0),),
    )
    assert jnp.isfinite(tangent)
    with pytest.raises(Exception, match="requires an explicit tangent_frame"):
        space.evaluate(coefficients, north)
    space = _complex_space(9, 8)
    coefficients = _modal(space)
    value = space.evaluate_angles(coefficients, 0.7, -0.2)
    assert jnp.isfinite(value)
