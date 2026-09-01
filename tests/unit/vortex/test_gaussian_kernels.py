#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from phydrax.operators.integral.vortex._gaussian2d import (
    gaussian_vortex_velocity_2d,
    gaussian_vortex_velocity_gradient_2d,
    gaussian_vortex_vorticity_2d,
)
from phydrax.operators.integral.vortex._gaussian3d import GaussianErfVortexKernel3D


def test_gaussian_2d_kernel_has_lamb_oseen_profile_and_regular_center_limit():
    displacement = jnp.asarray(((0.0, 0.0), (0.7, -0.4), (4.0, 0.0)))
    circulation = jnp.asarray((1.3, -0.8, 2.0))
    core_radius = jnp.asarray((0.35, 0.6, 0.25))

    velocity = gaussian_vortex_velocity_2d(displacement, circulation, core_radius)
    gradient = gaussian_vortex_velocity_gradient_2d(
        displacement, circulation, core_radius
    )
    vorticity = gaussian_vortex_vorticity_2d(displacement, circulation, core_radius)
    squared_distance = jnp.sum(displacement * displacement, axis=-1)
    factor = circulation * (-jnp.expm1(-squared_distance / core_radius**2))
    factor = factor / (
        2.0 * jnp.pi * jnp.where(squared_distance > 0.0, squared_distance, 1.0)
    )
    expected_velocity = factor[:, None] * jnp.stack(
        (-displacement[:, 1], displacement[:, 0]), axis=-1
    )
    expected_velocity = expected_velocity.at[0].set(jnp.zeros(2))
    expected_vorticity = (
        circulation
        * jnp.exp(-squared_distance / core_radius**2)
        / (jnp.pi * core_radius**2)
    )

    np.testing.assert_allclose(velocity, expected_velocity, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(vorticity, expected_vorticity, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        gradient[0],
        circulation[0]
        / (2.0 * jnp.pi * core_radius[0] ** 2)
        * jnp.asarray(((0.0, -1.0), (1.0, 0.0))),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(jnp.trace(gradient, axis1=-2, axis2=-1), 0.0, atol=3e-7)
    np.testing.assert_allclose(
        gradient[:, 1, 0] - gradient[:, 0, 1],
        vorticity,
        rtol=3e-6,
        atol=3e-7,
    )
    assert jnp.all(jnp.isfinite(velocity))
    assert jnp.all(jnp.isfinite(gradient))


def test_gaussian_erf_3d_kernel_matches_biot_savart_and_gaussian_vorticity():
    kernel = GaussianErfVortexKernel3D()
    displacement = jnp.asarray(((0.0, 0.0, 0.0), (0.8, -0.3, 0.5), (5.0, 0.0, 0.0)))
    strength = jnp.asarray(((1.0, -0.5, 0.25), (0.2, 1.1, -0.7), (0.0, 2.0, 0.0)))
    core_radius = jnp.asarray((0.4, 0.65, 0.3))

    evaluation = kernel.evaluate(displacement, strength, core_radius)
    distance = jnp.linalg.norm(displacement, axis=-1)
    safe_distance = jnp.where(distance > 0.0, distance, 1.0)
    scaled = safe_distance / core_radius
    cutoff = jsp.erf(scaled / math.sqrt(2.0)) - math.sqrt(
        2.0 / math.pi
    ) * scaled * jnp.exp(-0.5 * scaled**2)
    expected_velocity = (
        (
            cutoff[:, None]
            * jnp.cross(strength, displacement)
            / (4.0 * jnp.pi * safe_distance[:, None] ** 3)
        )
        .at[0]
        .set(jnp.zeros(3))
    )
    squared_scaled = jnp.sum((displacement / core_radius[:, None]) ** 2, axis=-1)
    expected_vorticity = (
        jnp.exp(-0.5 * squared_scaled) / ((2.0 * jnp.pi) ** 1.5 * core_radius**3)
    )[:, None] * strength

    np.testing.assert_allclose(
        evaluation.velocity, expected_velocity, rtol=3e-6, atol=3e-7
    )
    np.testing.assert_allclose(
        evaluation.vorticity, expected_vorticity, rtol=3e-6, atol=3e-7
    )
    np.testing.assert_allclose(
        jnp.trace(evaluation.velocity_gradient, axis1=-2, axis2=-1),
        0.0,
        atol=4e-7,
    )
    assert bool(evaluation.coincident[0])
    assert bool(evaluation.finite)
    assert jnp.all(jnp.isfinite(evaluation.velocity_gradient))
