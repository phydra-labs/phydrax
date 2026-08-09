#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _points():
    diagonal = 1.0 / jnp.sqrt(2.0)
    return jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [diagonal, 0.0, diagonal]])


def _scalar_kernel(length_scale=0.7):
    return phx.kernels.SphereSpectralKernel(
        2,
        5,
        phx.kernels.MaternSpectralMultiplier(length_scale, 1.4),
    )


def test_projected_tangent_covariance_is_intrinsic_symmetric_and_psd():
    points = _points()
    kernel = phx.kernels.sphere_tangent_kernel(_scalar_kernel())
    covariance = kernel.matrix(points, points)

    assert covariance.shape == (9, 9)
    assert jnp.allclose(covariance, covariance.T, atol=1e-10)
    assert jnp.allclose(jnp.diag(covariance), kernel.diagonal(points))
    assert np.min(np.linalg.eigvalsh(np.asarray(covariance))) >= -1e-9

    for left in points:
        for right in points:
            block = kernel.block(left, right)
            assert jnp.allclose(left @ block, 0.0, atol=1e-9)
            assert jnp.allclose(block @ right, 0.0, atol=1e-9)


def test_sphere_tangent_covariance_is_rotation_equivariant():
    points = _points()
    rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    kernel = phx.kernels.sphere_tangent_kernel(_scalar_kernel())

    for left in points:
        for right in points:
            expected = rotation @ kernel.block(left, right) @ rotation.T
            actual = kernel.block(rotation @ left, rotation @ right)
            assert jnp.allclose(actual, expected, atol=1e-9)


def test_sphere_lifts_share_scalar_membership_tolerance_and_canonicalization():
    point = jnp.sqrt(1.0005) * jnp.asarray([1.0, 0.0, 0.0])
    scalar = phx.kernels.SphereSpectralKernel(
        2,
        3,
        phx.kernels.HeatSpectralMultiplier(0.2),
        membership_tolerance=1e-3,
    )
    tangent = phx.kernels.sphere_tangent_kernel(scalar)
    one_form = phx.kernels.sphere_differential_form_kernel(scalar, 1)

    assert jnp.all(jnp.isfinite(tangent.block(point, point)))
    assert jnp.all(jnp.isfinite(one_form.block(point, point)))


def test_degree_one_form_kernel_matches_tangent_covariance():
    points = _points()
    scalar = _scalar_kernel()
    tangent = phx.kernels.sphere_tangent_kernel(scalar)
    one_form = phx.kernels.sphere_differential_form_kernel(scalar, 1)

    assert one_form.output_dimension == 3
    assert jnp.allclose(one_form.matrix(points, points), tangent.matrix(points, points))
    assert jnp.allclose(one_form.diagonal(points), tangent.diagonal(points))


def test_higher_form_covariance_is_positive_semidefinite_and_differentiable():
    points = _points()

    def objective(length_scale):
        kernel = phx.kernels.sphere_differential_form_kernel(
            _scalar_kernel(length_scale), 2
        )
        return jnp.sum(kernel.matrix(points, points))

    kernel = phx.kernels.sphere_differential_form_kernel(_scalar_kernel(), 2)
    covariance = kernel.matrix(points, points)

    assert kernel.output_dimension == 3
    assert covariance.shape == (9, 9)
    assert jnp.allclose(covariance, covariance.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(np.asarray(covariance))) >= -1e-9
    assert jnp.isfinite(jax.jit(jax.grad(objective))(jnp.asarray(0.7)))
