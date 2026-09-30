#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.exterior import FormType
from phydrax.kernels import ProjectedDifferentialFormKernel


def _points() -> Any:
    diagonal = 1.0 / jnp.sqrt(2.0)
    return jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [diagonal, 0.0, diagonal]])


def _scalar_kernel(length_scale: Any = 0.7) -> Any:
    return phx.kernels.SphereSpectralKernel(
        2,
        5,
        phx.kernels.MaternSpectralMultiplier(length_scale, 1.4),
    )


def test_operator_valued_scenario_1() -> None:
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
    points = _points()
    rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    kernel = phx.kernels.sphere_tangent_kernel(_scalar_kernel())

    for left in points:
        for right in points:
            expected = rotation @ kernel.block(left, right) @ rotation.T
            actual = kernel.block(rotation @ left, rotation @ right)
            assert jnp.allclose(actual, expected, atol=1e-9)
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
    points = _points()
    scalar = _scalar_kernel()
    tangent = phx.kernels.sphere_tangent_kernel(scalar)
    one_form = phx.kernels.sphere_differential_form_kernel(scalar, 1)

    assert one_form.output_dimension == 3
    assert jnp.allclose(one_form.matrix(points, points), tangent.matrix(points, points))
    assert jnp.allclose(one_form.diagonal(points), tangent.diagonal(points))


def test_higher_form_covariance_is_positive_semidefinite_and_differentiable() -> None:
    points = _points()

    def objective(length_scale: Any) -> Any:
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


def test_kernel_functional_terms_require_exact_integer_derivative_orders() -> None:
    with pytest.raises(TypeError, match="exact integers"):
        phx.kernels.KernelFunctionalTerm(
            "field",
            jnp.asarray([[0.0]]),
            # ty: ignore[invalid-argument-type]
            ((0.5,),),
            jnp.ones((1, 1, 1, 1)),
        )
    with pytest.raises(TypeError, match="exact integers"):
        phx.kernels.KernelFunctionalTerm(
            "field",
            jnp.asarray([[0.0]]),
            ((True,),),
            jnp.ones((1, 1, 1, 1)),
        )


def test_projected_form_identity_distinguishes_scientific_and_derivative_contracts() -> (
    None
):
    scalar = _scalar_kernel()

    def projector(point: Array) -> Array:
        return jnp.eye(point.shape[0], dtype=point.dtype)

    types = (
        FormType(2, 1),
        FormType(2, 1, ambient_dimension=3),
        FormType(2, 1, twist="twisted"),
        FormType(3, 1),
    )
    kernels = [
        ProjectedDifferentialFormKernel(
            scalar,
            projector,
            form_type,
            projector_id="identity",
            projector_derivative_order=order,
        )
        for form_type in types
        for order in (None, 0, 2)
    ]
    assert len({kernel.kernel_id for kernel in kernels}) == len(kernels)
    repeated = ProjectedDifferentialFormKernel(
        scalar, projector, FormType(2, 1), projector_id="identity"
    )
    assert repeated.kernel_id == kernels[0].kernel_id
    assert kernels[1].max_derivative_order == 0
    assert kernels[2].max_derivative_order == 2


def test_sphere_two_form_covariance_matches_oriented_area_oracle() -> None:
    scalar = _scalar_kernel()
    kernel = phx.kernels.sphere_differential_form_kernel(scalar, 2)
    assert (
        kernel.form_type.form_type_id == FormType(2, 2, ambient_dimension=3).form_type_id
    )

    def area(point: Array) -> Array:
        # Lexicographic (xy, xz, yz) coefficients of contraction with volume.
        return jnp.stack((point[2], -point[1], point[0]))

    for left in _points():
        for right in _points():
            left_area, right_area = area(left), area(right)
            expected = (
                scalar.pairwise(left, right)
                * jnp.vdot(left_area, right_area)
                * jnp.outer(left_area, right_area)
            )
            np.testing.assert_allclose(kernel.block(left, right), expected, atol=1e-10)


def test_zero_form_kernel_retains_explicit_scalar_component() -> None:
    scalar = _scalar_kernel()
    kernel = phx.kernels.sphere_differential_form_kernel(scalar, 0)
    assert kernel.output_dimension == 1
    points = _points()
    np.testing.assert_allclose(
        kernel.blocks(points, points)[..., 0, 0], scalar.matrix(points, points)
    )
    with pytest.raises(ValueError):
        phx.kernels.sphere_differential_form_kernel(scalar, 3)
