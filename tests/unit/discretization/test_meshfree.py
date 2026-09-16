#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _points():
    return jnp.asarray(
        [
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0),
            (-1.0, 0.0),
            (0.0, 0.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ]
    )


def _polynomial(points):
    x, y = points[:, 0], points[:, 1]
    return x**2 + 2.0 * x * y + 3.0 * y**2 + 4.0 * x - 2.0 * y + 1.0


@pytest.mark.parametrize("method", ("rbf-fd", "gmls"))
def test_meshfree_gradient_and_laplacian_reproduce_quadratics(method) -> None:
    points = _points()
    plan = phx.discretization.MeshfreeStencilPlan(
        points, stencil_size=9, polynomial_degree=2
    )
    gradient = plan.prepare("gradient", method=method, axis=0)
    laplacian = plan.prepare("laplacian", method=method)
    values = _polynomial(points)
    expected_gradient = 2.0 * points[:, 0] + 2.0 * points[:, 1] + 4.0

    actual_gradient = jax.jit(gradient.apply)(values)
    actual_laplacian = jax.jit(laplacian.apply)(values)

    assert gradient.evidence.passed
    assert laplacian.evidence.passed
    assert jnp.allclose(actual_gradient, expected_gradient, atol=2.0e-10)
    assert jnp.allclose(actual_laplacian, 8.0, atol=2.0e-10)


def test_meshfree_vector_payload_preserves_trailing_axes() -> None:
    points = _points()
    operator = phx.discretization.MeshfreeStencilPlan(
        points, stencil_size=9, polynomial_degree=2
    ).prepare("value", method="gmls")
    values = jnp.stack((_polynomial(points), 2.0 * _polynomial(points)), axis=-1)

    result = operator(values)

    assert result.shape == values.shape
    assert jnp.allclose(result, values, atol=2.0e-10)
