#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


@pytest.mark.parametrize(
    "cone",
    [
        phx.optim.ZeroCone(3),
        phx.optim.NonnegativeCone(3),
        phx.optim.SecondOrderCone(3),
        phx.optim.RotatedSecondOrderCone(4),
    ],
)
def test_cone_projection_is_idempotent_batched_and_jittable(cone):
    values = jnp.asarray([[-2.0, 1.0, 3.0, -1.0], [4.0, -3.0, 0.5, 2.0]])[
        ..., : cone.dimension
    ]
    projected = jax.jit(cone.project)(values)

    np.testing.assert_allclose(cone.project(projected), projected, atol=1e-7)
    assert projected.shape == values.shape
    assert jnp.all(cone.contains(projected, tolerance=1e-7))


def test_self_dual_cones_satisfy_moreau_decomposition():
    cones = (
        phx.optim.NonnegativeCone(3),
        phx.optim.SecondOrderCone(3),
        phx.optim.RotatedSecondOrderCone(4),
    )
    values = (
        jnp.asarray([-2.0, 1.0, 3.0]),
        jnp.asarray([-0.5, 2.0, -1.0]),
        jnp.asarray([-1.0, 2.0, 3.0, -4.0]),
    )
    for cone, value in zip(cones, values, strict=True):
        np.testing.assert_allclose(
            cone.project(value) - cone.project(-value),
            value,
            atol=1e-7,
        )


def test_soc_and_rotated_soc_boundary_and_apex_are_finite():
    soc = phx.optim.SecondOrderCone(3)
    rotated = phx.optim.RotatedSecondOrderCone(4)

    soc_values = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    rotated_values = jnp.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 1.0, 0.0], [-1.0, 2.0, 0.0, 0.0]]
    )

    assert jnp.all(jnp.isfinite(soc.project(soc_values)))
    assert jnp.all(jnp.isfinite(rotated.project(rotated_values)))
    assert jnp.all(soc.contains(soc.project(soc_values), tolerance=1e-7))
    assert jnp.all(rotated.contains(rotated.project(rotated_values), tolerance=1e-7))


def test_product_cone_preserves_block_layout_and_complementarity():
    cone = phx.optim.ProductCone(
        (
            phx.optim.ZeroCone(1),
            phx.optim.NonnegativeCone(2),
            phx.optim.SecondOrderCone(3),
        )
    )
    value = jnp.asarray([2.0, -1.0, 3.0, 0.0, 2.0, 0.0])
    projected = cone.project(value)

    assert cone.dimension == 6
    assert cone.split(projected)[0].shape == (1,)
    assert cone.split(projected)[1].shape == (2,)
    assert cone.split(projected)[2].shape == (3,)
    assert cone.block_complementarity(projected, jnp.zeros_like(projected)).shape == (3,)


def test_cones_reject_wrong_shape_and_complex_values():
    cone = phx.optim.SecondOrderCone(3)
    with pytest.raises(ValueError, match="must end in shape"):
        cone.project(jnp.zeros(2))
    with pytest.raises(TypeError, match="real floating-point"):
        cone.project(jnp.ones(3, dtype=jnp.complex64))
