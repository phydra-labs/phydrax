#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax._interpolation import apply_gather_stencil, rectilinear_stencil


def test_nonuniform_rectilinear_map_is_affine_exact_with_complex_payload():
    x = jnp.asarray([-1.0, -0.2, 0.4, 2.0])
    query = jnp.asarray([[-0.7], [0.1], [1.25]])
    values = jnp.stack((2.0 * x + 3.0, (1.0 - x) * (1.0 + 2.0j)), axis=-1)
    stencil = rectilinear_stencil((x,), query, boundary=("clamp",))

    output = apply_gather_stencil(values, stencil).values

    assert output.shape == (3, 2)
    assert jnp.allclose(output[:, 0], 2.0 * query[:, 0] + 3.0)
    assert jnp.allclose(output[:, 1], (1.0 - query[:, 0]) * (1.0 + 2.0j))


def test_batched_two_dimensional_map_uses_case_local_sources():
    x = jnp.asarray([0.0, 0.3, 1.0])
    y = jnp.asarray([-1.0, 0.5, 2.0])
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    base = xx + 2.0 * yy
    values = jnp.stack((base, base + 10.0), axis=0)[..., None]
    query = jnp.asarray(
        [
            [[0.15, -0.25], [0.65, 1.0]],
            [[0.15, -0.25], [0.65, 1.0]],
        ]
    )
    stencil = rectilinear_stencil(
        (x, y),
        query,
        boundary=("clamp", "clamp"),
        batch_shape=(2,),
    )

    output = apply_gather_stencil(values.reshape((-1, 1)), stencil).values[..., 0]
    expected = query[..., 0] + 2.0 * query[..., 1] + jnp.asarray([[0.0], [10.0]])

    assert jnp.allclose(output, expected)


def test_boundary_modes_have_distinct_explicit_support():
    nodes = jnp.asarray([0.0, 1.0])
    values = jnp.asarray([0.0, 1.0])
    query = jnp.asarray([[-0.25], [1.25]])

    clamp = rectilinear_stencil((nodes,), query, boundary=("clamp",))
    reflect = rectilinear_stencil((nodes,), query, boundary=("reflect",))
    constant = rectilinear_stencil((nodes,), query, boundary=("constant",))
    periodic = rectilinear_stencil(
        (jnp.asarray([0.0, 0.5]),),
        jnp.asarray([[0.75], [1.25]]),
        boundary=("periodic",),
        periods=(1.0,),
        axis_bounds=((0.0, 1.0),),
    )

    assert jnp.allclose(
        apply_gather_stencil(values, clamp).values,
        jnp.asarray([0.0, 1.0]),
    )
    assert jnp.allclose(
        apply_gather_stencil(values, reflect).values,
        jnp.asarray([0.25, 0.75]),
    )
    assert jnp.array_equal(
        apply_gather_stencil(values, constant).support,
        jnp.asarray([False, False]),
    )
    assert jnp.allclose(
        apply_gather_stencil(jnp.asarray([0.0, 1.0]), periodic).values,
        jnp.asarray([0.5, 0.5]),
    )


def test_rectilinear_source_masks_support_strict_and_renormalized_modes():
    nodes = jnp.asarray([0.0, 1.0])
    query = jnp.asarray([[0.25]])
    values = jnp.asarray([2.0, 10.0])
    mask = jnp.asarray([True, False])
    stencil = rectilinear_stencil((nodes,), query, boundary=("clamp",))

    strict = apply_gather_stencil(
        values,
        stencil,
        source_mask=mask,
        mask_mode="strict",
    )
    renormalized = apply_gather_stencil(
        values,
        stencil,
        source_mask=mask,
        mask_mode="renormalize",
    )

    assert not bool(strict.support[0])
    assert jnp.allclose(strict.values, 0.0)
    assert bool(renormalized.support[0])
    assert jnp.allclose(renormalized.values, 2.0)


def test_rectilinear_map_is_jittable_and_differentiable_inside_cells():
    nodes = jnp.asarray([-1.0, 0.0, 2.0])
    values = 4.0 * nodes - 3.0

    @jax.jit
    def evaluate(query):
        stencil = rectilinear_stencil(
            (nodes,),
            query.reshape((1, 1)),
            boundary=("clamp",),
        )
        return apply_gather_stencil(values, stencil).values[0]

    assert jnp.allclose(evaluate(jnp.asarray(0.5)), -1.0)
    assert jnp.allclose(jax.grad(evaluate)(jnp.asarray(0.5)), 4.0)
