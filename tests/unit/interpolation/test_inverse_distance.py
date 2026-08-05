#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax._interpolation import apply_gather_stencil, inverse_distance_stencil


def test_inverse_distance_reproduces_constants_at_arbitrary_distance():
    indices = jnp.asarray([[0, 1, 2]])
    distance_squared = jnp.asarray([[100.0, 400.0, 900.0]])
    stencil = inverse_distance_stencil(
        indices,
        distance_squared,
        source_size=3,
        power=2.0,
        regularization=1e-12,
    )

    result = apply_gather_stencil(jnp.asarray([7.0, 7.0, 7.0]), stencil)

    assert jnp.allclose(jnp.sum(stencil.weights, axis=-1), 1.0)
    assert jnp.allclose(result.values, 7.0)
    assert jnp.array_equal(result.support, jnp.asarray([True]))


def test_duplicate_anchor_snap_policy_is_explicit():
    indices = jnp.asarray([[0, 1, 2]])
    distance_squared = jnp.asarray([[0.0, 0.0, 1.0]])
    values = jnp.asarray([2.0, 4.0, 100.0])

    first = inverse_distance_stencil(
        indices,
        distance_squared,
        source_size=3,
        snap_policy="first",
    )
    average = inverse_distance_stencil(
        indices,
        distance_squared,
        source_size=3,
        snap_policy="average",
        snap_inclusive=True,
    )

    assert jnp.allclose(apply_gather_stencil(values, first).values, 2.0)
    assert jnp.allclose(apply_gather_stencil(values, average).values, 3.0)


def test_invalid_candidates_are_inert_and_no_candidate_has_no_support():
    indices = jnp.asarray([[0, 999], [999, -5]])
    distance_squared = jnp.asarray([[1.0, 0.0], [0.0, 0.0]])
    valid = jnp.asarray([[True, False], [False, False]])
    stencil = inverse_distance_stencil(
        indices,
        distance_squared,
        source_size=1,
        valid=valid,
    )

    result = apply_gather_stencil(jnp.asarray([5.0]), stencil)

    assert jnp.allclose(result.values, jnp.asarray([5.0, 0.0]))
    assert jnp.array_equal(result.support, jnp.asarray([True, False]))


def test_inverse_distance_supports_complex_payloads_jit_and_query_gradients():
    values = jnp.asarray([[1.0 + 2.0j, 3.0], [4.0 - 1.0j, -2.0]])
    indices = jnp.asarray([[0, 1]])

    @jax.jit
    def evaluate(query):
        distance_squared = jnp.stack((query**2, (2.0 - query) ** 2))[None, :]
        stencil = inverse_distance_stencil(
            indices,
            distance_squared,
            source_size=2,
            regularization=1e-6,
        )
        return apply_gather_stencil(values, stencil).values

    output = evaluate(jnp.asarray(0.75))
    gradient = jax.grad(lambda query: jnp.real(jnp.sum(evaluate(query))))(
        jnp.asarray(0.75)
    )

    assert output.shape == (1, 2)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.isfinite(gradient)
