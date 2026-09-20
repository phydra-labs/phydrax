import jax
import jax.numpy as jnp
import numpy as np

from phydrax.uq._free_energy_kernels import (
    bar_kernel,
    fep_kernel,
    mbar_asymptotic_covariance_kernel,
    mbar_kernel,
    thermodynamic_integration_kernel,
)


def _assert_tree_allclose(actual, expected):
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1.0e-10, atol=1.0e-12)


def test_fep_kernel_lowers_and_matches_eager():
    values = jnp.asarray([0.0, 0.2, -0.1, 0.1])
    mask = jnp.asarray([True, True, True, True])
    weight = jnp.ones_like(values)
    lowered = jax.jit(fep_kernel).lower(values, mask, weight).compile()
    _assert_tree_allclose(lowered(values, mask, weight), fep_kernel(values, mask, weight))


def test_bar_kernel_lowers_and_matches_eager():
    values = jnp.asarray([1.0, 1.1, 0.9, -1.0, -1.1, -0.9])
    forward = jnp.asarray([True, True, True, False, False, False])
    reverse = ~forward
    weight = jnp.ones_like(values)

    def execute(work, forward_mask, reverse_mask, sample_weight):
        return bar_kernel(
            work,
            forward_mask,
            reverse_mask,
            sample_weight,
            maximum_iterations=128,
            tolerance=1.0e-10,
        )

    lowered = jax.jit(execute).lower(values, forward, reverse, weight).compile()
    _assert_tree_allclose(
        lowered(values, forward, reverse, weight),
        execute(values, forward, reverse, weight),
    )


def test_ti_kernel_lowers_and_matches_eager():
    values = jnp.asarray(
        [
            [1.9, 2.0, 2.1, 2.0],
            [2.0, 2.1, 1.9, 2.0],
            [2.1, 2.0, 1.9, 2.0],
        ]
    )
    retained = jnp.ones_like(values, dtype="bool")
    path = jnp.asarray([0.0, 0.5, 1.0])
    lowered = (
        jax.jit(thermodynamic_integration_kernel).lower(values, retained, path).compile()
    )
    _assert_tree_allclose(
        lowered(values, retained, path),
        thermodynamic_integration_kernel(values, retained, path),
    )


def test_mbar_kernels_lower_and_match_eager():
    values = jnp.asarray(
        [
            [0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
            [0.2, 0.3, 0.2, 0.3, 0.2, 0.3],
        ]
    )
    origins = jnp.asarray([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    sample_weight = jnp.ones((6,))

    def solve(potential, origin, weight):
        return mbar_kernel(
            potential,
            origin,
            weight,
            reference_state=0,
            maximum_iterations=512,
            tolerance=1.0e-10,
        )

    lowered_solve = jax.jit(solve).lower(values, origins, sample_weight).compile()
    eager = solve(values, origins, sample_weight)
    compiled = lowered_solve(values, origins, sample_weight)
    _assert_tree_allclose(compiled, eager)
    _, _, _, counts, weights = eager

    def covariance(weight_matrix, observation_weight, state_counts):
        return mbar_asymptotic_covariance_kernel(
            weight_matrix,
            observation_weight,
            state_counts,
            reference_state=0,
            rank_tolerance=1.0e-12,
        )

    lowered_covariance = (
        jax.jit(covariance).lower(weights, sample_weight, counts).compile()
    )
    _assert_tree_allclose(
        lowered_covariance(weights, sample_weight, counts),
        covariance(weights, sample_weight, counts),
    )
