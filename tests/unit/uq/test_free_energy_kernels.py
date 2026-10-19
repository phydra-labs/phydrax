from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.uq._free_energy_kernels import (
    bar_kernel,
    fep_kernel,
    mbar_asymptotic_covariance_kernel,
    mbar_kernel,
    thermodynamic_integration_kernel,
)


def _assert_tree_allclose(actual: Any, expected: Any) -> None:
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1.0e-10, atol=1.0e-12)


def test_free_energy_kernels_scenario_1() -> None:
    values = jnp.asarray([0.0, 0.2, -0.1, 0.1])
    mask = jnp.asarray([True, True, True, True])
    weight = jnp.ones_like(values)
    lowered = jax.jit(fep_kernel).lower(values, mask, weight).compile()
    _assert_tree_allclose(lowered(values, mask, weight), fep_kernel(values, mask, weight))
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


def test_bar_kernel_lowers_and_matches_eager() -> None:
    values = jnp.asarray([1.0, 1.1, 0.9, -1.0, -1.1, -0.9])
    forward = jnp.asarray([True, True, True, False, False, False])
    reverse = ~forward
    weight = jnp.ones_like(values)

    def execute(
        work: Any, forward_mask: Any, reverse_mask: Any, sample_weight: Any
    ) -> Any:
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


def test_mbar_kernels_lower_and_match_eager() -> None:
    values = jnp.asarray(
        [
            [0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
            [0.2, 0.3, 0.2, 0.3, 0.2, 0.3],
        ]
    )
    origins = jnp.asarray([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    sample_weight = jnp.ones((6,))

    def solve(potential: Any, origin: Any, weight: Any) -> Any:
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

    def covariance(weight_matrix: Any, observation_weight: Any, state_counts: Any) -> Any:
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


@pytest.mark.parametrize(
    "cross_weight", (0.0, 0.25, 0.5), ids=("disconnected", "overlap", "identical")
)
@pytest.mark.parametrize("reference_state", (0, 1), ids=("gauge-0", "gauge-1"))
@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_mbar_covariance_preserves_full_identity_rank_scale(
    cross_weight: float, reference_state: int, compiled: bool
) -> None:
    count = 6
    weights = jnp.asarray(
        np.repeat(
            np.asarray(
                ((1 - cross_weight, cross_weight), (cross_weight, 1 - cross_weight)),
                dtype=np.float64,
            )
            / count,
            count,
            axis=1,
        )
    )
    sample_weight = jnp.ones((2 * count,), dtype=jnp.float64)
    counts = jnp.full((2,), count, dtype=jnp.float64)

    def covariance(
        weight_matrix: jax.Array, observation_weight: jax.Array, state_counts: jax.Array
    ) -> jax.Array:
        return mbar_asymptotic_covariance_kernel(
            weight_matrix,
            observation_weight,
            state_counts,
            reference_state=reference_state,
            rank_tolerance=1e-12,
        )

    execute = jax.jit(covariance) if compiled else covariance
    actual = execute(weights, sample_weight, counts)
    # For two equally sampled populations, the contrast channel has overlap
    # eigenvalue a=(1-2p)^2 and variance 2a/[n(1-a)]; a=1 is a null
    # channel of the Moore-Penrose covariance, not an estimable contrast.
    contrast = (1 - 2 * cross_weight) ** 2
    variance = 0.0 if contrast == 1 else 2 * contrast / (count * (1 - contrast))
    expected = np.zeros((2, 2), dtype=np.float64)
    expected[1 - reference_state, 1 - reference_state] = variance
    # At most 64 float64 roundoff contributions for this two-state fixture.
    allowance = 64 * np.finfo(np.float64).eps
    np.testing.assert_allclose(actual, expected, rtol=allowance, atol=allowance)


@pytest.mark.parametrize(
    ("sample_repeats", "unsampled_states", "weak_factor"),
    ((2, 0, 1.25), (1, 0, 0.75), (1, 1, 0.75)),
    ids=("more-samples", "square", "more-states"),
)
@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_mbar_covariance_near_cutoff_matches_unreduced_operator(
    sample_repeats: int, unsampled_states: int, weak_factor: float, compiled: bool
) -> None:
    # A symmetric stochastic three-population overlap has exact eigenchannels
    # 1, sqrt(1-weak), and sqrt(1/2). All weights remain positive. The cutoff
    # is deliberately well above roundoff so the near-cutoff channel is resolved.
    rank_tolerance = 2.0**-8
    weak = weak_factor * rank_tolerance
    contrast = np.asarray((2.0, -1.0, -1.0), dtype=np.float64) / np.sqrt(6.0)
    transverse = np.asarray((0.0, 1.0, -1.0), dtype=np.float64) / np.sqrt(2.0)
    population = (
        np.full((3, 3), 1.0 / 3.0, dtype=np.float64)
        + np.sqrt(1.0 - weak) * np.outer(contrast, contrast)
        + np.sqrt(0.5) * np.outer(transverse, transverse)
    )
    weights = np.repeat(population / sample_repeats, sample_repeats, axis=1)
    counts = np.full((3,), sample_repeats, dtype=np.float64)
    if unsampled_states:
        weights = np.concatenate(
            (
                weights,
                np.full((1, weights.shape[1]), 1.0 / weights.shape[1], dtype=np.float64),
            ),
            axis=0,
        )
        counts = np.concatenate((counts, np.zeros((1,), dtype=np.float64)))
    sample_weight = np.ones((weights.shape[1],), dtype=np.float64)
    # The three-channel construction and three-term normalization contribute
    # at most 32 float64 roundoff terms.
    normalization_roundoff = 32 * np.finfo(np.float64).eps
    np.testing.assert_allclose(
        counts @ weights, 1.0, rtol=0.0, atol=normalization_roundoff
    )
    np.testing.assert_allclose(
        weights @ sample_weight, 1.0, rtol=0.0, atol=normalization_roundoff
    )

    # Independently form the small unreduced sample-space operator. No thin
    # factorization, projected count matrix, or package rank helper is reused.
    weighted = np.sqrt(sample_weight)[:, None] * weights.T
    full_operator = (
        np.eye(weights.shape[1], dtype=np.float64)
        - (weighted * counts[None, :]) @ weighted.T
    )
    theta = (
        weighted.T
        @ np.linalg.pinv(full_operator, rcond=rank_tolerance, hermitian=True)
        @ weighted
    )
    gauge = np.eye(weights.shape[0], dtype=np.float64)
    gauge[:, 0] -= 1.0
    gauge[0, :] = 0.0
    expected = gauge @ theta @ gauge.T

    def covariance(
        weight_matrix: jax.Array, observation_weight: jax.Array, state_counts: jax.Array
    ) -> jax.Array:
        return mbar_asymptotic_covariance_kernel(
            weight_matrix,
            observation_weight,
            state_counts,
            reference_state=0,
            rank_tolerance=rank_tolerance,
        )

    execute = jax.jit(covariance) if compiled else covariance
    actual = execute(
        jnp.asarray(weights), jnp.asarray(sample_weight), jnp.asarray(counts)
    )
    # Reuse the existing eager/reference kernel agreement accuracy.
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_mbar_covariance_keeps_unconverged_full_operator_scale(compiled: bool) -> None:
    weights = np.repeat(
        np.asarray(((0.4,), (0.09,), (0.01,)), dtype=np.float64), 6, axis=1
    )
    counts = np.full((3,), 2.0, dtype=np.float64)
    weighted = weights.T
    full_operator = (
        np.eye(6, dtype=np.float64) - (weighted * counts[None, :]) @ weighted.T
    )
    assert np.linalg.norm(full_operator, ord=2) > 1.0
    theta = (
        weighted.T @ np.linalg.pinv(full_operator, rcond=1e-12, hermitian=True) @ weighted
    )
    gauge = np.asarray(
        ((0.0, 0.0, 0.0), (-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    expected = gauge @ theta @ gauge.T

    def covariance(weight_matrix: jax.Array) -> jax.Array:
        return mbar_asymptotic_covariance_kernel(
            weight_matrix,
            jnp.ones((6,), dtype=jnp.float64),
            jnp.asarray(counts),
            reference_state=0,
            rank_tolerance=1e-12,
        )

    execute = jax.jit(covariance) if compiled else covariance
    np.testing.assert_allclose(
        execute(jnp.asarray(weights)), expected, rtol=1e-10, atol=1e-12
    )
