#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_exact_weighted_univariate_wasserstein_matches_quantile_integral():
    source = jnp.asarray([0.0, 10.0])
    target = jnp.asarray([2.0, 8.0])
    source_weights = jnp.asarray([0.75, 0.25])
    target_weights = jnp.asarray([0.5, 0.5])

    distance_l1 = phx.transport.wasserstein_distance_1d(
        source,
        target,
        source_weights=source_weights,
        target_weights=target_weights,
        p=1.0,
    )
    distance_l2 = phx.transport.wasserstein_distance_1d(
        source,
        target,
        source_weights=source_weights,
        target_weights=target_weights,
        p=2.0,
    )

    assert jnp.allclose(distance_l1, 3.5)
    assert jnp.allclose(distance_l2, jnp.sqrt(19.0))


def test_univariate_distance_handles_duplicates_zero_mass_and_permutations():
    source = jnp.asarray([2.0, 0.0, 2.0, 7.0])
    source_weights = jnp.asarray([0.25, 0.5, 0.25, 0.0])
    target = jnp.asarray([1.0, 3.0])
    target_weights = jnp.asarray([0.5, 0.5])
    expected = phx.transport.wasserstein_distance_1d(
        jnp.asarray([0.0, 2.0]),
        target,
        source_weights=jnp.asarray([0.5, 0.5]),
        target_weights=target_weights,
        p=2.0,
    )
    actual = phx.transport.wasserstein_distance_1d(
        source,
        target,
        source_weights=source_weights,
        target_weights=target_weights,
        p=2.0,
    )
    permuted = phx.transport.wasserstein_distance_1d(
        source[jnp.asarray([3, 1, 0, 2])],
        target[::-1],
        source_weights=source_weights[jnp.asarray([3, 1, 0, 2])],
        target_weights=target_weights[::-1],
        p=2.0,
    )

    assert jnp.allclose(actual, expected)
    assert jnp.allclose(permuted, expected)
    with pytest.raises(ValueError, match="at least one"):
        phx.transport.wasserstein_distance_1d(source, target, p=0.5)


def test_univariate_and_sliced_distances_are_differentiable_away_from_ties():
    source = jnp.asarray([-1.0, 0.2, 2.0])
    target = jnp.asarray([-0.4, 0.9, 2.6])
    gradient = jax.grad(
        lambda values: phx.transport.wasserstein_distance_1d(values, target, p=2.0)
    )(source)
    sliced_gradient = jax.grad(
        lambda values: phx.transport.sliced_wasserstein_distance(
            values,
            jnp.asarray([[0.2, 0.4], [1.0, 1.3], [2.0, 2.4]]),
            projections=jnp.asarray([[1.0, 0.0], [1.0, 1.0]]),
        ).value
    )(jnp.asarray([[0.0, 0.1], [0.8, 1.0], [1.7, 2.0]]))

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(jnp.isfinite(sliced_gradient))


def test_sliced_wasserstein_reports_projection_level_structure_and_replay():
    source = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    target = source + jnp.asarray([1.0, 2.0])
    explicit = phx.transport.sliced_wasserstein_distance(
        source,
        target,
        projections=jnp.eye(2),
        p=2.0,
    )
    random = phx.transport.sliced_wasserstein_distance(
        source,
        target,
        num_projections=7,
        key=jr.key(12),
    )
    replay = phx.transport.sliced_wasserstein_distance(
        source,
        target,
        num_projections=7,
        key=jr.key(12),
    )

    assert explicit.sampling == "explicit"
    assert jnp.allclose(explicit.projection_distances, jnp.asarray([1.0, 2.0]))
    assert jnp.allclose(explicit.value, jnp.sqrt(2.5))
    assert random.sampling == "random-normal"
    assert random.projections.shape == (7, 2)
    assert jnp.array_equal(random.value, replay.value)
    assert jnp.array_equal(random.projections, replay.projections)
    with pytest.raises(ValueError, match="PRNG key"):
        phx.transport.sliced_wasserstein_distance(source, target)


def test_soft_sort_rank_and_payload_restoration_approach_hard_order():
    values = jnp.asarray([3.0, 1.0, 4.0, 2.0])
    payload = jnp.asarray([30.0, 10.0, 40.0, 20.0])
    sorted_values = phx.transport.soft_sort(values, epsilon=0.02)
    ranks = phx.transport.soft_rank(values, epsilon=0.02)
    sorted_payload = phx.transport.soft_sort_by(
        values,
        payload,
        epsilon=0.02,
    )

    assert jnp.all(jnp.diff(sorted_values) > 0.0)
    assert jnp.allclose(sorted_values, jnp.sort(values), atol=0.1)
    assert jnp.allclose(sorted_payload, jnp.sort(payload), atol=1.0)
    assert jnp.array_equal(jnp.argsort(ranks), jnp.argsort(values))
    assert jnp.allclose(jnp.sum(ranks), 6.0, atol=1e-8)


def test_soft_topk_masks_and_values_have_explicit_boundary_behavior():
    values = jnp.asarray([3.0, 1.0, 4.0, 2.0])
    mask = phx.transport.soft_topk_mask(values, 2, epsilon=0.02)
    top_values = phx.transport.soft_topk_values(values, 2, epsilon=0.02)

    assert jnp.all((mask >= 0.0) & (mask <= 1.0))
    assert jnp.allclose(jnp.sum(mask), 2.0, atol=1e-8)
    assert jnp.array_equal(jnp.argsort(mask)[-2:], jnp.asarray([0, 2]))
    assert jnp.allclose(top_values, jnp.asarray([3.0, 4.0]), atol=0.1)
    assert jnp.array_equal(phx.transport.soft_topk_mask(values, 0), jnp.zeros_like(values))
    assert jnp.array_equal(phx.transport.soft_topk_mask(values, 4), jnp.ones_like(values))
    assert phx.transport.soft_topk_values(values, 0).shape == (0,)
    with pytest.raises(ValueError, match=r"\[0, axis_size\]"):
        phx.transport.soft_topk_mask(values, 5)


def test_soft_quantiles_preserve_caller_order_endpoints_and_named_dimensions():
    values = jnp.asarray([[3.0, 1.0, 2.0], [5.0, -1.0, 1.0]])
    quantiles = phx.transport.soft_quantile(
        values,
        jnp.asarray([1.0, 0.5, 0.0]),
        axis=1,
        epsilon=0.1,
    )
    field = cx.Field(values, dims=("case", "sample"))
    named = phx.transport.soft_quantile(
        field,
        jnp.asarray([0.25, 0.75]),
        axis="sample",
        epsilon=0.1,
        quantile_dim="level",
    )

    assert quantiles.shape == (2, 3)
    assert jnp.array_equal(quantiles[:, 0], jnp.max(values, axis=1))
    assert jnp.array_equal(quantiles[:, -1], jnp.min(values, axis=1))
    assert jnp.allclose(quantiles[:, 1], jnp.asarray([2.0, 1.0]), atol=0.2)
    assert named.dims == ("case", "level")
    assert named.shape == (2, 2)


def test_soft_quantile_normalization_quantization_and_gradients_are_finite():
    values = jnp.asarray([3.0, 1.0, 2.0, 4.0])
    reference = jnp.asarray([-2.0, -1.0, 1.0, 2.0])
    normalized = phx.transport.soft_quantile_normalize(
        values,
        reference,
        epsilon=0.1,
    )
    quantized = phx.transport.soft_quantize(values, 2, epsilon=0.1)
    gradient = jax.grad(
        lambda candidate: jnp.sum(
            phx.transport.soft_sort(candidate, epsilon=0.1) ** 2
        )
    )(values)

    assert jnp.array_equal(jnp.argsort(normalized), jnp.argsort(values))
    assert jnp.allclose(jnp.mean(normalized), jnp.mean(reference), atol=1e-8)
    assert jnp.array_equal(jnp.argsort(quantized), jnp.argsort(values))
    assert jnp.max(quantized) - jnp.min(quantized) < jnp.max(values) - jnp.min(values)
    assert jnp.all(jnp.isfinite(gradient))


def test_soft_order_transport_exposes_solver_diagnostics_and_rejects_bad_weights():
    result = phx.transport.soft_order_transport(
        jnp.asarray([3.0, 1.0, 2.0]),
        weights=jnp.asarray([0.2, 0.3, 0.5]),
        epsilon=0.1,
    )
    assert result.converged
    assert result.problem.shape == (3, 3)
    assert jnp.allclose(result.source_marginal(), jnp.asarray([0.2, 0.3, 0.5]), atol=1e-7)

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="nonnegative"):
        invalid = phx.transport.soft_order_transport(
            jnp.asarray([3.0, 1.0, 2.0]),
            weights=jnp.asarray([0.2, -0.3, 0.5]),
        )
        jax.block_until_ready(invalid.source_potential)


@pytest.mark.parametrize(
    "operator",
    (
        lambda values: phx.transport.soft_sort(values, epsilon=0.15),
        lambda values: phx.transport.soft_rank(values, epsilon=0.15),
        lambda values: phx.transport.soft_quantile(values, 0.35, epsilon=0.15),
        lambda values: phx.transport.soft_topk_mask(values, 2, epsilon=0.15),
    ),
    ids=("sort", "rank", "quantile", "topk"),
)
def test_soft_order_operators_compose_with_forward_reverse_and_batch_transforms(
    operator,
):
    values = jnp.asarray([-1.3, 0.2, 2.1, 0.8])
    direction = jnp.asarray([0.3, -0.5, 0.2, 0.7])
    eager = operator(values)
    compiled = jax.jit(operator)(values)
    _, tangent = jax.jvp(operator, (values,), (direction,))
    jacobian = jax.jacfwd(operator)(values)
    expected_tangent = jnp.tensordot(jacobian, direction, axes=([-1], [0]))
    gradient = jax.grad(lambda candidate: jnp.sum(operator(candidate) ** 2))(values)
    batched = jax.vmap(operator)(jnp.stack((values, values + 1.0)))

    assert jnp.allclose(compiled, eager, rtol=1e-8, atol=1e-9)
    assert jnp.allclose(tangent, expected_tangent, rtol=1e-7, atol=1e-8)
    assert jnp.all(jnp.isfinite(jacobian))
    assert jnp.all(jnp.isfinite(gradient))
    assert batched.shape == (2,) + jnp.shape(eager)


def test_soft_order_has_finite_symmetric_second_derivatives_and_tie_sensitivities():
    values = jnp.asarray([-1.1, 0.4, 1.7, 3.2])
    hessian = jax.hessian(
        lambda candidate: jnp.sum(
            phx.transport.soft_sort(candidate, epsilon=0.15) ** 2
        )
    )(values)
    constant = jnp.ones((4,))
    constant_sorted = phx.transport.soft_sort(constant, epsilon=0.15)
    constant_jacobian = jax.jacfwd(
        lambda candidate: phx.transport.soft_sort(candidate, epsilon=0.15)
    )(constant)
    tied = jnp.asarray([1.0, 1.0, 2.0, 4.0])
    tied_ranks = phx.transport.soft_rank(tied, epsilon=0.15)
    tied_gradient = jax.grad(
        lambda candidate: jnp.sum(
            phx.transport.soft_rank(candidate, epsilon=0.15) ** 2
        )
    )(tied)

    assert jnp.all(jnp.isfinite(hessian))
    assert jnp.allclose(hessian, hessian.T, rtol=1e-7, atol=1e-8)
    assert jnp.array_equal(constant_sorted, constant)
    assert jnp.all(jnp.isfinite(constant_jacobian))
    assert jnp.allclose(constant_jacobian, constant_jacobian[0])
    assert tied_ranks[0] == tied_ranks[1]
    assert jnp.allclose(tied_gradient[0], tied_gradient[1], atol=1e-10)


def test_soft_order_preserves_order_invariances_and_coupling_mass():
    values = jnp.asarray([3.0, -1.0, 2.0, 0.5])
    permutation = jnp.asarray([2, 0, 3, 1])
    ordered = phx.transport.soft_sort(values, epsilon=0.15)
    ranks = phx.transport.soft_rank(values, epsilon=0.15)
    mask = phx.transport.soft_topk_mask(values, 2, epsilon=0.15)

    assert jnp.allclose(
        phx.transport.soft_sort(values + 7.0, epsilon=0.15),
        ordered + 7.0,
        rtol=1e-9,
        atol=1e-9,
    )
    assert jnp.allclose(
        phx.transport.soft_sort(2.5 * values - 4.0, epsilon=0.15),
        2.5 * ordered - 4.0,
        rtol=1e-8,
        atol=1e-8,
    )
    assert jnp.allclose(
        phx.transport.soft_sort(-2.0 * values + 1.0, epsilon=0.15),
        (-2.0 * ordered + 1.0)[::-1],
        rtol=1e-8,
        atol=1e-8,
    )
    assert jnp.allclose(
        phx.transport.soft_sort(values[permutation], epsilon=0.15), ordered
    )
    assert jnp.allclose(
        phx.transport.soft_rank(values[permutation], epsilon=0.15),
        ranks[permutation],
    )
    assert jnp.allclose(
        phx.transport.soft_topk_mask(values[permutation], 2, epsilon=0.15),
        mask[permutation],
    )
    assert jnp.allclose(jnp.sum(ranks), 6.0, atol=1e-8)
    assert jnp.allclose(jnp.sum(mask), 2.0, atol=1e-8)

    weights = jnp.asarray([0.1, 0.2, 0.3, 0.4])
    weighted_mask = phx.transport.soft_topk_mask(
        values,
        2,
        weights=weights,
        epsilon=0.15,
    )
    assert jnp.allclose(jnp.sum(weights * weighted_mask), 0.5, atol=1e-8)


def test_soft_order_weighted_named_and_blockwise_paths_share_one_contract():
    values = jnp.asarray([[3.0, 1.0, 4.0, 2.0], [0.5, -2.0, 1.5, 3.0]])
    field = cx.Field(values, dims=("case", "sample"))
    named = phx.transport.soft_sort(field, axis="sample", epsilon=0.2)
    plain = phx.transport.soft_sort(values, axis=1, epsilon=0.2)

    assert named.dims == field.dims
    assert jnp.allclose(named.data, plain)

    vector = values[0]
    zero_weights = jnp.asarray([0.2, 0.0, 0.3, 0.5])
    zero_weight_result = phx.transport.soft_sort(
        vector,
        weights=zero_weights,
        epsilon=0.2,
    )
    changed_inert_atom = phx.transport.soft_sort(
        vector.at[1].set(1e6),
        weights=zero_weights,
        epsilon=0.2,
    )
    assert jnp.allclose(zero_weight_result, changed_inert_atom)

    zero_weight_gradient = jax.grad(
        lambda candidate: jnp.dot(
            phx.transport.soft_topk_mask(
                candidate,
                2,
                weights=zero_weights,
                epsilon=0.2,
            ),
            jnp.arange(4.0),
        )
    )(vector)
    assert jnp.all(jnp.isfinite(zero_weight_gradient))
    assert zero_weight_gradient[1] == 0.0

    dense_solver = phx.transport.Sinkhorn(
        0.2,
        max_iterations=300,
        tolerance=1e-7,
        check_every=5,
    )
    block_solver = phx.transport.Sinkhorn(
        0.2,
        max_iterations=300,
        tolerance=1e-7,
        check_every=5,
        block_size=2,
    )
    dense = phx.transport.soft_sort(vector, weights=zero_weights, solver=dense_solver)
    block = phx.transport.soft_sort(vector, weights=zero_weights, solver=block_solver)
    assert jnp.allclose(block, dense, rtol=1e-8, atol=1e-8)

    positive_weights = jnp.asarray([0.2, 0.1, 0.3, 0.4])
    weight_gradient = jax.grad(
        lambda candidate_weights: jnp.sum(
            phx.transport.soft_sort(
                vector,
                weights=candidate_weights,
                solver=dense_solver,
            )
            ** 2
        )
    )(positive_weights)
    assert jnp.all(jnp.isfinite(weight_gradient))


def test_soft_order_provenance_and_explicit_solver_precedence_are_visible():
    solver = phx.transport.Sinkhorn(
        0.3,
        max_iterations=300,
        tolerance=1e-7,
        check_every=5,
    )
    values = jnp.asarray([3.0, 1.0, 2.0])
    result = phx.transport.soft_order_transport(
        values,
        epsilon=jnp.nan,
        solver=solver,
    )

    assert result.problem.provenance.source == (
        "soft-order-source:weighted-standardize-sigmoid"
    )
    assert result.problem.provenance.target == (
        "soft-order-target:probability-midpoints"
    )
    assert jnp.array_equal(result.epsilon, solver.epsilon)
    assert jnp.allclose(
        result.barycentric_source_to_target(values),
        phx.transport.soft_sort(values, solver=solver),
    )
