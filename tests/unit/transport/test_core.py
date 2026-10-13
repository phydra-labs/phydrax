#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.axes as cx


def _target(
    points: Any,
    weights: Any,
    *,
    normalized: Any = True,
    mask: Any = None,
    provenance: Any = "test-measure",
) -> Any:
    weight_field = cx.AxisArray(jnp.asarray(weights, dtype="float64"), dims=("atom",))
    mask_field = (
        None
        if mask is None
        else cx.AxisArray(jnp.asarray(mask, dtype="bool"), dims=("atom",))
    )
    point_values = points if isinstance(points, cx.AxisArray) else jnp.asarray(points)
    return phx.integration.discrete(
        point_values,
        weight_field,
        axes="atom",
        mask=mask_field,
        normalized=normalized,
        provenance=provenance,
    )


def _problem(
    source_points: Any,
    target_points: Any,
    *,
    source_weights: Any = None,
    target_weights: Any = None,
    normalized: Any = True,
    cost: Any = None,
) -> Any:
    source_count = jnp.asarray(source_points).shape[0]
    target_count = jnp.asarray(target_points).shape[0]
    if source_weights is None:
        source_weights = jnp.ones((source_count,))
    if target_weights is None:
        target_weights = jnp.ones((target_count,))
    source = _target(source_points, source_weights, normalized=normalized)
    target = _target(target_points, target_weights, normalized=normalized)
    return phx.transport.discrete_problem(
        source,
        target,
        cost=phx.transport.SquaredEuclideanCost() if cost is None else cost,
    )


def _solver(
    *, block_size: Any = None, max_iterations: Any = 1000, tolerance: Any = 1e-10
) -> Any:
    return phx.transport.Sinkhorn(
        0.5,
        max_iterations=max_iterations,
        tolerance=tolerance,
        check_every=5,
        block_size=block_size,
        early_stop=False,
        store_history=True,
    )


def test_core_scenario_1() -> None:
    points = cx.AxisArray(
        jnp.asarray(
            [
                [0.0, 1.0, jnp.nan],
                [2.0, 3.0, jnp.nan],
            ]
        ),
        dims=("feature", "atom"),
    )
    source = _target(
        points,
        [2.0, 1.0, jnp.nan],
        normalized=False,
        mask=[True, True, False],
        provenance="source-grid",
    )
    target = _target(
        cx.AxisArray(jnp.asarray([[0.0, 1.0], [2.0, 3.0]]), dims=("feature", "atom")),
        [2.0, 1.0],
        normalized=False,
        provenance="target-grid",
    )
    problem = phx.transport.discrete_problem(
        source,
        target,
        cost=phx.transport.SquaredEuclideanCost(),
    )

    assert problem.source.event_shape == (2,)
    assert problem.source.points.shape == (3, 2)
    assert jnp.array_equal(problem.source.points[-1], jnp.zeros((2,)))
    assert jnp.allclose(problem.source.probabilities, jnp.asarray([2 / 3, 1 / 3, 0]))
    assert jnp.allclose(problem.source_weights, jnp.asarray([2.0, 1.0, 0.0]))
    assert jnp.allclose(problem.mass, 3.0)
    assert problem.provenance.source == "source-grid"
    assert problem.provenance.target == "target-grid"
    left = jnp.asarray([0.9, 2.0])
    right = jnp.asarray([0.1, 4.0])

    assert jnp.allclose(phx.transport.SquaredEuclideanCost().pairwise(left, right), 4.64)
    assert jnp.allclose(
        # ty: ignore[invalid-argument-type]
        phx.transport.WeightedSquaredEuclideanCost([0.2, 2.0]).pairwise(left, right),
        17.0,
    )
    assert jnp.allclose(
        # ty: ignore[invalid-argument-type]
        phx.transport.PeriodicSquaredEuclideanCost([1.0, 10.0]).pairwise(left, right),
        4.04,
    )
    matrix = phx.transport.SquaredEuclideanCost().matrix(
        jnp.asarray([[0.0], [2.0]]),
        jnp.asarray([[1.0], [3.0]]),
    )
    assert jnp.array_equal(matrix, jnp.asarray([[1.0, 9.0], [1.0, 1.0]]))
    problem = _problem(
        [[0.0], [1.0]],
        [[0.0], [1.0]],
        # ty: ignore[invalid-argument-type]
        cost=phx.transport.PrecomputedCost([[0.0, 1.0], [1.0, 0.0]]),
    )
    result = _solver()(problem)
    ratio = jnp.exp(1.0 / result.epsilon)
    diagonal = 0.5 * ratio / (1.0 + ratio)
    off_diagonal = 0.5 / (1.0 + ratio)
    expected = jnp.asarray([[diagonal, off_diagonal], [off_diagonal, diagonal]])

    assert result.converged
    assert result.diagnostics.normalized_marginal_residual < 1e-10
    assert jnp.allclose(result.dense_plan(), expected, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(
        result.regularized_cost, result.transport_cost + result.regularization
    )
    assert jnp.allclose(result.regularized_cost, result.dual_cost, atol=1e-10)
    assert result.provenance.execution == "dense"
    assert result.diagnostics.residual_history.ndim == 1


def test_core_scenario_2() -> None:
    problem = _problem(
        [[0.0], [1.0], [2.0]],
        [[0.5], [1.5]],
        source_weights=[1.0, 2.0, 1.0],
        target_weights=[2.0, 2.0],
        normalized=False,
    )
    result = _solver()(problem)
    plan = result.dense_plan()
    source_payload = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target_payload = jnp.asarray([[2.0], [7.0]])

    assert result.converged
    assert jnp.allclose(jnp.sum(plan, axis=1), problem.source_weights, atol=1e-9)
    assert jnp.allclose(jnp.sum(plan, axis=0), problem.target_weights, atol=1e-9)
    assert jnp.allclose(result.source_marginal(), problem.source_weights, atol=1e-9)
    assert jnp.allclose(result.target_marginal(), problem.target_weights, atol=1e-9)
    assert jnp.allclose(
        result.apply_source_to_target(source_payload), plan.T @ source_payload
    )
    assert jnp.allclose(
        result.apply_target_to_source(target_payload), plan @ target_payload
    )
    assert jnp.allclose(
        result.barycentric_source_to_target(source_payload),
        (plan.T @ source_payload) / problem.target_weights[:, None],
    )

    normalized_problem = _problem(
        [[0.0], [1.0], [2.0]],
        [[0.5], [1.5]],
        source_weights=[1.0, 2.0, 1.0],
        target_weights=[2.0, 2.0],
        normalized=True,
    )
    normalized_result = _solver()(normalized_problem)
    assert jnp.allclose(result.regularized_cost, 4.0 * normalized_result.regularized_cost)
    for block_size in [1, 2, 4, 7]:
        problem = _problem(
            jnp.linspace(-1.0, 1.0, 5)[:, None],
            jnp.linspace(-0.7, 1.4, 7)[:, None],
            source_weights=[1.0, 2.0, 3.0, 2.0, 1.0],
            target_weights=[1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0],
        )
        dense = _solver()(problem)
        blockwise = _solver(block_size=block_size)(problem)
        payload = jnp.arange(15.0).reshape((5, 3))

        assert dense.converged & blockwise.converged
        assert blockwise.provenance.execution == "blockwise"
        assert jnp.allclose(
            blockwise.regularized_cost, dense.regularized_cost, rtol=1e-9, atol=1e-9
        )
        assert jnp.allclose(
            blockwise.source_potential, dense.source_potential, rtol=1e-8, atol=1e-8
        )
        assert jnp.allclose(
            blockwise.target_potential, dense.target_potential, rtol=1e-8, atol=1e-8
        )
        assert jnp.allclose(
            blockwise.dense_plan(), dense.dense_plan(), rtol=1e-8, atol=1e-9
        )
        assert jnp.allclose(
            blockwise.apply_source_to_target(payload),
            dense.apply_source_to_target(payload),
            rtol=1e-8,
            atol=1e-9,
        )
    source = _target(jnp.asarray([[0.0], [1.0], [2.0]]), [0.2, 0.3, 0.5])
    target = _target(jnp.asarray([[0.3], [1.4], [2.4]]), [0.4, 0.2, 0.4])
    problem = phx.transport.discrete_problem(
        source,
        target,
        cost=phx.transport.SquaredEuclideanCost(),
    )
    solver = _solver(tolerance=1e-9)
    direct = phx.transport.sinkhorn_divergence(problem, solver)
    reference = phx.transport.prepare_sinkhorn_reference(
        target,
        cost=phx.transport.SquaredEuclideanCost(),
        solver=solver,
    )
    prepared = phx.transport.sinkhorn_divergence_against(source, reference)
    identity_problem = phx.transport.discrete_problem(
        source,
        source,
        cost=phx.transport.SquaredEuclideanCost(),
    )
    identity = phx.transport.sinkhorn_divergence(identity_problem, solver)

    assert direct.converged & prepared.converged & identity.converged
    assert jnp.allclose(prepared.value, direct.value, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(identity.value, 0.0, atol=1e-12)
    assert prepared.target_self is reference.target_self


def test_solver_is_permutation_invariant_jittable_and_differentiable() -> None:
    target_points = jnp.asarray([[-0.5], [0.7], [1.8]])
    source_weights = jnp.asarray([0.2, 0.3, 0.5])
    target_weights = jnp.asarray([0.4, 0.1, 0.5])
    solver = phx.transport.Sinkhorn(
        0.7,
        max_iterations=300,
        tolerance=1e-9,
        check_every=5,
        early_stop=False,
    )

    def objective(source_points: Any) -> Any:
        problem = _problem(
            source_points,
            target_points,
            source_weights=source_weights,
            target_weights=target_weights,
        )
        return solver(problem).regularized_cost

    source_points = jnp.asarray([[-1.0], [0.2], [1.1]])
    compiled = jax.jit(objective)(source_points)
    gradient = jax.grad(objective)(source_points)
    step = 1e-4
    direction = jnp.asarray([[0.3], [-0.2], [0.5]])
    finite_difference = (
        objective(source_points + step * direction)
        - objective(source_points - step * direction)
    ) / (2.0 * step)

    permutation = jnp.asarray([2, 0, 1])
    permuted = _problem(
        source_points[permutation],
        target_points,
        source_weights=source_weights[permutation],
        target_weights=target_weights,
    )
    permuted_value = solver(permuted).regularized_cost

    assert jnp.isfinite(compiled)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(
        jnp.sum(gradient * direction), finite_difference, rtol=2e-4, atol=2e-5
    )
    assert jnp.allclose(permuted_value, compiled, rtol=1e-10, atol=1e-10)


def test_nonconvergence_and_invalid_measures_remain_explicit() -> None:
    problem = _problem([[0.0], [1.0]], [[10.0], [11.0]])
    result = phx.transport.Sinkhorn(
        0.01,
        max_iterations=1,
        tolerance=0.0,
        check_every=1,
    )(problem)
    assert not result.converged
    assert result.diagnostics.status == int(
        phx.transport.TransportStatus.MAXIMUM_ITERATIONS_REACHED
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        checked = phx.transport.require_converged(result)
        assert isinstance(checked, phx.transport.SinkhornResult)
        jax.block_until_ready(checked.source_potential)

    invalid_source = _target([[0.0], [1.0]], [1.0, -1.0])
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="nonnegative"):
        invalid_problem = phx.transport.discrete_problem(
            invalid_source,
            _target([[0.0], [1.0]], [0.5, 0.5]),
            cost=phx.transport.SquaredEuclideanCost(),
        )
        jax.block_until_ready(invalid_problem.source.probabilities)

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="equal positive"):
        mismatched = phx.transport.discrete_problem(
            _target([[0.0], [1.0]], [1.0, 1.0], normalized=False),
            _target([[0.0], [1.0]], [1.0, 2.0], normalized=False),
            cost=phx.transport.SquaredEuclideanCost(),
        )
        jax.block_until_ready(mismatched.mass)


@pytest.mark.parametrize("iterations", [1, 100], ids=["iteration-limit", "converged"])
def test_budget_blocks_preserve_dense_values_gradients_and_status(
    iterations: int,
) -> None:
    points = jnp.asarray([[-0.9], [-0.2], [0.1], [0.7], [1.3]], dtype=jnp.float64)
    targets = jnp.linspace(-1.1, 1.6, 17, dtype=jnp.float64)[:, None]
    dense_solver = phx.transport.Sinkhorn(
        0.5, max_iterations=iterations, tolerance=1e-9, block_size=None
    )
    bounded_solver = phx.transport.Sinkhorn(
        0.5, max_iterations=iterations, tolerance=1e-9, sinkhorn_workspace_bytes=1632
    )
    problem = _problem(points, targets)
    dense = dense_solver(problem)
    bounded = bounded_solver(problem)
    assert bounded.provenance.execution == "blockwise"
    assert bounded.provenance.block_size == bounded.block_size == 3
    assert bounded.provenance.sinkhorn_workspace_bytes == 1632
    assert dense.provenance.execution == "dense"
    assert dense.provenance.block_size is None
    assert bounded.diagnostics.status == dense.diagnostics.status
    assert bool(bounded.converged) is (iterations == 100)
    assert jnp.allclose(bounded.dense_plan(), dense.dense_plan(), rtol=1e-10, atol=1e-12)
    assert jnp.allclose(
        bounded.regularized_cost, dense.regularized_cost, rtol=1e-10, atol=1e-12
    )
    dense_gradient = jax.grad(
        lambda values: dense_solver(_problem(values, targets)).regularized_cost
    )(points)
    bounded_gradient = jax.grad(
        lambda values: bounded_solver(_problem(values, targets)).regularized_cost
    )(points)
    assert jnp.allclose(bounded_gradient, dense_gradient, rtol=1e-9, atol=1e-11)
    payload = jnp.stack((targets[:, 0], targets[:, 0] ** 2), axis=1)
    assert jnp.allclose(
        bounded.apply_target_to_source(payload),
        dense.apply_target_to_source(payload),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "execution,block_size",
    [("dense", 2), ("blockwise", None), ("blockwise", 0), ("factorized", 2)],
)
def test_transport_provenance_refuses_inconsistent_pair_tiles(
    execution: str, block_size: int | None
) -> None:
    with pytest.raises(ValueError, match="block_size"):
        phx.transport.TransportProvenance(
            "test",
            "cost",
            execution,
            "unrolled",
            "source",
            "target",
            block_size=block_size,
        )


@pytest.mark.parametrize(
    "block_size", [None, 256, "auto"], ids=["dense", "blockwise", "automatic"]
)
def test_transport_result_persists_static_provenance_and_plan(
    block_size: Literal["auto"] | int | None,
) -> None:
    import io

    import numpy as np

    result = phx.transport.Sinkhorn(
        0.5, max_iterations=100, tolerance=1e-9, block_size=block_size
    )(_problem([[0.0], [1.0]], [[0.2], [0.8]]))
    assert result.converged
    assert jax.tree.leaves(result.provenance) == []
    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, result)
    buffer.seek(0)
    restored = eqx.tree_deserialise_leaves(buffer, result)
    assert restored.provenance == result.provenance
    assert restored.provenance.block_size == (
        None if block_size == "auto" else block_size
    )
    assert restored.provenance.sinkhorn_workspace_bytes == 128 * 1024 * 1024
    assert jax.tree.leaves(restored.provenance) == []
    assert restored.diagnostics.status == result.diagnostics.status
    np.testing.assert_array_equal(restored.dense_plan(), result.dense_plan())
    np.testing.assert_array_equal(restored.regularized_cost, result.regularized_cost)


@pytest.mark.parametrize("budget", [480, 128 * 1024 * 1024], ids=["exact-fit", "default"])
def test_automatic_full_fit_is_exactly_dense(budget: int) -> None:
    problem = _problem([[-0.9], [0.1], [1.3]], [[-1.1], [-0.2], [0.4], [0.8], [1.6]])
    automatic = phx.transport.Sinkhorn(
        0.5, max_iterations=10, sinkhorn_workspace_bytes=budget
    )(problem)
    dense = phx.transport.Sinkhorn(0.5, max_iterations=10, block_size=None)(problem)
    assert automatic.block_size is automatic.provenance.block_size is None
    assert automatic.provenance.execution == "dense"
    assert automatic.provenance.sinkhorn_workspace_bytes == budget
    assert jnp.array_equal(automatic.source_potential, dense.source_potential)
    assert jnp.array_equal(automatic.target_potential, dense.target_potential)
    assert jnp.array_equal(automatic.regularized_cost, dense.regularized_cost)
    assert jnp.array_equal(automatic.dense_plan(), dense.dense_plan())
    assert automatic.diagnostics.status == dense.diagnostics.status


@pytest.mark.parametrize(
    "block_size,budget",
    [(None, 1), (2, 1), (2, 128 * 1024 * 1024)],
    ids=["dense-over-budget", "tile-over-budget", "tile-full-fit"],
)
def test_explicit_sinkhorn_routes_override_valid_budget(
    block_size: int | None, budget: int
) -> None:
    result = phx.transport.Sinkhorn(
        0.5, max_iterations=1, block_size=block_size, sinkhorn_workspace_bytes=budget
    )(_problem([[0.0], [1.0]], [[0.2], [0.8]]))
    assert result.block_size == result.provenance.block_size == block_size
    assert result.provenance.execution == ("dense" if block_size is None else "blockwise")
    assert result.provenance.sinkhorn_workspace_bytes == budget


@pytest.mark.parametrize("family", ["balanced", "unbalanced"])
@pytest.mark.parametrize("budget", [0, -1], ids=["zero", "negative"])
@pytest.mark.parametrize("block_size", ["auto", None, 2])
def test_sinkhorn_refuses_nonpositive_budget(
    family: str, budget: int, block_size: Literal["auto"] | int | None
) -> None:
    solver = (
        phx.transport.Sinkhorn
        if family == "balanced"
        else phx.transport.UnbalancedSinkhorn
    )
    with pytest.raises(ValueError, match="sinkhorn_workspace_bytes must be positive"):
        solver(0.5, block_size=block_size, sinkhorn_workspace_bytes=budget)


@pytest.mark.parametrize("family", ["balanced", "unbalanced"])
@pytest.mark.parametrize(
    "budget", [True, 1.5, "1024", None], ids=["boolean", "float", "string", "none"]
)
def test_sinkhorn_refuses_noninteger_budget(family: str, budget: Any) -> None:
    solver = (
        phx.transport.Sinkhorn
        if family == "balanced"
        else phx.transport.UnbalancedSinkhorn
    )
    with pytest.raises(TypeError, match="sinkhorn_workspace_bytes must be an integer"):
        solver(0.5, sinkhorn_workspace_bytes=budget)


@pytest.mark.parametrize("family", ["balanced", "unbalanced"])
@pytest.mark.parametrize("block_size", [True, 1.5, "dense", 0, -2])
def test_sinkhorn_refuses_invalid_block_override(family: str, block_size: Any) -> None:
    solver = (
        phx.transport.Sinkhorn
        if family == "balanced"
        else phx.transport.UnbalancedSinkhorn
    )
    with pytest.raises((TypeError, ValueError), match="block_size"):
        solver(0.5, block_size=block_size)


def test_automatic_sinkhorn_clamps_to_one_pair_tile() -> None:
    problem = _problem([[0.0], [1.0]], [[0.2], [0.8]])
    bounded = phx.transport.Sinkhorn(0.5, max_iterations=10, sinkhorn_workspace_bytes=1)(
        problem
    )
    dense = phx.transport.Sinkhorn(0.5, max_iterations=10, block_size=None)(problem)
    assert bounded.block_size == bounded.provenance.block_size == 1
    assert jnp.allclose(bounded.dense_plan(), dense.dense_plan(), rtol=1e-10, atol=1e-12)
