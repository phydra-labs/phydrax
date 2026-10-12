#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.discretization import PreparedTensorGrid
from phydrax.discretization.finite_difference import _multigrid
from phydrax.linalg import ArraySpace


def _cell_grid(points: Any, *, dimension: Any = 1) -> Any:
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(points) for _ in range(dimension)),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))


def _field_space(grid: PreparedTensorGrid, name: str) -> ArraySpace:
    space = grid.field_space(name).vector_space
    assert isinstance(space, ArraySpace)
    return space


def test_structured_multigrid_scenario_1() -> None:
    fine = _cell_grid(16, dimension=2)
    coarse = _cell_grid(8, dimension=2)
    transfer = phx.discretization.StructuredTransferPlan(fine, coarse)
    fine_space = fine.field_space("fine").vector_space
    coarse_space = coarse.field_space("coarse").vector_space
    restriction, prolongation = transfer.prepare(fine_space, coarse_space)
    field = jnp.arange(fine.size, dtype="float64").reshape(fine.shape)

    restricted = restriction.mv(field)
    prolonged_constant = prolongation.mv(jnp.ones(coarse.shape))

    assert transfer.report.passed
    # ty: ignore[unsupported-operator]
    assert transfer.report.conservation_residual < 1e-12
    np.testing.assert_allclose(
        jnp.sum(fine.quadrature_weights * field),
        jnp.sum(coarse.quadrature_weights * restricted),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(prolonged_constant, 1.0, rtol=0.0, atol=1e-14)
    fine = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(17),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    coarse = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(9),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    transfer = phx.discretization.StructuredTransferPlan(fine, coarse)
    restriction, prolongation = transfer.prepare(
        # ty: ignore[invalid-argument-type]
        fine.field_space("fine").vector_space,
        # ty: ignore[invalid-argument-type]
        coarse.field_space("coarse").vector_space,
    )
    fine_linear = 2.0 * fine.axes[0].nodes - 0.3
    coarse_linear = 2.0 * coarse.axes[0].nodes - 0.3

    np.testing.assert_allclose(
        restriction.mv(fine_linear),
        coarse_linear,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        prolongation.mv(coarse_linear),
        fine_linear,
        rtol=0.0,
        atol=2e-14,
    )
    factors = []
    for points in (32, 64, 128):
        multigrid = _prepared_multigrid(points)
        grid = multigrid.grids[0]
        exact = jnp.sin(jnp.pi * grid.axes[0].nodes)
        rhs = multigrid.level_operators[0].mv(exact)

        result = multigrid.solve(rhs, cycles=7, tolerance=1e-7)
        factors.append(
            float((result.residual_norms[-1] / result.residual_norms[0]) ** (1 / 7))
        )

        assert result.residual_norms[-1] < 2e-5 * result.residual_norms[0]
        np.testing.assert_allclose(result.value, exact, rtol=2e-6, atol=2e-7)

    assert max(factors) - min(factors) < 0.04
    assert max(factors) < 0.2


def _prepared_multigrid(
    points: Any, *, dimension: Any = 1, coefficient: Any = 1.0
) -> Any:
    grid = _cell_grid(points, dimension=dimension)
    boundaries = {axis: ("dirichlet", "dirichlet") for axis in grid.axis_names}
    diffusion = phx.discretization.ConservativeDiffusionPlan(
        grid,
        # ty: ignore[invalid-argument-type]
        boundaries=boundaries,
    ).prepare(coefficient)
    return phx.discretization.StructuredMultigridPlan(
        diffusion,
        minimum_coarse_points=4,
        pre_smoothing=2,
        post_smoothing=2,
    ).prepare()


def test_structured_multigrid_scenario_2() -> None:
    grid = _cell_grid(32, dimension=2)
    x = grid.axes[0].nodes[:, None]
    coefficient = jnp.where(x < 0.5, 1.0, 20.0)
    coefficient = jnp.broadcast_to(coefficient, grid.shape)
    boundaries = {axis: ("dirichlet", "dirichlet") for axis in grid.axis_names}
    diffusion = phx.discretization.ConservativeDiffusionPlan(
        grid,
        # ty: ignore[invalid-argument-type]
        boundaries=boundaries,
    ).prepare(coefficient)
    multigrid = phx.discretization.StructuredMultigridPlan(
        diffusion,
        minimum_coarse_points=4,
        cycle_kind="w",
    ).prepare()
    state = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * grid.axes[1].nodes[None, :])
    rhs = multigrid.level_operators[0].mv(state)
    apply = eqx.filter_jit(multigrid.apply)

    correction = apply(rhs)
    before = jnp.linalg.norm(rhs)
    after = jnp.linalg.norm(rhs - multigrid.level_operators[0].mv(correction))

    assert len(multigrid.grids) >= 3
    assert after < 0.35 * before
    grid = _cell_grid(64)
    diffusion = phx.discretization.ConservativeDiffusionPlan(grid).prepare(1.0)
    multigrid = phx.discretization.StructuredMultigridPlan(
        diffusion,
        minimum_coarse_points=4,
    ).prepare()
    x = grid.axes[0].nodes
    state = jnp.cos(2.0 * jnp.pi * x)
    rhs = multigrid.level_operators[0].mv(state)

    result = multigrid.solve(rhs, cycles=10, tolerance=1e-7)
    residual = rhs - multigrid.level_operators[0].mv(result.value)

    assert jnp.linalg.norm(residual) < 1e-6 * jnp.linalg.norm(rhs)
    np.testing.assert_allclose(jnp.mean(result.value), 0.0, atol=2e-10)


@pytest.mark.parametrize("dimension", [2, 3])
def test_sparse_transfer_dense_reference_and_transpose(dimension: int) -> None:
    """Unequal axis ratios exercise tensor order and conservative routes."""
    fine_counts = (12, 10, 8)[:dimension]
    coarse_counts = (4, 5, 4)[:dimension]
    bounds = jnp.asarray(
        [[0.0, -1.0, 0.0][:dimension], [2.0, 3.0, 1.0][:dimension]], dtype=jnp.float64
    )
    fine = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(n) for n in fine_counts),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(bounds)
    coarse = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(n) for n in coarse_counts),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(bounds)
    plan = phx.discretization.StructuredTransferPlan(fine, coarse)
    restriction, _ = plan.prepare(
        _field_space(fine, "fine"), _field_space(coarse, "coarse")
    )
    axis_x = np.zeros((4, 12), dtype=np.float64)
    axis_y = np.zeros((5, 10), dtype=np.float64)
    for row in range(4):
        axis_x[row, 3 * row : 3 * row + 3] = 1.0 / 3.0
    for row in range(5):
        axis_y[row, 2 * row : 2 * row + 2] = 0.5
    dense = np.kron(axis_x, axis_y)
    if dimension == 3:
        axis_z = np.zeros((4, 8), dtype=np.float64)
        for row in range(4):
            axis_z[row, 2 * row : 2 * row + 2] = 0.5
        dense = np.kron(dense, axis_z)
    field = np.arange(fine.size, dtype=np.float64).reshape(fine.shape)
    dual = np.arange(coarse.size, dtype=np.float64).reshape(coarse.shape)
    np.testing.assert_allclose(
        restriction.mv(field).reshape(-1), dense @ field.reshape(-1), atol=1e-12
    )
    np.testing.assert_allclose(
        restriction.transpose_mv(dual).reshape(-1), dense.T @ dual.reshape(-1), atol=1e-12
    )
    expected = np.max(
        np.abs(
            np.asarray(coarse.quadrature_weights).reshape(-1) @ dense
            - np.asarray(fine.quadrature_weights).reshape(-1)
        )
    )
    assert plan.report.conservation_residual is not None
    np.testing.assert_allclose(plan.report.conservation_residual, expected, atol=1e-14)
    assert sum(route.weights.size for route in plan.restriction_routes) <= sum(
        fine_counts
    ) + sum(coarse_counts)


@pytest.mark.parametrize("fault", ["wrong_sign", "dropped_boundary"])
def test_transfer_certificate_rejects_fault(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    original = _multigrid._cell_restriction

    def faulty(fine_axis: Any, coarse_axis: Any, precision: Any, /) -> Any:
        route = original(fine_axis, coarse_axis, precision)
        weights = route.weights.at[0].set(
            -route.weights[0] if fault == "wrong_sign" else 0.0
        )
        return eqx.tree_at(lambda value: value.weights, route, weights)

    monkeypatch.setattr(_multigrid, "_cell_restriction", faulty)
    with pytest.raises(RuntimeError, match="constant/conservation"):
        phx.discretization.StructuredTransferPlan(_cell_grid(12), _cell_grid(4))


def test_nonuniform_cell_transfer_independent_overlap_reference() -> None:
    fine_edges = np.asarray(
        [0.0, 0.08, 0.25, 0.42, 0.6, 0.73, 0.9, 1.0], dtype=np.float64
    )
    coarse_edges = np.asarray([0.0, 0.17, 0.45, 0.8, 1.0], dtype=np.float64)
    fine = phx.discretization.TensorGridPlan(
        (phx.discretization.NonuniformCellAxisSpec(fine_edges),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]], dtype=jnp.float64))
    coarse = phx.discretization.TensorGridPlan(
        (phx.discretization.NonuniformCellAxisSpec(coarse_edges),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]], dtype=jnp.float64))
    reference = (
        np.maximum(
            0.0,
            np.minimum(coarse_edges[1:, None], fine_edges[None, 1:])
            - np.maximum(coarse_edges[:-1, None], fine_edges[None, :-1]),
        )
        / np.diff(coarse_edges)[:, None]
    )
    plan = phx.discretization.StructuredTransferPlan(fine, coarse)
    restriction, _ = plan.prepare(
        _field_space(fine, "fine"), _field_space(coarse, "coarse")
    )
    values = np.asarray([0.2, -0.5, 1.4, 0.9, -2.0, 0.7, 0.3], dtype=np.float64)
    np.testing.assert_allclose(restriction.mv(values), reference @ values, atol=1e-14)
    np.testing.assert_allclose(
        np.diff(coarse_edges) @ reference, np.diff(fine_edges), atol=1e-14
    )
    assert plan.report.passed


def test_nonuniform_mixed_axis_tensor_certificate_matches_dense() -> None:
    fine_edges = np.asarray([0.0, 0.1, 0.3, 0.55, 0.8, 1.0], dtype=np.float64)
    coarse_edges = np.asarray([0.0, 0.2, 0.45, 0.7, 1.0], dtype=np.float64)
    bounds = jnp.asarray([[0.0, -1.0], [1.0, 2.0]], dtype=jnp.float64)
    fine = phx.discretization.TensorGridPlan(
        (
            phx.discretization.NonuniformCellAxisSpec(fine_edges),
            phx.discretization.UniformCellAxisSpec(8),
        ),
        axis_names=("x", "y"),
    ).prepare(bounds)
    coarse = phx.discretization.TensorGridPlan(
        (
            phx.discretization.NonuniformCellAxisSpec(coarse_edges),
            phx.discretization.UniformCellAxisSpec(4),
        ),
        axis_names=("x", "y"),
    ).prepare(bounds)
    x_reference = (
        np.maximum(
            0.0,
            np.minimum(coarse_edges[1:, None], fine_edges[None, 1:])
            - np.maximum(coarse_edges[:-1, None], fine_edges[None, :-1]),
        )
        / np.diff(coarse_edges)[:, None]
    )
    y_reference = np.zeros((4, 8), dtype=np.float64)
    for row in range(4):
        y_reference[row, 2 * row : 2 * row + 2] = 0.5
    reference = np.kron(x_reference, y_reference)
    plan = phx.discretization.StructuredTransferPlan(fine, coarse)
    values = np.sin(np.arange(fine.size, dtype=np.float64)).reshape(fine.shape)
    np.testing.assert_allclose(
        plan.restrict_array(values).reshape(-1),
        reference @ values.reshape(-1),
        atol=1e-14,
    )
    dense_defect = np.max(
        np.abs(
            np.asarray(coarse.quadrature_weights).reshape(-1) @ reference
            - np.asarray(fine.quadrature_weights).reshape(-1)
        )
    )
    assert plan.report.conservation_residual is not None
    np.testing.assert_allclose(
        plan.report.conservation_residual, dense_defect, atol=1e-14
    )


def test_transfer_operator_identity_matches_dense_factor_identity() -> None:
    fine = _cell_grid(16)
    coarse = _cell_grid(8)
    plan = phx.discretization.StructuredTransferPlan(fine, coarse)
    restriction, _ = plan.prepare(
        _field_space(fine, "fine"), _field_space(coarse, "coarse")
    )
    reference = np.zeros((8, 16), dtype=np.float64)
    for row in range(8):
        reference[row, 2 * row : 2 * row + 2] = 0.5
    expected = canonical_fingerprint(
        {
            "kind": "structured-tensor-transfer",
            "transfer_kind": "restriction",
            "source": restriction.source.space_id,
            "target": restriction.target.space_id,
            "matrices": [array_tree_fingerprint(reference)],
            "precision": plan.precision.policy_id,
        }
    )
    assert restriction.operator_id == expected
