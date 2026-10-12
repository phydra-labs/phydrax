# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import PreparedTensorGrid


pytestmark = pytest.mark.strict_jax


def _grid(*, nonuniform: bool = False, periodic: bool = False) -> PreparedTensorGrid:
    axis = (
        phx.discretization.NonuniformCellAxisSpec(
            np.asarray([0.0, 0.1, 0.35, 0.7, 1.0], dtype=np.float64), periodic=periodic
        )
        if nonuniform
        else phx.discretization.UniformCellAxisSpec(4, periodic=periodic)
    )
    return phx.discretization.TensorGridPlan(
        (axis, phx.discretization.UniformCellAxisSpec(3)), axis_names=("x", "y")
    ).prepare(np.asarray([[0.0, 0.0], [2.0, 3.0]], dtype=np.float64))


@pytest.mark.parametrize("nonuniform", [False, True])
def test_series_transmissibility_and_diffusion_equivalence(nonuniform: bool) -> None:
    grid = _grid(nonuniform=nonuniform)
    plan = phx.discretization.StructuredFaceConductancePlan(
        grid, np.ones(grid.shape, dtype=np.bool_)
    )
    coefficient = jnp.broadcast_to(jnp.asarray([1.0, 1.0, 9.0, 9.0])[:, None], grid.shape)
    lower = np.asarray(plan.relation.source_indices)
    upper = np.asarray(plan.relation.target_indices)
    widths = np.asarray(grid.structured_axes[0].interval_widths)
    expected = []
    for a, b in zip(lower, upper):
        left, right = np.unravel_index(a, grid.shape), np.unravel_index(b, grid.shape)
        if left[0] != right[0]:
            expected.append(
                1.0
                / (
                    widths[left[0]] / (2 * float(coefficient[left]))
                    + widths[right[0]] / (2 * float(coefficient[right]))
                )
            )
        else:
            expected.append(float(coefficient[left]) * widths[left[0]])
    conductances = plan.conductances(coefficient)
    np.testing.assert_allclose(conductances, expected, rtol=1e-12)
    u = jnp.arange(12, dtype=jnp.float64).reshape(grid.shape) ** 2
    flux = conductances * (u.ravel()[upper] - u.ravel()[lower])
    graph_action = (
        jnp.zeros(12, dtype=jnp.float64).at[lower].add(flux).at[upper].add(-flux)
    )
    operator = phx.discretization.ConservativeDiffusionPlan(grid).prepare(coefficient)
    np.testing.assert_allclose(
        grid.quadrature_weights * operator.mv(u),
        graph_action.reshape(grid.shape),
        rtol=1e-12,
        atol=1e-11,
    )
    np.testing.assert_allclose(plan.conductances(2.0), 2 * plan.geometry, rtol=1e-12)


def test_mask_periodic_diagonal_and_node_refusals() -> None:
    grid = _grid(periodic=True)
    mask = np.ones(grid.shape, dtype=np.bool_)
    mask[1, 1] = False
    plan = phx.discretization.StructuredFaceConductancePlan(grid, mask)
    full = phx.topology.grid_adjacency_relation(grid.shape, (True, False))
    keep = (
        mask.ravel()[np.asarray(full.source_indices)]
        & mask.ravel()[np.asarray(full.target_indices)]
    )
    assert plan.relation.source_indices.size == np.count_nonzero(keep)
    nodes = plan.node_indices(np.asarray([[0, 0], [3, 0]], dtype=np.int32))
    pairs = set(
        zip(
            np.asarray(plan.relation.source_indices),
            np.asarray(plan.relation.target_indices),
        )
    )
    assert tuple(nodes) in pairs
    diagonal = jnp.broadcast_to(jnp.asarray([2.0, 7.0]), grid.shape + (2,))
    lower = np.asarray(full.source_indices)[keep]
    upper = np.asarray(full.target_indices)[keep]
    axes = np.argmax(
        np.stack(np.unravel_index(lower, grid.shape), -1)
        != np.stack(np.unravel_index(upper, grid.shape), -1),
        axis=1,
    )
    np.testing.assert_allclose(
        plan.conductances(diagonal), np.asarray([2.0, 7.0])[axes] * plan.geometry
    )
    for cells in ([[1, 1]], [[4, 0]], [[-1, 0]], [[0]], [[0.2, 1.0]]):
        with pytest.raises(ValueError):
            plan.node_indices(np.asarray(cells))
    for selector in ("upwind", "callable"):
        with pytest.raises(ValueError, match="arithmetic or harmonic"):
            phx.discretization.StructuredFaceConductancePlan(
                grid, mask, interpolation=selector
            )
    with pytest.raises(ValueError, match="full tensors"):
        plan.conductances(jnp.ones(grid.shape + (2, 2)))
    small = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(2, periodic=True),), axis_names=("x",)
    ).prepare(np.asarray([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="three cells"):
        phx.discretization.StructuredFaceConductancePlan(
            small, np.asarray([True, True], dtype=np.bool_)
        )


def test_conductance_derivatives_and_identity() -> None:
    grid = _grid(nonuniform=True, periodic=True)
    mask = np.ones(grid.shape, dtype=np.bool_)
    plan = phx.discretization.StructuredFaceConductancePlan(grid, mask)
    assert (
        plan.plan_id
        == phx.discretization.StructuredFaceConductancePlan(grid, mask).plan_id
    )
    coefficient = jnp.arange(12, dtype=jnp.float64).reshape(grid.shape) + 1

    def objective(value: jax.Array) -> jax.Array:
        return jnp.sum(plan.conductances(value) ** 2)

    gradient = jax.grad(objective)(coefficient)
    direction = jnp.cos(coefficient)
    step = 1e-5
    difference = (
        objective(coefficient + step * direction)
        - objective(coefficient - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(jnp.sum(gradient * direction), difference, rtol=1e-8)
    np.testing.assert_array_equal(plan.conductances(jnp.zeros(grid.shape)), 0.0)


def test_nonuniform_periodic_wrap_series_and_arithmetic() -> None:
    grid = _grid(nonuniform=True, periodic=True)
    mask = np.ones(grid.shape, dtype=np.bool_)
    harmonic = phx.discretization.StructuredFaceConductancePlan(grid, mask)
    arithmetic = phx.discretization.StructuredFaceConductancePlan(
        grid, mask, interpolation="arithmetic"
    )
    coefficient = jnp.broadcast_to(jnp.asarray([1.0, 2.0, 4.0, 8.0])[:, None], grid.shape)
    widths = np.asarray(grid.structured_axes[0].interval_widths)
    expected = []
    arithmetic_expected = []
    for a, b in zip(
        np.asarray(harmonic.relation.source_indices),
        np.asarray(harmonic.relation.target_indices),
    ):
        left = np.unravel_index(a, grid.shape)
        right = np.unravel_index(b, grid.shape)
        kl, kr = float(coefficient[left]), float(coefficient[right])
        if left[0] != right[0]:
            expected.append(
                1 / (widths[left[0]] / (2 * kl) + widths[right[0]] / (2 * kr))
            )
            arithmetic_expected.append((kl + kr) / (widths[left[0]] + widths[right[0]]))
        else:
            expected.append(kl * widths[left[0]])
            arithmetic_expected.append(kl * widths[left[0]])
    np.testing.assert_allclose(harmonic.conductances(coefficient), expected, rtol=1e-12)
    np.testing.assert_allclose(
        arithmetic.conductances(coefficient), arithmetic_expected, rtol=1e-12
    )
    empty = phx.discretization.StructuredFaceConductancePlan(
        grid, np.zeros(grid.shape, dtype=np.bool_)
    )
    assert empty.conductances(coefficient).shape == (0,)
