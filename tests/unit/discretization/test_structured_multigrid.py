#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _cell_grid(points, *, dimension=1):
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(points) for _ in range(dimension)),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))


def test_cell_restriction_is_conservative_and_both_transfers_preserve_constants():
    fine = _cell_grid(16, dimension=2)
    coarse = _cell_grid(8, dimension=2)
    transfer = phx.discretization.StructuredTransferPlan(fine, coarse)
    fine_space = fine.field_space("fine").vector_space
    coarse_space = coarse.field_space("coarse").vector_space
    restriction, prolongation = transfer.prepare(fine_space, coarse_space)
    field = jnp.arange(fine.size, dtype=float).reshape(fine.shape)

    restricted = restriction.mv(field)
    prolonged_constant = prolongation.mv(jnp.ones(coarse.shape))

    assert transfer.report.passed
    assert transfer.report.conservation_residual < 1e-12
    np.testing.assert_allclose(
        jnp.sum(fine.quadrature_weights * field),
        jnp.sum(coarse.quadrature_weights * restricted),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(prolonged_constant, 1.0, rtol=0.0, atol=1e-14)


def test_nodal_transfer_injects_nested_nodes_and_linearly_interpolates():
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
        fine.field_space("fine").vector_space,
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


def _prepared_multigrid(points, *, dimension=1, coefficient=1.0):
    grid = _cell_grid(points, dimension=dimension)
    boundaries = {
        axis: ("dirichlet", "dirichlet") for axis in grid.axis_names
    }
    diffusion = phx.discretization.ConservativeDiffusionPlan(
        grid,
        boundaries=boundaries,
    ).prepare(coefficient)
    return phx.discretization.StructuredMultigridPlan(
        diffusion,
        minimum_coarse_points=4,
        pre_smoothing=2,
        post_smoothing=2,
    ).prepare()


def test_structured_v_cycle_has_resolution_independent_convergence_factor():
    factors = []
    for points in (32, 64, 128):
        multigrid = _prepared_multigrid(points)
        grid = multigrid.grids[0]
        exact = jnp.sin(jnp.pi * grid.axes[0].nodes)
        rhs = multigrid.level_operators[0].mv(exact)

        result = multigrid.solve(rhs, cycles=7, tolerance=1e-7)
        factors.append(float((result.residual_norms[-1] / result.residual_norms[0]) ** (1 / 7)))

        assert result.residual_norms[-1] < 2e-5 * result.residual_norms[0]
        np.testing.assert_allclose(result.value, exact, rtol=2e-6, atol=2e-7)

    assert max(factors) - min(factors) < 0.04
    assert max(factors) < 0.2


def test_variable_coefficient_two_dimensional_hierarchy_is_jittable_and_contracts_residual():
    grid = _cell_grid(32, dimension=2)
    x = grid.axes[0].nodes[:, None]
    coefficient = jnp.where(x < 0.5, 1.0, 20.0)
    coefficient = jnp.broadcast_to(coefficient, grid.shape)
    boundaries = {axis: ("dirichlet", "dirichlet") for axis in grid.axis_names}
    diffusion = phx.discretization.ConservativeDiffusionPlan(
        grid,
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


def test_all_neumann_coarse_pseudoinverse_handles_compatible_nullspace_rhs():
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
