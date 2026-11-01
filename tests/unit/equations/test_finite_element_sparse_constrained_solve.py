#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse elimination of strong Dirichlet constraints and its native solve route."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_mesh import CellBlock, CellMesh


la = phx.linalg
_SPD = la.OperatorProperties(
    self_adjoint=True,
    positive_definite=True,
    evidence={"positive_definite": "asserted"},
)


def _triangles(count: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, count + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=-1)
    index = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)
    lower = index[:-1, :-1].ravel()
    cells = np.concatenate(
        (
            np.stack((lower, lower + count + 1, lower + count + 2), axis=-1),
            np.stack((lower, lower + count + 2, lower + 1), axis=-1),
        )
    )
    return points, cells.astype(np.int32)


def _mesh(points: np.ndarray, cells: np.ndarray, kind: str) -> Any:
    if kind == "triangle":
        return CellMesh.from_triangles(jnp.asarray(points), jnp.asarray(cells))
    return CellMesh(
        points,
        (CellBlock("cells", kind, cells, global_ids=tuple(range(cells.shape[0]))),),
    )


def _annulus(parameters: np.ndarray) -> np.ndarray:
    radius = 1.0 + parameters[..., 0]
    angle = 0.5 * np.pi * parameters[..., 1]
    return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)


def _curved_p2_space(kind: str) -> Any:
    """P2 field on a quarter annulus whose P2 coordinate map interpolates the chart."""
    points, cells = _triangles(3)
    nodes = phx.discretization.FiniteElementDofMap(
        _mesh(points, cells, kind), (phx.discretization.lagrange_element(kind, 2),)
    )
    mesh = _mesh(_annulus(points), cells, kind)
    geometry = phx.discretization.CellGeometrySpec(
        {mesh.blocks[0].name: phx.discretization.lagrange_element(kind, 2)},
        {mesh.blocks[0].name: np.asarray(nodes.cell_dofs[0])},
        _annulus(np.asarray(nodes.dof_coordinates)),
    )
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element(kind, 2)
        ),
        coordinate_spec=geometry,
    ).prepare()


def _dirichlet_laplace(space: Any, values: Any, *, properties: Any = _SPD) -> Any:
    return phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "dirichlet-laplace",
            "u",
            (phx.equations.DiffusionAction("u", 1.0),),
            properties=properties,
        ),
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=values,
    )


_NATIVE_CHOLESKY = la.LinearSolvePolicy(
    la.PCG(),
    tolerance=la.TolerancePolicy(relative=1.0e-14, absolute=0.0, max_steps=10),
    preconditioning=la.PreconditioningPolicy(
        la.SparseFactorizationPreconditionerBuilder(
            la.SparseFactorizationPolicy("cholesky", ordering="nested-dissection")
        )
    ),
)


def _factorization(prepared: Any) -> Any:
    """The native sparse factor retained by a prepared preconditioned solve."""
    state = prepared.preconditioning_state
    assert state is not None
    action = state.action
    assert isinstance(action, la.SparseFactorizationPreconditioner)
    return action.factorization


def test_curved_p2_dirichlet_laplace_reproduces_x_through_the_sparse_route() -> None:
    space = _curved_p2_space("triangle")
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    problem = _dirichlet_laplace(space, lambda value: value[..., 0])
    system, rhs = problem.linear_system(storage="sparse")
    action_system, action_rhs = problem.linear_system()

    operator = system.operator
    free = int(problem.state_space.size)
    assert isinstance(operator, phx.sparse.SparseCoordinateOperator)
    assert operator.source.compatible(problem.state_space)
    assert operator.properties.certifies("positive_definite")
    # The eliminated block is the action operator exactly; Dirichlet columns
    # live in the right-hand side through the lift.
    identity = np.eye(free)
    sparse_columns = np.stack([np.asarray(operator.mv(column)) for column in identity])
    action_columns = np.stack(
        [np.asarray(action_system.operator.mv(column)) for column in identity]
    )
    np.testing.assert_allclose(sparse_columns, action_columns, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(np.asarray(rhs), np.asarray(action_rhs))
    csr = problem.to_scipy_csr()
    assert csr.shape == (free, free)
    assert csr.nnz == operator.sparse_storage().nnz < free * free

    prepared = la.prepare(system, _NATIVE_CHOLESKY)
    factorization = _factorization(prepared)
    assert int(factorization.status) == int(la.SparseFactorizationStatus.SUCCESS)
    assert float(factorization.diagnostics.minimum_pivot) > 0.0
    result = la.solve(prepared, rhs)
    assert bool(result.successful)
    assert float(result.diagnostics.relative_residual) < 1.0e-14
    assert int(result.diagnostics.iterations) <= 1
    np.testing.assert_allclose(
        np.asarray(problem.expand(result.value)), points[:, 0], rtol=0.0, atol=1.0e-13
    )
    condition = la.estimate_condition_number(operator, max_dimension=free)
    assert 1.0 < float(condition.lower_bound) <= float(condition.value) < 1.0e4

    # One factorization serves every lift: u = y and u = x - 2y share it.
    context = phx.equations.FiniteElementExecutionContext
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    exact = (points[:, 1], points[:, 0] - 2.0 * points[:, 1])
    contexts = tuple(
        context(space.default_runtime, lift=constraint.lift(jnp.asarray(value)))
        for value in exact
    )
    columns = jnp.stack(
        tuple(problem.right_hand_side(item) for item in contexts), axis=-1
    )
    many = la.solve_many(prepared, columns)
    assert bool(jnp.all(many.successful))
    for index, (item, value) in enumerate(zip(contexts, exact, strict=True)):
        np.testing.assert_allclose(
            np.asarray(problem.expand(many.value[:, index], item)),
            value,
            rtol=0.0,
            atol=1.0e-13,
        )


def test_sparse_route_without_definiteness_claims_uses_native_lu() -> None:
    space = _curved_p2_space("simplex:2")
    problem = _dirichlet_laplace(
        space, lambda value: value[..., 0], properties=la.OperatorProperties()
    )
    system, rhs = problem.linear_system(storage="sparse")
    assert not system.operator.properties.self_adjoint
    policy = la.LinearSolvePolicy(
        la.GMRES(),
        tolerance=la.TolerancePolicy(relative=1.0e-14, absolute=0.0, max_steps=10),
        preconditioning=la.PreconditioningPolicy(
            la.SparseFactorizationPreconditionerBuilder(
                la.SparseFactorizationPolicy("lu", ordering="nested-dissection")
            )
        ),
    )
    result = la.solve(system, rhs, policy=policy)
    assert bool(result.successful)
    np.testing.assert_allclose(
        np.asarray(problem.expand(result.value)),
        np.asarray(space.dof_maps[0].dof_coordinates)[:, 0],
        rtol=0.0,
        atol=1.0e-13,
    )


def test_sparse_route_refuses_without_structural_sparse_storage() -> None:
    space = _curved_p2_space("triangle")
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "matrix-free", "u", (phx.equations.DiffusionAction("u", 1.0),)
        ),
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=0.0,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free"
        ),
    )
    with pytest.raises(ValueError, match="structural sparse actions"):
        problem.linear_system(storage="sparse")
    with pytest.raises(ValueError, match="storage must be"):
        problem.linear_system(storage="dense")  # ty: ignore[invalid-argument-type]


def test_constraint_map_elimination_refuses_non_selection_prolongation() -> None:
    full = la.ArraySpace((3,))
    reduced = la.ArraySpace((2,))
    prolongation = phx.sparse.SparseCoordinateOperator(
        phx.sparse.EdgeRelation(
            np.asarray([0, 1, 1], dtype=np.int32),
            np.asarray([0, 1, 2], dtype=np.int32),
            source_size=2,
            target_size=3,
        ),
        jnp.ones((3,)),
        source=reduced,
        target=full,
    )
    hanging = la.ConstraintMap(full, reduced, prolongation)
    assert hanging.coordinate_selection() is None
    with pytest.raises(la.LinearCapabilityError, match="coordinate-selection"):
        hanging.reduce_sparse_operator(la.IdentityLinearOperator(full))


def test_large_dirichlet_laplace_stays_sparse() -> None:
    """A 50k-DOF P2 Dirichlet problem solves on O(nnz) storage, never densified.

    The large case uses the preconditioned-Krylov branch of the sparse route
    (IC(0) on the eliminated block); the complete native factor shares the same
    eliminated operator and is exercised on the curved case above.
    """
    mesh = _mesh(*_triangles(112), "triangle")
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 2)
        ),
    ).prepare()
    assert space.dof_maps[0].global_dof_count >= 50_000
    problem = _dirichlet_laplace(space, lambda value: value[..., 0] + 2.0 * value[..., 1])
    system, rhs = problem.linear_system(storage="sparse")
    size = system.operator.source.size
    nnz = system.operator.sparse_storage().nnz
    assert size >= 49_000
    assert nnz < 20 * size
    policy = la.LinearSolvePolicy(
        la.PCG(),
        tolerance=la.TolerancePolicy(relative=1.0e-12, absolute=0.0, max_steps=2000),
        preconditioning=la.PreconditioningPolicy(
            la.IncompleteCholeskyPreconditionerBuilder(ordering="reverse-cuthill-mckee")
        ),
    )
    prepared = la.prepare(system, policy)
    factorization = _factorization(prepared)
    assert int(factorization.status) == int(la.SparseFactorizationStatus.SUCCESS)
    assert factorization.plan.factor_nnz <= nnz
    result = la.solve(prepared, rhs)
    assert bool(result.successful)
    assert float(result.diagnostics.relative_residual) < 1.0e-12
    dofs = np.asarray(space.dof_maps[0].dof_coordinates)
    np.testing.assert_allclose(
        np.asarray(problem.expand(result.value)),
        dofs[:, 0] + 2.0 * dofs[:, 1],
        rtol=0.0,
        atol=1.0e-8,
    )


@pytest.mark.parametrize("kind", ("triangle", "simplex:2"))
def test_per_cell_exact_charts_prepare_and_solve(kind: str) -> None:
    """Adapted curved results carry per-cell exact charts; FE prepares and solves."""
    from phydrax.discretization._cell_geometry import (
        PerCellRestrictedCellGeometryElement,
    )
    from phydrax.discretization._coordinate_enclosure import axes

    shared = _curved_p2_space(kind)
    block = shared.mesh.blocks[0]
    cells = tuple(int(value) for value in np.asarray(block.global_ids).tolist())
    p2 = phx.discretization.lagrange_element(kind, 2)
    restricted = PerCellRestrictedCellGeometryElement(
        p2, kind, cells, cells, (axes(2),) * len(cells)
    )
    geometry = phx.discretization.CellGeometrySpec(
        {block.name: restricted},
        {block.name: np.asarray(shared.dof_maps[0].cell_dofs[0])},
        np.asarray(shared.dof_maps[0].dof_coordinates),
    )
    space = phx.discretization.FiniteElementPlan(
        shared.mesh,
        phx.discretization.FiniteElementFieldSpec("u", p2),
        coordinate_spec=geometry,
    ).prepare()
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    np.testing.assert_allclose(
        points, np.asarray(shared.dof_maps[0].dof_coordinates), rtol=0.0, atol=1e-14
    )
    problem = _dirichlet_laplace(space, lambda value: value[..., 0])
    system, rhs = problem.linear_system(storage="sparse")
    result = la.solve(system, rhs, policy=_NATIVE_CHOLESKY)
    assert bool(result.successful)
    np.testing.assert_allclose(
        np.asarray(problem.expand(result.value)), points[:, 0], rtol=0.0, atol=1e-13
    )
    action_system, action_rhs = problem.linear_system()
    np.testing.assert_allclose(
        np.asarray(action_system.operator.mv(result.value)),
        np.asarray(action_rhs),
        rtol=0.0,
        atol=1e-12,
    )
