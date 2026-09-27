#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import itertools
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import masked_simplex_facet_neighbors, MaskedSimplexMesh


def _oriented(points: Any, cells: Any) -> Any:
    edges = points[cells[:, 1:]] - points[cells[:, :1]]
    flip = np.linalg.det(edges) < 0.0
    cells = cells.copy()
    cells[flip, 0], cells[flip, 1] = cells[flip, 1], cells[flip, 0].copy()
    return cells


def _square(count: Any) -> Any:
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    index = np.arange((count + 1) ** 2).reshape((count + 1, count + 1))
    cells = []
    for i, j in itertools.product(range(count), repeat=2):
        a, b, c, d = index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]
        cells.extend(((a, b, c), (a, c, d)))
    return points, _oriented(points, np.asarray(cells, dtype=np.int32))


def _cube(count: Any) -> Any:
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    index = np.arange((count + 1) ** 3).reshape((count + 1,) * 3)
    cells = []
    for corner in itertools.product(range(count), repeat=3):
        for order in itertools.permutations(range(3)):
            vertex = np.asarray(corner)
            path = [index[tuple(vertex)]]
            for axis_index in order:
                vertex = vertex + np.eye(3, dtype=np.int64)[axis_index]
                path.append(index[tuple(vertex)])
            cells.append(path)
    return points, _oriented(points, np.asarray(cells, dtype=np.int32))


def _masked_layout(
    points: Any, cells: Any, vertex_capacity: Any, cell_capacity: Any, seed: Any
) -> Any:
    """Pad a compact mesh into capacity slots with interleaved inactive lanes."""
    generator = np.random.default_rng(seed)
    vertex_slots = np.sort(
        generator.choice(vertex_capacity, points.shape[0], replace=False)
    )
    cell_slots = np.sort(generator.choice(cell_capacity, cells.shape[0], replace=False))
    coordinates = np.full((vertex_capacity, points.shape[1]), np.nan)
    coordinates[vertex_slots] = points
    vertex_ids = np.full((vertex_capacity,), -1, dtype=np.int64)
    vertex_ids[vertex_slots] = np.arange(points.shape[0])
    vertex_active = vertex_ids >= 0
    slot_cells = np.zeros((cell_capacity, cells.shape[1]), dtype=np.int32)
    slot_cells[cell_slots] = vertex_slots[cells]
    cell_ids = np.full((cell_capacity,), -1, dtype=np.int64)
    cell_ids[cell_slots] = np.arange(cells.shape[0])
    cell_active = cell_ids >= 0
    slot_cells = jnp.asarray(slot_cells)
    cell_active = jnp.asarray(cell_active)
    mesh = MaskedSimplexMesh(
        jnp.asarray(coordinates),
        jnp.asarray(vertex_ids),
        jnp.asarray(vertex_active),
        slot_cells,
        jnp.asarray(cell_ids),
        cell_active,
        masked_simplex_facet_neighbors(slot_cells, cell_active),
    )
    return mesh, vertex_slots


def _compact(points: Any, cells: Any) -> Any:
    kind = "triangle" if cells.shape[1] == 3 else "tetrahedron"
    mesh = (
        phx.discretization.CellMesh.from_triangles
        if kind == "triangle"
        else phx.discretization.CellMesh.from_tetrahedra
    )(jnp.asarray(points), jnp.asarray(cells))
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element(kind, 1)
    )
    return phx.discretization.FiniteElementPlan(mesh, field).prepare()


def _source(points: Any) -> Any:
    return 1.0 + points[..., 0] + 2.0 * points[..., 1]


_CASES = {
    "triangle": (lambda: _square(5), 52, 70),
    "tetrahedron": (lambda: _cube(2), 41, 60),
}


@pytest.fixture(params=sorted(_CASES))
def case(request: Any) -> Any:
    build, vertex_capacity, cell_capacity = _CASES[request.param]
    points, cells = build()
    mesh, slots = _masked_layout(points, cells, vertex_capacity, cell_capacity, 7)
    plan = phx.discretization.MaskedFiniteElementPlan(mesh)
    system = phx.discretization.assemble_masked_finite_element(plan, mesh)
    return points, cells, mesh, slots, system


def test_masked_poisson_matches_compact_dirichlet_solve(case: Any) -> None:
    points, cells, mesh, slots, system = case
    discretization = _compact(points, cells)
    form = phx.equations.FiniteElementForm(
        "poisson",
        "u",
        (
            phx.equations.DiffusionAction("u"),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    lambda x, _: _source(x), coefficient_id="affine-source"
                ),
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=phx.discretization.dirichlet_constraint(discretization, "u"),
        dirichlet_values=0.0,
    )
    compact_system, compact_rhs = compiled.linear_system()
    compact_result = phx.linalg.solve(compact_system, compact_rhs)
    expected = compiled.expand(compact_result.value)

    source = jnp.where(mesh.vertex_active, _source(mesh.coordinates), 0.0)
    rhs = jnp.where(system.boundary_dofs, 0.0, system.mass.mv(source))
    operator = phx.discretization.constrain_masked_dofs(
        system.stiffness, system.boundary_dofs
    )
    result = phx.linalg.solve(phx.linalg.LinearSystem(operator), rhs)
    padding = np.setdiff1d(np.arange(mesh.vertex_capacity), slots)

    assert jnp.all(compact_result.successful) and jnp.all(result.successful)
    assert jnp.array_equal(system.boundary_dofs[slots], discretization.boundary_dof_mask)
    assert not jnp.any(system.boundary_dofs[padding])
    assert jnp.max(jnp.abs(expected)) > 1.0e-3
    np.testing.assert_allclose(result.value[slots], expected, rtol=1e-9, atol=1e-11)
    assert jnp.all(result.value[padding] == 0.0)


def test_masked_operators_match_compact_actions_and_transpose(case: Any) -> None:
    points, cells, mesh, slots, system = case
    discretization = _compact(points, cells)
    padding = np.setdiff1d(np.arange(mesh.vertex_capacity), slots)
    probe = jnp.asarray(np.random.default_rng(3).standard_normal((mesh.vertex_capacity,)))
    for masked, compact in (
        (system.mass, discretization.mass),
        (system.stiffness, discretization.stiffness),
    ):
        action = masked.mv(probe)

        assert action.shape == (mesh.vertex_capacity,)
        assert jnp.all(jnp.isfinite(action))
        np.testing.assert_allclose(masked.transpose_mv(probe), action, atol=1e-13)
        np.testing.assert_allclose(masked.adjoint_mv(probe), action, atol=1e-13)
        np.testing.assert_allclose(
            action[slots], compact.mv(probe[slots]), rtol=1e-12, atol=1e-13
        )
        assert jnp.array_equal(action[padding], probe[padding])


def test_masked_mass_integrates_constants_and_stiffness_annihilates_them(
    case: Any,
) -> None:
    points, cells, mesh, slots, system = case
    discretization = _compact(points, cells)
    ones = mesh.vertex_active.astype(jnp.float64)
    total = jnp.sum(discretization.measures[0].weights)

    assert jnp.array_equal(system.dof_active, mesh.vertex_active)
    np.testing.assert_allclose(jnp.sum(system.mass.mv(ones)[slots]), total, rtol=1e-13)
    np.testing.assert_allclose(total, 1.0, rtol=1e-13)
    np.testing.assert_allclose(system.stiffness.mv(ones), 0.0, atol=1e-12)


def test_masked_plan_rejects_non_vertex_degrees_and_foreign_buckets() -> None:
    points, cells = _square(2)
    mesh, _ = _masked_layout(points, cells, 12, 10, 0)
    other, _ = _masked_layout(points, cells, 13, 10, 0)
    plan = phx.discretization.MaskedFiniteElementPlan(mesh)

    with pytest.raises(ValueError, match="vertex DOFs only"):
        phx.discretization.MaskedFiniteElementPlan(mesh, degree=2)
    with pytest.raises(ValueError, match="capacity bucket"):
        phx.discretization.assemble_masked_finite_element(plan, other)
