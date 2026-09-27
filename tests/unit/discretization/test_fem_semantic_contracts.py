#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import pytest

import phydrax as phx


def _triangle_mesh() -> Any:
    return phx.discretization.CellMesh.from_triangles(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
    )


def _unit_hex_mesh() -> Any:
    coordinates = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    block = phx.discretization.CellBlock(
        "hexes",
        "hexahedron",
        jnp.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=jnp.int32),
    )
    return phx.discretization.CellMesh(coordinates, (block,))


def test_fem_semantic_contracts_scenario_1() -> None:
    mesh = _triangle_mesh()
    h1 = phx.discretization.lagrange_element("triangle", 1)
    l2 = phx.discretization.discontinuous_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        (
            phx.discretization.FiniteElementFieldSpec("continuous", h1),
            phx.discretization.FiniteElementFieldSpec("discontinuous", l2),
        ),
    ).prepare()

    assert h1.conformity == "H1"
    assert l2.conformity == "L2"
    assert h1.representation == "point_value"
    assert l2.representation == "point_value"
    assert tuple(space.representation for space in discretization.field_spaces) == (
        "point_value",
        "point_value",
    )
    assert (
        phx.discretization.raviart_thomas_element("triangle").representation
        == "flux_moment"
    )
    assert (
        phx.discretization.nedelec_element("triangle").representation
        == "circulation_moment"
    )
    discretization = phx.discretization.FiniteElementPlan(
        _triangle_mesh(),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    cells = discretization.mesh.topology.entity_sets[2]
    facets = discretization.mesh.topology.entity_sets[1]
    cell = phx.equations.coefficient(
        jnp.asarray([2.0]),
        location="cell",
        support_id=discretization.support.support_id,
        entity_set_id=cells.entity_set_id,
    )
    facet = phx.equations.coefficient(
        jnp.ones((facets.count,)),
        location="facet",
        support_id=discretization.support.support_id,
        entity_set_id=facets.entity_set_id,
        side="plus",
    )
    cell_changed = phx.equations.coefficient(
        jnp.asarray([3.0]),
        location="cell",
        support_id=discretization.support.support_id,
        entity_set_id=cells.entity_set_id,
    )
    quadrature = phx.equations.coefficient(
        jnp.ones((1, 2)),
        location="quadrature",
        support_id=discretization.support.support_id,
        entity_set_id=cells.entity_set_id,
        rule_id="two-point-rule",
    )

    assert (
        len({cell.coefficient_id, facet.coefficient_id, quadrature.coefficient_id}) == 3
    )
    with pytest.raises(ValueError, match="require support_id"):
        phx.equations.coefficient(jnp.ones((1,)), location="cell")
    with pytest.raises(ValueError, match="rule_id"):
        phx.equations.coefficient(
            jnp.ones((1, 2)),
            location="quadrature",
            support_id=discretization.support.support_id,
            entity_set_id=cells.entity_set_id,
        )
    assert cell.layout_id == cell_changed.layout_id
    assert cell.coefficient_id != cell_changed.coefficient_id
    with pytest.raises(ValueError, match="support identity"):
        cell.evaluate(
            jnp.zeros((1, 2, 2)),
            entity_indices=jnp.asarray([0]),
            support_id="wrong-support",
        )
    with pytest.raises(ValueError, match="canonical layout axes"):
        phx.equations.coefficient(
            jnp.asarray([2.0]),
            location="cell",
            support_id=discretization.support.support_id,
            entity_set_id=cells.entity_set_id,
            layout_axes=("quadrature",),
        )
    dof = phx.equations.coefficient(
        jnp.asarray([2.0, 3.0]),
        location="dof",
        support_id=discretization.support.support_id,
        field_space_id=discretization.field_spaces[0].field_space_id,
    )
    oriented = dof.evaluate(
        jnp.zeros((1, 2, 2)),
        dof_indices=jnp.asarray([[0, 1]], dtype=jnp.int32),
        dof_orientations=jnp.asarray([[1.0, -1.0]]),
        basis_values=jnp.eye(2),
        support_id=discretization.support.support_id,
        field_space_id=discretization.field_spaces[0].field_space_id,
    )
    assert jnp.array_equal(oriented, jnp.asarray([[2.0, -3.0]]))
    configuration = phx.solver.FiniteElementRunConfiguration(
        realization="matrix_free",
        local_kernel="sum_factorized",
        accumulation="deterministic",
    )
    policy = configuration.execution_policy()

    assert policy.realization == "matrix_free"
    assert policy.local_kernel == "sum_factorized"
    assert policy.accumulation == "deterministic"
    with pytest.raises(ValueError, match="local-kernel"):
        phx.equations.FiniteElementExecutionPolicy(local_kernel="element-tensor")
    with pytest.raises(ValueError, match="Sparse realization"):
        phx.equations.FiniteElementExecutionPolicy(
            realization="sparse", local_kernel="sum_factorized"
        )


def test_fem_semantic_contracts_scenario_2() -> None:
    source_mass = jnp.asarray([0.5, 0.5])
    target_mass = jnp.asarray([0.25, 0.5, 0.25])
    primal = jnp.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    adjoint_matrix = (primal.T * target_mass[None, :]) / source_mass[:, None]
    hilbert_adjoint = phx.linalg.DenseLinearOperator(
        adjoint_matrix,
        source=phx.linalg.ArraySpace((3,)),
        target=phx.linalg.ArraySpace((2,)),
    )
    transfer = phx.discretization.fem.vertex_interpolation_transfer(
        jnp.asarray([[0, 0], [0, 1], [1, 1]], dtype=jnp.int32),
        jnp.asarray([[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]]),
        jnp.asarray([[True, False], [True, True], [True, False]]),
        source_size=2,
        source_topology_id="coarse",
        target_topology_id="fine",
        hilbert_adjoint=hilbert_adjoint,
    )
    source = jnp.asarray([2.0, -1.0])
    target = jnp.asarray([0.5, 3.0, -2.0])
    raw = transfer.pullback(target)
    # ty: ignore[unresolved-attribute]
    adjoint = transfer.hilbert_adjoint.mv(target)

    assert jnp.allclose(raw, primal.T @ target)
    assert not jnp.allclose(raw, adjoint)
    assert jnp.allclose(
        jnp.vdot(transfer.apply(source), target_mass * target),
        jnp.vdot(source, source_mass * adjoint),
    )
    mesh = _unit_hex_mesh()
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("hexahedron", 1)
        ),
    ).prepare()
    ones = jnp.ones((8,))

    assert discretization.dof_maps[0].association == "vertex"
    assert jnp.allclose(jnp.sum(discretization.measures[0].weights), 1.0)
    assert jnp.allclose(jnp.sum(discretization.mass.mv(ones)), 1.0)
    assert jnp.allclose(discretization.stiffness.mv(ones), 0.0, atol=1.0e-12)
    matrix = jnp.asarray(discretization.stiffness.to_scipy().toarray())
    assert jnp.allclose(matrix, matrix.T, atol=1.0e-12)
