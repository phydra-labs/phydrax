from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import (
    canonical_nodal_entity_identity,
    lagrange_element,
    lagrange_reference_node_data,
)


def test_simplex_p3_lobatto_edges_keep_exact_root_and_execution_shadow() -> None:
    element = lagrange_element("triangle", 3)
    nodes = lagrange_reference_node_data(element)
    edge_dofs = element.entity_dofs[1][0]
    root = RadicalElement.sqrt(Fraction(1, 5))
    expected = {Fraction(1, 2) - root / 2, Fraction(1, 2) + root / 2}

    exact_x = set()
    for dof in edge_dofs:
        node = nodes[dof]
        assert node.exact_point is not None
        exact_x.add(node.exact_point[0])
        assert node.exact_point[1] == 0
        assert node.reference_point[0].operation == "lobatto-unit"
        assert Fraction(node.execution_shadow[0]) != node.exact_point[0]
        assert node.projection_error_bound[0] >= abs(
            Fraction(node.execution_shadow[0]) - node.exact_point[0]
        )
    assert exact_x == expected


def test_reversed_shared_edge_and_tetra_prism_face_have_canonical_node_ids() -> None:
    triangle = lagrange_element("triangle", 3)
    triangle_nodes = lagrange_reference_node_data(triangle)
    forward_vertices = (10, 20, 30)
    reversed_vertices = (20, 10, 40)
    forward_keys = {
        canonical_nodal_entity_identity(triangle_nodes[dof], forward_vertices)[0]
        for dof in triangle.entity_dofs[1][0]
    }
    reverse_keys = {
        canonical_nodal_entity_identity(triangle_nodes[dof], reversed_vertices)[0]
        for dof in triangle.entity_dofs[1][0]
    }
    assert forward_keys == reverse_keys

    tetrahedron = lagrange_element("tetrahedron", 3)
    tetra_nodes = lagrange_reference_node_data(tetrahedron)
    prism = lagrange_element("prism", 3)
    prism_nodes = lagrange_reference_node_data(prism)
    tetra_vertices = (90, 10, 20, 30)
    prism_vertices = (10, 30, 20, 40, 50, 60)
    tetra_face = set(reference_cell_topology("tetrahedron").entities[2][0])
    prism_face = set(reference_cell_topology("prism").entities[2][0])
    tetra_keys = {
        canonical_nodal_entity_identity(node, tetra_vertices)[0]
        for node in tetra_nodes
        if set(node.entity_vertices) <= tetra_face
    }
    prism_keys = {
        canonical_nodal_entity_identity(node, prism_vertices)[0]
        for node in prism_nodes
        if set(node.entity_vertices) <= prism_face
    }
    assert tetra_keys == prism_keys


def test_p3_algebraic_reference_point_evaluates_exact_linear_geometry() -> None:
    mesh = CellMesh.from_simplices(
        np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 3.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
        dimension=2,
    )
    element = lagrange_element("simplex:2", 1)
    geometry = CellGeometrySpec(
        {"simplices": element},
        {"simplices": np.asarray(((0, 1, 2),), dtype=np.int64)},
        np.asarray(mesh.coordinates),
    )
    p3 = lagrange_reference_node_data(lagrange_element("triangle", 3))
    node = next(
        item for item in p3 if item.entity_dimension == 1 and item.exact_point is not None
    )
    exact_point = node.exact_point
    assert exact_point is not None
    cell_id = int(mesh.blocks[0].global_ids[0])
    evaluated = geometry.evaluate_exact(
        mesh,
        np.asarray((cell_id,), dtype=np.int64),
        ((exact_point,),),
    )[0][0]
    assert evaluated == (
        2 * exact_point[0],
        3 * exact_point[1],
    )


def test_affine_qualified_simplex_uses_exact_form_basis_coefficients() -> None:
    mesh = CellMesh.from_simplices(
        np.asarray(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        dimension=3,
    )
    geometry = CellGeometrySpec.affine(mesh)
    cell_id = int(mesh.blocks[0].global_ids[0])
    evaluated = geometry.evaluate_exact(
        mesh,
        np.asarray((cell_id,), dtype=np.int64),
        (((Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)),),),
    )
    assert evaluated == (((Fraction(1, 2), Fraction(3, 4), Fraction(1)),),)


def test_cell_interior_identity_and_expression_only_higher_order_sources() -> None:
    cell = lagrange_element("triangle", 0)
    node = lagrange_reference_node_data(cell)[0]
    with pytest.raises(ValueError, match="global cell ID"):
        canonical_nodal_entity_identity(node, (5, 6, 7))
    first = canonical_nodal_entity_identity(node, (5, 6, 7), cell_global_id=11)[0]
    second = canonical_nodal_entity_identity(node, (5, 6, 7), cell_global_id=12)[0]
    assert first != second

    p5_nodes = lagrange_reference_node_data(lagrange_element("triangle", 5))
    expression_node = next(
        item
        for item in p5_nodes
        if item.entity_dimension == 1 and item.exact_point is None
    )
    shadow = Fraction(expression_node.execution_shadow[0])
    error_bound = expression_node.projection_error_bound[0]
    lower, upper = expression_node.reference_point[0].interval()
    assert shadow - error_bound <= lower
    assert upper <= shadow + error_bound

    for cell_kind, degree in (
        ("interval", 3),
        ("hexahedron", 2),
        ("prism", 2),
        ("pyramid", 2),
        ("tetrahedron", 5),
    ):
        element = lagrange_element(cell_kind, degree)
        data = lagrange_reference_node_data(element)
        assert len(data) == element.local_dof_count
        assert all(
            len(item.execution_shadow) == element.topological_dimension for item in data
        )
        assert all(
            len(item.projection_error_bound) == element.topological_dimension
            for item in data
        )
        assert all(item.reference_point for item in data)
