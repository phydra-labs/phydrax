#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

import phydrax as phx


def _mesh():
    coordinates = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    block = phx.discretization.CellBlock(
        "triangles",
        "triangle",
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        global_ids=np.asarray((101, 303), dtype=np.int64),
    )
    return phx.discretization.CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=np.asarray((10, 20, 30, 40), dtype=np.int64),
        entity_global_ids={
            1: np.asarray((701, 709, 719, 727, 733), dtype=np.int64),
        },
        numeric_version="revision-1",
    )


def _scope(mesh, dimension, rows, **overrides):
    entities = mesh.entity_set(dimension)
    return phx.meshing.MeshingScope(
        overrides.get("source_id", mesh.mesh_id),
        overrides.get("source_revision", mesh.numeric_version),
        overrides.get("entity_kind", phx.meshing.MeshingEntityKind.MESH),
        overrides.get("entity_dimension", dimension),
        overrides.get("entity_set_id", entities.entity_set_id),
        np.asarray(entities.entity_ids)[np.asarray(rows, dtype=np.int32)],
    )


def _facet_row(mesh, vertices):
    expected = frozenset(vertices)
    return next(
        row
        for row, edge in enumerate(np.asarray(mesh.connectivity.edges, dtype=np.int32))
        if frozenset(int(vertex) for vertex in edge) == expected
    )


def _discretization(mesh, *, degree=1):
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", degree)
        ),
    ).prepare()


def test_resolve_mesh_scope_returns_exact_persistent_id_selection():
    mesh = _mesh()
    scope = _scope(mesh, 2, (1,))

    selection = phx.meshing.resolve_mesh_scope(mesh, scope)

    np.testing.assert_array_equal(selection.mask, (False, True))
    assert selection.entity_set_id == mesh.entity_set(2).entity_set_id


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"source_id": "another-mesh"}, "another mesh source"),
        ({"source_revision": "stale-revision"}, "another mesh revision"),
        (
            {"entity_kind": phx.meshing.MeshingEntityKind.GEOMETRY},
            "mesh entities",
        ),
        ({"entity_dimension": 1}, "entity set"),
        ({"entity_set_id": "another-entity-set"}, "entity set"),
    ),
)
def test_resolve_mesh_scope_rejects_foreign_or_stale_bindings(overrides, message):
    mesh = _mesh()

    with pytest.raises(ValueError, match=message):
        phx.meshing.resolve_mesh_scope(mesh, _scope(mesh, 2, (0,), **overrides))


def test_resolve_mesh_scope_rejects_undeclared_global_ids():
    mesh = _mesh()
    cells = mesh.entity_set(2)
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        np.asarray((999_999,), dtype=np.int64),
    )

    with pytest.raises(ValueError, match="undeclared"):
        phx.meshing.resolve_mesh_scope(mesh, scope)


def test_p1_cell_and_boundary_facet_selections_resolve_exact_dofs():
    mesh = _mesh()
    discretization = _discretization(mesh)
    cell_selection = phx.meshing.resolve_mesh_scope(mesh, _scope(mesh, 2, (1,)))
    boundary_facet = _facet_row(mesh, (2, 3))
    facet_selection = phx.meshing.resolve_mesh_scope(
        mesh, _scope(mesh, 1, (boundary_facet,))
    )

    np.testing.assert_array_equal(
        discretization.dof_indices("u", cell_selection), (0, 2, 3)
    )
    np.testing.assert_array_equal(
        discretization.dof_indices("u", facet_selection), (2, 3)
    )
    np.testing.assert_array_equal(
        discretization.dof_mask("u", facet_selection),
        (False, False, True, True),
    )
    domain = discretization.integration_domain("cell", cell_selection)
    np.testing.assert_array_equal(domain.entity_indices, (1,))
    np.testing.assert_array_equal(domain.owner_cells, (1,))
    assert domain.selection_id == cell_selection.selection_id

    constraint = phx.discretization.dirichlet_constraint(
        discretization,
        "u",
        boundary_selection=facet_selection,
    )
    np.testing.assert_array_equal(constraint.constrained_dofs, (2, 3))
    np.testing.assert_array_equal(constraint.free_dofs, (0, 1))


def test_dirichlet_selection_rejects_interior_facets():
    mesh = _mesh()
    discretization = _discretization(mesh)
    interior_facet = int(
        np.flatnonzero(~np.asarray(mesh.connectivity.boundary_edges, dtype=bool))[0]
    )
    selection = phx.meshing.resolve_mesh_scope(mesh, _scope(mesh, 1, (interior_facet,)))

    with pytest.raises(ValueError, match="only exterior facets"):
        phx.discretization.dirichlet_constraint(
            discretization,
            "u",
            boundary_selection=selection,
        )


def test_entity_selection_to_dofs_rejects_high_order_fields():
    mesh = _mesh()
    discretization = _discretization(mesh, degree=2)
    selection = phx.meshing.resolve_mesh_scope(mesh, _scope(mesh, 2, (0,)))

    with pytest.raises(ValueError, match="only vertex-associated P1"):
        discretization.dof_indices("u", selection)
