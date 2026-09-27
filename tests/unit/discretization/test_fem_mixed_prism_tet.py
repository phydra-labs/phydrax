#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _mixed_mesh() -> Any:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.0, 0.0, 2.0),
        )
    )
    prism = phx.discretization.CellBlock(
        "prisms",
        "prism",
        np.asarray(((0, 1, 2, 3, 4, 5),), dtype=np.int32),
        global_ids=np.asarray((101,), dtype=np.int64),
    )
    tetrahedron = phx.discretization.CellBlock(
        "tetrahedra",
        "tetrahedron",
        np.asarray(((6, 3, 5, 4),), dtype=np.int32),
        global_ids=np.asarray((303,), dtype=np.int64),
    )
    return phx.discretization.CellMesh(
        coordinates,
        (prism, tetrahedron),
        numeric_version="mixed-r1",
    )


def _field(*, degree: Any = 1, component_shape: Any = ()) -> Any:
    return phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            "prisms": phx.discretization.lagrange_element("prism", degree),
            "tetrahedra": phx.discretization.lagrange_element("tetrahedron", degree),
        },
        component_shape=component_shape,
    )


def _discretization() -> Any:
    return phx.discretization.FiniteElementPlan(_mixed_mesh(), _field()).prepare()


def test_mixed_prism_contracts() -> None:
    discretization = _discretization()
    mesh = discretization.mesh
    dof_map = discretization.dof_maps[0]

    assert isinstance(mesh.connectivity, phx.discretization.PolyhedralConnectivity)
    assert dof_map.association == "vertex"
    assert dof_map.global_dof_count == 7
    prism_routes = np.asarray(dof_map.cell_dofs[0])[0]
    tetrahedron_routes = np.asarray(dof_map.cell_dofs[1])[0]
    np.testing.assert_array_equal(prism_routes[[3, 4, 5]], (3, 4, 5))
    np.testing.assert_array_equal(tetrahedron_routes[[1, 3, 2]], (3, 4, 5))

    connectivity = mesh.connectivity
    shared_face = int(np.flatnonzero(np.asarray(connectivity.face_neighbor) >= 0)[0])
    start, stop = np.asarray(connectivity.face_vertex_offsets)[
        shared_face : shared_face + 2
    ]
    np.testing.assert_array_equal(
        np.sort(np.asarray(connectivity.face_vertex_values)[start:stop]),
        (3, 4, 5),
    )
    discretization = _discretization()
    mesh = discretization.mesh
    cells = mesh.entity_set(3)
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        3,
        cells.entity_set_id,
        np.asarray((303,), dtype=np.int64),
    )
    selection = phx.meshing.resolve_mesh_scope(mesh, scope)
    domain = discretization.integration_domain("cell", selection)
    functional = phx.equations.FiniteElementFunctional(
        "selected-tetrahedron-volume",
        "u",
        lambda values, gradients, points, context: jnp.ones(values.shape[:2]),
        domain=domain,
    )

    np.testing.assert_array_equal(domain.entity_indices, (1,))
    np.testing.assert_allclose(
        functional.evaluate(discretization, jnp.zeros((7,))),
        1.0 / 6.0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    mesh = _mixed_mesh()

    with pytest.raises(ValueError, match="conforming.*degree 1 only"):
        phx.discretization.FiniteElementPlan(mesh, _field(degree=2))
    mesh = _mixed_mesh()
    discontinuous = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            "prisms": phx.discretization.discontinuous_element("prism", 0),
            "tetrahedra": phx.discretization.discontinuous_element("tetrahedron", 0),
        },
    )

    with pytest.raises(ValueError, match="only scalar P1 H1"):
        phx.discretization.FiniteElementPlan(mesh, discontinuous)
    with pytest.raises(ValueError, match="only scalar P1 H1"):
        phx.discretization.FiniteElementPlan(mesh, _field(component_shape=(2,)))
