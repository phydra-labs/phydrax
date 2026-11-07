# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Consumer-visible canonical FV receptor-face incidence contracts."""

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization.finite_volume._unstructured_dynamics import (
    PreparedUnstructuredFiniteVolumeDynamics,
)
from phydrax.meshing._overset_finite_volume import _unique_receptor_faces


def test_exact_face_coorientation_retains_a_small_positive_component() -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 1.0e-16), (1.0, 1.0), (0.0, 1.0)))
    geometry = phx.discretization.UnstructuredFiniteVolumePlan(
        points,
        quadrilaterals=np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        vertex_global_ids=np.asarray((10, 11, 12, 13)),
        cell_global_ids=np.asarray((101,)),
    ).prepare()
    _, quadrature, normals, _, _ = _unique_receptor_faces(
        geometry, np.asarray((0,), dtype=np.int32)
    )
    assert np.all(quadrature[0, :, 1] < 1.0e-15)
    assert np.all(normals[0, :, 0] > 0.0)


def test_embedded_surface_edge_selection_survives_storage_and_route_permutations() -> (
    None
):
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0))
    )
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    vertex_ids = np.asarray((10, 11, 12, 13))
    cell_ids = np.asarray((101, 103))

    def select(
        vertex_order: np.ndarray, cell_order: np.ndarray, ids: tuple[int, ...]
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        geometry = phx.discretization.UnstructuredFiniteVolumePlan(
            vertices[vertex_order],
            triangles=np.argsort(vertex_order)[cells[cell_order]],
            vertex_global_ids=vertex_ids[vertex_order],
            cell_global_ids=cell_ids[cell_order],
        ).prepare()
        row_of = {
            int(value): index for index, value in enumerate(geometry.cell_global_ids)
        }
        rows = np.asarray([row_of[identifier] for identifier in ids], dtype=np.int32)
        faces, points, normals, weights, _ = _unique_receptor_faces(geometry, rows)
        assert np.unique(faces).size == len(ids)
        result = {}
        for index, identifier in enumerate(ids):
            order = np.lexsort(points[index].T[::-1])
            result[identifier] = (
                points[index, order],
                normals[index, order],
                weights[index, order],
            )
        return result

    baseline = select(np.arange(4), np.arange(2), (101, 103))
    for current in (
        select(np.arange(4), np.arange(2), (101,)),
        select(np.asarray((3, 1, 0, 2)), np.asarray((1, 0)), (103, 101)),
    ):
        for identifier, entries in current.items():
            for actual, expected in zip(entries, baseline[identifier], strict=True):
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)


def test_periodic_self_face_neighbor_incidence_reaches_runtime() -> None:
    system = phx.equations.EulerSystem(2)
    vertices = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (2.0, 1.0),
        )
    )
    plan = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(((0, 1, 2, 3), (4, 5, 6, 7)), dtype=np.int32),
        vertex_global_ids=np.asarray((10, 11, 12, 13, 14, 15, 16, 17)),
        cell_global_ids=np.asarray((101, 103)),
        boundary_patches={
            "__periodic_seam__": np.asarray(((0, 3), (1, 2)), dtype=np.int32),
            "walls": np.asarray(
                ((0, 1), (2, 3), (4, 5), (5, 6), (6, 7), (7, 4)), dtype=np.int32
            ),
        },
        component_names=system.component_names,
    )
    assert plan.mesh is not None
    assert isinstance(plan.mesh.connectivity, PolygonalConnectivity)
    edges = np.asarray(plan.mesh.connectivity.edges)
    left = next(i for i, edge in enumerate(edges) if set(edge) == {0, 3})
    right = next(i for i, edge in enumerate(edges) if set(edge) == {1, 2})
    plan = phx.discretization.UnstructuredFiniteVolumePlan._with_owner_local_joins(
        plan,
        None,
        np.asarray(((left, right),), dtype=np.int32),
        np.asarray(((1.0, 0.0),)),
    )
    geometry = plan.prepare()
    rows = np.asarray((0,), dtype=np.int32)
    faces, points, normals, measures, orientations = _unique_receptor_faces(
        geometry, rows
    )
    assert int(np.asarray(geometry.owner_cells)[faces[0]]) == 0
    assert int(np.asarray(geometry.neighbor_cells)[faces[0]]) == 0
    np.testing.assert_array_equal(orientations, (-1,))
    np.testing.assert_allclose(points[0, :, 0], 1.0)
    np.testing.assert_allclose(normals[0], ((1.0, 0.0), (1.0, 0.0)))

    def overset(
        side: np.ndarray, qpoints: np.ndarray, qnormals: np.ndarray
    ) -> phx.discretization.UnstructuredOversetPlan:
        return phx.discretization.UnstructuredOversetPlan(
            geometry,
            geometry,
            rows,
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((1,), dtype=np.int32),
            np.asarray((1.0,)),
            receptor_face_ids=faces,
            receptor_face_points=qpoints,
            receptor_face_normals=qnormals,
            receptor_face_measures=measures,
            receptor_face_cells=rows,
            receptor_face_orientations=side,
            receptor_route_faces=np.asarray((0,), dtype=np.int32),
        )

    coupling = overset(orientations, points, normals)
    np.testing.assert_array_equal(coupling.receptor_face_orientations, (-1,))
    with pytest.raises(ValueError, match="stale|match"):
        overset(-orientations, points, normals)
    owner_side = overset(
        -orientations,
        np.asarray(geometry.face_quadrature_points)[faces],
        -normals,
    )
    assert owner_side.face_artifact_id != coupling.face_artifact_id
    assert owner_side.identity != coupling.identity
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        geometry.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in geometry.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR("self-face", "state", system, boundaries),
        geometry,
        method,
        coupling=phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
            overset=coupling
        ),
    ).dynamics
    assert isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
    state = system.primitive_to_conserved(
        jnp.asarray(((1.0, 0.3, 0.0, 1.0), (1.0, 0.3, 0.0, 1.0)))
    )
    metrics = phx.discretization.lower_static_unstructured_stage_metrics(geometry)
    block, _, _ = dynamics._overset_correction(state, metrics, None)
    assert block is not None
    np.testing.assert_allclose(np.asarray(block.flux_rate)[0, 0], -0.3, rtol=1.0e-6)
