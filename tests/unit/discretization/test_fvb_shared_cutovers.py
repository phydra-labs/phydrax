import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import PeriodicCell
from phydrax.discretization._cell_complex import (
    polyhedral_cell_complex,
    polyhedral_connectivity,
    PolyhedralConnectivity,
    PolyhedralWorksetLimitError,
    prepare_polyhedral_worksets,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization.finite_volume import (
    prepare_polyhedral_finite_volume_geometry,
)
from phydrax.discretization.vem import prepare_polyhedral_h1_virtual_element_3d


def test_rank_two_periodic_cell_preserves_orthogonal_component():
    cell = PeriodicCell([[2.0, 0.0, 0.0], [0.5, 1.5, 0.0]])
    point = jnp.asarray([2.25, -0.1, 3.0])
    wrapped, images = cell.wrap(point)
    assert cell.rank == 2
    assert cell.ambient_dimension == 3
    assert wrapped[2] == point[2]
    assert images.shape == (2,)
    assert jnp.allclose(
        cell.vectors @ cell.reciprocal_vectors.T, 2.0 * jnp.pi * jnp.eye(2)
    )


def test_root_polyhedral_connectivity_drives_degree_one_vem():
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = (
        (0, 2, 1),
        (0, 1, 3),
        (0, 3, 2),
        (1, 2, 3),
    )
    mesh = CellMesh.from_polyhedra(points, (faces,))
    assert isinstance(mesh.connectivity, PolyhedralConnectivity)
    assert bool(jnp.all(mesh.connectivity.boundary_faces))
    prepared = prepare_polyhedral_h1_virtual_element_3d(mesh)
    constant = jnp.ones((4,))
    assert jnp.linalg.norm(prepared.mv(constant)) < 1e-10
    assert prepared.evidence.minimum_volume > 0.0


def test_pathological_polyhedral_arity_keeps_actual_storage_linear():
    arity = 256
    tetrahedron_count = 128
    pyramid = (tuple(reversed(range(arity))),) + tuple(
        (vertex, (vertex + 1) % arity, arity) for vertex in range(arity)
    )
    tetrahedra = (
        np.arange(4 * tetrahedron_count, dtype=np.int32).reshape((-1, 4)) + arity + 1
    )
    vertex_count = arity + 1 + 4 * tetrahedron_count
    connectivity = polyhedral_connectivity(
        (pyramid, ("tetrahedron", tetrahedra)), vertex_count
    )
    input_entries = 4 * arity + 12 * tetrahedron_count
    stored_bytes = sum(array.nbytes for array in jax.tree.leaves(connectivity))
    assert stored_bytes <= 64 * (input_entries + vertex_count + connectivity.cell_count)
    with pytest.raises(PolyhedralWorksetLimitError):
        prepare_polyhedral_worksets(connectivity, maximum_entries=input_entries)


def test_mixed_polyhedral_incidence_preserves_ids_orientation_and_workset_budget():
    prism_faces = (
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (2, 0, 3, 5),
    )
    tetrahedron_faces = ((3, 5, 4), (3, 4, 6), (3, 6, 5), (4, 5, 6))
    identifiers = {
        "vertex_global_ids": np.asarray((50, 10, 70, 20, 80, 30, 90)),
        "edge_global_ids": np.arange(12) + 1000,
        "face_global_ids": np.arange(8) + 2000,
        "cell_global_ids": np.asarray((901, 42)),
    }
    connectivity = polyhedral_connectivity(
        (("prism", np.arange(6)[None, :]), tetrahedron_faces), 7, **identifiers
    )
    rotated = polyhedral_connectivity(
        tuple(
            tuple(face[1:] + face[:1] for face in faces)
            for faces in (prism_faces, tetrahedron_faces)
        ),
        7,
        **identifiers,
    )
    topology = polyhedral_cell_complex(connectivity)
    assert polyhedral_cell_complex(rotated).topology_id == topology.topology_id
    for degree, name in enumerate(
        ("vertex_global_ids", "edge_global_ids", "face_global_ids", "cell_global_ids")
    ):
        np.testing.assert_array_equal(
            topology.entities(degree).entity_ids, identifiers[name]
        )
    boundary = topology.incidences[2].scipy_boundary() @ np.ones(2)
    shared = np.flatnonzero(np.asarray(connectivity.face_cell_counts) == 2)
    np.testing.assert_array_equal(boundary[shared], np.zeros(1))
    assert np.sum(np.abs(boundary)) == 7

    with pytest.raises(PolyhedralWorksetLimitError):
        prepare_polyhedral_worksets(connectivity, maximum_entries=213)
    worksets = prepare_polyhedral_worksets(connectivity, maximum_entries=214)
    assert (
        worksets.allocated_entries
        == sum(array.size for array in jax.tree.leaves(worksets))
        == 214
    )
    dense_boundary = np.bincount(
        np.asarray(worksets.cell_faces)[np.asarray(worksets.cell_face_valid)],
        weights=np.asarray(worksets.cell_face_signs)[
            np.asarray(worksets.cell_face_valid)
        ],
        minlength=connectivity.face_count,
    )
    np.testing.assert_array_equal(dense_boundary, boundary)


def test_mixed_standard_polyhedral_geometry_preserves_reference_vertex_order():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.3, 0.3, 2.0),
        )
    )
    mesh = CellMesh.from_mixed_3d(
        points,
        (
            CellBlock(
                "prism",
                "prism",
                ((1, 2, 0, 4, 5, 3),),
                global_ids=(901,),
            ),
        ),
        polyhedra={"cap": (((3, 5, 4), (3, 4, 6), (3, 6, 5), (4, 5, 6)),)},
        vertex_global_ids=(50, 10, 70, 20, 80, 30, 90),
        polyhedral_cell_global_ids={"cap": (42,)},
    )
    geometry = prepare_polyhedral_finite_volume_geometry(mesh)
    np.testing.assert_allclose(geometry.cell_volumes, (0.5, 1.0 / 6.0))
    np.testing.assert_array_equal(mesh.entity_set(3).entity_ids, (901, 42))
    prepared = prepare_polyhedral_h1_virtual_element_3d(mesh)
    np.testing.assert_allclose(prepared.mv(jnp.ones(7)), np.zeros(7), atol=1e-10)
