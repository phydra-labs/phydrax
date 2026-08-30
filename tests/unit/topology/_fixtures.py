#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def filled_triangle_topology():
    return phx.geometry.simplicial.TriangleTopology(
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        num_vertices=3,
    ).cell_complex_topology()


def filled_triangle_filtration():
    topology = filled_triangle_topology()
    complex = phx.topology.CellSubcomplex.full(topology)
    filtration = phx.topology.CellFiltration(
        complex,
        (
            jnp.asarray([0.0, 0.5, 1.0]),
            jnp.asarray([0.5, 1.0, 1.0]),
            jnp.asarray([2.0]),
        ),
        source_id="filled-triangle",
    )
    return topology, complex, filtration


def filled_triangle_vertex_support(topology=None):
    topology = filled_triangle_topology() if topology is None else topology
    return phx.topology.cell_vertex_support(
        topology,
        (
            np.asarray([[0], [1], [2]], dtype=np.int32),
            np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )


def annulus_complex():
    outer = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    vertices = np.concatenate((outer, 0.4 * outer), axis=0)
    faces = np.asarray(
        [(index, (index + 1) % 4, 4 + (index + 1) % 4) for index in range(4)]
        + [(index, 4 + (index + 1) % 4, 4 + index) for index in range(4)],
        dtype=np.int32,
    )
    return phx.graph.triangle_mesh_to_cochain_complex(vertices, faces)


def projective_plane_topology():
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 4],
            [0, 3, 5],
            [0, 4, 5],
            [1, 2, 5],
            [1, 3, 4],
            [1, 4, 5],
            [2, 3, 4],
            [2, 3, 5],
        ],
        dtype=np.int32,
    )
    edge_values = sorted(
        {
            tuple(sorted((int(start), int(end))))
            for face in faces
            for start, end in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
        }
    )
    edges = np.asarray(edge_values, dtype=np.int32)
    edge_index = {edge: index for index, edge in enumerate(edge_values)}
    face_edges = []
    face_signs = []
    for face in faces:
        for start, end in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = tuple(sorted((start, end)))
            face_edges.append(edge_index[edge])
            face_signs.append(1.0 if edge == (start, end) else -1.0)
    vertices = phx.discretization.EntitySet("vertices", 0, np.arange(6))
    edge_entities = phx.discretization.EntitySet(
        "edges",
        1,
        np.arange(edges.shape[0]),
    )
    face_entities = phx.discretization.EntitySet(
        "faces",
        2,
        np.arange(faces.shape[0]),
    )
    vertex_edge = phx.discretization.OrientedIncidence(
        1,
        vertices,
        edge_entities,
        phx.sparse.EdgeRelation(
            edges.reshape((-1,)),
            np.repeat(np.arange(edges.shape[0]), 2),
            source_size=6,
            target_size=edges.shape[0],
        ),
        np.tile(np.asarray([-1.0, 1.0]), edges.shape[0]),
    )
    edge_face = phx.discretization.OrientedIncidence(
        2,
        edge_entities,
        face_entities,
        phx.sparse.EdgeRelation(
            np.asarray(face_edges),
            np.repeat(np.arange(faces.shape[0]), 3),
            source_size=edges.shape[0],
            target_size=faces.shape[0],
        ),
        np.asarray(face_signs),
    )
    return phx.discretization.CellComplexTopology(
        (vertices, edge_entities, face_entities),
        (vertex_edge, edge_face),
    )
