import jax.numpy as jnp

import phydrax as phx
from phydrax.graph import (
    mesh_to_geometry_graph,
    mesh_to_graph,
    point_cloud_to_graph,
)


class _DummyGeometry:
    def __init__(self):
        self.mesh_vertices = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.mesh_faces = jnp.array([[0, 1, 2]], dtype=jnp.int32)


def test_mesh_to_graph_bidirected_triangle():
    geom = _DummyGeometry()
    graph = mesh_to_graph(geom.mesh_vertices, geom.mesh_faces)
    assert graph.num_nodes == 3
    assert graph.num_edges == 6
    assert graph.nodes.shape == (3, 3)
    assert graph.edges.shape == (6, 3)


def test_mesh_to_graph_distance_edges():
    geom = _DummyGeometry()
    graph = mesh_to_graph(geom.mesh_vertices, geom.mesh_faces, edge_features="distance")
    assert graph.num_nodes == 3
    assert graph.num_edges == 6
    assert graph.edges.shape == (6, 1)


def test_mesh_to_graph_geometry_features():
    geom = _DummyGeometry()
    graph = mesh_to_graph(
        geom.mesh_vertices,
        geom.mesh_faces,
        node_features="geometry",
        edge_features="geometry",
    )

    assert set(graph.nodes) == {"area", "is_boundary", "normal", "positions"}
    assert set(graph.edges) == {
        "distance",
        "face_count",
        "is_boundary",
        "relative",
        "unit",
    }
    assert graph.nodes["positions"].shape == (3, 3)
    assert graph.nodes["normal"].shape == (3, 3)
    assert graph.nodes["area"].shape == (3, 1)
    assert jnp.allclose(graph.nodes["area"][:, 0], jnp.full((3,), 1.0 / 6.0))
    assert jnp.allclose(
        graph.nodes["normal"], jnp.tile(jnp.array([[0.0, 0.0, 1.0]]), (3, 1))
    )
    assert jnp.all(graph.nodes["is_boundary"])
    assert graph.edges["distance"].shape == (6, 1)
    assert jnp.all(graph.edges["is_boundary"])


def test_mesh_to_geometry_graph_exposes_boundary_components():
    geom = _DummyGeometry()
    bundle = mesh_to_geometry_graph(geom.mesh_vertices, geom.mesh_faces)

    assert bundle.graph.num_nodes == 3
    assert jnp.allclose(bundle.boundary_nodes, jnp.array([0, 1, 2], dtype=jnp.int32))
    assert bundle.interior_nodes.shape == (0,)
    assert bundle.boundary_edges.shape == (6,)
    assert bundle.interface_edges.shape == (0,)

    domain = phx.domain.GraphDomain(bundle.graph)
    component = domain.component({"graph": bundle.boundary_nodes_component()})
    batch = component.sample(
        phx.domain.PointSampling(3, layout=phx.domain.SampleLayout((("graph",),)))
    )

    assert jnp.allclose(batch["graph"]["positions"].data, geom.mesh_vertices)


def test_mesh_to_geometry_graph_marks_shared_interface_edges():
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)

    bundle = mesh_to_geometry_graph(vertices, faces)

    assert bundle.graph.num_edges == 10
    assert bundle.interface_edges.shape == (2,)
    assert jnp.all(bundle.graph.edges["face_count"][bundle.interface_edges, 0] == 2)
    assert jnp.all(bundle.graph.edges["is_boundary"][bundle.boundary_edges, 0])


def test_point_cloud_to_graph_knn_geometry_features():
    points = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
        ]
    )
    graph = point_cloud_to_graph(points, k=1, edge_features="geometry")

    assert graph.num_nodes == 3
    assert graph.num_edges == 3
    assert graph.nodes.shape == (3, 2)
    assert graph.edges["relative"].shape == (3, 2)
    assert graph.edges["distance"].shape == (3, 1)
    assert jnp.allclose(jnp.sort(graph.receivers), jnp.array([0, 1, 2], dtype=jnp.int32))
