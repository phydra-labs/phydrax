#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _path_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [2.0]]),
        edges={"features": jnp.array([[1.0], [3.0]])},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


def _square_mesh():
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    return vertices, faces


def test_line_graph_builds_edge_as_node_path_graph():
    bundle = phx.graph.line_graph(_path_graph())
    graph = bundle.graph

    assert graph.num_nodes == 2
    assert graph.num_edges == 1
    assert jnp.allclose(graph.senders, jnp.array([0], dtype=jnp.int32))
    assert jnp.allclose(graph.receivers, jnp.array([1], dtype=jnp.int32))
    assert jnp.allclose(graph.nodes["features"][:, 0], jnp.array([1.0, 3.0]))
    assert jnp.allclose(
        graph.nodes["original_sender"], jnp.array([0, 1], dtype=jnp.int32)
    )
    assert jnp.allclose(
        graph.nodes["original_receiver"], jnp.array([1, 2], dtype=jnp.int32)
    )
    assert jnp.allclose(graph.edges["shared_node"], jnp.array([1], dtype=jnp.int32))


def test_line_graph_shared_node_connectivity_adds_undirected_edge_adjacency():
    graph = phx.graph.GraphIR(
        senders=jnp.array([0, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 1], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    line = phx.graph.line_graph(graph, connectivity="shared_node").graph

    assert line.num_nodes == 2
    assert line.num_edges == 2
    assert jnp.allclose(line.senders, jnp.array([0, 1], dtype=jnp.int32))
    assert jnp.allclose(line.receivers, jnp.array([1, 0], dtype=jnp.int32))


def test_line_graph_integrates_with_graph_domain_model():
    bundle = phx.graph.line_graph(_path_graph())
    domain = phx.domain.GraphDomain(bundle.graph)
    component = domain.component({"graph": bundle.original_edges_component()})
    batch = component.sample(
        phx.domain.PointSampling(2, layout=phx.domain.SampleLayout((("graph",),)))
    )

    @domain.Function("graph")
    def u(edge_node):
        return edge_node.get("features")[0]

    model = domain.GraphModel(phx.graph.GraphDiffusion(), input_fn=u)

    assert jnp.allclose(model(batch).data, jnp.array([-2.0, 2.0]))


def test_mesh_to_dual_graph_builds_face_centered_graph():
    vertices, faces = _square_mesh()
    bundle = phx.graph.mesh_to_dual_graph(vertices, faces)
    graph = bundle.graph

    assert graph.num_nodes == 2
    assert graph.num_edges == 2
    assert jnp.allclose(graph.senders, jnp.array([0, 1], dtype=jnp.int32))
    assert jnp.allclose(graph.receivers, jnp.array([1, 0], dtype=jnp.int32))
    assert jnp.allclose(graph.edges["shared_edge_vertices"], jnp.array([[0, 2], [0, 2]]))
    assert graph.nodes["centroid"].shape == (2, 3)
    assert jnp.allclose(graph.nodes["area"][:, 0], jnp.array([0.5, 0.5]))
    assert jnp.allclose(bundle.boundary_faces, jnp.array([0, 1], dtype=jnp.int32))
    assert bundle.interior_faces.shape == (0,)


def test_mesh_dual_graph_boundary_component_samples_faces():
    vertices, faces = _square_mesh()
    bundle = phx.graph.mesh_to_dual_graph(vertices, faces)
    domain = phx.domain.GraphDomain(bundle.graph, measure="count")
    boundary = domain.component({"graph": bundle.boundary_faces_component()})
    batch = boundary.sample(
        phx.domain.PointSampling(2, layout=phx.domain.SampleLayout((("graph",),)))
    )

    assert jnp.allclose(
        batch["graph"]["face_index"].data, jnp.array([0, 1], dtype=jnp.int32)
    )
    assert boundary.mass.value == 2.0
