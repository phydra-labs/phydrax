#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.graph._derived import LineGraphConnectivity


def _path_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [2.0]]),
        edges={"features": jnp.array([[1.0], [3.0]])},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


def _square_mesh() -> Any:
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


def test_graph_derived_scenario_1() -> None:
    bundle = phx.graph.line_graph(_path_graph())
    graph = bundle.graph
    assert graph.senders is not None
    assert graph.receivers is not None

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
    graph = phx.graph.GraphIR(
        senders=jnp.array([0, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 1], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    line = phx.graph.line_graph(graph, connectivity="shared_node").graph
    assert line.senders is not None
    assert line.receivers is not None

    assert line.num_nodes == 2
    assert line.num_edges == 2
    assert jnp.allclose(line.senders, jnp.array([0, 1], dtype=jnp.int32))
    assert jnp.allclose(line.receivers, jnp.array([1, 0], dtype=jnp.int32))
    vertices, faces = _square_mesh()
    bundle = phx.graph.mesh_to_dual_graph(vertices, faces)
    graph = bundle.graph
    assert graph.senders is not None
    assert graph.receivers is not None

    assert graph.num_nodes == 2
    assert graph.num_edges == 2
    assert jnp.allclose(graph.senders, jnp.array([0, 1], dtype=jnp.int32))
    assert jnp.allclose(graph.receivers, jnp.array([1, 0], dtype=jnp.int32))
    assert jnp.allclose(graph.edges["shared_edge_vertices"], jnp.array([[0, 2], [0, 2]]))
    assert graph.nodes["centroid"].shape == (2, 3)
    assert jnp.allclose(graph.nodes["area"][:, 0], jnp.array([0.5, 0.5]))
    assert jnp.allclose(bundle.boundary_faces, jnp.array([0, 1], dtype=jnp.int32))
    assert bundle.interior_faces.shape == (0,)
    vertices, faces = _square_mesh()
    bundle = phx.graph.mesh_to_dual_graph(vertices, faces)
    domain = phx.domain.GraphDomain(bundle.graph, measure="count")
    boundary = domain.component({"graph": bundle.boundary_faces_component()})
    batch = boundary.sample(
        phx.domain.PointSampling(2, layout=phx.domain.SampleLayout((("graph",),)))
    )

    assert jnp.allclose(
        jnp.asarray(batch["graph"]["face_index"].data), jnp.array([0, 1], dtype=jnp.int32)
    )
    # ty: ignore[unresolved-attribute]
    assert boundary.mass.value == 2.0


def test_line_graph_integrates_with_graph_domain_model() -> None:
    bundle = phx.graph.line_graph(_path_graph())
    domain = phx.domain.GraphDomain(bundle.graph)
    component = domain.component({"graph": bundle.original_edges_component()})
    batch = component.sample(
        phx.domain.PointSampling(2, layout=phx.domain.SampleLayout((("graph",),)))
    )

    @domain.Function("graph")
    def u(edge_node: Any) -> Any:
        return edge_node.get("features")[0]

    model = domain.GraphModel(phx.graph.GraphDiffusion(), input_fn=u)

    assert jnp.allclose(jnp.asarray(model(batch).data), jnp.array([-2.0, 2.0]))


@pytest.mark.parametrize("connectivity", ["directed_path", "shared_node"])
@pytest.mark.parametrize("include_self", [False, True], ids=["no-self", "self"])
def test_line_graph_endpoint_joins_match_ordered_brute_force(
    connectivity: LineGraphConnectivity,
    include_self: bool,
) -> None:
    # Parallel/reversed edges, self-loops, and an isolate exercise duplicate
    # joins and the smallest-common-endpoint evidence.
    senders = np.asarray([2, 0, 1, 0, 2, 4], dtype=np.int32)
    receivers = np.asarray([0, 2, 1, 2, 3, 4], dtype=np.int32)
    expected: list[tuple[int, int, int]] = []
    for i in range(senders.size):
        for j in range(senders.size):
            if i == j and not include_self:
                continue
            if connectivity == "directed_path":
                common = [int(receivers[i])] if receivers[i] == senders[j] else []
            else:
                common = sorted(
                    {int(senders[i]), int(receivers[i])}
                    & {int(senders[j]), int(receivers[j])}
                )
            if common:
                expected.append((i, j, common[0]))
    graph = phx.graph.GraphIR(
        senders=senders,
        receivers=receivers,
        n_node=jnp.asarray([5], dtype=jnp.int32),
        n_edge=jnp.asarray([6], dtype=jnp.int32),
    )
    line = phx.graph.line_graph(
        graph,
        connectivity=connectivity,
        include_self_transitions=include_self,
    ).graph
    assert line.senders is not None
    assert line.receivers is not None
    np.testing.assert_array_equal(
        np.stack((line.senders, line.receivers, line.edges["shared_node"]), axis=-1),
        np.asarray(expected, dtype=np.int32),
    )
