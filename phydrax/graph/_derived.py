from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ._geometry import _face_geometry, _validate_mesh_arrays
from ._graph import ensure_graph
from ._ir import GraphIR


LineGraphConnectivity = Literal["directed_path", "shared_node"]


def _as_feature_mapping(value: Any, /) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"features": value}


def _edge_graph_ids(graph: GraphIR, /) -> np.ndarray:
    graph_ids = np.arange(int(graph.n_edge.shape[0]), dtype=np.int32)
    return np.repeat(graph_ids, np.asarray(graph.n_edge, dtype=np.int32), axis=0)


def _node_payload_from_edges(graph: GraphIR, /) -> dict[str, Any]:
    if graph.senders is None or graph.receivers is None:
        raise ValueError("line_graph requires explicit graph senders/receivers.")
    n_edge = int(graph.senders.shape[0])
    nodes = _as_feature_mapping(graph.edges)
    nodes["original_edge_index"] = jnp.arange(n_edge, dtype=jnp.int32)
    nodes["original_sender"] = graph.senders
    nodes["original_receiver"] = graph.receivers
    return nodes


def _line_graph_pairs(
    senders: np.ndarray,
    receivers: np.ndarray,
    /,
    *,
    connectivity: LineGraphConnectivity,
    include_self_transitions: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_edges: list[int] = []
    target_edges: list[int] = []
    shared_nodes: list[int] = []
    n_edge = int(senders.shape[0])
    for i in range(n_edge):
        for j in range(n_edge):
            if i == j and not include_self_transitions:
                continue
            shared = -1
            if connectivity == "directed_path":
                if int(receivers[i]) != int(senders[j]):
                    continue
                shared = int(receivers[i])
            elif connectivity == "shared_node":
                endpoints_i = (int(senders[i]), int(receivers[i]))
                endpoints_j = (int(senders[j]), int(receivers[j]))
                common = sorted(set(endpoints_i).intersection(endpoints_j))
                if not common:
                    continue
                shared = int(common[0])
            else:
                raise ValueError("connectivity must be 'directed_path' or 'shared_node'.")
            source_edges.append(i)
            target_edges.append(j)
            shared_nodes.append(shared)
    return (
        np.asarray(source_edges, dtype=np.int32),
        np.asarray(target_edges, dtype=np.int32),
        np.asarray(shared_nodes, dtype=np.int32),
    )


class LineGraph(eqx.Module):
    """A derived graph whose nodes are edges of an original `GraphIR`."""

    graph: GraphIR
    original_edges: jnp.ndarray
    transition_edges: jnp.ndarray
    source_edge_indices: jnp.ndarray
    target_edge_indices: jnp.ndarray

    def __init__(
        self,
        graph: GraphIR,
        /,
        *,
        original_edges: Any,
        transition_edges: Any,
        source_edge_indices: Any,
        target_edge_indices: Any,
    ):
        self.graph = graph
        self.original_edges = jnp.asarray(original_edges, dtype=jnp.int32)
        self.transition_edges = jnp.asarray(transition_edges, dtype=jnp.int32)
        self.source_edge_indices = jnp.asarray(source_edge_indices, dtype=jnp.int32)
        self.target_edge_indices = jnp.asarray(target_edge_indices, dtype=jnp.int32)

    def original_edges_component(self):
        from ..domain.graph import Nodes

        return Nodes()

    def transition_edges_component(self):
        from ..domain.graph import Edges

        return Edges()


def line_graph(
    graph: GraphIR,
    /,
    *,
    connectivity: LineGraphConnectivity = "directed_path",
    include_self_transitions: bool = False,
    validate: bool = True,
) -> LineGraph:
    """Build the line graph whose nodes are edges of `graph`."""
    graph = ensure_graph(graph, validate=validate)
    if graph.senders is None or graph.receivers is None:
        raise ValueError("line_graph requires explicit graph senders/receivers.")
    senders_np = np.asarray(graph.senders, dtype=np.int32)
    receivers_np = np.asarray(graph.receivers, dtype=np.int32)
    source_edges, target_edges, shared_nodes = _line_graph_pairs(
        senders_np,
        receivers_np,
        connectivity=connectivity,
        include_self_transitions=include_self_transitions,
    )
    edge_graph_ids = _edge_graph_ids(graph)
    if int(source_edges.shape[0]) == 0:
        n_line_edge = np.zeros((int(graph.n_edge.shape[0]),), dtype=np.int32)
    else:
        n_line_edge = np.bincount(
            edge_graph_ids[source_edges],
            minlength=int(graph.n_edge.shape[0]),
        ).astype(np.int32)
    edges = {
        "source_edge_index": jnp.asarray(source_edges, dtype=jnp.int32),
        "target_edge_index": jnp.asarray(target_edges, dtype=jnp.int32),
        "shared_node": jnp.asarray(shared_nodes, dtype=jnp.int32),
    }
    line = GraphIR(
        nodes=_node_payload_from_edges(graph),
        edges=edges,
        senders=jnp.asarray(source_edges, dtype=jnp.int32),
        receivers=jnp.asarray(target_edges, dtype=jnp.int32),
        globals=graph.globals,
        n_node=graph.n_edge,
        n_edge=jnp.asarray(n_line_edge, dtype=jnp.int32),
        validate=validate,
    )
    return LineGraph(
        line,
        original_edges=jnp.arange(int(graph.senders.shape[0]), dtype=jnp.int32),
        transition_edges=jnp.arange(int(source_edges.shape[0]), dtype=jnp.int32),
        source_edge_indices=source_edges,
        target_edge_indices=target_edges,
    )


def _face_edge_map(faces: np.ndarray, /) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for edge in (face[[0, 1]], face[[1, 2]], face[[2, 0]]):
            first, second = int(edge[0]), int(edge[1])
            key = (min(first, second), max(first, second))
            out.setdefault(key, []).append(face_index)
    return out


def _dual_pairs(
    edge_to_faces: dict[tuple[int, int], list[int]],
    /,
    *,
    add_reverse_edges: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    senders: list[int] = []
    receivers: list[int] = []
    shared_edges: list[tuple[int, int]] = []
    for edge, faces in sorted(edge_to_faces.items()):
        if len(faces) < 2:
            continue
        for i, source_face in enumerate(faces):
            for target_face in faces[i + 1 :]:
                senders.append(int(source_face))
                receivers.append(int(target_face))
                shared_edges.append(edge)
                if add_reverse_edges:
                    senders.append(int(target_face))
                    receivers.append(int(source_face))
                    shared_edges.append(edge)
    if not senders:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
        )
    return (
        np.asarray(senders, dtype=np.int32),
        np.asarray(receivers, dtype=np.int32),
        np.asarray(shared_edges, dtype=np.int32),
    )


class MeshDualGraph(eqx.Module):
    """A face-centered dual graph for a triangular mesh."""

    graph: GraphIR
    face_nodes: jnp.ndarray
    dual_edges: jnp.ndarray
    boundary_faces: jnp.ndarray
    interior_faces: jnp.ndarray

    def __init__(
        self,
        graph: GraphIR,
        /,
        *,
        face_nodes: Any,
        dual_edges: Any,
        boundary_faces: Any,
        interior_faces: Any,
    ):
        self.graph = graph
        self.face_nodes = jnp.asarray(face_nodes, dtype=jnp.int32)
        self.dual_edges = jnp.asarray(dual_edges, dtype=jnp.int32)
        self.boundary_faces = jnp.asarray(boundary_faces, dtype=jnp.int32)
        self.interior_faces = jnp.asarray(interior_faces, dtype=jnp.int32)

    def face_nodes_component(self):
        from ..domain.graph import Nodes

        return Nodes()

    def dual_edges_component(self):
        from ..domain.graph import Edges

        return Edges()

    def boundary_faces_component(self):
        from ..domain.graph import BoundaryNodes

        return BoundaryNodes(self.boundary_faces)

    def interior_faces_component(self):
        from ..domain.graph import InteriorNodes

        return InteriorNodes(self.interior_faces)


def mesh_to_dual_graph(
    mesh_vertices: Any,
    mesh_faces: Any,
    /,
    *,
    face_features: Any | None = None,
    add_reverse_edges: bool = True,
    globals: Any = None,
    validate: bool = True,
) -> MeshDualGraph:
    """Convert a triangular mesh into a face-centered dual `GraphIR`."""
    vertices, faces = _validate_mesh_arrays(mesh_vertices, mesh_faces)
    n_face = int(faces.shape[0])
    area, normal, centroid = _face_geometry(vertices, faces)
    edge_to_faces = _face_edge_map(faces)
    senders, receivers, shared_edges = _dual_pairs(
        edge_to_faces,
        add_reverse_edges=add_reverse_edges,
    )
    nodes = _as_feature_mapping(face_features)
    if "features" in nodes:
        features = jnp.asarray(nodes["features"])
        if features.ndim == 0 or int(features.shape[0]) != n_face:
            raise ValueError("face_features must have leading axis matching mesh faces.")
    nodes["face_index"] = jnp.arange(n_face, dtype=jnp.int32)
    nodes["centroid"] = jnp.asarray(centroid, dtype=float)
    nodes["normal"] = jnp.asarray(normal, dtype=float)
    nodes["area"] = jnp.asarray(area[:, None], dtype=float)
    if int(senders.shape[0]) == 0:
        relative = np.zeros((0, vertices.shape[1]), dtype=float)
        distance = np.zeros((0, 1), dtype=float)
    else:
        relative = centroid[receivers] - centroid[senders]
        distance = np.linalg.norm(relative, axis=-1, keepdims=True)
    unit = relative / np.maximum(distance, 1e-30)
    edges = {
        "shared_edge_vertices": jnp.asarray(shared_edges, dtype=jnp.int32),
        "relative": jnp.asarray(relative, dtype=float),
        "distance": jnp.asarray(distance, dtype=float),
        "unit": jnp.asarray(unit, dtype=float),
    }
    boundary = sorted(face_indices[0] for face_indices in edge_to_faces.values() if len(face_indices) == 1)
    boundary_faces = np.asarray(sorted(set(boundary)), dtype=np.int32)
    all_faces = np.arange(n_face, dtype=np.int32)
    interior_faces = np.setdiff1d(all_faces, boundary_faces, assume_unique=False).astype(np.int32)
    graph = GraphIR(
        nodes=nodes,
        edges=edges,
        senders=jnp.asarray(senders, dtype=jnp.int32),
        receivers=jnp.asarray(receivers, dtype=jnp.int32),
        globals=globals,
        n_node=jnp.asarray([n_face], dtype=jnp.int32),
        n_edge=jnp.asarray([int(senders.shape[0])], dtype=jnp.int32),
        validate=validate,
    )
    return MeshDualGraph(
        graph,
        face_nodes=jnp.arange(n_face, dtype=jnp.int32),
        dual_edges=jnp.arange(int(senders.shape[0]), dtype=jnp.int32),
        boundary_faces=boundary_faces,
        interior_faces=interior_faces,
    )


__all__ = [
    "LineGraph",
    "LineGraphConnectivity",
    "MeshDualGraph",
    "line_graph",
    "mesh_to_dual_graph",
]
