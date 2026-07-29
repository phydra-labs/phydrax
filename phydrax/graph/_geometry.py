from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ._ir import GraphIR


NodeFeatureMode = Literal["positions", "geometry"] | Callable[[jnp.ndarray], Any] | None
EdgeFeatureMode = (
    Literal["relative", "distance", "geometry"]
    | Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any]
    | None
)
MollifierKind = Literal["wendland_c2", "bump", "hat", "gaussian"]


class GeometryGraph(eqx.Module):
    """Graph plus geometry-derived entity-set metadata."""

    graph: GraphIR
    boundary_nodes: jnp.ndarray
    interior_nodes: jnp.ndarray
    boundary_edges: jnp.ndarray
    interface_edges: jnp.ndarray

    def __init__(
        self,
        graph: GraphIR,
        /,
        *,
        boundary_nodes: Any,
        interior_nodes: Any,
        boundary_edges: Any,
        interface_edges: Any,
    ):
        self.graph = graph
        self.boundary_nodes = jnp.asarray(boundary_nodes, dtype=jnp.int32)
        self.interior_nodes = jnp.asarray(interior_nodes, dtype=jnp.int32)
        self.boundary_edges = jnp.asarray(boundary_edges, dtype=jnp.int32)
        self.interface_edges = jnp.asarray(interface_edges, dtype=jnp.int32)

    def boundary_nodes_component(self):
        from ..domain.graph import BoundaryNodes

        return BoundaryNodes(self.boundary_nodes)

    def interior_nodes_component(self):
        from ..domain.graph import InteriorNodes

        return InteriorNodes(self.interior_nodes)

    def boundary_edges_component(self):
        from ..domain.graph import BoundaryEdges

        return BoundaryEdges(self.boundary_edges)

    def interface_edges_component(self):
        from ..domain.graph import InterfaceEdges

        return InterfaceEdges(self.interface_edges)


class QueryGraph(eqx.Module):
    """A typed bipartite query graph from source points to target points."""

    graph: GraphIR
    source_nodes: jnp.ndarray
    target_nodes: jnp.ndarray
    query_edges: jnp.ndarray
    source_type: int = eqx.field(static=True)
    target_type: int = eqx.field(static=True)
    query_edge_type: int = eqx.field(static=True)

    def __init__(
        self,
        graph: GraphIR,
        /,
        *,
        source_nodes: Any,
        target_nodes: Any,
        query_edges: Any,
        source_type: int,
        target_type: int,
        query_edge_type: int,
    ):
        self.graph = graph
        self.source_nodes = jnp.asarray(source_nodes, dtype=jnp.int32)
        self.target_nodes = jnp.asarray(target_nodes, dtype=jnp.int32)
        self.query_edges = jnp.asarray(query_edges, dtype=jnp.int32)
        self.source_type = int(source_type)
        self.target_type = int(target_type)
        self.query_edge_type = int(query_edge_type)

    def source_nodes_component(self):
        from ..domain.graph import NodeType

        return NodeType(self.source_type, name="source_nodes")

    def target_nodes_component(self):
        from ..domain.graph import NodeType

        return NodeType(self.target_type, name="target_nodes")

    def query_edges_component(self):
        from ..domain.graph import EdgeType

        return EdgeType(self.query_edge_type, name="query_edges")


def _triangle_adjacency(
    faces: np.ndarray,
    *,
    add_reverse_edges: bool,
    add_self_edges: bool,
    n_vertices: int,
) -> np.ndarray:
    directed = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    if add_reverse_edges:
        directed = np.concatenate([directed, directed[:, ::-1]], axis=0)
    if add_self_edges:
        self_edges = np.stack(
            [
                np.arange(n_vertices, dtype=np.int32),
                np.arange(n_vertices, dtype=np.int32),
            ],
            axis=1,
        )
        directed = np.concatenate([directed, self_edges], axis=0)
    return np.unique(directed.astype(np.int32), axis=0)


def _validate_mesh_arrays(mesh_vertices: Any, mesh_faces: Any, /) -> tuple[np.ndarray, np.ndarray]:
    vertices_np = np.asarray(mesh_vertices, dtype=float)
    faces_np = np.asarray(mesh_faces, dtype=np.int32)

    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        raise ValueError(
            "mesh_vertices must have shape (n_vertex, 3); "
            f"got {vertices_np.shape!r}."
        )
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(
            "mesh_faces must have shape (n_face, 3); "
            f"got {faces_np.shape!r}."
        )

    n_vertex = int(vertices_np.shape[0])
    if n_vertex == 0:
        raise ValueError("mesh_vertices must contain at least one vertex.")
    if np.any(faces_np < 0) or np.any(faces_np >= n_vertex):
        raise ValueError("mesh_faces contain out-of-range vertex indices.")
    return vertices_np, faces_np


def _face_geometry(vertices: np.ndarray, faces: np.ndarray, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=-1)
    area = 0.5 * norm
    normal = cross / np.maximum(norm[:, None], 1e-30)
    return area, normal, tri.mean(axis=1)


def _vertex_geometry(vertices: np.ndarray, faces: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    area, face_normals, _centroids = _face_geometry(vertices, faces)
    vertex_area = np.zeros((vertices.shape[0],), dtype=float)
    vertex_normal = np.zeros_like(vertices, dtype=float)
    for face, face_area, face_normal in zip(faces, area, face_normals, strict=True):
        vertex_area[face] += face_area / 3.0
        vertex_normal[face] += face_normal * face_area
    normal_norm = np.linalg.norm(vertex_normal, axis=-1, keepdims=True)
    vertex_normal = vertex_normal / np.maximum(normal_norm, 1e-30)
    return vertex_area, vertex_normal


def _canonical_edge_counts(faces: np.ndarray, /) -> dict[tuple[int, int], int]:
    undirected = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    canonical = np.sort(undirected.astype(np.int32), axis=1)
    unique, counts = np.unique(canonical, axis=0, return_counts=True)
    return {
        (int(edge[0]), int(edge[1])): int(count)
        for edge, count in zip(unique, counts, strict=True)
    }


def _edge_face_counts(pairs: np.ndarray, edge_counts: dict[tuple[int, int], int], /) -> np.ndarray:
    out = np.zeros((pairs.shape[0],), dtype=np.int32)
    for i, (sender, receiver) in enumerate(pairs):
        key = tuple(sorted((int(sender), int(receiver))))
        out[i] = edge_counts.get(key, 0)
    return out


def _mesh_boundary_metadata(
    pairs: np.ndarray,
    faces: np.ndarray,
    *,
    n_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_counts = _canonical_edge_counts(faces)
    face_counts = _edge_face_counts(pairs, edge_counts)
    boundary_edge_mask = face_counts == 1
    interface_edge_mask = face_counts > 1
    boundary_edges = np.nonzero(boundary_edge_mask)[0].astype(np.int32)
    interface_edges = np.nonzero(interface_edge_mask)[0].astype(np.int32)
    if boundary_edges.size == 0:
        boundary_nodes = np.zeros((0,), dtype=np.int32)
    else:
        boundary_nodes = np.unique(pairs[boundary_edges].reshape((-1,))).astype(np.int32)
    all_nodes = np.arange(n_vertices, dtype=np.int32)
    interior_nodes = np.setdiff1d(all_nodes, boundary_nodes, assume_unique=False).astype(
        np.int32
    )
    return boundary_nodes, interior_nodes, boundary_edges, interface_edges, face_counts


def _node_features_from_mode(
    mode: NodeFeatureMode,
    positions: jnp.ndarray,
    *,
    vertex_area: np.ndarray,
    vertex_normal: np.ndarray,
    boundary_nodes: np.ndarray,
) -> Any:
    if mode == "positions":
        return positions
    if mode == "geometry":
        boundary_mask = np.zeros((positions.shape[0],), dtype=bool)
        boundary_mask[boundary_nodes] = True
        return {
            "positions": positions,
            "normal": jnp.asarray(vertex_normal, dtype=float),
            "area": jnp.asarray(vertex_area[:, None], dtype=float),
            "is_boundary": jnp.asarray(boundary_mask[:, None]),
        }
    if mode is None:
        return None
    if callable(mode):
        return mode(positions)
    raise ValueError("Invalid `node_features` mode.")


def _edge_features_from_mode(
    mode: EdgeFeatureMode,
    positions: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    *,
    face_counts: np.ndarray | None = None,
) -> Any:
    relative = positions[receivers] - positions[senders]
    distance = jnp.linalg.norm(relative, axis=-1, keepdims=True)
    if mode == "relative":
        return relative
    if mode == "distance":
        return distance
    if mode == "geometry":
        if face_counts is None:
            face_counts = np.zeros((int(senders.shape[0]),), dtype=np.int32)
        unit = relative / jnp.maximum(distance, 1e-30)
        face_counts_arr = jnp.asarray(face_counts[:, None], dtype=jnp.int32)
        return {
            "relative": relative,
            "distance": distance,
            "unit": unit,
            "face_count": face_counts_arr,
            "is_boundary": face_counts_arr == 1,
        }
    if mode is None:
        return None
    if callable(mode):
        return mode(positions, senders, receivers)
    raise ValueError("Invalid `edge_features` mode.")


def mesh_to_graph(
    mesh_vertices: Any,
    mesh_faces: Any,
    *,
    node_features: NodeFeatureMode = "positions",
    edge_features: EdgeFeatureMode = "relative",
    add_reverse_edges: bool = True,
    add_self_edges: bool = False,
    globals: Any = None,
    validate: bool = True,
) -> GraphIR:
    """Convert triangular mesh arrays into canonical `GraphIR`."""
    vertices_np, faces_np = _validate_mesh_arrays(mesh_vertices, mesh_faces)
    n_vertex = int(vertices_np.shape[0])

    pairs = _triangle_adjacency(
        faces_np,
        add_reverse_edges=add_reverse_edges,
        add_self_edges=add_self_edges,
        n_vertices=n_vertex,
    )

    senders = jnp.asarray(pairs[:, 0], dtype=jnp.int32)
    receivers = jnp.asarray(pairs[:, 1], dtype=jnp.int32)
    positions = jnp.asarray(vertices_np, dtype=float)
    vertex_area, vertex_normal = _vertex_geometry(vertices_np, faces_np)
    boundary_nodes, _interior_nodes, _boundary_edges, _interface_edges, face_counts = (
        _mesh_boundary_metadata(pairs, faces_np, n_vertices=n_vertex)
    )
    nodes = _node_features_from_mode(
        node_features,
        positions,
        vertex_area=vertex_area,
        vertex_normal=vertex_normal,
        boundary_nodes=boundary_nodes,
    )
    edges = _edge_features_from_mode(
        edge_features,
        positions,
        senders,
        receivers,
        face_counts=face_counts,
    )

    return GraphIR(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals,
        n_node=jnp.asarray([n_vertex], dtype=jnp.int32),
        n_edge=jnp.asarray([int(senders.shape[0])], dtype=jnp.int32),
        validate=validate,
    )


def mesh_to_geometry_graph(
    mesh_vertices: Any,
    mesh_faces: Any,
    *,
    node_features: NodeFeatureMode = "geometry",
    edge_features: EdgeFeatureMode = "geometry",
    add_reverse_edges: bool = True,
    add_self_edges: bool = False,
    globals: Any = None,
    validate: bool = True,
) -> GeometryGraph:
    """Convert triangular mesh arrays into a graph with geometry set metadata."""
    vertices_np, faces_np = _validate_mesh_arrays(mesh_vertices, mesh_faces)
    graph = mesh_to_graph(
        vertices_np,
        faces_np,
        node_features=node_features,
        edge_features=edge_features,
        add_reverse_edges=add_reverse_edges,
        add_self_edges=add_self_edges,
        globals=globals,
        validate=validate,
    )
    pairs = np.stack(
        [np.asarray(graph.senders, dtype=np.int32), np.asarray(graph.receivers, dtype=np.int32)],
        axis=1,
    )
    boundary_nodes, interior_nodes, boundary_edges, interface_edges, _face_counts = (
        _mesh_boundary_metadata(pairs, faces_np, n_vertices=int(vertices_np.shape[0]))
    )
    return GeometryGraph(
        graph,
        boundary_nodes=boundary_nodes,
        interior_nodes=interior_nodes,
        boundary_edges=boundary_edges,
        interface_edges=interface_edges,
    )


def _point_cloud_edges(
    points: np.ndarray,
    /,
    *,
    radius: float | None,
    k: int | None,
    add_self_edges: bool,
) -> np.ndarray:
    if radius is None and k is None:
        raise ValueError("point_cloud_to_graph requires either radius or k.")
    if radius is not None and radius < 0:
        raise ValueError("radius must be non-negative.")
    if k is not None and k < 0:
        raise ValueError("k must be non-negative.")

    n = int(points.shape[0])
    diff = points[None, :, :] - points[:, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    edges: list[tuple[int, int]] = []

    if radius is not None:
        for receiver in range(n):
            for sender in range(n):
                if sender == receiver and not add_self_edges:
                    continue
                if dist[receiver, sender] <= radius:
                    edges.append((sender, receiver))

    if k is not None and k > 0:
        for receiver in range(n):
            order = np.argsort(dist[receiver], kind="stable")
            chosen: list[int] = []
            for sender in order:
                if sender == receiver and not add_self_edges:
                    continue
                chosen.append(int(sender))
                if len(chosen) >= k:
                    break
            edges.extend((sender, receiver) for sender in chosen)

    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.unique(np.asarray(edges, dtype=np.int32), axis=0)


def point_cloud_to_graph(
    points: Any,
    *,
    radius: float | None = None,
    k: int | None = None,
    node_features: NodeFeatureMode = "positions",
    edge_features: EdgeFeatureMode = "geometry",
    add_self_edges: bool = False,
    globals: Any = None,
    validate: bool = True,
) -> GraphIR:
    """Construct a sparse graph from point-cloud radius and/or kNN neighborhoods."""
    points_np = np.asarray(points, dtype=float)
    if points_np.ndim != 2:
        raise ValueError(f"points must have shape (n_point, dim); got {points_np.shape!r}.")
    if int(points_np.shape[0]) == 0:
        raise ValueError("points must contain at least one point.")
    pairs = _point_cloud_edges(
        points_np,
        radius=radius,
        k=k,
        add_self_edges=add_self_edges,
    )
    senders = jnp.asarray(pairs[:, 0], dtype=jnp.int32)
    receivers = jnp.asarray(pairs[:, 1], dtype=jnp.int32)
    positions = jnp.asarray(points_np, dtype=float)
    zeros_area = np.zeros((int(points_np.shape[0]),), dtype=float)
    zeros_normal = np.zeros_like(points_np, dtype=float)
    nodes = _node_features_from_mode(
        node_features,
        positions,
        vertex_area=zeros_area,
        vertex_normal=zeros_normal,
        boundary_nodes=np.zeros((0,), dtype=np.int32),
    )
    edges = _edge_features_from_mode(
        edge_features,
        positions,
        senders,
        receivers,
        face_counts=None,
    )
    return GraphIR(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals,
        n_node=jnp.asarray([int(points_np.shape[0])], dtype=jnp.int32),
        n_edge=jnp.asarray([int(senders.shape[0])], dtype=jnp.int32),
        validate=validate,
    )


def _validate_points(name: str, value: Any, /) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"{name} must have shape (n_point, dim); got {points.shape!r}.")
    if int(points.shape[0]) == 0:
        raise ValueError(f"{name} must contain at least one point.")
    return points


def _periodic_box(periodic_box: Any | None, dim: int, /) -> np.ndarray | None:
    if periodic_box is None:
        return None
    box = np.asarray(periodic_box, dtype=float)
    if box.ndim == 0:
        box = np.full((dim,), float(box), dtype=float)
    if box.shape != (dim,):
        raise ValueError(f"periodic_box must be scalar or shape ({dim},); got {box.shape!r}.")
    if np.any(box <= 0):
        raise ValueError("periodic_box entries must be positive.")
    return box


def _minimum_image(relative: np.ndarray, periodic_box: np.ndarray | None, /) -> np.ndarray:
    if periodic_box is None:
        return relative
    return relative - periodic_box * np.round(relative / periodic_box)


def _validate_query_points(
    source_points: Any,
    target_points: Any | None,
    periodic_box: Any | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
    source = _validate_points("source_points", source_points)
    target = source if target_points is None else _validate_points("target_points", target_points)
    if int(source.shape[1]) != int(target.shape[1]):
        raise ValueError("source_points and target_points must have the same dimension.")
    box = _periodic_box(periodic_box, int(source.shape[1]))
    return source, target, box, target_points is None


def _query_relative_and_distance(
    source: np.ndarray,
    target: np.ndarray,
    periodic_box: np.ndarray | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    relative = target[:, None, :] - source[None, :, :]
    relative = _minimum_image(relative, periodic_box)
    distance = np.linalg.norm(relative, axis=-1)
    return relative, distance


def _radius_query_pairs(
    distance: np.ndarray,
    radius: float,
    /,
    *,
    same_points: bool,
    include_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    r = float(radius)
    if r < 0:
        raise ValueError("radius must be non-negative.")
    target_idx, source_idx = np.nonzero(distance <= r)
    if same_points and not include_self:
        keep = source_idx != target_idx
        source_idx = source_idx[keep]
        target_idx = target_idx[keep]
    return source_idx.astype(np.int32), target_idx.astype(np.int32)


def _knn_query_pairs(
    distance: np.ndarray,
    k: int,
    /,
    *,
    same_points: bool,
    include_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    k_int = int(k)
    if k_int < 0:
        raise ValueError("k must be non-negative.")
    source_parts: list[int] = []
    target_parts: list[int] = []
    if k_int == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    for target_index in range(int(distance.shape[0])):
        order = np.argsort(distance[target_index], kind="stable")
        chosen: list[int] = []
        for source_index in order:
            if same_points and not include_self and int(source_index) == target_index:
                continue
            chosen.append(int(source_index))
            if len(chosen) >= k_int:
                break
        source_parts.extend(chosen)
        target_parts.extend([target_index] * len(chosen))
    return np.asarray(source_parts, dtype=np.int32), np.asarray(target_parts, dtype=np.int32)


def _validate_query_indices(
    name: str,
    value: Any,
    size: int,
    /,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be rank-1; got {arr.shape!r}.")
    if np.any(arr < 0) or np.any(arr >= int(size)):
        raise ValueError(f"{name} entries must be in [0, {size}).")
    return arr


def _combine_query_features(
    source_features: Any | None,
    target_features: Any | None,
    n_source: int,
    n_target: int,
    /,
) -> jnp.ndarray | None:
    if source_features is None and target_features is None:
        return None
    source = None if source_features is None else jnp.asarray(source_features, dtype=float)
    target = None if target_features is None else jnp.asarray(target_features, dtype=float)
    if source is not None and source.ndim == 0:
        raise ValueError("source_features must have a leading point axis.")
    if target is not None and target.ndim == 0:
        raise ValueError("target_features must have a leading point axis.")
    if source is not None and int(source.shape[0]) != n_source:
        raise ValueError(f"source_features leading axis must be {n_source}.")
    if target is not None and int(target.shape[0]) != n_target:
        raise ValueError(f"target_features leading axis must be {n_target}.")

    template = source if source is not None else target
    if template is None:
        return None
    trailing = template.shape[1:]
    dtype = template.dtype
    if source is None:
        source = jnp.zeros((n_source,) + trailing, dtype=dtype)
    if target is None:
        target = jnp.zeros((n_target,) + trailing, dtype=dtype)
    if source.shape[1:] != target.shape[1:]:
        raise ValueError("source_features and target_features must share trailing shape.")
    return jnp.concatenate([source.astype(dtype), target.astype(dtype)], axis=0)


def mollified_kernel_weight(
    distance: Any,
    radius: float,
    /,
    *,
    kind: MollifierKind = "wendland_c2",
) -> jnp.ndarray:
    """Return compact or smooth radial weights for query-graph kernels."""
    r = float(radius)
    if r <= 0:
        raise ValueError("radius must be positive for mollified kernel weights.")
    d = jnp.asarray(distance, dtype=float)
    q = d / r
    if kind == "wendland_c2":
        one_minus_q = jnp.maximum(1.0 - q, 0.0)
        return one_minus_q**4 * (4.0 * q + 1.0)
    if kind == "bump":
        inside = q < 1.0
        q2 = jnp.square(q)
        return jnp.where(inside, jnp.exp(-1.0 / jnp.maximum(1.0 - q2, 1e-30)), 0.0)
    if kind == "hat":
        return jnp.maximum(1.0 - q, 0.0)
    if kind == "gaussian":
        return jnp.exp(-jnp.square(q))
    raise ValueError("Unsupported mollifier kind.")


def query_graph_from_edges(
    source_points: Any,
    target_points: Any,
    source_indices: Any,
    target_indices: Any,
    /,
    *,
    source_features: Any | None = None,
    target_features: Any | None = None,
    source_measure: Any | None = None,
    source_measure_key: str = "quadrature_weight",
    periodic_box: Any | None = None,
    weight_kind: MollifierKind | None = "wendland_c2",
    weight_radius: float | None = None,
    source_type: int = 0,
    target_type: int = 1,
    query_edge_type: int = 0,
    validate: bool = True,
) -> QueryGraph:
    """Build a typed source-to-target query graph from cached neighbor indices."""
    source, target, box, _same_points = _validate_query_points(
        source_points,
        target_points,
        periodic_box,
    )
    source_idx = _validate_query_indices("source_indices", source_indices, int(source.shape[0]))
    target_idx = _validate_query_indices("target_indices", target_indices, int(target.shape[0]))
    if int(source_idx.shape[0]) != int(target_idx.shape[0]):
        raise ValueError("source_indices and target_indices must have the same length.")

    n_source = int(source.shape[0])
    n_target = int(target.shape[0])
    relative = _minimum_image(target[target_idx] - source[source_idx], box)
    distance = np.linalg.norm(relative, axis=-1, keepdims=True)
    unit = relative / np.maximum(distance, 1e-30)
    positions = jnp.asarray(np.concatenate([source, target], axis=0), dtype=float)
    features = _combine_query_features(
        source_features,
        target_features,
        n_source,
        n_target,
    )
    nodes: dict[str, Any] = {
        "positions": positions,
        "type": jnp.concatenate(
            [
                jnp.full((n_source,), int(source_type), dtype=jnp.int32),
                jnp.full((n_target,), int(target_type), dtype=jnp.int32),
            ],
            axis=0,
        ),
        "local_index": jnp.concatenate(
            [
                jnp.arange(n_source, dtype=jnp.int32),
                jnp.arange(n_target, dtype=jnp.int32),
            ],
            axis=0,
        ),
        "is_source": jnp.concatenate(
            [jnp.ones((n_source,), dtype=bool), jnp.zeros((n_target,), dtype=bool)],
            axis=0,
        ),
        "is_target": jnp.concatenate(
            [jnp.zeros((n_source,), dtype=bool), jnp.ones((n_target,), dtype=bool)],
            axis=0,
        ),
    }
    if source_measure is not None:
        source_measure_ = jnp.asarray(source_measure, dtype=float).reshape((-1,))
        if source_measure_.shape != (n_source,):
            raise ValueError(
                "source_measure must have one scalar quadrature weight per source node."
            )
        nodes[str(source_measure_key)] = jnp.concatenate(
            (source_measure_, jnp.zeros((n_target,), dtype=source_measure_.dtype)),
            axis=0,
        )
    if features is not None:
        nodes["features"] = features

    edges: dict[str, Any] = {
        "type": jnp.full((int(source_idx.shape[0]),), int(query_edge_type), dtype=jnp.int32),
        "source_index": jnp.asarray(source_idx, dtype=jnp.int32),
        "target_index": jnp.asarray(target_idx, dtype=jnp.int32),
        "relative": jnp.asarray(relative, dtype=float),
        "distance": jnp.asarray(distance, dtype=float),
        "unit": jnp.asarray(unit, dtype=float),
    }
    if weight_kind is not None:
        if int(source_idx.shape[0]) == 0:
            edges["kernel_weight"] = jnp.zeros((0, 1), dtype=float)
        else:
            radius = float(np.max(distance)) if weight_radius is None else float(weight_radius)
            if radius <= 0:
                radius = 1.0
            edges["kernel_weight"] = mollified_kernel_weight(
                edges["distance"],
                radius,
                kind=weight_kind,
            )

    senders = jnp.asarray(source_idx, dtype=jnp.int32)
    receivers = jnp.asarray(n_source + target_idx, dtype=jnp.int32)
    graph = GraphIR(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=jnp.asarray([n_source + n_target], dtype=jnp.int32),
        n_edge=jnp.asarray([int(source_idx.shape[0])], dtype=jnp.int32),
        validate=validate,
    )
    return QueryGraph(
        graph,
        source_nodes=jnp.arange(n_source, dtype=jnp.int32),
        target_nodes=n_source + jnp.arange(n_target, dtype=jnp.int32),
        query_edges=jnp.arange(int(source_idx.shape[0]), dtype=jnp.int32),
        source_type=source_type,
        target_type=target_type,
        query_edge_type=query_edge_type,
    )


def radius_query_graph(
    source_points: Any,
    target_points: Any | None = None,
    /,
    *,
    radius: float,
    source_features: Any | None = None,
    target_features: Any | None = None,
    source_measure: Any | None = None,
    source_measure_key: str = "quadrature_weight",
    periodic_box: Any | None = None,
    include_self: bool = False,
    weight_kind: MollifierKind | None = "wendland_c2",
    source_type: int = 0,
    target_type: int = 1,
    query_edge_type: int = 0,
    validate: bool = True,
) -> QueryGraph:
    """Build a bipartite query graph from all source points within `radius`."""
    source, target, box, same_points = _validate_query_points(
        source_points,
        target_points,
        periodic_box,
    )
    _relative, distance = _query_relative_and_distance(source, target, box)
    source_idx, target_idx = _radius_query_pairs(
        distance,
        radius,
        same_points=same_points,
        include_self=include_self,
    )
    return query_graph_from_edges(
        source,
        target,
        source_idx,
        target_idx,
        source_features=source_features,
        target_features=target_features,
        source_measure=source_measure,
        source_measure_key=source_measure_key,
        periodic_box=box,
        weight_kind=weight_kind,
        weight_radius=radius,
        source_type=source_type,
        target_type=target_type,
        query_edge_type=query_edge_type,
        validate=validate,
    )


def knn_query_graph(
    source_points: Any,
    target_points: Any | None = None,
    /,
    *,
    k: int,
    source_features: Any | None = None,
    target_features: Any | None = None,
    source_measure: Any | None = None,
    source_measure_key: str = "quadrature_weight",
    periodic_box: Any | None = None,
    include_self: bool = False,
    weight_kind: MollifierKind | None = None,
    weight_radius: float | None = None,
    source_type: int = 0,
    target_type: int = 1,
    query_edge_type: int = 0,
    validate: bool = True,
) -> QueryGraph:
    """Build a bipartite query graph from `k` nearest source points per target."""
    source, target, box, same_points = _validate_query_points(
        source_points,
        target_points,
        periodic_box,
    )
    _relative, distance = _query_relative_and_distance(source, target, box)
    source_idx, target_idx = _knn_query_pairs(
        distance,
        k,
        same_points=same_points,
        include_self=include_self,
    )
    return query_graph_from_edges(
        source,
        target,
        source_idx,
        target_idx,
        source_features=source_features,
        target_features=target_features,
        source_measure=source_measure,
        source_measure_key=source_measure_key,
        periodic_box=box,
        weight_kind=weight_kind,
        weight_radius=weight_radius,
        source_type=source_type,
        target_type=target_type,
        query_edge_type=query_edge_type,
        validate=validate,
    )


def radius_graph(
    points: Any,
    /,
    *,
    radius: float,
    node_features: NodeFeatureMode = "positions",
    edge_features: EdgeFeatureMode = "geometry",
    add_self_edges: bool = False,
    globals: Any = None,
    validate: bool = True,
) -> GraphIR:
    """Construct a homogeneous point-cloud graph from radius neighborhoods."""
    return point_cloud_to_graph(
        points,
        radius=radius,
        node_features=node_features,
        edge_features=edge_features,
        add_self_edges=add_self_edges,
        globals=globals,
        validate=validate,
    )


def knn_graph(
    points: Any,
    /,
    *,
    k: int,
    node_features: NodeFeatureMode = "positions",
    edge_features: EdgeFeatureMode = "geometry",
    add_self_edges: bool = False,
    globals: Any = None,
    validate: bool = True,
) -> GraphIR:
    """Construct a homogeneous point-cloud graph from k-nearest neighborhoods."""
    return point_cloud_to_graph(
        points,
        k=k,
        node_features=node_features,
        edge_features=edge_features,
        add_self_edges=add_self_edges,
        globals=globals,
        validate=validate,
    )


def geometry3d_to_graph(
    geometry: Any,
    *,
    node_features: NodeFeatureMode = "positions",
    edge_features: EdgeFeatureMode = "relative",
    add_reverse_edges: bool = True,
    add_self_edges: bool = False,
    globals: Any = None,
    validate: bool = True,
) -> GraphIR:
    """Convert a Phydrax 3D geometry object into `GraphIR`."""
    return mesh_to_graph(
        geometry.mesh_vertices,
        geometry.mesh_faces,
        node_features=node_features,
        edge_features=edge_features,
        add_reverse_edges=add_reverse_edges,
        add_self_edges=add_self_edges,
        globals=globals,
        validate=validate,
    )


def geometry3d_to_geometry_graph(
    geometry: Any,
    *,
    node_features: NodeFeatureMode = "geometry",
    edge_features: EdgeFeatureMode = "geometry",
    add_reverse_edges: bool = True,
    add_self_edges: bool = False,
    globals: Any = None,
    validate: bool = True,
) -> GeometryGraph:
    """Convert a Phydrax 3D geometry object into a geometry-feature graph bundle."""
    return mesh_to_geometry_graph(
        geometry.mesh_vertices,
        geometry.mesh_faces,
        node_features=node_features,
        edge_features=edge_features,
        add_reverse_edges=add_reverse_edges,
        add_self_edges=add_self_edges,
        globals=globals,
        validate=validate,
    )


__all__ = [
    "GeometryGraph",
    "MollifierKind",
    "QueryGraph",
    "geometry3d_to_geometry_graph",
    "geometry3d_to_graph",
    "knn_graph",
    "knn_query_graph",
    "mesh_to_geometry_graph",
    "mesh_to_graph",
    "mollified_kernel_weight",
    "point_cloud_to_graph",
    "query_graph_from_edges",
    "radius_graph",
    "radius_query_graph",
]
