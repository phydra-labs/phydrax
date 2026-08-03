from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from ._geometry import (
    _face_geometry,
    _triangle_adjacency,
    _validate_mesh_arrays,
    _vertex_geometry,
    GeometryGraph,
    mesh_to_geometry_graph,
)
from ._graph import ensure_graph
from ._ir import GraphIR
from ._kernels import segment_sum


MeshLaplacianSign = Literal["neighbor_minus_self", "self_minus_neighbor"]


def _tree_leading_size(tree: Any) -> int:
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        raise ValueError("Feature tree must contain at least one array leaf.")
    return int(jnp.asarray(leaves[0]).shape[0])


def _tree_index(tree: Any, index: jnp.ndarray, /) -> Any:
    return jtu.tree_map(lambda x: x[index], tree)


def _multiply_leaf(value: Any, weight: Any, /) -> jnp.ndarray:
    value_arr = jnp.asarray(value)
    weight_arr = jnp.asarray(weight)
    if (
        value_arr.ndim != weight_arr.ndim
        and value_arr.ndim > 0
        and weight_arr.ndim > 0
        and int(value_arr.shape[0]) == int(weight_arr.shape[0])
    ):
        while value_arr.ndim < weight_arr.ndim:
            value_arr = jnp.expand_dims(value_arr, axis=-1)
        while weight_arr.ndim < value_arr.ndim:
            weight_arr = jnp.expand_dims(weight_arr, axis=-1)
    return value_arr * weight_arr


def _multiply_tree(tree: Any, weight: Any, /) -> Any:
    if jtu.tree_structure(tree) == jtu.tree_structure(weight):
        return jtu.tree_map(
            _multiply_leaf,
            tree,
            weight,
        )
    return jtu.tree_map(lambda x: _multiply_leaf(x, weight), tree)


def _mask_tree(tree: Any, mask: jnp.ndarray | None, /) -> Any:
    if mask is None:
        return tree
    return jtu.tree_map(
        lambda x: _multiply_leaf(x, mask.astype(x.dtype)),
        tree,
    )


def _tree_segment_sum(tree: Any, segment_ids: jnp.ndarray, num_segments: int, /) -> Any:
    return jtu.tree_map(lambda x: segment_sum(x, segment_ids, num_segments), tree)


def _num_nodes(graph: GraphIR, nodes: Any, /) -> int:
    if graph.node_mask is not None:
        return int(graph.node_mask.shape[0])
    return _tree_leading_size(nodes)


def _as_feature_mapping(name: str, value: Any, /) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"features": value}


def _cotangent_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray, /) -> float:
    u = b - a
    v = c - a
    cross_norm = float(np.linalg.norm(np.cross(u, v)))
    if cross_norm <= 1e-30:
        return 0.0
    return float(np.dot(u, v) / cross_norm)


def _cotangent_weight_map(
    vertices: np.ndarray,
    faces: np.ndarray,
    /,
) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = {}
    for face in faces:
        i, j, k = (int(face[0]), int(face[1]), int(face[2]))
        vi, vj, vk = vertices[[i, j, k]]
        contributions = (
            ((j, k), _cotangent_at_vertex(vi, vj, vk)),
            ((k, i), _cotangent_at_vertex(vj, vk, vi)),
            ((i, j), _cotangent_at_vertex(vk, vi, vj)),
        )
        for edge, cotangent in contributions:
            key = (min(edge), max(edge))
            weights[key] = weights.get(key, 0.0) + 0.5 * cotangent
    return weights


def _cotangent_weights_for_pairs(
    pairs: np.ndarray,
    weight_map: dict[tuple[int, int], float],
    /,
) -> np.ndarray:
    weights = np.zeros((pairs.shape[0],), dtype=float)
    for i, (sender, receiver) in enumerate(pairs):
        if int(sender) == int(receiver):
            continue
        key = tuple(sorted((int(sender), int(receiver))))
        weights[i] = weight_map.get(key, 0.0)
    return weights


def mesh_face_areas(mesh_vertices: Any, mesh_faces: Any, /) -> jnp.ndarray:
    """Return the area of each triangular mesh face."""
    vertices, faces = _validate_mesh_arrays(mesh_vertices, mesh_faces)
    area, _normal, _centroid = _face_geometry(vertices, faces)
    return jnp.asarray(area, dtype=float)


def mesh_face_normals(mesh_vertices: Any, mesh_faces: Any, /) -> jnp.ndarray:
    """Return unit normals for triangular mesh faces."""
    vertices, faces = _validate_mesh_arrays(mesh_vertices, mesh_faces)
    _area, normal, _centroid = _face_geometry(vertices, faces)
    return jnp.asarray(normal, dtype=float)


def mesh_lumped_vertex_areas(mesh_vertices: Any, mesh_faces: Any, /) -> jnp.ndarray:
    """Return barycentric lumped vertex areas for a triangular mesh."""
    vertices, faces = _validate_mesh_arrays(mesh_vertices, mesh_faces)
    area, _normal = _vertex_geometry(vertices, faces)
    return jnp.asarray(area, dtype=float)


def mesh_vertex_normals(mesh_vertices: Any, mesh_faces: Any, /) -> jnp.ndarray:
    """Return area-weighted unit vertex normals for a triangular mesh."""
    vertices, faces = _validate_mesh_arrays(mesh_vertices, mesh_faces)
    _area, normal = _vertex_geometry(vertices, faces)
    return jnp.asarray(normal, dtype=float)


def mesh_cotangent_weights(
    mesh_vertices: Any,
    mesh_faces: Any,
    /,
    *,
    add_reverse_edges: bool = True,
    add_self_edges: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return directed edge indices and cotangent FEM weights.

    The returned weights use the standard half-cotangent convention
    `0.5 * (cot(alpha) + cot(beta))` for each undirected mesh edge. Boundary
    edges have a single opposite angle contribution.
    """
    vertices, faces = _validate_mesh_arrays(mesh_vertices, mesh_faces)
    pairs = _triangle_adjacency(
        faces,
        add_reverse_edges=add_reverse_edges,
        add_self_edges=add_self_edges,
        n_vertices=int(vertices.shape[0]),
    )
    weights = _cotangent_weights_for_pairs(pairs, _cotangent_weight_map(vertices, faces))
    return (
        jnp.asarray(pairs[:, 0], dtype=jnp.int32),
        jnp.asarray(pairs[:, 1], dtype=jnp.int32),
        jnp.asarray(weights, dtype=float),
    )


def mesh_to_cotangent_graph(
    mesh_vertices: Any,
    mesh_faces: Any,
    /,
    *,
    weight_key: str = "cotangent_weight",
    mass_key: str = "mass",
    add_reverse_edges: bool = True,
    add_self_edges: bool = False,
    globals: Any = None,
    validate: bool = True,
) -> GeometryGraph:
    """Convert a triangular mesh into a geometry graph with cotangent data.

    The graph carries `edges[weight_key]` and `nodes[mass_key]`, so
    `MeshCotangentLaplacian` can be applied without recomputing geometry.
    """
    vertices, faces = _validate_mesh_arrays(mesh_vertices, mesh_faces)
    bundle = mesh_to_geometry_graph(
        vertices,
        faces,
        node_features="geometry",
        edge_features="geometry",
        add_reverse_edges=add_reverse_edges,
        add_self_edges=add_self_edges,
        globals=globals,
        validate=validate,
    )
    pairs = np.stack(
        [
            np.asarray(bundle.graph.senders, dtype=np.int32),
            np.asarray(bundle.graph.receivers, dtype=np.int32),
        ],
        axis=1,
    )
    weight = _cotangent_weights_for_pairs(pairs, _cotangent_weight_map(vertices, faces))
    mass = mesh_lumped_vertex_areas(vertices, faces)

    nodes = _as_feature_mapping("nodes", bundle.graph.nodes)
    nodes[mass_key] = mass
    edges = _as_feature_mapping("edges", bundle.graph.edges)
    edges[weight_key] = jnp.asarray(weight, dtype=float)

    graph = bundle.graph.replace(nodes=nodes, edges=edges, validate=validate)
    return GeometryGraph(
        graph,
        boundary_nodes=bundle.boundary_nodes,
        interior_nodes=bundle.interior_nodes,
        boundary_edges=bundle.boundary_edges,
        interface_edges=bundle.interface_edges,
    )


class MeshCotangentLaplacian(eqx.Module):
    """Mass-aware cotangent Laplacian block for mesh graphs.

    The block reads a node field, applies the sparse cotangent stencil, and
    writes the result as the graph's node payload or into `output_key`.
    """

    weight: Any
    mass: Any
    weight_key: str | None = eqx.field(static=True)
    mass_key: str | None = eqx.field(static=True)
    input_key: str | None = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)
    normalize_by_mass: bool = eqx.field(static=True)
    sign: MeshLaplacianSign = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        sign: MeshLaplacianSign,
        weight: Any = None,
        mass: Any = None,
        weight_key: str | None = "cotangent_weight",
        mass_key: str | None = "mass",
        input_key: str | None = None,
        output_key: str | None = None,
        normalize_by_mass: bool = True,
    ):
        if sign not in ("neighbor_minus_self", "self_minus_neighbor"):
            raise ValueError(
                "MeshCotangentLaplacian sign must be 'neighbor_minus_self' "
                "or 'self_minus_neighbor'."
            )
        self.weight = weight
        self.mass = mass
        self.weight_key = weight_key
        self.mass_key = mass_key
        self.input_key = input_key
        self.output_key = output_key
        self.normalize_by_mass = bool(normalize_by_mass)
        self.sign = sign

    def _node_field(self, graph: GraphIR, /) -> Any:
        if graph.nodes is None:
            raise ValueError("MeshCotangentLaplacian requires node features.")
        if self.input_key is None:
            return graph.nodes
        if not isinstance(graph.nodes, Mapping):
            raise TypeError("input_key requires mapping-valued graph nodes.")
        if self.input_key not in graph.nodes:
            raise KeyError(f"Graph nodes do not contain input_key {self.input_key!r}.")
        return graph.nodes[self.input_key]

    def _edge_weight(self, graph: GraphIR, /) -> Any:
        if self.weight is not None:
            return self.weight
        if self.weight_key is None:
            if graph.senders is None:
                raise ValueError("Cannot infer unit weights without graph edges.")
            return jnp.ones((graph.senders.shape[0],), dtype=float)
        if not isinstance(graph.edges, Mapping):
            raise TypeError("weight_key requires mapping-valued graph edges.")
        if self.weight_key not in graph.edges:
            raise KeyError(f"Graph edges do not contain weight_key {self.weight_key!r}.")
        return graph.edges[self.weight_key]

    def _mass(self, graph: GraphIR, /) -> Any:
        if self.mass is not None:
            return self.mass
        if self.mass_key is None:
            raise ValueError("mass_key=None requires normalize_by_mass=False or mass=...")
        if not isinstance(graph.nodes, Mapping):
            raise TypeError("mass_key requires mapping-valued graph nodes.")
        if self.mass_key not in graph.nodes:
            raise KeyError(f"Graph nodes do not contain mass_key {self.mass_key!r}.")
        return graph.nodes[self.mass_key]

    def _with_output(self, graph: GraphIR, value: Any, /) -> Any:
        if self.output_key is None:
            return value
        nodes = _as_feature_mapping("nodes", graph.nodes)
        nodes[self.output_key] = value
        return nodes

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.senders is None or graph.receivers is None:
            raise ValueError(
                "MeshCotangentLaplacian requires explicit senders/receivers."
            )

        nodes = self._node_field(graph)
        sent = _tree_index(nodes, graph.senders)
        recv = _tree_index(nodes, graph.receivers)
        if self.sign == "neighbor_minus_self":
            messages = jtu.tree_map(lambda s, r: s - r, sent, recv)
        else:
            messages = jtu.tree_map(lambda s, r: r - s, sent, recv)

        messages = _multiply_tree(messages, self._edge_weight(graph))
        messages = _mask_tree(messages, graph.edge_mask)
        out = _tree_segment_sum(messages, graph.receivers, _num_nodes(graph, nodes))

        if self.normalize_by_mass:
            mass = jnp.asarray(self._mass(graph))
            inverse_mass = jnp.where(mass != 0, 1.0 / mass, 0.0)
            out = _multiply_tree(out, inverse_mass)

        out = _mask_tree(out, graph.node_mask)
        return graph.replace(nodes=self._with_output(graph, out), validate=False)


__all__ = [
    "MeshCotangentLaplacian",
    "MeshLaplacianSign",
    "mesh_cotangent_weights",
    "mesh_face_areas",
    "mesh_face_normals",
    "mesh_lumped_vertex_areas",
    "mesh_to_cotangent_graph",
    "mesh_vertex_normals",
]
