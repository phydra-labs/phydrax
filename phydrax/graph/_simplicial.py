from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ._graph import ensure_graph
from ._ir import GraphIR
from ._kernels import segment_sum
from ._typed import edge_type_ids, node_type_ids


FormDegree = Literal[0, 1, 2]


def _validate_faces(faces: Any, num_vertices: int | None, /) -> tuple[np.ndarray, int]:
    faces_np = np.asarray(faces, dtype=np.int32)
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(f"mesh_faces must have shape (n_face, 3); got {faces_np.shape!r}.")
    if faces_np.shape[0] == 0:
        raise ValueError("mesh_faces must contain at least one face.")
    if np.any(faces_np < 0):
        raise ValueError("mesh_faces must not contain negative vertex indices.")

    if num_vertices is None:
        n_vertex = int(faces_np.max()) + 1
    else:
        n_vertex = int(num_vertices)
        if n_vertex < 0:
            raise ValueError("num_vertices must be non-negative.")
    if n_vertex == 0:
        raise ValueError("num_vertices must be positive.")
    if np.any(faces_np >= n_vertex):
        raise ValueError("mesh_faces contain out-of-range vertex indices.")
    return faces_np, n_vertex


def _triangle_edge_cells(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundary = np.stack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=1,
    ).reshape((-1, 2))
    canonical = np.sort(boundary, axis=1)
    edge_vertices, inverse = np.unique(canonical, axis=0, return_inverse=True)
    signs = np.where(boundary[:, 0] == canonical[:, 0], 1, -1).astype(np.float32)
    return (
        edge_vertices.astype(np.int32),
        inverse.reshape((faces.shape[0], 3)).astype(np.int32),
        signs.reshape((faces.shape[0], 3)),
    )


def _feature_array(name: str, value: Any, expected: int, /) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=float)
    if arr.ndim == 0:
        raise ValueError(f"{name} features must have a leading cell axis.")
    if int(arr.shape[0]) != int(expected):
        raise ValueError(
            f"{name} features must have leading axis {expected}; got {arr.shape[0]}."
        )
    return arr


def _combine_cell_features(
    vertex_features: Any | None,
    edge_features: Any | None,
    face_features: Any | None,
    n_vertex: int,
    n_edge: int,
    n_face: int,
    /,
) -> jnp.ndarray | None:
    provided = [
        _feature_array("vertex", vertex_features, n_vertex)
        if vertex_features is not None
        else None,
        _feature_array("edge", edge_features, n_edge)
        if edge_features is not None
        else None,
        _feature_array("face", face_features, n_face)
        if face_features is not None
        else None,
    ]
    template = next((arr for arr in provided if arr is not None), None)
    if template is None:
        return None

    trailing = template.shape[1:]
    dtype = template.dtype
    sizes = (n_vertex, n_edge, n_face)
    parts = []
    for label, arr, size in zip(("vertex", "edge", "face"), provided, sizes, strict=True):
        if arr is None:
            parts.append(jnp.zeros((size,) + trailing, dtype=dtype))
            continue
        if arr.shape[1:] != trailing:
            raise ValueError(
                "vertex, edge, and face features must share trailing shape; "
                f"{label} features have {arr.shape[1:]}, expected {trailing}."
            )
        parts.append(arr.astype(dtype))
    return jnp.concatenate(parts, axis=0)


def _as_feature_mapping(value: Any, /) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"features": value}


class SimplicialComplexGraph(eqx.Module):
    """A 2D simplicial complex encoded as a typed `GraphIR`.

    Vertices, edge cells, and triangular face cells are graph nodes. Signed
    boundary/incidence relations are graph edges, so standard graph-domain
    subsets and graph models can operate on cells of any degree.
    """

    graph: GraphIR
    vertex_cells: jnp.ndarray
    edge_cells: jnp.ndarray
    face_cells: jnp.ndarray
    edge_vertices: jnp.ndarray
    face_vertices: jnp.ndarray
    face_edges: jnp.ndarray
    face_edge_signs: jnp.ndarray
    vertex_to_edge_edges: jnp.ndarray
    edge_to_vertex_edges: jnp.ndarray
    edge_to_face_edges: jnp.ndarray
    face_to_edge_edges: jnp.ndarray
    vertex_type: int = eqx.field(static=True)
    edge_type: int = eqx.field(static=True)
    face_type: int = eqx.field(static=True)
    vertex_to_edge_type: int = eqx.field(static=True)
    edge_to_vertex_type: int = eqx.field(static=True)
    edge_to_face_type: int = eqx.field(static=True)
    face_to_edge_type: int = eqx.field(static=True)

    def __init__(
        self,
        graph: GraphIR,
        /,
        *,
        vertex_cells: Any,
        edge_cells: Any,
        face_cells: Any,
        edge_vertices: Any,
        face_vertices: Any,
        face_edges: Any,
        face_edge_signs: Any,
        vertex_to_edge_edges: Any,
        edge_to_vertex_edges: Any,
        edge_to_face_edges: Any,
        face_to_edge_edges: Any,
        vertex_type: int,
        edge_type: int,
        face_type: int,
        vertex_to_edge_type: int,
        edge_to_vertex_type: int,
        edge_to_face_type: int,
        face_to_edge_type: int,
    ):
        self.graph = graph
        self.vertex_cells = jnp.asarray(vertex_cells, dtype=jnp.int32)
        self.edge_cells = jnp.asarray(edge_cells, dtype=jnp.int32)
        self.face_cells = jnp.asarray(face_cells, dtype=jnp.int32)
        self.edge_vertices = jnp.asarray(edge_vertices, dtype=jnp.int32)
        self.face_vertices = jnp.asarray(face_vertices, dtype=jnp.int32)
        self.face_edges = jnp.asarray(face_edges, dtype=jnp.int32)
        self.face_edge_signs = jnp.asarray(face_edge_signs, dtype=float)
        self.vertex_to_edge_edges = jnp.asarray(vertex_to_edge_edges, dtype=jnp.int32)
        self.edge_to_vertex_edges = jnp.asarray(edge_to_vertex_edges, dtype=jnp.int32)
        self.edge_to_face_edges = jnp.asarray(edge_to_face_edges, dtype=jnp.int32)
        self.face_to_edge_edges = jnp.asarray(face_to_edge_edges, dtype=jnp.int32)
        self.vertex_type = int(vertex_type)
        self.edge_type = int(edge_type)
        self.face_type = int(face_type)
        self.vertex_to_edge_type = int(vertex_to_edge_type)
        self.edge_to_vertex_type = int(edge_to_vertex_type)
        self.edge_to_face_type = int(edge_to_face_type)
        self.face_to_edge_type = int(face_to_edge_type)

    def vertex_cells_component(self):
        from ..domain.graph import NodeType

        return NodeType(self.vertex_type, name="vertex_cells")

    def edge_cells_component(self):
        from ..domain.graph import NodeType

        return NodeType(self.edge_type, name="edge_cells")

    def face_cells_component(self):
        from ..domain.graph import NodeType

        return NodeType(self.face_type, name="face_cells")

    def vertex_to_edge_component(self):
        from ..domain.graph import EdgeType

        return EdgeType(self.vertex_to_edge_type, name="vertex_to_edge")

    def edge_to_vertex_component(self):
        from ..domain.graph import EdgeType

        return EdgeType(self.edge_to_vertex_type, name="edge_to_vertex")

    def edge_to_face_component(self):
        from ..domain.graph import EdgeType

        return EdgeType(self.edge_to_face_type, name="edge_to_face")

    def face_to_edge_component(self):
        from ..domain.graph import EdgeType

        return EdgeType(self.face_to_edge_type, name="face_to_edge")


def triangle_mesh_to_simplicial_graph(
    mesh_faces: Any,
    /,
    *,
    num_vertices: int | None = None,
    vertex_features: Any | None = None,
    edge_features: Any | None = None,
    face_features: Any | None = None,
    globals: Any = None,
    add_reverse_edges: bool = True,
    vertex_type: int = 0,
    edge_type: int = 1,
    face_type: int = 2,
    vertex_to_edge_type: int = 0,
    edge_to_vertex_type: int = 1,
    edge_to_face_type: int = 2,
    face_to_edge_type: int = 3,
    validate: bool = True,
) -> SimplicialComplexGraph:
    """Convert triangular faces into a signed simplicial-complex `GraphIR`."""
    faces, n_vertex = _validate_faces(mesh_faces, num_vertices)
    edge_vertices, face_edges, face_edge_signs = _triangle_edge_cells(faces)
    n_edge_cell = int(edge_vertices.shape[0])
    n_face = int(faces.shape[0])
    n_total = n_vertex + n_edge_cell + n_face

    vertex_cells = np.arange(n_vertex, dtype=np.int32)
    edge_cells = n_vertex + np.arange(n_edge_cell, dtype=np.int32)
    face_cells = n_vertex + n_edge_cell + np.arange(n_face, dtype=np.int32)

    ev_first = edge_vertices[:, 0]
    ev_second = edge_vertices[:, 1]
    edge_node_ids = edge_cells
    v_to_e_senders = np.concatenate([ev_first, ev_second], axis=0)
    v_to_e_receivers = np.concatenate([edge_node_ids, edge_node_ids], axis=0)
    v_to_e_signs = np.concatenate(
        [
            -np.ones((n_edge_cell,), dtype=np.float32),
            np.ones((n_edge_cell,), dtype=np.float32),
        ],
        axis=0,
    )
    v_to_e_lower = v_to_e_senders
    v_to_e_upper = np.concatenate(
        [
            np.arange(n_edge_cell, dtype=np.int32),
            np.arange(n_edge_cell, dtype=np.int32),
        ],
        axis=0,
    )

    face_ids = np.repeat(np.arange(n_face, dtype=np.int32), 3)
    e_to_f_lower = face_edges.reshape((-1,))
    e_to_f_upper = face_ids
    e_to_f_senders = edge_cells[e_to_f_lower]
    e_to_f_receivers = face_cells[e_to_f_upper]
    e_to_f_signs = face_edge_signs.reshape((-1,)).astype(np.float32)

    senders_parts = [v_to_e_senders, e_to_f_senders]
    receivers_parts = [v_to_e_receivers, e_to_f_receivers]
    type_parts = [
        np.full((v_to_e_senders.shape[0],), int(vertex_to_edge_type), dtype=np.int32),
        np.full((e_to_f_senders.shape[0],), int(edge_to_face_type), dtype=np.int32),
    ]
    sign_parts = [v_to_e_signs, e_to_f_signs]
    lower_index_parts = [v_to_e_lower, e_to_f_lower]
    upper_index_parts = [v_to_e_upper, e_to_f_upper]
    lower_dim_parts = [
        np.zeros((v_to_e_senders.shape[0],), dtype=np.int32),
        np.ones((e_to_f_senders.shape[0],), dtype=np.int32),
    ]
    upper_dim_parts = [
        np.ones((v_to_e_senders.shape[0],), dtype=np.int32),
        np.full((e_to_f_senders.shape[0],), 2, dtype=np.int32),
    ]

    v_to_e_edges = np.arange(v_to_e_senders.shape[0], dtype=np.int32)
    e_to_f_start = int(v_to_e_senders.shape[0])
    e_to_f_edges = e_to_f_start + np.arange(e_to_f_senders.shape[0], dtype=np.int32)
    e_to_v_edges = np.zeros((0,), dtype=np.int32)
    f_to_e_edges = np.zeros((0,), dtype=np.int32)

    if add_reverse_edges:
        e_to_v_start = e_to_f_start + int(e_to_f_senders.shape[0])
        e_to_v_senders = v_to_e_receivers
        e_to_v_receivers = v_to_e_senders
        e_to_v_edges = e_to_v_start + np.arange(e_to_v_senders.shape[0], dtype=np.int32)
        f_to_e_start = e_to_v_start + int(e_to_v_senders.shape[0])
        f_to_e_senders = e_to_f_receivers
        f_to_e_receivers = e_to_f_senders
        f_to_e_edges = f_to_e_start + np.arange(f_to_e_senders.shape[0], dtype=np.int32)

        senders_parts.extend([e_to_v_senders, f_to_e_senders])
        receivers_parts.extend([e_to_v_receivers, f_to_e_receivers])
        type_parts.extend(
            [
                np.full(
                    (e_to_v_senders.shape[0],),
                    int(edge_to_vertex_type),
                    dtype=np.int32,
                ),
                np.full(
                    (f_to_e_senders.shape[0],),
                    int(face_to_edge_type),
                    dtype=np.int32,
                ),
            ]
        )
        sign_parts.extend([v_to_e_signs, e_to_f_signs])
        lower_index_parts.extend([v_to_e_lower, e_to_f_lower])
        upper_index_parts.extend([v_to_e_upper, e_to_f_upper])
        lower_dim_parts.extend(
            [
                np.zeros((e_to_v_senders.shape[0],), dtype=np.int32),
                np.ones((f_to_e_senders.shape[0],), dtype=np.int32),
            ]
        )
        upper_dim_parts.extend(
            [
                np.ones((e_to_v_senders.shape[0],), dtype=np.int32),
                np.full((f_to_e_senders.shape[0],), 2, dtype=np.int32),
            ]
        )

    features = _combine_cell_features(
        vertex_features,
        edge_features,
        face_features,
        n_vertex,
        n_edge_cell,
        n_face,
    )
    nodes: dict[str, Any] = {
        "type": jnp.concatenate(
            [
                jnp.full((n_vertex,), int(vertex_type), dtype=jnp.int32),
                jnp.full((n_edge_cell,), int(edge_type), dtype=jnp.int32),
                jnp.full((n_face,), int(face_type), dtype=jnp.int32),
            ],
            axis=0,
        ),
        "cell_dim": jnp.concatenate(
            [
                jnp.zeros((n_vertex,), dtype=jnp.int32),
                jnp.ones((n_edge_cell,), dtype=jnp.int32),
                jnp.full((n_face,), 2, dtype=jnp.int32),
            ],
            axis=0,
        ),
        "local_index": jnp.concatenate(
            [
                jnp.arange(n_vertex, dtype=jnp.int32),
                jnp.arange(n_edge_cell, dtype=jnp.int32),
                jnp.arange(n_face, dtype=jnp.int32),
            ],
            axis=0,
        ),
    }
    if features is not None:
        nodes["features"] = features

    senders = np.concatenate(senders_parts, axis=0)
    receivers = np.concatenate(receivers_parts, axis=0)
    edge_type_arr = np.concatenate(type_parts, axis=0)
    edges = {
        "type": jnp.asarray(edge_type_arr, dtype=jnp.int32),
        "incidence_sign": jnp.asarray(np.concatenate(sign_parts, axis=0), dtype=float),
        "lower_index": jnp.asarray(np.concatenate(lower_index_parts, axis=0), dtype=jnp.int32),
        "upper_index": jnp.asarray(np.concatenate(upper_index_parts, axis=0), dtype=jnp.int32),
        "lower_cell_dim": jnp.asarray(np.concatenate(lower_dim_parts, axis=0), dtype=jnp.int32),
        "upper_cell_dim": jnp.asarray(np.concatenate(upper_dim_parts, axis=0), dtype=jnp.int32),
    }
    graph = GraphIR(
        nodes=nodes,
        edges=edges,
        senders=jnp.asarray(senders, dtype=jnp.int32),
        receivers=jnp.asarray(receivers, dtype=jnp.int32),
        globals=globals,
        n_node=jnp.asarray([n_total], dtype=jnp.int32),
        n_edge=jnp.asarray([int(senders.shape[0])], dtype=jnp.int32),
        validate=validate,
    )
    return SimplicialComplexGraph(
        graph,
        vertex_cells=vertex_cells,
        edge_cells=edge_cells,
        face_cells=face_cells,
        edge_vertices=edge_vertices,
        face_vertices=faces,
        face_edges=face_edges,
        face_edge_signs=face_edge_signs,
        vertex_to_edge_edges=v_to_e_edges,
        edge_to_vertex_edges=e_to_v_edges,
        edge_to_face_edges=e_to_f_edges,
        face_to_edge_edges=f_to_e_edges,
        vertex_type=vertex_type,
        edge_type=edge_type,
        face_type=face_type,
        vertex_to_edge_type=vertex_to_edge_type,
        edge_to_vertex_type=edge_to_vertex_type,
        edge_to_face_type=edge_to_face_type,
        face_to_edge_type=face_to_edge_type,
    )


def _as_array(name: str, value: Any, /) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=float)
    if arr.ndim == 0:
        raise ValueError(f"{name} must have a leading cell axis.")
    return arr


def _node_field(graph: GraphIR, input_key: str | None, /) -> jnp.ndarray:
    if graph.nodes is None:
        raise ValueError("SimplicialHodgeLaplacian requires node/cell features.")
    if input_key is None:
        if isinstance(graph.nodes, Mapping):
            raise TypeError(
                "mapping-valued simplicial-complex nodes require input_key."
            )
        return _as_array("nodes", graph.nodes)
    if not isinstance(graph.nodes, Mapping):
        raise TypeError("input_key requires mapping-valued simplicial-complex nodes.")
    if input_key not in graph.nodes:
        raise KeyError(f"Graph nodes do not contain input_key {input_key!r}.")
    return _as_array(f"nodes[{input_key!r}]", graph.nodes[input_key])


def _edge_signs(
    graph: GraphIR,
    sign_key: str,
    edge_type_key: str,
    wanted_type: int,
    /,
) -> jnp.ndarray:
    if not isinstance(graph.edges, Mapping):
        raise TypeError("SimplicialHodgeLaplacian requires mapping-valued graph edges.")
    if sign_key not in graph.edges:
        raise KeyError(f"Graph edges do not contain sign_key {sign_key!r}.")
    signs = jnp.asarray(graph.edges[sign_key], dtype=float).reshape((-1,))
    types = edge_type_ids(graph, type_key=edge_type_key)
    keep = types == int(wanted_type)
    if graph.edge_mask is not None:
        keep = keep & graph.edge_mask
    return jnp.where(keep, signs, 0.0)


def _broadcast_weight(weight: jnp.ndarray, values: jnp.ndarray, /) -> jnp.ndarray:
    while weight.ndim < values.ndim:
        weight = jnp.expand_dims(weight, axis=-1)
    return weight


def _incidence_apply(
    graph: GraphIR,
    values: jnp.ndarray,
    edge_type: int,
    /,
    *,
    sign_key: str,
    edge_type_key: str,
) -> jnp.ndarray:
    if graph.senders is None or graph.receivers is None:
        raise ValueError("SimplicialHodgeLaplacian requires explicit senders/receivers.")
    signs = _edge_signs(graph, sign_key, edge_type_key, edge_type)
    messages = values[graph.senders] * _broadcast_weight(signs, values[graph.senders])
    return segment_sum(messages, graph.receivers, int(values.shape[0]))


def _mask_cell_type(
    graph: GraphIR,
    values: jnp.ndarray,
    type_id: int,
    /,
    *,
    node_type_key: str,
) -> jnp.ndarray:
    keep = node_type_ids(graph, type_key=node_type_key) == int(type_id)
    if graph.node_mask is not None:
        keep = keep & graph.node_mask
    return values * _broadcast_weight(keep.astype(values.dtype), values)


def _with_node_output(graph: GraphIR, value: jnp.ndarray, output_key: str | None, /) -> Any:
    if output_key is None:
        return value
    nodes = _as_feature_mapping(graph.nodes)
    nodes[output_key] = value
    return nodes


class SimplicialHodgeLaplacian(eqx.Module):
    """Unweighted Hodge Laplacian on 0-, 1-, or 2-forms.

    The operator reads a cell field from graph nodes and applies
    `L_k = delta_{k+1} d_k + d_{k-1} delta_k` using signed incidence edges.
    Results are written back to the selected cell degree; all other cells are
    zero in the returned payload.
    """

    form_degree: int = eqx.field(static=True)
    input_key: str | None = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)
    node_type_key: str = eqx.field(static=True)
    edge_type_key: str = eqx.field(static=True)
    sign_key: str = eqx.field(static=True)
    vertex_type: int = eqx.field(static=True)
    edge_type: int = eqx.field(static=True)
    face_type: int = eqx.field(static=True)
    vertex_to_edge_type: int = eqx.field(static=True)
    edge_to_vertex_type: int = eqx.field(static=True)
    edge_to_face_type: int = eqx.field(static=True)
    face_to_edge_type: int = eqx.field(static=True)

    def __init__(
        self,
        form_degree: FormDegree,
        /,
        *,
        input_key: str | None = None,
        output_key: str | None = None,
        node_type_key: str = "type",
        edge_type_key: str = "type",
        sign_key: str = "incidence_sign",
        vertex_type: int = 0,
        edge_type: int = 1,
        face_type: int = 2,
        vertex_to_edge_type: int = 0,
        edge_to_vertex_type: int = 1,
        edge_to_face_type: int = 2,
        face_to_edge_type: int = 3,
    ):
        if form_degree not in (0, 1, 2):
            raise ValueError("form_degree must be 0, 1, or 2.")
        self.form_degree = int(form_degree)
        self.input_key = input_key
        self.output_key = output_key
        self.node_type_key = str(node_type_key)
        self.edge_type_key = str(edge_type_key)
        self.sign_key = str(sign_key)
        self.vertex_type = int(vertex_type)
        self.edge_type = int(edge_type)
        self.face_type = int(face_type)
        self.vertex_to_edge_type = int(vertex_to_edge_type)
        self.edge_to_vertex_type = int(edge_to_vertex_type)
        self.edge_to_face_type = int(edge_to_face_type)
        self.face_to_edge_type = int(face_to_edge_type)

    def _d(self, graph: GraphIR, values: jnp.ndarray, edge_type: int, /) -> jnp.ndarray:
        return _incidence_apply(
            graph,
            values,
            edge_type,
            sign_key=self.sign_key,
            edge_type_key=self.edge_type_key,
        )

    def _delta(
        self,
        graph: GraphIR,
        values: jnp.ndarray,
        edge_type: int,
        /,
    ) -> jnp.ndarray:
        return _incidence_apply(
            graph,
            values,
            edge_type,
            sign_key=self.sign_key,
            edge_type_key=self.edge_type_key,
        )

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        values = _node_field(graph, self.input_key)
        num_cells = int(node_type_ids(graph, type_key=self.node_type_key).shape[0])
        if int(values.shape[0]) != num_cells:
            raise ValueError(
                "SimplicialHodgeLaplacian input leading axis must match graph cells."
            )

        if self.form_degree == 0:
            x = _mask_cell_type(
                graph,
                values,
                self.vertex_type,
                node_type_key=self.node_type_key,
            )
            out = self._delta(graph, self._d(graph, x, self.vertex_to_edge_type), self.edge_to_vertex_type)
            out = _mask_cell_type(graph, out, self.vertex_type, node_type_key=self.node_type_key)
        elif self.form_degree == 1:
            x = _mask_cell_type(
                graph,
                values,
                self.edge_type,
                node_type_key=self.node_type_key,
            )
            lower = self._d(
                graph,
                self._delta(graph, x, self.edge_to_vertex_type),
                self.vertex_to_edge_type,
            )
            upper = self._delta(
                graph,
                self._d(graph, x, self.edge_to_face_type),
                self.face_to_edge_type,
            )
            out = _mask_cell_type(
                graph,
                lower + upper,
                self.edge_type,
                node_type_key=self.node_type_key,
            )
        else:
            x = _mask_cell_type(
                graph,
                values,
                self.face_type,
                node_type_key=self.node_type_key,
            )
            out = self._d(
                graph,
                self._delta(graph, x, self.face_to_edge_type),
                self.edge_to_face_type,
            )
            out = _mask_cell_type(graph, out, self.face_type, node_type_key=self.node_type_key)

        return graph.replace(nodes=_with_node_output(graph, out, self.output_key), validate=False)


__all__ = [
    "FormDegree",
    "SimplicialComplexGraph",
    "SimplicialHodgeLaplacian",
    "triangle_mesh_to_simplicial_graph",
]
