from __future__ import annotations

from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax._strict import StrictModule

from ..discretization._cell_complex import polygonal_cell_complex
from ..discretization._topology import CellComplexTopology
from ._ir import GraphIR


if TYPE_CHECKING:
    from ..domain.graph import EdgeType, NodeType


def _validate_faces(faces: Any, num_vertices: int | None, /) -> tuple[np.ndarray, int]:
    faces_np = np.asarray(faces, dtype=np.int32)
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(
            f"mesh_faces must have shape (n_face, 3); got {faces_np.shape!r}."
        )
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


def _feature_array(name: str, value: Any, expected: int, /) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.ndim == 0:
        raise ValueError(f"{name} features must have a leading cell axis.")
    if arr.shape[0] != int(expected):
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


@final
class SimplicialComplexGraph(StrictModule):
    """A 2D simplicial complex encoded as a typed `GraphIR`.

    Vertices, edge cells, and triangular face cells are graph nodes. Signed
    boundary/incidence relations are graph edges, so standard graph-domain
    subsets and graph models can operate on cells of any degree.
    """

    graph: GraphIR
    topology: CellComplexTopology
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
        topology: CellComplexTopology,
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
    ) -> None:
        if not isinstance(graph, GraphIR) or not isinstance(
            topology, CellComplexTopology
        ):
            raise TypeError("Simplicial graph requires GraphIR and CellComplexTopology.")
        self.graph = graph
        self.topology = topology
        self.vertex_cells = jnp.asarray(vertex_cells, dtype=jnp.int32)
        self.edge_cells = jnp.asarray(edge_cells, dtype=jnp.int32)
        self.face_cells = jnp.asarray(face_cells, dtype=jnp.int32)
        self.edge_vertices = jnp.asarray(edge_vertices, dtype=jnp.int32)
        self.face_vertices = jnp.asarray(face_vertices, dtype=jnp.int32)
        self.face_edges = jnp.asarray(face_edges, dtype=jnp.int32)
        self.face_edge_signs = jnp.asarray(face_edge_signs, dtype=jnp.float64)
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

    def vertex_cells_component(self) -> NodeType:
        from ..domain.graph import NodeType

        return NodeType(self.vertex_type, name="vertex_cells")

    def edge_cells_component(self) -> NodeType:
        from ..domain.graph import NodeType

        return NodeType(self.edge_type, name="edge_cells")

    def face_cells_component(self) -> NodeType:
        from ..domain.graph import NodeType

        return NodeType(self.face_type, name="face_cells")

    def vertex_to_edge_component(self) -> EdgeType:
        from ..domain.graph import EdgeType

        return EdgeType(self.vertex_to_edge_type, name="vertex_to_edge")

    def edge_to_vertex_component(self) -> EdgeType:
        from ..domain.graph import EdgeType

        return EdgeType(self.edge_to_vertex_type, name="edge_to_vertex")

    def edge_to_face_component(self) -> EdgeType:
        from ..domain.graph import EdgeType

        return EdgeType(self.edge_to_face_type, name="edge_to_face")

    def face_to_edge_component(self) -> EdgeType:
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
    topology = polygonal_cell_complex(faces, None, n_vertex)
    edge_vertices = np.asarray(topology.incidences[0].relation.source_indices).reshape(
        (-1, 2)
    )
    face_edges = np.asarray(topology.incidences[1].relation.source_indices).reshape(
        (-1, 3)
    )
    face_edge_signs = np.asarray(topology.incidences[1].signs).reshape((-1, 3))
    n_edge_cell = edge_vertices.shape[0]
    n_face = faces.shape[0]
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
    e_to_f_start = v_to_e_senders.shape[0]
    e_to_f_edges = e_to_f_start + np.arange(e_to_f_senders.shape[0], dtype=np.int32)
    e_to_v_edges = np.zeros((0,), dtype=np.int32)
    f_to_e_edges = np.zeros((0,), dtype=np.int32)

    if add_reverse_edges:
        e_to_v_start = e_to_f_start + e_to_f_senders.shape[0]
        e_to_v_senders = v_to_e_receivers
        e_to_v_receivers = v_to_e_senders
        e_to_v_edges = e_to_v_start + np.arange(e_to_v_senders.shape[0], dtype=np.int32)
        f_to_e_start = e_to_v_start + e_to_v_senders.shape[0]
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
        "incidence_sign": jnp.asarray(
            np.concatenate(sign_parts, axis=0), dtype=jnp.float64
        ),
        "lower_index": jnp.asarray(
            np.concatenate(lower_index_parts, axis=0), dtype=jnp.int32
        ),
        "upper_index": jnp.asarray(
            np.concatenate(upper_index_parts, axis=0), dtype=jnp.int32
        ),
        "lower_cell_dim": jnp.asarray(
            np.concatenate(lower_dim_parts, axis=0), dtype=jnp.int32
        ),
        "upper_cell_dim": jnp.asarray(
            np.concatenate(upper_dim_parts, axis=0), dtype=jnp.int32
        ),
    }
    graph = GraphIR(
        nodes=nodes,
        edges=edges,
        senders=jnp.asarray(senders, dtype=jnp.int32),
        receivers=jnp.asarray(receivers, dtype=jnp.int32),
        globals=globals,
        n_node=jnp.asarray([n_total], dtype=jnp.int32),
        n_edge=jnp.asarray([senders.shape[0]], dtype=jnp.int32),
        validate=validate,
    )
    return SimplicialComplexGraph(
        graph,
        topology=topology,
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


__all__ = [
    "SimplicialComplexGraph",
    "triangle_mesh_to_simplicial_graph",
]
