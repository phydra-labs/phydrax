from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from phydrax._strict import StrictModule

from ..sparse import EdgeRelation, gather_routes
from ..sparse._streamed import StreamedRelationPlan
from ..typing import parse
from ._graph import ensure_graph
from ._ir import GraphIR
from ._route_payload import declared_payload, require_route_local, RouteLocal
from ._typed import GraphFlow


def _as_feature_mapping(value: Any, /) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"features": value}


def _mapping_value(payload: Any, key: str, kind: str, /) -> jnp.ndarray:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{kind} key access requires mapping-valued graph {kind}.")
    if key not in payload:
        raise KeyError(f"Graph {kind} payload does not contain key {key!r}.")
    return jnp.asarray(payload[key])


def _positions(graph: GraphIR, position_key: str, /) -> jnp.ndarray:
    pos = _mapping_value(graph.nodes, position_key, "nodes")
    if pos.ndim != 2:
        raise ValueError(
            f"Graph node positions must have shape (n, dim); got {pos.shape!r}."
        )
    return pos


def _node_scalar(graph: GraphIR, input_key: str | None, /) -> jnp.ndarray:
    if graph.nodes is None:
        raise ValueError("EquivariantGraphConvolution requires node features.")
    if input_key is None:
        if isinstance(graph.nodes, Mapping):
            raise TypeError("mapping-valued graph nodes require input_key.")
        arr = jnp.asarray(graph.nodes, dtype=jnp.float64)
    else:
        arr = _mapping_value(graph.nodes, input_key, "nodes").astype("float64")
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(
            f"Node scalar field must be rank-1 or rank-2; got {arr.shape!r}."
        )
    return arr


def _edge_weight(graph: GraphIR, edge_weight_key: str | None, /) -> jnp.ndarray | None:
    if edge_weight_key is None:
        return None
    if not isinstance(graph.edges, Mapping):
        raise TypeError("edge_weight_key requires mapping-valued graph edges.")
    if edge_weight_key not in graph.edges:
        raise KeyError(f"Graph edges do not contain edge_weight_key {edge_weight_key!r}.")
    weight = jnp.asarray(graph.edges[edge_weight_key], dtype=jnp.float64)
    if weight.ndim == 2 and weight.shape[1] == 1:
        weight = weight[:, 0]
    if weight.ndim != 1:
        raise ValueError("edge weights must have shape (n_edge,) or (n_edge, 1).")
    return weight


def _oriented_relation(
    graph: GraphIR, flow: GraphFlow, node_count: int, /
) -> EdgeRelation:
    if graph.senders is None or graph.receivers is None:
        raise ValueError(
            "Equivariant graph operators require explicit senders/receivers."
        )
    flow = parse(flow, GraphFlow, "flow")
    return graph.edge_relation(node_count=node_count, flow=flow)


def _relative_geometry(
    graph: GraphIR,
    /,
    *,
    position_key: str,
    flow: GraphFlow,
    eps: float,
) -> tuple[EdgeRelation, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    pos = _positions(graph, position_key)
    relation = _oriented_relation(graph, flow, pos.shape[0])
    relative = gather_routes(relation.transpose(), pos) - gather_routes(relation, pos)
    squared_distance = jnp.sum(jnp.square(relative), axis=-1, keepdims=True)
    distance = jnp.sqrt(jnp.maximum(squared_distance, float(eps)))
    unit = relative / distance
    return relation, relative, distance, unit


def euclidean_edge_features(
    graph: GraphIR,
    /,
    *,
    position_key: str = "positions",
    relative_key: str = "relative",
    distance_key: str = "distance",
    unit_key: str = "unit",
    squared_distance_key: str = "squared_distance",
    flow: GraphFlow = "source_to_target",
    eps: float = 1e-30,
) -> GraphIR:
    """Attach Euclidean relative, distance, unit, and squared-distance edge features."""
    graph = ensure_graph(graph, validate=False)
    _relation, relative, distance, unit = _relative_geometry(
        graph,
        position_key=position_key,
        flow=flow,
        eps=eps,
    )
    edges = _as_feature_mapping(graph.edges)
    edges[relative_key] = relative
    edges[distance_key] = distance
    edges[unit_key] = unit
    edges[squared_distance_key] = jnp.sum(jnp.square(relative), axis=-1, keepdims=True)
    return graph.replace(edges=edges, validate=False)


def gaussian_radial_basis(
    distance: Any,
    centers: Any,
    /,
    *,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """Evaluate Gaussian radial basis features from pairwise distances."""
    d = jnp.asarray(distance, dtype=jnp.float64)
    c = jnp.asarray(centers, dtype=jnp.float64).reshape((1, -1))
    if d.ndim == 1:
        d = d[:, None]
    if d.ndim != 2 or d.shape[1] != 1:
        raise ValueError("distance must have shape (n_edge,) or (n_edge, 1).")
    return jnp.exp(-float(gamma) * jnp.square(d - c))


def _convolution_message(
    convolution: EquivariantGraphConvolution,
    source: tuple[jnp.ndarray, jnp.ndarray],
    receiver: tuple[jnp.ndarray, jnp.ndarray],
    edge: tuple[Any, jnp.ndarray | None],
    /,
) -> dict[str, jnp.ndarray]:
    """Scalar and relative-vector messages of one route, with its weight magnitude."""
    source_position, sent = source
    receiver_position, received = receiver
    edges, edge_weight = edge
    relative = receiver_position - source_position
    weight = jnp.ones((), dtype=sent.dtype)
    if edge_weight is not None:
        weight = weight * edge_weight.astype(weight.dtype)
    if convolution.radial_fn is not None:
        squared_distance = jnp.sum(jnp.square(relative), axis=-1, keepdims=True)
        distance = jnp.sqrt(jnp.maximum(squared_distance, convolution.eps))
        unit = relative / distance
        radial = jnp.asarray(
            convolution.radial_fn(edges, distance, unit, sent, received),
            dtype=sent.dtype,
        )
        if radial.shape == (1,):
            radial = radial[0]
        if radial.shape != ():
            raise ValueError("radial_fn must return shape () or (1,) for one route.")
        weight = weight * radial
    message = {
        "scalar": sent * weight,
        "vector": relative[:, None] * sent[None, :] * weight,
    }
    if convolution.normalize:
        message["weight_magnitude"] = jnp.abs(weight)
    return message


def _convolution_output(
    convolution: EquivariantGraphConvolution,
    receiver: tuple[jnp.ndarray, jnp.ndarray],
    aggregate: dict[str, jnp.ndarray],
    /,
) -> dict[str, jnp.ndarray]:
    """Receiver epilogue normalizing by its complete incoming weight magnitude."""
    del receiver
    scalar, vector = aggregate["scalar"], aggregate["vector"]
    if convolution.normalize:
        denom = aggregate["weight_magnitude"]
        scale = jnp.where(denom > 0, 1.0 / denom, 0.0)
        scalar = scalar * scale
        vector = vector * scale
    return {"scalar": scalar, "vector": vector}


def _mask_by_node_mask(value: jnp.ndarray, mask: jnp.ndarray | None, /) -> jnp.ndarray:
    if mask is None:
        return value
    while mask.ndim < value.ndim:
        mask = jnp.expand_dims(mask, axis=-1)
    return value * mask.astype(value.dtype)


class EquivariantGraphConvolution(StrictModule):
    """SE(n)-equivariant scalar/vector graph convolution.

    Scalar messages aggregate invariant source features. Vector messages are
    built from relative displacement vectors multiplied by invariant scalar
    coefficients, giving translation invariance and rotation equivariance when
    positions are transformed rigidly. Each route of
    `graph.edge_relation(flow=flow)` evaluates its message once through the
    prepared streamed relation selected by ``execution``, and every receiver
    normalizes by its complete incoming weight magnitude only after its last
    route; routes with `edge_mask=False` are inert. ``radial_fn`` must be a
    `RouteLocal` callback ``(edge, distance, unit, sent, received)`` of one
    route's unbatched rows returning one scalar weight.
    """

    radial_fn: RouteLocal | None
    execution: StreamedRelationPlan
    input_key: str | None = eqx.field(static=True)
    position_key: str = eqx.field(static=True)
    scalar_output_key: str | None = eqx.field(static=True)
    vector_output_key: str | None = eqx.field(static=True)
    edge_weight_key: str | None = eqx.field(static=True)
    flow: GraphFlow = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        radial_fn: RouteLocal | None = None,
        /,
        *,
        input_key: str | None = "features",
        position_key: str = "positions",
        scalar_output_key: str | None = "scalar",
        vector_output_key: str | None = "vector",
        edge_weight_key: str | None = None,
        flow: GraphFlow = "source_to_target",
        normalize: bool = False,
        eps: float = 1e-30,
        execution: StreamedRelationPlan | None = None,
    ) -> None:
        flow = parse(flow, GraphFlow, "flow")
        if scalar_output_key is None and vector_output_key is None:
            raise ValueError("At least one output key must be provided.")
        if execution is not None and not isinstance(execution, StreamedRelationPlan):
            raise TypeError("execution must be a StreamedRelationPlan or None.")
        self.radial_fn = require_route_local(radial_fn, "radial_fn")
        self.execution = StreamedRelationPlan() if execution is None else execution
        self.input_key = input_key
        self.position_key = str(position_key)
        self.scalar_output_key = scalar_output_key
        self.vector_output_key = vector_output_key
        self.edge_weight_key = edge_weight_key
        self.flow = flow
        self.normalize = bool(normalize)
        self.eps = float(eps)

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        positions = _positions(graph, self.position_key)
        relation = _oriented_relation(graph, self.flow, positions.shape[0])
        scalars = _node_scalar(graph, self.input_key)
        endpoints = (positions, scalars)
        edge_data = (
            None if self.radial_fn is None else graph.edges,
            _edge_weight(graph, self.edge_weight_key),
        )
        payload = declared_payload(
            _convolution_message,
            _convolution_output,
            self,
            endpoints,
            endpoints,
            edge_data,
            edge_output=False,
        )
        prepared = self.execution.prepare(
            relation, owner_id="graph:EquivariantGraphConvolution"
        )
        outputs = prepared.evaluate(
            payload,
            _convolution_message,
            _convolution_output,
            self,
            endpoints,
            endpoints,
            edge_data,
        ).receiver_outputs
        scalar_out = _mask_by_node_mask(outputs["scalar"], graph.node_mask)
        vector_out = _mask_by_node_mask(outputs["vector"], graph.node_mask)

        nodes = _as_feature_mapping(graph.nodes)
        if self.scalar_output_key is not None:
            nodes[self.scalar_output_key] = scalar_out
        if self.vector_output_key is not None:
            nodes[self.vector_output_key] = vector_out
        return graph.replace(nodes=nodes, validate=False)


__all__ = [
    "EquivariantGraphConvolution",
    "GraphFlow",
    "euclidean_edge_features",
    "gaussian_radial_basis",
]
