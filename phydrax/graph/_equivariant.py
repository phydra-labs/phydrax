from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp

from ._graph import ensure_graph
from ._ir import GraphIR
from ._kernels import segment_sum


GraphFlow = Literal["source_to_target", "target_to_source"]


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
        raise ValueError(f"Graph node positions must have shape (n, dim); got {pos.shape!r}.")
    return pos


def _node_scalar(graph: GraphIR, input_key: str | None, /) -> jnp.ndarray:
    if graph.nodes is None:
        raise ValueError("EquivariantGraphConvolution requires node features.")
    if input_key is None:
        if isinstance(graph.nodes, Mapping):
            raise TypeError("mapping-valued graph nodes require input_key.")
        arr = jnp.asarray(graph.nodes, dtype=float)
    else:
        arr = _mapping_value(graph.nodes, input_key, "nodes").astype(float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Node scalar field must be rank-1 or rank-2; got {arr.shape!r}.")
    return arr


def _edge_weight(graph: GraphIR, edge_weight_key: str | None, /) -> jnp.ndarray | None:
    if edge_weight_key is None:
        return None
    if not isinstance(graph.edges, Mapping):
        raise TypeError("edge_weight_key requires mapping-valued graph edges.")
    if edge_weight_key not in graph.edges:
        raise KeyError(f"Graph edges do not contain edge_weight_key {edge_weight_key!r}.")
    weight = jnp.asarray(graph.edges[edge_weight_key], dtype=float)
    if weight.ndim == 2 and int(weight.shape[1]) == 1:
        weight = weight[:, 0]
    if weight.ndim != 1:
        raise ValueError("edge weights must have shape (n_edge,) or (n_edge, 1).")
    return weight


def _oriented_edges(graph: GraphIR, flow: GraphFlow, /) -> tuple[jnp.ndarray, jnp.ndarray]:
    if graph.senders is None or graph.receivers is None:
        raise ValueError("Equivariant graph operators require explicit senders/receivers.")
    if flow == "source_to_target":
        return graph.senders, graph.receivers
    if flow == "target_to_source":
        return graph.receivers, graph.senders
    raise ValueError("flow must be 'source_to_target' or 'target_to_source'.")


def _relative_geometry(
    graph: GraphIR,
    /,
    *,
    position_key: str,
    flow: GraphFlow,
    eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    source, target = _oriented_edges(graph, flow)
    pos = _positions(graph, position_key)
    relative = pos[target] - pos[source]
    squared_distance = jnp.sum(jnp.square(relative), axis=-1, keepdims=True)
    distance = jnp.sqrt(jnp.maximum(squared_distance, float(eps)))
    unit = relative / distance
    return source, target, relative, distance, unit


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
    _source, _target, relative, distance, unit = _relative_geometry(
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
    d = jnp.asarray(distance, dtype=float)
    c = jnp.asarray(centers, dtype=float).reshape((1, -1))
    if d.ndim == 1:
        d = d[:, None]
    if d.ndim != 2 or int(d.shape[1]) != 1:
        raise ValueError("distance must have shape (n_edge,) or (n_edge, 1).")
    return jnp.exp(-float(gamma) * jnp.square(d - c))


def _broadcast_edge_weight(weight: jnp.ndarray, values: jnp.ndarray, /) -> jnp.ndarray:
    while weight.ndim < values.ndim:
        weight = jnp.expand_dims(weight, axis=-1)
    return weight


def _mask_by_node_mask(value: jnp.ndarray, mask: jnp.ndarray | None, /) -> jnp.ndarray:
    if mask is None:
        return value
    while mask.ndim < value.ndim:
        mask = jnp.expand_dims(mask, axis=-1)
    return value * mask.astype(value.dtype)


class EquivariantGraphConvolution(eqx.Module):
    """SE(n)-equivariant scalar/vector graph convolution.

    Scalar messages aggregate invariant source features. Vector messages are
    built from relative displacement vectors multiplied by invariant scalar
    coefficients, giving translation invariance and rotation equivariance when
    positions are transformed rigidly.
    """

    radial_fn: Callable | None
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
        radial_fn: Callable | None = None,
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
    ):
        if flow not in ("source_to_target", "target_to_source"):
            raise ValueError("flow must be 'source_to_target' or 'target_to_source'.")
        if scalar_output_key is None and vector_output_key is None:
            raise ValueError("At least one output key must be provided.")
        self.radial_fn = radial_fn
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
        source, target, relative, distance, unit = _relative_geometry(
            graph,
            position_key=self.position_key,
            flow=self.flow,
            eps=self.eps,
        )
        scalars = _node_scalar(graph, self.input_key)
        sent = scalars[source]
        recv = scalars[target]

        weight = jnp.ones((int(source.shape[0]),), dtype=scalars.dtype)
        edge_weight = _edge_weight(graph, self.edge_weight_key)
        if edge_weight is not None:
            weight = weight * edge_weight.astype(weight.dtype)
        if self.radial_fn is not None:
            radial = jnp.asarray(
                self.radial_fn(graph.edges, distance, unit, sent, recv),
                dtype=scalars.dtype,
            )
            if radial.ndim == 2 and int(radial.shape[1]) == 1:
                radial = radial[:, 0]
            if radial.ndim != 1:
                raise ValueError("radial_fn must return shape (n_edge,) or (n_edge, 1).")
            weight = weight * radial
        if graph.edge_mask is not None:
            weight = weight * graph.edge_mask.astype(weight.dtype)

        scalar_messages = sent * _broadcast_edge_weight(weight, sent)
        scalar_out = segment_sum(scalar_messages, target, int(scalars.shape[0]))

        vector_messages = (
            relative[:, :, None]
            * sent[:, None, :]
            * _broadcast_edge_weight(weight, sent)[:, None, :]
        )
        vector_out = segment_sum(vector_messages, target, int(scalars.shape[0]))

        if self.normalize:
            denom = segment_sum(jnp.abs(weight), target, int(scalars.shape[0]))
            scale = jnp.where(denom > 0, 1.0 / denom, 0.0)
            scalar_out = scalar_out * scale[:, None]
            vector_out = vector_out * scale[:, None, None]

        scalar_out = _mask_by_node_mask(scalar_out, graph.node_mask)
        vector_out = _mask_by_node_mask(vector_out, graph.node_mask)

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
