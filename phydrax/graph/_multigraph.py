from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ._geometry import QueryGraph
from ._graph import ensure_graph
from ._ir import GraphIR
from ._neural_operators import GraphNeuralOperator


def _as_feature_mapping(value: Any, /) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"features": value}


def _node_field(graph: GraphIR, key: str | None, /) -> jnp.ndarray:
    if graph.nodes is None:
        raise ValueError("Source graph must contain node features.")
    if key is None:
        if isinstance(graph.nodes, Mapping):
            raise TypeError("mapping-valued source graph nodes require source_key.")
        return jnp.asarray(graph.nodes)
    if not isinstance(graph.nodes, Mapping):
        raise TypeError("source_key requires mapping-valued source graph nodes.")
    if key not in graph.nodes:
        raise KeyError(f"Source graph nodes do not contain key {key!r}.")
    return jnp.asarray(graph.nodes[key])


def query_graph_with_source_features(
    query: QueryGraph,
    source_features: Any,
    /,
    *,
    input_key: str = "features",
) -> GraphIR:
    """Install source features into the source side of a `QueryGraph`."""
    features = jnp.asarray(source_features)
    if features.ndim == 0:
        raise ValueError("source_features must have a leading source-node axis.")
    n_source = int(query.source_nodes.shape[0])
    if int(features.shape[0]) != n_source:
        raise ValueError(
            f"source_features leading axis must match query source count {n_source}; "
            f"got {features.shape[0]}."
        )
    total_nodes = int(jnp.asarray(query.graph.n_node).sum())
    values = jnp.zeros((total_nodes,) + features.shape[1:], dtype=features.dtype)
    values = values.at[query.source_nodes].set(features)
    nodes = _as_feature_mapping(query.graph.nodes)
    nodes[str(input_key)] = values
    return query.graph.replace(nodes=nodes, validate=False)


def query_target_features(
    graph: GraphIR,
    query: QueryGraph,
    key: str,
    /,
) -> jnp.ndarray:
    """Read a node payload from the target side of a query graph."""
    if not isinstance(graph.nodes, Mapping):
        raise TypeError("query_target_features requires mapping-valued graph nodes.")
    if key not in graph.nodes:
        raise KeyError(f"Query graph nodes do not contain key {key!r}.")
    return jnp.asarray(graph.nodes[key])[query.target_nodes]


class QueryGraphOperator(eqx.Module):
    """Transfer source-graph node features through a fixed query graph.

    The operator copies a source graph node field onto the source side of a
    `QueryGraph`, applies a `GraphIR -> GraphIR` operator over that query graph,
    and returns the resulting query graph.
    """

    query: QueryGraph
    operator: Callable[[GraphIR], GraphIR]
    source_indices: jnp.ndarray | None
    source_key: str | None = eqx.field(static=True)
    input_key: str = eqx.field(static=True)

    def __init__(
        self,
        query: QueryGraph,
        /,
        *,
        operator: Callable[[GraphIR], GraphIR] | None = None,
        source_key: str | None = "features",
        source_indices: Any | None = None,
        input_key: str = "features",
        output_key: str = "features",
        edge_weight_key: str | None = "kernel_weight",
        normalize: bool = True,
    ):
        self.query = query
        self.source_key = source_key
        self.source_indices = (
            None if source_indices is None else jnp.asarray(source_indices, dtype=jnp.int32)
        )
        self.input_key = str(input_key)
        self.operator = (
            GraphNeuralOperator(
                input_key=self.input_key,
                output_key=str(output_key),
                edge_weight_key=edge_weight_key,
                normalize=normalize,
                target_node_type=query.target_type,
            )
            if operator is None
            else operator
        )

    def __call__(self, source_graph: GraphIR) -> GraphIR:
        source_graph = ensure_graph(source_graph, validate=False)
        source_features = _node_field(source_graph, self.source_key)
        if self.source_indices is not None:
            source_features = source_features[self.source_indices]
        query_graph = query_graph_with_source_features(
            self.query,
            source_features,
            input_key=self.input_key,
        )
        out = self.operator(query_graph)
        if not isinstance(out, GraphIR):
            raise TypeError("QueryGraphOperator operator must return a GraphIR.")
        return out


class GraphEncodeProcessDecode(eqx.Module):
    """Compose source-to-latent transfer, latent processing, and latent-to-target transfer."""

    encoder: QueryGraphOperator
    processor: Callable[[GraphIR], GraphIR] | None
    decoder: QueryGraphOperator

    def __init__(
        self,
        encoder: QueryGraphOperator,
        decoder: QueryGraphOperator,
        /,
        *,
        processor: Callable[[GraphIR], GraphIR] | None = None,
    ):
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder

    def __call__(self, source_graph: GraphIR) -> GraphIR:
        latent = self.encoder(source_graph)
        if self.processor is not None:
            latent = self.processor(latent)
            if not isinstance(latent, GraphIR):
                raise TypeError("GraphEncodeProcessDecode processor must return a GraphIR.")
        return self.decoder(latent)


def query_encode_process_decode(
    encoder_query: QueryGraph,
    decoder_query: QueryGraph,
    /,
    *,
    encoder_operator: Callable[[GraphIR], GraphIR] | None = None,
    processor: Callable[[GraphIR], GraphIR] | None = None,
    decoder_operator: Callable[[GraphIR], GraphIR] | None = None,
    source_key: str | None = "features",
    encoder_input_key: str = "features",
    latent_key: str = "latent",
    decoder_input_key: str = "latent",
    output_key: str = "features",
    edge_weight_key: str | None = "kernel_weight",
    normalize: bool = True,
) -> GraphEncodeProcessDecode:
    """Build a query-graph encode-process-decode operator."""
    encoder = QueryGraphOperator(
        encoder_query,
        operator=encoder_operator,
        source_key=source_key,
        input_key=encoder_input_key,
        output_key=latent_key,
        edge_weight_key=edge_weight_key,
        normalize=normalize,
    )
    decoder = QueryGraphOperator(
        decoder_query,
        operator=decoder_operator,
        source_key=latent_key,
        source_indices=encoder_query.target_nodes,
        input_key=decoder_input_key,
        output_key=output_key,
        edge_weight_key=edge_weight_key,
        normalize=normalize,
    )
    return GraphEncodeProcessDecode(encoder, decoder, processor=processor)


__all__ = [
    "GraphEncodeProcessDecode",
    "QueryGraphOperator",
    "query_encode_process_decode",
    "query_graph_with_source_features",
    "query_target_features",
]
