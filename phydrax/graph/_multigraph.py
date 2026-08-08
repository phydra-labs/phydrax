from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from .._model import register_artifact_value
from ._geometry import QueryGraph
from ._graph import ensure_graph
from ._ir import GraphIR
from ._neural_operators import (
    GraphAttentionOperator,
    GraphNeuralOperator,
)


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
    total_nodes = n_source + int(query.target_nodes.shape[0])
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
    source_measure: Any
    source_key: str | None = eqx.field(static=True)
    input_key: str = eqx.field(static=True)

    source_measure_key: str | None = eqx.field(static=True)

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
        source_measure_key: str | None = None,
        source_measure: Any | None = None,
        normalize: bool = True,
    ):
        self.query = query
        self.source_key = source_key
        self.source_indices = (
            None
            if source_indices is None
            else jnp.asarray(source_indices, dtype=jnp.int32)
        )
        self.input_key = str(input_key)
        self.source_measure_key = source_measure_key
        self.source_measure = source_measure
        self.operator = (
            GraphNeuralOperator(
                input_key=self.input_key,
                output_key=str(output_key),
                edge_weight_key=edge_weight_key,
                normalize=normalize,
                source_measure_key=self.source_measure_key,
                source_measure=self.source_measure,
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


class GraphFieldProcessor(eqx.Module):
    """Apply an array model to a structured latent field stored on graph nodes."""

    model: Callable
    node_indices: jnp.ndarray
    coordinates: tuple[jnp.ndarray, ...]
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    input_key: str = eqx.field(static=True)
    output_key: str = eqx.field(static=True)

    def __init__(
        self,
        model: Callable,
        node_indices: Any,
        spatial_shape: Sequence[int],
        /,
        *,
        coordinates: Sequence[Any] = (),
        input_key: str = "latent",
        output_key: str = "latent",
    ):
        self.model = model
        self.node_indices = jnp.asarray(node_indices, dtype=jnp.int32)
        self.spatial_shape = tuple(int(size) for size in spatial_shape)
        self.coordinates = tuple(jnp.asarray(axis, dtype=float) for axis in coordinates)
        self.input_key = str(input_key)
        self.output_key = str(output_key)
        if not self.spatial_shape or any(size <= 0 for size in self.spatial_shape):
            raise ValueError("spatial_shape must contain positive sizes.")
        expected = 1
        for size in self.spatial_shape:
            expected *= size
        if int(self.node_indices.shape[0]) != expected:
            raise ValueError(
                "node_indices length must equal the latent spatial-shape product."
            )
        if self.coordinates and len(self.coordinates) != len(self.spatial_shape):
            raise ValueError("coordinates must contain one axis per spatial dimension.")

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        values = _node_field(graph, self.input_key)
        selected = values[self.node_indices]
        trailing = tuple(int(size) for size in selected.shape[1:])
        field = selected.reshape(self.spatial_shape + trailing)
        output = (
            self.model((field, *self.coordinates))
            if self.coordinates
            else self.model(field)
        )
        output_ = jnp.asarray(output)
        output_trailing = tuple(
            int(size) for size in output_.shape[len(self.spatial_shape) :]
        )
        flattened = output_.reshape((int(self.node_indices.shape[0]),) + output_trailing)
        total_nodes = int(jnp.asarray(graph.n_node).sum())
        installed = jnp.zeros(
            (total_nodes,) + output_trailing,
            dtype=flattened.dtype,
        )
        installed = installed.at[self.node_indices].set(flattened)
        nodes = _as_feature_mapping(graph.nodes)
        nodes[self.output_key] = installed
        return graph.replace(nodes=nodes, validate=False)


class RegionalGraphProcessor(eqx.Module):
    """Process encoder latents on a separate regional/slice graph."""

    regional_graph: GraphIR
    block: Callable[[GraphIR], GraphIR]
    latent_indices: jnp.ndarray
    steps: int = eqx.field(static=True)
    input_key: str = eqx.field(static=True)
    output_key: str = eqx.field(static=True)

    def __init__(
        self,
        regional_graph: GraphIR,
        block: Callable[[GraphIR], GraphIR],
        latent_indices: Any,
        /,
        *,
        steps: int,
        input_key: str = "latent",
        output_key: str = "latent",
    ):
        self.regional_graph = ensure_graph(regional_graph, validate=False)
        self.block = block
        self.latent_indices = jnp.asarray(latent_indices, dtype=jnp.int32)
        self.steps = int(steps)
        self.input_key = str(input_key)
        self.output_key = str(output_key)
        regional_count = int(jnp.asarray(self.regional_graph.n_node).sum())
        if regional_count != int(self.latent_indices.shape[0]):
            raise ValueError(
                "Regional graph node count must match the encoder latent-node count."
            )
        if self.steps <= 0:
            raise ValueError("Regional graph processor steps must be positive.")

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        latent_values = _node_field(graph, self.input_key)[self.latent_indices]
        regional_nodes = _as_feature_mapping(self.regional_graph.nodes)
        regional_nodes[self.input_key] = latent_values
        regional = self.regional_graph.replace(nodes=regional_nodes, validate=False)
        for _ in range(self.steps):
            regional = self.block(regional)
            if not isinstance(regional, GraphIR):
                raise TypeError("Regional processor block must return a GraphIR.")
        processed = _node_field(regional, self.output_key)
        total_nodes = int(jnp.asarray(graph.n_node).sum())
        installed = jnp.zeros(
            (total_nodes,) + processed.shape[1:],
            dtype=processed.dtype,
        )
        installed = installed.at[self.latent_indices].set(processed)
        nodes = _as_feature_mapping(graph.nodes)
        nodes[self.output_key] = installed
        return graph.replace(nodes=nodes, validate=False)


register_artifact_value(
    "phydrax.graph.model:RegionalGraphProcessor@1",
    RegionalGraphProcessor,
)


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
                raise TypeError(
                    "GraphEncodeProcessDecode processor must return a GraphIR."
                )
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
    encoder_source_measure_key: str | None = None,
    encoder_source_measure: Any | None = None,
    decoder_source_measure_key: str | None = None,
    decoder_source_measure: Any | None = None,
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
        source_measure_key=encoder_source_measure_key,
        source_measure=encoder_source_measure,
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
        source_measure_key=decoder_source_measure_key,
        source_measure=decoder_source_measure,
        normalize=normalize,
    )
    return GraphEncodeProcessDecode(encoder, decoder, processor=processor)


def gino_operator(
    encoder_query: QueryGraph,
    decoder_query: QueryGraph,
    processor_model: Callable,
    latent_shape: Sequence[int],
    /,
    *,
    latent_axes: Sequence[Any] = (),
    source_measure_key: str | None = "quadrature_weight",
    latent_measure_key: str | None = "quadrature_weight",
    source_key: str | None = "features",
    latent_key: str = "latent",
    output_key: str = "features",
) -> GraphEncodeProcessDecode:
    """Build a GINO configuration with graph transfers and a latent-grid model."""
    processor = GraphFieldProcessor(
        processor_model,
        encoder_query.target_nodes,
        latent_shape,
        coordinates=latent_axes,
        input_key=latent_key,
        output_key=latent_key,
    )
    return query_encode_process_decode(
        encoder_query,
        decoder_query,
        processor=processor,
        source_key=source_key,
        latent_key=latent_key,
        output_key=output_key,
        encoder_source_measure_key=source_measure_key,
        decoder_source_measure_key=latent_measure_key,
    )


def rigno_operator(
    encoder_query: QueryGraph,
    decoder_query: QueryGraph,
    regional_graph: GraphIR,
    regional_block: Callable[[GraphIR], GraphIR],
    /,
    *,
    steps: int,
    source_measure_key: str | None = "quadrature_weight",
    latent_measure_key: str | None = "quadrature_weight",
    source_key: str | None = "features",
    latent_key: str = "latent",
    output_key: str = "features",
) -> GraphEncodeProcessDecode:
    """Build a RIGNO configuration with a dedicated regional latent graph."""
    processor = RegionalGraphProcessor(
        regional_graph,
        regional_block,
        encoder_query.target_nodes,
        steps=steps,
        input_key=latent_key,
        output_key=latent_key,
    )
    return query_encode_process_decode(
        encoder_query,
        decoder_query,
        processor=processor,
        source_key=source_key,
        latent_key=latent_key,
        output_key=output_key,
        encoder_source_measure_key=source_measure_key,
        decoder_source_measure_key=latent_measure_key,
    )


def gaot_operator(
    encoder_query: QueryGraph,
    decoder_query: QueryGraph,
    regional_graph: GraphIR,
    attention_block: GraphAttentionOperator,
    /,
    *,
    steps: int,
    encoder_operator: Callable[[GraphIR], GraphIR] | None = None,
    decoder_operator: Callable[[GraphIR], GraphIR] | None = None,
    source_key: str | None = "features",
    latent_key: str = "latent",
    output_key: str = "features",
) -> GraphEncodeProcessDecode:
    """Build a GAOT configuration with graph attention in regional latent space."""
    processor = RegionalGraphProcessor(
        regional_graph,
        attention_block,
        encoder_query.target_nodes,
        steps=steps,
        input_key=latent_key,
        output_key=latent_key,
    )
    return query_encode_process_decode(
        encoder_query,
        decoder_query,
        encoder_operator=encoder_operator,
        processor=processor,
        decoder_operator=decoder_operator,
        source_key=source_key,
        latent_key=latent_key,
        output_key=output_key,
    )


def transolver_operator(
    encoder_query: QueryGraph,
    decoder_query: QueryGraph,
    slice_graph: GraphIR,
    slice_attention: GraphAttentionOperator,
    /,
    *,
    steps: int,
    source_measure_key: str | None = "quadrature_weight",
    slice_measure_key: str | None = "quadrature_weight",
    source_key: str | None = "features",
    slice_key: str = "latent",
    output_key: str = "features",
) -> GraphEncodeProcessDecode:
    """Build a Transolver configuration whose latent graph nodes are physical slices."""
    processor = RegionalGraphProcessor(
        slice_graph,
        slice_attention,
        encoder_query.target_nodes,
        steps=steps,
        input_key=slice_key,
        output_key=slice_key,
    )
    return query_encode_process_decode(
        encoder_query,
        decoder_query,
        processor=processor,
        source_key=source_key,
        latent_key=slice_key,
        output_key=output_key,
        encoder_source_measure_key=source_measure_key,
        decoder_source_measure_key=slice_measure_key,
    )


__all__ = [
    "GraphEncodeProcessDecode",
    "GraphFieldProcessor",
    "RegionalGraphProcessor",
    "QueryGraphOperator",
    "query_encode_process_decode",
    "gaot_operator",
    "gino_operator",
    "query_graph_with_source_features",
    "rigno_operator",
    "transolver_operator",
    "query_target_features",
]
