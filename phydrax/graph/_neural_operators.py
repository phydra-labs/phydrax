from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from ..sparse import gather_routes, mask_routes, route_reduce
from ._graph import ensure_graph
from ._ir import GraphIR
from ._kernels import segment_softmax, segment_sum
from ._typed import node_type_ids


ArrayTree = Any
FiniteVolumeSign = Literal["in_minus_out", "out_minus_in"]


def _tree_leading_size(tree: ArrayTree) -> int:
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        raise ValueError("Feature tree must contain at least one array leaf.")
    return int(jnp.asarray(leaves[0]).shape[0])


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


def _multiply_tree(tree: ArrayTree, weight: Any, /) -> ArrayTree:
    if jtu.tree_structure(tree) == jtu.tree_structure(weight):
        return jtu.tree_map(
            _multiply_leaf,
            tree,
            weight,
        )
    return jtu.tree_map(lambda x: _multiply_leaf(x, weight), tree)


def _mask_tree(tree: ArrayTree, mask: jnp.ndarray | None, /) -> ArrayTree:
    if mask is None:
        return tree

    def mask_leaf(value: Any, /) -> jnp.ndarray:
        array = jnp.asarray(value)
        expanded = mask.reshape(mask.shape + (1,) * (array.ndim - mask.ndim))
        return jnp.where(expanded, array, jnp.zeros((), dtype=array.dtype))

    return jtu.tree_map(mask_leaf, tree)


def _pad_tree_leading(tree: ArrayTree, target: int, /) -> ArrayTree:
    n = _tree_leading_size(tree)
    pad = int(target) - n
    if pad <= 0:
        return tree
    return jtu.tree_map(
        lambda x: jnp.concatenate(
            [x, jnp.zeros((pad,) + x.shape[1:], dtype=x.dtype)],
            axis=0,
        ),
        tree,
    )


def _repeat_globals_for_entities(
    globals_: ArrayTree | None,
    counts: jnp.ndarray,
    total_length: int,
    /,
) -> ArrayTree | None:
    if globals_ is None:
        return None
    real_length = int(jnp.asarray(counts).sum())
    repeated = jtu.tree_map(
        lambda x: jnp.repeat(x, counts, axis=0, total_repeat_length=real_length),
        globals_,
    )
    return _pad_tree_leading(repeated, total_length)


def _as_feature_mapping(value: Any, /) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"features": value}


def _node_field(graph: GraphIR, input_key: str | None, /, *, name: str) -> ArrayTree:
    if graph.nodes is None:
        raise ValueError(f"{name} requires node features.")
    if input_key is None:
        if isinstance(graph.nodes, Mapping):
            raise TypeError(f"mapping-valued graph nodes require input_key for {name}.")
        return graph.nodes
    if not isinstance(graph.nodes, Mapping):
        raise TypeError("input_key requires mapping-valued graph nodes.")
    if input_key not in graph.nodes:
        raise KeyError(f"Graph nodes do not contain input_key {input_key!r}.")
    return graph.nodes[input_key]


def _node_array(graph: GraphIR, input_key: str | None, /, *, name: str) -> jnp.ndarray:
    arr = jnp.asarray(_node_field(graph, input_key, name=name), dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} node features must be rank-1 or rank-2.")
    return arr


def _with_node_output(graph: GraphIR, value: ArrayTree, output_key: str | None, /) -> Any:
    if output_key is None:
        return value
    nodes = _as_feature_mapping(graph.nodes)
    nodes[output_key] = value
    return nodes


def _edge_weight(graph: GraphIR, edge_weight_key: str | None, /) -> jnp.ndarray | None:
    if edge_weight_key is None:
        return None
    if not isinstance(graph.edges, Mapping):
        raise TypeError("edge_weight_key requires mapping-valued graph edges.")
    if edge_weight_key not in graph.edges:
        raise KeyError(f"Graph edges do not contain edge_weight_key {edge_weight_key!r}.")
    weight = jnp.asarray(graph.edges[edge_weight_key], dtype=float)
    if weight.ndim == 2 and int(weight.shape[1]) == 1:
        return weight[:, 0]
    if weight.ndim != 1:
        raise ValueError("edge weights must have shape (n_edge,) or (n_edge, 1).")
    return weight


def _edge_array(
    graph: GraphIR,
    edge_key: str | None,
    /,
    *,
    name: str,
) -> jnp.ndarray:
    if graph.edges is None:
        raise ValueError(f"{name} requires edge features.")
    if edge_key is None:
        if isinstance(graph.edges, Mapping):
            raise TypeError(f"mapping-valued graph edges require edge_key for {name}.")
        arr = jnp.asarray(graph.edges, dtype=float)
    else:
        if not isinstance(graph.edges, Mapping):
            raise TypeError("edge_key requires mapping-valued graph edges.")
        if edge_key not in graph.edges:
            raise KeyError(f"Graph edges do not contain edge_key {edge_key!r}.")
        arr = jnp.asarray(graph.edges[edge_key], dtype=float)
    if arr.ndim == 0:
        raise ValueError(f"{name} edge features must have a leading edge axis.")
    return arr


def _node_volume(
    graph: GraphIR,
    volume_key: str | None,
    volume: Any | None,
    /,
    *,
    n_node: int,
) -> jnp.ndarray:
    if volume is not None:
        out = jnp.asarray(volume, dtype=float)
    elif volume_key is None:
        out = jnp.ones((n_node,), dtype=float)
    else:
        if not isinstance(graph.nodes, Mapping):
            raise TypeError("volume_key requires mapping-valued graph nodes.")
        if volume_key not in graph.nodes:
            raise KeyError(f"Graph nodes do not contain volume_key {volume_key!r}.")
        out = jnp.asarray(graph.nodes[volume_key], dtype=float)
    if out.ndim == 2 and int(out.shape[1]) == 1:
        out = out[:, 0]
    if out.ndim != 1:
        raise ValueError(
            "Finite-volume node volumes must have shape (n_node,) or (n_node, 1)."
        )
    if int(out.shape[0]) != n_node:
        raise ValueError("Finite-volume node volume length must match graph nodes.")
    return out


def _node_measure(
    graph: GraphIR,
    measure_key: str | None,
    measure: Any | None,
    /,
    *,
    n_node: int,
) -> jnp.ndarray:
    out = _node_volume(
        graph,
        measure_key,
        measure,
        n_node=n_node,
    )
    return jnp.maximum(out, 0.0)


def _broadcast_node_volume(volume: jnp.ndarray, values: jnp.ndarray, /) -> jnp.ndarray:
    while volume.ndim < values.ndim:
        volume = jnp.expand_dims(volume, axis=-1)
    return volume


def _finite_volume_divergence(
    graph: GraphIR,
    flux: jnp.ndarray,
    /,
    *,
    volume_key: str | None,
    volume: Any | None,
    normalize_by_volume: bool,
    sign: FiniteVolumeSign,
) -> jnp.ndarray:
    n_node = _num_graph_nodes(graph)
    relation = graph.edge_relation(node_count=n_node)
    incoming = route_reduce(relation, flux)
    outgoing = route_reduce(relation.transpose(), flux)
    if sign == "in_minus_out":
        out = incoming - outgoing
    elif sign == "out_minus_in":
        out = outgoing - incoming
    else:
        raise ValueError("sign must be 'in_minus_out' or 'out_minus_in'.")
    if normalize_by_volume:
        vol = _node_volume(graph, volume_key, volume, n_node=n_node)
        inv_vol = jnp.where(vol != 0, 1.0 / vol, 0.0)
        out = out * _broadcast_node_volume(inv_vol, out)
    return _mask_tree(out, graph.node_mask)


def _edge_bias(graph: GraphIR, edge_bias_key: str | None, /) -> jnp.ndarray | None:
    if edge_bias_key is None:
        return None
    if not isinstance(graph.edges, Mapping):
        raise TypeError("edge_bias_key requires mapping-valued graph edges.")
    if edge_bias_key not in graph.edges:
        raise KeyError(f"Graph edges do not contain edge_bias_key {edge_bias_key!r}.")
    bias = jnp.asarray(graph.edges[edge_bias_key], dtype=float)
    if bias.ndim == 2 and int(bias.shape[1]) == 1:
        return bias[:, 0]
    if bias.ndim not in (1, 2):
        raise ValueError(
            "edge attention bias must have shape (n_edge,), (n_edge, 1), or (n_edge, n_head)."
        )
    return bias


def _mask_node_type(
    graph: GraphIR,
    tree: ArrayTree,
    target_node_type: int | None,
    /,
    *,
    node_type_key: str,
) -> ArrayTree:
    if target_node_type is None:
        return tree
    keep = node_type_ids(graph, type_key=node_type_key) == int(target_node_type)
    if graph.node_mask is not None:
        keep = keep & graph.node_mask
    return _multiply_tree(tree, keep.astype(float))


def _num_nodes(graph: GraphIR, nodes: ArrayTree, /) -> int:
    if graph.node_mask is not None:
        return int(graph.node_mask.shape[0])
    return _tree_leading_size(nodes)


def _num_edges(graph: GraphIR, /) -> int:
    if graph.edge_mask is not None:
        return int(graph.edge_mask.shape[0])
    if graph.senders is None:
        return int(jnp.asarray(graph.n_edge).sum())
    return int(graph.senders.shape[0])


def _num_graph_nodes(graph: GraphIR, /) -> int:
    if graph.node_mask is not None:
        return int(graph.node_mask.shape[0])
    if graph.nodes is not None:
        return _tree_leading_size(graph.nodes)
    return int(jnp.asarray(graph.n_node).sum())


class GraphKernelIntegral(eqx.Module):
    """Edge-kernel integral operator over a sparse graph.

    The block sends source node features along directed edges, optionally
    multiplies them by a learned/analytic kernel, aggregates to receivers, and
    writes the aggregate as the graph's node payload.
    """

    kernel_fn: Callable | None
    source_fn: Callable | None
    update_node_fn: Callable | None
    source_measure: Any
    aggregate_fn: Callable = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    source_measure_key: str | None = eqx.field(static=True)

    def __init__(
        self,
        kernel_fn: Callable | None = None,
        /,
        *,
        source_fn: Callable | None = None,
        update_node_fn: Callable | None = None,
        source_measure_key: str | None = None,
        source_measure: Any | None = None,
        aggregate_fn: Callable = segment_sum,
        normalize: bool = False,
    ):
        self.kernel_fn = kernel_fn
        self.source_fn = source_fn
        self.update_node_fn = update_node_fn
        self.source_measure_key = source_measure_key
        self.source_measure = source_measure
        self.aggregate_fn = aggregate_fn
        self.normalize = bool(normalize)

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.nodes is None:
            raise ValueError("GraphKernelIntegral requires node features.")
        if graph.senders is None or graph.receivers is None:
            raise ValueError("GraphKernelIntegral requires explicit senders/receivers.")

        nodes = graph.nodes
        num_nodes = _num_nodes(graph, nodes)
        relation = graph.edge_relation(node_count=num_nodes)
        source = nodes if self.source_fn is None else self.source_fn(nodes)
        sent_source = gather_routes(relation, source)
        sent_nodes = gather_routes(relation, nodes)
        recv_nodes = gather_routes(relation.transpose(), nodes)
        num_edges = _num_edges(graph)
        glob_edge = _repeat_globals_for_entities(graph.globals, graph.n_edge, num_edges)

        edge_measure = None
        if self.source_measure_key is not None or self.source_measure is not None:
            node_measure = _node_measure(
                graph,
                self.source_measure_key,
                self.source_measure,
                n_node=num_nodes,
            )
            edge_measure = node_measure[relation.source_indices]
        messages = sent_source
        if self.kernel_fn is not None:
            weight = self.kernel_fn(graph.edges, sent_nodes, recv_nodes, glob_edge)
            messages = _multiply_tree(messages, weight)
        if edge_measure is not None:
            messages = _multiply_tree(messages, edge_measure)

        messages = mask_routes(relation, messages)
        aggregated = jtu.tree_map(
            lambda x: self.aggregate_fn(x, relation.target_indices, num_nodes),
            messages,
        )

        if self.normalize:
            if edge_measure is None:
                normalizer = jnp.ones((num_edges,), dtype=float)
            else:
                normalizer = edge_measure
            degree = route_reduce(relation, normalizer)
            scale = jnp.where(degree > 0, 1.0 / degree, 0.0)
            aggregated = _multiply_tree(aggregated, scale)

        glob_node = _repeat_globals_for_entities(graph.globals, graph.n_node, num_nodes)
        if self.update_node_fn is not None:
            aggregated = self.update_node_fn(nodes, aggregated, glob_node)
        aggregated = _mask_tree(aggregated, graph.node_mask)
        return graph.replace(nodes=aggregated, validate=False)


class GraphDiffusion(eqx.Module):
    """Physics-encoded incidence diffusion operator over node features."""

    conductivity_fn: Callable | None
    update_node_fn: Callable | None
    sign: str = eqx.field(static=True)

    def __init__(
        self,
        conductivity_fn: Callable | None = None,
        /,
        *,
        update_node_fn: Callable | None = None,
        sign: str = "in_minus_out",
    ):
        if sign not in ("in_minus_out", "out_minus_in"):
            raise ValueError(
                "GraphDiffusion sign must be 'in_minus_out' or 'out_minus_in'."
            )
        self.conductivity_fn = conductivity_fn
        self.update_node_fn = update_node_fn
        self.sign = sign

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.nodes is None:
            raise ValueError("GraphDiffusion requires node features.")
        if graph.senders is None or graph.receivers is None:
            raise ValueError("GraphDiffusion requires explicit senders/receivers.")

        nodes = graph.nodes
        num_nodes = _num_nodes(graph, nodes)
        relation = graph.edge_relation(node_count=num_nodes)
        sent = gather_routes(relation, nodes)
        recv = gather_routes(relation.transpose(), nodes)
        gradient = jtu.tree_map(lambda r, s: r - s, recv, sent)
        num_edges = _num_edges(graph)
        glob_edge = _repeat_globals_for_entities(graph.globals, graph.n_edge, num_edges)
        if self.conductivity_fn is not None:
            conductivity = self.conductivity_fn(graph.edges, sent, recv, glob_edge)
            gradient = _multiply_tree(gradient, conductivity)
        gradient = mask_routes(relation, gradient)

        incoming = route_reduce(relation, gradient)
        outgoing = route_reduce(relation.transpose(), gradient)
        if self.sign == "in_minus_out":
            diffused = jtu.tree_map(lambda inc, out: inc - out, incoming, outgoing)
        else:
            diffused = jtu.tree_map(lambda inc, out: out - inc, incoming, outgoing)

        glob_node = _repeat_globals_for_entities(graph.globals, graph.n_node, num_nodes)
        if self.update_node_fn is not None:
            diffused = self.update_node_fn(nodes, diffused, glob_node)
        diffused = _mask_tree(diffused, graph.node_mask)
        return graph.replace(nodes=diffused, validate=False)


class GraphNeuralOperator(eqx.Module):
    """Weighted graph neural operator for source-to-target query graphs.

    The block reads a named node field, sends source values across graph edges,
    multiplies them by an optional analytic/learned kernel and scalar edge
    weights, aggregates to receivers, and writes the result back as graph nodes
    or into `output_key`.
    """

    kernel_fn: Callable | None
    source_fn: Callable | None
    update_node_fn: Callable | None
    source_measure: Any
    aggregate_fn: Callable = eqx.field(static=True)
    input_key: str | None = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)
    edge_weight_key: str | None = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    source_measure_key: str | None = eqx.field(static=True)
    node_type_key: str = eqx.field(static=True)
    target_node_type: int | None = eqx.field(static=True)

    def __init__(
        self,
        kernel_fn: Callable | None = None,
        /,
        *,
        source_fn: Callable | None = None,
        update_node_fn: Callable | None = None,
        aggregate_fn: Callable = segment_sum,
        input_key: str | None = None,
        output_key: str | None = None,
        edge_weight_key: str | None = "kernel_weight",
        source_measure_key: str | None = None,
        source_measure: Any | None = None,
        normalize: bool = True,
        node_type_key: str = "type",
        target_node_type: int | None = None,
    ):
        self.kernel_fn = kernel_fn
        self.source_fn = source_fn
        self.update_node_fn = update_node_fn
        self.aggregate_fn = aggregate_fn
        self.input_key = input_key
        self.output_key = output_key
        self.edge_weight_key = edge_weight_key
        self.source_measure_key = source_measure_key
        self.source_measure = source_measure
        self.normalize = bool(normalize)
        self.node_type_key = str(node_type_key)
        self.target_node_type = (
            None if target_node_type is None else int(target_node_type)
        )

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.senders is None or graph.receivers is None:
            raise ValueError("GraphNeuralOperator requires explicit senders/receivers.")

        nodes = _node_field(graph, self.input_key, name="GraphNeuralOperator")
        num_nodes = _num_nodes(graph, nodes)
        relation = graph.edge_relation(node_count=num_nodes)
        source = nodes if self.source_fn is None else self.source_fn(nodes)
        sent_source = gather_routes(relation, source)
        sent_nodes = gather_routes(relation, nodes)
        recv_nodes = gather_routes(relation.transpose(), nodes)
        num_edges = _num_edges(graph)
        glob_edge = _repeat_globals_for_entities(graph.globals, graph.n_edge, num_edges)

        messages = sent_source
        if self.kernel_fn is not None:
            kernel = self.kernel_fn(graph.edges, sent_nodes, recv_nodes, glob_edge)
            messages = _multiply_tree(messages, kernel)

        edge_measure = None
        if self.source_measure_key is not None or self.source_measure is not None:
            node_measure = _node_measure(
                graph,
                self.source_measure_key,
                self.source_measure,
                n_node=num_nodes,
            )
            edge_measure = node_measure[relation.source_indices]
        edge_weight = _edge_weight(graph, self.edge_weight_key)
        if edge_weight is not None:
            messages = _multiply_tree(messages, edge_weight)
        if edge_measure is not None:
            messages = _multiply_tree(messages, edge_measure)
        messages = mask_routes(relation, messages)

        aggregated = jtu.tree_map(
            lambda x: self.aggregate_fn(x, relation.target_indices, num_nodes),
            messages,
        )
        if self.normalize:
            normalizer = jnp.ones((num_edges,), dtype=float)
            if edge_weight is not None:
                normalizer = normalizer * edge_weight
            if edge_measure is not None:
                normalizer = normalizer * edge_measure
            degree = route_reduce(relation, normalizer)
            scale = jnp.where(degree > 0, 1.0 / degree, 0.0)
            aggregated = _multiply_tree(aggregated, scale)

        glob_node = _repeat_globals_for_entities(graph.globals, graph.n_node, num_nodes)
        if self.update_node_fn is not None:
            aggregated = self.update_node_fn(nodes, aggregated, glob_node)
        aggregated = _mask_node_type(
            graph,
            aggregated,
            self.target_node_type,
            node_type_key=self.node_type_key,
        )
        aggregated = _mask_tree(aggregated, graph.node_mask)
        return graph.replace(
            nodes=_with_node_output(graph, aggregated, self.output_key),
            validate=False,
        )


class GraphAttentionOperator(eqx.Module):
    """Edge-aware attention operator over a sparse graph.

    By default this computes scaled dot-product attention from source nodes to
    receiver nodes. Optional callbacks can provide query/key/value maps, custom
    logits, edge bias, and a final node update while preserving the same
    `GraphIR -> GraphIR` surface as other graph SciML operators.
    """

    query_fn: Callable | None
    key_fn: Callable | None
    value_fn: Callable | None
    logit_fn: Callable | None
    update_node_fn: Callable | None
    source_measure: Any
    input_key: str | None = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)
    edge_bias_key: str | None = eqx.field(static=True)
    flow: str = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    head_reduction: str = eqx.field(static=True)
    node_type_key: str = eqx.field(static=True)
    source_measure_key: str | None = eqx.field(static=True)
    measure_eps: float = eqx.field(static=True)
    target_node_type: int | None = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        query_fn: Callable | None = None,
        key_fn: Callable | None = None,
        value_fn: Callable | None = None,
        logit_fn: Callable | None = None,
        update_node_fn: Callable | None = None,
        input_key: str | None = None,
        output_key: str | None = None,
        edge_bias_key: str | None = None,
        source_measure_key: str | None = None,
        source_measure: Any | None = None,
        measure_eps: float = 1e-12,
        flow: str = "source_to_target",
        temperature: float | None = None,
        head_reduction: str = "concat",
        node_type_key: str = "type",
        target_node_type: int | None = None,
    ):
        if flow not in ("source_to_target", "target_to_source"):
            raise ValueError("flow must be 'source_to_target' or 'target_to_source'.")
        if head_reduction not in ("concat", "mean"):
            raise ValueError("head_reduction must be 'concat' or 'mean'.")
        self.query_fn = query_fn
        self.key_fn = key_fn
        self.value_fn = value_fn
        self.logit_fn = logit_fn
        self.update_node_fn = update_node_fn
        self.input_key = input_key
        self.output_key = output_key
        self.edge_bias_key = edge_bias_key
        self.source_measure_key = source_measure_key
        self.source_measure = source_measure
        self.measure_eps = float(measure_eps)
        self.flow = flow
        self.temperature = 0.0 if temperature is None else float(temperature)
        self.head_reduction = head_reduction
        self.node_type_key = str(node_type_key)
        self.target_node_type = (
            None if target_node_type is None else int(target_node_type)
        )

    def _oriented_edges(self, graph: GraphIR, /) -> tuple[jnp.ndarray, jnp.ndarray]:
        if graph.senders is None or graph.receivers is None:
            raise ValueError(
                "GraphAttentionOperator requires explicit senders/receivers."
            )
        if self.flow == "source_to_target":
            return graph.senders, graph.receivers
        return graph.receivers, graph.senders

    def _logits(
        self,
        graph: GraphIR,
        queries: jnp.ndarray,
        keys: jnp.ndarray,
        source: jnp.ndarray,
        target: jnp.ndarray,
        glob_edge: ArrayTree | None,
        /,
    ) -> jnp.ndarray:
        if self.logit_fn is None:
            raw = jnp.sum(keys[source] * queries[target], axis=-1)
            temperature = (
                jnp.sqrt(jnp.asarray(keys.shape[-1], dtype=raw.dtype))
                if self.temperature <= 0.0
                else jnp.asarray(self.temperature, dtype=raw.dtype)
            )
            logits = raw / jnp.maximum(temperature, jnp.asarray(1e-12, dtype=raw.dtype))
        else:
            logits = jnp.asarray(
                self.logit_fn(graph.edges, keys[source], queries[target], glob_edge),
                dtype=float,
            )
            if logits.ndim == 2 and int(logits.shape[1]) == 1:
                logits = logits[:, 0]
            if logits.ndim not in (1, 2):
                raise ValueError(
                    "GraphAttentionOperator logit_fn must return shape "
                    "(n_edge,), (n_edge, 1), or (n_edge, n_head)."
                )
        bias = _edge_bias(graph, self.edge_bias_key)
        if bias is not None:
            logits = logits + bias
        return logits

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        nodes = _node_array(graph, self.input_key, name="GraphAttentionOperator")
        queries = (
            nodes
            if self.query_fn is None
            else jnp.asarray(self.query_fn(nodes), dtype=float)
        )
        keys = (
            nodes if self.key_fn is None else jnp.asarray(self.key_fn(nodes), dtype=float)
        )
        values = (
            nodes
            if self.value_fn is None
            else jnp.asarray(self.value_fn(nodes), dtype=float)
        )
        if queries.ndim != 2 or keys.ndim != 2 or values.ndim != 2:
            raise ValueError("query_fn, key_fn, and value_fn must return rank-2 arrays.")
        if queries.shape != keys.shape:
            raise ValueError("GraphAttentionOperator queries and keys must share shape.")

        source, target = self._oriented_edges(graph)
        num_edges = _num_edges(graph)
        glob_edge = _repeat_globals_for_entities(graph.globals, graph.n_edge, num_edges)
        logits = self._logits(graph, queries, keys, source, target, glob_edge)
        if graph.edge_mask is not None:
            mask = graph.edge_mask
            while mask.ndim < logits.ndim:
                mask = jnp.expand_dims(mask, axis=-1)
            logits = jnp.where(mask, logits, jnp.asarray(-1e30, dtype=logits.dtype))

        if self.source_measure_key is not None or self.source_measure is not None:
            node_measure = _node_measure(
                graph,
                self.source_measure_key,
                self.source_measure,
                n_node=int(nodes.shape[0]),
            )
            edge_measure = node_measure[source]
            measure_logits = jnp.where(
                edge_measure > 0.0,
                jnp.log(jnp.maximum(edge_measure, self.measure_eps)),
                jnp.asarray(-1e30, dtype=logits.dtype),
            )
            while measure_logits.ndim < logits.ndim:
                measure_logits = jnp.expand_dims(measure_logits, axis=-1)
            logits = logits + measure_logits
        weights = segment_softmax(logits, target, int(nodes.shape[0]))
        if graph.edge_mask is not None:
            mask = graph.edge_mask
            while mask.ndim < weights.ndim:
                mask = jnp.expand_dims(mask, axis=-1)
            weights = weights * mask.astype(weights.dtype)

        sent_values = values[source]
        if weights.ndim == 1:
            messages = sent_values * weights[:, None]
            out = segment_sum(messages, target, int(nodes.shape[0]))
        else:
            messages = sent_values[:, None, :] * weights[:, :, None]
            headed = segment_sum(messages, target, int(nodes.shape[0]))
            if self.head_reduction == "mean":
                out = jnp.mean(headed, axis=1)
            else:
                out = headed.reshape((headed.shape[0], headed.shape[1] * headed.shape[2]))

        glob_node = _repeat_globals_for_entities(
            graph.globals, graph.n_node, int(nodes.shape[0])
        )
        if self.update_node_fn is not None:
            out = self.update_node_fn(nodes, out, glob_node)
        out = _mask_node_type(
            graph,
            out,
            self.target_node_type,
            node_type_key=self.node_type_key,
        )
        out = _mask_tree(out, graph.node_mask)
        return graph.replace(
            nodes=_with_node_output(graph, out, self.output_key), validate=False
        )


class GraphFiniteVolumeDivergence(eqx.Module):
    """Conservative finite-volume divergence from edge fluxes to graph nodes."""

    flux_key: str | None = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)
    volume_key: str | None = eqx.field(static=True)
    volume: Any
    normalize_by_volume: bool = eqx.field(static=True)
    sign: FiniteVolumeSign = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        flux_key: str | None = "flux",
        output_key: str | None = "divergence",
        volume_key: str | None = "area",
        volume: Any | None = None,
        normalize_by_volume: bool = True,
        sign: FiniteVolumeSign = "in_minus_out",
    ):
        if sign not in ("in_minus_out", "out_minus_in"):
            raise ValueError("sign must be 'in_minus_out' or 'out_minus_in'.")
        self.flux_key = flux_key
        self.output_key = output_key
        self.volume_key = volume_key
        self.volume = volume
        self.normalize_by_volume = bool(normalize_by_volume)
        self.sign = sign

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        flux = _edge_array(graph, self.flux_key, name="GraphFiniteVolumeDivergence")
        out = _finite_volume_divergence(
            graph,
            flux,
            volume_key=self.volume_key,
            volume=self.volume,
            normalize_by_volume=self.normalize_by_volume,
            sign=self.sign,
        )
        return graph.replace(
            nodes=_with_node_output(graph, out, self.output_key), validate=False
        )


class GraphFiniteVolumeDiffusion(eqx.Module):
    """Finite-volume diffusion operator over cell-centered graph node fields."""

    input_key: str | None = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)
    conductivity_key: str | None = eqx.field(static=True)
    distance_key: str | None = eqx.field(static=True)
    volume_key: str | None = eqx.field(static=True)
    conductivity: Any
    volume: Any
    normalize_by_volume: bool = eqx.field(static=True)
    sign: FiniteVolumeSign = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        input_key: str | None = None,
        output_key: str | None = "diffusion",
        conductivity_key: str | None = None,
        conductivity: Any | None = None,
        distance_key: str | None = "distance",
        volume_key: str | None = "area",
        volume: Any | None = None,
        normalize_by_volume: bool = True,
        sign: FiniteVolumeSign = "in_minus_out",
        eps: float = 1e-12,
    ):
        if sign not in ("in_minus_out", "out_minus_in"):
            raise ValueError("sign must be 'in_minus_out' or 'out_minus_in'.")
        self.input_key = input_key
        self.output_key = output_key
        self.conductivity_key = conductivity_key
        self.conductivity = conductivity
        self.distance_key = distance_key
        self.volume_key = volume_key
        self.volume = volume
        self.normalize_by_volume = bool(normalize_by_volume)
        self.sign = sign
        self.eps = float(eps)

    def _conductivity(self, graph: GraphIR, flux: jnp.ndarray, /) -> jnp.ndarray:
        if self.conductivity is not None:
            return jnp.asarray(self.conductivity, dtype=flux.dtype)
        if self.conductivity_key is None:
            return jnp.asarray(1.0, dtype=flux.dtype)
        return _edge_array(
            graph,
            self.conductivity_key,
            name="GraphFiniteVolumeDiffusion conductivity",
        ).astype(flux.dtype)

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.senders is None or graph.receivers is None:
            raise ValueError(
                "GraphFiniteVolumeDiffusion requires explicit senders/receivers."
            )
        nodes = _node_array(graph, self.input_key, name="GraphFiniteVolumeDiffusion")
        sent = nodes[graph.senders]
        recv = nodes[graph.receivers]
        flux = sent - recv
        if self.distance_key is not None:
            distance = _edge_array(
                graph,
                self.distance_key,
                name="GraphFiniteVolumeDiffusion distance",
            )
            if distance.ndim == 2 and int(distance.shape[1]) == 1:
                distance = distance[:, 0]
            if distance.ndim != 1:
                raise ValueError(
                    "distance_key must reference shape (n_edge,) or (n_edge, 1)."
                )
            flux = flux / jnp.maximum(distance, self.eps)[:, None]
        flux = _multiply_leaf(flux, self._conductivity(graph, flux))
        out = _finite_volume_divergence(
            graph,
            flux,
            volume_key=self.volume_key,
            volume=self.volume,
            normalize_by_volume=self.normalize_by_volume,
            sign=self.sign,
        )
        return graph.replace(
            nodes=_with_node_output(graph, out, self.output_key), validate=False
        )


class GraphProcessor(eqx.Module):
    """Sequential processor for `GraphIR -> GraphIR` blocks."""

    blocks: tuple[Callable[[GraphIR], GraphIR], ...]

    def __init__(self, blocks: Sequence[Callable[[GraphIR], GraphIR]], /):
        if len(blocks) == 0:
            raise ValueError("GraphProcessor requires at least one block.")
        self.blocks = tuple(blocks)

    def __call__(self, graph: GraphIR) -> GraphIR:
        out = graph
        for block in self.blocks:
            out = block(out)
            if not isinstance(out, GraphIR):
                raise TypeError("GraphProcessor blocks must return GraphIR values.")
        return out


class RepeatedGraphProcessor(eqx.Module):
    """Apply one graph block repeatedly."""

    block: Callable[[GraphIR], GraphIR]
    steps: int = eqx.field(static=True)

    def __init__(self, block: Callable[[GraphIR], GraphIR], /, *, steps: int):
        self.block = block
        self.steps = int(steps)
        if self.steps < 0:
            raise ValueError("steps must be non-negative.")

    def __call__(self, graph: GraphIR) -> GraphIR:
        out = graph
        for _ in range(self.steps):
            out = self.block(out)
            if not isinstance(out, GraphIR):
                raise TypeError("RepeatedGraphProcessor block must return a GraphIR.")
        return out


__all__ = [
    "GraphAttentionOperator",
    "GraphDiffusion",
    "GraphFiniteVolumeDiffusion",
    "GraphFiniteVolumeDivergence",
    "GraphKernelIntegral",
    "GraphNeuralOperator",
    "GraphProcessor",
    "RepeatedGraphProcessor",
]
