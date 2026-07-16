from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from ._graph import ensure_graph
from ._ir import GraphIR
from ._kernels import segment_sum


GraphFilterOperator = Literal["laplacian", "adjacency"]
GraphFlow = Literal["source_to_target", "target_to_source"]
GraphLaplacianNormalization = Literal["none", "random_walk", "symmetric"]


def _tree_leading_size(tree: Any, /) -> int:
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        raise ValueError("Feature tree must contain at least one array leaf.")
    return int(jnp.asarray(leaves[0]).shape[0])


def _num_nodes(graph: GraphIR, nodes: Any, /) -> int:
    if graph.node_mask is not None:
        return int(graph.node_mask.shape[0])
    return _tree_leading_size(nodes)


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
        return jtu.tree_map(_multiply_leaf, tree, weight)
    return jtu.tree_map(lambda x: _multiply_leaf(x, weight), tree)


def _mask_tree(tree: Any, mask: jnp.ndarray | None, /) -> Any:
    if mask is None:
        return tree
    return jtu.tree_map(lambda x: _multiply_leaf(x, mask.astype(x.dtype)), tree)


def _tree_add(a: Any, b: Any, /) -> Any:
    return jtu.tree_map(lambda x, y: x + y, a, b)


def _tree_sub(a: Any, b: Any, /) -> Any:
    return jtu.tree_map(lambda x, y: x - y, a, b)


def _tree_zeros_like(tree: Any, /) -> Any:
    return jtu.tree_map(jnp.zeros_like, tree)


def _tree_index(tree: Any, index: jnp.ndarray, /) -> Any:
    return jtu.tree_map(lambda x: x[index], tree)


def _as_feature_mapping(value: Any, /) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"features": value}


def _node_field(graph: GraphIR, input_key: str | None, /) -> Any:
    if graph.nodes is None:
        raise ValueError("Spectral graph operators require node features.")
    if input_key is None:
        return graph.nodes
    if not isinstance(graph.nodes, Mapping):
        raise TypeError("input_key requires mapping-valued graph nodes.")
    if input_key not in graph.nodes:
        raise KeyError(f"Graph nodes do not contain input_key {input_key!r}.")
    return graph.nodes[input_key]


def _with_node_output(graph: GraphIR, value: Any, output_key: str | None, /) -> Any:
    if output_key is None:
        return value
    nodes = _as_feature_mapping(graph.nodes)
    nodes[output_key] = value
    return nodes


def _edge_weight(
    graph: GraphIR,
    /,
    *,
    weight: Any = None,
    weight_key: str | None = None,
) -> jnp.ndarray:
    if graph.senders is None:
        raise ValueError("Spectral graph operators require explicit senders/receivers.")
    if weight is not None:
        out = jnp.asarray(weight, dtype=float)
    elif weight_key is not None:
        if not isinstance(graph.edges, Mapping):
            raise TypeError("weight_key requires mapping-valued graph edges.")
        if weight_key not in graph.edges:
            raise KeyError(f"Graph edges do not contain weight_key {weight_key!r}.")
        out = jnp.asarray(graph.edges[weight_key], dtype=float)
    else:
        out = jnp.ones((graph.senders.shape[0],), dtype=float)
    if out.ndim == 2 and int(out.shape[1]) == 1:
        out = out[:, 0]
    if out.ndim != 1:
        raise ValueError("Graph spectral edge weights must have shape (n_edge,) or (n_edge, 1).")
    if int(out.shape[0]) != int(graph.senders.shape[0]):
        raise ValueError("Graph spectral edge weights must match edge count.")
    if graph.edge_mask is not None:
        out = out * graph.edge_mask.astype(out.dtype)
    return out


def _oriented_edges(graph: GraphIR, flow: GraphFlow, /) -> tuple[jnp.ndarray, jnp.ndarray]:
    if graph.senders is None or graph.receivers is None:
        raise ValueError("Spectral graph operators require explicit senders/receivers.")
    if flow == "source_to_target":
        return graph.senders, graph.receivers
    if flow == "target_to_source":
        return graph.receivers, graph.senders
    raise ValueError("flow must be 'source_to_target' or 'target_to_source'.")


def graph_adjacency_apply(
    graph: GraphIR,
    nodes: Any | None = None,
    /,
    *,
    weight: Any = None,
    weight_key: str | None = None,
    flow: GraphFlow = "source_to_target",
    normalization: GraphLaplacianNormalization = "none",
) -> Any:
    """Apply sparse weighted adjacency to node features."""
    graph = ensure_graph(graph, validate=False)
    if normalization not in ("none", "random_walk", "symmetric"):
        raise ValueError("normalization must be 'none', 'random_walk', or 'symmetric'.")
    x = graph.nodes if nodes is None else nodes
    if x is None:
        raise ValueError("graph_adjacency_apply requires node features.")
    source, target = _oriented_edges(graph, flow)
    n = _num_nodes(graph, x)
    edge_weight = _edge_weight(graph, weight=weight, weight_key=weight_key)
    degree = segment_sum(edge_weight, target, n)
    messages = _tree_index(x, source)

    if normalization == "random_walk":
        inv_degree = jnp.where(degree > 0, 1.0 / degree, 0.0)
        edge_weight = edge_weight * inv_degree[target]
    elif normalization == "symmetric":
        inv_sqrt = jnp.where(degree > 0, jax_lax_rsqrt(degree), 0.0)
        edge_weight = edge_weight * inv_sqrt[source] * inv_sqrt[target]

    messages = _multiply_tree(messages, edge_weight)
    out = jtu.tree_map(lambda y: segment_sum(y, target, n), messages)
    return _mask_tree(out, graph.node_mask)


def jax_lax_rsqrt(x: jnp.ndarray, /) -> jnp.ndarray:
    return jnp.asarray(1.0, dtype=x.dtype) / jnp.sqrt(jnp.maximum(x, 1e-30))


def graph_laplacian_apply(
    graph: GraphIR,
    nodes: Any | None = None,
    /,
    *,
    weight: Any = None,
    weight_key: str | None = None,
    flow: GraphFlow = "source_to_target",
    normalization: GraphLaplacianNormalization = "symmetric",
) -> Any:
    """Apply a sparse graph Laplacian to node features."""
    graph = ensure_graph(graph, validate=False)
    if normalization not in ("none", "random_walk", "symmetric"):
        raise ValueError("normalization must be 'none', 'random_walk', or 'symmetric'.")
    x = graph.nodes if nodes is None else nodes
    if x is None:
        raise ValueError("graph_laplacian_apply requires node features.")
    source, target = _oriented_edges(graph, flow)
    n = _num_nodes(graph, x)
    edge_weight = _edge_weight(graph, weight=weight, weight_key=weight_key)
    degree = segment_sum(edge_weight, target, n)

    if normalization == "none":
        adjacency = graph_adjacency_apply(
            graph,
            x,
            weight=edge_weight,
            flow=flow,
            normalization="none",
        )
        diagonal = _multiply_tree(x, degree)
        out = _tree_sub(diagonal, adjacency)
    else:
        adjacency = graph_adjacency_apply(
            graph,
            x,
            weight=edge_weight,
            flow=flow,
            normalization=normalization,
        )
        out = _tree_sub(x, adjacency)
    return _mask_tree(out, graph.node_mask)


def _apply_operator(
    graph: GraphIR,
    nodes: Any,
    /,
    *,
    operator: GraphFilterOperator,
    weight: Any,
    weight_key: str | None,
    flow: GraphFlow,
    normalization: GraphLaplacianNormalization,
) -> Any:
    if operator == "laplacian":
        return graph_laplacian_apply(
            graph,
            nodes,
            weight=weight,
            weight_key=weight_key,
            flow=flow,
            normalization=normalization,
        )
    if operator == "adjacency":
        return graph_adjacency_apply(
            graph,
            nodes,
            weight=weight,
            weight_key=weight_key,
            flow=flow,
            normalization=normalization,
        )
    raise ValueError("operator must be 'laplacian' or 'adjacency'.")


def _apply_coefficient_leaf(term: Any, coeff: Any, /) -> jnp.ndarray:
    term_arr = jnp.asarray(term)
    coeff_arr = jnp.asarray(coeff, dtype=term_arr.dtype)
    if coeff_arr.ndim == 0:
        return term_arr * coeff_arr
    if (
        coeff_arr.ndim == 1
        and term_arr.ndim >= 2
        and int(coeff_arr.shape[0]) == int(term_arr.shape[-1])
    ):
        return term_arr * coeff_arr
    if (
        coeff_arr.ndim == 2
        and term_arr.ndim == 2
        and int(coeff_arr.shape[0]) == int(term_arr.shape[-1])
    ):
        return term_arr @ coeff_arr
    return term_arr * coeff_arr


def _apply_coefficient(term: Any, coeff: Any, /) -> Any:
    if jtu.tree_structure(term) == jtu.tree_structure(coeff):
        return jtu.tree_map(_apply_coefficient_leaf, term, coeff)
    return jtu.tree_map(lambda x: _apply_coefficient_leaf(x, coeff), term)


def _coeff_at(coefficients: Any, index: int, /) -> Any:
    if isinstance(coefficients, Mapping):
        return {key: _coeff_at(value, index) for key, value in coefficients.items()}
    return jnp.asarray(coefficients)[index]


class GraphLaplacianOperator(eqx.Module):
    """`GraphIR -> GraphIR` sparse graph Laplacian block."""

    weight: Any
    weight_key: str | None = eqx.field(static=True)
    input_key: str | None = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)
    flow: GraphFlow = eqx.field(static=True)
    normalization: GraphLaplacianNormalization = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        weight: Any = None,
        weight_key: str | None = None,
        input_key: str | None = None,
        output_key: str | None = None,
        flow: GraphFlow = "source_to_target",
        normalization: GraphLaplacianNormalization = "symmetric",
    ):
        self.weight = weight
        self.weight_key = weight_key
        self.input_key = input_key
        self.output_key = output_key
        self.flow = flow
        self.normalization = normalization

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        out = graph_laplacian_apply(
            graph,
            _node_field(graph, self.input_key),
            weight=self.weight,
            weight_key=self.weight_key,
            flow=self.flow,
            normalization=self.normalization,
        )
        return graph.replace(nodes=_with_node_output(graph, out, self.output_key), validate=False)


class GraphPolynomialFilter(eqx.Module):
    """Polynomial graph filter over powers of sparse adjacency or Laplacian."""

    coefficients: Any
    weight: Any
    weight_key: str | None = eqx.field(static=True)
    input_key: str | None = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)
    operator: GraphFilterOperator = eqx.field(static=True)
    flow: GraphFlow = eqx.field(static=True)
    normalization: GraphLaplacianNormalization = eqx.field(static=True)

    def __init__(
        self,
        coefficients: Any,
        /,
        *,
        weight: Any = None,
        weight_key: str | None = None,
        input_key: str | None = None,
        output_key: str | None = None,
        operator: GraphFilterOperator = "laplacian",
        flow: GraphFlow = "source_to_target",
        normalization: GraphLaplacianNormalization = "symmetric",
    ):
        coeff_leaves = jtu.tree_leaves(coefficients)
        if not coeff_leaves:
            raise ValueError("GraphPolynomialFilter coefficients must be non-empty.")
        if int(jnp.asarray(coeff_leaves[0]).shape[0]) <= 0:
            raise ValueError("GraphPolynomialFilter requires at least one coefficient.")
        self.coefficients = coefficients
        self.weight = weight
        self.weight_key = weight_key
        self.input_key = input_key
        self.output_key = output_key
        self.operator = operator
        self.flow = flow
        self.normalization = normalization

    @property
    def order(self) -> int:
        return int(jnp.asarray(jtu.tree_leaves(self.coefficients)[0]).shape[0]) - 1

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        x0 = _node_field(graph, self.input_key)
        term = x0
        out = _apply_coefficient(term, _coeff_at(self.coefficients, 0))
        for k in range(1, self.order + 1):
            term = _apply_operator(
                graph,
                term,
                operator=self.operator,
                weight=self.weight,
                weight_key=self.weight_key,
                flow=self.flow,
                normalization=self.normalization,
            )
            out = _tree_add(out, _apply_coefficient(term, _coeff_at(self.coefficients, k)))
        out = _mask_tree(out, graph.node_mask)
        return graph.replace(nodes=_with_node_output(graph, out, self.output_key), validate=False)


class GraphChebyshevFilter(eqx.Module):
    """Chebyshev polynomial filter over the scaled graph Laplacian."""

    coefficients: Any
    weight: Any
    weight_key: str | None = eqx.field(static=True)
    input_key: str | None = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)
    flow: GraphFlow = eqx.field(static=True)
    normalization: GraphLaplacianNormalization = eqx.field(static=True)
    lambda_max: jnp.ndarray

    def __init__(
        self,
        coefficients: Any,
        /,
        *,
        weight: Any = None,
        weight_key: str | None = None,
        input_key: str | None = None,
        output_key: str | None = None,
        flow: GraphFlow = "source_to_target",
        normalization: GraphLaplacianNormalization = "symmetric",
        lambda_max: float = 2.0,
    ):
        coeff_leaves = jtu.tree_leaves(coefficients)
        if not coeff_leaves:
            raise ValueError("GraphChebyshevFilter coefficients must be non-empty.")
        if int(jnp.asarray(coeff_leaves[0]).shape[0]) <= 0:
            raise ValueError("GraphChebyshevFilter requires at least one coefficient.")
        if float(lambda_max) <= 0:
            raise ValueError("lambda_max must be positive.")
        self.coefficients = coefficients
        self.weight = weight
        self.weight_key = weight_key
        self.input_key = input_key
        self.output_key = output_key
        self.flow = flow
        self.normalization = normalization
        self.lambda_max = jnp.asarray(lambda_max, dtype=float)

    @property
    def order(self) -> int:
        return int(jnp.asarray(jtu.tree_leaves(self.coefficients)[0]).shape[0]) - 1

    def _scaled_laplacian(self, graph: GraphIR, nodes: Any, /) -> Any:
        lap = graph_laplacian_apply(
            graph,
            nodes,
            weight=self.weight,
            weight_key=self.weight_key,
            flow=self.flow,
            normalization=self.normalization,
        )
        scaled = _multiply_tree(lap, 2.0 / self.lambda_max)
        return _tree_sub(scaled, nodes)

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        t0 = _node_field(graph, self.input_key)
        out = _apply_coefficient(t0, _coeff_at(self.coefficients, 0))
        if self.order >= 1:
            t1 = self._scaled_laplacian(graph, t0)
            out = _tree_add(out, _apply_coefficient(t1, _coeff_at(self.coefficients, 1)))
            prev, current = t0, t1
            for k in range(2, self.order + 1):
                next_term = _tree_sub(_multiply_tree(self._scaled_laplacian(graph, current), 2.0), prev)
                out = _tree_add(
                    out,
                    _apply_coefficient(next_term, _coeff_at(self.coefficients, k)),
                )
                prev, current = current, next_term
        out = _mask_tree(out, graph.node_mask)
        return graph.replace(nodes=_with_node_output(graph, out, self.output_key), validate=False)


__all__ = [
    "GraphChebyshevFilter",
    "GraphFilterOperator",
    "GraphFlow",
    "GraphLaplacianNormalization",
    "GraphLaplacianOperator",
    "GraphPolynomialFilter",
    "graph_adjacency_apply",
    "graph_laplacian_apply",
]
