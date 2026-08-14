from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import scipy.sparse as scipy_sparse

from .._spectral._modal import SpectralDiscretization
from ..sparse import linear_apply, route_reduce
from ._graph import ensure_graph
from ._ir import GraphIR


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

    def mask_leaf(value: Any, /) -> jnp.ndarray:
        array = jnp.asarray(value)
        expanded = mask.reshape(mask.shape + (1,) * (array.ndim - mask.ndim))
        return jnp.where(expanded, array, jnp.zeros((), dtype=array.dtype))

    return jtu.tree_map(mask_leaf, tree)


def _tree_add(a: Any, b: Any, /) -> Any:
    return jtu.tree_map(lambda x, y: x + y, a, b)


def _tree_sub(a: Any, b: Any, /) -> Any:
    return jtu.tree_map(lambda x, y: x - y, a, b)


def _tree_zeros_like(tree: Any, /) -> Any:
    return jtu.tree_map(jnp.zeros_like, tree)


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
        raise ValueError(
            "Graph spectral edge weights must have shape (n_edge,) or (n_edge, 1)."
        )
    if int(out.shape[0]) != int(graph.senders.shape[0]):
        raise ValueError("Graph spectral edge weights must match edge count.")
    return out


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
    n = _num_nodes(graph, x)
    relation = graph.edge_relation(node_count=n, flow=flow)
    edge_weight = _edge_weight(graph, weight=weight, weight_key=weight_key)
    degree = route_reduce(relation, edge_weight)

    if normalization == "random_walk":
        inv_degree = jnp.where(degree > 0, 1.0 / degree, 0.0)
        edge_weight = edge_weight * inv_degree[relation.target_indices]
    elif normalization == "symmetric":
        inv_sqrt = jnp.where(degree > 0, jax_lax_rsqrt(degree), 0.0)
        edge_weight = (
            edge_weight
            * inv_sqrt[relation.source_indices]
            * inv_sqrt[relation.target_indices]
        )

    out = linear_apply(relation, edge_weight, x)
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
    n = _num_nodes(graph, x)
    relation = graph.edge_relation(node_count=n, flow=flow)
    edge_weight = _edge_weight(graph, weight=weight, weight_key=weight_key)
    degree = route_reduce(relation, edge_weight)

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
        return graph.replace(
            nodes=_with_node_output(graph, out, self.output_key), validate=False
        )


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
            out = _tree_add(
                out, _apply_coefficient(term, _coeff_at(self.coefficients, k))
            )
        out = _mask_tree(out, graph.node_mask)
        return graph.replace(
            nodes=_with_node_output(graph, out, self.output_key), validate=False
        )


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
                next_term = _tree_sub(
                    _multiply_tree(self._scaled_laplacian(graph, current), 2.0), prev
                )
                out = _tree_add(
                    out,
                    _apply_coefficient(next_term, _coeff_at(self.coefficients, k)),
                )
                prev, current = current, next_term
        out = _mask_tree(out, graph.node_mask)
        return graph.replace(
            nodes=_with_node_output(graph, out, self.output_key), validate=False
        )


def spectral_discretization_from_graph(
    graph: GraphIR,
    /,
    *,
    n_modes: int,
    weight: Any = None,
    weight_key: str | None = None,
    mass: Any = None,
    mass_key: str | None = None,
    symmetrize: bool = True,
    group_tolerance: float = 1e-7,
    basis_id: str | None = None,
    max_construction_bytes: int = 512 * 1024**2,
) -> SpectralDiscretization:
    """Build a host-side weighted graph-Laplacian eigenbasis.

    Directed adjacency is averaged with its transpose when ``symmetrize`` is true;
    otherwise the weighted adjacency must already be symmetric.
    """
    graph_value = ensure_graph(graph, validate=True)
    counts = np.asarray(graph_value.n_node).reshape((-1,))
    if counts.size != 1:
        raise ValueError("Graph spectral plans require exactly one graph.")
    num_nodes = int(counts[0])
    if num_nodes <= 0:
        raise ValueError("Graph spectral plans require at least one node.")
    if graph_value.node_mask is not None and not bool(
        np.all(np.asarray(graph_value.node_mask))
    ):
        raise ValueError("Graph spectral plans do not accept padded or masked nodes.")
    if graph_value.senders is None or graph_value.receivers is None:
        raise ValueError("Graph spectral plans require explicit senders and receivers.")
    senders = np.asarray(graph_value.senders, dtype=np.int64).reshape((-1,))
    receivers = np.asarray(graph_value.receivers, dtype=np.int64).reshape((-1,))
    weights = np.asarray(
        _edge_weight(
            graph_value,
            weight=weight,
            weight_key=weight_key,
        ),
        dtype=float,
    )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Graph spectral edge weights must be finite and non-negative.")
    if min(max_construction_bytes, 0) == max_construction_bytes:
        raise ValueError("max_construction_bytes must be positive.")
    assembly_bytes = 6 * max(1, weights.size) * np.dtype(float).itemsize
    if assembly_bytes > int(max_construction_bytes):
        raise ValueError(
            "Graph Laplacian assembly exceeds max_construction_bytes; "
            f"estimated {assembly_bytes} bytes."
        )
    adjacency = scipy_sparse.coo_matrix(
        (weights, (senders, receivers)),
        shape=(num_nodes, num_nodes),
        dtype=float,
    ).tocsr()
    adjacency.sum_duplicates()
    if symmetrize:
        adjacency = 0.5 * (adjacency + adjacency.T)
    else:
        difference = adjacency - adjacency.T
        error = float(np.max(np.abs(difference.data))) if difference.data.size else 0.0
        scale = float(np.max(np.abs(adjacency.data))) if adjacency.data.size else 1.0
        if error > 1e-10 * max(1.0, scale):
            raise ValueError(
                "Graph adjacency must be symmetric when symmetrize is false."
            )
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1), dtype=float).reshape((-1,))
    stiffness = scipy_sparse.diags(degree, format="csr") - adjacency
    if mass is not None and mass_key is not None:
        raise ValueError("Specify mass or mass_key, not both.")
    if mass_key is not None:
        if not isinstance(graph_value.nodes, Mapping):
            raise TypeError("mass_key requires mapping-valued graph nodes.")
        if mass_key not in graph_value.nodes:
            raise KeyError(f"Graph nodes do not contain mass_key {mass_key!r}.")
        measure = graph_value.nodes[mass_key]
    elif mass is None:
        measure = np.ones((num_nodes,), dtype=float)
    else:
        measure = mass
    return SpectralDiscretization.from_stiffness(
        stiffness,
        measure,
        n_modes=n_modes,
        group_tolerance=group_tolerance,
        basis_id=basis_id,
        max_construction_bytes=max_construction_bytes,
    )


def spectral_discretization_from_triangle_mesh(
    mesh: Any,
    /,
    *,
    n_modes: int,
    group_tolerance: float = 1e-7,
    basis_id: str | None = None,
    max_construction_bytes: int = 512 * 1024**2,
) -> SpectralDiscretization:
    """Build a cotangent-FEM eigenbasis from a ``TriangleMesh``."""
    from phydrax.geometry.simplicial import DDGOperators, TriangleMesh

    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh.")
    operators = DDGOperators(mesh)
    edges = np.asarray(operators.edges, dtype=np.int64)
    weights = np.asarray(operators.edge_weights, dtype=float)
    num_nodes = int(mesh.vertices.shape[0])
    assembly_bytes = 8 * max(1, weights.size) * np.dtype(float).itemsize
    if int(max_construction_bytes) <= 0:
        raise ValueError("max_construction_bytes must be positive.")
    if assembly_bytes > int(max_construction_bytes):
        raise ValueError(
            "Cotangent Laplacian assembly exceeds max_construction_bytes; "
            f"estimated {assembly_bytes} bytes."
        )
    first = edges[:, 0]
    second = edges[:, 1]
    rows = np.concatenate((first, second, first, second))
    columns = np.concatenate((second, first, first, second))
    data = np.concatenate((-weights, -weights, weights, weights))
    stiffness = scipy_sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(num_nodes, num_nodes),
        dtype=float,
    ).tocsr()
    stiffness.sum_duplicates()
    return SpectralDiscretization.from_stiffness(
        stiffness,
        np.asarray(operators.vertex_mass, dtype=float),
        n_modes=n_modes,
        group_tolerance=group_tolerance,
        basis_id=basis_id,
        max_construction_bytes=max_construction_bytes,
    )


__all__ = [
    "GraphChebyshevFilter",
    "GraphFilterOperator",
    "GraphFlow",
    "GraphLaplacianNormalization",
    "GraphLaplacianOperator",
    "GraphPolynomialFilter",
    "graph_adjacency_apply",
    "spectral_discretization_from_graph",
    "spectral_discretization_from_triangle_mesh",
    "graph_laplacian_apply",
]
