#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse metric discrete-exterior-calculus operators on :class:`GraphIR`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import core as jax_core
from jaxtyping import Array

from .._strict import StrictModule
from ._cochain import CochainBoundaryKind, CochainBoundaryPolicy
from ._ir import GraphIR
from ._kernels import segment_sum


HodgeLaplacianComponent: TypeAlias = Literal["lower", "upper", "complete"]


def _cochain_payload(graph: GraphIR, /) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(graph, GraphIR):
        raise TypeError("Metric cochain operators require a GraphIR.")
    if not isinstance(graph.nodes, Mapping) or not isinstance(graph.edges, Mapping):
        raise ValueError("Metric cochain operators require named node and edge payloads.")
    required_nodes = {"cell_dim", "hodge_star", "boundary"}
    required_edges = {
        "cochain_incidence",
        "incidence_degree",
        "incidence_direction",
        "incidence_sign",
    }
    if not required_nodes.issubset(graph.nodes):
        raise ValueError("GraphIR node payload is missing canonical cochain metadata.")
    if not required_edges.issubset(graph.edges):
        raise ValueError("GraphIR edge payload is missing canonical incidence metadata.")
    if graph.senders is None or graph.receivers is None:
        raise ValueError("Metric cochain operators require explicit incidence indices.")
    return graph.nodes, graph.edges


def _node_count(graph: GraphIR, nodes: Mapping[str, Any], /) -> int:
    return int(jnp.asarray(nodes["cell_dim"]).shape[0])


def _reshape_coefficient(coefficient: Array, values: Array, /) -> Array:
    return coefficient.reshape((coefficient.shape[0],) + (1,) * (values.ndim - 1))


def _active_nodes(
    graph: GraphIR,
    nodes: Mapping[str, Any],
    degree: int,
    boundary_policy: CochainBoundaryPolicy,
    /,
) -> Array:
    active = jnp.asarray(nodes["cell_dim"]) == int(degree)
    if boundary_policy.kind == "relative":
        active = active & ~jnp.asarray(nodes["boundary"], dtype=bool)
    if graph.node_mask is not None:
        active = active & graph.node_mask
    return active


def _forward_incidence_mask(
    graph: GraphIR,
    edges: Mapping[str, Any],
    incidence_degree: int,
    active_nodes: Array,
    /,
) -> Array:
    assert graph.senders is not None
    assert graph.receivers is not None
    mask = (
        jnp.asarray(edges["cochain_incidence"], dtype=bool)
        & (jnp.asarray(edges["incidence_direction"]) == 1)
        & (jnp.asarray(edges["incidence_degree"]) == int(incidence_degree))
        & active_nodes[graph.senders]
        & active_nodes[graph.receivers]
    )
    if graph.edge_mask is not None:
        mask = mask & graph.edge_mask
    return mask


def _validate_values(values: Any, node_count: int, /) -> Array:
    array = jnp.asarray(values)
    if array.ndim == 0 or int(array.shape[0]) != node_count:
        raise ValueError(
            f"Cochain values require leading graph-node size {node_count}; got {array.shape}."
        )
    return array


def cochain_exterior_derivative(
    graph: GraphIR,
    values: Any,
    degree: int,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
) -> Array:
    """Apply ``d_degree = B_(degree+1)^T`` to full graph-node cochain values."""
    nodes, edges = _cochain_payload(graph)
    source_degree = int(degree)
    if source_degree < 0:
        raise ValueError("Exterior derivative degree must be non-negative.")
    policy = CochainBoundaryPolicy(boundary_policy)
    array = _validate_values(values, _node_count(graph, nodes))
    source_active = _active_nodes(graph, nodes, source_degree, policy)
    target_active = _active_nodes(graph, nodes, source_degree + 1, policy)
    route_active = source_active | target_active
    mask = _forward_incidence_mask(
        graph, edges, source_degree + 1, route_active
    )
    assert graph.senders is not None
    assert graph.receivers is not None
    signs = jnp.asarray(edges["incidence_sign"], dtype=array.dtype)
    coefficient = _reshape_coefficient(signs * mask.astype(array.dtype), array)
    messages = coefficient * array[graph.senders]
    output = segment_sum(messages, graph.receivers, _node_count(graph, nodes))
    target_shape = (target_active.shape[0],) + (1,) * (array.ndim - 1)
    return jnp.where(target_active.reshape(target_shape), output, 0)


def cochain_codifferential(
    graph: GraphIR,
    values: Any,
    degree: int,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
) -> Array:
    """Apply ``δ_degree = M_(degree-1)^-1 B_degree M_degree``."""
    nodes, edges = _cochain_payload(graph)
    source_degree = int(degree)
    if source_degree <= 0:
        raise ValueError("Codifferential degree must be positive.")
    policy = CochainBoundaryPolicy(boundary_policy)
    array = _validate_values(values, _node_count(graph, nodes))
    lower_active = _active_nodes(graph, nodes, source_degree - 1, policy)
    upper_active = _active_nodes(graph, nodes, source_degree, policy)
    route_active = lower_active | upper_active
    mask = _forward_incidence_mask(graph, edges, source_degree, route_active)
    assert graph.senders is not None
    assert graph.receivers is not None
    signs = jnp.asarray(edges["incidence_sign"], dtype=array.dtype)
    star = jnp.asarray(nodes["hodge_star"], dtype=array.dtype)
    coefficient = signs * mask.astype(array.dtype) * star[graph.receivers]
    messages = _reshape_coefficient(coefficient, array) * array[graph.receivers]
    accumulated = segment_sum(messages, graph.senders, _node_count(graph, nodes))
    inverse_star = jnp.where(star > 0, 1.0 / star, 0.0)
    output = accumulated * inverse_star.reshape(
        (inverse_star.shape[0],) + (1,) * (array.ndim - 1)
    )
    target_shape = (lower_active.shape[0],) + (1,) * (array.ndim - 1)
    return jnp.where(lower_active.reshape(target_shape), output, 0)


def cochain_hodge_laplacian(
    graph: GraphIR,
    values: Any,
    degree: int,
    /,
    *,
    component: HodgeLaplacianComponent = "complete",
    boundary_policy: CochainBoundaryKind = "absolute",
) -> Array:
    """Apply a lower, upper, or complete metric Hodge Laplacian."""
    if component not in ("lower", "upper", "complete"):
        raise ValueError("Hodge Laplacian component must be 'lower', 'upper', or 'complete'.")
    nodes, _ = _cochain_payload(graph)
    resolved_degree = int(degree)
    if resolved_degree < 0:
        raise ValueError("Hodge Laplacian degree must be non-negative.")
    array = _validate_values(values, _node_count(graph, nodes))
    output = jnp.zeros_like(array)
    if component in ("lower", "complete") and resolved_degree > 0:
        lowered = cochain_codifferential(
            graph,
            array,
            resolved_degree,
            boundary_policy=boundary_policy,
        )
        output = output + cochain_exterior_derivative(
            graph,
            lowered,
            resolved_degree - 1,
            boundary_policy=boundary_policy,
        )
    if component in ("upper", "complete"):
        raised = cochain_exterior_derivative(
            graph,
            array,
            resolved_degree,
            boundary_policy=boundary_policy,
        )
        output = output + cochain_codifferential(
            graph,
            raised,
            resolved_degree + 1,
            boundary_policy=boundary_policy,
        )
    return output


def cochain_harmonic_projection(
    graph: GraphIR,
    values: Any,
    degree: int,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
) -> Array:
    """Apply the basis-independent metric projector onto ``ker(Δ_degree)``."""
    nodes, _ = _cochain_payload(graph)
    if "harmonic_basis" not in nodes or not isinstance(graph.globals, Mapping):
        raise ValueError("GraphIR has no precomputed harmonic subspace.")
    if "harmonic_rank" not in graph.globals or "harmonic_boundary_policy" not in graph.globals:
        raise ValueError("GraphIR harmonic metadata is incomplete.")
    policy = CochainBoundaryPolicy(boundary_policy)
    boundary_code = jnp.asarray(graph.globals["harmonic_boundary_policy"])
    expected_code = 0 if policy.kind == "absolute" else 1
    if not isinstance(boundary_code, jax_core.Tracer) and bool(
        jnp.any(boundary_code != expected_code)
    ):
        raise ValueError("Harmonic subspace uses a different boundary policy.")
    array = _validate_values(values, _node_count(graph, nodes))
    original_shape = array.shape
    flat = array.reshape((array.shape[0], -1))
    basis = jnp.asarray(nodes["harmonic_basis"], dtype=array.dtype)[:, int(degree), :]
    graph_ids = jnp.repeat(
        jnp.arange(graph.num_graphs, dtype=jnp.int32),
        graph.n_node,
        total_repeat_length=int(array.shape[0]),
    )
    ranks = jnp.asarray(graph.globals["harmonic_rank"])[:, int(degree)]
    mode_ids = jnp.arange(basis.shape[1], dtype=jnp.int32)
    mode_mask = mode_ids[None, :] < ranks[graph_ids, None]
    basis = jnp.where(mode_mask, basis, 0)
    star = jnp.asarray(nodes["hodge_star"], dtype=array.dtype)
    weighted = basis[:, :, None] * star[:, None, None] * flat[:, None, :]
    coefficients = segment_sum(weighted, graph_ids, graph.num_graphs)
    projected = jnp.sum(basis[:, :, None] * coefficients[graph_ids], axis=1)
    active = _active_nodes(graph, nodes, int(degree), policy)
    projected = jnp.where(active[:, None], projected, 0)
    return projected.reshape(original_shape)


def _replace_node_output(
    graph: GraphIR,
    output_key: str,
    values: Array,
    /,
) -> GraphIR:
    if not isinstance(graph.nodes, Mapping):
        raise ValueError("Cochain graph nodes must be a mapping.")
    return graph.replace(nodes={**graph.nodes, output_key: values}, validate=False)


class CochainExteriorDerivative(StrictModule):
    """GraphIR wrapper for a metric exterior derivative."""

    degree: int = eqx.field(static=True)
    input_key: str = eqx.field(static=True)
    output_key: str = eqx.field(static=True)
    boundary_policy: CochainBoundaryKind = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        input_key: str,
        output_key: str,
        boundary_policy: CochainBoundaryKind = "absolute",
    ):
        self.degree = int(degree)
        self.input_key = str(input_key)
        self.output_key = str(output_key)
        self.boundary_policy = CochainBoundaryPolicy(boundary_policy).kind

    def __call__(self, graph: GraphIR, /) -> GraphIR:
        if not isinstance(graph.nodes, Mapping) or self.input_key not in graph.nodes:
            raise KeyError(f"Missing cochain node field {self.input_key!r}.")
        values = cochain_exterior_derivative(
            graph,
            graph.nodes[self.input_key],
            self.degree,
            boundary_policy=self.boundary_policy,
        )
        return _replace_node_output(graph, self.output_key, values)


class CochainCodifferential(StrictModule):
    """GraphIR wrapper for a metric codifferential."""

    degree: int = eqx.field(static=True)
    input_key: str = eqx.field(static=True)
    output_key: str = eqx.field(static=True)
    boundary_policy: CochainBoundaryKind = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        input_key: str,
        output_key: str,
        boundary_policy: CochainBoundaryKind = "absolute",
    ):
        self.degree = int(degree)
        self.input_key = str(input_key)
        self.output_key = str(output_key)
        self.boundary_policy = CochainBoundaryPolicy(boundary_policy).kind

    def __call__(self, graph: GraphIR, /) -> GraphIR:
        if not isinstance(graph.nodes, Mapping) or self.input_key not in graph.nodes:
            raise KeyError(f"Missing cochain node field {self.input_key!r}.")
        values = cochain_codifferential(
            graph,
            graph.nodes[self.input_key],
            self.degree,
            boundary_policy=self.boundary_policy,
        )
        return _replace_node_output(graph, self.output_key, values)


class CochainHodgeLaplacian(StrictModule):
    """GraphIR wrapper for a split or complete metric Hodge Laplacian."""

    degree: int = eqx.field(static=True)
    input_key: str = eqx.field(static=True)
    output_key: str = eqx.field(static=True)
    component: HodgeLaplacianComponent = eqx.field(static=True)
    boundary_policy: CochainBoundaryKind = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        input_key: str,
        output_key: str,
        component: HodgeLaplacianComponent = "complete",
        boundary_policy: CochainBoundaryKind = "absolute",
    ):
        if component not in ("lower", "upper", "complete"):
            raise ValueError("Unknown Hodge Laplacian component.")
        self.degree = int(degree)
        self.input_key = str(input_key)
        self.output_key = str(output_key)
        self.component = component
        self.boundary_policy = CochainBoundaryPolicy(boundary_policy).kind

    def __call__(self, graph: GraphIR, /) -> GraphIR:
        if not isinstance(graph.nodes, Mapping) or self.input_key not in graph.nodes:
            raise KeyError(f"Missing cochain node field {self.input_key!r}.")
        values = cochain_hodge_laplacian(
            graph,
            graph.nodes[self.input_key],
            self.degree,
            component=self.component,
            boundary_policy=self.boundary_policy,
        )
        return _replace_node_output(graph, self.output_key, values)


class CochainHarmonicProjection(StrictModule):
    """GraphIR wrapper for exact metric harmonic projection."""

    degree: int = eqx.field(static=True)
    input_key: str = eqx.field(static=True)
    output_key: str = eqx.field(static=True)
    boundary_policy: CochainBoundaryKind = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        input_key: str,
        output_key: str,
        boundary_policy: CochainBoundaryKind = "absolute",
    ):
        self.degree = int(degree)
        self.input_key = str(input_key)
        self.output_key = str(output_key)
        self.boundary_policy = CochainBoundaryPolicy(boundary_policy).kind

    def __call__(self, graph: GraphIR, /) -> GraphIR:
        if not isinstance(graph.nodes, Mapping) or self.input_key not in graph.nodes:
            raise KeyError(f"Missing cochain node field {self.input_key!r}.")
        values = cochain_harmonic_projection(
            graph,
            graph.nodes[self.input_key],
            self.degree,
            boundary_policy=self.boundary_policy,
        )
        return _replace_node_output(graph, self.output_key, values)


__all__ = [
    "CochainCodifferential",
    "CochainExteriorDerivative",
    "CochainHarmonicProjection",
    "CochainHodgeLaplacian",
    "HodgeLaplacianComponent",
    "cochain_codifferential",
    "cochain_exterior_derivative",
    "cochain_harmonic_projection",
    "cochain_hodge_laplacian",
]
