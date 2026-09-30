#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Graph-coordinate execution of the prepared native cochain calculus."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import prod
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..discretization._cochain import CochainDiscretization
from ..exterior._complex import ComplexBoundary
from ..linalg import apply_real_map_componentwise
from ..linalg._complexes import HodgeLaplacianPart
from ..typing import parse
from ._cochain_execution import _cochain_metric_valid
from ._ir import GraphIR


def _cochain_payload(graph: GraphIR, /) -> Mapping[str, Any]:
    if not isinstance(graph, GraphIR):
        raise TypeError("Metric cochain operators require a GraphIR.")
    if not isinstance(graph.nodes, Mapping) or "cell_dim" not in graph.nodes:
        raise ValueError("Metric cochain operators require named cell metadata.")
    if not graph.cochain_bindings:
        raise ValueError("GraphIR has no prepared native cochain realization.")
    return graph.nodes


def _apply_native(
    graph: GraphIR,
    values: ArrayLike,
    source_degree: int,
    target_degree: int,
    action: Callable[[CochainDiscretization, Array], Array],
    /,
) -> Array:
    nodes = _cochain_payload(graph)
    array = jnp.asarray(values)
    count = jnp.asarray(nodes["cell_dim"]).shape[0]
    if array.ndim == 0 or array.shape[0] != count:
        raise ValueError(
            f"Cochain values require leading graph-node size {count}; got {array.shape}."
        )
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.float64)
    output = None
    for binding in graph.cochain_bindings:
        realization = binding.discretization
        realization._degree(source_degree)
        realization._degree(target_degree)
        source_start = binding.degree_start(source_degree)
        source_end = source_start + realization.cell_counts[source_degree]
        local = array[source_start:source_end]
        if graph.node_mask is not None:
            mask = graph.node_mask[source_start:source_end]
            local = jnp.where(
                mask.reshape(mask.shape + (1,) * (array.ndim - 1)), local, 0
            )
        if array.ndim == 1:
            result = action(realization, local)
        else:
            flat = local.reshape((local.shape[0], prod(array.shape[1:])))
            result = jax.vmap(
                lambda vector: action(realization, vector), in_axes=1, out_axes=1
            )(flat).reshape((realization.cell_counts[target_degree],) + array.shape[1:])
        if output is None:
            output = jnp.zeros(array.shape, dtype=result.dtype)
        if graph.graph_mask is not None:
            result = jnp.where(graph.graph_mask[binding.graph_index], result, 0)
        target_start = binding.degree_start(target_degree)
        target_end = target_start + realization.cell_counts[target_degree]
        output = output.at[target_start:target_end].set(result)
    assert output is not None
    if graph.node_mask is not None:
        output = jnp.where(
            graph.node_mask.reshape((count,) + (1,) * (array.ndim - 1)), output, 0
        )
    return eqx.error_if(
        output,
        ~_cochain_metric_valid(graph.cochain_bindings),
        "Native cochain pairing must remain finite and positive definite.",
    )


def cochain_exterior_derivative(
    graph: GraphIR,
    values: ArrayLike,
    degree: int,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> Array:
    """Apply the native ``d_degree`` in graph coordinates."""
    policy = parse(boundary, ComplexBoundary, "boundary")
    return _apply_native(
        graph,
        values,
        degree,
        degree + 1,
        lambda owner, local: owner.exterior_derivative(degree, local, boundary=policy),
    )


def cochain_codifferential(
    graph: GraphIR,
    values: ArrayLike,
    degree: int,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> Array:
    """Apply the native positive Hilbert adjoint, including full sparse Gram solves."""
    policy = parse(boundary, ComplexBoundary, "boundary")
    return _apply_native(
        graph,
        values,
        degree,
        degree - 1,
        lambda owner, local: owner.codifferential(degree, local, boundary=policy),
    )


def cochain_hodge_laplacian(
    graph: GraphIR,
    values: ArrayLike,
    degree: int,
    /,
    *,
    part: HodgeLaplacianPart = "complete",
    boundary: ComplexBoundary = "absolute",
) -> Array:
    """Apply the native degree-valid split or complete Hodge Laplacian."""
    policy = parse(boundary, ComplexBoundary, "boundary")
    part = parse(part, HodgeLaplacianPart, "part")
    return _apply_native(
        graph,
        values,
        degree,
        degree,
        lambda owner, local: owner.hodge_laplacian(
            degree, local, part=part, boundary=policy
        ),
    )


def _cochain_riesz(graph: GraphIR, values: ArrayLike, degree: int, /) -> Array:
    """Apply the full native metric, never its graph display diagonal."""
    return _apply_native(
        graph,
        values,
        degree,
        degree,
        lambda owner, local: owner.hodge_star(degree, local),
    )


def cochain_harmonic_projection(
    graph: GraphIR,
    values: ArrayLike,
    degree: int,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> Array:
    """Project with native harmonic evidence and the native full Gram pairing."""
    nodes = _cochain_payload(graph)
    policy = parse(boundary, ComplexBoundary, "boundary")
    array = jnp.asarray(values)
    count = jnp.asarray(nodes["cell_dim"]).shape[0]
    if array.ndim == 0 or array.shape[0] != count:
        raise ValueError("Harmonic values must align with graph nodes.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.float64)
    output = None
    for binding in graph.cochain_bindings:
        owner = binding.discretization
        owner._degree(degree)
        if binding.boundary != policy:
            raise ValueError("Harmonic subspace uses a different boundary policy.")
        harmonic = binding.harmonic[degree]
        if harmonic is None:
            raise ValueError(
                "GraphIR has no precomputed harmonic subspace at this degree."
            )
        start = binding.degree_start(degree)
        indices = owner.active_indices(degree, boundary=policy)
        local = array[start + indices]
        if graph.node_mask is not None:
            mask = graph.node_mask[start + indices]
            local = jnp.where(
                mask.reshape(mask.shape + (1,) * (array.ndim - 1)), local, 0
            )
        local = eqx.error_if(
            local, ~harmonic.valid, "Native harmonic evidence is invalid."
        )
        space = owner.hilbert_complex(boundary=policy).space(degree)

        def project(vector: Array) -> Array:
            return apply_real_map_componentwise(
                lambda value: harmonic.project(space, value), vector
            )

        if array.ndim == 1:
            projected = project(local)
        else:
            projected = jax.vmap(project, in_axes=1, out_axes=1)(
                local.reshape((local.shape[0], prod(array.shape[1:])))
            ).reshape(local.shape)
        if output is None:
            output = jnp.zeros(array.shape, dtype=projected.dtype)
        if graph.graph_mask is not None:
            projected = jnp.where(graph.graph_mask[binding.graph_index], projected, 0)
        output = output.at[start + indices].set(projected)
    assert output is not None
    return eqx.error_if(
        output,
        ~_cochain_metric_valid(graph.cochain_bindings),
        "Native cochain pairing must remain finite and positive definite.",
    )


def _replace_node_output(
    graph: GraphIR,
    output_key: str,
    values: Array,
    /,
) -> GraphIR:
    if not isinstance(graph.nodes, Mapping):
        raise ValueError("Cochain graph nodes must be a mapping.")
    return graph.replace(nodes={**graph.nodes, output_key: values}, validate=False)


@final
class CochainExteriorDerivative(StrictModule):
    """GraphIR wrapper for a metric exterior derivative."""

    degree: int = eqx.field(static=True)
    input_key: str = eqx.field(static=True)
    output_key: str = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        input_key: str,
        output_key: str,
        boundary: ComplexBoundary = "absolute",
    ) -> None:
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        self.degree = int(degree)
        self.input_key = str(input_key)
        self.output_key = str(output_key)
        self.boundary = boundary_

    def __call__(self, graph: GraphIR, /) -> GraphIR:
        if not isinstance(graph.nodes, Mapping) or self.input_key not in graph.nodes:
            raise KeyError(f"Missing cochain node field {self.input_key!r}.")
        values = cochain_exterior_derivative(
            graph,
            graph.nodes[self.input_key],
            self.degree,
            boundary=self.boundary,
        )
        return _replace_node_output(graph, self.output_key, values)


@final
class CochainCodifferential(StrictModule):
    """GraphIR wrapper for a metric codifferential."""

    degree: int = eqx.field(static=True)
    input_key: str = eqx.field(static=True)
    output_key: str = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        input_key: str,
        output_key: str,
        boundary: ComplexBoundary = "absolute",
    ) -> None:
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        self.degree = int(degree)
        self.input_key = str(input_key)
        self.output_key = str(output_key)
        self.boundary = boundary_

    def __call__(self, graph: GraphIR, /) -> GraphIR:
        if not isinstance(graph.nodes, Mapping) or self.input_key not in graph.nodes:
            raise KeyError(f"Missing cochain node field {self.input_key!r}.")
        values = cochain_codifferential(
            graph,
            graph.nodes[self.input_key],
            self.degree,
            boundary=self.boundary,
        )
        return _replace_node_output(graph, self.output_key, values)


@final
class CochainHodgeLaplacian(StrictModule):
    """GraphIR wrapper for a split or complete metric Hodge Laplacian."""

    degree: int = eqx.field(static=True)
    input_key: str = eqx.field(static=True)
    output_key: str = eqx.field(static=True)
    part: HodgeLaplacianPart = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        input_key: str,
        output_key: str,
        part: HodgeLaplacianPart = "complete",
        boundary: ComplexBoundary = "absolute",
    ) -> None:
        part = parse(part, HodgeLaplacianPart, "part")
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        self.degree = int(degree)
        self.input_key = str(input_key)
        self.output_key = str(output_key)
        self.part = part
        self.boundary = boundary_

    def __call__(self, graph: GraphIR, /) -> GraphIR:
        if not isinstance(graph.nodes, Mapping) or self.input_key not in graph.nodes:
            raise KeyError(f"Missing cochain node field {self.input_key!r}.")
        values = cochain_hodge_laplacian(
            graph,
            graph.nodes[self.input_key],
            self.degree,
            part=self.part,
            boundary=self.boundary,
        )
        return _replace_node_output(graph, self.output_key, values)


@final
class CochainHarmonicProjection(StrictModule):
    """GraphIR wrapper for exact metric harmonic projection."""

    degree: int = eqx.field(static=True)
    input_key: str = eqx.field(static=True)
    output_key: str = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        input_key: str,
        output_key: str,
        boundary: ComplexBoundary = "absolute",
    ) -> None:
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        self.degree = int(degree)
        self.input_key = str(input_key)
        self.output_key = str(output_key)
        self.boundary = boundary_

    def __call__(self, graph: GraphIR, /) -> GraphIR:
        if not isinstance(graph.nodes, Mapping) or self.input_key not in graph.nodes:
            raise KeyError(f"Missing cochain node field {self.input_key!r}.")
        values = cochain_harmonic_projection(
            graph,
            graph.nodes[self.input_key],
            self.degree,
            boundary=self.boundary,
        )
        return _replace_node_output(graph, self.output_key, values)


__all__ = [
    "CochainCodifferential",
    "CochainExteriorDerivative",
    "CochainHarmonicProjection",
    "CochainHodgeLaplacian",
    "cochain_codifferential",
    "cochain_exterior_derivative",
    "cochain_harmonic_projection",
    "cochain_hodge_laplacian",
]
