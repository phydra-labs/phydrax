#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._interpolation import (
    bspline_cross_gram,
    bspline_mass_matrix,
    BSplineGrid,
    BSplineGridBank,
    BSplineGridTransfer,
)
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ..._utils import _get_size
from ._kan import KAN, KANEdgeBlock, KANLayer
from ._kan_basis import BSplineEdgeBasis


EdgePath = tuple[int, int, int]
CapacityOperation = Literal["refine", "coarsen"]


class KANCapacityAdaptationReport(StrictModule, NonTrainableState):
    """Diagnostics for an explicit between-phase edge-capacity transformation."""

    operation: CapacityOperation = eqx.field(static=True)
    paths: tuple[EdgePath, ...]
    old_coefficient_counts: tuple[int, ...]
    new_coefficient_counts: tuple[int, ...]
    transfer_conditioning: tuple[float, ...]
    projection_error_bounds: tuple[float, ...]

    def __init__(
        self,
        *,
        operation: CapacityOperation,
        paths: tuple[EdgePath, ...],
        old_coefficient_counts: tuple[int, ...],
        new_coefficient_counts: tuple[int, ...],
        transfer_conditioning: tuple[float, ...],
        projection_error_bounds: tuple[float, ...],
    ):
        self.operation = operation
        self.paths = paths
        self.old_coefficient_counts = old_coefficient_counts
        self.new_coefficient_counts = new_coefficient_counts
        self.transfer_conditioning = transfer_conditioning
        self.projection_error_bounds = projection_error_bounds


@dataclass(frozen=True, slots=True)
class _SplineEdge:
    output_index: int
    input_index: int
    grid: BSplineGrid
    coefficients: Array
    regularization_order: int


@dataclass(frozen=True, slots=True)
class _EdgeTransfer:
    grid: BSplineGrid
    coefficients: Array
    condition: float
    error_bound: float


def _layer_spline_edges(layer: KANLayer, layer_index: int) -> tuple[_SplineEdge, ...]:
    edges: list[_SplineEdge] = []
    if layer.edge_blocks:
        for block in layer.edge_blocks:
            if (
                not isinstance(block.edge_basis, BSplineEdgeBasis)
                or not isinstance(block.edge_basis.grid, BSplineGrid)
                or not eqx.is_array(block.coeffs)
            ):
                raise TypeError(
                    f"KAN layer {layer_index} contains a non-polynomial spline block."
                )
            for position, (output_index, input_index) in enumerate(
                zip(block.output_indices, block.input_indices, strict=True)
            ):
                edges.append(
                    _SplineEdge(
                        output_index,
                        input_index,
                        block.edge_basis.grid,
                        block.coeffs[position, 0],
                        block.edge_basis.regularization_order,
                    )
                )
        return tuple(edges)

    if (
        not isinstance(layer.edge_basis, BSplineEdgeBasis)
        or layer.coeffs is None
        or not eqx.is_array(layer.coeffs)
    ):
        raise TypeError(f"KAN layer {layer_index} is not a polynomial B-spline layer.")
    basis = layer.edge_basis
    if isinstance(basis.grid, BSplineGrid):
        input_grids = (basis.grid,) * int(layer.coeffs.shape[1])
    elif isinstance(basis.grid, BSplineGridBank):
        input_grids = basis.grid.grids
    else:
        raise TypeError(
            f"KAN layer {layer_index} uses trainable knots and cannot change capacity."
        )
    for output_index in range(int(layer.coeffs.shape[0])):
        for input_index, grid in enumerate(input_grids):
            edges.append(
                _SplineEdge(
                    output_index,
                    input_index,
                    grid,
                    layer.coeffs[output_index, input_index],
                    basis.regularization_order,
                )
            )
    return tuple(edges)


def _edge_map(edges: tuple[_SplineEdge, ...]) -> dict[tuple[int, int], _SplineEdge]:
    return {(edge.output_index, edge.input_index): edge for edge in edges}


def _grid_key(edge: _SplineEdge) -> tuple[int, int, tuple[float, ...]]:
    return (
        edge.grid.degree,
        edge.regularization_order,
        tuple(float(value) for value in np.asarray(edge.grid.knots)),
    )


def _block_layer(layer: KANLayer, edges: tuple[_SplineEdge, ...]) -> KANLayer:
    groups: dict[tuple[int, int, tuple[float, ...]], list[_SplineEdge]] = defaultdict(
        list
    )
    for edge in sorted(edges, key=lambda item: (item.output_index, item.input_index)):
        groups[_grid_key(edge)].append(edge)
    blocks: list[KANEdgeBlock] = []
    for grouped in groups.values():
        exemplar = grouped[0]
        coefficients = jnp.stack(tuple(edge.coefficients for edge in grouped))[:, None, :]
        blocks.append(
            KANEdgeBlock(
                output_indices=tuple(edge.output_index for edge in grouped),
                input_indices=tuple(edge.input_index for edge in grouped),
                edge_basis=BSplineEdgeBasis(
                    grid=exemplar.grid,
                    regularization_order=exemplar.regularization_order,
                ),
                coeffs=coefficients,
            )
        )
    return eqx.tree_at(
        lambda current: (current.edge_basis, current.coeffs, current.edge_blocks),
        layer,
        (None, None, tuple(blocks)),
        is_leaf=lambda value: value is None,
    )


def _validate_budget(budget: int) -> int:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("KAN capacity budget must be a positive integer.")
    return budget


def _coefficient_projection_error(
    old_grid: BSplineGrid,
    new_grid: BSplineGrid,
    transfer: BSplineGridTransfer,
    coefficients: Array,
    /,
) -> float:
    cross_gram = bspline_cross_gram(old_grid, new_grid)
    residual_gram = bspline_mass_matrix(old_grid) - cross_gram.T @ transfer.matrix
    residual = jnp.real(jnp.conj(coefficients) @ residual_gram @ coefficients)
    return float(jnp.sqrt(jnp.maximum(residual, 0.0)))


def refine_kan_edges(
    model: KAN,
    span_indicators: Mapping[EdgePath, ArrayLike],
    /,
    *,
    budget: int,
) -> tuple[KAN, KANCapacityAdaptationReport]:
    """Insert high-scoring span midpoints and transfer affected edge functions exactly."""
    if not isinstance(model, KAN):
        raise TypeError("refine_kan_edges expects a KAN model.")
    if not isinstance(span_indicators, Mapping):
        raise TypeError("span_indicators must map edge paths to per-span scores.")
    budget_ = _validate_budget(budget)
    layer_edges: dict[int, tuple[_SplineEdge, ...]] = {}
    candidates: list[tuple[float, EdgePath, int]] = []
    for path, indicator in span_indicators.items():
        if not isinstance(path, tuple) or len(path) != 3:
            raise ValueError("Every refinement path must be (layer, output, input).")
        layer_index, output_index, input_index = (int(value) for value in path)
        if not 0 <= layer_index < len(model.layers):
            raise ValueError(f"Unknown KAN refinement layer: {layer_index}.")
        if layer_index not in layer_edges:
            layer_edges[layer_index] = _layer_spline_edges(
                model.layers[layer_index], layer_index
            )
        edges = _edge_map(layer_edges[layer_index])
        if (output_index, input_index) not in edges:
            raise ValueError(f"Unknown KAN refinement edge: {path!r}.")
        edge = edges[(output_index, input_index)]
        scores = np.asarray(indicator, dtype=float)
        if scores.ndim != 1 or scores.size != edge.grid.num_intervals:
            raise ValueError(
                f"Refinement scores for {path!r} must contain one value per positive span."
            )
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
            raise ValueError("KAN refinement scores must be finite and nonnegative.")
        for span_index, score in enumerate(scores):
            if score > 0.0:
                candidates.append((float(score), path, span_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = candidates[:budget_]
    insertions: dict[EdgePath, set[int]] = defaultdict(set)
    for _, path, span_index in selected:
        insertions[path].add(span_index)

    layers = list(model.layers)
    report_paths: list[EdgePath] = []
    old_counts: list[int] = []
    new_counts: list[int] = []
    conditions: list[float] = []
    errors: list[float] = []
    for layer_index, original_edges in layer_edges.items():
        updated_edges: list[_SplineEdge] = []
        for edge in original_edges:
            path = (layer_index, edge.output_index, edge.input_index)
            selected_spans = insertions.get(path)
            if not selected_spans:
                updated_edges.append(edge)
                continue
            breakpoints = np.asarray(edge.grid.breakpoints)
            new_knots = jnp.sort(
                jnp.concatenate(
                    (
                        edge.grid.knots,
                        jnp.asarray(
                            [
                                0.5 * (breakpoints[index] + breakpoints[index + 1])
                                for index in sorted(selected_spans)
                            ],
                            dtype=edge.grid.knots.dtype,
                        ),
                    )
                )
            )
            new_grid = BSplineGrid(new_knots, edge.grid.degree)
            transfer = BSplineGridTransfer(edge.grid, new_grid)
            new_coefficients = transfer(edge.coefficients)
            coefficient_norm = float(np.linalg.norm(np.asarray(edge.coefficients)))
            updated_edges.append(
                _SplineEdge(
                    edge.output_index,
                    edge.input_index,
                    new_grid,
                    new_coefficients,
                    edge.regularization_order,
                )
            )
            report_paths.append(path)
            old_counts.append(edge.grid.coefficient_count)
            new_counts.append(new_grid.coefficient_count)
            conditions.append(transfer.condition_estimate)
            errors.append(transfer.projection_error_bound * coefficient_norm)
        layers[layer_index] = _block_layer(layers[layer_index], tuple(updated_edges))
    adapted = model._replace_layers(tuple(layers)) if report_paths else model
    return adapted, KANCapacityAdaptationReport(
        operation="refine",
        paths=tuple(report_paths),
        old_coefficient_counts=tuple(old_counts),
        new_coefficient_counts=tuple(new_counts),
        transfer_conditioning=tuple(conditions),
        projection_error_bounds=tuple(errors),
    )


def coarsen_kan_edges(
    model: KAN,
    tolerances: float | Mapping[EdgePath, float],
    /,
    *,
    budget: int,
) -> tuple[KAN, KANCapacityAdaptationReport]:
    """Remove at most one knot per selected edge when projection error is certified."""
    if not isinstance(model, KAN):
        raise TypeError("coarsen_kan_edges expects a KAN model.")
    budget_ = _validate_budget(budget)
    if isinstance(tolerances, Mapping):
        requested = {
            tuple(int(value) for value in path): float(limit)
            for path, limit in tolerances.items()
        }
    else:
        limit = float(tolerances)
        requested = {
            (layer_index, output_index, input_index): limit
            for layer_index, layer in enumerate(model.layers)
            for output_index in range(_get_size(layer.out_size))
            for input_index in range(_get_size(layer.in_size))
        }
    if any(not isfinite(limit) or limit < 0.0 for limit in requested.values()):
        raise ValueError("KAN coarsening tolerances must be finite and nonnegative.")

    layer_edges: dict[int, tuple[_SplineEdge, ...]] = {}
    candidate_transfers: list[tuple[float, EdgePath, _EdgeTransfer]] = []
    for path, tolerance in requested.items():
        if len(path) != 3:
            raise ValueError("Every coarsening path must be (layer, output, input).")
        layer_index, output_index, input_index = path
        if not 0 <= layer_index < len(model.layers):
            if isinstance(tolerances, Mapping):
                raise ValueError(f"Unknown KAN coarsening layer: {layer_index}.")
            continue
        if layer_index not in layer_edges:
            layer_edges[layer_index] = _layer_spline_edges(
                model.layers[layer_index], layer_index
            )
        edges = _edge_map(layer_edges[layer_index])
        if (output_index, input_index) not in edges:
            if isinstance(tolerances, Mapping):
                raise ValueError(f"Unknown KAN coarsening edge: {path!r}.")
            continue
        edge = edges[(output_index, input_index)]
        interior_positions = range(edge.grid.degree + 1, edge.grid.coefficient_count)
        best: tuple[float, _EdgeTransfer] | None = None
        for knot_position in interior_positions:
            new_grid = BSplineGrid(
                jnp.delete(edge.grid.knots, knot_position),
                edge.grid.degree,
            )
            transfer = BSplineGridTransfer(edge.grid, new_grid)
            new_coefficients = transfer(edge.coefficients)
            error_bound = _coefficient_projection_error(
                edge.grid,
                new_grid,
                transfer,
                edge.coefficients,
            )
            candidate = _EdgeTransfer(
                new_grid,
                new_coefficients,
                transfer.condition_estimate,
                error_bound,
            )
            if best is None or error_bound < best[0]:
                best = (error_bound, candidate)
        if best is not None and best[0] <= tolerance:
            candidate_transfers.append((best[0], path, best[1]))
    candidate_transfers.sort(key=lambda item: (item[0], item[1]))
    selected = candidate_transfers[:budget_]
    selected_by_path = {path: transfer for _, path, transfer in selected}

    layers = list(model.layers)
    report_paths: list[EdgePath] = []
    old_counts: list[int] = []
    new_counts: list[int] = []
    conditions: list[float] = []
    errors: list[float] = []
    for layer_index, original_edges in layer_edges.items():
        updated_edges: list[_SplineEdge] = []
        for edge in original_edges:
            path = (layer_index, edge.output_index, edge.input_index)
            transfer = selected_by_path.get(path)
            if transfer is None:
                updated_edges.append(edge)
                continue
            updated_edges.append(
                _SplineEdge(
                    edge.output_index,
                    edge.input_index,
                    transfer.grid,
                    transfer.coefficients,
                    edge.regularization_order,
                )
            )
            report_paths.append(path)
            old_counts.append(edge.grid.coefficient_count)
            new_counts.append(transfer.grid.coefficient_count)
            conditions.append(transfer.condition)
            errors.append(transfer.error_bound)
        layers[layer_index] = _block_layer(layers[layer_index], tuple(updated_edges))
    adapted = model._replace_layers(tuple(layers)) if report_paths else model
    return adapted, KANCapacityAdaptationReport(
        operation="coarsen",
        paths=tuple(report_paths),
        old_coefficient_counts=tuple(old_counts),
        new_coefficient_counts=tuple(new_counts),
        transfer_conditioning=tuple(conditions),
        projection_error_bounds=tuple(errors),
    )


__all__ = [
    "CapacityOperation",
    "EdgePath",
    "KANCapacityAdaptationReport",
    "coarsen_kan_edges",
    "refine_kan_edges",
]
