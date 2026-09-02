#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._interpolation import (
    bspline_cross_gram,
    bspline_mass_matrix,
    BSplineGrid,
    BSplineGridBank,
    BSplineGridTransfer,
    TrainableBSplineGrid,
    TrainableBSplineGridBank,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._utils import _get_size
from ._kan import KAN, KANEdgeBlock, KANLayer
from ._kan_basis import (
    BSplineEdgeBasis,
    RationalBSplineEdgeBasis,
    RationalBSplineEdgeParameters,
)


EdgePath = tuple[int, int, int]
CapacityOperation = Literal["refine", "coarsen"]
BasisKind = Literal["polynomial", "rational"]
_Grid = BSplineGrid | TrainableBSplineGrid


class KANCapacityAdaptationReport(StrictModule, NonTrainableState):
    """Evidence for one explicit, nondifferentiable topology transition."""

    operation: CapacityOperation = eqx.field(static=True)
    paths: tuple[EdgePath, ...]
    old_coefficient_counts: tuple[int, ...]
    new_coefficient_counts: tuple[int, ...]
    transfer_conditioning: tuple[float, ...]
    projection_error_bounds: tuple[float, ...]
    basis_kinds: tuple[BasisKind, ...] = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    rejected_candidates: tuple[tuple[EdgePath, str], ...] = eqx.field(static=True)
    numerator_projection_bounds: tuple[float, ...]
    denominator_projection_bounds: tuple[float, ...]
    denominator_lower_bounds: tuple[float, ...]
    differentiability_certified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        operation: CapacityOperation,
        paths: tuple[EdgePath, ...],
        old_coefficient_counts: tuple[int, ...],
        new_coefficient_counts: tuple[int, ...],
        transfer_conditioning: tuple[float, ...],
        projection_error_bounds: tuple[float, ...],
        basis_kinds: tuple[BasisKind, ...] = (),
        source_topology_id: str = "",
        target_topology_id: str = "",
        rejected_candidates: tuple[tuple[EdgePath, str], ...] = (),
        numerator_projection_bounds: tuple[float, ...] = (),
        denominator_projection_bounds: tuple[float, ...] = (),
        denominator_lower_bounds: tuple[float, ...] = (),
    ):
        self.operation = operation
        self.paths = paths
        self.old_coefficient_counts = old_coefficient_counts
        self.new_coefficient_counts = new_coefficient_counts
        self.transfer_conditioning = transfer_conditioning
        self.projection_error_bounds = projection_error_bounds
        self.basis_kinds = basis_kinds
        self.source_topology_id = source_topology_id
        self.target_topology_id = target_topology_id
        self.rejected_candidates = rejected_candidates
        self.numerator_projection_bounds = numerator_projection_bounds
        self.denominator_projection_bounds = denominator_projection_bounds
        self.denominator_lower_bounds = denominator_lower_bounds
        self.differentiability_certified = False


@dataclass(frozen=True, slots=True)
class _SplineEdge:
    output_index: int
    input_index: int
    basis: BSplineEdgeBasis | RationalBSplineEdgeBasis
    grid: _Grid
    coefficients: Array | RationalBSplineEdgeParameters

    @property
    def fixed_grid(self) -> BSplineGrid:
        if isinstance(self.grid, BSplineGrid):
            return self.grid
        return BSplineGrid(self.grid.knots, self.grid.degree)

    @property
    def basis_kind(self) -> BasisKind:
        return (
            "rational"
            if isinstance(self.basis, RationalBSplineEdgeBasis)
            else "polynomial"
        )


@dataclass(frozen=True, slots=True)
class _EdgeTransfer:
    grid: _Grid
    coefficients: Array | RationalBSplineEdgeParameters
    condition: float
    error_bound: float
    numerator_error: float
    denominator_error: float
    denominator_lower_bound: float


def _single_grids(
    grid: BSplineGrid | BSplineGridBank | TrainableBSplineGrid | TrainableBSplineGridBank,
    count: int,
    /,
) -> tuple[_Grid, ...]:
    if isinstance(grid, (BSplineGrid, TrainableBSplineGrid)):
        return (grid,) * count
    if isinstance(grid, BSplineGridBank):
        if grid.num_grids != count:
            raise ValueError("B-spline grid-bank size must match the KAN input size.")
        return grid.grids
    if grid.num_grids != count:
        raise ValueError(
            "Trainable B-spline grid-bank size must match the KAN input size."
        )
    return tuple(
        TrainableBSplineGrid(
            grid.raw_span_logits[index],
            grid.degree,
            interval=(
                float(np.asarray(grid.intervals[index, 0])),
                float(np.asarray(grid.intervals[index, 1])),
            ),
            minimum_span=float(np.asarray(grid.minimum_spans[index])),
        )
        for index in range(grid.num_grids)
    )


def _edge_parameters(parameters: Any, first: int, second: int, /):
    if isinstance(parameters, RationalBSplineEdgeParameters):
        return RationalBSplineEdgeParameters(
            parameters.control_values[first, second][None, None, :],
            parameters.raw_log_weights[first, second][None, None, :],
        )
    if not eqx.is_array(parameters):
        raise TypeError(
            "Spline KAN edge parameters must be arrays or rational parameters."
        )
    return parameters[first, second]


def _layer_spline_edges(layer: KANLayer, layer_index: int) -> tuple[_SplineEdge, ...]:
    edges: list[_SplineEdge] = []
    supported = (BSplineEdgeBasis, RationalBSplineEdgeBasis)
    if layer.edge_blocks:
        for block in layer.edge_blocks:
            if not isinstance(block.edge_basis, supported):
                raise TypeError(f"KAN layer {layer_index} contains a non-spline block.")
            grids = _single_grids(block.edge_basis.grid, 1)
            for position, (output_index, input_index) in enumerate(
                zip(block.output_indices, block.input_indices, strict=True)
            ):
                edges.append(
                    _SplineEdge(
                        output_index,
                        input_index,
                        block.edge_basis,
                        grids[0],
                        _edge_parameters(block.coeffs, position, 0),
                    )
                )
        return tuple(edges)

    if not isinstance(layer.edge_basis, supported) or layer.coeffs is None:
        raise TypeError(f"KAN layer {layer_index} is not a B-spline layer.")
    basis = layer.edge_basis
    input_count = _get_size(layer.in_size)
    grids = _single_grids(basis.grid, input_count)
    for output_index in range(_get_size(layer.out_size)):
        for input_index, grid in enumerate(grids):
            edges.append(
                _SplineEdge(
                    output_index,
                    input_index,
                    basis,
                    grid,
                    _edge_parameters(layer.coeffs, output_index, input_index),
                )
            )
    return tuple(edges)


def _edge_map(edges: tuple[_SplineEdge, ...]) -> dict[tuple[int, int], _SplineEdge]:
    return {(edge.output_index, edge.input_index): edge for edge in edges}


def _basis_key(edge: _SplineEdge) -> tuple[Any, ...]:
    basis = edge.basis
    common = (
        edge.basis_kind,
        edge.grid.degree,
        basis.regularization_order,
        type(edge.grid).__name__,
        tuple(float(value) for value in np.asarray(edge.grid.knots)),
    )
    if isinstance(basis, BSplineEdgeBasis):
        return common + (basis.knot_entropy_weight, basis.knot_neighbor_weight)
    return common + (
        basis.maximum_log_weight,
        basis.weight_magnitude_weight,
        basis.weight_variation_weight,
        basis.minimum_denominator,
        basis.denominator_weight,
    )


def _basis_with_grid(edge: _SplineEdge):
    basis = edge.basis
    if isinstance(basis, BSplineEdgeBasis):
        return BSplineEdgeBasis(
            grid=edge.grid,
            regularization_order=basis.regularization_order,
            knot_entropy_weight=basis.knot_entropy_weight,
            knot_neighbor_weight=basis.knot_neighbor_weight,
        )
    return RationalBSplineEdgeBasis(
        grid=edge.grid,
        regularization_order=basis.regularization_order,
        maximum_log_weight=basis.maximum_log_weight,
        weight_magnitude_weight=basis.weight_magnitude_weight,
        weight_variation_weight=basis.weight_variation_weight,
        minimum_denominator=basis.minimum_denominator,
        denominator_weight=basis.denominator_weight,
    )


def _block_layer(layer: KANLayer, edges: tuple[_SplineEdge, ...]) -> KANLayer:
    groups: dict[tuple[Any, ...], list[_SplineEdge]] = defaultdict(list)
    for edge in sorted(edges, key=lambda item: (item.output_index, item.input_index)):
        groups[_basis_key(edge)].append(edge)
    blocks: list[KANEdgeBlock] = []
    for grouped in groups.values():
        exemplar = grouped[0]
        if exemplar.basis_kind == "polynomial":
            coefficients: Any = jnp.stack(
                tuple(jnp.asarray(edge.coefficients) for edge in grouped)
            )[:, None, :]
        else:
            rational = tuple(
                edge.coefficients
                for edge in grouped
                if isinstance(edge.coefficients, RationalBSplineEdgeParameters)
            )
            if len(rational) != len(grouped):
                raise TypeError("Rational edge group contains polynomial coefficients.")
            coefficients = RationalBSplineEdgeParameters(
                jnp.concatenate(
                    tuple(value.control_values for value in rational), axis=0
                ),
                jnp.concatenate(
                    tuple(value.raw_log_weights for value in rational), axis=0
                ),
            )
        blocks.append(
            KANEdgeBlock(
                output_indices=tuple(edge.output_index for edge in grouped),
                input_indices=tuple(edge.input_index for edge in grouped),
                edge_basis=_basis_with_grid(exemplar),
                coeffs=coefficients,
            )
        )
    return eqx.tree_at(
        lambda current: (current.edge_basis, current.coeffs, current.edge_blocks),
        layer,
        (None, None, tuple(blocks)),
        is_leaf=lambda value: value is None,
    )


def _topology_id(model: KAN, /) -> str:
    digest = sha256()
    for layer_index, layer in enumerate(model.layers):
        digest.update(str(layer_index).encode())
        for edge in _layer_spline_edges(layer, layer_index):
            digest.update(
                str((edge.output_index, edge.input_index, edge.basis_kind)).encode()
            )
            digest.update(np.asarray(edge.grid.knots).tobytes())
    return digest.hexdigest()[:24]


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


def _target_grid(edge: _SplineEdge, fixed: BSplineGrid, /) -> _Grid | None:
    if isinstance(edge.grid, BSplineGrid):
        return fixed
    minimum = edge.grid.minimum_span
    widths = np.diff(np.asarray(fixed.breakpoints))
    if np.any(widths <= minimum):
        return None
    return TrainableBSplineGrid.from_grid(fixed, minimum_span=minimum)


def _rational_homogeneous(edge: _SplineEdge) -> tuple[Array, Array]:
    if not isinstance(edge.basis, RationalBSplineEdgeBasis) or not isinstance(
        edge.coefficients, RationalBSplineEdgeParameters
    ):
        raise TypeError("Rational homogeneous transfer requires rational edge data.")
    controls = edge.coefficients.control_values[0, 0]
    raw = edge.coefficients.raw_log_weights[0, 0]
    weights, _ = edge.basis._positive_weights(raw)
    return controls * weights, weights


def _rational_parameters(
    basis: RationalBSplineEdgeBasis,
    numerator: Array,
    denominator: Array,
    /,
) -> RationalBSplineEdgeParameters | None:
    denominator_host = np.asarray(denominator, dtype=float)
    if not np.all(np.isfinite(denominator_host)) or np.any(
        denominator_host <= basis.minimum_denominator
    ):
        return None
    controls = numerator / denominator
    log_weights = jnp.log(denominator)
    log_weights = log_weights - jnp.mean(log_weights)
    limit = basis.maximum_log_weight
    if np.max(np.abs(np.asarray(log_weights))) >= limit:
        return None
    raw = jnp.arctanh(log_weights / limit)
    return RationalBSplineEdgeParameters(
        controls[None, None, :],
        raw[None, None, :],
    )


def _refine_edge(
    edge: _SplineEdge, spans: set[int], /
) -> tuple[_EdgeTransfer | None, str]:
    old_grid = edge.fixed_grid
    breakpoints = np.asarray(old_grid.breakpoints)
    inserted = jnp.asarray(
        [0.5 * (breakpoints[index] + breakpoints[index + 1]) for index in sorted(spans)],
        dtype=old_grid.knots.dtype,
    )
    fixed = BSplineGrid(
        jnp.sort(jnp.concatenate((old_grid.knots, inserted))), old_grid.degree
    )
    target = _target_grid(edge, fixed)
    if target is None:
        return None, "inserted spans violate the trainable grid minimum span"
    transfer = BSplineGridTransfer(old_grid, fixed)
    if edge.basis_kind == "polynomial":
        coefficients = transfer(jnp.asarray(edge.coefficients))
        norm = float(np.linalg.norm(np.asarray(edge.coefficients)))
        return (
            _EdgeTransfer(
                target,
                coefficients,
                transfer.condition_estimate,
                transfer.projection_error_bound * norm,
                transfer.projection_error_bound * norm,
                0.0,
                float("inf"),
            ),
            "accepted exact polynomial insertion",
        )
    numerator, denominator = _rational_homogeneous(edge)
    new_numerator = transfer(numerator)
    new_denominator = transfer(denominator)
    parameters = _rational_parameters(edge.basis, new_numerator, new_denominator)
    if parameters is None:
        return (
            None,
            "transferred rational weights violate positivity or bounded-log representation",
        )
    return (
        _EdgeTransfer(
            target,
            parameters,
            transfer.condition_estimate,
            0.0,
            0.0,
            0.0,
            float(np.min(np.asarray(new_denominator))),
        ),
        "accepted exact homogeneous insertion",
    )


def refine_kan_edges(
    model: KAN,
    span_indicators: Mapping[EdgePath, ArrayLike],
    /,
    *,
    budget: int,
) -> tuple[KAN, KANCapacityAdaptationReport]:
    """Insert selected knots between optimizer epochs with exact spline transfer."""
    if not isinstance(model, KAN):
        raise TypeError("refine_kan_edges expects a KAN model.")
    if not isinstance(span_indicators, Mapping):
        raise TypeError("span_indicators must map edge paths to per-span scores.")
    budget_ = _validate_budget(budget)
    source_id = _topology_id(model)
    layer_edges: dict[int, tuple[_SplineEdge, ...]] = {}
    candidates: list[tuple[float, EdgePath, int]] = []
    for raw_path, indicator in span_indicators.items():
        if not isinstance(raw_path, tuple) or len(raw_path) != 3:
            raise ValueError("Every refinement path must be (layer, output, input).")
        path = tuple(int(value) for value in raw_path)
        layer_index, output_index, input_index = path
        if not 0 <= layer_index < len(model.layers):
            raise ValueError(f"Unknown KAN refinement layer: {layer_index}.")
        if layer_index not in layer_edges:
            layer_edges[layer_index] = _layer_spline_edges(
                model.layers[layer_index], layer_index
            )
        edge = _edge_map(layer_edges[layer_index]).get((output_index, input_index))
        if edge is None:
            raise ValueError(f"Unknown KAN refinement edge: {path!r}.")
        scores = np.asarray(indicator, dtype=float)
        if scores.ndim != 1 or scores.size != edge.fixed_grid.num_intervals:
            raise ValueError(
                f"Refinement scores for {path!r} must contain one value per span."
            )
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
            raise ValueError("KAN refinement scores must be finite and nonnegative.")
        candidates.extend(
            (float(score), path, span_index)
            for span_index, score in enumerate(scores)
            if score > 0.0
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    insertions: dict[EdgePath, set[int]] = defaultdict(set)
    for _, path, span_index in candidates[:budget_]:
        insertions[path].add(span_index)

    selected: dict[EdgePath, _EdgeTransfer] = {}
    rejected: list[tuple[EdgePath, str]] = []
    for path, spans in insertions.items():
        edge = _edge_map(layer_edges[path[0]])[(path[1], path[2])]
        transfer, reason = _refine_edge(edge, spans)
        if transfer is None:
            rejected.append((path, reason))
        else:
            selected[path] = transfer

    layers = list(model.layers)
    report_edges: list[_SplineEdge] = []
    report_transfers: list[_EdgeTransfer] = []
    report_paths: list[EdgePath] = []
    for layer_index, original in layer_edges.items():
        updated: list[_SplineEdge] = []
        for edge in original:
            path = (layer_index, edge.output_index, edge.input_index)
            transfer = selected.get(path)
            if transfer is None:
                updated.append(edge)
                continue
            updated_edge = _SplineEdge(
                edge.output_index,
                edge.input_index,
                edge.basis,
                transfer.grid,
                transfer.coefficients,
            )
            updated.append(updated_edge)
            report_edges.append(edge)
            report_transfers.append(transfer)
            report_paths.append(path)
        layers[layer_index] = _block_layer(layers[layer_index], tuple(updated))
    adapted = model._replace_layers(tuple(layers)) if report_edges else model
    target_id = _topology_id(adapted)
    return adapted, KANCapacityAdaptationReport(
        operation="refine",
        paths=tuple(report_paths),
        old_coefficient_counts=tuple(
            edge.fixed_grid.coefficient_count for edge in report_edges
        ),
        new_coefficient_counts=tuple(
            value.grid.coefficient_count for value in report_transfers
        ),
        transfer_conditioning=tuple(value.condition for value in report_transfers),
        projection_error_bounds=tuple(value.error_bound for value in report_transfers),
        basis_kinds=tuple(edge.basis_kind for edge in report_edges),
        source_topology_id=source_id,
        target_topology_id=target_id,
        rejected_candidates=tuple(rejected),
        numerator_projection_bounds=tuple(
            value.numerator_error for value in report_transfers
        ),
        denominator_projection_bounds=tuple(
            value.denominator_error for value in report_transfers
        ),
        denominator_lower_bounds=tuple(
            value.denominator_lower_bound for value in report_transfers
        ),
    )


def _coarsening_transfer(
    edge: _SplineEdge, knot_position: int, /
) -> _EdgeTransfer | None:
    old_grid = edge.fixed_grid
    fixed = BSplineGrid(jnp.delete(old_grid.knots, knot_position), old_grid.degree)
    target = _target_grid(edge, fixed)
    if target is None:
        return None
    transfer = BSplineGridTransfer(old_grid, fixed)
    if edge.basis_kind == "polynomial":
        coefficients = jnp.asarray(edge.coefficients)
        projected = transfer(coefficients)
        error = _coefficient_projection_error(old_grid, fixed, transfer, coefficients)
        return _EdgeTransfer(
            target,
            projected,
            transfer.condition_estimate,
            error,
            error,
            0.0,
            float("inf"),
        )
    numerator, denominator = _rational_homogeneous(edge)
    projected_numerator = transfer(numerator)
    projected_denominator = transfer(denominator)
    parameters = _rational_parameters(
        edge.basis, projected_numerator, projected_denominator
    )
    if parameters is None:
        return None
    numerator_error = _coefficient_projection_error(old_grid, fixed, transfer, numerator)
    denominator_error = _coefficient_projection_error(
        old_grid, fixed, transfer, denominator
    )
    lower = float(np.min(np.asarray(projected_denominator)))
    numerator_bound = float(np.max(np.abs(np.asarray(projected_numerator))))
    quotient_bound = numerator_error / lower + numerator_bound * denominator_error / (
        lower * lower
    )
    return _EdgeTransfer(
        target,
        parameters,
        transfer.condition_estimate,
        quotient_bound,
        numerator_error,
        denominator_error,
        lower,
    )


def coarsen_kan_edges(
    model: KAN,
    tolerances: float | Mapping[EdgePath, float],
    /,
    *,
    budget: int,
) -> tuple[KAN, KANCapacityAdaptationReport]:
    """Remove knots only when the polynomial or rational certificate is accepted."""
    if not isinstance(model, KAN):
        raise TypeError("coarsen_kan_edges expects a KAN model.")
    budget_ = _validate_budget(budget)
    source_id = _topology_id(model)
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
    rejected: list[tuple[EdgePath, str]] = []
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
        edge = _edge_map(layer_edges[layer_index]).get((output_index, input_index))
        if edge is None:
            if isinstance(tolerances, Mapping):
                raise ValueError(f"Unknown KAN coarsening edge: {path!r}.")
            continue
        best: _EdgeTransfer | None = None
        for position in range(
            edge.fixed_grid.degree + 1, edge.fixed_grid.coefficient_count
        ):
            candidate = _coarsening_transfer(edge, position)
            if candidate is not None and (
                best is None or candidate.error_bound < best.error_bound
            ):
                best = candidate
        if best is None:
            rejected.append((path, "no representable positive reduced topology"))
        elif best.error_bound <= tolerance:
            candidate_transfers.append((best.error_bound, path, best))
        else:
            rejected.append(
                (path, "best conservative projection bound exceeds tolerance")
            )
    candidate_transfers.sort(key=lambda item: (item[0], item[1]))
    selected = candidate_transfers[:budget_]
    rejected.extend(
        (path, "global coarsening budget exhausted")
        for _, path, _ in candidate_transfers[budget_:]
    )
    selected_by_path = {path: transfer for _, path, transfer in selected}

    layers = list(model.layers)
    report_edges: list[_SplineEdge] = []
    report_transfers: list[_EdgeTransfer] = []
    report_paths = []
    for layer_index, original in layer_edges.items():
        updated: list[_SplineEdge] = []
        for edge in original:
            path = (layer_index, edge.output_index, edge.input_index)
            transfer = selected_by_path.get(path)
            if transfer is None:
                updated.append(edge)
                continue
            updated.append(
                _SplineEdge(
                    edge.output_index,
                    edge.input_index,
                    edge.basis,
                    transfer.grid,
                    transfer.coefficients,
                )
            )
            report_edges.append(edge)
            report_transfers.append(transfer)
            report_paths.append(path)
        layers[layer_index] = _block_layer(layers[layer_index], tuple(updated))
    adapted = model._replace_layers(tuple(layers)) if report_edges else model
    target_id = _topology_id(adapted)
    return adapted, KANCapacityAdaptationReport(
        operation="coarsen",
        paths=tuple(report_paths),
        old_coefficient_counts=tuple(
            edge.fixed_grid.coefficient_count for edge in report_edges
        ),
        new_coefficient_counts=tuple(
            value.grid.coefficient_count for value in report_transfers
        ),
        transfer_conditioning=tuple(value.condition for value in report_transfers),
        projection_error_bounds=tuple(value.error_bound for value in report_transfers),
        basis_kinds=tuple(edge.basis_kind for edge in report_edges),
        source_topology_id=source_id,
        target_topology_id=target_id,
        rejected_candidates=tuple(rejected),
        numerator_projection_bounds=tuple(
            value.numerator_error for value in report_transfers
        ),
        denominator_projection_bounds=tuple(
            value.denominator_error for value in report_transfers
        ),
        denominator_lower_bounds=tuple(
            value.denominator_lower_bound for value in report_transfers
        ),
    )


__all__ = [
    "BasisKind",
    "CapacityOperation",
    "EdgePath",
    "KANCapacityAdaptationReport",
    "coarsen_kan_edges",
    "refine_kan_edges",
]
