#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._interpolation import (
    BSplineGrid,
    BSplineGridBank,
    BSplineGridTransfer,
    TrainableBSplineGrid,
)
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ..._utils import _get_size
from ..wrappers._separable_wrappers import Separable
from ._kan import KAN, KANLayer
from ._kan_basis import BSplineEdgeBasis
from ._separable_kan import SeparableKAN


QuantileMethod = Literal["linear", "nearest", "midpoint"]
DegenerateGridPolicy = Literal["retain", "uniform"]


class KANGridAdaptationPlan(StrictModule, NonTrainableState):
    """Policy for pure, fixed-count quantile adaptation of B-spline KAN grids."""

    blend: float
    minimum_span: float
    quantile_method: QuantileMethod
    degenerate_policy: DegenerateGridPolicy
    per_input: bool

    def __init__(
        self,
        *,
        blend: float = 0.1,
        minimum_span: float = 1.0e-3,
        quantile_method: QuantileMethod = "linear",
        degenerate_policy: DegenerateGridPolicy = "retain",
        per_input: bool = False,
    ):
        blend_ = float(blend)
        minimum_span_ = float(minimum_span)
        if not isfinite(blend_) or not 0.0 <= blend_ <= 1.0:
            raise ValueError("adaptation blend must lie between zero and one.")
        if not isfinite(minimum_span_) or minimum_span_ <= 0.0:
            raise ValueError("adaptation minimum_span must be finite and positive.")
        if quantile_method not in ("linear", "nearest", "midpoint"):
            raise ValueError(f"Unknown quantile method: {quantile_method!r}.")
        if degenerate_policy not in ("retain", "uniform"):
            raise ValueError(f"Unknown degenerate-grid policy: {degenerate_policy!r}.")
        self.blend = blend_
        self.minimum_span = minimum_span_
        self.quantile_method = quantile_method
        self.degenerate_policy = degenerate_policy
        self.per_input = bool(per_input)


class KANGridAdaptationReport(StrictModule, NonTrainableState):
    """Numerical diagnostics for a completed KAN grid transformation."""

    paths: tuple[tuple[int, int], ...]
    input_indices: tuple[int | None, ...]
    old_grids: tuple[BSplineGrid, ...]
    new_grids: tuple[BSplineGrid, ...]
    transfer_conditioning: tuple[float, ...]
    projection_error_bounds: tuple[float, ...]
    activation_counts: tuple[int, ...]
    skipped_paths: tuple[tuple[int, int], ...]
    degenerate_paths: tuple[tuple[int, int], ...]
    degenerate_grid_paths: tuple[tuple[int, int, int], ...]

    def __init__(
        self,
        *,
        paths: tuple[tuple[int, int], ...],
        input_indices: tuple[int | None, ...],
        old_grids: tuple[BSplineGrid, ...],
        new_grids: tuple[BSplineGrid, ...],
        transfer_conditioning: tuple[float, ...],
        projection_error_bounds: tuple[float, ...],
        activation_counts: tuple[int, ...],
        skipped_paths: tuple[tuple[int, int], ...],
        degenerate_paths: tuple[tuple[int, int], ...],
        degenerate_grid_paths: tuple[tuple[int, int, int], ...],
    ):
        self.paths = paths
        self.input_indices = input_indices
        self.old_grids = old_grids
        self.new_grids = new_grids
        self.transfer_conditioning = transfer_conditioning
        self.projection_error_bounds = projection_error_bounds
        self.activation_counts = activation_counts
        self.skipped_paths = skipped_paths
        self.degenerate_paths = degenerate_paths
        self.degenerate_grid_paths = degenerate_grid_paths


class _AdaptationRecords:
    def __init__(self):
        self.paths: list[tuple[int, int]] = []
        self.input_indices: list[int | None] = []
        self.old_grids: list[BSplineGrid] = []
        self.new_grids: list[BSplineGrid] = []
        self.transfer_conditioning: list[float] = []
        self.projection_error_bounds: list[float] = []
        self.activation_counts: list[int] = []
        self.skipped_paths: list[tuple[int, int]] = []
        self.degenerate_paths: list[tuple[int, int]] = []
        self.degenerate_grid_paths: list[tuple[int, int, int]] = []

    def report(self) -> KANGridAdaptationReport:
        return KANGridAdaptationReport(
            paths=tuple(self.paths),
            input_indices=tuple(self.input_indices),
            old_grids=tuple(self.old_grids),
            new_grids=tuple(self.new_grids),
            transfer_conditioning=tuple(self.transfer_conditioning),
            projection_error_bounds=tuple(self.projection_error_bounds),
            activation_counts=tuple(self.activation_counts),
            skipped_paths=tuple(self.skipped_paths),
            degenerate_paths=tuple(self.degenerate_paths),
            degenerate_grid_paths=tuple(self.degenerate_grid_paths),
        )


def _calibration_batch(model: KAN, calibration_inputs: ArrayLike) -> Array:
    values = jnp.asarray(calibration_inputs)
    in_size = _get_size(model.in_size)
    if model.in_size == "scalar":
        if values.ndim == 0:
            batch = values.reshape((1,))
        elif values.ndim == 1:
            batch = values
        elif values.ndim == 2 and values.shape[1] == 1:
            batch = values[:, 0]
        else:
            raise ValueError(
                "Scalar KAN calibration inputs must have shape (samples,) or "
                "(samples, 1)."
            )
    else:
        if values.ndim == 1 and values.shape[0] == in_size:
            batch = values.reshape((1, in_size))
        elif values.ndim == 2 and values.shape[1] == in_size:
            batch = values
        else:
            raise ValueError(
                f"KAN calibration inputs must have shape (samples, {in_size})."
            )
    if batch.shape[0] == 0:
        raise ValueError("KAN calibration inputs must contain at least one sample.")
    if not np.all(np.isfinite(np.asarray(batch))):
        raise ValueError("KAN calibration inputs must be finite.")
    return batch


def _minimum_span_knots(
    candidates: np.ndarray,
    lower: float,
    upper: float,
    minimum_span: float,
) -> np.ndarray:
    count = candidates.size
    if minimum_span * (count + 1) >= upper - lower:
        raise ValueError(
            "adaptation minimum_span is too large for the grid coefficient count."
        )
    lower_bounds = lower + minimum_span * np.arange(1, count + 1)
    upper_bounds = upper - minimum_span * np.arange(count, 0, -1)
    knots = np.clip(candidates, lower_bounds, upper_bounds)
    for index in range(1, count):
        knots[index] = max(knots[index], knots[index - 1] + minimum_span)
    for index in range(count - 2, -1, -1):
        knots[index] = min(knots[index], knots[index + 1] - minimum_span)
    return knots


def _adapted_grid(
    old_grid: BSplineGrid,
    activations: Array,
    plan: KANGridAdaptationPlan,
) -> tuple[BSplineGrid, bool]:
    lower, upper = (float(value) for value in old_grid.active_interval)
    interior_count = old_grid.coefficient_count - old_grid.degree - 1
    if interior_count == 0:
        return old_grid, False

    samples = np.clip(np.asarray(activations, dtype=float).reshape((-1,)), lower, upper)
    degenerate = (
        samples.size < interior_count + 1
        or np.unique(samples).size < 2
        or float(np.ptp(samples)) < plan.minimum_span
    )
    if degenerate and plan.degenerate_policy == "retain":
        return old_grid, True

    uniform = np.linspace(lower, upper, interior_count + 2)[1:-1]
    if degenerate:
        candidates = uniform
    else:
        probabilities = np.arange(1, interior_count + 1) / (interior_count + 1)
        quantiles = np.quantile(samples, probabilities, method=plan.quantile_method)
        candidates = (1.0 - plan.blend) * quantiles + plan.blend * uniform
    interior = _minimum_span_knots(
        np.asarray(candidates, dtype=float),
        lower,
        upper,
        plan.minimum_span,
    )
    knots = jnp.concatenate(
        (
            jnp.full((old_grid.degree + 1,), lower, dtype=old_grid.knots.dtype),
            jnp.asarray(interior, dtype=old_grid.knots.dtype),
            jnp.full((old_grid.degree + 1,), upper, dtype=old_grid.knots.dtype),
        )
    )
    return BSplineGrid(knots, old_grid.degree), degenerate


def _adapt_layer(
    layer: KANLayer,
    normalized_inputs: Array,
    plan: KANGridAdaptationPlan,
    path: tuple[int, int],
    records: _AdaptationRecords,
) -> KANLayer:
    if not isinstance(layer.edge_basis, BSplineEdgeBasis):
        records.skipped_paths.append(path)
        return layer

    if isinstance(layer.edge_basis.grid, TrainableBSplineGrid):
        raise ValueError(
            "Explicit grid adaptation only supports fixed B-spline grids; "
            "trainable knot grids must be optimized through solver phases."
        )
    old_grid = layer.edge_basis.grid
    if isinstance(old_grid, BSplineGridBank) or plan.per_input:
        old_grids = (
            old_grid.grids
            if isinstance(old_grid, BSplineGridBank)
            else (old_grid,) * int(layer.coeffs.shape[1])
        )
        new_grids: list[BSplineGrid] = []
        coefficient_columns: list[Array] = []
        for input_index, input_grid in enumerate(old_grids):
            input_values = normalized_inputs[..., input_index]
            new_grid, degenerate = _adapted_grid(input_grid, input_values, plan)
            transfer = BSplineGridTransfer(input_grid, new_grid)
            old_coefficients = layer.coeffs[:, input_index, :]
            coefficient_columns.append(transfer(old_coefficients))
            new_grids.append(new_grid)
            coefficient_norms = np.linalg.norm(np.asarray(old_coefficients), axis=-1)
            records.paths.append(path)
            records.input_indices.append(input_index)
            records.old_grids.append(input_grid)
            records.new_grids.append(new_grid)
            records.transfer_conditioning.append(transfer.condition_estimate)
            records.projection_error_bounds.append(
                transfer.projection_error_bound * float(np.max(coefficient_norms))
            )
            records.activation_counts.append(int(input_values.size))
            if degenerate:
                records.degenerate_paths.append(path)
                records.degenerate_grid_paths.append((*path, input_index))
        new_basis = BSplineEdgeBasis(
            grid=BSplineGridBank.from_grids(tuple(new_grids)),
            regularization_order=layer.edge_basis.regularization_order,
            per_input=True,
        )
        return eqx.tree_at(
            lambda current: (current.edge_basis, current.coeffs),
            layer,
            (new_basis, jnp.stack(tuple(coefficient_columns), axis=1)),
        )

    new_grid, degenerate = _adapted_grid(old_grid, normalized_inputs, plan)
    transfer = BSplineGridTransfer(old_grid, new_grid)
    new_coefficients = transfer(layer.coeffs)
    new_basis = BSplineEdgeBasis(
        grid=new_grid,
        regularization_order=layer.edge_basis.regularization_order,
    )
    new_layer = eqx.tree_at(
        lambda current: (current.edge_basis, current.coeffs),
        layer,
        (new_basis, new_coefficients),
    )

    coefficient_norms = np.linalg.norm(
        np.asarray(layer.coeffs).reshape((-1, old_grid.coefficient_count)),
        axis=1,
    )
    records.paths.append(path)
    records.input_indices.append(None)
    records.old_grids.append(old_grid)
    records.new_grids.append(new_grid)
    records.transfer_conditioning.append(transfer.condition_estimate)
    records.projection_error_bounds.append(
        transfer.projection_error_bound * float(np.max(coefficient_norms))
    )
    records.activation_counts.append(int(normalized_inputs.size))
    if degenerate:
        records.degenerate_paths.append(path)
    return new_layer


def _adapt_single_kan(
    model: KAN,
    calibration_inputs: ArrayLike,
    plan: KANGridAdaptationPlan,
    model_index: int,
    records: _AdaptationRecords,
) -> KAN:
    activations = _calibration_batch(model, calibration_inputs)
    layers: list[KANLayer] = []
    for layer_index, layer in enumerate(model.layers):
        normalized_inputs = jax.vmap(layer._normalized_edge_inputs)(activations)
        new_layer = _adapt_layer(
            layer,
            normalized_inputs,
            plan,
            (model_index, layer_index),
            records,
        )
        layers.append(new_layer)
        activations = jax.vmap(new_layer)(activations)
    return model._replace_layers(tuple(layers))


def _adapt_separable_kan(
    model: SeparableKAN,
    calibration_inputs: ArrayLike,
    plan: KANGridAdaptationPlan,
    records: _AdaptationRecords,
) -> SeparableKAN:
    wrapper = model.model
    if not isinstance(wrapper, Separable):
        raise TypeError("SeparableKAN must contain a Separable wrapper.")
    values = jnp.asarray(calibration_inputs)
    if wrapper._replicated_scalar_input:
        if values.ndim == 2 and values.shape[1] == 1:
            coordinate_batches = (values[:, 0],)
        elif values.ndim == 1:
            coordinate_batches = (values,)
        else:
            raise ValueError(
                "Replicated scalar SeparableKAN calibration inputs must have shape "
                "(samples,) or (samples, 1)."
            )
    else:
        if values.ndim == 1 and values.shape[0] == wrapper._base_in_dim:
            values = values.reshape((1, wrapper._base_in_dim))
        if values.ndim != 2 or values.shape[1] != wrapper._base_in_dim:
            raise ValueError(
                "SeparableKAN calibration inputs must have shape "
                f"(samples, {wrapper._base_in_dim})."
            )
        coordinate_batches = tuple(
            values[:, index] for index in range(wrapper._base_in_dim)
        )

    adapted_models = []
    for model_index, coordinate_model in enumerate(wrapper.models):
        if not isinstance(coordinate_model, KAN):
            raise TypeError("SeparableKAN coordinate models must be KAN instances.")
        coordinate_index = model_index // wrapper._clones
        adapted_models.append(
            _adapt_single_kan(
                coordinate_model,
                coordinate_batches[coordinate_index],
                plan,
                model_index,
                records,
            )
        )
    split_input = wrapper._clones if wrapper._clones > 1 else None
    adapted_wrapper = Separable(
        in_size=wrapper.in_size,
        out_size=wrapper.out_size,
        latent_size=wrapper.latent_size,
        models=tuple(adapted_models),
        output_activation=wrapper.output_activation,
        keep_outputs_complex=wrapper.keep_outputs_complex,
        split_input=split_input,
        scan=wrapper.scan,
    )
    return eqx.tree_at(lambda current: current.model, model, adapted_wrapper)


@overload
def adapt_kan_grids(
    model: KAN,
    calibration_inputs: ArrayLike,
    /,
    *,
    plan: KANGridAdaptationPlan | None = None,
) -> tuple[KAN, KANGridAdaptationReport]: ...


@overload
def adapt_kan_grids(
    model: SeparableKAN,
    calibration_inputs: ArrayLike,
    /,
    *,
    plan: KANGridAdaptationPlan | None = None,
) -> tuple[SeparableKAN, KANGridAdaptationReport]: ...


def adapt_kan_grids(
    model: KAN | SeparableKAN,
    calibration_inputs: ArrayLike,
    /,
    *,
    plan: KANGridAdaptationPlan | None = None,
) -> tuple[KAN | SeparableKAN, KANGridAdaptationReport]:
    """Return a regridded model and diagnostics without mutating the input model."""
    plan_ = KANGridAdaptationPlan() if plan is None else plan
    if not isinstance(plan_, KANGridAdaptationPlan):
        raise TypeError("plan must be a KANGridAdaptationPlan.")
    records = _AdaptationRecords()
    if isinstance(model, KAN):
        adapted = _adapt_single_kan(model, calibration_inputs, plan_, 0, records)
    elif isinstance(model, SeparableKAN):
        adapted = _adapt_separable_kan(model, calibration_inputs, plan_, records)
    else:
        raise TypeError("adapt_kan_grids supports KAN and SeparableKAN models.")
    return adapted, records.report()


__all__ = [
    "DegenerateGridPolicy",
    "KANGridAdaptationPlan",
    "KANGridAdaptationReport",
    "QuantileMethod",
    "adapt_kan_grids",
]
