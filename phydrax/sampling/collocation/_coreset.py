#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Key

from phydrax.coresets import (
    CoresetSelection,
    kernel_herd,
    KernelHerding,
    RadialKernel,
    weighted_mmd,
)
from phydrax.domain import PointBatch, PointSampling

from ..._doc import DOC_KEY0
from ..._sampling import DesignLike, resolve_design, UnitDesign
from ..._strict import StrictModule
from ._adaptive import (
    _collocation_population_metrics,
    _concat_batches,
    _point_sampling,
    _single_axis_and_size,
    _take_batch,
    _validate_axis_field,
    AbstractCollocationPolicy,
    CollocationPopulation,
    PointwiseSamplingTerm,
)


if TYPE_CHECKING:
    from phydrax.domain import DomainFunction


class _CoresetCollocationDiagnostics(StrictModule):
    """Last-refresh importance, kernel, and coverage evidence."""

    selection_valid: Array
    selection_accepted: Array
    selection_mmd: Array
    importance_effective_sample_size: Array
    effective_uniform_fraction: Array
    ess_guard_triggered: Array
    kernel_length_scale_min: Array
    kernel_length_scale_max: Array
    coverage_fill_distance: Array
    coverage_baseline_fill_distance: Array
    coverage_guard_triggered: Array
    selection_kernel_evaluations: Array

    def __init__(
        self,
        *,
        selection_valid: bool | Array = False,
        selection_accepted: bool | Array = False,
        selection_mmd: float | Array = 0.0,
        importance_effective_sample_size: float | Array = 0.0,
        effective_uniform_fraction: float | Array = 0.0,
        ess_guard_triggered: bool | Array = False,
        kernel_length_scale_min: float | Array = 0.0,
        kernel_length_scale_max: float | Array = 0.0,
        coverage_fill_distance: float | Array = 0.0,
        coverage_baseline_fill_distance: float | Array = 0.0,
        coverage_guard_triggered: bool | Array = False,
        selection_kernel_evaluations: int | Array = 0,
    ):
        self.selection_valid = jnp.asarray(selection_valid, dtype=bool)
        self.selection_accepted = jnp.asarray(selection_accepted, dtype=bool)
        self.selection_mmd = jnp.asarray(selection_mmd, dtype=float)
        self.importance_effective_sample_size = jnp.asarray(
            importance_effective_sample_size,
            dtype=float,
        )
        self.effective_uniform_fraction = jnp.asarray(
            effective_uniform_fraction,
            dtype=float,
        )
        self.ess_guard_triggered = jnp.asarray(ess_guard_triggered, dtype=bool)
        self.kernel_length_scale_min = jnp.asarray(
            kernel_length_scale_min,
            dtype=float,
        )
        self.kernel_length_scale_max = jnp.asarray(
            kernel_length_scale_max,
            dtype=float,
        )
        self.coverage_fill_distance = jnp.asarray(
            coverage_fill_distance,
            dtype=float,
        )
        self.coverage_baseline_fill_distance = jnp.asarray(
            coverage_baseline_fill_distance,
            dtype=float,
        )
        self.coverage_guard_triggered = jnp.asarray(
            coverage_guard_triggered,
            dtype=bool,
        )
        self.selection_kernel_evaluations = jnp.asarray(
            selection_kernel_evaluations,
            dtype=jnp.int64,
        )


def _coreset_diagnostics(
    population: CollocationPopulation,
    /,
) -> _CoresetCollocationDiagnostics:
    if not isinstance(population.diagnostics, _CoresetCollocationDiagnostics):
        raise TypeError("CoresetCollocationPolicy requires its initialized population.")
    return population.diagnostics


class CoresetCollocationPolicy(AbstractCollocationPolicy):
    """Robust residual-weighted kernel herding over refreshed candidates.

    Candidate scores are normalized before mixing with a uniform measure. The
    effective uniform fraction is raised when needed to enforce the requested
    importance ESS. Candidate coordinates are range-normalized; omitting ``kernel``
    selects a median-distance scale on each refresh. A proposal that regresses
    candidate fill distance too far relative to the retained population is rejected.
    """

    refresh_every: int
    start_at: int
    sampler: UnitDesign
    candidate_multiplier: int
    exponent: Array
    uniform_fraction: Array
    minimum_ess_fraction: Array
    max_fill_distance_ratio: Array
    epsilon: Array
    kernel: RadialKernel | None
    kernel_scale_factor: Array
    block_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        refresh_every: int = 100,
        start_at: int | None = None,
        sampler: DesignLike = "halton_scrambled",
        candidate_multiplier: int = 10,
        exponent: float = 0.5,
        uniform_fraction: float = 0.5,
        minimum_ess_fraction: float = 0.5,
        max_fill_distance_ratio: float = 3.0,
        epsilon: float = 1e-12,
        kernel: RadialKernel | None = None,
        kernel_scale_factor: float = 1.0,
        block_size: int = 256,
    ):
        refresh = int(refresh_every)
        activation = 2 * refresh if start_at is None else int(start_at)
        multiplier = int(candidate_multiplier)
        exponent_value = float(exponent)
        uniform = float(uniform_fraction)
        minimum_ess = float(minimum_ess_fraction)
        fill_ratio = float(max_fill_distance_ratio)
        epsilon_value = float(epsilon)
        scale_factor = float(kernel_scale_factor)
        block = int(block_size)
        if refresh <= 0:
            raise ValueError("refresh_every must be positive.")
        if activation <= 0:
            raise ValueError("start_at must be positive.")
        if multiplier < 2:
            raise ValueError("candidate_multiplier must be at least two.")
        if not isfinite(exponent_value) or exponent_value < 0.0:
            raise ValueError("exponent must be finite and non-negative.")
        if not isfinite(uniform) or not 0.0 <= uniform <= 1.0:
            raise ValueError("uniform_fraction must lie in [0, 1].")
        if not isfinite(minimum_ess) or not 0.0 < minimum_ess <= 1.0:
            raise ValueError("minimum_ess_fraction must lie in (0, 1].")
        if not isfinite(fill_ratio) or fill_ratio < 1.0:
            raise ValueError("max_fill_distance_ratio must be finite and at least one.")
        if not isfinite(epsilon_value) or epsilon_value <= 0.0:
            raise ValueError("epsilon must be finite and strictly positive.")
        if kernel is not None and not isinstance(kernel, RadialKernel):
            raise TypeError("kernel must be a RadialKernel or None.")
        if not isfinite(scale_factor) or scale_factor <= 0.0:
            raise ValueError("kernel_scale_factor must be finite and strictly positive.")
        if block <= 0:
            raise ValueError("block_size must be positive.")
        self.refresh_every = refresh
        self.start_at = activation
        self.sampler = resolve_design(sampler)
        self.candidate_multiplier = multiplier
        self.exponent = jnp.asarray(exponent_value, dtype=float)
        self.uniform_fraction = jnp.asarray(uniform, dtype=float)
        self.minimum_ess_fraction = jnp.asarray(minimum_ess, dtype=float)
        self.max_fill_distance_ratio = jnp.asarray(fill_ratio, dtype=float)
        self.epsilon = jnp.asarray(epsilon_value, dtype=float)
        self.kernel = kernel
        self.kernel_scale_factor = jnp.asarray(scale_factor, dtype=float)
        self.block_size = block

    def initialize(
        self,
        term: PointwiseSamplingTerm,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> CollocationPopulation:
        batch = term.sample(key=key)
        if not isinstance(batch, PointBatch):
            raise TypeError("Coreset collocation requires a PointBatch.")
        _, size = _single_axis_and_size(batch)
        return CollocationPopulation(
            batch,
            diagnostics=_CoresetCollocationDiagnostics(
                importance_effective_sample_size=size,
                effective_uniform_fraction=self.uniform_fraction,
            ),
        )

    def should_refresh(
        self,
        population: CollocationPopulation,
        iter_: int | Array,
    ) -> Array:
        step = jnp.asarray(iter_, dtype=jnp.int32)
        activated = step >= self.start_at
        due = (step - population.last_refresh) >= self.refresh_every
        return activated & due

    def loss_batch_and_weight(
        self,
        population: CollocationPopulation,
        /,
    ) -> tuple[PointBatch, cx.Field | None]:
        return population.batch, None

    def data_metrics(
        self,
        population: CollocationPopulation,
        /,
    ) -> dict[str, Array]:
        metrics = _collocation_population_metrics(population)
        _, size = _single_axis_and_size(population.batch)
        candidate_count = size * self.candidate_multiplier
        diagnostics = _coreset_diagnostics(population)
        metrics.update(
            {
                "coreset_candidate_count": jnp.asarray(candidate_count, dtype=float),
                "coreset_candidate_multiplier": jnp.asarray(
                    self.candidate_multiplier,
                    dtype=float,
                ),
                "coreset_start_at": jnp.asarray(self.start_at, dtype=float),
                "coreset_score_exponent": jnp.asarray(self.exponent),
                "coreset_uniform_fraction": jnp.asarray(self.uniform_fraction),
                "coreset_effective_uniform_fraction": jnp.asarray(
                    diagnostics.effective_uniform_fraction,
                    dtype=float,
                ),
                "coreset_minimum_ess_fraction": jnp.asarray(
                    self.minimum_ess_fraction
                ),
                "coreset_max_fill_distance_ratio": jnp.asarray(
                    self.max_fill_distance_ratio
                ),
                "coreset_importance_effective_sample_size": jnp.asarray(
                    diagnostics.importance_effective_sample_size,
                    dtype=float,
                ),
                "coreset_importance_effective_sample_fraction": jnp.asarray(
                    diagnostics.importance_effective_sample_size / candidate_count,
                    dtype=float,
                ),
                "coreset_ess_guard_triggered": jnp.asarray(
                    diagnostics.ess_guard_triggered,
                    dtype=float,
                ),
                "coreset_kernel_automatic": jnp.asarray(
                    self.kernel is None,
                    dtype=float,
                ),
                "coreset_kernel_scale_factor": jnp.asarray(
                    self.kernel_scale_factor
                ),
                "coreset_kernel_length_scale_min": jnp.asarray(
                    diagnostics.kernel_length_scale_min,
                    dtype=float,
                ),
                "coreset_kernel_length_scale_max": jnp.asarray(
                    diagnostics.kernel_length_scale_max,
                    dtype=float,
                ),
                "coreset_coverage_fill_distance": jnp.asarray(
                    diagnostics.coverage_fill_distance,
                    dtype=float,
                ),
                "coreset_coverage_baseline_fill_distance": jnp.asarray(
                    diagnostics.coverage_baseline_fill_distance,
                    dtype=float,
                ),
                "coreset_coverage_guard_triggered": jnp.asarray(
                    diagnostics.coverage_guard_triggered,
                    dtype=float,
                ),
                "coreset_selection_kernel_evaluations": jnp.asarray(
                    diagnostics.selection_kernel_evaluations,
                    dtype=float,
                ),
                "coreset_selection_valid": jnp.asarray(
                    diagnostics.selection_valid,
                    dtype=float,
                ),
                "coreset_selection_accepted": jnp.asarray(
                    diagnostics.selection_accepted,
                    dtype=float,
                ),
                "coreset_selection_mmd": jnp.asarray(
                    diagnostics.selection_mmd,
                    dtype=float,
                ),
            }
        )
        return metrics

    def refresh_residual_evaluations(
        self,
        population: CollocationPopulation,
        /,
    ) -> int:
        _, size = _single_axis_and_size(population.batch)
        return size * self.candidate_multiplier

    def refresh(
        self,
        term: PointwiseSamplingTerm,
        functions: Mapping[str, DomainFunction],
        population: CollocationPopulation,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array,
    ) -> CollocationPopulation:
        _coreset_diagnostics(population)
        sampling = _point_sampling(term)
        axis, size = _single_axis_and_size(population.batch)
        new_count = size * (self.candidate_multiplier - 1)
        replacement = term.component.sample(
            PointSampling(
                new_count,
                layout=sampling.layout,
                design=self.sampler,
            ),
            key=jr.fold_in(key, 1),
        )
        if not isinstance(replacement, PointBatch):
            raise TypeError("Coreset candidate sampling requires a PointBatch.")
        candidates = _concat_batches(population.batch, replacement)
        scores = term.pointwise_score(
            functions,
            candidates,
            key=jr.fold_in(key, 2),
        )
        candidate_count = size * self.candidate_multiplier
        _validate_axis_field(
            scores,
            axis=axis,
            size=candidate_count,
            name="pointwise loss",
        )
        score_values = jax.lax.stop_gradient(jnp.asarray(scores.data, dtype=float))
        importance, importance_ess, effective_uniform, ess_guard_triggered = (
            _normalized_importance(
                score_values,
                exponent=self.exponent,
                uniform_fraction=self.uniform_fraction,
                minimum_ess_fraction=self.minimum_ess_fraction,
                epsilon=self.epsilon,
            )
        )
        point_features = _point_feature_matrix(candidates, axis, candidate_count)
        normalized_features = _normalize_point_features(point_features)
        resolved_kernel = _resolve_selection_kernel(
            normalized_features,
            kernel=self.kernel,
            scale_factor=self.kernel_scale_factor,
        )
        selection = _compiled_kernel_herd(
            normalized_features,
            KernelHerding(
                size,
                kernel=resolved_kernel,
                block_size=self.block_size,
                unique=True,
            ),
            jnp.log(importance),
        )
        baseline_fill_distance = _coverage_fill_distance(
            normalized_features,
            normalized_features[:size],
            block_size=self.block_size,
        )
        selection_valid = bool(selection.diagnostics.valid)
        if selection_valid:
            proposal_fill_distance = _coverage_fill_distance(
                normalized_features,
                normalized_features[selection.indices],
                block_size=self.block_size,
            )
            numerical_slack = jnp.sqrt(jnp.finfo(normalized_features.dtype).eps)
            coverage_valid = bool(
                proposal_fill_distance
                <= self.max_fill_distance_ratio * baseline_fill_distance
                + numerical_slack
            )
        else:
            proposal_fill_distance = jnp.asarray(jnp.inf, dtype=float)
            coverage_valid = False
        selection_accepted = selection_valid and coverage_valid
        if selection_accepted:
            selected_indices = selection.indices
            selection_mmd = selection.diagnostics.mmd
            fallback_kernel_evaluations = 0
        else:
            selected_indices = jnp.arange(size, dtype=jnp.int32)
            selection_mmd = _compiled_weighted_mmd(
                normalized_features,
                normalized_features[:size],
                jnp.log(importance),
                resolved_kernel,
                self.block_size,
            )
            fallback_kernel_evaluations = (
                candidate_count * candidate_count
                + 2 * candidate_count * size
                + size * size
            )
        selected_batch = _take_batch(candidates, selected_indices)
        candidate_age = jnp.concatenate(
            (
                jnp.asarray(population.age.data, dtype=jnp.int32) + 1,
                jnp.zeros((new_count,), dtype=jnp.int32),
            )
        )
        age = cx.Field(candidate_age[selected_indices], dims=(axis,))
        step = jnp.asarray(iter_, dtype=jnp.int32)
        kernel_evaluations = (
            2 * candidate_count * candidate_count
            + 2 * candidate_count * size
            + size * size
            + fallback_kernel_evaluations
        )
        length_scale = jnp.asarray(resolved_kernel.length_scale, dtype=float)
        return CollocationPopulation(
            selected_batch,
            age=age,
            refresh_count=population.refresh_count + 1,
            last_refresh=step,
            diagnostics=_CoresetCollocationDiagnostics(
                selection_valid=selection.diagnostics.valid,
                selection_accepted=selection_accepted,
                selection_mmd=selection_mmd,
                importance_effective_sample_size=importance_ess,
                effective_uniform_fraction=effective_uniform,
                ess_guard_triggered=ess_guard_triggered,
                kernel_length_scale_min=jnp.min(length_scale),
                kernel_length_scale_max=jnp.max(length_scale),
                coverage_fill_distance=proposal_fill_distance,
                coverage_baseline_fill_distance=baseline_fill_distance,
                coverage_guard_triggered=selection_valid and not coverage_valid,
                selection_kernel_evaluations=kernel_evaluations,
            ),
        )


@eqx.filter_jit
def _compiled_kernel_herd(
    points: Array,
    method: KernelHerding,
    log_weights: Array,
    /,
) -> CoresetSelection:
    return kernel_herd(points, method, log_weights=log_weights)


@eqx.filter_jit
def _compiled_weighted_mmd(
    source_points: Array,
    comparison_points: Array,
    source_log_weights: Array,
    kernel: RadialKernel,
    block_size: int,
    /,
) -> Array:
    return weighted_mmd(
        source_points,
        comparison_points,
        source_log_weights=source_log_weights,
        kernel=kernel,
        block_size=block_size,
    )


@jax.jit
def _normalized_importance(
    scores: Array,
    /,
    *,
    exponent: Array,
    uniform_fraction: Array,
    minimum_ess_fraction: Array,
    epsilon: Array,
) -> tuple[Array, Array, Array, Array]:
    values = jnp.asarray(scores, dtype=float)
    count = int(values.shape[0])
    values = jnp.nan_to_num(
        values,
        nan=0.0,
        posinf=jnp.finfo(values.dtype).max,
        neginf=0.0,
    )
    log_signal = exponent * jnp.log(jnp.maximum(values, 0.0) + epsilon)
    log_signal = jnp.nan_to_num(
        log_signal,
        nan=0.0,
        posinf=jnp.finfo(values.dtype).max,
        neginf=-jnp.finfo(values.dtype).max,
    )
    signal_probability = jax.nn.softmax(log_signal)
    uniform_probability = jnp.asarray(1.0 / count, dtype=values.dtype)
    signal_concentration = jnp.sum(signal_probability * signal_probability)
    target_ess = jnp.maximum(minimum_ess_fraction * count, 1.0)
    target_concentration = 1.0 / target_ess
    excess_signal_concentration = signal_concentration - uniform_probability
    tolerance = 16.0 * jnp.finfo(values.dtype).eps
    admissible_signal_fraction_squared = jnp.where(
        excess_signal_concentration > tolerance,
        (target_concentration - uniform_probability)
        / excess_signal_concentration,
        1.0,
    )
    required_uniform_fraction = 1.0 - jnp.sqrt(
        jnp.clip(admissible_signal_fraction_squared, 0.0, 1.0)
    )
    effective_uniform_fraction = jnp.maximum(
        uniform_fraction,
        required_uniform_fraction,
    )
    importance = (
        effective_uniform_fraction * uniform_probability
        + (1.0 - effective_uniform_fraction) * signal_probability
    )
    importance_ess = 1.0 / jnp.sum(importance * importance)
    guard_triggered = effective_uniform_fraction > uniform_fraction + tolerance
    return (
        importance,
        importance_ess,
        effective_uniform_fraction,
        guard_triggered,
    )


@jax.jit
def _normalize_point_features(features: Array, /) -> Array:
    values = jnp.asarray(features, dtype=float)
    lower = jnp.min(values, axis=0)
    extent = jnp.max(values, axis=0) - lower
    tolerance = jnp.sqrt(jnp.finfo(values.dtype).eps)
    active = extent > tolerance
    safe_extent = jnp.where(active, extent, 1.0)
    normalized = (values - lower) / safe_extent
    return jnp.where(active, normalized, 0.0)


def _resolve_selection_kernel(
    features: Array,
    /,
    *,
    kernel: RadialKernel | None,
    scale_factor: Array,
) -> RadialKernel:
    coordinate_size = int(features.shape[1])
    if kernel is not None:
        length_scale = jnp.asarray(kernel.length_scale)
        if (
            length_scale.ndim == 1
            and length_scale.shape[0] not in (1, coordinate_size)
        ):
            raise ValueError(
                "Explicit coreset kernel length_scale must be scalar or match "
                "the normalized coordinate size."
            )
        return kernel
    length_scale = scale_factor * _median_pairwise_distance(features)
    minimum_scale = jnp.sqrt(jnp.finfo(features.dtype).eps)
    return RadialKernel(length_scale=jnp.maximum(length_scale, minimum_scale))


@jax.jit
def _median_pairwise_distance(features: Array, /) -> Array:
    point_count = int(features.shape[0])
    sample_count = min(point_count, 256)
    if sample_count < 2:
        return jnp.asarray(1.0, dtype=features.dtype)
    sample_indices = (
        jnp.arange(sample_count, dtype=jnp.int32) * (point_count - 1)
        // (sample_count - 1)
    )
    sample = features[sample_indices]
    left_indices, right_indices = jnp.triu_indices(sample_count, k=1)
    delta = sample[left_indices] - sample[right_indices]
    distances = jnp.sqrt(jnp.sum(delta * delta, axis=1))
    positive = distances > jnp.sqrt(jnp.finfo(distances.dtype).eps)
    ordered = jnp.sort(jnp.where(positive, distances, jnp.inf))
    positive_count = jnp.sum(positive, dtype=jnp.int32)
    median_index = jnp.maximum((positive_count - 1) // 2, 0)
    return jnp.where(
        positive_count > 0,
        ordered[median_index],
        jnp.asarray(1.0, dtype=features.dtype),
    )


@eqx.filter_jit
def _coverage_fill_distance(
    points: Array,
    support: Array,
    /,
    *,
    block_size: int,
) -> Array:
    source = jnp.asarray(points, dtype=float)
    retained = jnp.asarray(support, dtype=float)
    if source.ndim != 2 or retained.ndim != 2:
        raise ValueError("Coverage point arrays must be two-dimensional.")
    if source.shape[1] != retained.shape[1]:
        raise ValueError("Coverage point arrays must have equal coordinate size.")
    source_count, coordinate_size = map(int, source.shape)
    retained_count = int(retained.shape[0])
    if source_count == 0 or retained_count == 0:
        raise ValueError("Coverage point arrays must be non-empty.")
    block = min(int(block_size), max(source_count, retained_count))
    source_blocks = (source_count + block - 1) // block
    retained_blocks = (retained_count + block - 1) // block
    source_padded = jnp.pad(
        source,
        ((0, source_blocks * block - source_count), (0, 0)),
    )
    retained_padded = jnp.pad(
        retained,
        ((0, retained_blocks * block - retained_count), (0, 0)),
    )
    offsets = jnp.arange(block, dtype=jnp.int32)

    def source_body(source_block, maximum_squared_distance):
        source_start = source_block * block
        left = jax.lax.dynamic_slice(
            source_padded,
            (source_start, 0),
            (block, coordinate_size),
        )

        def retained_body(retained_block, minimum_squared_distance):
            retained_start = retained_block * block
            right = jax.lax.dynamic_slice(
                retained_padded,
                (retained_start, 0),
                (block, coordinate_size),
            )
            delta = left[:, None, :] - right[None, :, :]
            squared_distance = jnp.sum(delta * delta, axis=2)
            right_valid = retained_start + offsets < retained_count
            block_minimum = jnp.min(
                jnp.where(right_valid[None, :], squared_distance, jnp.inf),
                axis=1,
            )
            return jnp.minimum(minimum_squared_distance, block_minimum)

        minimum_squared_distance = jax.lax.fori_loop(
            0,
            retained_blocks,
            retained_body,
            jnp.full((block,), jnp.inf, dtype=source.dtype),
        )
        left_valid = source_start + offsets < source_count
        block_maximum = jnp.max(
            jnp.where(left_valid, minimum_squared_distance, 0.0)
        )
        return jnp.maximum(maximum_squared_distance, block_maximum)

    maximum_squared_distance = jax.lax.fori_loop(
        0,
        source_blocks,
        source_body,
        jnp.asarray(0.0, dtype=source.dtype),
    )
    return jnp.sqrt(jnp.maximum(maximum_squared_distance, 0.0))


def _point_feature_matrix(batch: PointBatch, axis: str, count: int, /) -> Array:
    matrices = []
    leaves = jtu.tree_leaves(
        batch.points,
        is_leaf=lambda value: isinstance(value, cx.Field),
    )
    for leaf in leaves:
        if not isinstance(leaf, cx.Field) or axis not in leaf.named_dims:
            continue
        position = leaf.dims.index(axis)
        values = jnp.moveaxis(jnp.asarray(leaf.data, dtype=float), position, 0)
        if values.shape[0] != count:
            raise ValueError("Candidate point fields disagree on sample count.")
        matrices.append(values.reshape((count, -1)))
    if not matrices:
        raise ValueError("Could not derive coreset coordinates from candidate points.")
    features = jnp.concatenate(tuple(matrices), axis=1)
    if bool(jnp.any(~jnp.isfinite(features))):
        raise ValueError("Coreset candidate coordinates must be finite.")
    return features


def CoresetCollocation(**kwargs) -> CoresetCollocationPolicy:
    """Construct residual-weighted, diversity-preserving paired collocation."""
    return CoresetCollocationPolicy(**kwargs)


__all__ = ["CoresetCollocation", "CoresetCollocationPolicy"]
