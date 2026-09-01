#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from ._peaks import find_top_peaks
from ._types import CorrelationBatch, EnsemblePIVAccumulator, PeakBatch, PIVResult


def initialize_ensemble(prepared: object, /) -> EnsemblePIVAccumulator:
    """Allocate a fixed final-pass ensemble correlation accumulator."""
    from ._plan import PreparedPIV

    if not isinstance(prepared, PreparedPIV):
        raise TypeError("prepared must be a PreparedPIV.")
    grid = prepared.grids[-1]
    margin = grid.search_margin
    surface_shape = grid.grid_shape + (2 * margin[0] + 1, 2 * margin[1] + 1)
    dtype = (
        jnp.float64
        if prepared.report.resolved_compute_dtype == "float64"
        else jnp.float32
    )
    rows, columns = jnp.meshgrid(
        jnp.arange(-margin[0], margin[0] + 1, dtype=jnp.int32),
        jnp.arange(-margin[1], margin[1] + 1, dtype=jnp.int32),
        indexing="ij",
    )
    lags = jnp.stack((rows, columns), axis=-1)
    return EnsemblePIVAccumulator(
        jnp.zeros(surface_shape, dtype=dtype),
        jnp.zeros(surface_shape, dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros(surface_shape, dtype=jnp.int32),
        lags,
        prepared.prepared_id,
        prepared.report.resolved_compute_dtype,
    )


def accumulate_ensemble(
    accumulator: EnsemblePIVAccumulator,
    result: PIVResult,
    /,
) -> EnsemblePIVAccumulator:
    """Add one retained final-pass surface without treating missing lags as zero."""
    if not isinstance(accumulator, EnsemblePIVAccumulator):
        raise TypeError("accumulator must be an EnsemblePIVAccumulator.")
    if not isinstance(result, PIVResult):
        raise TypeError("result must be a PIVResult.")
    if not result.retention.retained:
        raise ValueError("PIVResult must retain correlation for ensemble accumulation.")
    if result.prepared_id != accumulator.prepared_id:
        raise ValueError("PIVResult and ensemble accumulator preparation differ.")
    if result.retention.correlation.shape != accumulator.correlation_sum.shape:
        raise ValueError("Retained correlation shape does not match the accumulator.")
    valid = jnp.isfinite(result.retention.correlation)
    return EnsemblePIVAccumulator(
        accumulator.correlation_sum + jnp.where(valid, result.retention.correlation, 0.0),
        accumulator.overlap_sum + jnp.where(valid, result.retention.overlap, 0.0),
        accumulator.sample_count + 1,
        accumulator.valid_count + valid.astype(jnp.int32),
        accumulator.lags_rc,
        accumulator.prepared_id,
        accumulator.resolved_compute_dtype,
    )


def merge_ensembles(
    left: EnsemblePIVAccumulator,
    right: EnsemblePIVAccumulator,
    /,
) -> EnsemblePIVAccumulator:
    """Merge independent ensemble partitions exactly through sufficient sums."""
    if left.prepared_id != right.prepared_id:
        raise ValueError("Ensemble preparations differ.")
    if left.correlation_sum.shape != right.correlation_sum.shape:
        raise ValueError("Ensemble capacities differ.")
    return EnsemblePIVAccumulator(
        left.correlation_sum + right.correlation_sum,
        left.overlap_sum + right.overlap_sum,
        left.sample_count + right.sample_count,
        left.valid_count + right.valid_count,
        left.lags_rc,
        left.prepared_id,
        left.resolved_compute_dtype,
    )


def ensemble_correlation(accumulator: EnsemblePIVAccumulator, /) -> CorrelationBatch:
    """Materialize the mean surface with explicit per-lag observation support."""
    if not isinstance(accumulator, EnsemblePIVAccumulator):
        raise TypeError("accumulator must be an EnsemblePIVAccumulator.")
    valid = accumulator.valid_count > 0
    divisor = jnp.maximum(accumulator.valid_count, 1)
    values = accumulator.correlation_sum / divisor
    overlap = accumulator.overlap_sum / divisor
    surface_shape = values.shape[-2:]
    return CorrelationBatch(
        jnp.where(valid, values, -jnp.inf).reshape((-1,) + surface_shape),
        overlap.reshape((-1,) + surface_shape),
        valid.reshape((-1,) + surface_shape),
        accumulator.lags_rc,
        "ensemble",
    )


def ensemble_peaks(
    accumulator: EnsemblePIVAccumulator,
    /,
    *,
    top_k: int = 2,
    method: str = "gaussian",
) -> PeakBatch:
    """Fit deterministic peaks to the accumulated mean correlation."""
    return find_top_peaks(
        ensemble_correlation(accumulator),
        top_k=top_k,
        method=method,
    )


__all__ = [
    "accumulate_ensemble",
    "ensemble_correlation",
    "ensemble_peaks",
    "initialize_ensemble",
    "merge_ensembles",
]
