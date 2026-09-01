#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._assay import CountAssay


PSEUDOBULK_SUCCESS = 0
PSEUDOBULK_EMPTY_UNIT = 1


def _pseudobulk_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "technical-count-pseudobulk",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement="Counts are summed only across observed cells in each declared unit.",
        truncation_statement="No cells or units are truncated; unit capacity is explicit.",
        capacity_semantics="num_units fixes output capacity and every included route is preflighted.",
        assumptions=(
            "Rows assigned to one unit are technical observations of the same experimental unit.",
        ),
        nondifferentiable_outputs=(
            "counts",
            "contributing_cells",
            "contributing_observations",
            "valid",
            "status",
        ),
    )


class PseudobulkResult(StrictModule):
    """Aggregated count assay with technical-replication evidence."""

    assay: CountAssay
    contributing_cells: Array
    contributing_observations: Array
    source_to_unit: Array
    included_sources: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract


def pseudobulk_counts(
    assay: CountAssay,
    unit_indices: ArrayLike,
    /,
    *,
    num_units: int,
    included: ArrayLike | None = None,
) -> PseudobulkResult:
    """Sum technical observations into declared biological experimental units."""

    if not isinstance(assay, CountAssay):
        raise TypeError("assay must be a CountAssay.")
    units = jnp.asarray(unit_indices)
    if units.shape != (assay.num_samples,):
        raise ValueError(
            f"unit_indices must have shape ({assay.num_samples},); got {units.shape}."
        )
    if not jnp.issubdtype(units.dtype, jnp.integer):
        raise TypeError("unit_indices must have an integer dtype.")
    capacity = int(num_units)
    if capacity < 1:
        raise ValueError("num_units must be positive.")
    source_mask = (
        jnp.ones((assay.num_samples,), dtype=bool)
        if included is None
        else jnp.asarray(included, dtype=bool)
    )
    if source_mask.shape != (assay.num_samples,):
        raise ValueError(f"included must have shape ({assay.num_samples},).")
    units = eqx.error_if(
        units.astype(jnp.int32),
        jnp.any(source_mask & ((units < 0) | (units >= capacity))),
        "An included unit index lies outside declared num_units.",
    )

    counts, observed, structural, missing = assay.dense_components()
    safe_units = jnp.where(source_mask, units, 0)
    contributing_cells = jax.ops.segment_sum(
        source_mask.astype(jnp.int32), safe_units, num_segments=capacity
    )
    contributing_observations = jax.ops.segment_sum(
        jnp.where(source_mask[:, None], observed, False).astype(jnp.int32),
        safe_units,
        num_segments=capacity,
    )
    structural_count = jax.ops.segment_sum(
        jnp.where(source_mask[:, None], structural, False).astype(jnp.int32),
        safe_units,
        num_segments=capacity,
    )
    missing_count = jax.ops.segment_sum(
        jnp.where(source_mask[:, None], missing, False).astype(jnp.int32),
        safe_units,
        num_segments=capacity,
    )
    aggregated_counts = jax.ops.segment_sum(
        jnp.where(source_mask[:, None] & observed, counts, 0),
        safe_units,
        num_segments=capacity,
    )
    observed_output = contributing_observations > 0
    structural_output = (
        (~observed_output)
        & (structural_count == contributing_cells[:, None])
        & (contributing_cells[:, None] > 0)
    )
    missing_output = ~observed_output & ~structural_output
    aggregated = CountAssay(
        aggregated_counts,
        missing=missing_output,
        structural_absence=structural_output,
    )
    valid = contributing_cells > 0
    status = jnp.where(valid, PSEUDOBULK_SUCCESS, PSEUDOBULK_EMPTY_UNIT).astype(jnp.int32)
    evidence = jnp.stack(
        (
            contributing_cells.astype(counts.dtype),
            jnp.sum(contributing_observations, axis=1).astype(counts.dtype),
            jnp.sum(structural_count, axis=1).astype(counts.dtype),
            jnp.sum(missing_count, axis=1).astype(counts.dtype),
        ),
        axis=1,
    )
    return PseudobulkResult(
        assay=aggregated,
        contributing_cells=contributing_cells,
        contributing_observations=contributing_observations,
        source_to_unit=units,
        included_sources=source_mask,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_pseudobulk_contract(),
    )


__all__ = [
    "PSEUDOBULK_EMPTY_UNIT",
    "PSEUDOBULK_SUCCESS",
    "PseudobulkResult",
    "pseudobulk_counts",
]
