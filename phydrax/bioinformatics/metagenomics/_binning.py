#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation, route_reduce
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class BinningStatus(IntEnum):
    SUCCESS = 0
    INVALID_ASSIGNMENT = 1
    INVALID_MARKER_COUNT = 2
    EMPTY_BIN = 3
    NO_EXPECTED_MARKERS = 4


def _supplied_binning_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "supplied contig-binning boundary",
        MethodKind.HEURISTIC,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.PARTITION,
        conditioning_statement=(
            "Contig-to-bin labels are supplied by an external or heuristic binning method."
        ),
        truncation_statement=(
            "This boundary validates supplied labels and makes no claim about bin discovery."
        ),
        capacity_semantics=(
            "There is one explicit route slot per contig and a declared fixed bin capacity."
        ),
        assumptions=("Each assigned contig belongs to exactly one bin.",),
        nondifferentiable_outputs=("relation", "contig_valid", "bin_valid", "status"),
        input_dtype="int32/bool",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


def _metrics_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "single-copy-marker bin quality metrics",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.ARRAY,
        conditioning_statement=(
            "Completeness is observed expected markers / expected markers; contamination is "
            "excess expected-marker copies / expected markers."
        ),
        truncation_statement="Every expected marker and every assigned contig is included.",
        capacity_semantics="Contig, bin, and marker capacities are fixed by validated arrays.",
        assumptions=("Expected marker copy number is one.",),
        nondifferentiable_outputs=("marker_copy_counts", "status"),
        input_dtype="int/float/bool",
        compute_dtype="float32/int32",
        output_dtype="float32/int32",
    )


class ContigBinning(StrictModule, NonTrainableState):
    """One supplied hard partition represented by a native contig-to-bin relation."""

    contig_ids: Array
    contig_valid: Array
    relation: EdgeRelation
    bin_valid: Array
    method_contract: BioinformaticsMethodContract
    binning_id: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    @property
    def contig_capacity(self) -> int:
        return int(self.contig_ids.shape[0])

    @property
    def bin_capacity(self) -> int:
        return self.relation.target_size


class ContigBinningResult(StrictModule):
    binning: ContigBinning
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract


class BinningMetricsResult(StrictModule):
    marker_copy_counts: Array
    observed_marker_counts: Array
    excess_marker_copies: Array
    completeness: Array
    contamination: Array
    quality_score: Array
    contig_counts: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    binning_id: str = eqx.field(static=True)


def supplied_contig_binning(
    contig_ids: ArrayLike,
    bin_indices: ArrayLike,
    assigned_valid: ArrayLike,
    /,
    *,
    bin_capacity: int,
    contig_valid: ArrayLike | None = None,
    provenance: str = "caller-supplied",
    binning_id: str | None = None,
) -> ContigBinningResult:
    """Validate externally generated hard labels without implementing a clustering heuristic."""
    ids = jnp.asarray(contig_ids, dtype=jnp.int32)
    bins = jnp.asarray(bin_indices, dtype=jnp.int32)
    assigned = jnp.asarray(assigned_valid, dtype=bool)
    if ids.ndim != 1 or bins.shape != ids.shape or assigned.shape != ids.shape:
        raise ValueError("Contig IDs, bin indices, and assignment mask must match.")
    bins_capacity = int(bin_capacity)
    if bins_capacity < 1:
        raise ValueError("bin_capacity must be positive.")
    active = (
        jnp.ones(ids.shape, dtype=bool)
        if contig_valid is None
        else jnp.asarray(contig_valid, dtype=bool)
    )
    if active.shape != ids.shape:
        raise ValueError("contig_valid must match contig capacity.")
    in_bounds = (bins >= 0) & (bins < bins_capacity)
    invalid_route = assigned & ((~active) | (~in_bounds))
    route_valid = assigned & active & in_bounds
    safe_bins = jnp.clip(bins, 0, bins_capacity - 1)
    relation = EdgeRelation(
        jnp.arange(ids.size, dtype=jnp.int32),
        safe_bins,
        source_size=ids.size,
        target_size=bins_capacity,
        valid=route_valid,
    )
    bin_counts = route_reduce(relation, route_valid.astype(jnp.int32), reduction="sum")
    populated = bin_counts > 0
    provenance_ = str(provenance).strip()
    if not provenance_:
        raise ValueError("Binning provenance must be non-empty.")
    identity = binning_id or canonical_fingerprint(
        {
            "kind": "supplied-contig-binning",
            "bin_capacity": bins_capacity,
            "provenance": provenance_,
            "arrays": array_tree_fingerprint((ids, bins, assigned, active)),
        }
    )
    if not identity:
        raise ValueError("binning_id must be non-empty.")
    contract = _supplied_binning_contract()
    binning = ContigBinning(
        ids,
        active,
        relation,
        populated,
        contract,
        identity,
        provenance_,
    )
    valid = ~jnp.any(invalid_route)
    status = jnp.where(
        valid,
        int(BinningStatus.SUCCESS),
        int(BinningStatus.INVALID_ASSIGNMENT),
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        (
            jnp.sum(active, dtype=jnp.int32),
            jnp.sum(route_valid, dtype=jnp.int32),
            jnp.sum(active & (~route_valid), dtype=jnp.int32),
            jnp.sum(invalid_route, dtype=jnp.int32),
            jnp.sum(populated, dtype=jnp.int32),
        ),
        dtype=jnp.int32,
    )
    return ContigBinningResult(binning, valid, status, evidence, contract)


def evaluate_binning_markers(
    binning: ContigBinning,
    marker_copy_counts: ArrayLike,
    /,
    *,
    expected_markers: ArrayLike | None = None,
) -> BinningMetricsResult:
    """Compute exact completeness and contamination from expected single-copy markers."""
    marker_counts = jnp.asarray(marker_copy_counts)
    if marker_counts.ndim != 2 or marker_counts.shape[0] != binning.contig_capacity:
        raise ValueError(
            "marker_copy_counts must have shape (contig_capacity, marker_capacity)."
        )
    marker_capacity = marker_counts.shape[1]
    if marker_capacity < 1:
        raise ValueError("marker capacity must be positive.")
    if not jnp.issubdtype(marker_counts.dtype, jnp.inexact):
        marker_counts = marker_counts.astype(jnp.float32)
    expected = (
        jnp.ones((marker_capacity,), dtype=bool)
        if expected_markers is None
        else jnp.asarray(expected_markers, dtype=bool)
    )
    if expected.shape != (marker_capacity,):
        raise ValueError("expected_markers must match marker capacity.")
    finite_nonnegative = jnp.isfinite(marker_counts) & (marker_counts >= 0.0)
    marker_input_valid = jnp.all((~binning.contig_valid[:, None]) | finite_nonnegative)
    safe_counts = jnp.where(
        binning.contig_valid[:, None] & finite_nonnegative,
        marker_counts,
        0.0,
    )
    per_bin = route_reduce(binning.relation, safe_counts, reduction="sum")
    expected_count = jnp.sum(expected, dtype=jnp.int32)
    denominator = jnp.maximum(expected_count, 1).astype(per_bin.dtype)
    observed = jnp.sum((per_bin > 0.0) & expected[None, :], axis=1, dtype=jnp.int32)
    excess = jnp.sum(
        jnp.where(expected[None, :], jnp.maximum(per_bin - 1.0, 0.0), 0.0),
        axis=1,
    )
    completeness = observed.astype(per_bin.dtype) / denominator
    contamination = excess / denominator
    quality = completeness - contamination
    contig_counts = route_reduce(
        binning.relation,
        binning.relation.valid.astype(jnp.int32),
        reduction="sum",
    )
    valid = binning.bin_valid & marker_input_valid & (expected_count > 0)
    status = jnp.where(
        expected_count == 0,
        int(BinningStatus.NO_EXPECTED_MARKERS),
        jnp.where(
            ~marker_input_valid,
            int(BinningStatus.INVALID_MARKER_COUNT),
            jnp.where(
                binning.bin_valid,
                int(BinningStatus.SUCCESS),
                int(BinningStatus.EMPTY_BIN),
            ),
        ),
    ).astype(jnp.int32)
    completeness = jnp.where(valid, completeness, 0.0)
    contamination = jnp.where(valid, contamination, 0.0)
    quality = jnp.where(valid, quality, 0.0)
    contract = _metrics_contract()
    unbinned_count = jnp.sum(
        binning.contig_valid
        & (~jax_route_source_mask(binning.relation, binning.contig_capacity)),
        dtype=jnp.int32,
    )
    evidence = jnp.stack(
        (
            contig_counts.astype(per_bin.dtype),
            observed.astype(per_bin.dtype),
            excess,
            jnp.full((binning.bin_capacity,), expected_count, dtype=per_bin.dtype),
            jnp.full((binning.bin_capacity,), unbinned_count, dtype=per_bin.dtype),
        ),
        axis=1,
    )
    return BinningMetricsResult(
        per_bin,
        observed,
        excess,
        completeness,
        contamination,
        quality,
        contig_counts,
        valid,
        status,
        evidence,
        contract,
        binning.binning_id,
    )


def jax_route_source_mask(relation: EdgeRelation, source_capacity: int, /) -> Array:
    """Return which source slots participate in at least one valid supplied bin route."""
    safe_source = jnp.where(relation.valid, relation.source_indices, 0)
    counts = jnp.zeros((source_capacity,), dtype=jnp.int32)
    counts = counts.at[safe_source].add(relation.valid.astype(jnp.int32))
    return counts > 0


__all__ = [
    "BinningMetricsResult",
    "BinningStatus",
    "ContigBinning",
    "ContigBinningResult",
    "evaluate_binning_markers",
    "supplied_contig_binning",
]
