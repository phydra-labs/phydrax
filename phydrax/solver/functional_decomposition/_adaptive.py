#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...domain import (
    AxisPartition,
    LocalFieldFamily,
    partition_of_unity_field,
    SubdomainCover,
)
from .._functional_correction import freeze_domain_function


class AdaptiveTopologyEvidence(StrictModule, NonTrainableState):
    old_cover_id: str = eqx.field(static=True)
    new_cover_id: str = eqx.field(static=True)
    maximum_transfer_error: float = eqx.field(static=True)
    coverage_verified: bool = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        old_cover_id: str,
        new_cover_id: str,
        maximum_transfer_error: float,
        coverage_verified: bool,
        accepted: bool,
    ):
        self.old_cover_id = str(old_cover_id)
        self.new_cover_id = str(new_cover_id)
        self.maximum_transfer_error = float(maximum_transfer_error)
        self.coverage_verified = bool(coverage_verified)
        self.accepted = bool(accepted)


class AdaptiveTopologyTransaction(StrictModule):
    """Prepared candidate topology with a frozen field-preserving transfer."""

    source: LocalFieldFamily
    candidate_cover: SubdomainCover
    transferred: LocalFieldFamily
    evidence: AdaptiveTopologyEvidence

    def __init__(
        self,
        source: LocalFieldFamily,
        candidate_cover: SubdomainCover,
        transferred: LocalFieldFamily,
        evidence: AdaptiveTopologyEvidence,
        /,
    ):
        self.source = source
        self.candidate_cover = candidate_cover
        self.transferred = transferred
        self.evidence = evidence

    def commit(self) -> LocalFieldFamily:
        if not self.evidence.accepted:
            raise ValueError("Cannot commit a rejected adaptive topology transaction.")
        return self.transferred


class AdaptiveRefinementPlan(StrictModule, NonTrainableState):
    transfer_tolerance: float = eqx.field(static=True)
    maximum_patches: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        transfer_tolerance: float = 1.0e-8,
        maximum_patches: int = 1024,
    ):
        tolerance = float(transfer_tolerance)
        maximum = int(maximum_patches)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("transfer_tolerance must be finite and non-negative.")
        if maximum <= 0:
            raise ValueError("maximum_patches must be positive.")
        self.transfer_tolerance = tolerance
        self.maximum_patches = maximum


def refine_axis_partition(
    partition: AxisPartition,
    cell_index: int,
    /,
    *,
    fraction: float = 0.5,
) -> AxisPartition:
    """Split one axis cell while preserving overlap and periodic semantics."""
    if not isinstance(partition, AxisPartition):
        raise TypeError("partition must be an AxisPartition.")
    index = int(cell_index)
    fraction_ = float(fraction)
    if not 0 <= index < partition.count:
        raise IndexError("cell_index is outside the axis partition.")
    if not math.isfinite(fraction_) or not 0.0 < fraction_ < 1.0:
        raise ValueError("fraction must be finite and in (0, 1).")
    left = partition.boundaries[index]
    right = partition.boundaries[index + 1]
    point = left + fraction_ * (right - left)
    boundaries = (
        *partition.boundaries[: index + 1],
        point,
        *partition.boundaries[index + 1 :],
    )
    return AxisPartition(
        boundaries,
        overlap_fraction=partition.overlap_fraction,
        periodic=partition.periodic,
    )


def prepare_adaptive_topology_transaction(
    source: LocalFieldFamily,
    candidate_cover: SubdomainCover,
    audit_points: Any,
    plan: AdaptiveRefinementPlan | None = None,
    /,
) -> AdaptiveTopologyTransaction:
    """Validate a candidate cover and preserve the represented ambient field."""
    if not isinstance(source, LocalFieldFamily):
        raise TypeError("source must be a LocalFieldFamily.")
    if not isinstance(candidate_cover, SubdomainCover):
        raise TypeError("candidate_cover must be a SubdomainCover.")
    plan_ = AdaptiveRefinementPlan() if plan is None else plan
    if not isinstance(plan_, AdaptiveRefinementPlan):
        raise TypeError("plan must be an AdaptiveRefinementPlan or None.")
    if len(candidate_cover.patches) > plan_.maximum_patches:
        raise ValueError("Candidate cover exceeds maximum_patches.")
    if not source.cover.ambient.same_support(candidate_cover.ambient):
        raise ValueError("Adaptive source and candidate covers need one ambient domain.")
    if any(patch.window is None for patch in source.cover.patches):
        raise ValueError("Adaptive field transfer currently requires POU source windows.")
    if any(patch.window is None for patch in candidate_cover.patches):
        raise ValueError("Adaptive candidate cover requires POU windows.")

    source_field = freeze_domain_function(partition_of_unity_field(source))
    local_fields = {
        patch.patch_id: patch.restrict(source_field) for patch in candidate_cover.patches
    }
    transferred = LocalFieldFamily(source.field_id, candidate_cover, local_fields)
    candidate_field = partition_of_unity_field(transferred)
    source_values = jnp.asarray(source_field(audit_points).data)
    candidate_values = jnp.asarray(candidate_field(audit_points).data)
    transfer_error = float(jnp.max(jnp.abs(candidate_values - source_values)))
    coverage = (
        candidate_cover.structural_evidence()
        if candidate_cover.exact_coverage
        else candidate_cover.audit(audit_points)
    )
    accepted = coverage.verified and transfer_error <= plan_.transfer_tolerance
    evidence = AdaptiveTopologyEvidence(
        old_cover_id=source.cover.cover_id,
        new_cover_id=candidate_cover.cover_id,
        maximum_transfer_error=transfer_error,
        coverage_verified=coverage.verified,
        accepted=accepted,
    )
    return AdaptiveTopologyTransaction(
        source,
        candidate_cover,
        transferred,
        evidence,
    )


def coarsen_axis_partition(
    partition: AxisPartition,
    boundary_index: int,
    /,
) -> AxisPartition:
    """Remove one interior axis boundary while preserving overlap semantics."""
    if not isinstance(partition, AxisPartition):
        raise TypeError("partition must be an AxisPartition.")
    index = int(boundary_index)
    if not 0 < index < len(partition.boundaries) - 1:
        raise IndexError("boundary_index must select an interior boundary.")
    boundaries = (
        *partition.boundaries[:index],
        *partition.boundaries[index + 1 :],
    )
    return AxisPartition(
        boundaries,
        overlap_fraction=partition.overlap_fraction,
        periodic=partition.periodic,
    )


class TrainablePartitionEvidence(StrictModule, NonTrainableState):
    minimum_width: float = eqx.field(static=True)
    monotone: bool = eqx.field(static=True)
    verified: bool = eqx.field(static=True)

    def __init__(self, *, minimum_width: float, monotone: bool):
        self.minimum_width = float(minimum_width)
        self.monotone = bool(monotone)
        self.verified = self.monotone and self.minimum_width > 0.0


class TrainableAxisPartition(StrictModule):
    """Differentiable positive-width axis partition with immutable cell count."""

    width_logits: jax.Array
    start: float = eqx.field(static=True)
    end: float = eqx.field(static=True)
    minimum_fraction: float = eqx.field(static=True)
    overlap_fraction: float = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)

    def __init__(
        self,
        partition: AxisPartition,
        /,
        *,
        minimum_fraction: float = 1.0e-3,
    ):
        if not isinstance(partition, AxisPartition):
            raise TypeError("partition must be an AxisPartition.")
        minimum = float(minimum_fraction)
        if (
            not math.isfinite(minimum)
            or minimum <= 0.0
            or minimum * partition.count >= 1.0
        ):
            raise ValueError(
                "minimum_fraction must be positive and leave free simplex mass."
            )
        widths = jnp.diff(jnp.asarray(partition.boundaries))
        normalized = widths / jnp.sum(widths)
        free = (normalized - minimum) / (1.0 - partition.count * minimum)
        if bool(jnp.any(free <= 0.0)):
            raise ValueError("Existing partition violates minimum_fraction.")
        self.width_logits = jnp.log(free)
        self.start = partition.boundaries[0]
        self.end = partition.boundaries[-1]
        self.minimum_fraction = minimum
        self.overlap_fraction = partition.overlap_fraction
        self.periodic = partition.periodic

    def normalized_widths(self):
        count = int(self.width_logits.shape[0])
        free_mass = 1.0 - count * self.minimum_fraction
        return self.minimum_fraction + free_mass * jax.nn.softmax(self.width_logits)

    def boundary_array(self):
        widths = (self.end - self.start) * self.normalized_widths()
        return jnp.concatenate(
            (
                jnp.asarray([self.start], dtype=widths.dtype),
                self.start + jnp.cumsum(widths),
            )
        )

    def regularizer(self):
        widths = self.normalized_widths()
        return jnp.mean(jnp.square(widths - jnp.mean(widths)))

    def evidence(self) -> TrainablePartitionEvidence:
        boundaries = self.boundary_array()
        widths = jnp.diff(boundaries)
        minimum = float(jnp.min(widths))
        return TrainablePartitionEvidence(
            minimum_width=minimum,
            monotone=bool(jnp.all(widths > 0.0)),
        )

    def materialize(self) -> AxisPartition:
        evidence = self.evidence()
        if not evidence.verified:
            raise ValueError("Trainable partition failed monotonicity evidence.")
        return AxisPartition(
            tuple(float(value) for value in self.boundary_array()),
            overlap_fraction=self.overlap_fraction,
            periodic=self.periodic,
        )


__all__ = [
    "TrainableAxisPartition",
    "TrainablePartitionEvidence",
    "AdaptiveRefinementPlan",
    "AdaptiveTopologyEvidence",
    "AdaptiveTopologyTransaction",
    "coarsen_axis_partition",
    "prepare_adaptive_topology_transaction",
    "refine_axis_partition",
]
