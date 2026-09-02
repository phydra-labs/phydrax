#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize
from .._strict import StrictModule
from ._contraction import (
    ContractionPlan,
    execute_schedule,
    plan_contraction,
    PreparedContraction,
)
from ._topology import ContractionOperand, ContractionStructure


class SlicingResourcePolicy(StrictModule):
    maximum_slices: int = eqx.field(static=True)
    maximum_batch_elements: int = eqx.field(static=True)
    maximum_checkpoint_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_slices: int = 1_000_000,
        maximum_batch_elements: int = 100_000_000,
        maximum_checkpoint_bytes: int = 2**30,
    ):
        values = (
            int(maximum_slices),
            int(maximum_batch_elements),
            int(maximum_checkpoint_bytes),
        )
        if any(value < 1 for value in values):
            raise ValueError("Slicing resource limits must be positive.")
        (
            self.maximum_slices,
            self.maximum_batch_elements,
            self.maximum_checkpoint_bytes,
        ) = values
        self.policy_id = canonical_fingerprint(
            {"kind": "slicing-resource-policy", "limits": values}
        )


class SliceRange(StrictModule):
    start: int = eqx.field(static=True)
    stop: int = eqx.field(static=True)
    total: int = eqx.field(static=True)
    range_id: str = eqx.field(static=True)

    def __init__(self, start: int, stop: int, total: int, /):
        values = (int(start), int(stop), int(total))
        if not 0 <= values[0] < values[1] <= values[2]:
            raise ValueError("Slice ranges require 0 <= start < stop <= total.")
        self.start, self.stop, self.total = values
        self.range_id = canonical_fingerprint(
            {
                "kind": "slice-range",
                "start": values[0],
                "stop": values[1],
                "total": values[2],
            }
        )


class SlicedContractionPlan(StrictModule):
    original: ContractionPlan
    residual: ContractionPlan
    labels: tuple[str, ...] = eqx.field(static=True)
    dimensions: tuple[int, ...] = eqx.field(static=True)
    slice_count: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    resources: SlicingResourcePolicy
    exact: bool = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class SliceExecutionEvidence(StrictModule):
    plan_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    range_id: str = eqx.field(static=True)
    order: str = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    logarithmic_scaling: bool = eqx.field(static=True)
    processed_slices: int = eqx.field(static=True)
    total_slices: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    peak_batch_elements: int = eqx.field(static=True)
    complete: Array
    finite: Array
    accepted: Array
    exact: Array
    claim: str = eqx.field(static=True)


class SliceCheckpoint(StrictModule):
    scaled_sum: Array
    log_scale: Array
    finite: Array
    plan_id: str = eqx.field(static=True)
    range: SliceRange
    checkpoint_id: str = eqx.field(static=True)


class SlicedContractionResult(StrictModule):
    value: Array
    checkpoint: SliceCheckpoint
    evidence: SliceExecutionEvidence


def mixed_radix_assignments(
    dimensions: Sequence[int],
    /,
    *,
    start: int = 0,
    stop: int | None = None,
    maximum_assignments: int = 1_000_000,
) -> Array:
    """Enumerate deterministic lexicographic assignments, last label fastest."""

    dimensions_ = tuple(int(value) for value in dimensions)
    if not dimensions_ or any(value < 1 for value in dimensions_):
        raise ValueError("Mixed-radix dimensions must be a nonempty positive tuple.")
    total = prod(dimensions_)
    if total > 2_147_483_647:
        raise MemoryError("Mixed-radix index space exceeds signed int32 capacity.")
    start_ = int(start)
    stop_ = total if stop is None else int(stop)
    if not 0 <= start_ <= stop_ <= total:
        raise ValueError("Assignment range lies outside the mixed-radix extent.")
    count = stop_ - start_
    maximum = int(maximum_assignments)
    if maximum < 1:
        raise ValueError("maximum_assignments must be positive.")
    if count > maximum:
        raise MemoryError("Mixed-radix assignments exceed maximum_assignments.")
    indices = jnp.arange(start_, stop_, dtype=jnp.int32)
    divisors = tuple(prod(dimensions_[index + 1 :]) for index in range(len(dimensions_)))
    return jnp.stack(
        tuple(
            (indices // divisor) % dimension
            for divisor, dimension in zip(divisors, dimensions_, strict=True)
        ),
        axis=-1,
    ).astype(jnp.int32)


def checkpoint_slice_ranges(
    slice_count: int,
    checkpoint_size: int,
    /,
    *,
    maximum_ranges: int = 1_000_000,
) -> tuple[SliceRange, ...]:
    count = int(slice_count)
    size = int(checkpoint_size)
    maximum = int(maximum_ranges)
    if count < 1 or size < 1 or maximum < 1:
        raise ValueError(
            "slice_count, checkpoint_size, and maximum_ranges must be positive."
        )
    range_count = (count + size - 1) // size
    if range_count > maximum:
        raise MemoryError("Slice checkpoint ranges exceed maximum_ranges.")
    return tuple(
        SliceRange(start, min(start + size, count), count)
        for start in range(0, count, size)
    )


def plan_sliced_contraction(
    plan: ContractionPlan,
    labels: Sequence[str],
    /,
    *,
    batch_size: int = 1,
    resources: SlicingResourcePolicy | None = None,
) -> SlicedContractionPlan:
    if not isinstance(plan, ContractionPlan):
        raise TypeError("plan must be ContractionPlan.")
    labels_ = tuple(str(label) for label in labels)
    if not labels_ or len(set(labels_)) != len(labels_):
        raise ValueError("Slicing labels must be nonempty and unique.")
    if any(label not in plan.structure.labels for label in labels_):
        raise ValueError("Every slicing label must occur in the contraction.")
    if any(label in plan.structure.outputs for label in labels_):
        raise ValueError("Exact slicing cannot remove an explicit output label.")
    resources_ = SlicingResourcePolicy() if resources is None else resources
    if not isinstance(resources_, SlicingResourcePolicy):
        raise TypeError("resources must be SlicingResourcePolicy or None.")
    batch = int(batch_size)
    if batch < 1:
        raise ValueError("batch_size must be positive.")
    dimensions = tuple(plan.structure.dimension(label) for label in labels_)
    slice_count = prod(dimensions)
    if slice_count > resources_.maximum_slices:
        raise MemoryError("Sliced contraction exceeds maximum_slices.")
    peak_batch_elements = min(batch, slice_count) * plan.structure.output_elements
    if peak_batch_elements > resources_.maximum_batch_elements:
        raise MemoryError("Sliced contraction exceeds maximum_batch_elements.")
    checkpoint_bytes = plan.structure.output_elements * precision_itemsize(plan.dtype)
    if checkpoint_bytes > resources_.maximum_checkpoint_bytes:
        raise MemoryError("Slice checkpoint exceeds maximum_checkpoint_bytes.")

    operands = tuple(
        ContractionOperand(
            operand.operand_id,
            tuple(leg for leg in operand.legs if leg.label not in labels_),
        )
        for operand in plan.structure.operands
    )
    residual_structure = ContractionStructure(operands, plan.structure.outputs)
    residual = plan_contraction(
        residual_structure,
        precision=plan.precision,
        resources=plan.resources,
        planner=plan.planner,
        optimizer=plan.optimizer,
        dtype=plan.dtype,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "exact-sliced-contraction",
            "original": plan.plan_id,
            "residual": residual.plan_id,
            "labels": labels_,
            "dimensions": dimensions,
            "batch_size": batch,
            "resources": resources_.policy_id,
        }
    )
    return SlicedContractionPlan(
        plan,
        residual,
        labels_,
        dimensions,
        slice_count,
        batch,
        resources_,
        True,
        "exact exhaustive mixed-radix slice sum",
        plan_id,
    )


def _slice_operands(
    prepared: PreparedContraction,
    plan: SlicedContractionPlan,
    assignment: Array,
    /,
) -> tuple[Array, ...]:
    label_positions = {label: index for index, label in enumerate(plan.labels)}
    sliced = []
    for operand, specification in zip(
        prepared.operands, prepared.plan.structure.operands, strict=True
    ):
        value = operand
        axes = tuple(
            (axis, label_positions[leg.label])
            for axis, leg in enumerate(specification.legs)
            if leg.label in label_positions
        )
        for axis, assignment_index in sorted(axes, reverse=True):
            value = jnp.take(value, assignment[assignment_index], axis=axis)
        sliced.append(value)
    return tuple(sliced)


def execute_slice_assignment(
    prepared: PreparedContraction,
    plan: SlicedContractionPlan,
    assignment: Array,
    /,
) -> Array:
    if prepared.plan.plan_id != plan.original.plan_id:
        raise ValueError("Prepared contraction belongs to another sliced plan.")
    operands = _slice_operands(prepared, plan, assignment)
    converted = plan.residual.precision.contraction(operands)
    return plan.residual.precision.output(
        execute_schedule(plan.residual.schedule, converted)
    )


def _scaled_add(
    scaled_sum: Array,
    log_scale: Array,
    value: Array,
    /,
) -> tuple[Array, Array]:
    magnitude = jnp.max(jnp.abs(value))
    value_log = jnp.where(magnitude > 0, jnp.log(magnitude), -jnp.inf)
    new_scale = jnp.maximum(log_scale, value_log)
    old_factor = jnp.where(jnp.isfinite(log_scale), jnp.exp(log_scale - new_scale), 0.0)
    new_factor = jnp.where(jnp.isfinite(value_log), jnp.exp(-new_scale), 0.0)
    return scaled_sum * old_factor + value * new_factor, new_scale


def _checkpoint(
    plan: SlicedContractionPlan,
    range_: SliceRange,
    scaled_sum: Array,
    log_scale: Array,
    /,
) -> SliceCheckpoint:
    finite = jnp.all(jnp.isfinite(scaled_sum)) & (
        jnp.isfinite(log_scale) | jnp.all(scaled_sum == 0)
    )
    checkpoint_id = canonical_fingerprint(
        {
            "kind": "slice-checkpoint",
            "plan": plan.plan_id,
            "range": range_.range_id,
        }
    )
    return SliceCheckpoint(
        scaled_sum, log_scale, finite, plan.plan_id, range_, checkpoint_id
    )


def execute_sliced_contraction(
    prepared: PreparedContraction,
    plan: SlicedContractionPlan,
    /,
    *,
    slice_range: SliceRange | None = None,
    mode: str = "serial",
    logarithmic_scaling: bool = False,
) -> SlicedContractionResult:
    """Execute an exact full or checkpoint range; partial ranges are not accepted aggregates."""

    if not isinstance(prepared, PreparedContraction) or not isinstance(
        plan, SlicedContractionPlan
    ):
        raise TypeError("prepared and plan have invalid types.")
    if prepared.plan.plan_id != plan.original.plan_id:
        raise ValueError("Prepared contraction belongs to another sliced plan.")
    if mode not in ("serial", "batched"):
        raise ValueError("mode must be 'serial' or 'batched'.")
    range_ = (
        SliceRange(0, plan.slice_count, plan.slice_count)
        if slice_range is None
        else slice_range
    )
    if not isinstance(range_, SliceRange) or range_.total != plan.slice_count:
        raise ValueError("slice_range belongs to another sliced plan extent.")
    output_shape = tuple(
        plan.original.structure.dimension(label)
        for label in plan.original.structure.outputs
    )
    scaled_sum = jnp.zeros(output_shape, dtype=prepared.operands[0].dtype)
    log_scale = jnp.asarray(-jnp.inf, dtype=jnp.real(scaled_sum).dtype)

    if mode == "serial":
        assignments = mixed_radix_assignments(
            plan.dimensions,
            start=range_.start,
            stop=range_.stop,
            maximum_assignments=plan.resources.maximum_slices,
        )
        for offset in range(range_.stop - range_.start):
            value = execute_slice_assignment(prepared, plan, assignments[offset])
            if logarithmic_scaling:
                scaled_sum, log_scale = _scaled_add(scaled_sum, log_scale, value)
            else:
                scaled_sum = scaled_sum + value
        if not logarithmic_scaling:
            log_scale = jnp.asarray(0.0, dtype=jnp.real(scaled_sum).dtype)
    else:
        for start in range(range_.start, range_.stop, plan.batch_size):
            stop = min(start + plan.batch_size, range_.stop)
            assignments = mixed_radix_assignments(
                plan.dimensions,
                start=start,
                stop=stop,
                maximum_assignments=plan.batch_size,
            )
            block = jax.vmap(
                lambda assignment: execute_slice_assignment(prepared, plan, assignment)
            )(assignments)
            for offset in range(stop - start):
                if logarithmic_scaling:
                    scaled_sum, log_scale = _scaled_add(
                        scaled_sum, log_scale, block[offset]
                    )
                else:
                    scaled_sum = scaled_sum + block[offset]
        if not logarithmic_scaling:
            log_scale = jnp.asarray(0.0, dtype=jnp.real(scaled_sum).dtype)

    checkpoint = _checkpoint(plan, range_, scaled_sum, log_scale)
    value = jnp.where(
        jnp.isfinite(log_scale), scaled_sum * jnp.exp(log_scale), scaled_sum
    )
    complete = jnp.asarray(range_.start == 0 and range_.stop == plan.slice_count)
    finite = checkpoint.finite & jnp.all(jnp.isfinite(value))
    accepted = complete & finite
    replay_id = canonical_fingerprint(
        {
            "kind": "sliced-contraction-replay",
            "plan": plan.plan_id,
            "range": range_.range_id,
            "mode": mode,
            "logarithmic_scaling": bool(logarithmic_scaling),
        }
    )
    evidence = SliceExecutionEvidence(
        plan.plan_id,
        replay_id,
        range_.range_id,
        "lexicographic-last-label-fastest",
        mode,
        bool(logarithmic_scaling),
        range_.stop - range_.start,
        plan.slice_count,
        plan.batch_size,
        min(plan.batch_size if mode == "batched" else 1, range_.stop - range_.start)
        * plan.original.structure.output_elements,
        complete,
        finite,
        accepted,
        accepted,
        "exact only when the complete admitted slice range is finite",
    )
    return SlicedContractionResult(value, checkpoint, evidence)


def merge_slice_checkpoints(
    plan: SlicedContractionPlan,
    checkpoints: Sequence[SliceCheckpoint],
    /,
) -> SlicedContractionResult:
    checkpoints_ = tuple(checkpoints)
    if not checkpoints_:
        raise ValueError("At least one slice checkpoint is required.")
    expected = 0
    for checkpoint in checkpoints_:
        if checkpoint.plan_id != plan.plan_id or checkpoint.range.start != expected:
            raise ValueError(
                "Slice checkpoints must be plan-matched, contiguous, and ordered."
            )
        expected = checkpoint.range.stop
    if expected != plan.slice_count:
        raise ValueError("Slice checkpoints do not cover the full slice extent.")
    scaled = jnp.zeros_like(checkpoints_[0].scaled_sum)
    scale = jnp.asarray(-jnp.inf, dtype=checkpoints_[0].log_scale.dtype)
    for checkpoint in checkpoints_:
        new_scale = jnp.maximum(scale, checkpoint.log_scale)
        old_factor = jnp.where(jnp.isfinite(scale), jnp.exp(scale - new_scale), 0.0)
        checkpoint_factor = jnp.where(
            jnp.isfinite(checkpoint.log_scale),
            jnp.exp(checkpoint.log_scale - new_scale),
            0.0,
        )
        scaled = scaled * old_factor + checkpoint.scaled_sum * checkpoint_factor
        scale = new_scale
    merged_range = SliceRange(0, plan.slice_count, plan.slice_count)
    merged = _checkpoint(plan, merged_range, scaled, scale)
    value = jnp.where(jnp.isfinite(scale), scaled * jnp.exp(scale), scaled)
    finite = jnp.all(
        jnp.stack(tuple(checkpoint.finite for checkpoint in checkpoints_))
    ) & jnp.all(jnp.isfinite(value))
    replay_id = canonical_fingerprint(
        {
            "kind": "merged-slice-replay",
            "plan": plan.plan_id,
            "checkpoints": tuple(checkpoint.checkpoint_id for checkpoint in checkpoints_),
        }
    )
    evidence = SliceExecutionEvidence(
        plan.plan_id,
        replay_id,
        merged_range.range_id,
        "lexicographic-last-label-fastest",
        "checkpoint-merge",
        True,
        plan.slice_count,
        plan.slice_count,
        plan.batch_size,
        plan.original.structure.output_elements,
        jnp.asarray(True),
        finite,
        finite,
        finite,
        "exact exhaustive checkpoint merge when every range is finite",
    )
    return SlicedContractionResult(value, merged, evidence)


__all__ = [
    "SliceCheckpoint",
    "SliceExecutionEvidence",
    "SliceRange",
    "SlicedContractionPlan",
    "SlicedContractionResult",
    "SlicingResourcePolicy",
    "checkpoint_slice_ranges",
    "execute_slice_assignment",
    "execute_sliced_contraction",
    "merge_slice_checkpoints",
    "mixed_radix_assignments",
    "plan_sliced_contraction",
]
