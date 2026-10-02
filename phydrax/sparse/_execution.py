#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._numerics._compensated import two_sum
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import AnyShape, Bool, Inexact, Integer, parse
from ._key_groups import KeyGroupPlan, KeyGroupState
from ._relation import EdgeRelation, RowRelation, SparseRelation


RelationAccumulation: TypeAlias = Literal["fast", "deterministic", "compensated"]
RelationReduction: TypeAlias = Literal["sum", "mean", "min", "max"]
RelationOutput: TypeAlias = Literal["compact", "dense"]


class KeyGroupAccumulation(NonTrainableState, StrictModule):
    """Seedable compact sums with their uncollapsed compensation."""

    __strict_contract__ = True

    high: Inexact[AnyShape] | Integer[AnyShape]
    correction: Inexact[AnyShape] | Integer[AnyShape]

    def __init__(self, high: ArrayLike, correction: ArrayLike) -> None:
        high_array = parse(
            jnp.asarray(high), Inexact[AnyShape] | Integer[AnyShape], "high"
        )
        correction_array = parse(
            jnp.asarray(correction), Inexact[AnyShape] | Integer[AnyShape], "correction"
        )
        if high_array.shape != correction_array.shape:
            raise ValueError("high and correction must have the same shape.")
        if high_array.dtype != correction_array.dtype:
            raise TypeError("high and correction must have the same dtype.")
        self.high = high_array
        self.correction = correction_array

    @property
    def value(self) -> Array:
        return self.high + self.correction


class KeyGroupReductionEvidence(NonTrainableState, StrictModule):
    """Per-case numerical and grouping admission for a compact sum."""

    __strict_contract__ = True

    finite: Bool[AnyShape]
    successful: Bool[AnyShape]


def reduce_key_groups(
    groups: KeyGroupState,
    values: ArrayLike,
    *,
    accumulation: RelationAccumulation = "compensated",
    initial: KeyGroupAccumulation | None = None,
    value_valid: ArrayLike | None = None,
) -> tuple[KeyGroupAccumulation, KeyGroupReductionEvidence]:
    """Add canonical item events to seeded compact sums without chunk subtotals."""
    accumulation = parse(accumulation, RelationAccumulation, "accumulation")
    array = parse(jnp.asarray(values), Inexact[AnyShape] | Integer[AnyShape], "values")
    item_shape = groups.plan.case_shape + (groups.plan.item_capacity,)
    if array.shape[: len(item_shape)] != item_shape:
        raise ValueError(
            f"values must begin with item shape {item_shape}; got {array.shape}."
        )
    trailing = array.shape[len(item_shape) :]
    output_shape = groups.plan.case_shape + (groups.plan.group_capacity,) + trailing
    if initial is None:
        high = jnp.zeros(output_shape, dtype=array.dtype)
        correction = jnp.zeros_like(high)
    else:
        if not isinstance(initial, KeyGroupAccumulation):
            raise TypeError("initial must be a KeyGroupAccumulation.")
        if initial.high.shape != output_shape:
            raise ValueError(f"initial must have shape {output_shape}.")
        if initial.high.dtype != array.dtype:
            raise TypeError("initial and values must have the same dtype.")
        high, correction = initial.high, initial.correction
    if value_valid is None:
        active = groups.item_valid
    else:
        active = jnp.asarray(value_valid, dtype=jnp.bool_)
        if active.shape != item_shape:
            raise ValueError(f"value_valid must have shape {item_shape}.")
        active = active & groups.item_valid
    batch_size = 1
    for size in groups.plan.case_shape:
        batch_size *= size
    flat_values = array.reshape((batch_size, groups.plan.item_capacity) + trailing)
    flat_order = groups.storage_to_logical.reshape(
        (batch_size, groups.plan.item_capacity)
    )
    flat_valid = groups.sorted_item_valid.reshape((batch_size, groups.plan.item_capacity))
    flat_active = active.reshape((batch_size, groups.plan.item_capacity))
    flat_slots = groups.item_group_slots.reshape((batch_size, groups.plan.item_capacity))
    seed_shape = (batch_size, groups.plan.group_capacity) + trailing

    def reduce_case(
        case_values: Array,
        order: Array,
        valid: Array,
        value_active: Array,
        slots: Array,
        seed_high: Array,
        seed_correction: Array,
    ) -> tuple[Array, Array, Array]:
        enabled = valid & value_active[order]
        usable = (
            enabled & (slots[order] >= 0) & (slots[order] < groups.plan.group_capacity)
        )
        mask_shape = (groups.plan.item_capacity,) + (1,) * len(trailing)
        safe_values = jnp.where(
            enabled.reshape(mask_shape), case_values[order], jnp.zeros((), array.dtype)
        )
        next_high, next_correction = _sum_segments(
            safe_values,
            jnp.where(usable, slots[order], 0),
            usable,
            groups.plan.group_capacity,
            accumulation=accumulation,
            initial=(seed_high, seed_correction),
        )
        finite = (
            jnp.all(jnp.isfinite(safe_values))
            & jnp.all(jnp.isfinite(next_high))
            & jnp.all(jnp.isfinite(next_correction))
            & jnp.all(jnp.isfinite(next_high + next_correction))
        )
        return next_high, next_correction, finite

    result_high, result_correction, finite = jax.vmap(reduce_case)(
        flat_values,
        flat_order,
        flat_valid,
        flat_active,
        flat_slots,
        high.reshape(seed_shape),
        correction.reshape(seed_shape),
    )
    finite = finite.reshape(groups.plan.case_shape)
    return (
        KeyGroupAccumulation(
            result_high.reshape(output_shape), result_correction.reshape(output_shape)
        ),
        KeyGroupReductionEvidence(
            finite=finite, successful=groups.evidence.successful & finite
        ),
    )


def canonical_row_route_ids(
    stable_row_order: ArrayLike,
    route_width: int,
    /,
) -> Array:
    """Return route IDs ordered by stable row rank and then route slot."""
    order = jnp.asarray(stable_row_order, dtype=jnp.int32)
    if order.ndim != 1:
        raise ValueError("stable_row_order must be a rank-1 permutation.")
    width = int(route_width)
    if width <= 0:
        raise ValueError("route_width must be positive.")
    row_count = order.shape[0]
    inverse = (
        jnp.zeros((row_count,), dtype=jnp.int32)
        .at[order]
        .set(jnp.arange(row_count, dtype=jnp.int32))
    )
    return (
        inverse[:, None] * width + jnp.arange(width, dtype=jnp.int32)[None, :]
    ).reshape((-1,))


class RelationReductionEvidence(NonTrainableState, StrictModule):
    """Runtime evidence for one grouped route reduction."""

    valid_routes: Array
    active_targets: Array
    maximum_target_contention: Array
    finite: Array
    successful: Array
    accumulation: RelationAccumulation = eqx.field(static=True)
    reduction: RelationReduction = eqx.field(static=True)


class RelationExecutionPlan(StrictModule):
    """Prepare canonical target-grouped execution for a sparse relation."""

    maximum_active_targets: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, maximum_active_targets: int | None = None) -> None:
        if maximum_active_targets is not None and maximum_active_targets <= 0:
            raise ValueError("maximum_active_targets must be positive when provided.")
        self.maximum_active_targets = (
            None if maximum_active_targets is None else int(maximum_active_targets)
        )
        self.plan_id = canonical_fingerprint(
            {
                "type": "relation-execution-plan",
                "maximum_active_targets": self.maximum_active_targets,
            }
        )

    def prepare(
        self,
        relation: SparseRelation,
        *,
        stable_route_ids: ArrayLike | None = None,
    ) -> RelationExecutionState:
        """Prepare target groups without changing the relation's route semantics."""
        if isinstance(relation, RowRelation):
            edge = relation.as_edge_relation()
            output_shape = relation.output_shape
        elif isinstance(relation, EdgeRelation):
            edge = relation
            output_shape = relation.output_shape
        else:
            raise TypeError("relation must be an EdgeRelation or RowRelation.")
        group_capacity = min(edge.capacity, edge.target_size)
        if self.maximum_active_targets is not None:
            group_capacity = min(group_capacity, self.maximum_active_targets)
        group_capacity = max(group_capacity, 1)
        grouping = KeyGroupPlan(
            edge.capacity,
            group_capacity,
            max(edge.target_size - 1, 0),
        ).build(
            edge.target_indices,
            edge.valid,
            stable_ids=stable_route_ids,
        )
        execution_id = canonical_fingerprint(
            {
                "type": "relation-execution",
                "plan_id": self.plan_id,
                "source_size": edge.source_size,
                "target_size": edge.target_size,
                "route_capacity": edge.capacity,
                "output_shape": output_shape,
            }
        )
        return RelationExecutionState(
            plan=self,
            relation=edge,
            groups=grouping,
            output_shape=output_shape,
            execution_id=execution_id,
        )


class RelationExecutionState(NonTrainableState, StrictModule):
    """Target-grouped route order and reduction operations."""

    plan: RelationExecutionPlan
    relation: EdgeRelation
    groups: KeyGroupState
    output_shape: tuple[int, ...] = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)

    @property
    def route_capacity(self) -> int:
        return self.relation.capacity

    @property
    def compact_target_capacity(self) -> int:
        return self.groups.plan.group_capacity

    def reduce(
        self,
        route_values: PyTree[ArrayLike],
        *,
        reduction: RelationReduction = "sum",
        accumulation: RelationAccumulation = "fast",
        output: RelationOutput = "dense",
    ) -> tuple[PyTree[Array], RelationReductionEvidence]:
        """Reduce route-aligned values to compact groups or the dense target space."""
        reduction = parse(reduction, RelationReduction, "reduction")
        accumulation = parse(accumulation, RelationAccumulation, "accumulation")
        output = parse(output, RelationOutput, "output")
        if accumulation == "compensated" and reduction not in ("sum", "mean"):
            raise ValueError("Compensated accumulation supports sum and mean only.")

        leaves, treedef = jax.tree_util.tree_flatten(route_values)
        arrays = [jnp.asarray(leaf) for leaf in leaves]
        for value in arrays:
            if value.ndim == 0 or value.shape[0] != self.route_capacity:
                raise ValueError(
                    f"Every route value leaf must begin with route capacity {self.route_capacity}; got {value.shape}."
                )
        reduced = [
            self._reduce_leaf(
                value,
                reduction=reduction,
                accumulation=accumulation,
                output=output,
            )
            for value in arrays
        ]
        result = jax.tree_util.tree_unflatten(treedef, reduced)
        finite = jnp.asarray(True)
        for value in reduced:
            finite = finite & jnp.all(jnp.isfinite(value))
        evidence = RelationReductionEvidence(
            valid_routes=self.groups.evidence.active_items,
            active_targets=self.groups.evidence.required_groups,
            maximum_target_contention=self.groups.evidence.maximum_group_size,
            finite=finite,
            successful=self.groups.evidence.successful & finite,
            accumulation=accumulation,
            reduction=reduction,
        )
        return result, evidence

    def _reduce_leaf(
        self,
        values: Array,
        *,
        reduction: RelationReduction,
        accumulation: RelationAccumulation,
        output: RelationOutput,
    ) -> Array:
        order = self.groups.storage_to_logical
        sorted_values = values[order]
        sorted_valid = self.groups.sorted_item_valid
        group_slots = self.groups.item_group_slots[order]
        trailing = values.shape[1:]
        valid_shape = (self.route_capacity,) + (1,) * len(trailing)
        safe_values = jnp.where(
            sorted_valid.reshape(valid_shape), sorted_values, jnp.zeros((), values.dtype)
        )
        safe_slots = jnp.where(sorted_valid, group_slots, 0)

        if reduction in ("min", "max") and jnp.issubdtype(
            values.dtype, jnp.complexfloating
        ):
            raise TypeError(
                "min and max relation reductions do not support complex values."
            )

        if reduction in ("sum", "mean"):
            high, correction = _sum_segments(
                safe_values,
                safe_slots,
                sorted_valid,
                self.compact_target_capacity,
                accumulation=accumulation,
            )
            compact = high + correction if accumulation == "compensated" else high
            if reduction == "mean":
                counts = self.groups.group_counts.astype(values.dtype)
                count_shape = counts.shape + (1,) * len(trailing)
                compact = jnp.where(
                    self.groups.group_active.reshape(count_shape),
                    compact / jnp.maximum(counts, 1).reshape(count_shape),
                    jnp.zeros((), compact.dtype),
                )
        else:
            compact = _extreme_segments(
                safe_values,
                safe_slots,
                sorted_valid,
                self.compact_target_capacity,
                reduction=reduction,
            )
        active_shape = self.groups.group_active.shape + (1,) * len(trailing)
        usable = self.groups.group_active & self.groups.evidence.successful
        compact = jnp.where(
            usable.reshape(active_shape),
            compact,
            jnp.zeros((), compact.dtype),
        )
        if output == "compact":
            return compact
        dense = jnp.zeros((self.relation.target_size,) + trailing, dtype=compact.dtype)
        safe_targets = jnp.where(usable, self.groups.group_keys, 0)
        return (
            dense.at[safe_targets]
            .add(
                jnp.where(
                    usable.reshape(active_shape),
                    compact,
                    jnp.zeros((), compact.dtype),
                )
            )
            .reshape(self.output_shape + trailing)
        )


def _sum_segments(
    values: Array,
    group_slots: Array,
    valid: Array,
    group_capacity: int,
    *,
    accumulation: RelationAccumulation,
    initial: tuple[Array, Array] | None = None,
) -> tuple[Array, Array]:
    trailing = values.shape[1:]
    if initial is None:
        high = jnp.zeros((group_capacity,) + trailing, dtype=values.dtype)
        correction = jnp.zeros_like(high)
    else:
        high, correction = initial
    if values.shape[0] == 0:
        return high, correction
    mask_shape = valid.shape + (1,) * len(trailing)
    safe_values = jnp.where(
        valid.reshape(mask_shape), values, jnp.zeros((), values.dtype)
    )
    match accumulation:
        case "fast":
            subtotal = jax.ops.segment_sum(
                safe_values,
                group_slots,
                num_segments=group_capacity,
                indices_are_sorted=True,
            )
            return (
                subtotal
                if initial is None
                else jnp.where(subtotal != 0, high + subtotal, high),
                correction,
            )
        case "deterministic":

            def add_one(index: int, total: Array) -> Array:
                slot = group_slots[index]
                value = safe_values[index]
                previous = total[slot]
                next_total = jnp.where(value != 0, previous + value, previous)
                return total.at[slot].set(next_total)

            return (
                jax.lax.fori_loop(0, values.shape[0], add_one, high),
                correction,
            )
        case "compensated":

            def add_compensated(
                index: int, carry: tuple[Array, Array]
            ) -> tuple[Array, Array]:
                total, residual = carry
                slot = group_slots[index]
                value = safe_values[index]
                previous = total[slot]
                previous_residual = residual[slot]
                next_total, error = two_sum(previous, value)
                # Zero components, padding and key-retention items are true no-ops.
                changed = value != 0
                return (
                    total.at[slot].set(jnp.where(changed, next_total, previous)),
                    residual.at[slot].set(
                        jnp.where(changed, previous_residual + error, previous_residual)
                    ),
                )

            return jax.lax.fori_loop(
                0, values.shape[0], add_compensated, (high, correction)
            )
        case _:
            raise ValueError(f"Invalid accumulation: {accumulation!r}.")


def _extreme_segments(
    values: Array,
    group_slots: Array,
    valid: Array,
    group_capacity: int,
    *,
    reduction: Literal["min", "max"],
) -> Array:
    if jnp.issubdtype(values.dtype, jnp.floating):
        identity = jnp.inf if reduction == "min" else -jnp.inf
    elif jnp.issubdtype(values.dtype, jnp.integer):
        info = jnp.iinfo(values.dtype)
        identity = info.max if reduction == "min" else info.min
    else:
        raise TypeError("min and max relation reductions require real numeric values.")
    safe = jnp.where(
        valid.reshape((valid.shape[0],) + (1,) * (values.ndim - 1)),
        values,
        jnp.asarray(identity, dtype=values.dtype),
    )
    operation = jax.ops.segment_min if reduction == "min" else jax.ops.segment_max
    return operation(
        safe,
        group_slots,
        num_segments=group_capacity,
        indices_are_sorted=True,
    )


__all__ = [
    "canonical_row_route_ids",
    "KeyGroupAccumulation",
    "KeyGroupReductionEvidence",
    "reduce_key_groups",
    "RelationAccumulation",
    "RelationExecutionPlan",
    "RelationExecutionState",
    "RelationOutput",
    "RelationReduction",
    "RelationReductionEvidence",
]
