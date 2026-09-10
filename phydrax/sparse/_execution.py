#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._numerics._compensated import two_sum
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._key_groups import KeyGroupPlan, KeyGroupState
from ._relation import EdgeRelation, RowRelation, SparseRelation


RelationAccumulation = Literal["fast", "deterministic", "compensated"]
RelationReduction = Literal["sum", "mean", "min", "max"]
RelationOutput = Literal["compact", "dense"]


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
    row_count = int(order.shape[0])
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
        if reduction not in ("sum", "mean", "min", "max"):
            raise ValueError(f"Unsupported relation reduction {reduction!r}.")
        if accumulation not in ("fast", "deterministic", "compensated"):
            raise ValueError(f"Unsupported relation accumulation {accumulation!r}.")
        if output not in ("compact", "dense"):
            raise ValueError(f"Unsupported relation output {output!r}.")
        if accumulation == "compensated" and reduction not in ("sum", "mean"):
            raise ValueError("Compensated accumulation supports sum and mean only.")

        leaves, treedef = jax.tree_util.tree_flatten(route_values)
        arrays = [jnp.asarray(leaf) for leaf in leaves]
        for value in arrays:
            if value.ndim == 0 or value.shape[0] != self.route_capacity:
                raise ValueError(
                    "Every route value leaf must begin with route capacity "
                    f"{self.route_capacity}; got {value.shape}."
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
            compact = _sum_segments(
                safe_values,
                safe_slots,
                sorted_valid,
                self.compact_target_capacity,
                accumulation=accumulation,
            )
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
) -> Array:
    trailing = values.shape[1:]
    if accumulation == "fast":
        return jax.ops.segment_sum(
            values,
            group_slots,
            num_segments=group_capacity,
            indices_are_sorted=True,
        )

    initial = jnp.zeros((group_capacity,) + trailing, dtype=values.dtype)
    if accumulation == "deterministic":

        def add_one(index: int, total: Array) -> Array:
            slot = group_slots[index]
            value = jnp.where(valid[index], values[index], jnp.zeros((), values.dtype))
            return total.at[slot].add(value)

        return jax.lax.fori_loop(0, values.shape[0], add_one, initial)

    correction = jnp.zeros_like(initial)

    def add_compensated(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        total, residual = carry
        slot = group_slots[index]
        value = jnp.where(valid[index], values[index], jnp.zeros((), values.dtype))
        next_total, error = two_sum(total[slot], value)
        return total.at[slot].set(next_total), residual.at[slot].add(error)

    total, residual = jax.lax.fori_loop(
        0, values.shape[0], add_compensated, (initial, correction)
    )
    return total + residual


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
    "RelationAccumulation",
    "RelationExecutionPlan",
    "RelationExecutionState",
    "RelationOutput",
    "RelationReduction",
    "RelationReductionEvidence",
]
