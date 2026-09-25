#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ._execution_pool import (
    PoolExecutionSignature,
    semantic_task_indices,
    semantic_task_keys,
)
from ._execution_runtime import (
    bind_execution_group,
    ExecutionGroup,
    ExecutionRuntime,
    partition_execution_group_specs,
)
from ._execution_tasks import HostTaskExecutor, InlineTaskExecutor
from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._identity import NumericRevision
from ._sampling._addressing import SampleAddress
from ._strict import StrictModule
from ._trainable import LaneLayout, NonTrainableState


ExecutionWorksetMode = Literal["serial", "vmap", "filter_vmap"]
# Semantic RNG family of workset items: an item's index is this address extended
# by its semantic ID, and its key folds that index and its restart counter.
_WORKSET_ITEM_ADDRESS = SampleAddress("execution", "workset", role="item")


def _item_tree(values: PyTree[ArrayLike], item_count: int, /) -> PyTree[Array]:
    arrays = jax.tree_util.tree_map(jnp.asarray, values)
    leaves = jax.tree_util.tree_leaves(arrays)
    if not leaves:
        raise ValueError("Execution workset values must contain at least one array leaf.")
    if any(value.ndim < 1 or value.shape[0] != item_count for value in leaves):
        raise ValueError(
            "Every execution workset value leaf must have one leading entry per item."
        )
    if any(
        not (
            jnp.issubdtype(value.dtype, jnp.number)
            or jnp.issubdtype(value.dtype, jnp.bool_)
        )
        for value in leaves
    ):
        raise TypeError("Execution workset values must be numeric or boolean arrays.")
    return arrays


def _expand_mask(mask: Array, value: Array, /) -> Array:
    return mask.reshape(mask.shape + (1,) * (value.ndim - mask.ndim))


def _mask_tree(values: PyTree[Array], mask: Array, /) -> PyTree[Array]:
    return jax.tree_util.tree_map(
        lambda value: jnp.where(_expand_mask(mask, value), value, jnp.zeros_like(value)),
        values,
    )


def _tree_finite(values: PyTree[Array], /) -> Array:
    leaves = jax.tree_util.tree_leaves(values)
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in leaves)))


def _item_layout(values: Any, layout: LaneLayout | None, /) -> LaneLayout:
    if layout is None:
        if not any(eqx.is_array(leaf) for leaf in jax.tree_util.tree_leaves(values)):
            raise ValueError("Filtered workset values require at least one array leaf.")
        return LaneLayout.from_predicate(values, lambda _: True, kind="item")
    if not isinstance(layout, LaneLayout) or layout.kind != "item":
        raise TypeError("layout must be a LaneLayout of kind 'item'.")
    return layout


class ExecutionWorksetPlan(StrictModule, NonTrainableState):
    """Canonical item order and fixed per-bucket capacity for one execution family."""

    semantic_ids: tuple[str, ...] = eqx.field(static=True)
    signatures: tuple[PoolExecutionSignature, ...]
    semantic_rng_indices: Array
    bucket_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        semantic_ids: tuple[str, ...],
        signatures: tuple[PoolExecutionSignature, ...],
        /,
        *,
        bucket_capacity: int = 8,
    ):
        identifiers = tuple(str(value).strip() for value in semantic_ids)
        signature_values = tuple(signatures)
        capacity = int(bucket_capacity)
        if not identifiers:
            raise ValueError("An execution workset plan requires at least one item.")
        if any(not value for value in identifiers) or len(set(identifiers)) != len(
            identifiers
        ):
            raise ValueError(
                "Execution workset semantic IDs must be non-empty and unique."
            )
        if len(signature_values) != len(identifiers) or not all(
            isinstance(value, PoolExecutionSignature) for value in signature_values
        ):
            raise TypeError(
                "signatures must contain one PoolExecutionSignature per semantic ID."
            )
        if capacity < 1 or capacity > 64:
            raise ValueError(
                "bucket_capacity must lie in the fixed modest range [1, 64]."
            )
        ordered = sorted(
            zip(identifiers, signature_values, strict=True), key=lambda x: x[0]
        )
        canonical_ids = tuple(value[0] for value in ordered)
        canonical_signatures = tuple(value[1] for value in ordered)
        rng_indices = semantic_task_indices(_WORKSET_ITEM_ADDRESS, canonical_ids)
        self.semantic_ids = canonical_ids
        self.signatures = canonical_signatures
        self.semantic_rng_indices = jnp.asarray(rng_indices, dtype=jnp.uint32)
        self.bucket_capacity = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-shape-execution-workset-plan",
                "items": [
                    {
                        "semantic_id": identifier,
                        "signature": signature.signature_id,
                        "rng_index": rng_index,
                    }
                    for identifier, signature, rng_index in zip(
                        canonical_ids,
                        canonical_signatures,
                        rng_indices,
                        strict=True,
                    )
                ],
                "bucket_capacity": capacity,
            }
        )

    @property
    def item_count(self) -> int:
        return len(self.semantic_ids)

    def prepare(self) -> PreparedExecutionWorksets:
        """Lower canonical items into deterministic homogeneous padded buckets."""
        return PreparedExecutionWorksets(self)


class PreparedExecutionWorksets(StrictModule, NonTrainableState):
    """Fixed-shape homogeneous buckets and their reversible item permutation."""

    plan: ExecutionWorksetPlan
    bucket_signatures: tuple[PoolExecutionSignature, ...]
    item_indices: Array
    valid_mask: Array
    bucket_rng_indices: Array
    item_bucket: Array
    item_slot: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ExecutionWorksetPlan, /):
        if not isinstance(plan, ExecutionWorksetPlan):
            raise TypeError("plan must be an ExecutionWorksetPlan.")
        grouped: dict[str, list[int]] = {}
        representatives: dict[str, PoolExecutionSignature] = {}
        for item, signature in enumerate(plan.signatures):
            grouped.setdefault(signature.signature_id, []).append(item)
            representatives.setdefault(signature.signature_id, signature)
        bucket_rows: list[list[int]] = []
        bucket_signatures: list[PoolExecutionSignature] = []
        for signature_id in sorted(grouped):
            items = grouped[signature_id]
            for start in range(0, len(items), plan.bucket_capacity):
                active = items[start : start + plan.bucket_capacity]
                bucket_rows.append(
                    active + [active[0]] * (plan.bucket_capacity - len(active))
                )
                bucket_signatures.append(representatives[signature_id])
        indices = np.asarray(bucket_rows, dtype=np.int32)
        valid = np.zeros_like(indices, dtype=np.bool_)
        item_bucket = np.empty((plan.item_count,), dtype=np.int32)
        item_slot = np.empty((plan.item_count,), dtype=np.int32)
        for bucket, signature in enumerate(bucket_signatures):
            active_count = min(
                plan.bucket_capacity,
                len(grouped[signature.signature_id])
                - sum(
                    plan.bucket_capacity
                    for earlier in bucket_signatures[:bucket]
                    if earlier.signature_id == signature.signature_id
                ),
            )
            valid[bucket, :active_count] = True
            for slot in range(active_count):
                item = int(indices[bucket, slot])
                item_bucket[item] = bucket
                item_slot[item] = slot
        rng_indices = np.asarray(plan.semantic_rng_indices)[indices]
        self.plan = plan
        self.bucket_signatures = tuple(bucket_signatures)
        self.item_indices = jnp.asarray(indices)
        self.valid_mask = jnp.asarray(valid)
        self.bucket_rng_indices = jnp.asarray(rng_indices, dtype=jnp.uint32)
        self.item_bucket = jnp.asarray(item_bucket)
        self.item_slot = jnp.asarray(item_slot)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fixed-shape-execution-worksets",
                "plan": plan.plan_id,
                "bucket_signatures": [value.signature_id for value in bucket_signatures],
                "item_indices": indices.tolist(),
                "valid_mask": valid.tolist(),
            }
        )

    @property
    def bucket_count(self) -> int:
        return len(self.bucket_signatures)

    @property
    def bucket_capacity(self) -> int:
        return self.plan.bucket_capacity

    @property
    def item_count(self) -> int:
        return self.plan.item_count

    def gather(self, values: PyTree[ArrayLike], /) -> PyTree[Array]:
        """Gather item-major values with safe duplicate values in padded lanes."""
        arrays = _item_tree(values, self.item_count)
        return jax.tree_util.tree_map(lambda value: value[self.item_indices], arrays)

    def gather_filtered(self, values: Any, /, *, layout: LaneLayout | None = None) -> Any:
        """Gather the item lane of the declared leaves; other leaves are shared.

        `layout` declares which leaves carry the leading item axis, independently
        of array roles (FIXED data may be mapped). By default every array leaf does.
        """
        lanes = _item_layout(values, layout)
        if lanes.lane_size(values) != self.item_count:
            raise ValueError(
                "Every mapped workset array leaf must have one leading item axis."
            )
        return lanes.take(values, self.item_indices)

    def scatter(self, bucket_values: PyTree[ArrayLike], /) -> PyTree[Array]:
        """Scatter every valid bucket lane back to canonical item order."""
        values = jax.tree_util.tree_map(jnp.asarray, bucket_values)
        leaves = jax.tree_util.tree_leaves(values)
        expected = (self.bucket_count, self.bucket_capacity)
        if not leaves:
            raise ValueError("Bucket values must contain at least one array leaf.")
        if any(value.ndim < 2 or value.shape[:2] != expected for value in leaves):
            raise ValueError(
                "Every bucket value leaf must begin with bucket_count and bucket_capacity."
            )
        masked = _mask_tree(values, self.valid_mask)
        flat_indices = self.item_indices.reshape((-1,))

        def scatter_leaf(value: Array) -> Array:
            flat = value.reshape((flat_indices.size,) + value.shape[2:])
            destination = jnp.zeros(
                (self.item_count + 1,) + value.shape[2:], dtype=value.dtype
            )
            safe = jnp.where(
                self.valid_mask.reshape((-1,)), flat_indices, self.item_count
            )
            return destination.at[safe].set(flat)[: self.item_count]

        return jax.tree_util.tree_map(scatter_leaf, masked)

    def semantic_keys(
        self,
        root_key: Array,
        rng_counters: ArrayLike,
        /,
    ) -> Array:
        """Derive keys from semantic item identity and explicit restartable counters."""
        counters = jnp.asarray(rng_counters, dtype=jnp.uint32)
        if counters.shape != (self.item_count,):
            raise ValueError("rng_counters must contain one scalar counter per item.")
        root = jnp.asarray(root_key)
        if jax.random.key_data(root).shape != (2,):
            raise ValueError("root_key must be one JAX PRNG key.")
        bucket_counters = counters[self.item_indices].reshape((-1,))
        keys = semantic_task_keys(
            root,
            _WORKSET_ITEM_ADDRESS,
            self.bucket_rng_indices.reshape((-1,)),
            bucket_counters,
        )
        return keys.reshape(self.item_indices.shape + keys.shape[1:])


class ExecutionWorksetEvidence(StrictModule, NonTrainableState):
    """Runtime evidence for one serial or vectorized workset evaluation."""

    finite: Array
    active_item_count: Array
    padded_lane_count: Array
    exact_coverage: Array
    mode: ExecutionWorksetMode = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.finite & self.exact_coverage


class ExecutionWorksetEvaluation(StrictModule):
    values: Any
    next_rng_counters: Array
    evidence: ExecutionWorksetEvidence


def _stack_trees(values: list[PyTree[Array]], /) -> PyTree[Array]:
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *values)


def _concatenate_trees(values: list[PyTree[Array]], /) -> PyTree[Array]:
    return jax.tree_util.tree_map(
        lambda *leaves: jnp.concatenate(leaves, axis=0), *values
    )


def _evaluate_execution_worksets(
    prepared: PreparedExecutionWorksets,
    operation: Callable[
        [PoolExecutionSignature, PyTree[Array], Array, Array], PyTree[Array]
    ],
    values: Any,
    root_key: Array,
    rng_counters: ArrayLike,
    /,
    *,
    mode: ExecutionWorksetMode,
    layout: LaneLayout | None = None,
) -> ExecutionWorksetEvaluation:
    if not isinstance(prepared, PreparedExecutionWorksets):
        raise TypeError("prepared must be PreparedExecutionWorksets.")
    if not callable(operation):
        raise TypeError("operation must be callable.")
    if mode == "filter_vmap":
        lanes = _item_layout(values, layout)
        gathered = prepared.gather_filtered(values, layout=lanes)
        lane_axes = (lanes.in_axes(gathered), 0, 0)
    else:
        gathered = prepared.gather(values)
    counters = jnp.asarray(rng_counters, dtype=jnp.uint32)
    keys = prepared.semantic_keys(root_key, counters)
    counter_overflow = jnp.any(counters == jnp.iinfo(jnp.uint32).max)
    advanced_counters = counters + jnp.asarray(1, dtype=jnp.uint32)
    if mode in ("vmap", "filter_vmap"):
        signature_group_outputs: list[PyTree[Array]] = []
        start = 0
        while start < prepared.bucket_count:
            signature = prepared.bucket_signatures[start]
            stop = start + 1
            while (
                stop < prepared.bucket_count
                and prepared.bucket_signatures[stop].signature_id
                == signature.signature_id
            ):
                stop += 1
            group_keys = keys[start:stop]
            group_indices = prepared.bucket_rng_indices[start:stop]

            def lane_operation(item, key, semantic_index, signature=signature):
                return operation(signature, item, key, semantic_index)

            if mode == "filter_vmap":
                group_values = lanes.take(gathered, slice(start, stop))
                mapper = eqx.filter_vmap(
                    eqx.filter_vmap(lane_operation, in_axes=lane_axes),
                    in_axes=lane_axes,
                )
            else:
                group_values = jax.tree_util.tree_map(
                    lambda value, start=start, stop=stop: value[start:stop],
                    gathered,
                )
                mapper = jax.vmap(jax.vmap(lane_operation))
            signature_group_outputs.append(
                mapper(group_values, group_keys, group_indices)
            )
            start = stop
        buckets = _concatenate_trees(signature_group_outputs)
    else:
        bucket_outputs: list[PyTree[Array]] = []
        for bucket, signature in enumerate(prepared.bucket_signatures):
            bucket_values = jax.tree_util.tree_map(
                lambda value, bucket=bucket: value[bucket],
                gathered,
            )
            bucket_keys = keys[bucket]
            bucket_indices = prepared.bucket_rng_indices[bucket]
            lanes = [
                operation(
                    signature,
                    jax.tree_util.tree_map(
                        lambda value, lane=lane: value[lane],
                        bucket_values,
                    ),
                    bucket_keys[lane],
                    bucket_indices[lane],
                )
                for lane in range(prepared.bucket_capacity)
            ]
            bucket_outputs.append(_stack_trees(lanes))
        buckets = _stack_trees(bucket_outputs)
    masked = _mask_tree(buckets, prepared.valid_mask)
    result = prepared.scatter(masked)
    result = jax.tree_util.tree_map(
        lambda value: eqx.error_if(
            value,
            counter_overflow,
            "Execution workset RNG counter overflow would break semantic key identity.",
        ),
        result,
    )
    # Preparation proves this invariant once on the host; runtime does not
    # re-materialize device arrays merely to restate it.
    exact_coverage = jnp.asarray(True)
    evidence = ExecutionWorksetEvidence(
        _tree_finite(result),
        jnp.asarray(prepared.item_count, dtype=jnp.int32),
        jnp.asarray(
            prepared.bucket_count * prepared.bucket_capacity - prepared.item_count,
            dtype=jnp.int32,
        ),
        exact_coverage,
        mode,
        prepared.prepared_id,
    )
    next_counters = jnp.where(evidence.successful, advanced_counters, counters)
    next_counters = eqx.error_if(
        next_counters,
        counter_overflow,
        "Execution workset RNG counter overflow would break semantic key identity.",
    )
    return ExecutionWorksetEvaluation(result, next_counters, evidence)


def evaluate_execution_worksets_serial(
    prepared: PreparedExecutionWorksets,
    operation: Callable[
        [PoolExecutionSignature, PyTree[Array], Array, Array], PyTree[Array]
    ],
    values: PyTree[ArrayLike],
    root_key: Array,
    rng_counters: ArrayLike,
    /,
) -> ExecutionWorksetEvaluation:
    """Evaluate each fixed lane serially using semantic, restartable RNG keys."""
    return _evaluate_execution_worksets(
        prepared,
        operation,
        values,
        root_key,
        rng_counters,
        mode="serial",
    )


def evaluate_execution_worksets_vmap(
    prepared: PreparedExecutionWorksets,
    operation: Callable[
        [PoolExecutionSignature, PyTree[Array], Array, Array], PyTree[Array]
    ],
    values: PyTree[ArrayLike],
    root_key: Array,
    rng_counters: ArrayLike,
    /,
) -> ExecutionWorksetEvaluation:
    """Evaluate fixed lanes with ``jax.vmap`` and scatter to canonical item order."""
    return _evaluate_execution_worksets(
        prepared,
        operation,
        values,
        root_key,
        rng_counters,
        mode="vmap",
    )


def evaluate_execution_worksets_filter_vmap(
    prepared: PreparedExecutionWorksets,
    operation: Callable[[PoolExecutionSignature, Any, Array, Array], PyTree[Array]],
    values: Any,
    root_key: Array,
    rng_counters: ArrayLike,
    /,
    *,
    layout: LaneLayout | None = None,
) -> ExecutionWorksetEvaluation:
    """Map homogeneous Equinox PyTrees over their declared item lane.

    `layout` (a `LaneLayout` of kind ``"item"``) declares which leaves carry the
    leading item axis; every other leaf, including static configuration and shared
    arrays, is broadcast within each signature bucket. Lanes are independent of
    array roles, so FIXED data may be mapped. By default every array leaf is
    mapped.
    """
    return _evaluate_execution_worksets(
        prepared,
        operation,
        values,
        root_key,
        rng_counters,
        mode="filter_vmap",
        layout=layout,
    )


class ExecutionWorksetCheckpoint(StrictModule, NonTrainableState):
    """Content-addressed host checkpoint for canonical item state and RNG counters.

    `numeric_revisions` are the `NumericRevision`s of the dynamic numeric content
    (for example learned weights) bound into the evaluated items; `()` declares
    that no such content was bound. Their sorted revision IDs enter the
    checkpoint identity, and restoring requires the same bound revisions.
    """

    state: Any
    rng_counters: Array
    prepared_id: str = eqx.field(static=True)
    semantic_ids: tuple[str, ...] = eqx.field(static=True)
    numeric_revision_ids: tuple[str, ...] = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedExecutionWorksets,
        state: PyTree[ArrayLike],
        rng_counters: ArrayLike,
        /,
        *,
        numeric_revisions: Sequence[NumericRevision],
    ):
        if not isinstance(prepared, PreparedExecutionWorksets):
            raise TypeError("prepared must be PreparedExecutionWorksets.")
        arrays = _item_tree(state, prepared.item_count)
        counters = jnp.asarray(rng_counters, dtype=jnp.uint32)
        if counters.shape != (prepared.item_count,):
            raise ValueError("rng_counters must contain one scalar counter per item.")
        revision_ids = _numeric_revision_ids(numeric_revisions)
        self.state = arrays
        self.rng_counters = counters
        self.prepared_id = prepared.prepared_id
        self.semantic_ids = prepared.plan.semantic_ids
        self.numeric_revision_ids = revision_ids
        self.checkpoint_id = _checkpoint_id(
            prepared.prepared_id,
            prepared.plan.semantic_ids,
            revision_ids,
            arrays,
            counters,
        )


def _numeric_revision_ids(
    numeric_revisions: Sequence[NumericRevision], /
) -> tuple[str, ...]:
    revisions = tuple(numeric_revisions)
    if any(not isinstance(revision, NumericRevision) for revision in revisions):
        raise TypeError("numeric_revisions must contain NumericRevision values.")
    revision_ids = tuple(sorted(revision.revision_id for revision in revisions))
    if len(set(revision_ids)) != len(revision_ids):
        raise ValueError("numeric_revisions must be distinct.")
    return revision_ids


def _checkpoint_id(
    prepared_id: str,
    semantic_ids: tuple[str, ...],
    numeric_revision_ids: tuple[str, ...],
    state: PyTree[Array],
    counters: Array,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "execution-workset-checkpoint",
            "prepared": prepared_id,
            "semantic_ids": list(semantic_ids),
            "numeric_revision_ids": list(numeric_revision_ids),
            "state": array_tree_fingerprint(state),
            "rng_counters": array_tree_fingerprint(counters),
        }
    )


def evaluate_execution_worksets_grouped(
    prepared: PreparedExecutionWorksets,
    runtime: ExecutionRuntime,
    evaluator: Callable[
        [int, str, PoolExecutionSignature, int, ExecutionGroup],
        Any,
    ],
    /,
    *,
    executor: HostTaskExecutor | None = None,
) -> tuple[Any, ...]:
    """Execute canonical independent items on disjoint child device groups."""

    if not isinstance(prepared, PreparedExecutionWorksets):
        raise TypeError("prepared must be PreparedExecutionWorksets.")
    if not isinstance(runtime, ExecutionRuntime):
        raise TypeError("runtime must be ExecutionRuntime.")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable.")
    executor_ = InlineTaskExecutor() if executor is None else executor
    owns_executor = executor is None
    sentinel = object()
    results: list[Any] = [sentinel] * prepared.plan.item_count
    rng_indices = tuple(
        int(value) for value in np.asarray(prepared.plan.semantic_rng_indices)
    )
    grouped: dict[str, list[int]] = {}
    signatures: dict[str, PoolExecutionSignature] = {}
    for item_index, signature in enumerate(prepared.plan.signatures):
        grouped.setdefault(signature.signature_id, []).append(item_index)
        signatures.setdefault(signature.signature_id, signature)
    try:
        for signature_id in sorted(grouped):
            signature = signatures[signature_id]
            if runtime.root_group.spec.device_count % signature.shard_count:
                raise ValueError(
                    "root execution group size must be divisible by item shard_count"
                )
            group_count = runtime.root_group.spec.device_count // signature.shard_count
            group_specs = partition_execution_group_specs(
                runtime.root_group.spec,
                group_count,
            )
            groups = tuple(bind_execution_group(spec) for spec in group_specs)
            if signature.execution_group_id is not None:
                if (
                    signature.execution_group_id == runtime.root_group.spec.group_id
                    and signature.shard_count == runtime.root_group.spec.device_count
                ):
                    groups = (runtime.root_group,)
                else:
                    groups = tuple(
                        group
                        for group in groups
                        if group.spec.group_id == signature.execution_group_id
                    )
                    if not groups:
                        raise ValueError(
                            "workset signature is bound to an unavailable execution group"
                        )
            parallel_count = 1 if runtime.inventory.process_count > 1 else len(groups)
            items = grouped[signature_id]
            for start in range(0, len(items), parallel_count):
                wave = items[start : start + parallel_count]
                handles = []
                for slot, item_index in enumerate(wave):
                    semantic_id = prepared.plan.semantic_ids[item_index]
                    rng_index = rng_indices[item_index]
                    group = groups[slot]
                    handles.append(
                        (
                            item_index,
                            executor_.submit(
                                semantic_id,
                                evaluator,
                                item_index,
                                semantic_id,
                                signature,
                                rng_index,
                                group,
                            ),
                        )
                    )
                for item_index, handle in handles:
                    results[item_index] = handle.result()
    finally:
        if owns_executor:
            executor_.close()
    if any(result is sentinel for result in results):
        raise RuntimeError("execution workset did not produce every canonical item")
    return tuple(results)


def restore_execution_workset_checkpoint(
    prepared: PreparedExecutionWorksets,
    checkpoint: ExecutionWorksetCheckpoint,
    /,
    *,
    numeric_revisions: Sequence[NumericRevision],
) -> tuple[PyTree[Array], Array]:
    """Validate topology, bound numeric revisions, and payload identity.

    `numeric_revisions` are the revisions currently bound into the items; they
    must be exactly the revisions the checkpoint was produced with.
    """
    if not isinstance(prepared, PreparedExecutionWorksets):
        raise TypeError("prepared must be PreparedExecutionWorksets.")
    if not isinstance(checkpoint, ExecutionWorksetCheckpoint):
        raise TypeError("checkpoint must be ExecutionWorksetCheckpoint.")
    if (
        checkpoint.prepared_id != prepared.prepared_id
        or checkpoint.semantic_ids != prepared.plan.semantic_ids
    ):
        raise ValueError("Execution workset checkpoint belongs to another runtime.")
    if _numeric_revision_ids(numeric_revisions) != checkpoint.numeric_revision_ids:
        raise ValueError(
            "Execution workset checkpoint was produced with other bound numeric revisions."
        )
    observed = _checkpoint_id(
        checkpoint.prepared_id,
        checkpoint.semantic_ids,
        checkpoint.numeric_revision_ids,
        checkpoint.state,
        checkpoint.rng_counters,
    )
    if observed != checkpoint.checkpoint_id:
        raise ValueError("Execution workset checkpoint content identity is corrupt.")
    return checkpoint.state, checkpoint.rng_counters


__all__ = [
    "ExecutionWorksetCheckpoint",
    "ExecutionWorksetEvaluation",
    "ExecutionWorksetEvidence",
    "ExecutionWorksetPlan",
    "PreparedExecutionWorksets",
    "evaluate_execution_worksets_filter_vmap",
    "evaluate_execution_worksets_grouped",
    "evaluate_execution_worksets_serial",
    "evaluate_execution_worksets_vmap",
    "restore_execution_workset_checkpoint",
]
