#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ._execution_pool import PoolExecutionSignature, semantic_task_keys
from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


ExecutionWorksetMode = Literal["serial", "vmap"]


def _semantic_rng_index(identifier: str, /) -> int:
    digest = hashlib.sha256(f"phydrax-execution-item:{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


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
            raise ValueError("Execution workset semantic IDs must be non-empty and unique.")
        if len(signature_values) != len(identifiers) or not all(
            isinstance(value, PoolExecutionSignature) for value in signature_values
        ):
            raise TypeError(
                "signatures must contain one PoolExecutionSignature per semantic ID."
            )
        if any(value.shard_count != 1 for value in signature_values):
            raise ValueError(
                "Execution worksets currently accept only real local unsharded signatures."
            )
        if capacity < 1 or capacity > 64:
            raise ValueError("bucket_capacity must lie in the fixed modest range [1, 64].")
        ordered = sorted(zip(identifiers, signature_values, strict=True), key=lambda x: x[0])
        canonical_ids = tuple(value[0] for value in ordered)
        canonical_signatures = tuple(value[1] for value in ordered)
        rng_indices = tuple(_semantic_rng_index(value) for value in canonical_ids)
        if len(set(rng_indices)) != len(rng_indices):
            raise ValueError(
                "Semantic RNG indices collide; choose distinct semantic identifiers."
            )
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
                    active
                    + [active[0]] * (plan.bucket_capacity - len(active))
                )
                bucket_signatures.append(representatives[signature_id])
        indices = np.asarray(bucket_rows, dtype=np.int32)
        valid = np.zeros_like(indices, dtype=bool)
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
                "bucket_signatures": [
                    value.signature_id for value in bucket_signatures
                ],
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
            safe = jnp.where(self.valid_mask.reshape((-1,)), flat_indices, self.item_count)
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
        item_keys = semantic_task_keys(root, self.plan.semantic_rng_indices)
        bucket_keys = item_keys[self.item_indices]
        bucket_counters = counters[self.item_indices]
        flat_keys = bucket_keys.reshape((-1,) + bucket_keys.shape[2:])
        flat_counters = bucket_counters.reshape((-1,))
        folded = jax.vmap(jax.random.fold_in)(flat_keys, flat_counters)
        return folded.reshape(bucket_keys.shape)


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
    operation: Callable[[PoolExecutionSignature, PyTree[Array], Array, Array], PyTree[Array]],
    values: PyTree[ArrayLike],
    root_key: Array,
    rng_counters: ArrayLike,
    /,
    *,
    mode: ExecutionWorksetMode,
) -> ExecutionWorksetEvaluation:
    if not isinstance(prepared, PreparedExecutionWorksets):
        raise TypeError("prepared must be PreparedExecutionWorksets.")
    if not callable(operation):
        raise TypeError("operation must be callable.")
    gathered = prepared.gather(values)
    counters = jnp.asarray(rng_counters, dtype=jnp.uint32)
    keys = prepared.semantic_keys(root_key, counters)
    counter_overflow = jnp.any(counters == jnp.iinfo(jnp.uint32).max)
    advanced_counters = counters + jnp.asarray(1, dtype=jnp.uint32)
    if mode == "vmap":
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
            group_values = jax.tree_util.tree_map(
                lambda value: value[start:stop], gathered
            )
            group_keys = keys[start:stop]
            group_indices = prepared.bucket_rng_indices[start:stop]

            def lane_operation(item, key, semantic_index):
                return operation(signature, item, key, semantic_index)

            signature_group_outputs.append(
                jax.vmap(jax.vmap(lane_operation))(
                    group_values, group_keys, group_indices
                )
            )
            start = stop
        buckets = _concatenate_trees(signature_group_outputs)
    else:
        bucket_outputs: list[PyTree[Array]] = []
        for bucket, signature in enumerate(prepared.bucket_signatures):
            bucket_values = jax.tree_util.tree_map(
                lambda value: value[bucket], gathered
            )
            bucket_keys = keys[bucket]
            bucket_indices = prepared.bucket_rng_indices[bucket]
            lanes = [
                operation(
                    signature,
                    jax.tree_util.tree_map(lambda value: value[lane], bucket_values),
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
    operation: Callable[[PoolExecutionSignature, PyTree[Array], Array, Array], PyTree[Array]],
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
    operation: Callable[[PoolExecutionSignature, PyTree[Array], Array, Array], PyTree[Array]],
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


class ExecutionWorksetCheckpoint(StrictModule, NonTrainableState):
    """Content-addressed host checkpoint for canonical item state and RNG counters."""

    state: Any
    rng_counters: Array
    prepared_id: str = eqx.field(static=True)
    semantic_ids: tuple[str, ...] = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedExecutionWorksets,
        state: PyTree[ArrayLike],
        rng_counters: ArrayLike,
        /,
    ):
        if not isinstance(prepared, PreparedExecutionWorksets):
            raise TypeError("prepared must be PreparedExecutionWorksets.")
        arrays = _item_tree(state, prepared.item_count)
        counters = jnp.asarray(rng_counters, dtype=jnp.uint32)
        if counters.shape != (prepared.item_count,):
            raise ValueError("rng_counters must contain one scalar counter per item.")
        self.state = arrays
        self.rng_counters = counters
        self.prepared_id = prepared.prepared_id
        self.semantic_ids = prepared.plan.semantic_ids
        self.checkpoint_id = _checkpoint_id(
            prepared.prepared_id,
            prepared.plan.semantic_ids,
            arrays,
            counters,
        )


def _checkpoint_id(
    prepared_id: str,
    semantic_ids: tuple[str, ...],
    state: PyTree[Array],
    counters: Array,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "execution-workset-checkpoint",
            "prepared": prepared_id,
            "semantic_ids": list(semantic_ids),
            "state": array_tree_fingerprint(state),
            "rng_counters": array_tree_fingerprint(counters),
        }
    )


def restore_execution_workset_checkpoint(
    prepared: PreparedExecutionWorksets,
    checkpoint: ExecutionWorksetCheckpoint,
    /,
) -> tuple[PyTree[Array], Array]:
    """Validate topology and payload identity before returning checkpoint state."""
    if not isinstance(prepared, PreparedExecutionWorksets):
        raise TypeError("prepared must be PreparedExecutionWorksets.")
    if not isinstance(checkpoint, ExecutionWorksetCheckpoint):
        raise TypeError("checkpoint must be ExecutionWorksetCheckpoint.")
    if (
        checkpoint.prepared_id != prepared.prepared_id
        or checkpoint.semantic_ids != prepared.plan.semantic_ids
    ):
        raise ValueError("Execution workset checkpoint belongs to another runtime.")
    observed = _checkpoint_id(
        checkpoint.prepared_id,
        checkpoint.semantic_ids,
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
    "evaluate_execution_worksets_serial",
    "evaluate_execution_worksets_vmap",
    "restore_execution_workset_checkpoint",
]
