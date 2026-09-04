#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.execution import (
    evaluate_execution_worksets_serial,
    evaluate_execution_worksets_vmap,
    ExecutionWorksetCheckpoint,
    ExecutionWorksetPlan,
    PoolExecutionSignature,
    restore_execution_workset_checkpoint,
)


def _signature(topology: str) -> PoolExecutionSignature:
    return PoolExecutionSignature(
        topology_id=topology,
        method_id="explicit-map",
        precision_id="float32",
        backend_id="jax-cpu",
    )


def _plan(capacity: int = 2) -> ExecutionWorksetPlan:
    fast = _signature("fast-fiber")
    slow = _signature("slow-fiber")
    return ExecutionWorksetPlan(
        ("unit-3", "unit-1", "unit-4", "unit-2", "unit-0"),
        (slow, fast, slow, fast, fast),
        bucket_capacity=capacity,
    )


def _operation(signature, item, key, semantic_index):
    topology_scale = 2.0 if signature.topology_id == "fast-fiber" else 3.0
    noise = jax.random.uniform(key, shape=item.shape, minval=-0.25, maxval=0.25)
    return item * topology_scale + noise + semantic_index.astype(item.dtype) * 0.0


def test_plan_canonicalizes_order_and_builds_homogeneous_fixed_buckets() -> None:
    plan = _plan()
    assert plan.semantic_ids == ("unit-0", "unit-1", "unit-2", "unit-3", "unit-4")
    prepared = plan.prepare()
    assert prepared.item_indices.shape == (3, 2)
    assert prepared.valid_mask.shape == (3, 2)
    assert int(jnp.sum(prepared.valid_mask)) == plan.item_count
    for bucket, signature in enumerate(prepared.bucket_signatures):
        active = prepared.item_indices[bucket][prepared.valid_mask[bucket]]
        assert all(
            plan.signatures[int(item)].signature_id == signature.signature_id
            for item in active
        )


def test_plan_identity_and_rng_indices_do_not_depend_on_input_order() -> None:
    first = _plan()
    ids = tuple(reversed(first.semantic_ids))
    signatures = tuple(reversed(first.signatures))
    second = ExecutionWorksetPlan(ids, signatures, bucket_capacity=2)
    assert first.plan_id == second.plan_id
    assert jnp.array_equal(first.semantic_rng_indices, second.semantic_rng_indices)


def test_gather_scatter_is_an_exact_stable_permutation() -> None:
    prepared = _plan().prepare()
    values = {
        "state": jnp.arange(15, dtype=jnp.float32).reshape((5, 3)),
        "accepted": jnp.asarray([True, False, True, True, False]),
    }
    recovered = prepared.scatter(prepared.gather(values))
    assert jnp.array_equal(recovered["state"], values["state"])
    assert jnp.array_equal(recovered["accepted"], values["accepted"])


def test_serial_and_vmap_modes_are_exactly_equivalent() -> None:
    prepared = _plan().prepare()
    values = jnp.arange(15, dtype=jnp.float32).reshape((5, 3)) / 7.0
    counters = jnp.asarray([2, 1, 7, 0, 4], dtype=jnp.uint32)
    key = jax.random.key(83)
    serial = evaluate_execution_worksets_serial(
        prepared, _operation, values, key, counters
    )
    vectorized = evaluate_execution_worksets_vmap(
        prepared, _operation, values, key, counters
    )
    assert jnp.array_equal(serial.values, vectorized.values)
    assert bool(serial.evidence.successful)
    assert bool(vectorized.evidence.successful)
    assert jnp.array_equal(serial.next_rng_counters, counters + 1)
    assert jnp.array_equal(vectorized.next_rng_counters, counters + 1)
    assert int(vectorized.evidence.padded_lane_count) == 1


def test_semantic_rng_keys_survive_a_bucket_capacity_change() -> None:
    first = _plan(capacity=2).prepare()
    second = _plan(capacity=4).prepare()
    counters = jnp.arange(5, dtype=jnp.uint32)
    key = jax.random.key(19)
    first_keys = first.scatter(jax.random.key_data(first.semantic_keys(key, counters)))
    second_keys = second.scatter(jax.random.key_data(second.semantic_keys(key, counters)))
    assert jnp.array_equal(first_keys, second_keys)


def test_checkpoint_validates_runtime_and_payload_identity() -> None:
    prepared = _plan().prepare()
    state = jnp.arange(10, dtype=jnp.float32).reshape((5, 2))
    counters = jnp.arange(5, dtype=jnp.uint32)
    checkpoint = ExecutionWorksetCheckpoint(prepared, state, counters)
    restored_state, restored_counters = restore_execution_workset_checkpoint(
        prepared, checkpoint
    )
    assert jnp.array_equal(restored_state, state)
    assert jnp.array_equal(restored_counters, counters)
    corrupt = eqx.tree_at(lambda value: value.state, checkpoint, state.at[0, 0].set(-1.0))
    with pytest.raises(ValueError, match="content identity"):
        restore_execution_workset_checkpoint(prepared, corrupt)
    with pytest.raises(ValueError, match="another runtime"):
        restore_execution_workset_checkpoint(_plan(capacity=4).prepare(), checkpoint)


@pytest.mark.parametrize(
    "evaluate",
    [evaluate_execution_worksets_serial, evaluate_execution_worksets_vmap],
)
def test_failed_evaluation_preserves_counters_and_retry_keys(evaluate) -> None:
    prepared = _plan().prepare()
    counters = jnp.asarray([2, 1, 7, 0, 4], dtype=jnp.uint32)
    values = jnp.ones((5, 1))
    root_key = jax.random.key(0)

    def operation(signature, item, key, index):
        del signature, index
        return {
            "diagnostic": jax.random.key_data(key),
            "value": item / jnp.asarray(0.0),
        }

    failed = evaluate(prepared, operation, values, root_key, counters)
    retry = evaluate(
        prepared, operation, values, root_key, failed.next_rng_counters
    )

    assert not bool(failed.evidence.successful)
    assert jnp.array_equal(failed.next_rng_counters, counters)
    assert jnp.array_equal(retry.next_rng_counters, counters)
    assert jnp.array_equal(
        retry.values["diagnostic"], failed.values["diagnostic"]
    )


def test_rng_counter_overflow_fails_without_a_continuation_state() -> None:
    prepared = _plan().prepare()
    counters = jnp.zeros((5,), dtype=jnp.uint32).at[2].set(
        jnp.iinfo(jnp.uint32).max
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="counter overflow"):
        evaluation = evaluate_execution_worksets_vmap(
            prepared,
            _operation,
            jnp.ones((5, 1)),
            jax.random.key(0),
            counters,
        )
        jax.block_until_ready(evaluation.next_rng_counters)


def test_pool_execution_signature_is_exported_by_public_execution_module() -> None:
    import phydrax.execution as execution

    assert execution.PoolExecutionSignature is PoolExecutionSignature
    assert "PoolExecutionSignature" in execution.__all__


def test_distributed_execution_is_not_exposed_without_a_real_device_path() -> None:
    import phydrax.execution as worksets

    assert not any("distributed" in name.lower() for name in worksets.__all__)
    sharded = PoolExecutionSignature(
        topology_id="fast-fiber",
        method_id="explicit-map",
        precision_id="float32",
        backend_id="jax",
        shard_count=2,
    )
    with pytest.raises(ValueError, match="unsharded"):
        ExecutionWorksetPlan(("unit-0",), (sharded,))
