#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.execution import (
    evaluate_execution_worksets_filter_vmap,
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


def _operation(signature: Any, item: Any, key: Any, semantic_index: Any) -> Any:
    topology_scale = 2.0 if signature.topology_id == "fast-fiber" else 3.0
    noise = jax.random.uniform(key, shape=item.shape, minval=-0.25, maxval=0.25)
    return item * topology_scale + noise + semantic_index.astype(item.dtype) * 0.0


def test_execution_workset_scenario_1() -> None:
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
    first = _plan()
    ids = tuple(reversed(first.semantic_ids))
    signatures = tuple(reversed(first.signatures))
    second = ExecutionWorksetPlan(ids, signatures, bucket_capacity=2)
    assert first.plan_id == second.plan_id
    assert jnp.array_equal(first.semantic_rng_indices, second.semantic_rng_indices)
    sharded = PoolExecutionSignature(
        topology_id="fast-fiber",
        method_id="explicit-map",
        precision_id="float32",
        backend_id="jax",
        shard_count=2,
    )
    plan = ExecutionWorksetPlan(("unit-0",), (sharded,))
    assert plan.signatures[0].shard_count == 2
    assert plan.prepare().bucket_signatures == (sharded,)
    prepared = _plan().prepare()
    values = {
        "state": jnp.arange(15, dtype=jnp.float32).reshape((5, 3)),
        "accepted": jnp.asarray([True, False, True, True, False]),
    }
    recovered = prepared.scatter(prepared.gather(values))
    assert jnp.array_equal(recovered["state"], values["state"])
    assert jnp.array_equal(recovered["accepted"], values["accepted"])
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


def test_filter_vmap_worksets_broadcast_static_module_leaves() -> None:
    class StackedState(eqx.Module):
        values: jax.Array
        label: str = eqx.field(static=True)

    prepared = _plan().prepare()
    state = StackedState(
        jnp.arange(10, dtype=jnp.float32).reshape((5, 2)),
        "shared-static-label",
    )
    counters = jnp.arange(5, dtype=jnp.uint32)

    def operation(signature: Any, item: Any, key: Any, semantic_index: Any) -> Any:
        del key, semantic_index
        factor = 2.0 if signature.topology_id == "fast-fiber" else 3.0
        assert item.label == "shared-static-label"
        return item.values * factor

    result = evaluate_execution_worksets_filter_vmap(
        prepared,
        operation,
        state,
        jax.random.key(0),
        counters,
    )
    expected = jnp.stack(
        tuple(
            state.values[index] * (2.0 if signature.topology_id == "fast-fiber" else 3.0)
            for index, signature in enumerate(prepared.plan.signatures)
        )
    )
    assert jnp.array_equal(result.values, expected)
    assert bool(result.evidence.successful)


def test_filter_vmap_worksets_map_only_the_declared_item_lane() -> None:
    class Normalized(phx.StrictModule, phx.ParameterOwner):
        weight: jax.Array
        shift: jax.Array = phx.fixed_field()

    prepared = _plan().prepare()
    shifts = jnp.arange(10, dtype=jnp.float32).reshape((5, 2))
    items = Normalized(jnp.asarray([2.0, 3.0]), shifts)
    counters = jnp.zeros((5,), dtype=jnp.uint32)

    def operation(signature: Any, item: Any, key: Any, semantic_index: Any) -> Any:
        del signature, key, semantic_index
        return item.weight * (1.0 - item.shift)

    # FIXED per-item normalizers are mapped while the parameter is shared.
    result = evaluate_execution_worksets_filter_vmap(
        prepared,
        operation,
        items,
        jax.random.key(0),
        counters,
        layout=phx.LaneLayout("item", (".shift",)),
    )
    assert jnp.array_equal(result.values, items.weight * (1.0 - shifts))
    assert jnp.array_equal(result.next_rng_counters, counters + 1)
    with pytest.raises(ValueError, match="share one lane size"):
        evaluate_execution_worksets_filter_vmap(
            prepared, operation, items, jax.random.key(0), counters
        )
    with pytest.raises(TypeError, match="kind 'item'"):
        evaluate_execution_worksets_filter_vmap(
            prepared,
            operation,
            items,
            jax.random.key(0),
            counters,
            layout=phx.LaneLayout("member", (".shift",)),
        )


def test_execution_workset_scenario_2() -> None:
    first = _plan(capacity=2).prepare()
    second = _plan(capacity=4).prepare()
    counters = jnp.arange(5, dtype=jnp.uint32)
    key = jax.random.key(19)
    first_keys = first.scatter(jax.random.key_data(first.semantic_keys(key, counters)))
    second_keys = second.scatter(jax.random.key_data(second.semantic_keys(key, counters)))
    assert jnp.array_equal(first_keys, second_keys)
    full = _plan().prepare()
    fast = _signature("fast-fiber")
    subset = ExecutionWorksetPlan(("unit-4", "unit-1"), (fast, fast)).prepare()
    key = jax.random.key(19)
    full_keys = full.scatter(
        jax.random.key_data(full.semantic_keys(key, jnp.full((5,), 3, jnp.uint32)))
    )
    subset_keys = subset.scatter(
        jax.random.key_data(subset.semantic_keys(key, jnp.full((2,), 3, jnp.uint32)))
    )
    by_id = dict(zip(full.plan.semantic_ids, full_keys, strict=True))
    for semantic_id, key_words in zip(subset.plan.semantic_ids, subset_keys, strict=True):
        assert jnp.array_equal(key_words, by_id[semantic_id])
    prepared = _plan().prepare()
    state = jnp.arange(10, dtype=jnp.float32).reshape((5, 2))
    counters = jnp.arange(5, dtype=jnp.uint32)
    checkpoint = ExecutionWorksetCheckpoint(
        prepared, state, counters, numeric_revisions=()
    )
    restored_state, restored_counters = restore_execution_workset_checkpoint(
        prepared, checkpoint, numeric_revisions=()
    )
    assert jnp.array_equal(restored_state, state)
    assert jnp.array_equal(restored_counters, counters)
    corrupt = eqx.tree_at(lambda value: value.state, checkpoint, state.at[0, 0].set(-1.0))
    with pytest.raises(ValueError, match="content identity"):
        restore_execution_workset_checkpoint(prepared, corrupt, numeric_revisions=())
    with pytest.raises(ValueError, match="another runtime"):
        restore_execution_workset_checkpoint(
            _plan(capacity=4).prepare(), checkpoint, numeric_revisions=()
        )
    prepared = _plan().prepare()
    state = jnp.ones((5, 2), dtype=jnp.float32)
    counters = jnp.zeros((5,), dtype=jnp.uint32)
    semantic = phx.SemanticProvenance({"kind": "workset-gain"})
    trained = phx.NumericRevision(semantic, {"gain": jnp.asarray(2.0)})
    updated = phx.NumericRevision(semantic, {"gain": jnp.asarray(3.0)})
    checkpoint = ExecutionWorksetCheckpoint(
        prepared, state, counters, numeric_revisions=(trained,)
    )
    unbound = ExecutionWorksetCheckpoint(prepared, state, counters, numeric_revisions=())

    # The same item state produced by other weights is another checkpoint.
    assert checkpoint.checkpoint_id != unbound.checkpoint_id
    restored, _ = restore_execution_workset_checkpoint(
        prepared, checkpoint, numeric_revisions=(trained,)
    )
    assert jnp.array_equal(restored, state)
    for bound in ((updated,), (), (trained, updated)):
        with pytest.raises(ValueError, match="other bound numeric revisions"):
            restore_execution_workset_checkpoint(
                prepared, checkpoint, numeric_revisions=bound
            )
    with pytest.raises(ValueError, match="distinct"):
        ExecutionWorksetCheckpoint(
            prepared, state, counters, numeric_revisions=(trained, trained)
        )
    prepared = _plan().prepare()
    counters = jnp.zeros((5,), dtype=jnp.uint32).at[2].set(jnp.iinfo(jnp.uint32).max)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="counter overflow"):
        evaluation = evaluate_execution_worksets_vmap(
            prepared,
            _operation,
            jnp.ones((5, 1)),
            jax.random.key(0),
            counters,
        )
        jax.block_until_ready(evaluation.next_rng_counters)


def test_failed_evaluation_preserves_counters_and_retry_keys() -> None:
    for evaluate in [
        evaluate_execution_worksets_serial,
        evaluate_execution_worksets_vmap,
    ]:
        prepared = _plan().prepare()
        counters = jnp.asarray([2, 1, 7, 0, 4], dtype=jnp.uint32)
        values = jnp.ones((5, 1))
        root_key = jax.random.key(0)

        def operation(signature: Any, item: Any, key: Any, index: Any) -> Any:
            del signature, index
            return {
                "diagnostic": jax.random.key_data(key),
                "value": item / jnp.asarray(0.0),
            }

        failed = evaluate(prepared, operation, values, root_key, counters)
        retry = evaluate(prepared, operation, values, root_key, failed.next_rng_counters)

        assert not bool(failed.evidence.successful)
        assert jnp.array_equal(failed.next_rng_counters, counters)
        assert jnp.array_equal(retry.next_rng_counters, counters)
        assert jnp.array_equal(retry.values["diagnostic"], failed.values["diagnostic"])


def test_pool_execution_signature_is_exported_by_public_execution_module() -> None:
    import phydrax.execution as execution

    assert execution.PoolExecutionSignature is PoolExecutionSignature
    assert "PoolExecutionSignature" in execution.__all__


def _fiber_topology(epoch: int) -> phx.lifecycle.CompositionEntry:
    return phx.lifecycle.CompositionEntry(
        jnp.arange(4 + epoch),
        entry_id="fiber/topology",
        role="topology",
        owner_id="fiber",
        structure_id=f"fiber-topology-{epoch}",
        revision_id=f"fiber-topology-{epoch}",
        semantics_id="fiber.bundle",
    )


def test_workset_entry_is_stale_until_reprepared_for_a_new_topology() -> None:
    topology = _fiber_topology(0)
    worksets = _plan().prepare()
    source = phx.lifecycle.Composition(
        (
            topology,
            worksets.composition_entry(
                entry_id="fiber/worksets",
                owner_id="fiber",
                dependencies=(topology.binding("structure"),),
            ),
        ),
        boundary_id="window-0",
    )
    refined = _fiber_topology(1)
    with pytest.raises(
        ValueError, match="'fiber/worksets' is stale against the structure"
    ):
        phx.lifecycle.CompositionRebind(
            source, retain=("fiber/worksets",), reprepare=(refined,)
        )
    fine = _signature("fine-fiber")
    replanned = ExecutionWorksetPlan(
        ("unit-3", "unit-1", "unit-4", "unit-2", "unit-0"),
        (fine, fine, fine, fine, _signature("slow-fiber")),
        bucket_capacity=2,
    ).prepare()
    staged = replanned.composition_entry(
        entry_id="fiber/worksets",
        owner_id="fiber",
        dependencies=(refined.binding("structure"),),
    )
    assert staged.structure_id != source.entry("fiber/worksets").structure_id
    receipt = phx.lifecycle.commit_composition_rebind(
        phx.lifecycle.CompositionRebind(source, reprepare=(refined, staged)),
        accepted_boundary=True,
    )
    assert receipt.published
    assert receipt.reprepared == ("fiber/topology", "fiber/worksets")
    assert receipt.composition.value("fiber/worksets") is replanned
    other_items = ExecutionWorksetPlan(("unit-0",), (fine,)).prepare()
    with pytest.raises(ValueError, match="must keep its role and semantics"):
        phx.lifecycle.CompositionRebind(
            source,
            reprepare=(
                refined,
                other_items.composition_entry(
                    entry_id="fiber/worksets",
                    owner_id="fiber",
                    dependencies=(refined.binding("structure"),),
                ),
            ),
        )
