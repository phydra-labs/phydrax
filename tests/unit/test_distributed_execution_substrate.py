#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from phydrax._data_plane import IndexEpochPlan
from phydrax.backends import APPLE_METAL_PROFILE
from phydrax.discretization._distributed_field import (
    DistributedHaloPlan,
    DistributedLocalOperator,
)
from phydrax.execution import (
    AxisBinding,
    DeviceResource,
    DistributedIndexEpochPlan,
    DistributionMode,
    evaluate_execution_worksets_grouped,
    ExecutionCandidate,
    ExecutionGroupSpec,
    ExecutionPolicy,
    ExecutionRequirements,
    ExecutionRuntime,
    ExecutionWorksetPlan,
    LogicalAxis,
    LogicalAxisKind,
    partition_execution_group_specs,
    PlacementKind,
    PoolExecutionSignature,
    ProviderBinding,
    resolve_execution_plan,
    ResourceInventory,
    ValuePlacement,
)
from phydrax.lifecycle import (
    assemble_distributed_checkpoint_from_repository,
    publish_process_checkpoint,
    restore_global_array_from_checkpoint,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.linalg import (
    DistributedKrylovPolicy,
    DistributedLinearOperator,
    DistributedPairing,
    solve_distributed_pcg,
)


def test_root_import_consumes_bootstrap_environment_before_package_loading() -> None:
    root = Path(__file__).parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    environment["PHYDRAX_CPU_COLLECTIVES"] = "gloo"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import phydrax; import jax; "
                "assert jax.config.jax_cpu_collectives_implementation == 'gloo'; "
                "info = phydrax.execution.initialize(); "
                "assert info.process_count == 1; "
                "assert info.global_device_count >= 1"
            ),
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _inventory() -> ResourceInventory:
    return ResourceInventory(
        1,
        0,
        (
            DeviceResource(0, 0, 0, "cpu", "cpu"),
            DeviceResource(0, 1, 1, "cpu", "cpu"),
        ),
    )


def test_inventory_identity_excludes_process_local_visibility() -> None:
    first = ResourceInventory(
        2,
        0,
        (
            DeviceResource(0, 0, 0, "cpu", "cpu"),
            DeviceResource(1, 2048, None, "cpu", "cpu"),
        ),
    )
    second = ResourceInventory(
        2,
        1,
        (
            DeviceResource(0, 0, None, "cpu", "cpu"),
            DeviceResource(1, 2048, 0, "cpu", "cpu"),
        ),
    )
    assert first.inventory_id == second.inventory_id


def test_execution_plan_admits_complete_distributed_candidate_and_round_trips() -> None:
    inventory = _inventory()
    requirements = ExecutionRequirements(
        "distributed-test",
        logical_axes=(LogicalAxis("case", 8, LogicalAxisKind.INDEPENDENT),),
        operations=("objective",),
        dtypes=("float64",),
        transformations=("jit", "vjp"),
        requires_process_local_input=True,
    )
    group = ExecutionGroupSpec(
        "group-two",
        (0,),
        ((0, 0), (0, 1)),
        mesh_axes=(("data", 2),),
    )
    candidate = ExecutionCandidate(
        "data-parallel",
        requirements.requirements_id,
        group,
        axis_bindings=(AxisBinding("case", ("data",)),),
        value_placements=(
            ValuePlacement(
                "cases",
                PlacementKind.PARTITIONED,
                logical_axes=("case",),
                mesh_axes=("data", None),
                global_shape=(8, 3),
                dtype="float64",
            ),
        ),
        providers=(ProviderBinding("array", "jax-native", "global-array"),),
    )
    plan = resolve_execution_plan(
        ExecutionPolicy(DistributionMode.DISTRIBUTED),
        inventory,
        (candidate,),
        precision_policy_id="float64",
        solver_policy_id="test-solver",
    )
    restored = type(plan).from_payload(plan.to_payload())
    assert restored.plan_fingerprint == plan.plan_fingerprint
    assert restored.group == group
    assert restored.value_placements[0].global_shape == (8, 3)


def test_process_symmetric_child_groups_are_disjoint() -> None:
    parent = ExecutionGroupSpec(
        "root",
        (0, 1),
        ((0, 0), (0, 1), (1, 2), (1, 3)),
        mesh_axes=(("device", 4),),
    )
    groups = partition_execution_group_specs(parent, 2)
    assert all(group.process_indices == (0, 1) for group in groups)
    assert set(groups[0].device_keys).isdisjoint(groups[1].device_keys)
    assert set(groups[0].device_keys) | set(groups[1].device_keys) == set(
        parent.device_keys
    )


def test_grouped_workset_preserves_canonical_identity_and_group_assignment() -> None:
    runtime = ExecutionRuntime.current()
    signature = PoolExecutionSignature(
        topology_id="grouped-test",
        method_id="identity",
        precision_id="host",
        backend_id="inline",
        shard_count=runtime.root_group.spec.device_count,
    )
    prepared = ExecutionWorksetPlan(
        ("item-b", "item-a"),
        (signature, signature),
    ).prepare()

    def evaluate(item_index, semantic_id, item_signature, rng_index, group):
        return (
            item_index,
            semantic_id,
            item_signature.signature_id,
            rng_index,
            group.spec.group_id,
            group.spec.parent_group_id,
        )

    results = evaluate_execution_worksets_grouped(prepared, runtime, evaluate)
    assert tuple(result[1] for result in results) == ("item-a", "item-b")
    assert all(result[2] == signature.signature_id for result in results)
    assert all(result[5] == runtime.root_group.spec.group_id for result in results)


def test_process_local_epoch_preserves_global_ids_and_fixed_capacity() -> None:
    global_plan = IndexEpochPlan(10, 7, False, 0, 0, False)
    two_processes = tuple(
        DistributedIndexEpochPlan(global_plan, 2, process, 2).batch(0)
        for process in range(2)
    )
    four_processes = tuple(
        DistributedIndexEpochPlan(global_plan, 4, process, 1).batch(0)
        for process in range(4)
    )

    def active(batches):
        return tuple(
            index
            for batch in batches
            for index, valid in zip(batch.indices, batch.valid, strict=True)
            if valid
        )

    assert active(two_processes) == tuple(range(7))
    assert active(four_processes) == tuple(range(7))
    assert all(len(batch.indices) == 4 for batch in two_processes)
    assert all(len(batch.indices) == 2 for batch in four_processes)


def test_generic_halo_shard_map_matches_forward_and_transpose_references() -> None:
    devices = tuple(jax.devices()[:2])
    if len(devices) < 2:
        pytest.skip("requires two real or explicitly configured JAX devices")
    halo = DistributedHaloPlan(
        jnp.asarray([0, 0, 1, 1]),
        jnp.asarray([[1, 2]]),
        2,
    )

    def action(part, local, global_ids, valid, owned):
        del part, global_ids
        halo_total = jnp.sum(jnp.where(valid & ~owned, local, 0))
        return jnp.where(owned, local + halo_total, 0)

    def transpose(part, local, global_ids, valid, owned):
        del part, global_ids
        owned_total = jnp.sum(jnp.where(owned, local, 0))
        return jnp.where(owned, local, jnp.where(valid, owned_total, 0))

    operator = DistributedLocalOperator(
        halo,
        action,
        transpose,
        operator_name="test-neighbor-sum",
    )
    values = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    cotangent = jnp.asarray([0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(
        operator.distributed(values),
        operator.serial_reference(values),
    )
    np.testing.assert_allclose(
        operator.distributed_transpose(cotangent),
        operator.serial_transpose_reference(cotangent),
    )


def _repository(tmp_path: Path) -> POSIXArtifactRepository:
    profile = HPCFilesystemProfile(
        "posix.distributed-test",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    policy = POSIXRepositoryPolicy(
        profile,
        maximum_chunk_bytes=64,
        maximum_metadata_bytes=64 * 1024,
    )
    return POSIXArtifactRepository(tmp_path / "repository", policy)


def test_addressable_checkpoint_publishes_once_and_restores_directly(
    tmp_path: Path,
) -> None:
    device = jax.devices()[0]
    mesh = Mesh(np.asarray((device,), dtype=object), ("data",))
    sharding = NamedSharding(mesh, PartitionSpec("data"))
    value = jax.device_put(jnp.arange(12, dtype=jnp.float64), sharding)
    repository = _repository(tmp_path)
    publication = publish_process_checkpoint(
        repository,
        "checkpoint-a",
        "execution-a",
        {"value": value},
        writer_id="writer-a",
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "checkpoint-a",
        "analysis-a",
        "revision-a",
        "execution-a",
        expected_process_count=1,
    )
    metadata = dict(publication.shards[0].metadata)
    restored = restore_global_array_from_checkpoint(
        repository,
        manifest,
        metadata["array_path"],
        sharding,
    )
    np.testing.assert_array_equal(np.asarray(restored), np.arange(12))


def test_distributed_pcg_uses_owned_pairing_and_global_consensus() -> None:
    operator = DistributedLinearOperator(
        lambda value: 4.0 * value,
        lambda value: 4.0 * value,
        (4,),
        (4,),
        operator_id="four-identity",
    )
    pairing = DistributedPairing(jnp.asarray([True, True, True, False]))
    right_hand_side = jnp.asarray([4.0, 8.0, 12.0, 0.0])
    result = solve_distributed_pcg(
        operator,
        right_hand_side,
        pairing,
        DistributedKrylovPolicy(8, relative_tolerance=1.0e-12),
    )
    np.testing.assert_allclose(result.value, jnp.asarray([1.0, 2.0, 3.0, 0.0]))
    assert bool(result.converged)


def test_experimental_vendor_profile_rejects_unsupported_precision() -> None:
    assert APPLE_METAL_PROFILE.experimental
    with pytest.raises(ValueError, match="float64"):
        APPLE_METAL_PROFILE.require(scope="single_device", dtype="float64")
