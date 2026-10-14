#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import phydrax.lifecycle._distributed_checkpoint as checkpoint_module
from phydrax import config as pxconfig
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
    LocalProcessLaunchPlan,
    LogicalAxis,
    LogicalAxisKind,
    partition_execution_group_specs,
    PlacementKind,
    PoolExecutionSignature,
    ProviderBinding,
    RankEnvironment,
    resolve_execution_plan,
    ResourceInventory,
    ValuePlacement,
)
from phydrax.lifecycle import (
    assemble_distributed_checkpoint_from_repository,
    CheckpointManifest,
    CheckpointShard,
    publish_process_checkpoint,
    RepositoryCorruptionError,
    restore_global_array_from_checkpoint,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.linalg import (
    DistributedAdditiveSchwarzPreconditioner,
    DistributedKrylovPolicy,
    DistributedLinearOperator,
    DistributedPairing,
    solve_distributed_block_krylov,
    solve_distributed_fgmres,
    solve_distributed_gmres,
    solve_distributed_minres,
    solve_distributed_pcg,
)


def test_distributed_execution_substrate_scenario_1() -> None:
    root = Path(__file__).parents[2]
    environment = pxconfig.environment(
        pxconfig.RuntimeConfig(cpu_collectives_implementation="gloo")
    )
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import phydrax; "
                "assert phydrax.config.snapshot().cpu_collectives_implementation == 'gloo'; "
                "assert phydrax.config.snapshot().backends_initialized is False; "
                "info = phydrax.execution.runtime_info(); "
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

    tampered = plan.to_payload()
    tampered["backend"] = "tampered-backend"
    with pytest.raises(ValueError, match="fingerprint"):
        type(plan).from_payload(tampered)

    missing_fingerprint = plan.to_payload()
    del missing_fingerprint["plan_fingerprint"]
    with pytest.raises(ValueError, match="plan_fingerprint"):
        type(plan).from_payload(missing_fingerprint)

    unknown_field = plan.to_payload()
    unknown_field["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        type(plan).from_payload(unknown_field)


def _inventory() -> ResourceInventory:
    return ResourceInventory(
        1,
        0,
        (
            DeviceResource(0, 0, 0, "cpu", "cpu"),
            DeviceResource(0, 1, 1, "cpu", "cpu"),
        ),
    )


def test_distributed_execution_substrate_scenario_2() -> None:
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
    for weights in (
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([1.0, -1.0]),
        jnp.asarray([1.0, jnp.nan]),
    ):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            DistributedPairing(jnp.asarray([True, True]), weights=weights)

        padded = DistributedPairing(
            jnp.asarray([True, False]),
            weights=jnp.asarray([2.0, jnp.nan]),
        )
        assert padded.inner(jnp.asarray([3.0, 1.0]), jnp.asarray([4.0, 1.0])) == 24.0

        with pytest.raises(TypeError, match="real numeric"):
            DistributedPairing(
                jnp.asarray([True, False]),
                weights=jnp.asarray([1.0 + 0.0j, 0.0 + 1.0j]),
            )
    for solver in (
        solve_distributed_gmres,
        solve_distributed_fgmres,
        solve_distributed_minres,
    ):
        diagonal = jnp.asarray([2.0, 3.0, 4.0, 1.0])
        operator = DistributedLinearOperator(
            lambda value: diagonal * value,
            lambda value: diagonal * value,
            (4,),
            (4,),
            operator_id="positive-diagonal",
        )
        pairing = DistributedPairing(jnp.asarray([True, True, True, False]))
        right_hand_side = jnp.asarray([2.0, 6.0, 12.0, 0.0])
        schwarz = DistributedAdditiveSchwarzPreconditioner(
            (lambda value: value / diagonal, lambda value: value / diagonal),
            (jnp.full((4,), 0.5), jnp.full((4,), 0.5)),
            preconditioner_id="two-copy-partition-of-unity",
        )

        result = solver(
            operator,
            right_hand_side,
            pairing,
            DistributedKrylovPolicy(
                8,
                relative_tolerance=1.0e-12,
                restart=4,
            ),
            preconditioner=schwarz,
        )

        np.testing.assert_allclose(result.value, jnp.asarray([1.0, 2.0, 3.0, 0.0]))
        assert bool(result.converged)
        assert result.residual_norm < 1.0e-10
    operator = DistributedLinearOperator(
        lambda value: 5.0 * value,
        lambda value: 5.0 * value,
        (3,),
        (3,),
        operator_id="five-identity",
    )
    pairing = DistributedPairing(jnp.ones((3,), dtype="bool"))
    right = jnp.asarray([[5.0, 10.0], [10.0, -5.0], [15.0, 20.0]])

    result = solve_distributed_block_krylov(
        operator,
        right,
        pairing,
        DistributedKrylovPolicy(6, relative_tolerance=1.0e-12, restart=3),
        method="gmres",
    )

    np.testing.assert_allclose(result.value, right / 5.0)
    np.testing.assert_array_equal(result.converged, jnp.asarray((True, True)))
    assert result.iterations.shape == (2,)
    assert APPLE_METAL_PROFILE.experimental
    with pytest.raises(ValueError, match="float64"):
        APPLE_METAL_PROFILE.require(scope="single_device", dtype="float64")


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

    def evaluate(
        item_index: Any, semantic_id: Any, item_signature: Any, rng_index: Any, group: Any
    ) -> Any:
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

    def active(batches: Any) -> Any:
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

    def action(part: Any, local: Any, global_ids: Any, valid: Any, owned: Any) -> Any:
        del part, global_ids
        halo_total = jnp.sum(jnp.where(valid & ~owned, local, 0))
        return jnp.where(owned, local + halo_total, 0)

    def transpose(part: Any, local: Any, global_ids: Any, valid: Any, owned: Any) -> Any:
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


def _repository(tmp_path: Path, maximum_chunk_bytes: int = 64) -> POSIXArtifactRepository:
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
        maximum_chunk_bytes=maximum_chunk_bytes,
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
        analysis_plan_id="analysis-a",
        numeric_revision_id="revision-a",
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
    forged_metadata = dict(publication.shards[0].metadata)
    forged_metadata["repository_id"] = "forged-repository"
    forged_shard = CheckpointShard(
        publication.shards[0].shard_id,
        publication.shards[0].payload_digest,
        publication.shards[0].byte_count,
        publication.shards[0].layout_ids,
        metadata=forged_metadata,
    )
    forged_manifest = CheckpointManifest(
        manifest.checkpoint_id,
        manifest.analysis_plan_id,
        manifest.numeric_revision_id,
        manifest.execution_plan_id,
        (forged_shard,),
        complete=True,
    )
    with pytest.raises(RepositoryCorruptionError, match="lifecycle ownership"):
        restore_global_array_from_checkpoint(
            repository,
            forged_manifest,
            metadata["array_path"],
            sharding,
        )
    publish_process_checkpoint(
        repository,
        "checkpoint-b",
        "execution-a",
        {"value": value + 1.0},
        writer_id="writer-b",
        analysis_plan_id="analysis-a",
        numeric_revision_id="revision-a",
        parent_manifest=manifest,
    )
    child = assemble_distributed_checkpoint_from_repository(
        repository,
        "checkpoint-b",
        "analysis-a",
        "revision-a",
        "execution-a",
        expected_process_count=1,
        parent_manifest=manifest,
    )
    assert child.parent_checkpoint_id == manifest.checkpoint_id
    assert child.parent_manifest_id == manifest.manifest_id
    child_path = dict(child.shards[0].metadata)["array_path"]
    child_restored = restore_global_array_from_checkpoint(
        repository,
        child,
        child_path,
        sharding,
        parent_manifest=manifest,
    )
    np.testing.assert_array_equal(np.asarray(child_restored), np.arange(12) + 1.0)


def test_distributed_checkpoint_rejects_huge_declared_shape_before_device_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = CheckpointShard(
        "shard-huge",
        "0" * 64,
        128,
        ("layout-huge",),
        metadata={
            "artifact_id": "artifact-huge",
            "artifact_manifest_id": "manifest-huge",
            "checkpoint_id": "checkpoint-huge",
            "analysis_plan_id": "analysis-huge",
            "numeric_revision_id": "revision-huge",
            "execution_plan_id": "execution-huge",
            "logical_name": "array-0",
            "array_path": "['value']",
            "global_shape": "[1000000000]",
            "dtype": "<f8",
            "index": "[[0,1,1]]",
            "process_index": "0",
            "device_id": "0",
            "replica_id": "0",
            "topology_epoch": "0",
            "repository_id": "repository-huge",
            "parent_checkpoint_id": "none",
            "parent_manifest_id": "none",
        },
    )
    manifest = CheckpointManifest(
        "checkpoint-huge",
        "analysis-huge",
        "revision-huge",
        "execution-huge",
        (shard,),
        complete=True,
    )
    monkeypatch.setattr(
        checkpoint_module.jax,
        "make_array_from_callback",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("device allocation was reached before manifest admission")
        ),
    )

    with pytest.raises(RepositoryCorruptionError, match="axis-length"):
        restore_global_array_from_checkpoint(
            # ty: ignore[invalid-argument-type]
            object(),
            manifest,
            "['value']",
            # ty: ignore[invalid-argument-type]
            None,
        )


def test_checkpoint_multichunk_disk_spool_returns_readonly_array(tmp_path: Path) -> None:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS

    repository = _repository(tmp_path, maximum_chunk_bytes=262144)
    value = jnp.arange(262144, dtype=jnp.float64)
    publication = publish_process_checkpoint(
        repository,
        "checkpoint-spool",
        "execution-spool",
        {"value": value},
        analysis_plan_id="analysis-spool",
        numeric_revision_id="revision-spool",
        writer_id="writer-spool",
    )
    shard = publication.shards[0]
    restored = checkpoint_module._read_shard_payload(
        repository, shard, value.shape, np.dtype(np.float64), DEFAULT_ARRAY_ARCHIVE_LIMITS
    )
    np.testing.assert_array_equal(restored, np.arange(262144, dtype=np.float64))
    assert not restored.flags.writeable
    with pytest.raises(ValueError):
        restored[0] = -1


@pytest.mark.parametrize("fault", ["digest", "shape", "dtype", "allocation"])
def test_checkpoint_shard_decode_refuses_invalid_payload_contract(
    tmp_path: Path, fault: str
) -> None:
    from phydrax._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS

    repository = _repository(tmp_path)
    value = jnp.arange(16, dtype=jnp.float64)
    publication = publish_process_checkpoint(
        repository,
        "checkpoint-fault",
        "execution-fault",
        {"value": value},
        analysis_plan_id="analysis-fault",
        numeric_revision_id="revision-fault",
        writer_id="writer-fault",
    )
    shard = publication.shards[0]
    shape = (15,) if fault == "shape" else value.shape
    dtype = np.dtype(np.float32) if fault == "dtype" else np.dtype(np.float64)
    limits = (
        ArrayArchiveLimits(max_member_bytes=128)
        if fault == "allocation"
        else DEFAULT_ARRAY_ARCHIVE_LIMITS
    )
    if fault == "digest":
        shard = CheckpointShard(
            shard.shard_id,
            "0" * 64,
            shard.byte_count,
            shard.layout_ids,
            metadata=dict(shard.metadata),
        )
    with pytest.raises(RepositoryCorruptionError):
        checkpoint_module._read_shard_payload(repository, shard, shape, dtype, limits)


def test_rank_projection_replaces_old_rank_and_retains_custom_timeouts() -> None:
    base = {
        "PHYDRAX_COORDINATOR_ADDRESS": "old:1234",
        "PHYDRAX_NUM_PROCESSES": "malformed-old-value",
        "PHYDRAX_PROCESS_ID": "old",
        "PHYDRAX_LOCAL_DEVICE_IDS": "7",
        "PHYDRAX_COORDINATOR_BIND_ADDRESS": "old:1234",
        "PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS": "29",
        "PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS": "47",
        "OTHER": "kept",
    }
    projected = RankEnvironment("127.0.0.1:1234", 2, 1).apply(base)
    assert projected == {
        "PHYDRAX_COORDINATOR_ADDRESS": "127.0.0.1:1234",
        "PHYDRAX_NUM_PROCESSES": "2",
        "PHYDRAX_PROCESS_ID": "1",
        "PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS": "29",
        "PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS": "47",
        "OTHER": "kept",
    }
    assert base["PHYDRAX_LOCAL_DEVICE_IDS"] == "7"


@pytest.mark.parametrize(
    ("base", "heartbeat", "initialization"),
    [
        ({}, 100, 300),
        ({"PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS": "29"}, 29, 300),
        ({"PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS": "47"}, 100, 47),
    ],
    ids=["constructor-defaults", "heartbeat-only", "initialization-only"],
)
def test_rank_projection_composes_independent_timeout_overrides(
    base: dict[str, str], heartbeat: int, initialization: int
) -> None:
    projected = RankEnvironment("127.0.0.1:1234", 2, 0).apply(base)
    assert projected["PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS"] == str(heartbeat)
    assert projected["PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS"] == str(initialization)


@pytest.mark.parametrize(
    "key",
    ["PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS", "PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS"],
    ids=["heartbeat", "initialization"],
)
def test_rank_projection_refuses_malformed_present_timeout(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        RankEnvironment("127.0.0.1:1234", 2, 0).apply({key: "0"})


def test_local_launch_rank_projection_retains_disjoint_device_ids() -> None:
    plan = LocalProcessLaunchPlan(2, "127.0.0.1:1234", devices_per_process=2)
    first = plan.rank(0).apply({})
    second = plan.rank(1).apply(first)
    assert first["PHYDRAX_LOCAL_DEVICE_IDS"] == "0,1"
    assert second["PHYDRAX_LOCAL_DEVICE_IDS"] == "2,3"
    assert first["PHYDRAX_COORDINATOR_BIND_ADDRESS"] == "127.0.0.1:1234"
    assert "PHYDRAX_COORDINATOR_BIND_ADDRESS" not in second
