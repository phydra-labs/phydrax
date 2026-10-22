# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import json
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._execution_runtime import ExecutionRuntime
from phydrax._meshcore import meshcore_runtime_identity
from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.discretization._cell_mesh import CellMeshStorage
from phydrax.lifecycle._meshing_sources import read_meshing_source_closure
from phydrax.meshing._device_adaptation import (
    commit_partitioned_mesh_restart,
    retained_partition_states,
    validate_partitioned_mesh_evidence,
)
from phydrax.meshing._restart_distribution import repack_simplex_restart
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.distributed_meshing.native_distributed_records import (
    _new_checks,
    _record_array,
    _require_instance,
)


_CONTRACT_KEYS = (
    "packed_cold_continuation_30",
    "packed_cold_continuation_33",
    "packed_cold_continuation_34",
    "packed_cold_continuation_37",
    "packed_cold_continuation_38",
    "packed_cold_replay_13",
    "packed_cold_replay_16",
    "packed_cold_replay_17",
    "packed_cold_replay_18",
    "packed_cold_replay_41",
    "packed_cold_replay_57",
    "packed_cold_replay_refusal_49",
)


def packed_cold_replay(
    archive: str,
    content_id: str,
    policy_id: str,
    source_result_id: str,
    expected_cells: str,
) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    print(
        "NATIVE_RUNTIME_IDENTITY",
        os.getpid(),
        json.dumps(meshcore_runtime_identity(), sort_keys=True),
        flush=True,
    )
    target, policy, hierarchy = read_meshing_source_closure(
        archive, expected_content_id=content_id, limits=ArrayArchiveLimits()
    )
    checks.setdefault("packed_cold_replay_13", []).append(
        bool(policy.policy_id == policy_id)
    )
    proof = target.collective_evidence
    validate_partitioned_mesh_evidence(target.mesh.storage, proof)
    checks.setdefault("packed_cold_replay_16", []).append(
        bool(proof.uniform_refinement.source.result_id == source_result_id)
    )
    checks.setdefault("packed_cold_replay_17", []).append(
        bool(proof.global_entity_counts[-1] == int(expected_cells))
    )
    checks.setdefault("packed_cold_replay_18", []).append(
        bool(
            not np.array_equal(
                dict(proof.logical_arrays)["epoch/cell_ids"],
                proof.compiled_states.mesh.cell_ids,
            )
        )
    )
    layout, saved, exterior = retained_partition_states(target)
    if np.count_nonzero(np.asarray(saved.mesh.cell_ids) >= 0) > layout.cell_capacity:
        before = tuple(
            (np.array(value, copy=True) for value in jax.tree_util.tree_leaves(saved))
        )
        ranks = jnp.arange(saved.mesh.cell_ids.shape[0], dtype=jnp.int32)[:, None]
        cell_owners = jnp.where(saved.mesh.cell_ids >= 0, ranks, -1)
        vertex_owners = jnp.full(
            saved.mesh.vertex_ids.shape, saved.mesh.vertex_ids.shape[0], dtype=jnp.int32
        )
        for owner in range(saved.mesh.vertex_ids.shape[0]):
            present = jnp.any(
                saved.mesh.vertex_ids[:, :, None]
                == saved.mesh.vertex_ids[owner, None, None, :],
                axis=-1,
            )
            vertex_owners = jnp.minimum(
                vertex_owners, jnp.where(present, owner, saved.mesh.vertex_ids.shape[0])
            )
        vertex_owners = jnp.where(saved.mesh.vertex_ids >= 0, vertex_owners, -1)
        repack = repack_simplex_restart(
            ExecutionRuntime.current().root_group,
            layout,
            saved,
            cell_owners,
            vertex_owners,
            jnp.zeros_like(cell_owners),
            source_result=target,
        )
        checks.setdefault("packed_cold_replay_41", []).append(
            bool(
                np.all(
                    np.asarray(repack.status)
                    & int(phx.discretization.AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
                )
            )
        )
        source_partition = phx.meshing.MeshPartitionPolicy(
            phx.meshing.MeshPartitionKind.PROVIDER, 2, halo_width=2
        )
        distribution = phx.meshing.prepare_mesh_distribution(
            phx.meshing.MeshPart("domain", target), policy=source_partition
        )
        restart_policy = phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
            device_policy=policy.device_policy,
            distribution=distribution,
            partition_policy=phx.meshing.MeshPartitionPolicy(
                phx.meshing.MeshPartitionKind.PROVIDER, 1, halo_width=2
            ),
        )
        with pytest.raises(ValueError, match=r"restart\.saved_states"):
            commit_partitioned_mesh_restart(
                repack,
                phx.meshing.prepare_mesh_adaptation(
                    target,
                    phx.meshing.MarkedMeshAdaptation(hierarchy=hierarchy),
                    policy=restart_policy,
                ),
                proof=phx.meshing.SimplexRestartRepackProof(repack),
            )
        checks["packed_cold_replay_refusal_49"].append(True)
        for value, original in zip(jax.tree_util.tree_leaves(saved), before, strict=True):
            _record_array(checks, "packed_cold_replay_57", value, original)
    return checks


def packed_cold_continuation(archive: str, content_id: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    print(
        "NATIVE_RUNTIME_IDENTITY",
        os.getpid(),
        json.dumps(meshcore_runtime_identity(), sort_keys=True),
        flush=True,
    )
    target, saved_policy, hierarchy = read_meshing_source_closure(
        archive, expected_content_id=content_id, limits=ArrayArchiveLimits()
    )
    M = phx.meshing
    partition_policy = M.MeshPartitionPolicy(
        M.MeshPartitionKind.PROVIDER, 2, halo_width=2, maximum_imbalance=2.0
    )
    distribution = M.prepare_mesh_distribution(
        M.MeshPart("domain", target), policy=partition_policy
    )
    policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=saved_policy.device_policy,
        distribution=distribution,
        partition_policy=partition_policy,
    )
    ids = np.asarray(target.collective_evidence.entity_ids[-1])[
        : target.collective_evidence.global_entity_counts[-1]
    ]
    original = target.collective_evidence.uniform_refinement.source
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(original.geometry)
    ) as budget:
        original_coefficients = original.geometry.source_coordinates(budget=budget)
    continued = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            target, M.MarkedMeshAdaptation(ids[:1], hierarchy=hierarchy), policy=policy
        )
    )
    checks.setdefault("packed_cold_continuation_30", []).append(
        bool(continued.status is M.MeshAdaptationStatus.COMPLETE)
    )
    continued.target.audit.require_passed()
    storage = _require_instance(continued.target.mesh.storage, CellMeshStorage)
    checks.setdefault("packed_cold_continuation_33", []).append(
        bool(isinstance(storage, CellMeshStorage))
    )
    with coordinate_reference_budget(
        queries_upper=(
            cell_geometry_preparation_query_upper(continued.target.geometry)
            + cell_geometry_preparation_query_upper(original.geometry)
        )
    ) as budget:
        checks.setdefault("packed_cold_continuation_34", []).append(
            bool(
                continued.target.geometry.source_coordinates(budget=budget)
                == original.geometry.source_coordinates(
                    storage.coordinate_global_ids, budget=budget
                )
            )
        )
    checks.setdefault("packed_cold_continuation_37", []).append(
        bool(target.collective_evidence.uniform_refinement.source is original)
    )
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(original.geometry)
    ) as budget:
        checks.setdefault("packed_cold_continuation_38", []).append(
            bool(
                original.geometry.source_coordinates(budget=budget)
                == original_coefficients
            )
        )
    return checks
