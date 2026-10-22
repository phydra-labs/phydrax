# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Owner-local restore admission and cold, native packed-geometry regressions.

The four-owner archive is produced in processes that exit before consumers start.
Only addressable shards and bounded digests cross the observation boundary.
"""

import copy
import hashlib
import json
import socket
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Never

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.sharding import (
    Mesh,
    NamedSharding,
    PartitionSpec,
    Sharding,
    SingleDeviceSharding,
)
from jax.typing import DTypeLike

import phydrax as phx
from phydrax import config as pxconfig
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax._fingerprint import logical_array_value_collection_digest, tree_content_equal
from phydrax.discretization._cell_geometry import CellGeometryStorageProjection
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.lifecycle import _distributed_checkpoint as checkpoint
from phydrax.lifecycle._meshing_checkpoint_binding import source_logical_array_name
from phydrax.lifecycle._meshing_source_records import stored_record_content_id
from phydrax.lifecycle._meshing_sources import (
    MeshingSourceValidation,
    register_meshing_source_artifacts,
)
from phydrax.lifecycle._models import CheckpointManifest
from phydrax.lifecycle._repository import POSIXArtifactRepository
from phydrax.lifecycle._restart_topology import (
    MeshingCheckpointState,
    TopologyRestartPolicy,
)
from phydrax.meshing import (
    _distribution as distribution_owner,
    _distribution_migration as migration_owner,
    _publication_lowering as publication_owner,
    _result as result_owner,
)
from phydrax.meshing._device_adaptation import (
    commit_partitioned_mesh_restart,
    prepare_partitioned_mesh_adaptation,
    restore_partitioned_mesh_state,
)
from phydrax.meshing._distribution import MeshDistributionTransition
from phydrax.meshing._distribution_migration import (
    prepare_graph_restart_proposal,
    prepare_ownerlocal_migration_state,
)
from phydrax.meshing._failures import MeshingFailure
from phydrax.meshing._initial_certification import InitialCollectiveMeshEvidence
from phydrax.meshing._restart_distribution import (
    repack_simplex_restart,
    SimplexRestartRepackProof,
)
from phydrax.meshing._result import CollectiveMeshEvidence, CollectiveMeshStorageBinding
from phydrax.meshing.qualification import (
    bind_distributed_checkpoint_source_arrays,
    prepare_distributed_restore_placement,
)
from tests._support.distributed_meshing.device_generation import (
    _prepare as _prepare_initial,
    _publish as _publish_initial,
)
from tests._support.distributed_meshing.native_distributed import _prepare_source
from tests._support.distributed_meshing.restart_builders import (
    build_repository,
    checkpoint_state,
    restart_relation,
    scientific_arrays,
)
from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices
from tools.meshing_qualification._controls import QualificationControls


_MODULE = "tests._support.distributed_meshing.owner_local_restore"


def run_owner_local_restore_workers(
    function: str, directory: str
) -> list[dict[str, Any]]:
    """Launch two Gloo ranks with one CPU device per process."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_on_forced_devices,
                _MODULE,
                function,
                (directory,),
                devices=1,
                env=gloo_rank_environment(coordinator, rank),
            )
            for rank in range(2)
        ]
        return [future.result() for future in futures]


def packet_assembly(directory: str) -> dict[str, Any]:
    """Column packets restore into row owners without a global device buffer."""
    devices = tuple(jax.local_devices())
    mesh = Mesh(np.asarray(devices, dtype=object), ("x",))
    reference = np.arange(24, dtype=np.float64).reshape((4, 6))
    repository = build_repository(Path(directory) / "repository")
    checkpoint.publish_process_checkpoint(
        repository,
        "owner-local-packets",
        "column-owners",
        {
            "value": jax.device_put(
                reference, NamedSharding(mesh, PartitionSpec(None, "x"))
            )
        },
        analysis_plan_id="owner-local-analysis",
        numeric_revision_id="owner-local-revision",
        writer_id="column-writer",
    )
    manifest = checkpoint.assemble_distributed_checkpoint_from_repository(
        repository,
        "owner-local-packets",
        "owner-local-analysis",
        "owner-local-revision",
        "column-owners",
        expected_process_count=1,
    )
    path = jax.tree_util.keystr((jax.tree_util.DictKey("value"),))
    allocated: list[tuple[int, ...]] = []
    zeros = checkpoint.jnp.zeros

    def recorded(
        shape: Any,
        dtype: DTypeLike | None = None,
        *,
        device: jax.Device | Sharding | None = None,
        out_sharding: NamedSharding | PartitionSpec | None = None,
    ) -> jax.Array:
        # Preserve JAX's dynamic shape boundary; observe its canonical output shape.
        value = zeros(shape, dtype, device=device, out_sharding=out_sharding)
        allocated.append(value.shape)
        return value

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(checkpoint.jnp, "zeros", recorded)
        rows = checkpoint.restore_global_array_from_checkpoint(
            repository,
            manifest,
            path,
            NamedSharding(mesh, PartitionSpec("x", None)),
            maximum_host_packet_bytes=reference[:, :3].nbytes,
        )
        row_buffers = list(allocated)
        allocated.clear()
        replicated = checkpoint.restore_global_array_from_checkpoint(
            repository,
            manifest,
            path,
            NamedSharding(mesh, PartitionSpec()),
            maximum_host_packet_bytes=reference[:, :3].nbytes,
        )
        replicated_buffers = list(allocated)
    proof = MeshingSourceValidation(
        (), {"value": rows}, logical_array_value_collection_digest({"value": rows})
    )
    assert proof.array_equal(rows, replicated)
    for row in (0, reference.shape[0] - 1):
        corrupted = replicated.at[row, 0].add(1)
        assert not proof.array_equal(rows, corrupted)
    return {
        "devices": len(devices),
        "row_values_exact": bool(np.array_equal(np.asarray(rows), reference)),
        "row_shards": sorted(
            (shard.index[0].start, tuple(shard.data.shape))
            for shard in rows.addressable_shards
        ),
        "row_buffers": row_buffers,
        "replicated_values_exact": bool(
            np.array_equal(np.asarray(replicated), reference)
        ),
        "replicated_shards": [
            tuple(shard.data.shape) for shard in replicated.addressable_shards
        ],
        "replicated_buffers": replicated_buffers,
    }


def collective_admission(directory: str) -> dict[str, Any]:
    """One rank's invalid restore destinations refuse the restore on every rank."""
    root = Path(directory)
    if jax.process_index() == 0:
        repository = build_repository(root / "repository")
        checkpoint.publish_process_meshing_checkpoint(
            repository,
            "mesh-accepted",
            "source-execution",
            checkpoint_state(),
            {
                name: jax.device_put(value, jax.local_devices()[0])
                for name, value in scientific_arrays().items()
            },
            analysis_plan_id="mesh-analysis",
            numeric_revision_id="accepted-revision",
            writer_id="process-writer",
        )
    multihost_utils.sync_global_devices("owner-local-restore-published")
    repository = build_repository(root / "repository")
    manifest = checkpoint.assemble_distributed_checkpoint_from_repository(
        repository,
        "mesh-accepted",
        "mesh-analysis",
        "accepted-revision",
        "source-execution",
        expected_process_count=1,
    )
    local = SingleDeviceSharding(jax.local_devices()[0])
    names = sorted(scientific_arrays())
    # Rank 1 omits one scientific destination; rank 0's request is valid.
    shardings = {
        name: local for name in (names if jax.process_index() == 0 else names[1:])
    }
    try:
        checkpoint.restore_meshing_checkpoint(
            repository,
            manifest,
            shardings,
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=checkpoint_state().checkpoint_state_id,
        )
    except ValueError as error:
        refusal = str(error)
    else:
        refusal = None
    admitted = checkpoint.restore_meshing_checkpoint(
        repository,
        manifest,
        {name: local for name in names},
        restart_relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=checkpoint_state().checkpoint_state_id,
    )
    return {
        "rank": jax.process_index(),
        "process_count": jax.process_count(),
        "refused": refusal is not None,
        "remote_refusal": refusal is not None and "collective_admission" in refusal,
        "local_refusal": refusal is not None and "destination_shardings" in refusal,
        "admitted_after_refusal": admitted.state.checkpoint_state_id
        == checkpoint_state().checkpoint_state_id,
    }


_PACKED_LIMITS = ArrayArchiveLimits(
    max_members=32768,
    max_manifest_nesting=128,
    max_manifest_bytes=128 * 1024 * 1024,
    max_central_directory_bytes=8 * 1024 * 1024,
)
_PACKED_CHUNK_BYTES = 16 * 1024
_PACKED_HOST_PACKET_BYTES = QualificationControls(
    scenario="native-distributed-lifecycle"
).distributed_host_packet_bytes


def run_packed_owner_workers(
    function: str, directory: str, count: int, *arguments: str
) -> list[dict[str, Any]]:
    """Run a fresh two- or four-process topology, with no surviving producer."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(
                run_on_forced_devices,
                _MODULE,
                function,
                (directory, *arguments),
                devices=1,
                env=pxconfig.environment(
                    pxconfig.RuntimeConfig(
                        bootstrap=pxconfig.RuntimeBootstrap(
                            coordinator_address=coordinator,
                            num_processes=count,
                            process_id=rank,
                            local_device_ids=(0,),
                        ),
                        cpu_collectives_implementation="gloo",
                    ),
                    base={},
                ),
            )
            for rank in range(count)
        ]
        return [future.result() for future in futures]


def _packed_repository(directory: str) -> POSIXArtifactRepository:
    return build_repository(
        Path(directory),
        maximum_chunk_bytes=_PACKED_CHUNK_BYTES,
        maximum_metadata_bytes=128 * 1024 * 1024,
    )


def _packed_group(permuted: bool) -> ExecutionGroup:
    devices = tuple(jax.devices())
    if permuted:
        # Non-monotone for four devices, reversed for two. Never relabel owners.
        order = (2, 0, 3, 1) if len(devices) == 4 else (1, 0)
        devices = tuple(devices[index] for index in order)
    return ExecutionGroup(
        ExecutionGroupSpec(
            "cold-packed-owner-restore",
            tuple(sorted({device.process_index for device in devices})),
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("owner", len(devices)),),
        ),
        devices,
    )


def _science_row(owner: int) -> dict[str, np.ndarray]:
    """Distinct IDs, signed-zero fields, history, RNG words, and an accepted clock."""
    return {
        "entity_ids": np.asarray([[101 + 71 * owner, 509 + 113 * owner]], np.int64),
        "field": np.asarray([[owner + 0.25, -0.0]], np.float64),
        "history": np.asarray([[owner - 0.5, owner + 0.75]], np.float64),
        "rng": np.asarray([[0xF0000000 + owner, 19 + 17 * owner]], np.uint32),
        "clock": np.asarray([[0.125, 7.0]], np.float64),
    }


def _bank_digest(arrays: dict[str, jax.Array]) -> str:
    return logical_array_value_collection_digest(
        arrays, maximum_chunk_bytes=_PACKED_CHUNK_BYTES
    )


def _local_bank_rows(arrays: dict[str, jax.Array]) -> dict[str, Any]:
    """Record hashes of local rows, never host-global scientific buffers."""
    records: dict[str, Any] = {}
    for name, value in arrays.items():
        rows: dict[str, str] = {}
        for shard in value.addressable_shards:
            host = np.asarray(shard.data)
            if not value.shape:
                rows["scalar"] = hashlib.sha256(host.tobytes()).hexdigest()
                continue
            for local, row in enumerate(range(*shard.index[0].indices(value.shape[0]))):
                rows[str(row)] = hashlib.sha256(host[local].tobytes()).hexdigest()
        records[name] = {"shape": list(value.shape), "dtype": value.dtype.str, "rows": rows}
    return records


def _target_identity(target: Any) -> dict[str, Any]:
    storage = target.mesh.storage
    assert storage is not None
    return {
        "result": target.result_id,
        "topology": target.mesh.topology_id,
        "geometry": cell_geometry_id(target.geometry),
        "geometry_layout": target.geometry.geometry_layout_id,
        "partition": storage.partition_index,
    }


def _publish_graph_restart(
    current: phx.meshing.CellMeshingResult,
) -> phx.meshing.MeshAdaptationResult:
    """Earn graph/repack ancestry with real traffic and scientific history packets."""
    storage = current.mesh.storage
    evidence = current.collective_evidence
    assert storage is not None and isinstance(evidence, CollectiveMeshEvidence)
    layout, raw, _ = restore_partitioned_mesh_state(storage, evidence)
    distribution = phx.meshing.prepare_mesh_distribution(
        phx.meshing.MeshPart("domain", current),
        policy=phx.meshing.MeshPartitionPolicy(
            phx.meshing.MeshPartitionKind.PROVIDER, 4, halo_width=2
        ),
    )
    # The two cells currently owned by zero each cost four; the other six
    # cost one. Keeping those owners violates 1.5 imbalance (8 / 3.5), so the
    # genuine graph solver must migrate work instead of merely retaining labels.
    graph = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.GRAPH, 4, halo_width=2, maximum_imbalance=1.5
    )
    adaptation = phx.meshing.prepare_mesh_adaptation(
        current,
        phx.meshing.MarkedMeshAdaptation(),
        policy=phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
            device_policy=phx.discretization.AdaptiveSimplexPolicy(
                vertex_capacity=32, cell_capacity=32
            ),
            distribution=distribution,
            partition_policy=graph,
        ),
    )
    partitioned = prepare_partitioned_mesh_adaptation(adaptation)
    cells = jnp.where(raw.mesh.cell_active, raw.mesh.cell_ids, -1)
    vertices = jnp.where(raw.mesh.vertex_active, raw.mesh.vertex_ids, -1)
    bound = prepare_ownerlocal_migration_state(
        partitioned,
        raw,
        cell_field_ids=cells,
        cell_fields=jnp.where(cells >= 0, cells * 0.25 + 0.5, 0.0),
        vertex_field_ids=vertices,
        vertex_fields=jnp.where(vertices >= 0, vertices * 0.125 - 0.25, 0.0),
        cell_epochs=jnp.where(cells >= 0, 1, 0).astype(jnp.int64),
        cell_history=(
            jnp.where(cells >= 0, cells + 17, 0).astype(jnp.uint32),
            jnp.where(cells >= 0, cells + 2**40, 0).astype(jnp.uint64),
            jnp.where(cells >= 0, cells * 0.5 + 0.125, 0.0),
        ),
        vertex_history=(jnp.where(vertices >= 0, vertices * 0.25, -0.0),),
    )
    proposal = prepare_graph_restart_proposal(
        partitioned.parts,
        raw,
        bound,
        graph,
        vertex_weights=jnp.where(
            bound.workset.cell_valid,
            jnp.where(bound.workset.cell_owner == 0, 4, 1),
            0,
        ).astype(jnp.int64),
    )
    assert bound.forest_cell_owners is not None
    assert bound.forest_vertex_owners is not None
    receipt = repack_simplex_restart(
        _packed_group(False),
        layout,
        raw,
        bound.forest_cell_owners,
        bound.forest_vertex_owners,
        proposal.requested_target_owners,
        cell_history=bound.cell_history,
        vertex_history=bound.vertex_history,
        source_result=current,
        graph_proposal=proposal,
    )
    for shard in receipt.status.addressable_shards:
        assert not np.any(np.asarray(shard.data))
    traffic = receipt.cell_migration_counts
    assert traffic is not None
    assert int(jax.device_get(jnp.sum(traffic) - jnp.trace(traffic))) > 0
    proof = SimplexRestartRepackProof(receipt)
    proof.require_receipt(receipt)
    forged = replace(
        receipt,
        cell_history=(receipt.cell_history[0] + jnp.uint32(1), *receipt.cell_history[1:]),
    )

    def refuse_second_planner(*args: Any, **kwargs: Any) -> Never:
        raise AssertionError("An accepted weighted restart must not invoke adaptation rebalancing")

    route_owners = migration_owner._restart_solver_workset_owners
    solver_view = migration_owner.restart_solver_family_view
    project = publication_owner._project
    observed = []
    publication_source = result_owner._restart_publication_source
    source_owners = []

    def observe_owners(raw_ids: jax.Array, owners: jax.Array, work: Any) -> Any:
        result = route_owners(raw_ids, owners, work)
        forged_work = replace(
            work,
            cell_ids=jnp.where(work.cell_valid, work.cell_ids + jnp.max(raw_ids) + 1, work.cell_ids),
        )
        _, valid_ids = route_owners(raw_ids, owners, forged_work)
        _, valid_owners = route_owners(raw_ids, jnp.full_like(owners, -1), work)
        assert not bool(jax.device_get(valid_ids))
        assert not bool(jax.device_get(valid_owners))
        return result

    def observe_solver_view(partitioned: Any, states: Any) -> Any:
        work, forest = solver_view(partitioned, states)
        physical = jnp.sum(states.mesh.cell_active, axis=1)
        scientific = jnp.sum(forest.mesh.cell_active, axis=1)
        assert bool(jax.device_get(jnp.any((physical == 0) & (scientific > 0))))
        observed.append(True)
        return work, forest

    def observe_projection(
        arrays: Any, tables: Any, identifiers: Any, owners: Any, **kwargs: Any,
    ) -> Any:
        rows, found = publication_owner._lookup(
            tables[-1], arrays["closure/cell_ids"][..., None], kwargs["counts"][-1],
        )
        valid = arrays["closure/cell_valid"]
        corners = arrays["closure/cell_vertices"]
        corner_rows, corner_found = publication_owner._lookup(
            tables[0], corners[..., None], kwargs["counts"][0],
        )
        vertex_mask = jnp.broadcast_to(valid[..., None], corners.shape)
        _, _, fits = jax.vmap(
            lambda keys, mask: publication_owner._compact(keys, mask, kwargs["vertex_capacity"])
        )(corners.reshape(corners.shape[0], -1, 1), vertex_mask.reshape(corners.shape[0], -1))
        _, cell_mask, _ = jax.vmap(
            lambda keys, mask: publication_owner._compact(keys, mask, valid.shape[1])
        )(arrays["closure/cell_ids"][..., None], valid)
        predicates = {
            "cell_ids_present": jnp.all(~valid | found),
            "cell_owners_match": jnp.all(~valid | (owners[-1][rows] == arrays["closure/cell_owner"])),
            "vertex_ids_present": jnp.all(~vertex_mask | corner_found),
            "coordinates_match": jnp.all(
                ~valid[..., None, None]
                | (arrays["coordinates"][corner_rows] == arrays["closure/cell_coordinates"])
            ),
            "vertex_capacity": jnp.all(fits),
            "unique_cells": jnp.all(jnp.sum(cell_mask, axis=1) == jnp.sum(valid, axis=1)),
        }
        flags = np.asarray(jax.device_get(jnp.stack(tuple(predicates.values()))), dtype=np.bool_)
        assert np.all(flags), dict(zip(predicates, flags.tolist(), strict=True))
        return project(arrays, tables, identifiers, owners, **kwargs)

    def observe_publication_source(partitioned: Any, arrays: Any) -> Any:
        owner = publication_source(partitioned, arrays)
        if owner is not None and isinstance(partitioned, phx.meshing.PartitionedAdaptiveSimplex):
            assert owner is partitioned.restart_repack
            missing = tuple((name, value) for name, value in arrays if name != "placement/proof_content_id")
            with pytest.raises(MeshingFailure):
                publication_source(partitioned, missing)
            for field in ("restart_repack", "restart_proof"):
                forged_source = eqx.tree_at(
                    lambda value: getattr(value, field), partitioned, None,
                )
                with pytest.raises(MeshingFailure):
                    publication_source(forged_source, arrays)
            source_owners.append(True)
        return owner

    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(distribution_owner, "prepare_partitioned_rebalance_proposal", refuse_second_planner)
        guard.setattr(migration_owner, "_restart_solver_workset_owners", observe_owners)
        guard.setattr(migration_owner, "restart_solver_family_view", observe_solver_view)
        guard.setattr(publication_owner, "_project", observe_projection)
        guard.setattr(result_owner, "_restart_publication_source", observe_publication_source)
        with pytest.raises(MeshingFailure):
            commit_partitioned_mesh_restart(forged, adaptation, proof=proof)
        (result,) = commit_partitioned_mesh_restart(
            receipt, adaptation, proof=proof
        ).local_results
    assert observed == [True]
    assert source_owners
    assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert result.distribution is not None
    assert result.distribution.restart_repack is not None
    assert result.distribution.restart_repack_id == proof.content_id
    retained = result.distribution.restart_repack
    retained_graph = retained.graph_proposal
    assert retained_graph is not None
    owned = proposal.workset.cell_valid & (
        proposal.workset.cell_owner == jnp.arange(4, dtype=jnp.int32)[:, None]
    )
    expected_weights = jnp.bincount(
        jnp.where(owned, proposal.parts, 0).reshape(-1),
        weights=jnp.where(owned, proposal.vertex_weights, 0).reshape(-1).astype(jnp.float64),
        length=4,
    )
    assert bool(jax.device_get(jnp.array_equal(
        result.distribution.target.evidence.part_weights, expected_weights,
    )))
    assert float(jax.device_get(result.distribution.target.evidence.imbalance)) <= graph.maximum_imbalance
    target_storage = result.target.mesh.storage
    assert target_storage is not None
    logical = dict(target_storage.logical_arrays)
    _, _, _, invalid_weights = distribution_owner._restart_graph_weight_values(
        proposal.workset, -proposal.vertex_weights, proposal.parts,
        logical["cell_global_ids"], logical["cell_owners"],
        target_storage.global_entity_counts[-1], target_storage.partition_count,
    )
    assert not bool(jax.device_get(invalid_weights))
    assert logical_array_value_collection_digest({
        "owners": retained.requested_target_owners,
        "weights": retained_graph.vertex_weights,
        "rng32": retained.cell_history[0],
        "rng64": retained.cell_history[1],
    }) == logical_array_value_collection_digest({
        "owners": proposal.requested_target_owners,
        "weights": proposal.vertex_weights,
        "rng32": receipt.cell_history[0],
        "rng64": receipt.cell_history[1],
    })
    return result


def _packed_scientific_source(
    kind: str,
) -> tuple[phx.meshing.CellMeshingResult, dict[str, Any], dict[str, Any]]:
    if kind == "empty":
        # Initial scientific owners may be empty. Adaptive publication would
        # legitimately rebalance them, irrespective of the imbalance tolerance.
        prepared = _prepare_initial((0, 2))
        target = _publish_initial(prepared)
        evidence = target.collective_evidence
        assert isinstance(evidence, InitialCollectiveMeshEvidence)
        premise = evidence.premise
        return target, {
            "collective_target": target,
            "generation_source": premise.authored_source,
            "generation_specification": premise.specification,
            "generation_schedule": premise.schedule,
            "generation_layout": premise.layout,
        }, {
            "source_topology_id": None,
            "source_geometry_id": None,
            "lineage_id": None,
            "transfer_id": None,
            "request_id": premise.specification.specification_id,
            "source_revision_id": premise.source_revision,
        }
    assert kind in ("adaptive", "graph", "planar")
    transfer = None
    if kind == "planar":
        from tests.unit.meshing.test_initial_planar_support import _native_planar_initial

        source, transfer = _native_planar_initial(width=4.0)
    else:
        source = _prepare_source("halo_depth", {}).source
        assert source.certification is not None
        source.certification.require_passed()
    if kind == "planar":
        assert source.mesh.storage is not None
        policy = phx.meshing.MeshPartitionPolicy(
            phx.meshing.MeshPartitionKind.PROVIDER, 4,
            halo_width=source.mesh.storage.neighborhood_depth,
        )
    else:
        policy = phx.meshing.MeshPartitionPolicy(
            phx.meshing.MeshPartitionKind.MORTON, 4, halo_width=2
        )
    distribution = phx.meshing.prepare_mesh_distribution(
        phx.meshing.MeshPart("domain", source), policy=policy
    )
    adaptation = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
        compatibility=(
            phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT
            if kind == "planar" else phx.meshing.BisectionCompatibility.REJECT
        ),
        device_policy=phx.discretization.AdaptiveSimplexPolicy(
            vertex_capacity=1000 if kind == "planar" else 32,
            cell_capacity=2000 if kind == "planar" else 32,
        ),
        distribution=distribution,
        partition_policy=policy,
        association_transfer=transfer,
    )
    if kind == "planar":
        partitioned = prepare_partitioned_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                source, phx.meshing.MarkedMeshAdaptation(), policy=adaptation,
            ),
        )
    else:
        partitioned = phx.meshing.partition_adaptive_simplex(
            phx.meshing.prepare_adaptive_simplex(source, policy=adaptation)
        )
    update = phx.discretization.refine_adaptive_simplex_parts(
        partitioned.layout,
        type(partitioned).execution_parts(partitioned),
        partitioned.states,
        partitioned.states.mesh.cell_active,
    )
    for shard in update.report.status.addressable_shards:
        assert not np.any(np.asarray(shard.data))
    (result,) = phx.meshing.commit_partitioned_adaptive_simplex(
        partitioned, update.state
    ).local_results
    assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    if kind == "planar":
        # Match the routine checkpoint's initial/refine/coarsen/refine epoch4.
        # Each accepted predecessor remains reachable through its actual evidence.
        for operation in (
            phx.discretization.coarsen_adaptive_simplex_parts,
            phx.discretization.refine_adaptive_simplex_parts,
        ):
            assert result.distribution is not None
            adaptation = phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
                compatibility=adaptation.compatibility,
                device_policy=adaptation.device_policy,
                distribution=result.distribution.target,
                partition_policy=policy,
                association_transfer=transfer,
            )
            partitioned = prepare_partitioned_mesh_adaptation(
                phx.meshing.prepare_mesh_adaptation(
                    result.target, phx.meshing.MarkedMeshAdaptation(), policy=adaptation,
                ),
            )
            update = operation(
                partitioned.layout,
                type(partitioned).execution_parts(partitioned),
                partitioned.states,
                partitioned.states.mesh.cell_active,
            )
            (result,) = phx.meshing.commit_partitioned_adaptive_simplex(
                partitioned, update.state,
            ).local_results
            assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    if kind == "graph":
        result = _publish_graph_restart(result.target)
    target = result.target
    evidence = target.collective_evidence
    assert isinstance(evidence, phx.meshing.CollectiveMeshEvidence)
    if kind == "planar":
        assert result.lineage is not None and result.transfer is not None
        initial = source.collective_evidence
        assert isinstance(initial, InitialCollectiveMeshEvidence)
        premise = initial.premise
        return target, {
            "collective_target": target,
            "generation_source": premise.authored_source,
            "generation_specification": premise.specification,
            "generation_schedule": premise.schedule,
            "generation_layout": premise.layout,
            "accepted_data": {"adaptation_results": (result,)},
        }, {
            "source_topology_id": evidence.source.mesh.topology_id,
            "source_geometry_id": cell_geometry_id(evidence.source.geometry),
            "request_id": evidence.preparation.request.request_id,
            "lineage_id": result.lineage.lineage_id,
            "transfer_id": result.transfer.transfer_id,
            "source_revision_id": premise.source_revision,
        }
    assert source.certification is not None
    return target, {
        "certification_inputs": source.certification.request,
        "report": source.certification,
        "associations": source.associations,
        "collective_target": target,
        "accepted_data": {"adaptation_results": (result,)},
    }, {
        "source_topology_id": evidence.source.mesh.topology_id,
        "source_geometry_id": cell_geometry_id(evidence.source.geometry),
        "request_id": evidence.preparation.request.request_id,
        "lineage_id": None if result.lineage is None else result.lineage.lineage_id,
        "transfer_id": None if result.transfer is None else result.transfer.transfer_id,
        "source_revision_id": source.certification.request.source_revision,
    }


def packed_geometry_publish(directory: str, kind: str = "adaptive") -> dict[str, Any]:
    """Publish real adaptive geometry, or a separate genuinely empty initial owner."""
    try:
        assert jax.process_count() == 4
        register_meshing_source_artifacts()
        target, closure, references = _packed_scientific_source(kind)
        target.audit.require_passed()
        storage = target.mesh.storage
        assert storage is not None
        projection = storage.geometry_projection
        assert isinstance(projection, CellGeometryStorageProjection)
        rank = jax.process_index()
        assert storage.partition_index == rank
        receipt = dict(projection.addressable_arrays(rank))
        bound = bind_distributed_checkpoint_source_arrays(closure, _PACKED_LIMITS)
        assert not bound.missing, bound.missing_details
        assert any("/geometry_projection/" in name for name in bound.bank)
        payload = checkpoint_state().to_payload()
        payload.update(references)
        if payload["source_revision_id"] is None:
            payload["source_revision_id"] = checkpoint_state().source_revision_id
        evidence = target.collective_evidence
        assert isinstance(
            evidence, (phx.meshing.CollectiveMeshEvidence, InitialCollectiveMeshEvidence)
        )
        logical = dict(storage.logical_arrays)
        mesh_epoch = {"empty": 0, "adaptive": 1, "graph": 2, "planar": 1}[kind]
        group = _packed_group(False)
        mesh_arrays = {
            "mesh-entity-ids": logical["vertex_global_ids"],
            "mesh-coordinates": logical["coordinates"],
            "mesh-topology": logical["cell_vertices"],
            "mesh-epoch": jax.device_put(
                np.asarray(mesh_epoch, dtype=np.int64),
                group.named_sharding(PartitionSpec()),
            ),
            "mesh-evidence": evidence.partition_checks,
        }
        mesh_roles = {
            "mesh-entity-ids": "entity-ids",
            "mesh-coordinates": "geometry",
            "mesh-topology": "topology",
            "mesh-epoch": "epoch",
            "mesh-evidence": "evidence",
        }
        if kind != "empty":
            mesh_arrays["mesh-lineage"] = logical["epoch/parents"]
            mesh_roles["mesh-lineage"] = "lineage"
        payload.update(
            topology_id=target.mesh.topology_id,
            geometry_id=cell_geometry_id(target.geometry),
            result_id=target.result_id,
            coordinate_contract_id=target.coordinate_contract.spatial_id,
            evidence_id=evidence.evidence_id,
            mesh_epoch=mesh_epoch,
            array_roles=(
                *tuple((name, "state") for name in _science_row(rank)),
                *tuple(mesh_roles.items()),
            ),
            source_closure=closure,
            source_owner_index=rank,
            source_array_bindings=bound.bindings,
            source_logical_arrays=bound.bank,
            limits=_PACKED_LIMITS,
        )
        state = MeshingCheckpointState(**payload)
        sharding = group.named_sharding(PartitionSpec("owner"))
        arrays = {
            name: jax.make_array_from_process_local_data(
                sharding, value, global_shape=(4, *value.shape[1:])
            )
            for name, value in _science_row(rank).items()
        }
        assert all(not value.is_fully_addressable for value in arrays.values())
        arrays.update(mesh_arrays)
        checkpoint.publish_process_meshing_checkpoint(
            _packed_repository(directory),
            "cold-packed-geometry",
            "packed-producer",
            state,
            arrays,
            analysis_plan_id="packed-analysis",
            numeric_revision_id="packed-revision",
            writer_id=f"packed-writer-{rank}",
            limits=_PACKED_LIMITS,
        )
        multihost_utils.sync_global_devices("cold-packed-geometry-published")
        (Path(directory) / f"bank-rows-{rank}.json").write_text(
            json.dumps(_local_bank_rows(bound.bank)), encoding="utf-8"
        )
        return {
            "rank": rank,
            "state_id": state.checkpoint_state_id,
            "closure_id": state.source_closure_id,
            "identity": _target_identity(target),
            "receipt_digest": _bank_digest(receipt),
            "empty": int(np.asarray(receipt["geometry/owned_cell_count"])) == 0,
            "graph": kind == "graph",
            "planar": kind == "planar",
            "transition_id": (
                closure["accepted_data"]["adaptation_results"][0].distribution.transition_id
                if kind == "graph" else None
            ),
        }
    finally:
        jax.distributed.shutdown()


def _packed_archive(
    directory: str,
) -> tuple[POSIXArtifactRepository, CheckpointManifest, tuple[MeshingCheckpointState, ...]]:
    register_meshing_source_artifacts()
    repository = _packed_repository(directory)
    manifest = checkpoint.assemble_distributed_checkpoint_from_repository(
        repository,
        "cold-packed-geometry",
        "packed-analysis",
        "packed-revision",
        "packed-producer",
        expected_process_count=4,
        limits=_PACKED_LIMITS,
    )
    states = checkpoint.read_meshing_checkpoint_states(
        repository, manifest, limits=_PACKED_LIMITS
    )
    return repository, manifest, states


def _publish_restored_planar(state: MeshingCheckpointState) -> None:
    """Consume authenticated historical owners through current PLC publication."""
    from phydrax.meshing import _plc_planar_support as support

    closure = state.source_closure
    assert closure is not None
    historical = closure["collective_target"]
    old = closure["accepted_data"]["adaptation_results"][0]
    transfer = old.policy.association_transfer
    sources = tuple(
        owner.source_closure["collective_target"]
        for owner in state.source_owner_states
        if owner.source_closure is not None
    )
    layout, raw, _ = restore_partitioned_mesh_state(
        historical.mesh.storage, historical.collective_evidence,
    )

    def owners(ids: jax.Array) -> jax.Array:
        ranks = jnp.broadcast_to(
            jnp.arange(ids.shape[0], dtype=jnp.int32)[:, None], ids.shape,
        ).reshape(-1)
        flat = ids.reshape(-1)
        order = jnp.lexsort((ranks, flat))
        positions = jnp.searchsorted(flat[order], ids)
        return jnp.where(ids >= 0, ranks[order][positions], -1).astype(jnp.int32)

    cells, vertices = owners(raw.mesh.cell_ids), owners(raw.mesh.vertex_ids)
    requested = jnp.where(cells >= 0, cells // (4 // jax.process_count()), 0)
    receipt = repack_simplex_restart(
        _packed_group(False), layout, raw, cells, vertices, requested,
        source_result=historical,
    )
    proof = SimplexRestartRepackProof(receipt)
    partition = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.PROVIDER, jax.process_count(),
        halo_width=old.policy.partition_policy.halo_width,
    )
    prepared = phx.meshing.prepare_mesh_adaptation(
        historical, phx.meshing.MarkedMeshAdaptation(),
        policy=phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
            device_policy=old.policy.device_policy,
            distribution=old.distribution.target, partition_policy=partition,
            association_transfer=transfer,
        ),
    )
    observed = []
    propagate = support.planar_propagate

    def current_support(*args: Any, **kwargs: Any) -> Any:
        current, lineage, target = args[1:4]
        assert current is not historical
        assert current.mesh is target
        assert current.collective_certificates is not None
        evidence = current.collective_evidence
        assert evidence is not None
        arrays = dict(evidence.logical_arrays)
        for record in (*current.patches, *current.zones, *current.labels):
            scope = record.scope.projection
            assert scope is not None
            publication = scope.publication
            type(publication).require_source(publication, evidence.logical_arrays)
            for name, value in publication.source_arrays:
                if not name.startswith("closure/"):
                    assert arrays[name] is value
            name, value = next(
                (name, value) for name, value in publication.source_arrays
                if not name.startswith("closure/")
            )
            cloned = tuple(
                (key, value.copy() if key == name else bank)
                for key, bank in evidence.logical_arrays
            )
            with pytest.raises(ValueError, match="different bank objects"):
                type(publication).require_source(publication, cloned)
        for dimension in range(3):
            record = lineage.entity_lineage(dimension)
            assert record.source_entity_set_id == current.mesh.entity_set(dimension).entity_set_id
            assert record.target_entity_set_id == target.entity_set(dimension).entity_set_id
        observed.append(current)
        return propagate(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(support, "planar_propagate", current_support)
        (result,) = commit_partitioned_mesh_restart(
            receipt, prepared, proof=proof, historical_sources=sources,
        ).local_results
    assert len(observed) == 1
    assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert result.lineage is not None
    for fault in ("missing", "duplicate", "foreign"):
        lineage = result.lineage
        if jax.process_index() == 0:
            record = lineage.entity_lineage(0)
            ids = record.source_global_ids
            changed = ids[:-1] if fault == "missing" else (
                jnp.concatenate((ids, ids[:1])) if fault == "duplicate"
                else ids.at[0].add(100000)
            )
            record = eqx.tree_at(lambda value: value.source_global_ids, record, changed)
            lineage = eqx.tree_at(
                lambda value: value.entities, lineage, (record, *lineage.entities[1:]),
            )
        with pytest.raises(MeshingFailure):
            support.restart_planar_lineage(
                historical, lineage, observed[0].mesh, result.target.mesh,
            )
        support.restart_planar_lineage(
            historical, result.lineage, observed[0].mesh, result.target.mesh,
        )


def packed_geometry_restore(
    directory: str, published_json: str, permutation: str, failure: str = ""
) -> dict[str, Any]:
    """Cold restore every local historical row without relocating old owner zero."""
    from phydrax.lifecycle import _meshing_sources as source_owner
    from phydrax.meshing import _collective_organization as organization

    try:
        published = json.loads(published_json)
        repository, manifest, states = _packed_archive(directory)
        group = _packed_group(permutation == "permuted")
        placement = prepare_distributed_restore_placement(
            states, manifest, group, _PACKED_LIMITS
        )
        current = group.named_sharding(PartitionSpec("owner"))
        expected_owners: list[int] = []
        for index in current.addressable_devices_indices_map((4,)).values():
            assert index is not None
            selection = index[0]
            assert isinstance(selection, slice)
            expected_owners.extend(range(*selection.indices(4)))
        expected = tuple(sorted(expected_owners))
        assert placement.source_owners == expected
        assert placement.decoded_owners == expected
        selected = placement.decoded_owners[0]
        calls: list[int] = []
        addressable = CellGeometryStorageProjection.addressable_arrays
        injected = False
        refused = False
        stage, _, failed_rank = failure.partition("-")
        replayed_transitions: list[str] = []
        transition_constructor = MeshDistributionTransition.from_stored_fields

        def same_local_array(left: Any, right: Any) -> bool:
            if isinstance(left, jax.Array) and isinstance(right, jax.Array):
                return _local_bank_rows({"array": left}) == _local_bank_rows({"array": right})
            a, b = np.asarray(left), np.asarray(right)
            return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

        def observed_transition(
            cls: type[MeshDistributionTransition],
            value: MeshDistributionTransition,
            /,
            *,
            collective: bool = False,
            restart_proof: SimplexRestartRepackProof | None = None,
        ) -> MeshDistributionTransition:
            nonlocal injected
            assert cls is MeshDistributionTransition
            receipt = value.restart_repack
            selected_graph = receipt is not None and receipt.graph_proposal is not None
            if selected_graph:
                assert collective
                assert isinstance(restart_proof, SimplexRestartRepackProof)
                restart_proof.require_receipt(receipt)
            original_owners = value.source_cell_owners
            fail_here = (
                selected_graph and stage in ("graphowner", "graphproof")
                and not injected and jax.process_index() == int(failed_rank)
            )
            if fail_here:
                injected = True
                if stage == "graphproof":
                    restart_proof = None
                else:
                    assert original_owners is not None
                    if isinstance(original_owners, jax.Array):
                        replacement = jax.make_array_from_single_device_arrays(
                            original_owners.shape,
                            original_owners.sharding,
                            [
                                jax.device_put(
                                    np.full(shard.data.shape, -1, original_owners.dtype),
                                    shard.device,
                                )
                                for shard in original_owners.addressable_shards
                            ],
                        )
                    else:
                        replacement = np.full_like(original_owners, -1)
                    object.__setattr__(value, "source_cell_owners", replacement)
            try:
                rebuilt = transition_constructor(
                    value, collective=collective, restart_proof=restart_proof
                )
            finally:
                if fail_here and stage == "graphowner":
                    object.__setattr__(value, "source_cell_owners", original_owners)
            if selected_graph:
                assert restart_proof is not None
                assert tree_content_equal(value, rebuilt, array_equal=same_local_array)
                assert rebuilt.transition_id == value.transition_id
                assert rebuilt.restart_repack_id == restart_proof.content_id
                replayed_transitions.append(rebuilt.transition_id)
            return rebuilt

        def observed(
            self: CellGeometryStorageProjection, owner: int, /
        ) -> tuple[tuple[str, jax.Array], ...]:
            nonlocal injected
            fail_here = (
                bool(failure) and not injected
                and jax.process_index() == int(failed_rank)
            )
            if fail_here and stage == "before":
                injected = True
                raise ValueError("packed projection injected failure before local receipt")
            result = addressable(self, owner)
            if fail_here and stage == "after":
                injected = True
                raise ValueError("packed projection injected failure after local receipt")
            calls.append(owner)
            return result

        def restore_once() -> checkpoint.MeshingCheckpointRestore:
            return checkpoint.restore_meshing_checkpoint(
                repository,
                manifest,
                placement.shardings,
                restart_relation(),
                TopologyRestartPolicy(allow_topology_change=True),
                expected_state_id={
                    record["rank"]: record["state_id"] for record in published
                },
                source_owner_index=selected,
                decoded_source_owners=placement.decoded_owners,
                limits=_PACKED_LIMITS,
                maximum_host_packet_bytes=_PACKED_HOST_PACKET_BYTES,
            )

        prepare_source_operand = organization._organization_source_operand

        def observed_operand(value: Any, target: jax.Array) -> jax.Array:
            result = prepare_source_operand(value, target)
            assert result.shape == value.shape
            assert result.dtype == value.dtype
            result_sharding, target_sharding = result.sharding, target.sharding
            assert isinstance(result_sharding, NamedSharding)
            assert isinstance(target_sharding, NamedSharding)
            assert result_sharding.mesh == target_sharding.mesh
            if not isinstance(value, jax.Array) or value.is_fully_addressable:
                assert result.sharding.is_fully_replicated
                assert {shard.device for shard in result.addressable_shards} == set(
                    target.sharding.addressable_devices
                )
                assert all(
                    shard.data.shape == value.shape for shard in result.addressable_shards
                )
            else:
                assert result is value
            return result

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(CellGeometryStorageProjection, "addressable_arrays", observed)
            patch.setattr(organization, "_organization_source_operand", observed_operand)
            if published[0]["graph"]:
                patch.setattr(
                    MeshDistributionTransition, "from_stored_fields",
                    classmethod(observed_transition),
                )
            if failure:
                if stage in ("operandmissing", "operandforeign"):
                    from phydrax._execution_array import _place_process_local_array

                    prepare_operand = organization._organization_source_operand

                    def refused_operand(value: Any, target: jax.Array) -> jax.Array:
                        nonlocal injected
                        complete_local = (
                            not isinstance(value, jax.Array) or value.is_fully_addressable
                        )
                        if complete_local and not injected and jax.process_index() == int(failed_rank):
                            injected = True
                            if stage == "operandmissing":
                                value = None
                            else:
                                target_sharding = target.sharding
                                assert isinstance(target_sharding, NamedSharding)
                                mesh = target_sharding.mesh
                                foreign = NamedSharding(
                                    Mesh(mesh.devices.reshape(-1)[::-1].reshape(mesh.devices.shape),
                                         mesh.axis_names),
                                    PartitionSpec(),
                                )
                                # A complete dense source, not an addressable shard
                                # masquerading as the original global bank.
                                value = _place_process_local_array(
                                    jax.device_put(value),
                                    foreign,
                                )
                        return prepare_operand(value, target)

                    def forbidden_binding(*args: Any, **kwargs: Any) -> Never:
                        raise AssertionError("Source binding entered after operand refusal")

                    with pytest.MonkeyPatch.context() as operand_patch:
                        operand_patch.setattr(
                            organization, "_organization_source_operand", refused_operand
                        )
                        operand_patch.setattr(
                            organization, "_bind_vertex_source_rows", forbidden_binding
                        )
                        with pytest.raises(phx.meshing.MeshingFailure) as refused_operand_error:
                            restore_once()
                        assert refused_operand_error.value.category is (
                            phx.meshing.MeshingFailureCategory.LINEAGE_FAILED
                        )
                elif stage == "preflight":
                    replay_plan = source_owner._meshing_source_replay_plan

                    def refused_plan(
                        *args: Any, **kwargs: Any
                    ) -> source_owner._MeshingSourceReplayPlan:
                        nonlocal injected
                        plan = replay_plan(*args, **kwargs)
                        if jax.process_index() == int(failed_rank):
                            injected = True
                            raise ValueError("packed canonical replay preflight failure")
                        return plan

                    def forbidden_replay(*args: Any, **kwargs: Any) -> Never:
                        raise AssertionError(
                            "Numerical epoch/projection replay entered after peer preflight failure"
                        )

                    with pytest.MonkeyPatch.context() as preflight_patch:
                        preflight_patch.setattr(
                            source_owner, "_meshing_source_replay_plan", refused_plan
                        )
                        preflight_patch.setattr(
                            source_owner, "validate_partitioned_mesh_evidence",
                            forbidden_replay,
                        )
                        preflight_patch.setattr(
                            CellGeometryStorageProjection, "__init__", forbidden_replay
                        )
                        preflight_patch.setattr(
                            InitialCollectiveMeshEvidence, "require_passed",
                            forbidden_replay,
                        )
                        with pytest.raises(ValueError):
                            restore_once()
                elif stage == "constructor":
                    validate_epochs = checkpoint.validate_collective_mesh_source_epochs

                    def forbidden_constructor(*args: Any, **kwargs: Any) -> Never:
                        raise AssertionError(
                            "Owner validation reran a changed projection constructor"
                        )

                    with pytest.MonkeyPatch.context() as constructor_patch:
                        def changed_after_replay(
                            *args: Any, **kwargs: Any
                        ) -> source_owner.MeshingSourceValidation:
                            nonlocal injected
                            proof = validate_epochs(*args, **kwargs)
                            if jax.process_index() == int(failed_rank):
                                constructor_patch.setattr(
                                    CellGeometryStorageProjection,
                                    "__init__",
                                    forbidden_constructor,
                                )
                                injected = True
                            return proof

                        constructor_patch.setattr(
                            checkpoint,
                            "validate_collective_mesh_source_epochs",
                            changed_after_replay,
                        )
                        with pytest.raises((ValueError, checkpoint.RepositoryCorruptionError)):
                            restore_once()
                elif stage in ("storedid", "changed", "unbound", "deleted"):
                    injected = _reject_shared_owner_fault(
                        restore_once, stage, int(failed_rank)
                    )
                else:
                    with pytest.raises(
                        (ValueError, checkpoint.RepositoryCorruptionError)
                    ) as refusal:
                        restore_once()
                    if stage in ("before", "after") and injected:
                        assert isinstance(refusal.value, checkpoint.RepositoryCorruptionError)
                        assert isinstance(refusal.value.__cause__, ValueError)
                        assert str(refusal.value.__cause__) == (
                            f"packed projection injected failure {stage} local receipt"
                        )
                refused = True
                assert injected == (jax.process_index() == int(failed_rank))
                calls.clear()
                replayed_transitions.clear()
            restored = restore_once()
            assert selected in calls
            assert set(calls) <= set(expected)
            state = restored.state
            assert state.checkpoint_state_id == published[selected]["state_id"]
            assert state.source_closure_id == published[selected]["closure_id"]
            assert state.to_payload() == states[selected].to_payload()
            assert state.source_closure is not None
            target = state.source_closure["collective_target"]
            assert _target_identity(target) == published[selected]["identity"]
            projection = target.mesh.storage.geometry_projection
            assert isinstance(projection, CellGeometryStorageProjection)
            for owner in expected:
                receipt = dict(projection.addressable_arrays(owner))
                assert _bank_digest(receipt) == published[owner]["receipt_digest"]
            for owner in set(range(4)) - set(expected):
                with pytest.raises(ValueError, match="not process-addressable"):
                    projection.addressable_arrays(owner)
        assert set(calls) == set(expected)
        assert all(
            not array.is_fully_addressable for _, array in projection.projected_arrays
        )
        # Every logical source row remains in the bank, and every assigned
        # historical typed source root was authenticated on this process.
        bound = bind_distributed_checkpoint_source_arrays(
            state.source_closure, _PACKED_LIMITS
        )
        assert not bound.missing
        producer_rows = [
            json.loads((Path(directory) / f"bank-rows-{owner}.json").read_text("utf-8"))
            for owner in range(4)
        ]
        actual_rows = _local_bank_rows(bound.bank)
        assert all(set(records) == set(actual_rows) for records in producer_rows)
        for name, actual in actual_rows.items():
            expected_rows: dict[str, str] = {}
            for records in producer_rows:
                archived = records[name]
                assert actual["shape"] == archived["shape"]
                assert actual["dtype"] == archived["dtype"]
                for row, digest in archived["rows"].items():
                    assert row not in expected_rows or expected_rows[row] == digest
                    expected_rows[row] = digest
            assert all(
                digest == expected_rows[row] for row, digest in actual["rows"].items()
            )
        assert tuple(owner.source_owner_index for owner in state.source_owner_states) == (
            0, 1, 2, 3
        )
        assert tuple(
            owner.source_owner_index
            for owner in state.source_owner_states
            if owner.source_closure is not None
        ) == placement.decoded_owners
        for name in _science_row(0):
            value = restored.arrays[name]
            assert isinstance(value, jax.Array)
            assert not value.is_fully_addressable
            for shard in value.addressable_shards:
                rows = range(*shard.index[0].indices(4))
                reference = np.concatenate([_science_row(row)[name] for row in rows])
                np.testing.assert_array_equal(
                    np.asarray(shard.data).view(np.uint8), reference.view(np.uint8)
                )
        if published[0]["graph"]:
            assert published[selected]["transition_id"] in replayed_transitions
        if published[0].get("planar"):
            _publish_restored_planar(state)
        return {
            "rank": jax.process_index(),
            "source_owners": list(expected),
            "decoded_owners": list(placement.decoded_owners),
            "addressed_owners": sorted(set(calls)),
            "empty_owners": [owner for owner in expected if published[owner]["empty"]],
            "science_exact": True,
            "bank_exact": True,
            "collectively_refused": refused,
            "injected_locally": injected,
            "replayed_transitions": replayed_transitions,
        }
    finally:
        jax.distributed.shutdown()


def _reject_shared_owner_fault(
    restore: Callable[[], checkpoint.MeshingCheckpointRestore],
    stage: str,
    failed_rank: int,
) -> bool:
    """Corrupt current authenticated owners, never a synthetic replacement proof."""
    from phydrax.lifecycle import _meshing_sources as source_owner

    injected = False
    owner_walk = False
    mutations: list[tuple[Any, str, Any]] = []
    original_evidence = CollectiveMeshEvidence.from_stored_fields
    original_binding = CollectiveMeshStorageBinding.__init__
    original_plan = source_owner._meshing_source_replay_plan
    original_validation = checkpoint.validate_collective_mesh_source_epochs

    def evidence_constructor(
        cls: type[CollectiveMeshEvidence],
        value: CollectiveMeshEvidence,
        /,
        *,
        collective: bool = False,
    ) -> CollectiveMeshEvidence:
        assert cls is CollectiveMeshEvidence
        if owner_walk:
            raise AssertionError("Owner walk reran a shared evidence constructor")
        rebuilt = original_evidence(value, collective=collective)
        if collective:
            assert rebuilt.stored_content_id == stored_record_content_id(
                rebuilt, excluded="stored_content_id"
            )
        return rebuilt

    def binding_constructor(
        self: CollectiveMeshStorageBinding, *args: Any, **kwargs: Any
    ) -> None:
        if owner_walk:
            raise AssertionError("Owner walk reran a shared storage-binding constructor")
        original_binding(self, *args, **kwargs)

    def corrupt_stored_id(*args: Any, **kwargs: Any) -> source_owner._MeshingSourceReplayPlan:
        nonlocal injected
        plan = original_plan(*args, **kwargs)
        if jax.process_index() == failed_rank:
            evidence = next(
                value for _, value in reversed(plan.epochs)
                if isinstance(value, CollectiveMeshEvidence)
            )
            mutations.append((evidence, "stored_content_id", evidence.stored_content_id))
            object.__setattr__(
                evidence, "stored_content_id",
                ("0" if evidence.stored_content_id[0] != "0" else "1")
                + evidence.stored_content_id[1:],
            )
            injected = True
        return plan

    def corrupt_shared_input(
        closures: tuple[Mapping[str, Any], ...], /, **kwargs: Any
    ) -> source_owner.MeshingSourceValidation:
        nonlocal injected, owner_walk
        proof = original_validation(closures, **kwargs)
        owner_walk = True
        if jax.process_index() == failed_rank:
            evidence = closures[0]["collective_target"].collective_evidence
            assert isinstance(evidence, CollectiveMeshEvidence)
            original = evidence.entity_ids[0]
            assert not original.is_fully_addressable
            if stage == "deleted":
                original.delete()
            else:
                local = []
                for shard in original.addressable_shards:
                    host = np.array(shard.data, copy=True)
                    if stage == "changed" and host.size:
                        host.reshape(-1)[0] += 1
                    local.append(jax.device_put(host, shard.device))
                replacement = jax.make_array_from_single_device_arrays(
                    original.shape, original.sharding, local
                )
                assert replacement is not original
                mutations.append((evidence, "entity_ids", evidence.entity_ids))
                object.__setattr__(
                    evidence, "entity_ids", (replacement, *evidence.entity_ids[1:])
                )
            injected = True
        return proof

    try:
        with pytest.MonkeyPatch.context() as patch:
            # Install guards before proof creation so their identity is unchanged
            # during owner reuse; only the deliberately corrupted input can refuse.
            patch.setattr(
                CollectiveMeshEvidence, "from_stored_fields",
                classmethod(evidence_constructor),
            )
            patch.setattr(CollectiveMeshStorageBinding, "__init__", binding_constructor)
            if stage == "storedid":
                patch.setattr(
                    source_owner, "_meshing_source_replay_plan", corrupt_stored_id
                )
            else:
                patch.setattr(
                    checkpoint, "validate_collective_mesh_source_epochs",
                    corrupt_shared_input,
                )
            with pytest.raises((ValueError, checkpoint.RepositoryCorruptionError)):
                restore()
    finally:
        for owner, name, original in reversed(mutations):
            object.__setattr__(owner, name, original)
    return injected


def packed_geometry_malformed(directory: str, fault: str) -> dict[str, Any]:
    """Reject incomplete, duplicate, and foreign metadata before reading payloads."""
    from phydrax.meshing.qualification import _distributed as placement_owner

    try:
        _, manifest, states = _packed_archive(directory)
        altered = list(states)
        layouts = placement_owner.meshing_checkpoint_array_layouts
        with pytest.MonkeyPatch.context() as patch:
            if fault == "missing-owner":
                altered.pop()
            elif fault in ("duplicate-owner", "foreign-owner"):
                altered[-1] = copy.copy(altered[-1])
                object.__setattr__(
                    altered[-1],
                    "source_owner_index",
                    0 if fault == "duplicate-owner" else 4,
                )
            elif fault == "missing-projection":
                for index, state in enumerate(altered):
                    altered[index] = copy.copy(state)
                    object.__setattr__(
                        altered[index],
                        "source_logical_array_names",
                        tuple(
                            name for name in state.source_logical_array_names
                            if "/geometry_projection/" not in name
                        ),
                    )
            elif fault == "foreign-layout":
                projection_names = {
                    source_logical_array_name(alias)
                    for alias in states[0].source_logical_array_names
                    if "/geometry_projection/" in alias
                }
                assert projection_names

                def foreign_layouts(
                    *args: Any, **kwargs: Any
                ) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
                    result = layouts(*args, **kwargs)
                    for name, (shape, dtype) in tuple(result.items()):
                        if name in projection_names:
                            result[name] = ((8, *shape[1:]), dtype)
                    return result

                patch.setattr(
                    placement_owner, "meshing_checkpoint_array_layouts", foreign_layouts
                )
            else:
                raise AssertionError(f"Unexpected metadata fault {fault!r}")
            with pytest.raises(ValueError) as refused:
                prepare_distributed_restore_placement(
                    altered, manifest, _packed_group(False), _PACKED_LIMITS
                )
        return {"rank": jax.process_index(), "fault": fault, "refusal": str(refused.value)}
    finally:
        jax.distributed.shutdown()
