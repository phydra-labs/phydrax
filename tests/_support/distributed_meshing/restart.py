# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Two-process checkpoint publication, refusal, and source restoration scenarios."""

import socket
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import phydrax as phx
from phydrax._model._structure import (
    model_from_logical_array_recipe,
    model_recipe_array_inventory,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.lifecycle._chunk_repository import RepositoryCorruptionError
from phydrax.lifecycle._distributed_checkpoint import (
    assemble_distributed_checkpoint_from_repository,
    publish_process_meshing_checkpoint,
    restore_meshing_checkpoint,
)
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from phydrax.lifecycle._repository import ObjectNotFoundError
from phydrax.lifecycle._restart_topology import (
    MeshingCheckpointState,
    TopologyRestartPolicy,
)
from phydrax.meshing._audit import CellMeshAuditReport
from tests._support.distributed_meshing.restart_builders import (
    build_repository,
    checkpoint_state,
    native_source_arrays,
    native_source_closure,
    native_source_state,
    restart_relation,
    scientific_arrays,
)
from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices


def run_restart_workers(
    tmp_path: Path,
    function: str,
    arguments: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Return concurrent rank records after provisioning their shared repository."""
    repository_root = tmp_path / "multiprocess-repository"
    repository_root.mkdir()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_on_forced_devices,
                "tests._support.distributed_meshing.restart",
                function,
                (str(repository_root), *arguments),
                devices=1,
                env=gloo_rank_environment(coordinator, rank),
            )
            for rank in range(2)
        ]
        return [future.result() for future in futures]


def _distributed_arrays(
    reference: Mapping[str, np.ndarray],
    placement: NamedSharding,
    process_index: int,
) -> dict[str, jax.Array]:
    return {
        name: jax.make_array_from_process_local_data(
            placement,
            value[
                process_index * (value.shape[0] // 2) : (process_index + 1)
                * (value.shape[0] // 2)
            ],
            global_shape=value.shape,
        )
        for name, value in reference.items()
    }


def _array_evidence(
    restored: Mapping[str, Any], reference: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name in sorted(reference):
        expected = reference[name]
        value = restored[name]
        match value:
            case jax.Array():
                evidence[name] = {
                    "logical_jax_array": True,
                    "shards": [
                        {
                            "actual": np.asarray(shard.data).tolist(),
                            "expected": expected[shard.index].tolist(),
                        }
                        for shard in value.addressable_shards
                    ],
                }
            case _:
                evidence[name] = {"logical_jax_array": False, "shards": []}
    return evidence


def process_checkpoint_restart(repository_root: str) -> dict[str, Any]:
    """Publish nonaddressable state and restore it with reversed ownership."""
    try:
        process_index = jax.process_index()
        devices = tuple(jax.devices())
        source = NamedSharding(
            Mesh(np.asarray(devices, dtype=np.object_), ("owner",)),
            PartitionSpec("owner"),
        )
        destination = NamedSharding(
            Mesh(np.asarray(devices[::-1], dtype=np.object_), ("owner",)),
            PartitionSpec("owner"),
        )
        reference = scientific_arrays()
        arrays = _distributed_arrays(reference, source, process_index)
        repository = build_repository(Path(repository_root))
        state = checkpoint_state()
        publication = publish_process_meshing_checkpoint(
            repository,
            "mesh-accepted",
            "source-execution",
            state,
            arrays,
            analysis_plan_id="mesh-analysis",
            numeric_revision_id="accepted-revision",
            writer_id=f"process-writer-{process_index}",
        )
        multihost_utils.sync_global_devices("meshing-checkpoint-published")
        manifest = assemble_distributed_checkpoint_from_repository(
            repository,
            "mesh-accepted",
            "mesh-analysis",
            "accepted-revision",
            "source-execution",
            expected_process_count=2,
        )
        checkpoint = restore_meshing_checkpoint(
            repository,
            manifest,
            {name: destination for name in arrays},
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=state.checkpoint_state_id,
        )
        restored_state = checkpoint.state
        restored = checkpoint.arrays
        admission = checkpoint.admission
        array_evidence = _array_evidence(restored, reference)
        return {
            "process_index": publication.process_index,
            "local_ids": array_evidence["vertex_ids"]["shards"][0]["actual"],
            "mesh_epoch": restored_state.mesh_epoch,
            "nonaddressable": all(
                not array.is_fully_addressable for array in arrays.values()
            ),
            "admitted": admission.admitted,
            "state_preserved": restored_state.to_payload() == state.to_payload(),
            "arrays": array_evidence,
        }
    finally:
        jax.distributed.shutdown()


def rejected_checkpoint(
    repository_root: str, fault: Literal["roles", "metadata"]
) -> dict[str, Any]:
    """Return local publication and collective assembly refusal evidence."""
    try:
        process_index = jax.process_index()
        placement = NamedSharding(
            Mesh(np.asarray(tuple(jax.devices()), dtype=np.object_), ("owner",)),
            PartitionSpec("owner"),
        )
        arrays = _distributed_arrays(scientific_arrays(), placement, process_index)
        state = checkpoint_state()
        if process_index == 1:
            if fault == "roles":
                arrays.pop("material_inventory")
            else:
                payload = state.to_payload()
                payload["result_id"] = "different-accepted-scientific-result"
                state = MeshingCheckpointState.from_payload(payload)
        repository = build_repository(Path(repository_root))
        published = False
        publication_error = None
        try:
            publish_process_meshing_checkpoint(
                repository,
                "mesh-rejected",
                "source-execution",
                state,
                arrays,
                analysis_plan_id="mesh-analysis",
                numeric_revision_id="accepted-revision",
                writer_id=f"process-writer-{process_index}",
            )
            published = True
        except ValueError as error:
            if fault != "roles" or process_index != 1:
                raise
            publication_error = type(error).__name__
        multihost_utils.sync_global_devices("meshing-checkpoint-local-refusal")
        complete_manifest_returned = False
        assembly_error = None
        try:
            assemble_distributed_checkpoint_from_repository(
                repository,
                "mesh-rejected",
                "mesh-analysis",
                "accepted-revision",
                "source-execution",
                expected_process_count=2,
            )
            complete_manifest_returned = True
        except (ObjectNotFoundError, RepositoryCorruptionError) as error:
            assembly_error = type(error).__name__
        return {
            "process_index": process_index,
            "local_published": published,
            "complete_manifest_returned": complete_manifest_returned,
            "publication_error": publication_error,
            "assembly_error": assembly_error,
        }
    finally:
        jax.distributed.shutdown()


def _distributed_native_source(
    source: NamedSharding, destination: NamedSharding, process_index: int
) -> tuple[
    CellMesh,
    CellGeometrySpec,
    CellMeshAuditReport,
    MeshingCheckpointState,
    dict[str, NamedSharding],
    dict[int, str],
]:
    replicated_source = NamedSharding(source.mesh, PartitionSpec())
    replicated_destination = NamedSharding(destination.mesh, PartitionSpec())
    mesh, geometry, audit, local_closure = native_source_closure()
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(local_closure)
    inventory = model_recipe_array_inventory(recipe, prefix="source-record")
    source_values = model_recipe_array_values(
        local_closure, recipe, prefix="source-record"
    )
    source_bindings: dict[str, str] = {}
    logical_bank: dict[str, jax.Array] = {}
    destination_by_alias: dict[str, NamedSharding] = {}
    for entry in inventory:
        if entry.backend == "numpy":
            continue
        value = np.asarray(source_values[entry.name])
        partitioned = bool(entry.shape) and entry.shape[0] % 2 == 0
        local_value = (
            value[
                process_index * (entry.shape[0] // 2) : (process_index + 1)
                * (entry.shape[0] // 2)
            ]
            if partitioned
            else value
        )
        logical = jax.make_array_from_process_local_data(
            source if partitioned else replicated_source,
            local_value,
            global_shape=entry.shape,
        )
        source_values[entry.name] = logical
        alias = f"source/{entry.path}"
        source_bindings[entry.name] = alias
        logical_bank[alias] = logical
        destination_by_alias[alias] = (
            destination if partitioned else replicated_destination
        )
    closure = model_from_logical_array_recipe(
        recipe, source_values, prefix="source-record"
    )
    state = native_source_state(
        closure, source_array_bindings=source_bindings, source_logical_arrays=logical_bank
    )
    destination_shardings = {
        state.source_array_name(entry.name): destination_by_alias[
            source_bindings[entry.name]
        ]
        for entry in inventory
        if entry.backend != "numpy"
    }
    trusted_ids = multihost_utils.process_allgather(
        np.frombuffer(bytes.fromhex(state.checkpoint_state_id), dtype=np.uint8),
        tiled=False,
    )
    expected_ids = {
        owner: np.asarray(identifier).tobytes().hex()
        for owner, identifier in enumerate(np.asarray(trusted_ids))
    }
    return mesh, geometry, audit, state, destination_shardings, expected_ids


def native_source_checkpoint_restart(repository_root: str) -> dict[str, Any]:
    """Restore every source owner record, query its source, and recertify it."""
    try:
        process_index = jax.process_index()
        devices = tuple(jax.devices())
        source = NamedSharding(
            Mesh(np.asarray(devices, dtype=np.object_), ("owner",)),
            PartitionSpec("owner"),
        )
        destination = NamedSharding(
            Mesh(np.asarray(devices[::-1], dtype=np.object_), ("owner",)),
            PartitionSpec("owner"),
        )
        mesh, geometry, audit, state, shardings, expected_ids = (
            _distributed_native_source(source, destination, process_index)
        )
        reference = native_source_arrays(mesh, state.mesh_epoch)
        arrays = _distributed_arrays(reference, source, process_index)
        shardings.update({name: destination for name in arrays})
        repository = build_repository(
            Path(repository_root),
            maximum_chunk_bytes=16 * 1024,
            maximum_metadata_bytes=1024 * 1024,
        )
        publish_process_meshing_checkpoint(
            repository,
            "native-source",
            "source-execution",
            state,
            arrays,
            analysis_plan_id="source-analysis",
            numeric_revision_id="source-revision",
            writer_id=f"source-writer-{process_index}",
        )
        multihost_utils.sync_global_devices("native-source-checkpoint-published")
        manifest = assemble_distributed_checkpoint_from_repository(
            repository,
            "native-source",
            "source-analysis",
            "source-revision",
            "source-execution",
            expected_process_count=2,
        )
        checkpoint = restore_meshing_checkpoint(
            repository,
            manifest,
            shardings,
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=expected_ids,
            source_owner_index=process_index,
        )
        restored_state = checkpoint.state
        restored = checkpoint.arrays
        admission = checkpoint.admission
        array_evidence = _array_evidence(restored, reference)
        retained = restored_state.source_closure
        record = {
            "process_index": process_index,
            "local_ids": array_evidence["vertex_ids"]["shards"][0]["actual"],
            "mesh_epoch": restored_state.mesh_epoch,
            "nonaddressable": all(
                not array.is_fully_addressable for array in arrays.values()
            ),
            "admitted": admission.admitted,
            "source_closure_id_preserved": restored_state.source_closure_id
            == state.source_closure_id,
            "source_owner_indices": [
                owner.source_owner_index for owner in restored_state.source_owner_states
            ],
            "source_owner_state_ids": {
                owner.source_owner_index: owner.checkpoint_state_id
                for owner in restored_state.source_owner_states
            },
            "expected_owner_state_ids": expected_ids,
            "owner_closures_retained": all(
                owner.source_closure is not None
                for owner in restored_state.source_owner_states
            ),
            "source_restored": retained is not None,
            "boundary_distance": None,
            "source_recertified": False,
            "report_id_preserved": False,
            "arrays": array_evidence,
        }
        if retained is not None:
            inputs = retained["certification_inputs"]
            record["boundary_distance"] = np.asarray(
                inputs.source.boundary_distance(
                    np.asarray(((2.0, 0.0, 0.0),), dtype=np.float64)
                ).upper
            ).tolist()
            renewed = phx.meshing.certify_meshing_acceptance(
                mesh,
                geometry,
                audit,
                schedule=inputs.schedule,
                source=inputs.source,
                domain=inputs.domain,
                cell_regions=inputs.cell_regions,
                fidelity_tolerance=inputs.fidelity_tolerance,
                fidelity_sample_order=inputs.fidelity_sample_order,
                limits=inputs.limits,
                junction_vertices=inputs.junction_vertices,
            )
            record["source_recertified"] = renewed.passed
            record["report_id_preserved"] = (
                renewed.report_id == retained["report"].report_id
            )
        return record
    finally:
        jax.distributed.shutdown()
