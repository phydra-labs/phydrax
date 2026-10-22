# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Two-process checkpoint publication, refusal, and source restoration scenarios."""

import json
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
from phydrax._array_archive import ArrayArchiveLimits
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
    read_meshing_checkpoint_states,
    restore_meshing_checkpoint,
)
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from phydrax.lifecycle._repository import ObjectNotFoundError, POSIXArtifactRepository
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


# Frozen distributed-qualification archive and repository limits.
_ACCEPTED_EPOCH_LIMITS = ArrayArchiveLimits(
    max_members=32768,
    max_manifest_nesting=128,
    max_manifest_bytes=128 * 1024 * 1024,
    max_central_directory_bytes=8 * 1024 * 1024,
)
_ACCEPTED_EPOCH_CHECKPOINT = {
    "checkpoint": "accepted-epoch",
    "execution": "accepted-epoch-execution",
    "analysis": "accepted-epoch-analysis",
    "revision": "accepted-epoch-revision",
}


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


def _accepted_epoch_repository(repository_root: str) -> POSIXArtifactRepository:
    return build_repository(
        Path(repository_root),
        maximum_chunk_bytes=16 * 1024,
        maximum_metadata_bytes=128 * 1024 * 1024,
    )


def accepted_epoch_checkpoint_publish(repository_root: str) -> dict[str, Any]:
    """Publish real accepted-epoch fields partitioned across both producer owners."""
    from phydrax.meshing.qualification import (
        bind_distributed_checkpoint_source_arrays,
        envelope_transport_field_templates,
        prepare_envelope_accepted_epoch,
    )
    from tests._support.native_family_generation import native_family_generation

    try:
        process_index = jax.process_index()
        devices = Mesh(np.asarray(jax.devices(), dtype=np.object_), ("owner",))
        owners = NamedSharding(devices, PartitionSpec("owner"))
        # The native surface carrier has 26 P1 rows, so every owner holds a
        # true half of each bank; the 7-row tetrahedral carrier cannot split.
        part, source, request, options = native_family_generation("surface")
        carrier = part.carrier
        if not isinstance(carrier, phx.meshing.CellMeshingResult):
            raise TypeError(
                "accepted_epoch.carrier must be a native CellMeshingResult; "
                f"got {type(carrier).__name__}"
            )
        certification = carrier.certification
        if certification is None:
            raise ValueError(
                "accepted_epoch.carrier.certification must retain the native report; got None"
            )
        scalar = phx.discretization.FiniteElementPlan(
            carrier.mesh,
            envelope_transport_field_templates(carrier.mesh.blocks[0].cell_kind)[0][1],
            coordinate_spec=carrier.geometry,
        ).prepare()
        coordinates = np.asarray(scalar.dof_maps[0].dof_coordinates)
        # Equal shapes keep alias selection independent of shape coincidence.
        reference = {
            "u": np.sum(coordinates**2, axis=-1),
            "density": np.full(coordinates.shape[0], 2.0),
            "history": coordinates[:, 0] + 0.5,
        }
        owner_count = jax.process_count()
        if coordinates.shape[0] % owner_count:
            raise ValueError(
                "accepted_epoch.coefficients must split evenly across producer owners; "
                f"got {coordinates.shape[0]} rows for {owner_count} owners"
            )
        rows = coordinates.shape[0] // owner_count
        values = {
            name: jax.make_array_from_process_local_data(
                owners,
                value[process_index * rows : (process_index + 1) * rows],
                global_shape=value.shape,
            )
            for name, value in reference.items()
        }
        # S01 is reachable only through owner-partitioned coefficient banks.
        if any(value.is_fully_addressable for value in values.values()):
            raise ValueError(
                "accepted_epoch.fields must be partitioned across producer processes; "
                "got fully addressable coefficient or history banks"
            )
        epoch = prepare_envelope_accepted_epoch(
            carrier,
            scalar,
            values["u"],
            (values["density"], values["history"]),
            0,
            part_name="implicit",
        )
        closure = {
            "certification_inputs": certification.request,
            "report": certification,
            "associations": carrier.associations,
            "generation_source": source,
            "generation_specification": request,
            "generation_options": options,
            "accepted_target": carrier,
            "accepted_data": {"accepted_epochs": (epoch,)},
        }
        register_meshing_source_artifacts()
        source_arrays = bind_distributed_checkpoint_source_arrays(
            closure, _ACCEPTED_EPOCH_LIMITS
        )
        record: dict[str, Any] = {
            "process_index": process_index,
            "nonaddressable_fields": all(
                isinstance(value, jax.Array) and not value.is_fully_addressable
                for value in epoch.fields.values()
            ),
            "epoch_id": epoch.epoch_id,
            "field_aliases": {
                name: next(
                    (
                        alias
                        for alias, array in source_arrays.bank.items()
                        if array is value
                    ),
                    None,
                )
                for name, value in epoch.fields.items()
            },
            "missing": source_arrays.missing,
            "reference": {name: value.tolist() for name, value in reference.items()},
            "published": False,
        }
        if source_arrays.missing:
            return record
        state = native_source_state(
            closure,
            source_array_bindings=source_arrays.bindings,
            source_logical_arrays=source_arrays.bank,
            limits=_ACCEPTED_EPOCH_LIMITS,
        )
        identities = _ACCEPTED_EPOCH_CHECKPOINT
        publish_process_meshing_checkpoint(
            _accepted_epoch_repository(repository_root),
            identities["checkpoint"],
            identities["execution"],
            state,
            _distributed_arrays(scientific_arrays(), owners, process_index),
            analysis_plan_id=identities["analysis"],
            numeric_revision_id=identities["revision"],
            writer_id=f"accepted-epoch-writer-{process_index}",
            limits=_ACCEPTED_EPOCH_LIMITS,
        )
        multihost_utils.sync_global_devices("accepted-epoch-checkpoint-published")
        record["published"] = True
        record["state_id"] = state.checkpoint_state_id
        return record
    finally:
        jax.distributed.shutdown()


def accepted_epoch_checkpoint_restore(
    repository_root: str, expected_state_ids: str
) -> dict[str, Any]:
    """Restore both producer owners in one process and replay their accepted epoch."""
    from phydrax.lifecycle import validate_meshing_accepted_epochs
    from phydrax.meshing.qualification import distributed_checkpoint_host_array_names

    register_meshing_source_artifacts()
    repository = _accepted_epoch_repository(repository_root)
    identities = _ACCEPTED_EPOCH_CHECKPOINT
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        identities["checkpoint"],
        identities["analysis"],
        identities["revision"],
        identities["execution"],
        expected_process_count=2,
        limits=_ACCEPTED_EPOCH_LIMITS,
    )
    states = read_meshing_checkpoint_states(
        repository, manifest, limits=_ACCEPTED_EPOCH_LIMITS
    )
    host_names: set[str] = set()
    for owner in states:
        if owner.source_recipe_json is None:
            raise ValueError(
                "accepted_epoch.restore.source_recipe_json must be retained by every owner; got None"
            )
        host_names |= distributed_checkpoint_host_array_names(
            owner, owner.source_recipe_json, _ACCEPTED_EPOCH_LIMITS
        )
    names = {name for owner in states for name, _ in owner.array_roles}
    placement = NamedSharding(
        Mesh(np.asarray(jax.devices(), dtype=np.object_), ("owner",)), PartitionSpec()
    )
    checkpoint = restore_meshing_checkpoint(
        repository,
        manifest,
        {name: placement for name in names - host_names},
        restart_relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id={
            int(owner): identifier
            for owner, identifier in json.loads(expected_state_ids).items()
        },
        source_owner_index=0,
        limits=_ACCEPTED_EPOCH_LIMITS,
    )
    restored = checkpoint.state
    closures = (
        restored.source_closure,
        *(owner.source_closure for owner in restored.source_owner_states),
    )
    if any(closure is None for closure in closures):
        raise ValueError(
            "accepted_epoch.restore.source_closure must be retained by every owner; got None"
        )
    histories = [
        validate_meshing_accepted_epochs(closure)
        for closure in closures
        if closure is not None
    ]
    fields = histories[0][0].fields
    return {
        "process_count": jax.process_count(),
        "source_owner_indices": [
            owner.source_owner_index for owner in restored.source_owner_states
        ],
        "epoch_ids": [[epoch.epoch_id for epoch in history] for history in histories],
        "fields": {name: np.asarray(value).tolist() for name, value in fields.items()},
        "history_shares_accepted_history": fields["history"]
        is fields["accepted-history"],
    }
