# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Native source checkpoint placement, evidence, and corruption contracts."""

import json
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._model._structure import (
    model_from_logical_array_recipe,
    model_recipe_array_inventory,
    model_recipe_array_values,
    ModelRecipeArray,
)
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.lifecycle import (
    assemble_distributed_checkpoint_from_repository,
    CheckpointManifest,
    MeshingCheckpointRestore,
    MeshingCheckpointState,
    MeshingSourceClosure,
    POSIXArtifactRepository,
    publish_process_meshing_checkpoint,
    RepositoryCorruptionError,
    restore_meshing_checkpoint,
    TopologyRestartPolicy,
)
from phydrax.meshing import CellMeshAuditReport
from tests._support.distributed_meshing.restart_builders import (
    build_repository,
    checkpoint_state,
    native_source_arrays,
    native_source_closure,
    native_source_state,
    restart_relation,
)


@dataclass(frozen=True, slots=True)
class _PublishedSourceCheckpoint:
    mesh: CellMesh
    geometry: CellGeometrySpec
    audit: CellMeshAuditReport
    closure: dict[str, Any]
    state: MeshingCheckpointState
    devices: tuple[jax.Device, ...]
    reference: dict[str, np.ndarray]
    recipe: dict[str, Any]
    inventory: tuple[ModelRecipeArray, ...]
    shardings: dict[str, Sharding]
    repository: POSIXArtifactRepository
    manifest: CheckpointManifest


@pytest.fixture(scope="module")
def published_source_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> _PublishedSourceCheckpoint:
    devices = tuple(jax.local_devices()[:2])
    if len(devices) != 2:
        pytest.skip("Source checkpoint placement scenario requires two local devices.")
    mesh, geometry, audit, closure = native_source_closure()
    state = native_source_state(closure)
    source = NamedSharding(
        Mesh(np.asarray(devices, dtype=np.object_), ("owner",)), PartitionSpec("owner")
    )
    destination = NamedSharding(
        Mesh(np.asarray(devices[::-1], dtype=np.object_), ("owner",)),
        PartitionSpec("owner"),
    )
    replicated = NamedSharding(
        Mesh(np.asarray(devices[::-1], dtype=np.object_), ("owner",)), PartitionSpec()
    )
    reference = native_source_arrays(mesh, state.mesh_epoch)
    arrays = {name: jax.device_put(value, source) for name, value in reference.items()}
    assert state.source_recipe_json is not None
    recipe = json.loads(state.source_recipe_json)
    inventory = model_recipe_array_inventory(recipe, prefix="source-record")
    shardings: dict[str, Sharding] = {name: destination for name in arrays}
    for entry in inventory:
        if entry.backend != "numpy":
            shardings[state.source_array_name(entry.name)] = (
                destination if entry.shape and entry.shape[0] % 2 == 0 else replicated
            )
    repository = build_repository(
        tmp_path_factory.mktemp("native-source-checkpoint"),
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
        writer_id="source-writer",
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "native-source",
        "source-analysis",
        "source-revision",
        "source-execution",
        expected_process_count=1,
    )
    return _PublishedSourceCheckpoint(
        mesh,
        geometry,
        audit,
        closure,
        state,
        devices,
        reference,
        recipe,
        inventory,
        shardings,
        repository,
        manifest,
    )


def test_native_source_checkpoint_changed_placement_requeries_and_recertifies(
    published_source_checkpoint: _PublishedSourceCheckpoint,
) -> None:
    case = published_source_checkpoint
    checkpoint = restore_meshing_checkpoint(
        case.repository,
        case.manifest,
        case.shardings,
        restart_relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=case.state.checkpoint_state_id,
    )
    assert isinstance(checkpoint, MeshingCheckpointRestore)
    restored_state = checkpoint.state
    restored_arrays = checkpoint.arrays
    assert checkpoint.admission.admitted
    assert restored_state.source_closure_id == case.state.source_closure_id
    assert restored_state.to_payload() == case.state.to_payload()
    assert restored_state.checkpoint_state_id == case.state.checkpoint_state_id
    restored = restored_state.source_closure
    assert restored is not None
    inputs = restored["certification_inputs"]
    report = restored["report"]
    assert report.request.request_id == inputs.request_id
    assert restored_state.request_id == checkpoint_state().request_id
    assert restored_state.evidence_id == checkpoint_state().evidence_id
    assert inputs.source_revision == case.state.source_revision_id
    assert (
        inputs.fidelity_tolerance
        == case.closure["certification_inputs"].fidelity_tolerance
    )
    np.testing.assert_allclose(
        np.asarray(
            inputs.source.boundary_distance(
                np.asarray(((2.0, 0.0, 0.0),), dtype=np.float64)
            ).upper
        ),
        np.asarray((1.0,), dtype=np.float64),
    )
    renewed = phx.meshing.certify_meshing_acceptance(
        case.mesh,
        case.geometry,
        case.audit,
        schedule=inputs.schedule,
        domain=inputs.domain,
        source=inputs.source,
        cell_regions=inputs.cell_regions,
        fidelity_tolerance=inputs.fidelity_tolerance,
        fidelity_sample_order=inputs.fidelity_sample_order,
        limits=inputs.limits,
        junction_vertices=inputs.junction_vertices,
    )
    renewed.require_passed()
    assert renewed.report_id == report.report_id
    for old, new in zip(
        case.closure["associations"], restored["associations"], strict=True
    ):
        assert (
            new.source_entities.occurrence_paths()
            == old.source_entities.occurrence_paths()
        )
        assert new.source_entities.entity_ids() == old.source_entities.entity_ids()
        np.testing.assert_array_equal(
            np.asarray(new.target_global_ids), np.asarray(old.target_global_ids)
        )
    for name, expected in case.reference.items():
        np.testing.assert_array_equal(np.asarray(restored_arrays[name]), expected)
    assert np.sum(np.asarray(restored_arrays["material_inventory"])) == np.sum(
        case.reference["material_inventory"]
    )
    restored_ids = restored_arrays["vertex_ids"]
    assert isinstance(restored_ids, jax.Array)
    vertex_ids = np.asarray(case.mesh.vertex_global_ids, dtype=np.int64)
    np.testing.assert_array_equal(
        np.asarray(
            next(
                shard.data
                for shard in restored_ids.addressable_shards
                if shard.device == case.devices[0]
            )
        ),
        vertex_ids[len(vertex_ids) // 2 :],
    )
    for entry in case.inventory:
        value = restored_arrays[case.state.source_array_name(entry.name)]
        if entry.backend == "numpy":
            assert isinstance(value, np.ndarray) and not value.flags.writeable


def test_native_source_checkpoint_refuses_scientific_identity_substitution(
    published_source_checkpoint: _PublishedSourceCheckpoint,
) -> None:
    case = published_source_checkpoint
    with pytest.raises(
        RepositoryCorruptionError,
        match=r"checkpoint_state\.checkpoint_state_id.*scientific binding",
    ):
        restore_meshing_checkpoint(
            case.repository,
            case.manifest,
            case.shardings,
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=canonical_fingerprint({"substitute": "source"}),
        )


def test_native_source_checkpoint_refuses_missing_source_shards(
    published_source_checkpoint: _PublishedSourceCheckpoint,
) -> None:
    case = published_source_checkpoint
    source_shard = next(
        shard
        for shard in case.manifest.shards
        if "source-record:" in dict(shard.metadata)["array_path"]
    )
    manifest = case.manifest
    incomplete = CheckpointManifest(
        manifest.checkpoint_id,
        manifest.analysis_plan_id,
        manifest.numeric_revision_id,
        manifest.execution_plan_id,
        tuple(
            shard for shard in manifest.shards if shard.shard_id != source_shard.shard_id
        ),
        complete=True,
    )
    with pytest.raises(
        RepositoryCorruptionError,
        match=r"checkpoint\.array\.coverage.*covers|manifest\.array_epoch_binding",
    ):
        restore_meshing_checkpoint(
            case.repository,
            incomplete,
            case.shardings,
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=case.state.checkpoint_state_id,
        )


def test_native_source_checkpoint_refuses_numerical_content_substitution(
    published_source_checkpoint: _PublishedSourceCheckpoint,
) -> None:
    case = published_source_checkpoint
    substituted_values = model_recipe_array_values(
        case.closure, case.recipe, prefix="source-record"
    )
    numerical = next(
        entry
        for entry in case.inventory
        if np.issubdtype(np.dtype(entry.dtype), np.floating) and np.prod(entry.shape) > 0
    )
    original = substituted_values[numerical.name]
    # A power-of-two perturbation exposes stale cached scientific IDs.
    substituted_values[numerical.name] = original + np.asarray(
        0.125, dtype=original.dtype
    )
    substituted_closure = model_from_logical_array_recipe(
        case.recipe, substituted_values, prefix="source-record"
    )
    with pytest.raises(ValueError, match=r"GeometryAssociation\.exact"):
        MeshingCheckpointState.from_payload(
            case.state.to_payload(), source_closure=substituted_closure
        )


def test_typed_source_closure_preserves_checkpoint_recipe_identity(
    published_source_checkpoint: _PublishedSourceCheckpoint,
) -> None:
    case = published_source_checkpoint
    typed = MeshingSourceClosure.from_records(case.closure)
    state = MeshingCheckpointState.from_payload(
        case.state.to_payload(), source_closure=typed
    )
    assert state.source_recipe_json == case.state.source_recipe_json
    assert state.source_content_digest == case.state.source_content_digest
    assert state.checkpoint_state_id == case.state.checkpoint_state_id
    assert state.to_payload() == case.state.to_payload()
