# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Earned source admission preserves checkpoint bytes and mutation refusals."""

import copy
import socket
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from phydrax._model._structure import (
    model_recipe_array_inventory,
    prepare_model_recipe_catalog,
)
from phydrax.lifecycle import (
    _distributed_checkpoint as checkpoint_owner,
    _meshing_checkpoint_binding as binding,
)
from phydrax.lifecycle._distributed_checkpoint import (
    ProcessCheckpointPublication,
    publish_process_meshing_checkpoint,
)
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from phydrax.lifecycle._models import CheckpointManifest
from phydrax.lifecycle._repository import ObjectNotFoundError, POSIXArtifactRepository
from phydrax.lifecycle._restart_topology import (
    MeshingCheckpointState,
    TopologyRestartPolicy,
)
from tests._support.distributed_meshing.restart_builders import (
    build_repository,
    native_source_arrays,
    native_source_state,
    restart_relation,
)
from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices
from tests._support.meshing_field_declarations import (
    prepare_single_native_field_scenario,
)


@dataclass(frozen=True, slots=True)
class _PublicationSource:
    state: MeshingCheckpointState
    arrays: dict[str, jax.Array]
    local: np.ndarray


@pytest.fixture
def publication_source() -> Iterator[_PublicationSource]:
    register_meshing_source_artifacts()
    scenario = prepare_single_native_field_scenario()
    mesh, closure, local = (
        scenario.carrier.mesh,
        scenario.closure,
        scenario.coefficients,
    )
    _, _, values = prepare_model_recipe_catalog(closure, prefix="source-record")
    writable = local.flags.writeable
    local.setflags(write=True)
    try:
        aliases = {
            name: f"checkpoint/{name}"
            for name, value in values.items()
            if isinstance(value, jax.Array)
        }
        bank = {
            aliases[name]: value
            for name, value in values.items()
            if isinstance(value, jax.Array)
        }
        state = native_source_state(
            closure, source_array_bindings=aliases, source_logical_arrays=bank
        )
        arrays = {
            name: jax.device_put(value)
            for name, value in native_source_arrays(mesh, state.mesh_epoch).items()
        }
        yield _PublicationSource(state, arrays, local)
    finally:
        local.setflags(write=writable)


def _without_publication_admission(
    state: MeshingCheckpointState,
) -> MeshingCheckpointState:
    bank = state.source_logical_arrays
    assert bank is not None and bank.source_validation is not None
    cold_bank = binding.AdmittedMeshingSourceArrays(
        bank,
        maximum_chunk_bytes=bank.maximum_chunk_bytes,
        source_validation=bank.source_validation,
    )
    cold_state = copy.copy(state)
    object.__setattr__(cold_state, "source_logical_arrays", cold_bank)
    return cold_state


def _count_semantic_validation(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []
    validate = binding.validate_meshing_source_closure

    def counted(closure: Any, *args: Any, **kwargs: Any) -> None:
        calls.append(closure)
        validate(closure, *args, **kwargs)

    monkeypatch.setattr(binding, "validate_meshing_source_closure", counted)
    return calls


def _publish(
    repository: POSIXArtifactRepository,
    state: MeshingCheckpointState,
    arrays: dict[str, jax.Array],
) -> ProcessCheckpointPublication:
    return publish_process_meshing_checkpoint(
        repository,
        "source-publication",
        "source-execution",
        state,
        arrays,
        analysis_plan_id="source-analysis",
        numeric_revision_id="source-revision",
        writer_id="source-writer",
        attempt_id="source-attempt",
    )


def _content_frame(
    repository: POSIXArtifactRepository, publication: ProcessCheckpointPublication
) -> tuple[object, ...]:
    manifest = publication.artifact_manifest
    assert manifest is not None
    # Storage roots, commit times, and transaction IDs belong to each repository;
    # the scientific metadata, stream boundaries, checksums, and bytes must agree.
    chunks = tuple(
        (
            chunk.logical_name,
            chunk.index,
            chunk.offset,
            chunk.plaintext_size,
            chunk.encoded_size,
            chunk.plaintext_sha256,
            chunk.encoded_sha256,
            chunk.encoding,
            repository.read_chunk(manifest, chunk),
        )
        for chunk in manifest.chunks
    )
    return publication.process_index, manifest.artifact_id, manifest.metadata, chunks


def _assert_cold_publication_metadata(
    publication: ProcessCheckpointPublication, state: MeshingCheckpointState
) -> None:
    retained = publication.meshing_state
    assert retained is not None
    assert retained.to_payload() == state.to_payload()
    assert retained.checkpoint_state_id == state.checkpoint_state_id
    assert retained.source_closure is None
    assert retained.source_logical_arrays is None


def test_admitted_source_publication_matches_full_validation_content_and_framing(
    publication_source: _PublicationSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, arrays = publication_source.state, publication_source.arrays
    full_state = _without_publication_admission(state)
    assert full_state.to_payload() == state.to_payload()
    admitted_repository = build_repository(
        tmp_path / "admitted",
        maximum_chunk_bytes=16 * 1024,
        maximum_metadata_bytes=1024 * 1024,
    )
    full_repository = build_repository(
        tmp_path / "full",
        maximum_chunk_bytes=16 * 1024,
        maximum_metadata_bytes=1024 * 1024,
    )
    calls = _count_semantic_validation(monkeypatch)
    admitted = _publish(admitted_repository, state, arrays)
    assert calls == []
    full = _publish(full_repository, full_state, arrays)
    assert len(calls) == 1
    assert _content_frame(admitted_repository, admitted) == _content_frame(
        full_repository, full
    )
    assert len(admitted.shards) == len(full.shards)
    _assert_cold_publication_metadata(admitted, state)
    _assert_cold_publication_metadata(full, full_state)


def test_mutated_local_source_publication_uses_full_validation_refusal(
    publication_source: _PublicationSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, arrays = publication_source.state, publication_source.arrays
    full_state = _without_publication_admission(state)
    admitted_repository = build_repository(tmp_path / "admitted")
    full_repository = build_repository(tmp_path / "full")
    calls = _count_semantic_validation(monkeypatch)
    local = publication_source.local
    saved = local.copy()
    local.flat[np.flatnonzero(np.isfinite(local))[0]] = np.nan
    try:
        with pytest.raises(ValueError) as full_error:
            _publish(full_repository, full_state, arrays)
        assert len(calls) == 1
        calls.clear()
        with pytest.raises(type(full_error.value)) as admitted_error:
            _publish(admitted_repository, state, arrays)
        assert len(calls) == 1
        assert str(admitted_error.value) == str(full_error.value)
        with pytest.raises(ObjectNotFoundError):
            admitted_repository.get_manifest(
                f"source-publication.process-{jax.process_index()}"
            )
    finally:
        local[...] = saved


def _published_restore_source(
    source: _PublicationSource, root: Path
) -> tuple[POSIXArtifactRepository, CheckpointManifest, dict[str, jax.sharding.Sharding]]:
    repository = build_repository(
        root, maximum_chunk_bytes=16 * 1024, maximum_metadata_bytes=1024 * 1024
    )
    _publish(repository, source.state, source.arrays)
    manifest = checkpoint_owner.assemble_distributed_checkpoint_from_repository(
        repository,
        "source-publication",
        "source-analysis",
        "source-revision",
        "source-execution",
        expected_process_count=1,
    )
    _, inventory, _ = prepare_model_recipe_catalog(
        source.state.source_closure, prefix="source-record"
    )
    host_names = {
        source.state.source_array_name(entry.name)
        for entry in inventory
        if entry.backend == "numpy"
    }
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    shardings: dict[str, jax.sharding.Sharding] = {
        name: sharding for name, _ in source.state.array_roles if name not in host_names
    }
    return repository, manifest, shardings


def test_cold_restore_reuses_selected_admission_only_within_its_invocation(
    publication_source: _PublicationSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, manifest, shardings = _published_restore_source(
        publication_source, tmp_path / "checkpoint"
    )
    calls = _count_semantic_validation(monkeypatch)
    restored = checkpoint_owner.restore_meshing_checkpoint(
        repository,
        manifest,
        shardings,
        restart_relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=publication_source.state.checkpoint_state_id,
    ).state
    assert len(calls) == 1
    assert restored.to_payload() == publication_source.state.to_payload()
    assert restored.checkpoint_state_id == publication_source.state.checkpoint_state_id
    assert len(restored.source_owner_states) == 1
    # The restore-scoped token cannot authorize a later independent construction.
    repeated = MeshingCheckpointState.from_payload(
        restored.to_payload(),
        source_closure=restored.source_closure,
        source_logical_arrays=restored.source_logical_arrays,
        source_owner_states=restored.source_owner_states,
        limits=checkpoint_owner._DISTRIBUTED_CHECKPOINT_LIMITS,
    )
    assert len(calls) == 2
    assert repeated.to_payload() == restored.to_payload()


@pytest.mark.parametrize("fault", ("foreign-restore-token", "mutated-bytes"))
def test_cold_restore_selected_rebuild_falls_back_to_full_admission(
    publication_source: _PublicationSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    repository, manifest, shardings = _published_restore_source(
        publication_source, tmp_path / "checkpoint"
    )

    def restore() -> Any:
        return checkpoint_owner.restore_meshing_checkpoint(
            repository,
            manifest,
            shardings,
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=publication_source.state.checkpoint_state_id,
        )

    foreign = None
    if fault == "foreign-restore-token":
        bank = restore().state.source_logical_arrays
        assert bank is not None
        foreign = bank._publication_slot.admission
        assert foreign is not None
    original = MeshingCheckpointState.from_payload

    def intercepted(cls: type, *args: Any, **kwargs: Any) -> MeshingCheckpointState:
        if kwargs.get("source_owner_states"):
            if fault == "foreign-restore-token":
                kwargs["source_logical_arrays"]._publication_slot.admission = foreign
            else:
                payload = kwargs["source_closure"]["accepted_data"]["fields"][
                    "temperature"
                ]
                payload.flat[0] = np.nan
        return original(*args, **kwargs)

    monkeypatch.setattr(MeshingCheckpointState, "from_payload", classmethod(intercepted))
    calls = _count_semantic_validation(monkeypatch)
    if fault == "mutated-bytes":
        with pytest.raises(ValueError, match="source numerical content changed"):
            restore()
    else:
        restored = restore().state
        assert restored.to_payload() == publication_source.state.to_payload()
    assert len(calls) == 2


def restore_phase_refusal_worker(
    directory: str, fault: str, refusing_rank: str
) -> dict[str, Any]:
    """Refuse a cold restore on one actual rank, then restore cleanly on both."""
    register_meshing_source_artifacts()
    repository = build_repository(
        Path(directory),
        maximum_chunk_bytes=16 * 1024,
        maximum_metadata_bytes=1024 * 1024,
    )
    manifest = checkpoint_owner.assemble_distributed_checkpoint_from_repository(
        repository,
        "source-publication",
        "source-analysis",
        "source-revision",
        "source-execution",
        expected_process_count=1,
    )
    state = checkpoint_owner.read_meshing_checkpoint_states(repository, manifest)[0]
    assert state.source_recipe_json is not None
    inventory = model_recipe_array_inventory(
        binding.meshing_source_recipe(state.source_recipe_json), prefix="source-record"
    )
    host_names = {
        state.source_array_name(entry.name)
        for entry in inventory
        if entry.backend == "numpy"
    }
    placement = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.asarray(jax.devices(), dtype=object), ("owner",)),
        jax.sharding.PartitionSpec(),
    )
    shardings = {
        name: placement for name, _ in state.array_roles if name not in host_names
    }
    rank = jax.process_index()
    marker = f"injected {fault} on rank {refusing_rank}"
    counts = {"decode": 0, "replay": 0, "owner": 0, "selected": 0}
    decode = checkpoint_owner.model_from_logical_array_recipe
    replay = checkpoint_owner.validate_collective_mesh_source_epochs
    from_payload = MeshingCheckpointState.from_payload
    validate_arrays = checkpoint_owner._validate_meshing_arrays

    def refuse(stage: str) -> None:
        if fault == stage and rank == int(refusing_rank):
            raise ValueError(marker)

    def decoded(*args: Any, **kwargs: Any) -> Any:
        closure = decode(*args, **kwargs)
        counts["decode"] += 1
        refuse("decode")
        return closure

    def replayed(*args: Any, **kwargs: Any) -> Any:
        counts["replay"] += 1
        return replay(*args, **kwargs)

    def reconstructed(cls: type, *args: Any, **kwargs: Any) -> MeshingCheckpointState:
        result = from_payload(*args, **kwargs)
        if kwargs.get("source_closure") is not None:
            stage = "selected" if kwargs.get("source_owner_states") else "owner"
            counts[stage] += 1
            # Fail after real validation, so a refusal must also discard any
            # earned owner/publication token from this incomplete invocation.
            refuse(stage)
        return result

    def arrays_validated(owner: MeshingCheckpointState, arrays: Any) -> None:
        validate_arrays(owner, arrays)
        if owner.source_closure is not None:
            refuse("owner-arrays")

    def restore() -> Any:
        return checkpoint_owner.restore_meshing_checkpoint(
            repository,
            manifest,
            shardings,
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=state.checkpoint_state_id,
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(checkpoint_owner, "model_from_logical_array_recipe", decoded)
        patch.setattr(checkpoint_owner, "validate_collective_mesh_source_epochs", replayed)
        patch.setattr(MeshingCheckpointState, "from_payload", classmethod(reconstructed))
        patch.setattr(checkpoint_owner, "_validate_meshing_arrays", arrays_validated)
        with pytest.raises(
            (ValueError, checkpoint_owner.RepositoryCorruptionError)
        ) as refusal:
            restore()
    refusal_counts = dict(counts)
    if rank == int(refusing_rank):
        original = refusal.value
        if fault in ("decode", "owner"):
            assert isinstance(original, checkpoint_owner.RepositoryCorruptionError)
            original = original.__cause__
        assert isinstance(original, ValueError)
        assert str(original) == marker
    else:
        assert "restore.collective_admission" in str(refusal.value)
        assert f"[{refusing_rank}]" in str(refusal.value)
    assert binding._source_restore_admission_scope.get() is None
    # Fresh validation, not a surviving token, must authorize the clean retry.
    with pytest.MonkeyPatch.context() as patch:
        validations = _count_semantic_validation(patch)
        restored = restore()
    assert len(validations) == 1
    assert restored.state.checkpoint_state_id == state.checkpoint_state_id
    assert restored.state.to_payload() == state.to_payload()
    assert len(restored.state.source_owner_states) == 1
    assert restored.state.source_closure is not None
    assert restored.state.source_logical_arrays is not None
    assert restored.state.source_logical_arrays.content_digest == (
        state.source_logical_content_digest
    )
    assert all(
        isinstance(restored.arrays[name], jax.Array)
        and not restored.arrays[name].is_fully_addressable
        for name in shardings
    )
    return {
        "rank": rank,
        "process_count": jax.process_count(),
        "counts": refusal_counts,
        "refused": True,
        "clean_restore_validations": len(validations),
        "restored_state_id": restored.state.checkpoint_state_id,
    }


@pytest.mark.parametrize("refusing_rank", (0, 1))
@pytest.mark.parametrize("fault", ("decode", "owner", "owner-arrays", "selected"))
def test_cold_restore_local_phase_refusal_rolls_back_on_every_process(
    publication_source: _PublicationSource,
    tmp_path: Path,
    fault: str,
    refusing_rank: int,
) -> None:
    directory = tmp_path / "checkpoint"
    _published_restore_source(publication_source, directory)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_on_forced_devices,
                __name__,
                "restore_phase_refusal_worker",
                (str(directory), fault, str(refusing_rank)),
                devices=1,
                env=gloo_rank_environment(coordinator, rank),
            )
            for rank in range(2)
        ]
        records = [future.result() for future in futures]
    assert [record["rank"] for record in records] == [0, 1]
    for record in records:
        assert record["process_count"] == 2
        assert record["refused"]
        counts = record["counts"]
        assert counts["decode"] == 1
        assert counts["replay"] == (0 if fault == "decode" else 1)
        assert counts["owner"] == (0 if fault == "decode" else 1)
        assert counts["selected"] == (1 if fault == "selected" else 0)
        assert record["clean_restore_validations"] == 1
        assert record["restored_state_id"] == publication_source.state.checkpoint_state_id
