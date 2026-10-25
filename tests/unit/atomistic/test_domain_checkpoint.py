#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Durability of distributed atomistic domain states.

Same-topology restores must replay bitwise under deterministic reductions;
changed-topology migration must preserve every scientific continuation leaf
and the RNG lineage with explicit evidence; incomplete publications and
tampered regions must never restore. Oracles: the uninterrupted trajectory of
the same runtime (bitwise) and the source state read in logical order.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.atomistic._domain_checkpoint as domain_checkpoint
from phydrax._array_archive import ArrayArchiveLimits
from phydrax.atomistic._domain_checkpoint import (
    AtomisticDomainCheckpoint,
    AtomisticDomainCheckpointError,
    AtomisticDomainMigrationCapacityError,
    checkpoint_domain_state,
    migrate_domain_checkpoint,
    publish_domain_checkpoint,
    restore_domain_checkpoint,
    restore_domain_state,
)
from phydrax.atomistic._domain_runtime import (
    prepare_atomistic_domain,
    PreparedAtomisticDomainRuntime,
)
from phydrax.atomistic._domain_state import AtomisticDomainState
from phydrax.atomistic._dynamics import VelocityVerletPlan
from phydrax.atomistic._training_data import AtomisticLoaderCursor
from phydrax.lifecycle._chunk_repository import RepositoryCorruptionError
from phydrax.lifecycle._distributed_checkpoint import (
    IncompleteDistributedCheckpointError,
    load_distributed_checkpoint_commit,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.lifecycle._restart_topology import TopologyRestartPolicy
from tests._support.atomistic_domain import (
    CELL,
    domain_plan,
    ingress,
    mace_model,
    structure,
)
from tests.unit.atomistic.test_domain_dynamics import _domain, _domain_step, _model


TIMESTEP = 0.02
# Regions remain smaller than the position buffers; the explicit member budget
# admits the complete region inventory.
REGION_LIMITS = ArrayArchiveLimits(max_members=2048)
# Generic artifact metadata values stay bounded by this fixed repository ceiling.
METADATA_VALUE_CEILING = 1 << 20
INVENTORY_LOGICAL_NAME = "checkpoint-shard-inventory"


def _repository(
    tmp_path: Path, maximum_chunk_bytes: int = 4096
) -> POSIXArtifactRepository:
    profile = HPCFilesystemProfile(
        "posix.domain-checkpoint-test",
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
        maximum_metadata_bytes=4 << 20,
    )
    return POSIXArtifactRepository(tmp_path / "repository", policy)


def _runtime(model: Any, **kwargs: Any) -> PreparedAtomisticDomainRuntime:
    kwargs.setdefault("streaming_accumulation", "deterministic")
    kwargs.setdefault("reduction", "deterministic")
    return prepare_atomistic_domain(domain_plan(model, **kwargs), model)


def _initial(
    runtime: PreparedAtomisticDomainRuntime,
    model: Any,
    positions: np.ndarray | None = None,
) -> AtomisticDomainState:
    coordinates, species_ids = structure()
    if positions is not None:
        coordinates = positions
    count = coordinates.shape[0]
    momenta = np.zeros_like(coordinates)
    momenta[:, 0] = 6.0
    return ingress(
        runtime,
        model,
        coordinates,
        species_ids,
        momenta=momenta,
        masses=np.linspace(1.0, 2.0, count),
        stable_ids=np.arange(count, dtype=np.int64) * 3 + 5,
        atom_payload={"history": np.arange(count, dtype=np.float64) * 0.25},
        replicated_payload={"thermostat": np.zeros((2,))},
    )


def _verlet(
    runtime: PreparedAtomisticDomainRuntime, state: AtomisticDomainState, steps: int
) -> AtomisticDomainState:
    """Velocity Verlet with a rebuild every step and keyed per-step noise."""
    for _ in range(steps):
        evaluation = runtime.evaluate(state)
        assert bool(evaluation.successful)
        safe = jnp.where(state.layout.active, state.masses, 1.0)
        half = state.momenta + 0.5 * TIMESTEP * evaluation.forces
        step = state.step_index + 1
        state = runtime.with_dynamics(
            state,
            state.positions + TIMESTEP * half / safe[:, None],
            half,
            step_index=step,
            time=state.time + TIMESTEP,
        )
        transition = runtime.rebuild(state)
        assert bool(transition.committed)
        state = transition.state
        following = runtime.evaluate(state)
        assert bool(following.successful)
        momentum = state.momenta + 0.5 * TIMESTEP * following.forces
        state = runtime.with_dynamics(
            state,
            state.positions,
            momentum,
            step_index=step,
            force=following.force_cache(step),
            replicated_payload={
                "thermostat": state.replicated_payload["thermostat"]
                + jax.random.uniform(
                    jax.random.fold_in(state.rng_key, step), (2,), jnp.float64
                )
            },
        )
    return state


def _assert_bitwise(left: AtomisticDomainState, right: AtomisticDomainState) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for a, b in zip(left_leaves, right_leaves, strict=True):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _cursor() -> AtomisticLoaderCursor:
    return AtomisticLoaderCursor(
        loader_id="loader-a",
        semantics_id="semantics-a",
        structure_id="structure-a",
        stream="train",
        source_ids=("source-a",),
        epoch=2,
        batch_index=5,
        payload_index=1,
    )


@pytest.mark.parametrize("devices", [False, True], ids=["lanes", "devices"])
def test_in_memory_checkpoint_replays_bitwise(devices: bool) -> None:
    model = mace_model(2)
    runtime = _runtime(model, devices=devices)
    initial = _initial(runtime, model)
    midway = _verlet(runtime, initial, 2)
    section = {
        "kernel": {
            "step": jnp.asarray(11, dtype=jnp.int64),
            "key": jax.random.key(3),
            "replicated": jax.device_put(
                jnp.asarray([3.0, 4.0], dtype=jnp.float64),
                runtime.plan.ownership.execution_group.named_sharding(()),
            ),
        }
    }
    checkpoint = checkpoint_domain_state(
        runtime,
        midway,
        sections=section,
        loader_cursors=(_cursor(),),
        training_plan_id="training-plan-a",
    )
    uninterrupted = _verlet(runtime, midway, 2)
    restore = restore_domain_state(
        runtime, checkpoint, initial, section_templates=section
    )
    assert restore.restart_class == "bitwise"
    assert restore.loader_cursors == (_cursor(),)
    assert restore.record.training_plan_id == "training-plan-a"
    _assert_bitwise(restore.state, midway)
    for name, actual_tree, template_tree in (
        ("state", restore.state, initial),
        ("sections", restore.sections, section),
    ):
        for (path, actual), expected in zip(
            jax.tree_util.tree_flatten_with_path(actual_tree)[0],
            jax.tree.leaves(template_tree),
            strict=True,
        ):
            location = name + jax.tree_util.keystr(path)
            assert actual.committed == expected.committed, location
            assert actual.sharding.is_equivalent_to(expected.sharding, expected.ndim), (
                location
            )
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(restore.sections["kernel"]["key"])),
        np.asarray(jax.random.key_data(jax.random.key(3))),
    )
    _assert_bitwise(_verlet(runtime, restore.state, 2), uninterrupted)


def test_in_memory_checkpoint_refuses_tamper_and_foreign_templates() -> None:
    model = mace_model(2)
    runtime = _runtime(model)
    initial = _initial(runtime, model)
    checkpoint = checkpoint_domain_state(runtime, initial)
    arrays = list(checkpoint.arrays)
    index = checkpoint.record.ordinal(("momenta",))
    arrays[index] = arrays[index] + 1.0
    forged = AtomisticDomainCheckpoint(
        arrays=tuple(arrays),
        sections=checkpoint.sections,
        record=checkpoint.record,
        checkpoint_id=checkpoint.checkpoint_id,
    )
    with pytest.raises(AtomisticDomainCheckpointError, match="content identity"):
        restore_domain_state(runtime, forged, initial)
    other_model = mace_model(2, key=3)
    other = prepare_atomistic_domain(runtime.plan, other_model)
    with pytest.raises(AtomisticDomainCheckpointError):
        restore_domain_state(other, checkpoint, _initial(other, other_model))
    payloadless = ingress(runtime, model, *structure())
    with pytest.raises(AtomisticDomainCheckpointError, match="structure"):
        restore_domain_state(runtime, checkpoint, payloadless)
    moved = _runtime(model, owners=3)
    with pytest.raises(AtomisticDomainCheckpointError):
        restore_domain_state(moved, checkpoint, _initial(moved, model))


def test_distributed_publication_restores_bitwise_with_sections(tmp_path: Path) -> None:
    from phydrax._execution_runtime import ExecutionRuntime
    from phydrax.atomistic._campaign import AtomisticTaskPacket
    from phydrax.atomistic._stage_transport import (
        AtomisticStageTransportPlan,
        AtomisticTransportEdge,
        AtomisticTransportStage,
        process_subset_group_spec,
    )

    model = mace_model(2)
    runtime = _runtime(model, devices=True)
    initial = _initial(runtime, model)
    midway = _verlet(runtime, initial, 2)
    root = ExecutionRuntime.current().root_group
    group = process_subset_group_spec(root.spec, (0,))
    template = AtomisticTaskPacket(positions=np.zeros((2, 3)), time=0.0)
    transport = AtomisticStageTransportPlan(
        (AtomisticTransportStage("a", group), AtomisticTransportStage("b", group)),
        (AtomisticTransportEdge("a", "b", template),),
        outbox_capacity=2,
        inbox_capacity=1,
        lane_slots=1,
        maximum_rounds=4,
    ).prepare(root, maximum_round_bytes=1 << 24)
    pending = transport.deposit(
        transport.initial_state(),
        "a",
        AtomisticTaskPacket(positions=np.ones((2, 3)), time=0.5),
    )
    section = {"kernel": {"step": jnp.asarray(11, jnp.int64)}}
    repository = _repository(tmp_path)
    publication = publish_domain_checkpoint(
        repository,
        runtime,
        midway,
        checkpoint_id="domain-same",
        writer_id="writer-0",
        sections=section,
        process_sections={"transport": pending},
        loader_cursors=(_cursor(),),
        region_bytes=256,
        limits=REGION_LIMITS,
    )
    assert publication.manifest.complete
    assert len(publication.manifest.shards) > len(publication.record.leaves)
    restore = restore_domain_checkpoint(
        repository,
        runtime,
        initial,
        "domain-same",
        section_templates=section,
        process_section_templates={"transport": transport.initial_state()},
        limits=REGION_LIMITS,
    )
    assert restore.restart_class == "bitwise"
    assert restore.loader_cursors == (_cursor(),)
    _assert_bitwise(restore.state, midway)
    np.testing.assert_array_equal(np.asarray(restore.sections["kernel"]["step"]), 11)
    restored_transport = restore.process_sections["transport"]
    for name in ("outbox_header", "outbox_bytes", "next_sequence", "transfer_digest"):
        np.testing.assert_array_equal(
            getattr(restored_transport, name), getattr(pending, name)
        )
    assert transport.outbox_free(restored_transport, "a") == 1
    _assert_bitwise(_verlet(runtime, restore.state, 2), _verlet(runtime, midway, 2))
    moved = _runtime(model, owners=3)
    with pytest.raises(AtomisticDomainCheckpointError):
        restore_domain_checkpoint(
            repository, moved, _initial(moved, model), "domain-same", limits=REGION_LIMITS
        )
    with pytest.raises(AtomisticDomainCheckpointError):
        migrate_domain_checkpoint(
            repository,
            moved,
            _initial(moved, model),
            "domain-same",
            policy=TopologyRestartPolicy(
                allow_topology_change=True,
                allow_tolerance_restart=True,
                maximum_relative_tolerance=1e-10,
            ),
            relative_tolerance=1e-10,
            limits=REGION_LIMITS,
        )


def test_incomplete_publication_is_never_restorable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = mace_model(2)
    runtime = _runtime(model)
    initial = _initial(runtime, model)
    repository = _repository(tmp_path)

    def crash(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("coordinator lost before the global commit")

    monkeypatch.setattr(domain_checkpoint, "commit_distributed_checkpoint", crash)
    with pytest.raises(RuntimeError, match="global commit"):
        publish_domain_checkpoint(
            repository, runtime, initial, checkpoint_id="domain-partial", writer_id="w"
        )
    with pytest.raises(IncompleteDistributedCheckpointError):
        load_distributed_checkpoint_commit(repository, "domain-partial")
    with pytest.raises(IncompleteDistributedCheckpointError):
        restore_domain_checkpoint(repository, runtime, initial, "domain-partial")


def test_tampered_region_is_refused(tmp_path: Path) -> None:
    model = mace_model(2)
    runtime = _runtime(model)
    initial = _initial(runtime, model)
    repository = _repository(tmp_path)
    publication = publish_domain_checkpoint(
        repository,
        runtime,
        initial,
        checkpoint_id="domain-tamper",
        writer_id="w",
        region_bytes=256,
        limits=REGION_LIMITS,
    )
    ordinal = publication.record.ordinal(("positions",))
    path = f"['state']['{ordinal:06d}']"
    shard = next(
        shard
        for shard in publication.manifest.shards
        if dict(shard.metadata)["array_path"] == path
    )
    metadata = dict(shard.metadata)
    artifact = repository.get_manifest(metadata["artifact_id"])
    chunk = next(
        chunk
        for chunk in artifact.chunks
        if chunk.logical_name == metadata["logical_name"]
    )
    target = tmp_path / "repository" / chunk.object_key
    target.chmod(0o600)
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))
    with pytest.raises(RepositoryCorruptionError):
        restore_domain_checkpoint(
            repository, runtime, initial, "domain-tamper", limits=REGION_LIMITS
        )


# Fixed large-inventory fixture: one global section whose 2048 rows are each
# exactly one 1024-byte region record. Every descriptor carries at least 22
# keys and three 64-hex digests (> 512 bytes), so these section records alone
# exceed the 1 MiB metadata-value ceiling. The state's own 1024-byte regions
# are fewer than its 256-byte regions, which REGION_LIMITS bounds by 2048.
LARGE_SECTION_ROWS = 2048
LARGE_REGION_BYTES = 1024
LARGE_INVENTORY_POLICY = ArrayArchiveLimits(max_members=8192, max_manifest_bytes=8 << 20)


@pytest.mark.parametrize("devices", [False, True], ids=["lanes", "devices"])
def test_large_shard_inventory_publishes_and_restores_under_explicit_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, devices: bool
) -> None:
    """A process inventory above the metadata-value ceiling is a chunked stream.

    Regression for the observed domain publication whose descriptor inventory
    exceeded one 1 MiB metadata value. The inventory is authenticated
    repository chunks; the explicit member/byte policy admits it exactly and a
    one-unit-smaller policy refuses before any inventory chunk is read.
    """
    model = mace_model(2)
    runtime = _runtime(model, devices=devices)
    initial = _initial(runtime, model)
    midway = _verlet(runtime, initial, 1)
    row_width = LARGE_REGION_BYTES // np.dtype(np.float64).itemsize
    section = {
        "ledger": jnp.arange(LARGE_SECTION_ROWS * row_width, dtype=jnp.float64).reshape(
            LARGE_SECTION_ROWS, row_width
        )
    }
    repository = _repository(tmp_path, maximum_chunk_bytes=1 << 16)
    checkpoint_id = "domain-large-inventory"
    publication = publish_domain_checkpoint(
        repository,
        runtime,
        midway,
        checkpoint_id=checkpoint_id,
        writer_id="w",
        sections=section,
        region_bytes=LARGE_REGION_BYTES,
        limits=LARGE_INVENTORY_POLICY,
    )
    assert publication.manifest.complete
    section_shards = [
        shard
        for shard in publication.manifest.shards
        if dict(shard.metadata)["array_path"].startswith("['sections']['ledger']")
    ]
    assert len(section_shards) == LARGE_SECTION_ROWS
    artifact = repository.get_manifest(f"{checkpoint_id}.process-{jax.process_index()}")
    metadata = dict(artifact.metadata)
    inventory = b"".join(
        repository.read_chunk(artifact, chunk)
        for chunk in artifact.chunks
        if chunk.logical_name == INVENTORY_LOGICAL_NAME
    )
    shard_count = int(metadata["shard_count"])
    assert len(inventory) > METADATA_VALUE_CEILING
    assert hashlib.sha256(inventory).hexdigest() == metadata["shard_inventory_digest"]
    assert all(
        len(value.encode("utf-8")) <= METADATA_VALUE_CEILING
        for value in metadata.values()
    )
    assert shard_count == len(publication.manifest.shards)
    assert shard_count <= LARGE_INVENTORY_POLICY.max_members

    exact = ArrayArchiveLimits(max_members=shard_count, max_manifest_bytes=len(inventory))
    restore = restore_domain_checkpoint(
        repository,
        runtime,
        initial,
        checkpoint_id,
        section_templates=section,
        limits=exact,
    )
    np.testing.assert_array_equal(
        np.asarray(restore.sections["ledger"]), np.asarray(section["ledger"])
    )
    assert restore.restart_class == "bitwise"
    _assert_bitwise(restore.state, midway)
    assert jax.dtypes.issubdtype(restore.state.rng_key.dtype, jax.dtypes.prng_key)
    # Values come from the checkpoint; placement comes from the restore template.
    for restored, template in zip(
        jax.tree_util.tree_leaves(restore.state),
        jax.tree_util.tree_leaves(initial),
        strict=True,
    ):
        assert isinstance(restored, jax.Array) == isinstance(template, jax.Array)
        if isinstance(template, jax.Array):
            assert restored.sharding.is_equivalent_to(template.sharding, restored.ndim)

    inventory_reads: list[str] = []
    read_chunk = POSIXArtifactRepository.read_chunk

    def observed_read(
        self: POSIXArtifactRepository,
        manifest: Any,
        chunk: Any,
        *args: Any,
        **kwargs: Any,
    ) -> bytes:
        if chunk.logical_name == INVENTORY_LOGICAL_NAME:
            inventory_reads.append(manifest.artifact_id)
        return read_chunk(self, manifest, chunk, *args, **kwargs)

    monkeypatch.setattr(POSIXArtifactRepository, "read_chunk", observed_read)
    for smaller in (
        dataclasses.replace(exact, max_manifest_bytes=len(inventory) - 1),
        dataclasses.replace(exact, max_members=shard_count - 1),
    ):
        inventory_reads.clear()
        with pytest.raises(RepositoryCorruptionError):
            restore_domain_checkpoint(
                repository,
                runtime,
                initial,
                checkpoint_id,
                section_templates=section,
                limits=smaller,
            )
        assert inventory_reads == []
    monkeypatch.undo()

    # Publication refuses an inventory above the explicit byte budget and
    # leaves nothing restorable. Both checkpoint IDs (embedded in every
    # descriptor) have the same encoded length, so the inventory size matches.
    refused_id = "domain-large-refusal-1"
    assert len(refused_id.encode("utf-8")) == len(checkpoint_id.encode("utf-8"))
    with pytest.raises(ValueError):
        publish_domain_checkpoint(
            repository,
            runtime,
            midway,
            checkpoint_id=refused_id,
            writer_id="w",
            sections=section,
            region_bytes=LARGE_REGION_BYTES,
            limits=dataclasses.replace(
                LARGE_INVENTORY_POLICY, max_manifest_bytes=len(inventory) - 1
            ),
        )
    with pytest.raises(IncompleteDistributedCheckpointError):
        load_distributed_checkpoint_commit(repository, refused_id)


def _migration_policy() -> TopologyRestartPolicy:
    return TopologyRestartPolicy(
        allow_topology_change=True,
        allow_tolerance_restart=True,
        maximum_relative_tolerance=1e-10,
    )


@pytest.mark.parametrize(
    ("source_owners", "target_owners", "devices"),
    [(2, 3, False), (3, 1, False), (2, 4, True)],
    ids=["lanes-2-to-3", "lanes-3-to-1", "devices-2-to-4"],
)
def test_changed_topology_migration_preserves_state_and_rng_lineage(
    tmp_path: Path, source_owners: int, target_owners: int, devices: bool
) -> None:
    model = mace_model(2)
    source_runtime = _runtime(model, owners=source_owners, devices=devices)
    midway = _verlet(source_runtime, _initial(source_runtime, model), 2)
    repository = _repository(tmp_path)
    publish_domain_checkpoint(
        repository,
        source_runtime,
        midway,
        checkpoint_id="domain-migrate",
        writer_id="w",
        loader_cursors=(_cursor(),),
        region_bytes=256,
        limits=REGION_LIMITS,
    )
    target_runtime = _runtime(model, owners=target_owners, devices=devices)
    template = _initial(target_runtime, model)
    with pytest.raises(AtomisticDomainCheckpointError):
        migrate_domain_checkpoint(
            repository,
            target_runtime,
            template,
            "domain-migrate",
            policy=TopologyRestartPolicy(allow_topology_change=False),
            relative_tolerance=1e-10,
            limits=REGION_LIMITS,
        )
    restore = migrate_domain_checkpoint(
        repository,
        target_runtime,
        template,
        "domain-migrate",
        policy=_migration_policy(),
        relative_tolerance=1e-10,
        limits=REGION_LIMITS,
    )
    state = restore.state
    evidence = restore.migration
    assert restore.restart_class == "tolerance"
    assert evidence is not None
    assert evidence.source_owner_count == source_owners
    assert evidence.target_owner_count == target_owners
    assert evidence.restored_rows == midway.layout.logical_count
    assert evidence.read_payload_bytes > 0
    assert evidence.rng_lineage_id == restore.record.rng_lineage_id
    assert restore.loader_cursors == (_cursor(),)
    assert state.run_id == midway.run_id
    for name in ("momenta", "masses", "species", "image_counts"):
        np.testing.assert_array_equal(
            np.asarray(state.layout.collect(getattr(state, name))),
            np.asarray(midway.layout.collect(getattr(midway, name))),
        )
    np.testing.assert_array_equal(
        np.asarray(state.layout.collect(state.unwrapped_positions())),
        np.asarray(midway.layout.collect(midway.unwrapped_positions())),
    )
    np.testing.assert_array_equal(
        np.asarray(state.layout.collect(state.layout.stable_ids)),
        np.asarray(midway.layout.collect(midway.layout.stable_ids)),
    )
    np.testing.assert_array_equal(
        np.asarray(state.layout.collect(state.atom_payload["history"])),
        np.asarray(midway.layout.collect(midway.atom_payload["history"])),
    )
    np.testing.assert_array_equal(
        np.asarray(state.layout.collect(state.force.forces)),
        np.asarray(midway.layout.collect(midway.force.forces)),
    )
    assert bool(state.force_cache_valid)
    np.testing.assert_array_equal(
        np.asarray(state.replicated_payload["thermostat"]),
        np.asarray(midway.replicated_payload["thermostat"]),
    )
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(state.rng_key)),
        np.asarray(jax.random.key_data(midway.rng_key)),
    )
    assert int(state.step_index) == int(midway.step_index)
    assert float(state.time) == float(midway.time)
    np.testing.assert_array_equal(
        np.asarray(state.case_identity_words), np.asarray(midway.case_identity_words)
    )
    np.testing.assert_allclose(
        float(target_runtime.evaluate(state).energy),
        float(source_runtime.evaluate(midway).energy),
        rtol=1e-11,
    )
    continued = _verlet(target_runtime, state, 2)
    reference = _verlet(source_runtime, midway, 2)
    np.testing.assert_allclose(
        np.asarray(continued.layout.collect(continued.unwrapped_positions())),
        np.asarray(reference.layout.collect(reference.unwrapped_positions())),
        rtol=1e-10,
        atol=1e-10,
    )
    with pytest.raises(AtomisticDomainCheckpointError):
        migrate_domain_checkpoint(
            repository,
            source_runtime,
            _initial(source_runtime, model),
            "domain-migrate",
            policy=_migration_policy(),
            relative_tolerance=1e-10,
            limits=REGION_LIMITS,
        )


def test_migration_with_empty_source_owners(tmp_path: Path) -> None:
    """Four source owners with two empty ones migrate onto two owners."""
    model = mace_model(2)
    source_runtime = _runtime(model, owners=4)
    positions, _ = structure()
    fractional = np.linalg.solve(CELL.T, positions.T).T % 1.0
    fractional[:, 0] = 0.05 + 0.4 * fractional[:, 0]
    source = _initial(source_runtime, model, fractional @ CELL)
    loads = np.asarray(source.layout.active).reshape((4, -1)).sum(axis=1)
    assert np.count_nonzero(loads == 0) == 2
    repository = _repository(tmp_path)
    publish_domain_checkpoint(
        repository, source_runtime, source, checkpoint_id="domain-empty", writer_id="w"
    )
    target_runtime = _runtime(model, owners=2)
    restore = migrate_domain_checkpoint(
        repository,
        target_runtime,
        _initial(target_runtime, model),
        "domain-empty",
        policy=_migration_policy(),
        relative_tolerance=1e-10,
    )
    np.testing.assert_allclose(
        float(target_runtime.evaluate(restore.state).energy),
        float(source_runtime.evaluate(source).energy),
        rtol=1e-11,
    )


def test_migration_capacity_refusal_is_not_scientific_failure(tmp_path: Path) -> None:
    model = mace_model(2)
    source_runtime = _runtime(model, owners=2)
    source = _initial(source_runtime, model)
    repository = _repository(tmp_path)
    publish_domain_checkpoint(
        repository, source_runtime, source, checkpoint_id="domain-small", writer_id="w"
    )
    target_runtime = _runtime(model, owners=1, local_capacity=4)
    with pytest.raises(AtomisticDomainMigrationCapacityError, match="owner rows"):
        migrate_domain_checkpoint(
            repository,
            target_runtime,
            ingress(target_runtime, model, *(array[:4] for array in structure())),
            "domain-small",
            policy=_migration_policy(),
            relative_tolerance=1e-10,
        )
    # The source commit is untouched and still restores bitwise.
    restore = restore_domain_checkpoint(
        repository, source_runtime, source, "domain-small"
    )
    _assert_bitwise(restore.state, source)


def test_migrated_replicated_trees_are_device_leaves_for_eager_diagnostics(
    tmp_path: Path,
) -> None:
    """Migration restores controller/ledger leaves as placed device arrays.

    Copied replicated leaves (and typed keys) are bitwise and placed like the
    template's; eager diagnostics of the migrated state admit them, and the
    2 -> 3 owner continuation matches the source within the tolerance class.
    """
    model = _model()
    source_dynamics, source = _domain(model, VelocityVerletPlan(0.5), 2, ensemble="nve")
    for _ in range(2):
        source = _domain_step(source_dynamics, source).accepted_state
    repository = _repository(tmp_path)
    publish_domain_checkpoint(
        repository,
        source_dynamics.runtime,
        source,
        checkpoint_id="domain-dynamics-migrate",
        writer_id="w",
    )
    target_dynamics, template = _domain(model, VelocityVerletPlan(0.5), 3, ensemble="nve")
    state = migrate_domain_checkpoint(
        repository,
        target_dynamics.runtime,
        template,
        "domain-dynamics-migrate",
        policy=_migration_policy(),
        relative_tolerance=1e-10,
    ).state
    for name in ("replicated_payload", "controllers", "ledger", "conditioning"):
        actual_tree, source_tree = getattr(state, name), getattr(source, name)
        assert jax.tree.structure(actual_tree) == jax.tree.structure(source_tree)
        for (path, actual), expected, placed in zip(
            jax.tree_util.tree_flatten_with_path(actual_tree)[0],
            jax.tree.leaves(source_tree),
            jax.tree.leaves(getattr(template, name)),
            strict=True,
        ):
            location = name + jax.tree_util.keystr(path)
            assert isinstance(actual, jax.Array), location
            assert actual.dtype == expected.dtype, location
            assert actual.shape == expected.shape, location
            assert actual.committed == placed.committed, location
            assert actual.sharding.is_equivalent_to(placed.sharding, placed.ndim), (
                location
            )
            if jax.dtypes.issubdtype(expected.dtype, jax.dtypes.prng_key):
                assert str(jax.random.key_impl(actual)) == str(
                    jax.random.key_impl(expected)
                ), location
                actual, expected = (
                    jax.random.key_data(actual),
                    jax.random.key_data(expected),
                )
            np.testing.assert_array_equal(
                np.atleast_1d(np.asarray(actual)).view(np.uint8),
                np.atleast_1d(np.asarray(expected)).view(np.uint8),
                err_msg=location,
            )

    diagnostics = target_dynamics.diagnostics(state)
    reference = source_dynamics.diagnostics(source)
    assert bool(diagnostics.successful)
    assert int(diagnostics.rejection_reasons) == int(reference.rejection_reasons) == 0
    assert int(diagnostics.degrees_of_freedom) == int(reference.degrees_of_freedom)
    potential = np.asarray(diagnostics.potential_energy)
    expected_potential = np.asarray(reference.potential_energy)
    assert potential.dtype == expected_potential.dtype == np.float64
    np.testing.assert_array_equal(
        potential.view(np.uint64), expected_potential.view(np.uint64)
    )
    # Owner-ordered reductions regroup across 2 -> 3 owners.
    for field in ("kinetic_energy", "total_energy", "extended_energy", "temperature"):
        np.testing.assert_allclose(
            float(getattr(diagnostics, field)),
            float(getattr(reference, field)),
            rtol=1e-12,
            err_msg=field,
        )

    continued, expected_state = state, source
    for _ in range(2):
        step = _domain_step(target_dynamics, continued)
        expected_step = _domain_step(source_dynamics, expected_state)
        assert bool(step.successful) and bool(expected_step.successful)
        continued, expected_state = step.accepted_state, expected_step.accepted_state
    np.testing.assert_allclose(
        np.asarray(continued.layout.collect(continued.unwrapped_positions())),
        np.asarray(expected_state.layout.collect(expected_state.unwrapped_positions())),
        rtol=1e-10,
        atol=1e-10,
    )
    assert int(continued.ledger.accepted_steps) == int(
        expected_state.ledger.accepted_steps
    )
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(continued.rng_key)),
        np.asarray(jax.random.key_data(expected_state.rng_key)),
    )
    assert bool(target_dynamics.diagnostics(continued).successful)
