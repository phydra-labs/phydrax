# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Cold multi-array restore authenticates each committed table once per restore."""

from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from jax.sharding import SingleDeviceSharding

from phydrax.lifecycle import _distributed_checkpoint as checkpoint
from phydrax.lifecycle._chunk_repository import RepositoryCorruptionError
from phydrax.lifecycle._models import CheckpointManifest, CheckpointShard
from phydrax.lifecycle._repository import POSIXArtifactRepository
from phydrax.lifecycle._restart_topology import TopologyRestartPolicy
from tests._support.distributed_meshing.restart_builders import (
    build_repository,
    checkpoint_state,
    restart_relation,
    scientific_arrays,
)
from tools.compile_counter import CompileCounter


def _published_checkpoint(
    root: Path,
) -> tuple[POSIXArtifactRepository, CheckpointManifest]:
    repository = build_repository(root)
    checkpoint.publish_process_meshing_checkpoint(
        repository,
        "mesh-accepted",
        "source-execution",
        checkpoint_state(),
        {name: jax.device_put(value) for name, value in scientific_arrays().items()},
        analysis_plan_id="mesh-analysis",
        numeric_revision_id="accepted-revision",
        writer_id="process-writer",
    )
    manifest = checkpoint.assemble_distributed_checkpoint_from_repository(
        repository,
        "mesh-accepted",
        "mesh-analysis",
        "accepted-revision",
        "source-execution",
        expected_process_count=1,
    )
    return repository, manifest


@pytest.mark.parametrize("packet_bytes", (None, 128))
def test_multi_array_restore_reuses_only_its_own_authenticated_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packet_bytes: int | None
) -> None:
    repository, manifest = _published_checkpoint(tmp_path / "checkpoint")
    sharding = SingleDeviceSharding(jax.local_devices()[0])
    reference = scientific_arrays()
    calls: list[str] = []
    original = checkpoint._process_shard_table

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[1].manifest_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(checkpoint, "_process_shard_table", counted)
    # The isolated public API is the prior per-array admission work oracle.
    for name in sorted(reference):
        value = checkpoint.restore_global_array_from_checkpoint(
            repository,
            manifest,
            checkpoint._meshing_array_path(name),
            sharding,
            maximum_host_packet_bytes=packet_bytes,
        )
        np.testing.assert_array_equal(
            np.asarray(value).view(np.uint8), reference[name].view(np.uint8)
        )
    assert len(calls) == len(reference)
    calls.clear()
    for repeat in (1, 2):
        restored = checkpoint.restore_meshing_checkpoint(
            repository,
            manifest,
            {name: sharding for name in reference},
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=checkpoint_state().checkpoint_state_id,
            maximum_host_packet_bytes=packet_bytes,
        )
        # One state-record table plus one shared scientific-array table. The
        # next restore earns both admissions anew; no global cache survives.
        assert len(calls) == 2 * repeat
        assert restored.state.to_payload() == checkpoint_state().to_payload()
        for name, expected in reference.items():
            np.testing.assert_array_equal(
                np.asarray(restored.arrays[name]).view(np.uint8), expected.view(np.uint8)
            )


def test_reused_table_still_refuses_a_substituted_later_array_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, manifest = _published_checkpoint(tmp_path / "checkpoint")
    selected = next(
        shard
        for shard in manifest.shards
        if dict(shard.metadata)["array_path"]
        == checkpoint._meshing_array_path("vertex_ids")
    )
    substituted = CheckpointShard(
        selected.shard_id + "-substituted",
        selected.payload_digest,
        selected.byte_count,
        selected.layout_ids,
        metadata=selected.metadata,
    )
    changed = CheckpointManifest(
        manifest.checkpoint_id,
        manifest.analysis_plan_id,
        manifest.numeric_revision_id,
        manifest.execution_plan_id,
        tuple(substituted if shard is selected else shard for shard in manifest.shards),
        complete=True,
    )
    calls: list[str] = []
    original = checkpoint._process_shard_table

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[1].manifest_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(checkpoint, "_process_shard_table", counted)
    sharding = SingleDeviceSharding(jax.local_devices()[0])
    with pytest.raises(RepositoryCorruptionError, match=r"checkpoint\.shard_descriptor"):
        checkpoint.restore_meshing_checkpoint(
            repository,
            changed,
            {name: sharding for name in scientific_arrays()},
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=checkpoint_state().checkpoint_state_id,
        )
    assert len(calls) == 2


@pytest.mark.parametrize("name", ("vertex_ids", "coordinates"))
def test_full_extent_packet_is_adopted_without_device_assembly_or_compilation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    repository, manifest = _published_checkpoint(tmp_path / "checkpoint")
    expected = scientific_arrays()[name]
    sharding = SingleDeviceSharding(jax.local_devices()[0])

    def redundant_assembly(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "restore.packet must adopt full-extent packets; got redundant device assembly"
        )

    monkeypatch.setattr(checkpoint.jnp, "zeros", redundant_assembly)
    monkeypatch.setattr(checkpoint, "_place_checkpoint_packet", redundant_assembly)
    with CompileCounter() as compiler:
        restored = checkpoint.restore_global_array_from_checkpoint(
            repository,
            manifest,
            checkpoint._meshing_array_path(name),
            sharding,
            maximum_host_packet_bytes=128,
        )
    assert compiler.count == 0
    np.testing.assert_array_equal(
        np.asarray(restored).view(np.uint8), expected.view(np.uint8)
    )
    with pytest.raises(ValueError, match="over-limit packet"):
        checkpoint.restore_global_array_from_checkpoint(
            repository,
            manifest,
            checkpoint._meshing_array_path(name),
            sharding,
            maximum_host_packet_bytes=1,
        )


def test_shard_selection_index_reads_metadata_once_and_retains_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, manifest = _published_checkpoint(tmp_path / "checkpoint")
    original = checkpoint._metadata
    calls = 0

    def counted(shard: CheckpointShard) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return original(shard)

    monkeypatch.setattr(checkpoint, "_metadata", counted)
    index = checkpoint._checkpoint_shards_by_path(manifest)
    assert calls == len(manifest.shards)
    for path, selected in index.items():
        assert selected == tuple(
            shard
            for shard in manifest.shards
            if dict(shard.metadata)["array_path"] == path
        )
    sharding = SingleDeviceSharding(jax.local_devices()[0])
    for name, expected in scientific_arrays().items():
        path = checkpoint._meshing_array_path(name)
        _, load, selected = checkpoint._checkpoint_array_loader(
            repository, manifest, path, shards_by_path=index
        )
        assert selected == index[path]
        restored = load(tuple(slice(0, size) for size in expected.shape))
        np.testing.assert_array_equal(restored.view(np.uint8), expected.view(np.uint8))
        restored_device = checkpoint._restore_checkpoint_array(
            repository,
            manifest,
            path,
            sharding,
            limits=checkpoint._DISTRIBUTED_CHECKPOINT_LIMITS,
            parent_manifest=None,
            maximum_host_packet_bytes=128,
            tables={},
            shards_by_path=index,
        )
        np.testing.assert_array_equal(
            np.asarray(restored_device).view(np.uint8), expected.view(np.uint8)
        )


def test_private_restore_worksets_do_not_walk_scientific_trees() -> None:
    walks = 0

    @jax.tree_util.register_pytree_node_class
    class CountedTree:
        def tree_flatten(self) -> tuple[tuple[()], None]:
            nonlocal walks
            walks += 1
            return (), None

        @classmethod
        def tree_unflatten(cls, auxiliary: None, children: tuple[()]) -> "CountedTree":
            return cls()

    tree = CountedTree()
    recipes = ({"scientific_tree": tree},)
    plan = checkpoint._MeshingRestorePlan((), recipes, (), frozenset(), frozenset())
    arrays = checkpoint._RestoredMeshingArrays({}, None, 128)
    sources = checkpoint._RestoredMeshingSources((checkpoint_state(),), None)
    assert walks == 0
    assert plan.recipes is recipes
    for workset in (plan, arrays, sources):
        leaves = jax.tree.leaves(workset)
        assert len(leaves) == 1
        assert leaves[0] is workset
    assert walks == 0
