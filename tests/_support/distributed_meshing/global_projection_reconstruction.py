# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Real Gloo ranks reconstruct constructor-derived collective geometry banks."""

import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import phydrax as phx
from phydrax._fingerprint import (
    logical_array_value_collection_digest,
    tree_content_equal,
)
from phydrax.discretization._cell_geometry import CellGeometryStorageProjection
from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
from phydrax.lifecycle._meshing_sources import (
    MeshingSourceValidation,
    reconstruct_meshing_source,
    register_meshing_source_artifacts,
)
from phydrax.meshing._result import CellMeshingResult
from tests._support.distributed_meshing.native_distributed import _prepare_source
from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices


_MAXIMUM_CHUNK_BYTES = 256


def run_global_projection_workers(function: str) -> list[dict[str, Any]]:
    """Launch two Gloo ranks with one CPU device per process."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_on_forced_devices,
                "tests._support.distributed_meshing.global_projection_reconstruction",
                function,
                (),
                devices=1,
                env=gloo_rank_environment(coordinator, rank),
            )
            for rank in range(2)
        ]
        return [future.result() for future in futures]


def _published_target() -> CellMeshingResult:
    source = _prepare_source("global_projection_reconstruction", {}).source
    assert source.certification is not None
    source.certification.require_passed()
    source.audit.require_passed()
    partition_policy = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.MORTON, 2, halo_width=2
    )
    distribution = phx.meshing.prepare_mesh_distribution(
        phx.meshing.MeshPart("domain", source), policy=partition_policy
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=phx.discretization.AdaptiveSimplexPolicy(
            vertex_capacity=32, cell_capacity=32
        ),
        distribution=distribution,
        partition_policy=partition_policy,
    )
    partitioned = phx.meshing.partition_adaptive_simplex(
        phx.meshing.prepare_adaptive_simplex(source, policy=policy)
    )
    assert partitioned.layout.vertex_capacity == 32
    assert partitioned.layout.cell_capacity == 32
    update = phx.discretization.refine_adaptive_simplex_parts(
        partitioned.layout,
        partitioned.parts,
        partitioned.states,
        partitioned.states.mesh.cell_active,
    )
    for shard in update.report.status.addressable_shards:
        np.testing.assert_array_equal(
            np.asarray(shard.data), np.zeros(shard.data.shape, dtype=np.int32)
        )
    (result,) = phx.meshing.commit_partitioned_adaptive_simplex(
        partitioned, update.state
    ).local_results
    assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert result.source.result_id == source.result_id
    result.target.audit.require_passed()
    return result.target


def _admit_projection(
    projection: CellGeometryStorageProjection,
) -> MeshingSourceValidation:
    bank = {
        f"projection/{index:04d}": leaf
        for index, leaf in enumerate(jax.tree.leaves(projection))
        if isinstance(leaf, jax.Array)
    }
    assert bank
    digest = logical_array_value_collection_digest(
        bank, maximum_chunk_bytes=_MAXIMUM_CHUNK_BYTES
    )
    proof = MeshingSourceValidation(
        (), bank, digest, maximum_chunk_bytes=_MAXIMUM_CHUNK_BYTES
    )
    proof.require_binding(bank, digest)
    return proof


def _assert_local_bytes_equal(
    original: CellGeometryStorageProjection,
    rebuilt: CellGeometryStorageProjection,
    proof: MeshingSourceValidation,
) -> int:
    assert type(rebuilt) is type(original)
    assert tree_content_equal(original, rebuilt, array_equal=proof.array_equal)
    first, first_structure = jax.tree.flatten(original)
    second, second_structure = jax.tree.flatten(rebuilt)
    assert first_structure == second_structure
    shard_count = 0
    for left, right in zip(first, second, strict=True):
        if not isinstance(left, jax.Array):
            continue
        assert isinstance(right, jax.Array)
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        assert left.is_fully_addressable == right.is_fully_addressable
        for a, b in zip(left.addressable_shards, right.addressable_shards, strict=True):
            assert a.index == b.index
            np.testing.assert_array_equal(
                np.asarray(a.data).view(np.uint8), np.asarray(b.data).view(np.uint8)
            )
            shard_count += 1
    assert shard_count > 0
    assert rebuilt.source_content_id == original.source_content_id
    assert (
        rebuilt.logical_coordinate_geometry_id == original.logical_coordinate_geometry_id
    )
    return shard_count


def _copy_projected_array(value: jax.Array, *, change_zero_sign: bool) -> jax.Array:
    """Copy only owner-local data, optionally flipping one zero on every rank."""
    local_arrays: list[jax.Array] = []
    for shard in value.addressable_shards:
        host = np.array(shard.data, copy=True)
        if change_zero_sign:
            assert host.dtype == np.dtype(np.float64)
            positions = np.flatnonzero(host.reshape(-1) == 0)
            assert positions.size > 0
            host.view(np.uint64).reshape(-1)[positions[0]] ^= np.uint64(1 << 63)
        local_arrays.append(jax.device_put(host, shard.device))
    return jax.make_array_from_single_device_arrays(
        value.shape, value.sharding, local_arrays
    )


def projection_reconstruction() -> dict[str, Any]:
    """Rebuild a real projection and refuse admitted but constructor-inconsistent bits."""
    try:
        assert jax.process_count() == 2
        assert jax.local_device_count() == 1
        register_meshing_source_artifacts()
        target = _published_target()
        storage = target.mesh.storage
        assert storage is not None
        assert storage.partition_index == jax.process_index()
        projection = storage.geometry_projection
        assert isinstance(projection, CellGeometryStorageProjection)
        nonaddressable = [
            value
            for _, value in projection.projected_arrays
            if not value.is_fully_addressable
        ]
        assert nonaddressable
        proof = _admit_projection(projection)
        bindings = {
            "logical_arrays": storage.logical_arrays,
            "source_elements": dict(projection.source_elements),
        }
        rebuilt = reconstruct_stored_fields(
            projection, bindings=bindings, array_equal=proof.array_equal
        )
        owner_rebuilt = reconstruct_meshing_source(
            projection, meshes=(target.mesh,), source_validation=proof
        )
        assert isinstance(owner_rebuilt, CellGeometryStorageProjection)
        shard_count = _assert_local_bytes_equal(projection, rebuilt, proof)
        assert _assert_local_bytes_equal(projection, owner_rebuilt, proof) == shard_count
        assert any(
            not value.is_fully_addressable for _, value in rebuilt.projected_arrays
        )
        action_name, action = next(
            (name, value)
            for name, value in projection.projected_arrays
            if name.startswith("geometry/action_weights/")
        )
        assert not action.is_fully_addressable
        fresh = _copy_projected_array(action, change_zero_sign=False)
        assert fresh is not action
        assert proof.array_equal(action, fresh)
        with pytest.raises(ValueError, match="common-bank object binding"):
            proof.array_equal(fresh, action)
        unbound = eqx.tree_at(
            lambda value: value.projected_arrays,
            projection,
            tuple(
                (name, fresh if name == action_name else value)
                for name, value in projection.projected_arrays
            ),
        )
        with pytest.raises(ValueError, match="common-bank object binding"):
            reconstruct_stored_fields(
                unbound, bindings=bindings, array_equal=proof.array_equal
            )
        changed = _copy_projected_array(action, change_zero_sign=True)
        altered = eqx.tree_at(
            lambda value: value.projected_arrays,
            projection,
            tuple(
                (name, changed if name == action_name else value)
                for name, value in projection.projected_arrays
            ),
        )
        altered_proof = _admit_projection(altered)
        assert not altered_proof.array_equal(changed, action)
        with pytest.raises(ValueError, match="changed projected_arrays"):
            reconstruct_stored_fields(
                altered, bindings=bindings, array_equal=altered_proof.array_equal
            )
        altered_storage = eqx.tree_at(
            lambda value: value.geometry_projection, storage, altered
        )
        altered_mesh = eqx.tree_at(
            lambda value: value.storage, target.mesh, altered_storage
        )
        bound_storage = altered_mesh.storage
        assert bound_storage is not None
        bound_projection = bound_storage.geometry_projection
        assert isinstance(bound_projection, CellGeometryStorageProjection)
        assert dict(bound_projection.projected_arrays)[action_name] is changed
        with pytest.raises(ValueError, match="changed projected_arrays"):
            reconstruct_meshing_source(
                bound_projection,
                meshes=(altered_mesh,),
                source_validation=altered_proof,
            )
        return {
            "rank": jax.process_index(),
            "process_count": jax.process_count(),
            "local_device_count": jax.local_device_count(),
            "nonaddressable_arrays": len(nonaddressable),
            "exact_local_shards": shard_count,
            "source_content_id": rebuilt.source_content_id,
            "coordinate_geometry_id": rebuilt.logical_coordinate_geometry_id,
            "global_cell_count": rebuilt.global_cell_count,
            "unbound_original_refused": True,
            "admitted_mutation_refused": True,
            "owner_mutation_refused": True,
        }
    finally:
        jax.distributed.shutdown()


def exact_payload_comparison() -> dict[str, Any]:
    """Authenticate aligned nonaddressable shards by bits, not numeric equality."""
    try:
        assert jax.process_count() == 2
        assert jax.local_device_count() == 1
        placement = NamedSharding(
            Mesh(np.asarray(jax.devices(), dtype=np.object_), ("owner",)),
            PartitionSpec("owner"),
        )

        def placed(bits: tuple[int, int]) -> jax.Array:
            local = np.asarray((bits,), dtype=np.uint64).view(np.float64)
            return jax.make_array_from_process_local_data(
                placement, local, global_shape=(2, 2)
            )

        original = placed((0, 0x7FF8000000000042))
        fresh = placed((0, 0x7FF8000000000042))
        negative_zero = placed((0x8000000000000000, 0x7FF8000000000042))
        other_nan = placed((0, 0x7FF8000000000043))
        assert all(
            not value.is_fully_addressable
            for value in (original, fresh, negative_zero, other_nan)
        )
        bank = {"payload": original}
        digest = logical_array_value_collection_digest(bank, maximum_chunk_bytes=8)
        proof = MeshingSourceValidation((), bank, digest, maximum_chunk_bytes=8)
        proof.require_binding(bank, digest)
        # Different call counts across ranks ensure comparison needs no collective
        # rendezvous; admission above, unlike owner-local equality, is collective.
        for _ in range(jax.process_index() + 1):
            assert proof.array_equal(original, fresh)
            assert not proof.array_equal(original, negative_zero)
            assert not proof.array_equal(original, other_nan)
            with pytest.raises(ValueError, match="common-bank object binding"):
                proof.array_equal(fresh, original)
            with pytest.raises(ValueError, match="common-bank object binding"):
                proof.array_equal(fresh, fresh)
        np.testing.assert_array_equal(
            np.asarray(original.addressable_shards[0].data).view(np.uint64),
            np.asarray(((0, 0x7FF8000000000042),), dtype=np.uint64),
        )
        return {
            "rank": jax.process_index(),
            "process_count": jax.process_count(),
            "local_device_count": jax.local_device_count(),
            "nonaddressable": True,
            "equal_copy_accepted": True,
            "signed_zero_refused": True,
            "nan_payload_refused": True,
            "unbound_original_refused": True,
            "comparison_rounds": jax.process_index() + 1,
            "maximum_chunk_bytes": proof.maximum_chunk_bytes,
            "digest": digest,
        }
    finally:
        jax.distributed.shutdown()


def local_layout_comparison() -> dict[str, Any]:
    """Compare equivalent fully addressable layouts with the default chunk bound."""
    assert jax.process_count() == 1
    assert jax.local_device_count() == 2
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("layout",))
    partitioned = NamedSharding(mesh, PartitionSpec("layout"))
    replicated = NamedSharding(mesh, PartitionSpec())
    bits = np.asarray(
        [0, 0x7FF8000000000001, 0x8000000000000000, 0x7FF8000000000002],
        dtype=np.uint64,
    )
    original = jax.device_put(bits.view(np.float64), partitioned)
    equal = jax.device_put(bits.view(np.float64), replicated)
    changed_zero = bits.copy()
    changed_zero[0] = np.uint64(0x8000000000000000)
    changed_nan = bits.copy()
    changed_nan[1] = np.uint64(0x7FF8000000000002)
    other_zero = jax.device_put(changed_zero.view(np.float64), replicated)
    other_nan = jax.device_put(changed_nan.view(np.float64), replicated)
    proof = MeshingSourceValidation((), {}, logical_array_value_collection_digest({}))
    assert proof.maximum_chunk_bytes is None
    assert original.is_fully_addressable and equal.is_fully_addressable
    assert original.addressable_shards[0].index != equal.addressable_shards[0].index
    assert proof.array_equal(original, equal)
    assert not proof.array_equal(original, other_zero)
    assert not proof.array_equal(original, other_nan)
    return {
        "equal_copy_accepted": True,
        "signed_zero_refused": True,
        "nan_payload_refused": True,
        "default_chunk_limit": True,
    }
