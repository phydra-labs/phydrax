# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Scientific and storage identity receipts for consecutive Gloo mesh epochs."""

from typing import Any

import jax
import numpy as np

import phydrax as phx
from phydrax.meshing._device_adaptation import prepare_partitioned_mesh_adaptation
from phydrax.meshing._result import CollectiveMeshEvidence
from tests._support.distributed_meshing.native_distributed import _prepare_source
from tests._support.distributed_meshing.native_distributed_records import (
    _require_instance,
    _require_present,
)


def consecutive_epochs() -> dict[str, Any]:
    """Commit two bisection epochs, consuming the first owner-local publication."""
    # Phydrax bootstrap consumes the runner's Gloo environment before backend use.
    assert jax.process_count() == 2
    assert jax.local_device_count() == 1
    rank = jax.process_index()
    setup = _prepare_source("rank_identity", {})
    np.testing.assert_array_equal(
        setup.points,
        np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        setup.cells, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    )
    source = setup.source
    certification = _require_present(source.certification)
    assert _require_present(certification.coverage).status == "certified"
    source.audit.require_passed()
    initial_source_result_id = source.result_id
    partition_policy = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.MORTON, 2, halo_width=2
    )
    distribution = phx.meshing.prepare_mesh_distribution(
        phx.meshing.MeshPart("domain", source), policy=partition_policy
    )
    device_policy = phx.discretization.AdaptiveSimplexPolicy(
        vertex_capacity=64, cell_capacity=64
    )
    epochs: list[dict[str, Any]] = []
    for epoch in range(2):
        source_storage = source.mesh.storage
        if epoch:
            assert source_storage is not None
            assert source_storage.partition_index == rank
            source_evidence = _require_instance(
                source.collective_evidence, CollectiveMeshEvidence
            )
            assert not (
                source_evidence.compiled_states.mesh.coordinates.is_fully_addressable
            )
        else:
            assert source_storage is None
        policy = phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
            device_policy=device_policy,
            distribution=distribution,
            partition_policy=partition_policy,
        )
        partitioned = prepare_partitioned_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                source, phx.meshing.MarkedMeshAdaptation(), policy=policy
            )
        )
        assert partitioned.layout.vertex_capacity == 64
        assert partitioned.layout.cell_capacity == 64
        witness = _require_present(partitioned.source_witness)
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
        target = result.target
        target.audit.require_passed()
        evidence = _require_instance(target.collective_evidence, CollectiveMeshEvidence)
        evidence.require_passed()
        storage = _require_present(target.mesh.storage)
        assert storage.partition_index == rank
        epochs.append(
            {
                "epoch": epoch,
                "partition": storage.partition_index,
                "source_is_owner_local": source_storage is not None,
                "source_result_id": result.source.result_id,
                "witness_source_result_id": witness.source_result_id,
                "predecessor_evidence_id": witness.predecessor_evidence_id,
                "result_id": target.result_id,
                "source_evidence_id": evidence.source_evidence_id,
                "theorem_evidence_id": evidence.evidence_id,
                "stored_content_id": target.stored_content_id,
                "child_stored_content_id": evidence.stored_content_id,
                "global_cell_count": evidence.global_entity_counts[-1],
            }
        )
        source = target
        distribution = _require_present(result.distribution).target
    return {
        "rank": rank,
        "initial_source_result_id": initial_source_result_id,
        "epochs": epochs,
    }
