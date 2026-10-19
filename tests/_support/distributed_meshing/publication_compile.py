# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Cold and repeated consumer publication compilation receipts."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import numpy as np

import phydrax as phx
from phydrax.geometry._collective_domain_coverage import _raw_partition
from phydrax.meshing._device_adaptation import _resized_epoch
from tests._support.cell_meshes import unit_square
from tests._support.distributed_meshing.native_distributed import _prepare_source
from tools.compile_counter import CompileCounter


def repeated_publication(source_kind: str) -> dict[str, Any]:
    if source_kind == "native-domain":
        source = _prepare_source("publication_compile", {}).source
    elif source_kind == "certified-mesh":
        source = phx.meshing.certify_cell_mesh(
            unit_square(), phx.SpatialCoordinateContract.si()
        )
    else:
        raise ValueError(
            f"publication_compile.source_kind must identify a test source; got {source_kind!r}"
        )
    partition_policy = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.MORTON, 2, halo_width=1
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
    prepared = phx.meshing.prepare_adaptive_simplex(source, policy=policy)
    partitioned = phx.meshing.partition_adaptive_simplex(prepared)
    update = phx.discretization.refine_adaptive_simplex_parts(
        partitioned.layout,
        partitioned.parts,
        partitioned.states,
        partitioned.states.mesh.cell_active,
    )
    jax.block_until_ready(update)
    partition_valid = jax.block_until_ready(
        _raw_partition(update.state, partitioned.states)
    )
    altered = eqx.tree_at(
        lambda value: value.mesh.cell_ids,
        update.state,
        update.state.mesh.cell_ids + 1,
    )
    resized = _resized_epoch(partitioned.states, partitioned.layout)
    for name in ("refine_rejected", "coarsen_marked", "counters"):
        original_leaf = getattr(partitioned.states, name)
        resized_leaf = getattr(resized, name)
        assert original_leaf.sharding == resized_leaf.sharding, name
        np.testing.assert_array_equal(np.asarray(resized_leaf), 0)
    with CompileCounter(capture_stack=True) as raw_warm:
        repeated_valid = jax.block_until_ready(
            _raw_partition(update.state, partitioned.states)
        )
        rejected = jax.block_until_ready(_raw_partition(altered, partitioned.states))
        resized_valid = jax.block_until_ready(_raw_partition(update.state, resized))
    assert bool(partition_valid) and bool(repeated_valid)
    assert not bool(rejected)
    assert bool(resized_valid)
    with CompileCounter(capture_stack=True) as cold:
        first = phx.meshing.commit_partitioned_adaptive_simplex(partitioned, update.state)
    with CompileCounter(capture_stack=True) as warm:
        second = phx.meshing.commit_partitioned_adaptive_simplex(
            partitioned, update.state
        )
    for left, right in zip(first.local_results, second.local_results, strict=True):
        assert left.target.result_id == right.target.result_id
        assert left.target.mesh.mesh_id == right.target.mesh.mesh_id
        np.testing.assert_array_equal(
            np.asarray(left.target.mesh.coordinates, dtype=np.float64).view(np.uint64),
            np.asarray(right.target.mesh.coordinates, dtype=np.float64).view(np.uint64),
        )
        for left_block, right_block in zip(
            left.target.mesh.blocks, right.target.mesh.blocks, strict=True
        ):
            np.testing.assert_array_equal(left_block.vertices, right_block.vertices)
            np.testing.assert_array_equal(left_block.global_ids, right_block.global_ids)
    return {
        "cold_count": cold.count,
        "warm_count": warm.count,
        "warm_events": [event.to_record() for event in warm.events],
        "warm_callsites": [
            [frame for frame in event.first_party_stack if frame.startswith("phydrax/")][
                -3:
            ]
            for event in warm.events
        ],
        "result_ids": [result.target.result_id for result in first.local_results],
        "raw_partition_warm_count": raw_warm.count,
    }
