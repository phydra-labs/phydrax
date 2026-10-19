# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Real native publication and placement-only field-history recovery."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._execution_runtime import ExecutionRuntime
from phydrax._fingerprint import tree_content_equal
from phydrax.discretization._adaptive_simplex import (
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
)
from phydrax.lifecycle._meshing_field_records import (
    MeshingFieldStateRole,
    prepare_meshing_field_index_binding,
    rebind_meshing_field_history,
    recover_meshing_field_history,
)
from phydrax.meshing._device_adaptation import commit_partitioned_mesh_restart
from phydrax.meshing._restart_distribution import (
    repack_simplex_restart,
    SimplexRestartRepack,
    SimplexRestartRepackProof,
)
from phydrax.meshing._result import CellMeshingResult, CollectiveMeshEvidence
from tests._support.distributed_meshing.native_distributed import _prepare_source


@dataclass(frozen=True)
class _HistoryScenario:
    source: CellMeshingResult
    foreign_source: CellMeshingResult
    target: CellMeshingResult
    receipt: SimplexRestartRepack
    proof: SimplexRestartRepackProof
    raw_role: MeshingFieldStateRole
    compact_role: MeshingFieldStateRole
    source_values: jax.Array


def _temperature(identifiers: np.ndarray) -> np.ndarray:
    # Dyadic values and signed zero expose bitwise preservation.
    values = identifiers.astype(np.float64) ** 2 + np.float64(0.25)
    return np.where(identifiers == 0, np.float64(-0.0), values)


def _prepare_history() -> _HistoryScenario:
    native = _prepare_source("field_history_proof", {}).source
    assert native.certification is not None
    native.certification.require_passed()
    partition = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.MORTON, 2, halo_width=1
    )
    distribution = phx.meshing.prepare_mesh_distribution(
        phx.meshing.MeshPart("domain", native), policy=partition
    )
    device_policy = phx.discretization.AdaptiveSimplexPolicy(
        vertex_capacity=32, cell_capacity=32
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=device_policy,
        distribution=distribution,
        partition_policy=partition,
    )
    prepared = phx.meshing.prepare_adaptive_simplex(native, policy=policy)
    partitioned = phx.meshing.partition_adaptive_simplex(prepared)
    # Publish the allocated source forest before moving its two roots.
    accepted = phx.meshing.commit_partitioned_adaptive_simplex(
        partitioned, partitioned.states
    ).local_results
    assert len(accepted) == 2
    assert all(
        result.status is phx.meshing.MeshAdaptationStatus.COMPLETE for result in accepted
    )
    source = accepted[0].target
    evidence = source.collective_evidence
    assert isinstance(evidence, CollectiveMeshEvidence)
    evidence.require_passed()
    source.audit.require_passed()
    saved = evidence.compiled_states
    raw_index = prepare_meshing_field_index_binding(source, "epoch/vertex_ids")
    np.testing.assert_array_equal(raw_index.index_values, saved.mesh.vertex_ids)
    assert saved.mesh.vertex_ids.is_fully_addressable
    source_values = jax.device_put(
        _temperature(np.asarray(saved.mesh.vertex_ids)), saved.mesh.vertex_ids.sharding
    )
    raw_role = MeshingFieldStateRole(
        "raw-temperature",
        "temperature",
        "material-history",
        source_values.shape,
        np.dtype(np.float64),
        1,
        value_units=(phx.units.KELVIN,),
        index_binding=raw_index,
    )
    ranks = jnp.arange(2, dtype=jnp.int32)[:, None]
    cell_owners = jnp.where(saved.mesh.cell_ids >= 0, ranks, -1)
    vertex_owners = jnp.full(saved.mesh.vertex_ids.shape, 2, dtype=jnp.int32)
    for owner in range(2):
        present = jnp.any(
            saved.mesh.vertex_ids[:, :, None]
            == saved.mesh.vertex_ids[owner, None, None, :],
            axis=-1,
        )
        vertex_owners = jnp.minimum(
            vertex_owners,
            jnp.where(present, jnp.int32(owner), jnp.int32(2)),
        )
    vertex_owners = jnp.where(saved.mesh.vertex_ids >= 0, vertex_owners, -1)
    receipt = repack_simplex_restart(
        ExecutionRuntime.current().root_group,
        partitioned.layout,
        saved,
        cell_owners,
        vertex_owners,
        jnp.broadcast_to(1 - ranks, saved.mesh.cell_ids.shape),
        vertex_history=(source_values,),
        source_result=source,
    )
    np.testing.assert_array_equal(receipt.status, np.zeros(2, dtype=np.int32))
    traffic = receipt.cell_migration_counts
    assert traffic is not None
    assert np.sum(np.asarray(traffic)) == np.sum(np.asarray(saved.mesh.cell_active))
    assert np.sum(np.asarray(traffic)) - np.trace(np.asarray(traffic)) > 0
    proof = SimplexRestartRepackProof(receipt)
    proof.require_receipt(receipt)
    source_partition = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.PROVIDER, 2, halo_width=partition.halo_width
    )
    distribution = phx.meshing.prepare_mesh_distribution(
        phx.meshing.MeshPart("domain", source), policy=source_partition
    )
    restart_policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=device_policy,
        distribution=distribution,
        partition_policy=source_partition,
    )
    adaptation = phx.meshing.prepare_mesh_adaptation(
        source, phx.meshing.MarkedMeshAdaptation(), policy=restart_policy
    )
    published = commit_partitioned_mesh_restart(receipt, adaptation).local_results
    assert len(published) == 2
    assert all(
        result.status is phx.meshing.MeshAdaptationStatus.COMPLETE for result in published
    )
    target = published[0].target
    assert target.collective_evidence is not None
    target.collective_evidence.require_passed()
    target.audit.require_passed()
    compact_index = prepare_meshing_field_index_binding(target, "mesh/vertex_ids")
    compact_role = MeshingFieldStateRole(
        "temperature",
        "temperature",
        "material-history",
        compact_index.index_values.shape,
        np.dtype(np.float64),
        1,
        value_units=(phx.units.KELVIN,),
        index_binding=compact_index,
    )
    return _HistoryScenario(
        source,
        accepted[1].target,
        target,
        receipt,
        proof,
        raw_role,
        compact_role,
        source_values,
    )


def _refusal(operation: Callable[[], Any]) -> str:
    try:
        operation()
    except ValueError as error:
        return str(error)
    return ""


def _consumer_refusals(
    scenario: _HistoryScenario,
    receipt: SimplexRestartRepack,
    rebound: MeshingFieldStateRole,
    values: jax.Array,
) -> dict[str, str]:
    return {
        "proof": _refusal(lambda: scenario.proof.require_receipt(receipt)),
        "rebind": _refusal(
            lambda: rebind_meshing_field_history(
                scenario.raw_role,
                scenario.source_values,
                scenario.target,
                receipt,
                history_index=0,
                proof=scenario.proof,
            )
        ),
        "recover": _refusal(
            lambda: recover_meshing_field_history(
                rebound,
                values,
                scenario.compact_role,
                scenario.target,
                receipt,
                history_index=0,
                proof=scenario.proof,
            )
        ),
    }


def _changed_arrays(
    receipt: SimplexRestartRepack,
) -> dict[str, SimplexRestartRepack]:
    return {
        "target-history": replace(
            receipt,
            vertex_history=(receipt.vertex_history[0].at[0, 0].add(0.5),),
        ),
        "saved-history": replace(
            receipt,
            saved_vertex_history=(receipt.saved_vertex_history[0].at[0, 0].add(0.5),),
        ),
        "target-forest": replace(
            receipt,
            states=eqx.tree_at(
                lambda state: state.mesh.coordinates,
                receipt.states,
                receipt.states.mesh.coordinates.at[0, 0, 0].add(0.5),
            ),
        ),
        "saved-forest": replace(
            receipt,
            saved_states=eqx.tree_at(
                lambda state: state.mesh.coordinates,
                receipt.saved_states,
                receipt.saved_states.mesh.coordinates.at[0, 0, 0].add(0.5),
            ),
        ),
        "requested-owners": replace(
            receipt,
            requested_target_owners=receipt.requested_target_owners.at[0, 0].add(1),
        ),
        "vertex-owners": replace(
            receipt, vertex_owners=receipt.vertex_owners.at[0, 0].add(1)
        ),
        "saved-locations": replace(
            receipt,
            vertex_saved_locations=receipt.vertex_saved_locations.at[0, 0, 0].add(1),
        ),
        "saved-controls": replace(
            receipt, saved_clocks=receipt.saved_clocks.at[0, 0].add(1)
        ),
        "record-counts": replace(
            receipt, vertices_after=receipt.vertices_after.at[0].add(1)
        ),
        "status": replace(receipt, status=receipt.status.at[0].add(1)),
    }


def _foreign_bindings(
    scenario: _HistoryScenario,
) -> dict[str, SimplexRestartRepack]:
    receipt = scenario.receipt
    layout = receipt.layout
    foreign_layout = AdaptiveSimplexLayout(
        layout.dimension,
        layout.ambient_dimension,
        vertex_capacity=layout.vertex_capacity,
        cell_capacity=layout.cell_capacity,
        protected_edge_capacity=layout.protected_edge_capacity,
        maximum_closure_iterations=layout.maximum_closure_iterations,
        maximum_coarsening_passes=layout.maximum_coarsening_passes,
        coordinate_dtype=np.dtype(layout.coordinate_dtype),
    )
    parts = receipt.execution_parts()
    foreign_parts = AdaptiveSimplexParts(
        tuple(parts.mesh.devices.flat)[::-1],
        axis_name=parts.axis_name,
        neighbor_pairs=parts.neighbor_pairs,
    )
    return {
        "source": replace(receipt, source_result=scenario.foreign_source),
        "layout": replace(receipt, layout=foreign_layout),
        "placement": replace(receipt, parts=foreign_parts),
    }


def earned_history_proof() -> dict[str, Any]:
    scenario = _prepare_history()
    receipt = scenario.receipt
    cold_role, cold_values = rebind_meshing_field_history(
        scenario.raw_role,
        scenario.source_values,
        scenario.target,
        receipt,
        history_index=0,
    )
    proven_role, proven_values = rebind_meshing_field_history(
        scenario.raw_role,
        scenario.source_values,
        scenario.target,
        receipt,
        history_index=0,
        proof=scenario.proof,
    )
    cold = recover_meshing_field_history(
        cold_role,
        cold_values,
        scenario.compact_role,
        scenario.target,
        receipt,
        history_index=0,
    )
    proven = recover_meshing_field_history(
        proven_role,
        proven_values,
        scenario.compact_role,
        scenario.target,
        receipt,
        history_index=0,
        proof=scenario.proof,
    )
    raw_index = proven_role.index_binding
    compact_index = scenario.compact_role.index_binding
    assert raw_index is not None and compact_index is not None
    raw_ids = np.asarray(receipt.states.mesh.vertex_ids)
    allocated = raw_ids >= 0
    expected_raw = _temperature(raw_ids[allocated])
    expected_compact = _temperature(np.asarray(compact_index.index_values))
    changed_source = scenario.source_values.at[0, 0].add(0.5)
    changed_recovered = proven_values.at[0, 0].add(0.5)
    payload_refusals = {
        "rebind": _refusal(
            lambda: rebind_meshing_field_history(
                scenario.raw_role,
                changed_source,
                scenario.target,
                receipt,
                history_index=0,
                proof=scenario.proof,
            )
        ),
        "recover": _refusal(
            lambda: recover_meshing_field_history(
                proven_role,
                changed_recovered,
                scenario.compact_role,
                scenario.target,
                receipt,
                history_index=0,
                proof=scenario.proof,
            )
        ),
    }
    # Payload edits leave the immutable repack proof bound.
    scenario.proof.require_receipt(receipt)
    return {
        "equivalence": {
            "raw_bitwise": tree_content_equal(cold_values, proven_values),
            "compact_bitwise": tree_content_equal(cold, proven),
            "role_identity": tree_content_equal(cold_role, proven_role),
            "raw_oracle": tree_content_equal(
                np.asarray(proven_values)[allocated], expected_raw
            ),
            "compact_oracle": tree_content_equal(np.asarray(proven), expected_compact),
            "raw_ids": tree_content_equal(
                raw_index.index_values, receipt.states.mesh.vertex_ids
            ),
            "target_identity": (
                raw_index.source_result_id
                == scenario.target.result_id
                == compact_index.source_result_id
            ),
            "banks": [raw_index.bank_name, compact_index.bank_name],
            "compact_diagnostics": {
                "ids": np.asarray(compact_index.index_values).tolist(),
                "expected_bits": expected_compact.view(np.uint64).tolist(),
                "actual_bits": np.asarray(proven).view(np.uint64).tolist(),
                "allocated_raw_ids": raw_ids[allocated].tolist(),
                "allocated_raw_bits": (
                    np.asarray(proven_values)[allocated].view(np.uint64).tolist()
                ),
            },
        },
        "array_refusals": {
            name: _consumer_refusals(scenario, value, proven_role, proven_values)
            for name, value in _changed_arrays(receipt).items()
        },
        "foreign_refusals": {
            name: _consumer_refusals(scenario, value, proven_role, proven_values)
            for name, value in _foreign_bindings(scenario).items()
        },
        "payload_refusals": payload_refusals,
        "proof_content_id": scenario.proof.content_id,
        "source_result_id": scenario.source.result_id,
        "target_result_id": scenario.target.result_id,
    }
