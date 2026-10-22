# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Real native publication and placement-only field-history recovery."""

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, replace
from typing import Any
from unittest.mock import patch

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
    MeshingFieldIndexBinding,
    MeshingFieldStateKind,
    MeshingFieldStateRole,
    MeshingRandomStream,
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
    result: phx.meshing.MeshAdaptationResult


def _temperature(identifiers: np.ndarray) -> np.ndarray:
    # Dyadic values and signed zero expose bitwise preservation.
    values = identifiers.astype(np.float64) ** 2 + np.float64(0.25)
    return np.where(identifiers == 0, np.float64(-0.0), values)


def _euler_state(identifiers: np.ndarray) -> np.ndarray:
    density = 1.0 + identifiers.astype(np.float64) / 8.0
    first = 0.25 + identifiers.astype(np.float64) / 8.0
    second = -0.125 - identifiers.astype(np.float64) / 16.0
    energy = 2.5 + (first**2 + second**2) / (2 * density)
    return np.stack((density, first, second, energy), axis=-1)


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
        cell_history=tuple(
            jnp.where(
                saved.mesh.cell_ids >= 0,
                saved.mesh.cell_ids.astype(dtype) + jnp.asarray(offset, dtype),
                jnp.asarray(0, dtype),
            )
            for dtype, offset in ((jnp.uint32, 17), (jnp.uint64, 2**40 + 17))
        ) + (
            jax.device_put(
                np.where(
                    np.asarray(saved.mesh.cell_ids)[..., None] >= 0,
                    _euler_state(np.asarray(saved.mesh.cell_ids)),
                    0.0,
                ),
                saved.mesh.cell_ids.sharding,
            ),
        ),
        vertex_history=(source_values,),
        source_result=source,
    )
    np.testing.assert_array_equal(receipt.status, np.zeros(2, dtype=np.int32))
    traffic = receipt.cell_migration_counts
    assert traffic is not None
    assert np.sum(np.asarray(traffic)) == np.sum(np.asarray(saved.mesh.cell_active))
    assert np.sum(np.asarray(traffic)) - np.trace(np.asarray(traffic)) > 0
    proof = SimplexRestartRepackProof(receipt)
    tree_leaves = jax.tree_util.tree_leaves

    def no_owner_warning_scan(tree: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            isinstance(tree, tuple)
            and len(tree) == 2
            and isinstance(tree[0], tuple)
            and isinstance(tree[1], dict)
            and any(value is proof or value is receipt for value in tree[0])
        ):
            raise AssertionError("Host proof admission must not build an Equinox bound method.")
        return tree_leaves(tree, *args, **kwargs)

    with patch("jax.tree_util.tree_leaves", side_effect=no_owner_warning_scan):
        SimplexRestartRepackProof.require_receipt(proof, receipt)
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
    with patch(
        "phydrax.meshing._restart_distribution.simplex_restart_repack_content_id",
        side_effect=AssertionError("Publication must consume its admitted proof, not rehash it."),
    ):
        with np.testing.assert_raises(ValueError):
            SimplexRestartRepackProof.require_receipt(proof, receipt)
    published = commit_partitioned_mesh_restart(
        receipt, adaptation, proof=proof
    ).local_results
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
        published[0],
    )


def _refusal(operation: Callable[[], Any]) -> str:
    try:
        operation()
    except (TypeError, ValueError) as error:
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


def _random_history_proof(scenario: _HistoryScenario) -> dict[str, Any]:
    """Integer counters travel as actual history leaves, without value conversion."""
    stream = MeshingRandomStream(
        "root-key", "threefry2x32", "meshing-test", "cell-probe", ("temperature",), "draw"
    )
    root = jax.random.key_data(jax.random.key(71, impl=stream.implementation))
    receipt = scenario.receipt
    source_index = prepare_meshing_field_index_binding(scenario.source, "epoch/cell_ids")
    compact_index = prepare_meshing_field_index_binding(scenario.target, "mesh/cell_ids")
    ids = np.asarray(compact_index.index_values)
    findings = {}
    for slot, offset in enumerate((17, 2**40 + 17)):
        saved = receipt.saved_cell_history[slot]
        raw = MeshingFieldStateRole(
            f"raw-counter-{slot}",
            "temperature",
            "rng-counter",
            saved.shape,
            np.dtype(saved.dtype),
            1,
            index_binding=source_index,
            random_stream=stream,
        )
        compact = replace(
            raw, entry_name=f"counter-{slot}", shape=ids.shape, index_binding=compact_index
        )
        rebound, moved = rebind_meshing_field_history(
            raw, saved, scenario.target, receipt, history_index=slot, proof=scenario.proof
        )
        recovered = recover_meshing_field_history(
            rebound,
            moved,
            compact,
            scenario.target,
            receipt,
            history_index=slot,
            proof=scenario.proof,
        )
        expected = ids.astype(saved.dtype) + np.asarray(offset, dtype=saved.dtype)
        expected_keys = stream.entity_keys(root, ids, expected)
        recovered_keys = stream.entity_keys(root, ids, np.asarray(recovered))
        sample = jax.vmap(lambda key: jax.random.bits(key, (8,)))
        foreign = replace(compact, random_stream=replace(stream, purpose="foreign"))
        changed = replace(
            receipt,
            cell_history=tuple(
                value.at[0, 0].add(1) if index == slot else value
                for index, value in enumerate(receipt.cell_history)
            ),
        )
        findings[str(saved.dtype)] = {
            "dtype_preserved": recovered.dtype == saved.dtype == moved.dtype,
            "counter_exact": bool(np.array_equal(recovered, expected)),
            "next_draws_exact": bool(
                np.array_equal(sample(expected_keys), sample(recovered_keys))
            ),
            "foreign_stream_refused": _refusal(
                lambda: recover_meshing_field_history(
                    rebound,
                    moved,
                    foreign,
                    scenario.target,
                    receipt,
                    history_index=slot,
                    proof=scenario.proof,
                )
            ),
            "counter_edit_refused": _refusal(lambda: SimplexRestartRepackProof(changed)),
        }
    return findings


def _euler_history_proof(scenario: _HistoryScenario) -> dict[str, bool]:
    from tools.meshing_qualification.scenarios._distributed_euler import state_units

    receipt = scenario.receipt
    saved = receipt.saved_cell_history[2]
    source_index = prepare_meshing_field_index_binding(scenario.source, "epoch/cell_ids")
    raw = MeshingFieldStateRole(
        "allocated-euler-conserved", "temperature", "stage-history",
        saved.shape, np.dtype(np.float64), 1,
        value_units=state_units(), index_binding=source_index,
    )
    compact_index = prepare_meshing_field_index_binding(scenario.target, "mesh/cell_ids")
    ids = np.asarray(compact_index.index_values)
    compact = replace(
        raw, entry_name="euler-conserved", shape=(ids.size, 4), index_binding=compact_index
    )
    rebound, moved = rebind_meshing_field_history(
        raw, saved, scenario.target, receipt, history_index=2, proof=scenario.proof
    )
    recovered = recover_meshing_field_history(
        rebound, moved, compact, scenario.target, receipt,
        history_index=2, proof=scenario.proof,
    )
    corrupted = replace(
        receipt, cell_history=(*receipt.cell_history[:2], moved.at[0, 0, 1].add(0.25))
    )
    return {
        "all_components_exact": bool(np.array_equal(recovered, _euler_state(ids))),
        "units_preserved": rebound.value_units == compact.value_units == state_units(),
        "momentum_edit_refused": bool(_refusal(lambda: SimplexRestartRepackProof(corrupted))),
    }


def _failure_evidence_proof(role: MeshingFieldStateRole) -> dict[str, Any]:
    from tools.meshing_qualification.scenarios.distributed import _distributed_field_difference

    ids = np.asarray([91, 7, 43], dtype=np.int64)
    owners = np.asarray([2, 1, 0], dtype=np.int32)
    exact = np.asarray([2**64 - 1, 2**63 + 1, 0], dtype=np.uint64)
    changed = np.asarray([0, 2**63, 0], dtype=np.uint64)
    integer = _distributed_field_difference(
        role, changed, exact, ids, owners,
        comparison="accepted-compact-transport", tolerance=0.0,
    )
    expected = np.asarray([1.0, 2.0, 3.0])
    inherited = expected + np.asarray([2e-12, 0.0, 0.0])
    transport = _distributed_field_difference(
        role, inherited, inherited, ids, owners,
        comparison="accepted-compact-transport", tolerance=0.0,
    )
    physical = _distributed_field_difference(
        role, inherited, expected, ids, owners,
        comparison="independent-physical-reference", tolerance=1e-12,
    )
    nonfinite = _distributed_field_difference(
        role, np.asarray([np.nan, 2.0, 3.0]), expected, ids, owners,
        comparison="independent-physical-reference", tolerance=1e-12,
    )
    return {"integer": integer, "transport": transport, "physical": physical,
            "nonfinite": nonfinite}


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
    assert isinstance(raw_index, MeshingFieldIndexBinding)
    assert isinstance(compact_index, MeshingFieldIndexBinding)
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
        "rng": _random_history_proof(scenario),
        "euler": _euler_history_proof(scenario),
        "failure_evidence": _failure_evidence_proof(scenario.compact_role),
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


def historical_owner_rearchive(path: str) -> dict[str, Any]:
    """Exercise provenance admission and archive identity with real accepted owners."""
    from phydrax.lifecycle._meshing_sources import (
        _validate_meshing_accepted_roles,
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    scenario = _prepare_history()
    owners = (scenario.source, scenario.foreign_source)
    accepted: dict[str, Any] = {
        "fields": {"temperature": scenario.source_values},
        "adaptation_results": (scenario.result,),
        "historical_owner_results": owners,
    }
    records: dict[str, Any] = {
        "collective_target": scenario.target,
        "field_declarations": {},
        "accepted_data": accepted,
    }
    for placement in (owners, owners[::-1], ()):
        accepted["historical_owner_results"] = placement
        _validate_meshing_accepted_roles(records)
    refusals = {}
    for name, invalid in (
        ("duplicate", (owners[0], owners[0])),
        ("foreign_epoch", (scenario.target,)),
        ("non_owner", ("owner",)),
    ):
        accepted["historical_owner_results"] = invalid
        refusals[name] = bool(_refusal(lambda: _validate_meshing_accepted_roles(records)))
    # Refusal must not poison a retry of the original accepted provenance.
    accepted["historical_owner_results"] = owners
    _validate_meshing_accepted_roles(records)
    payload = (scenario.result, owners)
    archived = write_meshing_source_closure(path, payload)
    restored = read_meshing_source_closure(path, expected_content_id=archived.content_id)
    rearchived = write_meshing_source_closure(path + "-rearchived", restored)
    return {
        "refusals": refusals,
        "content_id_equal": archived.content_id == rearchived.content_id,
        "current_id_equal": restored[0].target.result_id == scenario.target.result_id,
        "historical_ids_equal": tuple(owner.result_id for owner in restored[1])
        == tuple(owner.result_id for owner in owners),
    }


def save_live_publication_history(path: str) -> dict[str, Any]:
    """Archive real two-device history; deletion uses a separate full publication."""
    from dataclasses import fields
    from pathlib import Path

    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.lifecycle._meshing_sources import write_meshing_source_closure
    from phydrax.meshing import (
        _device_adaptation as device_owner,
        _result as result_owner,
    )

    original_builder = result_owner._build_collective_mesh_evidence
    original_run = device_owner._run_partitioned_adaptive_simplex_commit
    issued: list[Any] = []
    deleted: list[bool] = []

    def observe_builder(*args: Any, **kwargs: Any) -> Any:
        view = original_builder(*args, **kwargs)
        issued.append(view)
        return view

    def observe_run(partitioned: Any, raw: Any, **kwargs: Any) -> Any:
        result = original_run(partitioned, raw, **kwargs)
        if not deleted:
            # A complete second production run earns its own raw, geometry,
            # ownership, field and storage proofs. Never delete a shared bank
            # from the history scenario or substitute a fabricated positive.
            copied = jax.tree_util.tree_map(jnp.copy, raw)
            disposable = original_run(partitioned, copied, **kwargs)
            view = issued[-1]
            target = disposable.local_results[0].target
            storage = target.mesh.storage
            assert storage is not None and view.evidence is target.collective_evidence
            actual = view.require_states(
                storage, view.evidence, scope=kwargs["scope"], budget=kwargs["budget"]
            )
            assert actual is view._states
            assert copied.counters is not raw.counters
            copied.counters.delete()
            with np.testing.assert_raises(ValueError):
                view.require_states(
                    storage, view.evidence, scope=kwargs["scope"], budget=kwargs["budget"]
                )
            assert not raw.counters.is_deleted()
            deleted.append(True)
        return result

    with (
        patch.object(result_owner, "_build_collective_mesh_evidence", side_effect=observe_builder),
        patch.object(device_owner, "_run_partitioned_adaptive_simplex_commit", side_effect=observe_run),
    ):
        scenario = _prepare_history()
    target = scenario.target
    evidence = target.collective_evidence
    assert isinstance(evidence, CollectiveMeshEvidence)
    target.audit.require_passed()
    receipt = write_meshing_source_closure(Path(path), target, limits=ArrayArchiveLimits())
    for view in issued:
        with np.testing.assert_raises(ValueError):
            view.require_states(
                target.mesh.storage, evidence, scope=view._scope, budget=view._budget
            )
    return {
        "archive_content_id": receipt.content_id,
        "result_id": target.result_id,
        "evidence_id": evidence.evidence_id,
        "stored_content_id": evidence.stored_content_id,
        "persistent_fields": [member.name for member in fields(evidence)],
        "deleted_bank_refused_after_full_disposable_publication": deleted == [True],
        "expired_view_refused": True,
        "issued_count": len(issued),
    }


def replay_live_publication_history(path: str, content_id: str) -> dict[str, Any]:
    """Fresh interpreter: archive supplies fields, never live producer authority."""
    from dataclasses import fields, is_dataclass
    from pathlib import Path

    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle._meshing_sources import read_meshing_source_closure
    from phydrax.meshing import (
        _device_adaptation as device_owner,
        _distribution as distribution_owner,
        _result as result_owner,
    )

    target = read_meshing_source_closure(
        Path(path), expected_content_id=content_id, limits=ArrayArchiveLimits()
    )
    evidence = target.collective_evidence
    storage = target.mesh.storage
    assert isinstance(evidence, CollectiveMeshEvidence) and storage is not None
    original_replay = distribution_owner.replay_logical_bisection_checks
    original_raw = result_owner._prove_collective_raw_source
    original_physical = result_owner._prepare_collective_publication
    logical_inputs: list[tuple[Any, Any, Any]] = []
    raw_inputs: list[tuple[Any, Any]] = []
    physical: list[Any] = []
    requests: list[tuple[NativeExecutionScope, CoordinateEnclosureBudget]] = []

    def observe_replay(raw: Any, initial: Any, exterior: Any, **kwargs: Any) -> Any:
        value = original_replay(raw, initial, exterior, **kwargs)
        logical_inputs.append((raw, initial, exterior))
        return value

    def observe_raw(retained: Any, raw: Any, **kwargs: Any) -> Any:
        value = original_raw(retained, raw, **kwargs)
        raw_inputs.append((retained, raw))
        scope, budget = kwargs["scope"], kwargs["budget"]
        assert isinstance(scope, NativeExecutionScope)
        assert isinstance(budget, CoordinateEnclosureBudget)
        requests.append((scope, budget))
        return value

    def observe_physical(*args: Any, **kwargs: Any) -> Any:
        value = original_physical(*args, **kwargs)
        physical.append(args[2].states)
        return value

    with (
        patch.object(distribution_owner, "replay_logical_bisection_checks", side_effect=observe_replay),
        patch.object(result_owner, "_prove_collective_raw_source", side_effect=observe_raw),
        patch.object(result_owner, "_prepare_collective_publication", side_effect=observe_physical),
    ):
        layout, restored, exterior = device_owner.restore_partitioned_mesh_state(storage, evidence)
        assert len(logical_inputs) == 1
        assert logical_inputs[-1][0] is evidence.compiled_states
        assert logical_inputs[-1][1] is evidence.initial_states
        assert logical_inputs[-1][2] is evidence.source_exterior
        _, retained, _ = device_owner.retained_partition_states(target)
        assert len(logical_inputs) == 2
        assert tree_content_equal(restored, retained)
        device_owner.validate_partitioned_mesh_evidence(storage, evidence)
    assert layout.signature_id == evidence.layout.signature_id
    assert exterior is dict(storage.logical_arrays)["epoch/source_exterior"]
    assert sum(
        owner is evidence and raw is evidence.compiled_states for owner, raw in raw_inputs
    ) == 1
    assert physical
    epoch = dict(storage.logical_arrays)
    for state in (restored, retained):
        for owner in (state.mesh, state):
            for member in fields(owner):
                value = getattr(owner, member.name)
                if isinstance(value, jax.Array):
                    actual = np.asarray(value)
                    expected = np.asarray(epoch[f"epoch/{member.name}"])
                    assert actual.dtype == expected.dtype and actual.shape == expected.shape
                    assert actual.tobytes() == expected.tobytes()

    visited: set[int] = set()
    pending = [target]
    while pending:
        value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        assert not isinstance(value, result_owner._LiveCollectivePublication)
        if is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, member.name) for member in fields(value))
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
    for owner in (result_owner, device_owner):
        assert not any(
            isinstance(value, result_owner._LiveCollectivePublication)
            for value in vars(owner).values()
        )
    constructor = vars(result_owner)["_LiveCollectivePublication"]
    with np.testing.assert_raises(TypeError):
        constructor(evidence=evidence)
    assert requests
    scope, budget = requests[-1]
    with np.testing.assert_raises(ValueError):
        result_owner._LiveCollectivePublication.require_states(
            object.__new__(result_owner._LiveCollectivePublication),
            storage, evidence, scope=scope, budget=budget,
        )
    return {
        "result_id": target.result_id,
        "evidence_id": evidence.evidence_id,
        "stored_content_id": evidence.stored_content_id,
        "persistent_fields": [member.name for member in fields(evidence)],
        "public_restore_raw_replays": 1,
        "retained_source_raw_replays": 1,
        "cold_full_raw_reproof": 1,
        "cold_physical_publications": len(physical),
        "all_restored_state_fields_exact": tree_content_equal(restored, retained),
        "no_restored_or_global_live_view": True,
        "archive_cannot_issue_view": True,
    }


def projection_ring_matrix() -> dict[str, Any]:
    """Four historical identity groups routed under two/four current owners."""
    from jax.sharding import NamedSharding, PartitionSpec

    from phydrax.lifecycle._meshing_field_records import _ring_recovery

    results = {}
    historical = np.asarray([101, 509, 172, 622, 243, 735, 314, 848], np.int64)
    for count, permuted, empty in (
        (2, False, False), (4, False, False),
        (4, True, False), (4, True, True),
    ):
        devices = tuple(jax.devices())[:count]
        if permuted:
            devices = tuple(devices[index] for index in (2, 0, 3, 1))
        parts = AdaptiveSimplexParts(
            devices,
            neighbor_pairs=tuple(
                (left, right) for left in range(count) for right in range(count)
                if left != right
            ),
        )
        placement = NamedSharding(parts.mesh, PartitionSpec(parts.axis_name))
        ids = np.full((count, 8), -1, np.int64)
        for position, identity in enumerate(historical):
            owner = position % (count - 1 if empty else count)
            slot = int(np.sum(ids[owner] >= 0))
            ids[owner, slot] = identity
        owners = np.broadcast_to(np.arange(count, dtype=np.int32)[:, None], ids.shape)
        queries = np.tile(np.concatenate((historical[::-1], [-1])), (count, 1))
        if empty:
            queries[-1] = -1
        for dtype in (np.float64, np.complex128, np.uint32, np.uint64, np.bool_):
            values = (
                (ids % 2 == 0) if dtype is np.bool_
                else ids.astype(dtype) + np.asarray(17, dtype=dtype)
            )
            answer, found = _ring_recovery(
                parts,
                jax.device_put(ids, placement),
                jax.device_put(owners, placement),
                jax.device_put(values, placement),
                jax.device_put(queries, placement),
            )
            live = queries >= 0
            np.testing.assert_array_equal(np.asarray(found), live.astype(np.int32))
            expected = (
                queries % 2 == 0 if dtype is np.bool_
                else queries.astype(dtype) + np.asarray(17, dtype=dtype)
            )
            np.testing.assert_array_equal(np.asarray(answer)[live], expected[live])
        for fault in ("missing", "duplicate", "foreign-owner"):
            bad_ids, bad_owners = ids.copy(), owners.copy()
            location = np.argwhere(ids == historical[0])[0]
            if fault == "missing":
                bad_ids[tuple(location)] = -1
            elif fault == "duplicate":
                bad_ids[1, -1] = historical[0]
            else:
                bad_owners[tuple(location)] = (int(location[0]) + 1) % count
            _, found = _ring_recovery(
                parts,
                jax.device_put(bad_ids, placement),
                jax.device_put(bad_owners, placement),
                jax.device_put(ids.astype(np.float64), placement),
                jax.device_put(queries, placement),
            )
            expected_count = 2 if fault == "duplicate" else 0
            np.testing.assert_array_equal(
                np.asarray(found)[queries == historical[0]], expected_count
            )
        results[f"4-to-{count}-permuted-{permuted}-empty-{empty}"] = True
    return results


def native_projection_collective() -> dict[str, Any]:
    """Real current mesh ownership supplies active raw rows and agrees refusal."""
    from phydrax.lifecycle._meshing_field_records import (
        _owner_local_projection,
        project_meshing_field_history,
    )
    from phydrax.meshing._failures import MeshingFailure

    try:
        source = _prepare_source("halo_depth", {}).source
        partition = phx.meshing.MeshPartitionPolicy(
            phx.meshing.MeshPartitionKind.MORTON, jax.process_count(), halo_width=1
        )
        distribution = phx.meshing.prepare_mesh_distribution(
            phx.meshing.MeshPart("domain", source), policy=partition
        )
        policy = phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
            device_policy=phx.discretization.AdaptiveSimplexPolicy(
                vertex_capacity=64, cell_capacity=64
            ),
            distribution=distribution,
            partition_policy=partition,
        )
        prepared = phx.meshing.partition_adaptive_simplex(
            phx.meshing.prepare_adaptive_simplex(source, policy=policy)
        )
        (result,) = phx.meshing.commit_partitioned_adaptive_simplex(
            prepared, prepared.states
        ).local_results
        target = result.target
        evidence = target.collective_evidence
        assert evidence is not None
        checked = []
        for association in ("vertex", "cell"):
            compact = prepare_meshing_field_index_binding(
                target, "mesh/vertex_ids" if association == "vertex" else "mesh/cell_ids"
            )
            raw = prepare_meshing_field_index_binding(
                target, "epoch/vertex_ids" if association == "vertex" else "epoch/cell_ids"
            )
            active = dict(evidence.logical_arrays)[f"epoch/{association}_active"]
            assert isinstance(raw.index_values, jax.Array)
            for dtype in (np.float64, np.uint32, np.uint64, np.bool_):
                ids = np.asarray(compact.index_values)
                values = (
                    ids % 2 == 0 if dtype is np.bool_
                    else ids.astype(dtype) + np.asarray(17, dtype=dtype)
                )
                baseline = jnp.zeros(raw.index_values.shape, dtype=dtype, device=raw.index_values.sharding)
                compact_role = MeshingFieldStateRole(
                    "compact", "temperature", "stage-history", values.shape,
                    np.dtype(dtype), 1, index_binding=compact,
                )
                raw_role = MeshingFieldStateRole(
                    "raw", "temperature", "stage-history", baseline.shape,
                    np.dtype(dtype), 1, index_binding=raw,
                )
                projected = project_meshing_field_history(
                    compact_role, values, raw_role, baseline, target
                )
                assert isinstance(projected, jax.Array)
                local_ids = np.asarray(raw.index_values.addressable_shards[0].data)
                live = np.asarray(active.addressable_shards[0].data)
                expected = (
                    local_ids % 2 == 0 if dtype is np.bool_
                    else local_ids.astype(dtype) + np.asarray(17, dtype=dtype)
                )
                np.testing.assert_array_equal(
                    np.asarray(projected.addressable_shards[0].data),
                    np.where(live, expected, 0),
                )
                checked.append(f"{association}-{np.dtype(dtype)}")
            # One rank changes a scientific identity, then every rank must
            # refuse before the next ring. A subsequent valid call must work.
            changed = np.asarray(compact.index_values).copy()
            if jax.process_index() == 0:
                changed[0] = 999999
            bad = replace(compact, index_values=changed)
            try:
                _owner_local_projection(target, bad, raw, values, baseline, active)
            except MeshingFailure:
                pass
            else:
                raise AssertionError("One-rank foreign identity was not collectively refused")
            retry = _owner_local_projection(target, compact, raw, values, baseline, active)
            np.testing.assert_array_equal(
                np.asarray(retry.addressable_shards[0].data),
                np.asarray(projected.addressable_shards[0].data),
            )
        from phydrax.meshing._device_adaptation import prepare_partitioned_mesh_adaptation

        for operation in (
            phx.discretization.refine_adaptive_simplex_parts,
            phx.discretization.coarsen_adaptive_simplex_parts,
            phx.discretization.refine_adaptive_simplex_parts,
        ):
            assert result.distribution is not None
            policy = phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
                device_policy=policy.device_policy,
                distribution=result.distribution.target,
                partition_policy=partition,
            )
            prepared = prepare_partitioned_mesh_adaptation(
                phx.meshing.prepare_mesh_adaptation(
                    result.target, phx.meshing.MarkedMeshAdaptation(), policy=policy
                )
            )
            update = operation(
                prepared.layout, type(prepared).execution_parts(prepared),
                prepared.states, prepared.states.mesh.cell_active,
            )
            (result,) = phx.meshing.commit_partitioned_adaptive_simplex(
                prepared, update.state
            ).local_results
            target = result.target
            transition = result.transition
            assert transition is not None
            evidence = target.collective_evidence
            assert evidence is not None
            _assert_vertex_transition_aliases(transition)
            vertex_compact = prepare_meshing_field_index_binding(target, "mesh/vertex_ids")
            vertex_raw = prepare_meshing_field_index_binding(target, "epoch/vertex_ids")
            assert isinstance(vertex_raw.index_values, jax.Array)
            vertex_active = dict(evidence.logical_arrays)["epoch/vertex_active"]
            for dtype in (np.float64, np.complex128, np.uint32, np.uint64, np.bool_):
                vertex_values = np.asarray(vertex_compact.index_values).astype(dtype)
                vertex_baseline = jnp.ones(
                    vertex_raw.index_values.shape, dtype=dtype,
                    device=vertex_raw.index_values.sharding,
                )
                vertex_projected = _owner_local_projection(
                    target, vertex_compact, vertex_raw, vertex_values,
                    vertex_baseline, vertex_active, transition,
                )
                roles: tuple[MeshingFieldStateKind, ...] = (
                    "stage-history", "material-history", "rng-counter"
                )
                for role in roles:
                    if role == "rng-counter" and dtype not in (np.uint32, np.uint64):
                        continue
                    stream = (
                        MeshingRandomStream(
                            "root-key", "threefry2x32", "meshing-test", "vertex-probe",
                            ("temperature",), "draw",
                        ) if role == "rng-counter" else None
                    )
                    units = (phx.units.KELVIN,) if role == "material-history" else None
                    compact_role = MeshingFieldStateRole(
                        "compact", "temperature", role, vertex_values.shape,
                        np.dtype(dtype), 1, index_binding=vertex_compact,
                        random_stream=stream, value_units=units,
                    )
                    raw_role = MeshingFieldStateRole(
                        "raw", "temperature", role, vertex_baseline.shape,
                        np.dtype(dtype), 1, index_binding=vertex_raw,
                        random_stream=stream, value_units=units,
                    )
                    role_result = project_meshing_field_history(
                        compact_role, vertex_values, raw_role, vertex_baseline,
                        target, transition=transition,
                    )
                    assert isinstance(role_result, jax.Array)
                    np.testing.assert_array_equal(
                        np.asarray(role_result.addressable_shards[0].data),
                        np.asarray(vertex_projected.addressable_shards[0].data),
                    )
            # Presence in a target halo is not permission to discard a row when
            # its authoritative owner is missing.
            bad_target = target
            if jax.process_index() == 0:
                assert target.mesh.storage is not None
                bad_target = eqx.tree_at(
                    lambda value: value.mesh.storage.entity_owned[0],
                    target, jnp.zeros_like(target.mesh.storage.entity_owned[0]),
                )
            try:
                _owner_local_projection(
                    bad_target, vertex_compact, vertex_raw, vertex_values,
                    vertex_baseline, vertex_active,
                )
            except MeshingFailure:
                pass
            else:
                raise AssertionError("Missing vertex owner was classified as retirement")
            _owner_local_projection(
                target, vertex_compact, vertex_raw, vertex_values,
                vertex_baseline, vertex_active, transition,
            )
            compact = prepare_meshing_field_index_binding(target, "mesh/cell_ids")
            raw = prepare_meshing_field_index_binding(target, "epoch/cell_ids")
            assert isinstance(raw.index_values, jax.Array)
            active = dict(evidence.logical_arrays)["epoch/cell_active"]
            local_ids = np.asarray(raw.index_values.addressable_shards[0].data)
            live = np.asarray(active.addressable_shards[0].data)
            mapped = local_ids.copy()
            if transition.coarsened_cell_ids is not None:
                for child, parent in zip(
                    np.asarray(transition.coarsened_cell_ids),
                    np.asarray(transition.coarsened_into_ids), strict=True,
                ):
                    mapped[local_ids == child] = parent
            for dtype in (np.float64, np.complex128, np.uint32, np.uint64, np.bool_):
                ids = np.asarray(compact.index_values)
                values = (ids % 2 == 0) if dtype is np.bool_ else ids.astype(dtype) + 17
                baseline = jnp.ones(
                    raw.index_values.shape, dtype=dtype, device=raw.index_values.sharding
                )
                projected = _owner_local_projection(
                    target, compact, raw, values, baseline, active, transition
                )
                expected = (mapped % 2 == 0) if dtype is np.bool_ else mapped.astype(dtype) + 17
                np.testing.assert_array_equal(
                    np.asarray(projected.addressable_shards[0].data),
                    np.where(live, expected, 1),
                )
                checked.append(f"{operation.__name__}-{np.dtype(dtype)}")
                history_roles: tuple[MeshingFieldStateKind, ...] = (
                    "stage-history", "material-history", "rng-counter"
                )
                for role in history_roles:
                    if role == "rng-counter" and dtype not in (np.uint32, np.uint64):
                        continue
                    stream = (
                        MeshingRandomStream(
                            "root-key", "threefry2x32", "meshing-test", "cell-probe",
                            ("temperature",), "draw",
                        )
                        if role == "rng-counter" else None
                    )
                    compact_role = MeshingFieldStateRole(
                        "compact", "temperature", role, values.shape, np.dtype(dtype),
                        1, index_binding=compact, random_stream=stream,
                        value_units=(phx.units.KELVIN,) if role == "material-history" else None,
                    )
                    raw_role = MeshingFieldStateRole(
                        "raw", "temperature", role, baseline.shape, np.dtype(dtype),
                        1, index_binding=raw, random_stream=stream,
                        value_units=(phx.units.KELVIN,) if role == "material-history" else None,
                    )
                    role_result = project_meshing_field_history(
                        compact_role, values, raw_role, baseline, target,
                        transition=transition,
                    )
                    assert isinstance(role_result, jax.Array)
                    np.testing.assert_array_equal(
                        np.asarray(role_result.addressable_shards[0].data),
                        np.asarray(projected.addressable_shards[0].data),
                    )
            # Authenticate stored lineage on every rank before entering the ring.
            bad_transition = transition
            if jax.process_index() == 0:
                bad_transition = copy(transition)
                object.__setattr__(
                    bad_transition, "transition_id", "foreign-history-transition"
                )
            try:
                _owner_local_projection(
                    target, compact, raw, values, baseline, active, bad_transition
                )
            except MeshingFailure:
                pass
            else:
                raise AssertionError("Foreign transition was not collectively refused")
            retry = _owner_local_projection(
                target, compact, raw, values, baseline, active, transition
            )
            np.testing.assert_array_equal(
                np.asarray(retry.addressable_shards[0].data),
                np.asarray(projected.addressable_shards[0].data),
            )
        return {"checked": checked, "refusal_and_retry": True}
    finally:
        jax.distributed.shutdown()


def production_projection_prefix(root: str) -> dict[str, Any]:
    """Exercise the real producer entry only through coarsened field projection."""
    import json
    import os

    from phydrax.lifecycle import _meshing_field_records as history
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios import distributed as scenario

    class PrefixComplete(BaseException):
        pass

    args = QualificationControls(
        scenario="native-distributed-lifecycle", resolution=8, capacity=20000,
        timeout=600.0, maximum_work_units=32000000,
        maximum_memory_bytes=34359738368,
        distributed_parts=jax.process_count(), distributed_processes=jax.process_count(),
        restart_parts=jax.process_count() // 2,
        distributed_bisection_compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
        distributed_checkpoint_root=root,
    )
    os.environ[scenario._DISTRIBUTED_ROLE] = "producer"
    os.environ[scenario._DISTRIBUTED_REQUEST] = json.dumps(
        scenario._distributed_qualification_request(args)
    )
    observations: dict[str, Any] = {"retired_vertex_rows": 0}
    vertex_membership: np.ndarray | None = None
    observing_vertex = False
    projection = scenario._distributed_qualification_project
    owner_projection = history._owner_local_projection
    ring = history._ring_recovery

    def observe_ring(parts: Any, ids: Any, owners: Any, values: Any, queries: Any) -> Any:
        nonlocal vertex_membership
        answer, found = ring(parts, ids, owners, values, queries)
        local_queries = np.asarray(queries.addressable_shards[0].data)
        local_found = np.asarray(found.addressable_shards[0].data)
        if observing_vertex and vertex_membership is None:
            vertex_membership = local_found > 0
        bad = (local_queries >= 0) & (local_found != 1)
        observations["unmatched"] = local_queries[bad].tolist()
        observations["matches"] = local_found[bad].tolist()
        return answer, found

    def observe_owner(
        carrier: Any, compact: Any, raw: Any, values: Any, baseline: Any,
        active: Any, transition: Any = None,
    ) -> Any:
        nonlocal vertex_membership, observing_vertex
        vertex_membership = None
        observing_vertex = compact.bank_name == "mesh/vertex_ids"
        observations["bank"] = compact.bank_name
        observations["transition"] = None if transition is None else transition.transition_id
        observations["deleted"] = (
            [] if transition is None else np.asarray(
                transition.lineage.entity_lineage(
                    0 if compact.bank_name == "mesh/vertex_ids"
                    else carrier.mesh.topological_dimension
                ).deleted_source_ids
            ).tolist()
        )
        result = owner_projection(carrier, compact, raw, values, baseline, active, transition)
        if observing_vertex:
            assert vertex_membership is not None
            retired = (
                np.asarray(active.addressable_shards[0].data) & ~vertex_membership
            )
            np.testing.assert_array_equal(
                np.asarray(result.addressable_shards[0].data)[retired],
                np.asarray(baseline.addressable_shards[0].data)[retired],
            )
            observations["retired_vertex_rows"] += int(np.count_nonzero(retired))
        observing_vertex = False
        return result

    def prefix(*arguments: Any, **keywords: Any) -> Any:
        result = projection(*arguments, **keywords)
        transition = keywords.get("transition")
        if transition is not None and transition.transition_kind == "coarsen":
            raise PrefixComplete
        return result

    try:
        with (
            patch.object(scenario, "_distributed_qualification_project", prefix),
            patch.object(history, "_owner_local_projection", observe_owner),
            patch.object(history, "_ring_recovery", observe_ring),
        ):
            result = scenario._distributed_qualification_lifecycle(args)
        return {"passed": False, "failure": result.get("failure"), **observations}
    except PrefixComplete:
        return {"passed": True, **observations}
    finally:
        jax.distributed.shutdown()


def _assert_vertex_transition_aliases(transition: phx.meshing.CellMeshTransition) -> None:
    """Exercise canonical merged/deleted vertex provenance independently of cell IDs."""
    import pytest

    from phydrax.lifecycle._meshing_field_records import _projection_identity_rows

    target = transition.target
    compact = prepare_meshing_field_index_binding(target, "mesh/vertex_ids")
    vertices = transition.lineage.entity_lineage(0)
    deleted = np.asarray(vertices.deleted_source_ids)
    if deleted.size == 0:
        return
    ids = np.asarray(compact.index_values)
    # Both merge kinds may share a target; no ordinal relation to source IDs.
    destinations = np.full(deleted.shape, ids[-1], dtype=np.int64)

    def accepted(
        children: np.ndarray, parents: np.ndarray, target_set: str
    ) -> phx.meshing.CellMeshTransition:
        merged = phx.meshing.EntityLineage(
            0, vertices.source_entity_set_id, target_set,
            np.concatenate((np.asarray(vertices.source_global_ids), children)),
            np.concatenate((np.asarray(vertices.target_global_ids), parents)),
            np.concatenate((
                np.asarray(vertices.relation_kinds),
                np.where(
                    np.arange(children.size) % 2,
                    int(phx.meshing.EntityLineageKind.COLLAPSED_INTO),
                    int(phx.meshing.EntityLineageKind.MERGED_INTO),
                ),
            )),
            created_target_ids=vertices.created_target_ids,
            deleted_source_ids=np.setdiff1d(deleted, children),
        )
        lineage = phx.meshing.MeshLineage(
            transition.lineage.source_topology_id,
            transition.lineage.target_topology_id,
            (merged, *transition.lineage.entities[1:]),
        )
        return phx.meshing.CellMeshTransition(
            transition.source_mesh_id, transition.source_topology_id,
            target, lineage, phx.meshing.MeshTransitionKind.COARSEN,
        )

    merged = accepted(deleted, destinations, vertices.target_entity_set_id)
    joined, rows = _projection_identity_rows(target, compact, merged)
    np.testing.assert_array_equal(joined[:ids.size], ids)
    np.testing.assert_array_equal(joined[ids.size:], deleted)
    np.testing.assert_array_equal(ids[rows[ids.size:]], destinations)
    for children, parents, target_set in (
        (np.repeat(deleted[:1], 2), np.repeat(destinations[:1], 2), vertices.target_entity_set_id),
        (deleted, np.full_like(destinations, int(np.max(ids)) + 1), vertices.target_entity_set_id),
        (deleted, destinations, target.mesh.entity_set(target.mesh.topological_dimension).entity_set_id),
    ):
        with pytest.raises(ValueError, match="project_meshing_field_history.transition"):
            _projection_identity_rows(target, compact, accepted(children, parents, target_set))
