# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices


_MODULE = "tests._support.distributed_meshing."


# Fixed empirical allowances from test_native_distributed.py
# at a6827a9e5; no meshing acceptance policy is relaxed.
# Nodal prolongation, distributed load assembly and FV inventory on unit cells.
_DISTRIBUTED_TRANSFER_ATOL = 1e-12
# Distributed field solution against the retained serial solve record.
_DISTRIBUTED_SOLVE_ATOL = 1e-9


def _run_packed_scenario(
    scenario: str, semantic_owners_flag: str, archive_path: Path
) -> dict[str, Any]:
    # Each device-topology phase owns its deadline; the archive remains alive
    # across publication, cold replay, and cold continuation.
    record = run_on_forced_devices(
        _MODULE + "native_distributed_packed",
        f"native_{scenario}",
        (semantic_owners_flag, str(archive_path)),
        devices=2,
        env={},
    )
    packed_support = record["compact_packed_support"]
    assert packed_support["source_width"] >= (24 if scenario.endswith("tetra") else 6)
    assert packed_support["complete_parent_multisets"]
    assert packed_support["canonical_prefix"]
    restore = record["packed_restore"]
    record["packed_cold_replay"] = run_on_forced_devices(
        _MODULE + "native_distributed_cold",
        "packed_cold_replay",
        (
            str(archive_path),
            restore["content_id"],
            restore["policy_id"],
            restore["source_result_id"],
            restore["expected_cells"],
        ),
        devices=1,
        env={},
    )
    record["packed_cold_continuation"] = run_on_forced_devices(
        _MODULE + "native_distributed_cold",
        "packed_cold_continuation",
        (str(archive_path), restore["content_id"]),
        devices=2,
        env={},
    )
    return record


def _record_array(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray(payload["values"], dtype=np.dtype(payload["dtype"])).reshape(
        payload["shape"]
    )


def _assert_packed_cold_replay(record: dict[str, Any]) -> None:
    assert all(record["packed_cold_replay_13"]), "policy.policy_id == policy_id"
    assert all(record["packed_cold_replay_16"]), (
        "proof.uniform_refinement.source.result_id == source_result_id"
    )
    assert all(record["packed_cold_replay_17"]), (
        "proof.global_entity_counts[-1] == int(expected_cells)"
    )
    assert all(record["packed_cold_replay_18"]), (
        "not np.array_equal(dict(proof.logical_arrays)['epoch/cell_ids'], proof.compiled_states.mesh.cell_ids)"
    )
    assert all(record["packed_cold_replay_41"]), (
        "np.all(np.asarray(repack.status) & int(phx.discretization.AdaptiveSimplexStatus.CAPACITY_EXCEEDED))"
    )
    assert all(record["packed_cold_replay_refusal_49"]), "ValueError"
    for actual, expected in record["packed_cold_replay_57"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))


def _assert_packed_cold_continuation(record: dict[str, Any]) -> None:
    assert all(record["packed_cold_continuation_30"]), (
        "continued.status is M.MeshAdaptationStatus.COMPLETE"
    )
    assert all(record["packed_cold_continuation_33"]), (
        "isinstance(storage, CellMeshStorage)"
    )
    assert all(record["packed_cold_continuation_34"]), (
        "continued.target.geometry.source_coordinates() == original.geometry.source_coordinates(storage.coordinate_global_ids)"
    )
    assert all(record["packed_cold_continuation_37"]), (
        "target.collective_evidence.uniform_refinement.source is original"
    )
    assert all(record["packed_cold_continuation_38"]), (
        "original.geometry.source_coordinates() == original_coefficients"
    )


def _assert_distributed(record: dict[str, Any]) -> None:
    _assert_distributed_refinement(record)
    _assert_distributed_restriction_and_refusal(record)
    _assert_distributed_transfer_and_continuation(record)


def _assert_distributed_refinement(record: dict[str, Any]) -> None:
    assert all(record["distributed_126"]), (
        "lineage is not None and lineage.child_ids.shape[1] == (24 if scenario.endswith('tetra') else 6)"
    )
    assert all(record["distributed_127"]), "lineage.source is source"
    assert all(record["distributed_132"]), "np.all(owners >= 0)"
    assert all(record["distributed_172"]), "protected_position.size == 1"
    for actual, expected in record["distributed_174"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_178"]), (
        "any((result.hierarchy.uniform_refinement is not None for result in protected_results))"
    )
    for actual, expected in record["distributed_180"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_209"]), "binary.size > 0"
    for actual, expected in record["distributed_216"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_217"]), (
        "np.any(np.asarray(repack.states.mesh.cell_active) & (np.asarray(repack.solver_cell_owners) != np.arange(2)[:, None]))"
    )
    assert all(record["distributed_246"]), (
        "np.any(np.asarray(partitioned.states.mesh.cell_active) & (np.asarray(partitioned.solver_cell_owners) != np.arange(2)[:, None]))"
    )
    assert all(record["distributed_262"]), (
        "all((retained is submitted for retained, submitted in zip(jax.tree_util.tree_leaves(evidence.compiled_states), jax.tree_util.tree_leaves(compiled), strict=True)))"
    )
    assert all(record["distributed_270"]), (
        "evidence.uniform_refinement.lineage_id == lineage.lineage_id"
    )
    assert all(record["distributed_271"]), (
        "all((retained is submitted for retained, submitted in zip(jax.tree_util.tree_leaves(evidence.uniform_refinement), jax.tree_util.tree_leaves(lineage), strict=True)))"
    )
    for actual, expected in record["distributed_280"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_refusal_286"]), "(ValueError, M.MeshingFailure)"
    for actual, expected in record["distributed_293"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_297"]), (
        "evidence.global_entity_counts[-1] == expected_cells"
    )
    assert all(record["distributed_308"]), "len(pending) == siblings.size"
    assert all(record["distributed_312"]), "(owner, slot) not in visited"
    assert all(record["distributed_318"]), "np.all(children >= 0)"
    assert all(record["distributed_320"]), "support and np.all(np.asarray(support) >= 0)"
    assert all(record["distributed_321"]), (
        "published_owners[published[0]] == np.min(support)"
    )
    for actual, expected in record["distributed_323"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["distributed_325"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["distributed_326"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_327"]), (
        "evidence.coordinate_geometry_id == cell_geometry_id(source.geometry)"
    )
    assert all(record["distributed_328"]), (
        "not np.array_equal(banks['epoch/cell_ids'], compiled.mesh.cell_ids)"
    )
    assert all(record["distributed_329"]), (
        "any((result.hierarchy.uniform_refinement is not None for result in results)) == partial"
    )
    assert all(record["distributed_347"]), (
        "serial_elements[block.name] == element.element_id"
    )
    for actual, expected in record["distributed_355"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["distributed_356"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["distributed_358"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_359"]), (
        "tuple((zone.name for zone in result.target.zones)) == tuple((zone.name for zone in source_zones))"
    )
    for actual, expected in record["distributed_365"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_367"]), "isinstance(storage, CellMeshStorage)"
    assert all(record["distributed_368"]), (
        "result.target.geometry.source_coordinates() == source.geometry.source_coordinates(storage.coordinate_global_ids)"
    )
    assert all(record["distributed_371"]), "lineage.source is source"
    assert all(record["distributed_372"]), (
        "source.geometry.source_coordinates() == original_coefficients"
    )
    assert all(record["distributed_375"]), (
        "element.element_id == serial_elements[block.name]"
    )
    for actual, expected in record["distributed_378"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_382"]), "result.target.execution_evidence is not None"
    assert all(record["distributed_383"]), (
        "int(result.target.execution_evidence.externally_charged_work) > 0"
    )
    assert all(record["distributed_refusal_389"]), "ValueError"
    assert all(record["distributed_397"]), (
        "(limits.max_members, limits.max_manifest_nesting, limits.max_array_rank) == (257, 16, 8)"
    )


def _assert_distributed_restriction_and_refusal(record: dict[str, Any]) -> None:
    assert all(record["distributed_537"]), "not np.all(np.asarray(checks[:, 2]))"
    assert all(record["distributed_546"]), "np.all(np.asarray(changed.status_flags) == 0)"
    assert all(record["distributed_548"]), "not np.all(np.asarray(checks[:, 2]))"
    assert all(record["distributed_554"]), "not np.all(np.asarray(checks[:, 1]))"
    assert all(record["distributed_558"]), "not np.all(np.asarray(checks[:, 4]))"
    assert all(record["distributed_566"]), "len(results) == 2"
    assert all(record["distributed_569"]), "rounded.size"
    assert all(record["distributed_572"]), "isinstance(evidence, CollectiveMeshEvidence)"
    assert all(record["distributed_576"]), "np.array_equal(valid, issued)"
    assert all(record["distributed_577"]), "np.all(signs[valid] != 0)"
    assert all(record["distributed_579"]), "geometry.restriction_source is not None"
    assert all(record["distributed_581"]), (
        "isinstance(element, BarycentricCellGeometryElement)"
    )
    assert all(record["distributed_583"]), (
        "np.array_equal(np.round(2.0 * weights), 2.0 * weights)"
    )
    for actual, expected in record["distributed_584"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_586"]), (
        "all((np.any(np.all(points == row, axis=1)) for row in coefficients))"
    )
    assert all(record["distributed_587"]), (
        "not any((np.any(np.all(coefficients == row, axis=1)) for row in rounded))"
    )
    assert all(record["distributed_refusal_592"]), "M.MeshingFailure"
    assert all(record["distributed_595"]), (
        "error.category is M.MeshingFailureCategory.QUALITY_REJECTED"
    )
    assert all(record["distributed_596"]), (
        "error.stage == 'distributed-reference-restriction'"
    )
    assert all(record["distributed_refusal_602"]), "M.MeshingFailure"
    assert all(record["distributed_605"]), (
        "error.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED"
    )
    assert all(record["distributed_612"]), "seed.certification is None"
    assert all(record["distributed_617"]), (
        "embedding is not None and embedding.status == 'certified'"
    )
    assert all(record["distributed_619"]), (
        "embedding.validity_certificate_id == seed.audit.validity.certificate_id"
    )
    assert all(record["distributed_620"]), "proof.source_audit_id == seed.audit.report_id"
    assert all(record["distributed_621"]), (
        "proof.source_premise_id == partitioned.source_witness.scientific_certification_id"
    )
    assert all(record["distributed_622"]), (
        "proof.source_premise_id != seed.audit.report_id"
    )
    assert all(record["distributed_624"]), (
        "len({view.distribution_id for view in views}) == 1"
    )
    assert all(record["distributed_625"]), (
        "len({result.target.result_id for result in results.local_results}) == 1"
    )
    assert all(record["distributed_642"]), (
        "restored.identity.target_topology_id == seed.mesh.topology_id"
    )
    assert all(record["distributed_673"]), "overlap.collective_evidence is None"
    assert all(record["distributed_674"]), "overlap.certification is None"
    for actual, expected in record["distributed_675"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_refusal_679"]), "M.MeshingFailure"
    assert all(record["distributed_682"]), (
        "error.category is M.MeshingFailureCategory.LINEAGE_FAILED"
    )
    assert all(record["distributed_685"]), "source.result_id == source_result_id"
    for actual, expected in record["distributed_686"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_687"]), (
        "distribution.evidence.kind is M.MeshPartitionKind.MORTON"
    )


def _assert_distributed_transfer_and_continuation(record: dict[str, Any]) -> None:
    assert all(record["distributed_717"]), "value.mesh.cells.ndim == 2"
    assert all(record["distributed_725"]), "len(results) == 2"
    assert all(record["distributed_726"]), (
        "transfers and max(transfers) == partitioned.layout.cell_capacity"
    )
    for actual, expected in record["distributed_740"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["distributed_741"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["distributed_743"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_744"]), (
        "tuple((zone.name for zone in target.zones)) == ('cell:0', 'cell:1')"
    )
    for actual, expected in record["distributed_747"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_760"]), "storage is not None"
    assert all(record["distributed_761"]), (
        "target.mesh.topology_id == reference.target.mesh.topology_id"
    )
    assert all(record["distributed_762"]), (
        "target.mesh.geometry_id == reference.target.mesh.geometry_id"
    )
    assert all(record["distributed_763"]), "storage.global_entity_counts == (4, 5, 2)"
    assert all(record["distributed_764"]), (
        "target.collective_evidence.evidence_id == storage.evidence_id"
    )
    assert all(record["distributed_765"]), (
        "np.all(np.asarray(target.collective_evidence.global_checks))"
    )
    assert all(record["distributed_766"]), "result.distribution is not None"
    assert all(record["distributed_767"]), "not result.distribution.rebalanced"
    assert all(record["distributed_768"]), (
        "np.all(np.asarray(storage.local_neighborhood_complete))"
    )
    assert all(record["distributed_769"]), (
        "result.transfer.preserves_linear and result.transfer.preserves_constants"
    )
    assert all(record["distributed_770"]), "not result.transfer.conservative"
    for actual, expected in record["distributed_773"]:
        np.testing.assert_allclose(
            _record_array(actual),
            _record_array(expected),
            atol=_DISTRIBUTED_TRANSFER_ATOL,
        )
    for actual, expected in record["distributed_775"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_789"]), "receipt.array_digest == written.array_digest"
    for actual, expected in record["distributed_790"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["distributed_791"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["distributed_792"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["distributed_refusal_794"]), "ValueError"
    assert all(record["distributed_804"]), "len(cell_shards) == 2"
    assert all(record["distributed_808"]), "np.all(np.isin(held, shard.payload))"
    for actual, expected in record["distributed_841"]:
        np.testing.assert_allclose(
            _record_array(actual),
            _record_array(expected),
            atol=_DISTRIBUTED_TRANSFER_ATOL,
        )
    for actual, expected in record["distributed_842"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["distributed_845"]:
        np.testing.assert_allclose(
            _record_array(actual), _record_array(expected), atol=_DISTRIBUTED_SOLVE_ATOL
        )
    for actual, expected in record["distributed_895"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["distributed_896"]:
        np.testing.assert_allclose(
            _record_array(actual),
            _record_array(expected),
            atol=_DISTRIBUTED_TRANSFER_ATOL,
        )
    assert all(record["distributed_897"]), (
        "bool(advance.finite) and bool(advance.admissible)"
    )
    assert all(record["distributed_898"]), (
        "np.any(np.asarray(updated[0][dynamics[0].discretization.cell_owned, 0]) != np.asarray(initial[0][dynamics[0].discretization.cell_owned, 0]))"
    )
    assert all(record["distributed_913"]), (
        "not rejected.published and rejected.composition.composition_id == composition.composition_id"
    )
    assert all(record["distributed_915"]), "receipt.published"
    for actual, expected in record["distributed_916"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))


def _assert_halo_overflow(record: dict[str, Any]) -> None:
    assert record["distributed_698"], "the overflow route must emit its refusal evidence"
    assert all(record["distributed_698"]), (
        "np.all(flags & int(D.AdaptiveSimplexStatus.CAPACITY_EXCEEDED))"
    )


def _assert_halo_depth(record: dict[str, Any]) -> None:
    for key in (
        "distributed_700",
        "distributed_707",
        "distributed_708",
        "distributed_710",
        "distributed_711",
    ):
        assert record[key], f"the successful halo route must emit {key}"
    assert all(record["distributed_700"]), "not np.any(np.asarray(work.status))"
    for actual, expected in record["distributed_707"]:
        np.testing.assert_allclose(
            _record_array(actual), _record_array(expected), atol=0, rtol=0
        )
    for actual, expected in record["distributed_708"]:
        np.testing.assert_allclose(
            _record_array(actual), _record_array(expected), atol=0, rtol=0
        )
    assert all(record["distributed_710"]), "np.any(counts[1] > counts[0])"
    assert all(record["distributed_711"]), "np.all(counts[2] >= counts[1])"


def _assert_historical_routes(record: dict[str, Any]) -> None:
    assert all(record["historical_routes_7"]), "jax.device_count() == 1"
    for actual, expected in record["historical_routes_11"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["historical_routes_17"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))


def _assert_one_owner_halo(record: dict[str, Any]) -> None:
    for actual, expected in record["one_owner_halo_36"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["one_owner_halo_37"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["one_owner_halo_41"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["one_owner_halo_42"]), "halo.permutations == ()"


def _assert_migration(record: dict[str, Any]) -> None:
    for actual, expected in record["migration_59"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_60"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_61"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_62"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_63"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_64"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_65"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_66"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_67"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_68"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_69"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_70"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_71"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["migration_73"]), (
        "np.sum(np.asarray(target.cell_fields)[owned]) == 10.0"
    )
    for actual, expected in record["migration_74"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_75"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["migration_76"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["migration_77"]), "prepared.evidence.route_metadata_bytes == 4"
    assert all(record["migration_78"]), (
        "prepared.target_parts.neighbor_pairs == ((0, 1), (1, 0))"
    )
    for actual, expected in record["migration_80"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["migration_82"]), (
        "np.all(np.asarray(prepared.evidence.status) & int(flag))"
    )
    for actual, expected in record["migration_84"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))


def _assert_forest(record: dict[str, Any]) -> None:
    assert all(record["forest_40"]), "np.all(np.asarray(first.report.status) == 0)"
    assert all(record["forest_42"]), "np.all(np.asarray(undo.report.status) == 0)"
    assert all(record["forest_44"]), "np.all(np.asarray(second.report.status) == 0)"
    assert all(record["forest_46"]), "np.any(np.asarray(split.vertex_removal) >= 0)"
    assert all(record["forest_80"]), (
        "np.all(np.asarray(prepared.evidence.status) & int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED))"
    )
    for actual, expected in record["forest_81"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_83"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_85"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["forest_92"]), "graph_result.evidence.accepted"
    for actual, expected in record["forest_94"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_101"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_102"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["forest_104"]), "forest is not None"
    for actual, expected in record["forest_105"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_111"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_112"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_113"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_116"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_117"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_118"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_119"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_120"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_126"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["forest_128"]), (
        "np.all(np.asarray(forest.vertex_protected[p])[removed])"
    )
    for actual, expected in record["forest_129"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_130"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_132"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_136"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_140"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_143"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["forest_148"]), (
        "np.all(np.isin(resident_ids, np.asarray(solver_forest.mesh.cell_ids[p])))"
    )
    assert all(record["forest_151"]), (
        "np.sum(np.asarray(original.cell_fields)[old_owned]) == np.sum(np.asarray(accepted.cell_fields)[new_owned])"
    )
    for actual, expected in record["forest_153"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_154"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["forest_155"]), "prepared.forest_parts is not None"
    assert all(record["forest_157"]), "np.all(np.asarray(coarsened.report.status) == 0)"
    for actual, expected in record["forest_158"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["forest_161"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))


def _assert_coarsen(record: dict[str, Any]) -> None:
    assert all(record["coarsen_31"]), "np.all(np.asarray(refined.report.status) == 0)"
    assert all(record["coarsen_41"]), "np.all(np.asarray(full.report.status) == 0)"
    assert all(record["coarsen_42"]), "np.all(np.asarray(full.report.operations) == 4)"
    assert all(record["coarsen_43"]), "np.all(np.asarray(full.report.vertices) == 1)"
    assert all(record["coarsen_44"]), "np.all(np.asarray(full.report.accepted) == 8)"
    for actual, expected in record["coarsen_45"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["coarsen_46"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["coarsen_49"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["coarsen_50"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["coarsen_51"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["coarsen_52"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["coarsen_53"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["coarsen_69"]), "np.all(np.asarray(update.report.status) == 0)"
    assert all(record["coarsen_70"]), "np.all(np.asarray(update.report.operations) == 0)"
    assert all(record["coarsen_71"]), "np.all(np.asarray(update.report.accepted) == 0)"
    for actual, expected in record["coarsen_72"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["coarsen_73"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["coarsen_74"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["coarsen_75"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["coarsen_79"]), (
        "np.all(np.asarray(failed.report.status) & int(AdaptiveSimplexStatus.INVALID_GEOMETRY))"
    )
    assert all(record["coarsen_80"]), "np.all(np.asarray(failed.report.operations) == 0)"
    for actual, expected in record["coarsen_81"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["coarsen_83"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["coarsen_85"]), "np.all(np.asarray(refused.report.failed))"
    assert all(record["coarsen_86"]), "np.all(np.asarray(refused.report.iterations) == 0)"
    for actual, expected in record["coarsen_87"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["coarsen_97"]), "np.all(np.asarray(deep.report.status) == 0)"
    assert all(record["coarsen_99"]), (
        "np.all(np.asarray(limited.report.status) == int(AdaptiveSimplexStatus.PASS_LIMIT))"
    )
    assert all(record["coarsen_100"]), (
        "np.all(np.asarray(limited.report.iterations) == 1)"
    )
    assert all(record["coarsen_105"]), (
        "np.all(np.asarray(finished.report.status) == int(AdaptiveSimplexStatus.PASS_LIMIT))"
    )
    for actual, expected in record["coarsen_107"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["coarsen_108"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))


def _assert_proof(record: dict[str, Any]) -> None:
    assert all(record["proof_37"]), (
        "np.all(np.asarray(_compiled_bisection_checks(parts, first, source, exterior)))"
    )
    assert all(record["proof_39"]), "np.any(np.asarray(first.parents) >= 0)"
    assert all(record["proof_41"]), (
        "np.all(np.asarray(_compiled_bisection_checks(parts, second, first, first_exterior)))"
    )
    assert all(record["proof_46"]), "np.all(np.asarray(work.status) == 0)"
    assert all(record["proof_49"]), (
        "np.all(np.isin(np.asarray(work.root_cell_ids)[valid], initial_ids))"
    )
    assert all(record["proof_50"]), (
        "not np.any(np.isin(np.asarray(work.root_cell_ids)[valid], [10, 11]))"
    )
    assert all(record["proof_54"]), "np.any(support == 4)"
    for actual, expected in record["proof_69"]:
        np.testing.assert_allclose(
            _record_array(actual), _record_array(expected), rtol=0, atol=0
        )
    for actual, expected in record["proof_70"]:
        np.testing.assert_allclose(
            _record_array(actual), _record_array(expected), rtol=0, atol=0
        )
    assert all(record["proof_72"]), (
        "np.all(np.asarray(_compiled_bisection_checks(parts, coarse, first, first_exterior)))"
    )
    assert all(record["proof_74"]), "np.all(np.asarray(work.status) == 0)"
    for actual, expected in record["proof_81"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["proof_82"]), "int(work.root_cell_ids[rank, lane]) == -1"
    compact = record["compact_predecessors"]
    assert compact["widths"] == [1, 1, 1, 2]
    for name in (
        "dense_oracle_complete",
        "duplicates_negative_and_max_preserved",
        "undersized_collective_refused",
        "forged_zero_status_width_refused",
        "malformed_and_duplicate_witness_refused",
        "unsupported_refused",
        "representation_id_changed",
        "entity_and_affine_bytes_unchanged",
        "canonical_archive_roundtrip",
    ):
        assert compact[name], name
    for actual, expected in record["proof_86"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["proof_90"]), "not np.all(checks[:, 0])"
    assert all(record["proof_94"]), (
        "np.all(np.asarray(_compiled_bisection_checks(parts, again, coarse, coarse_exterior)))"
    )
    assert all(record["proof_97"]), (
        "not np.all(np.asarray(_compiled_bisection_checks(parts, bad, coarse, coarse_exterior))[:, 0])"
    )
    for actual, expected in record["proof_145"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["proof_146"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["proof_152"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["proof_154"]), "geometry_id == cell_geometry_id(original.geometry)"
    for actual, expected in record["proof_155"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["proof_156"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["proof_158"]), "geometry_id != cell_geometry_id(original.geometry)"
    assert all(record["proof_171"]), "exact != Fraction.from_float(float(actual))"
    assert all(record["proof_172"]), (
        "exact == (Fraction.from_float(float(points[0, 0])) + Fraction.from_float(float(points[2, 0]))) / 2"
    )


def _assert_solver(record: dict[str, Any]) -> None:
    for actual, expected in record["solver_34"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["solver_39"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["solver_43"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["solver_47"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["solver_63"]), (
        "np.all(np.asarray(status) & int(AdaptiveSimplexStatus.INVALID_GEOMETRY))"
    )
    for actual, expected in record["solver_65"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["solver_66"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["solver_73"]), "ancestor >= 0"
    assert all(record["solver_74"]), (
        "int(owners[rank, slot]) == int(bank[rank, ancestor])"
    )
    for actual, expected in record["solver_78"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))


def _assert_plc(record: dict[str, Any]) -> None:
    assert all(record["plc_30"]), (
        "source.certification is not None and source.certification.coverage is not None"
    )
    assert all(record["plc_48"]), (
        "np.all(np.asarray(update.report.status.addressable_shards[0].data) == 0)"
    )
    assert all(record["plc_50"]), "result.status is M.MeshAdaptationStatus.COMPLETE"
    assert all(record["plc_53"]), (
        "storage is not None and storage.partition_index == jax.process_index()"
    )
    assert all(record["plc_54"]), "target.collective_certificates is not None"
    assert all(record["plc_58"]), (
        "embedding.status == 'certified' and (not embedding.findings)"
    )
    assert all(record["plc_59"]), (
        "coverage.status == 'certified' and (not coverage.findings)"
    )
    assert all(record["plc_60"]), (
        "coverage.embedding_certificate_id == embedding.certificate_id"
    )
    assert all(record["plc_61"]), "coverage.domain_id == domain.domain_id"
    assert all(record["plc_62"]), (
        "coverage.premise_certificate_ids == (source.certification.coverage.certificate_id, storage.evidence_id)"
    )
    assert all(record["plc_63"]), (
        "coverage.covered_source_facet_count == coverage.source_facet_count"
    )
    assert all(record["plc_65"]), (
        "coverage.requested_region_measures == coverage.achieved_region_measures == (3.0,)"
    )
    assert all(record["plc_67"]), (
        "charged is not None and required is not None and (required >= charged)"
    )
    assert all(record["plc_80"]), "association.exact and association.complete"
    for actual, expected in record["plc_81"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["plc_90"]), (
        "point == authority[int(np.flatnonzero(transfer.vertex_indices == index)[0])]"
    )
    assert all(record["plc_92"]), (
        "on_segment(point, int(np.flatnonzero(transfer.edge_indices == index)[0]))"
    )
    assert all(record["plc_94"]), (
        "not any((on_segment(point, edge) for edge in range(transfer.edge_vertices.shape[0])))"
    )


def _assert_repack(record: dict[str, Any]) -> None:
    assert all(record["repack_60"]), (
        "np.all(np.asarray(report.status.addressable_shards[0].data) == 0)"
    )
    assert all(record["repack_91"]), "not raw.mesh.cell_ids.is_fully_addressable"
    assert all(record["repack_refusal_106"]), "ValueError"
    assert all(record["repack_110"]), "'maximum_host_packet_bytes' in str(error)"
    for actual, expected in record["repack_148"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["repack_151"]), "np.all(np.asarray(receipt.status) & int(required))"
    for actual, expected in record["repack_154"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["repack_158"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["repack_161"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["repack_162"]), "int(jnp.sum(initial_repack.cells_after)) == 4"
    assert all(record["repack_165"]), (
        "int(missing_status) & int(AdaptiveSimplexStatus.INVALID_GEOMETRY)"
    )
    assert all(record["repack_166"]), (
        "receipt.parts.part_count == count and receipt.states.mesh.cell_ids.shape == (count, 32)"
    )
    assert all(record["repack_167"]), "len(receipt.parts.mesh.devices.flat) == count"
    assert all(record["repack_168"]), "np.sum(np.asarray(receipt.cells_after)) == 20"
    for actual, expected in record["repack_173"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["repack_175"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["repack_176"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["repack_177"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["repack_182"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    assert all(record["repack_189"]), (
        "constraints and all((edge == [0, 1] for edge in constraints))"
    )
    for actual, expected in record["repack_190"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["repack_191"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["repack_192"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["repack_193"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["repack_195"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))
    for actual, expected in record["repack_197"]:
        np.testing.assert_array_equal(_record_array(actual), _record_array(expected))


@pytest.mark.parametrize("scenario", ("publication", "coarsen_publication"))
def test_native_owner_local_publication_transfer_fe_fv_and_atomic_rebind(
    scenario: str,
) -> None:
    record = run_on_forced_devices(
        _MODULE + "native_distributed", f"native_{scenario}", (), devices=2, env={}
    )
    _assert_distributed(record)


def test_native_publication_preserves_scientific_markers_and_zone_membership() -> None:
    record = run_on_forced_devices(
        _MODULE + "native_distributed",
        "native_scientific_receipts",
        (),
        devices=2,
        env={},
    )
    _assert_distributed(record)


@pytest.mark.parametrize("scenario", ("packed_inverse", "packed_inverse_tetra"))
def test_packed_uniform_collective_inverse_retains_raw_epoch_and_cold_scientific_source(
    scenario: str, tmp_path: Path
) -> None:
    record = _run_packed_scenario(scenario, "0", tmp_path / "packed-collective")
    _assert_distributed(record)
    _assert_packed_cold_replay(record["packed_cold_replay"])
    _assert_packed_cold_continuation(record["packed_cold_continuation"])


@pytest.mark.parametrize("scenario", ("packed_partial", "packed_partial_tetra"))
def test_packed_partial_collective_inverse_preserves_physical_source_and_cold_raw_epoch(
    scenario: str, tmp_path: Path
) -> None:
    record = _run_packed_scenario(scenario, "0", tmp_path / "packed-collective")
    _assert_distributed(record)
    _assert_packed_cold_replay(record["packed_cold_replay"])
    _assert_packed_cold_continuation(record["packed_cold_continuation"])


@pytest.mark.parametrize(
    "scenario",
    ("packed_inverse", "packed_inverse_tetra", "packed_partial", "packed_partial_tetra"),
)
def test_packed_inverse_after_actual_solver_placement_separation(
    scenario: str, tmp_path: Path
) -> None:
    record = _run_packed_scenario(scenario, "1", tmp_path / "packed-collective")
    _assert_distributed(record)
    _assert_packed_cold_replay(record["packed_cold_replay"])
    _assert_packed_cold_continuation(record["packed_cold_continuation"])


def test_retained_native_ownership_cannot_claim_an_unexecuted_partition_algorithm() -> (
    None
):
    record = run_on_forced_devices(
        _MODULE + "native_distributed", "native_algorithm_relabel", (), devices=2, env={}
    )
    _assert_distributed(record)


@pytest.mark.parametrize(
    "scenario",
    (
        "midpoint",
        "rounded_edge",
        "constraint_history",
        "child_coverage",
        "routes",
        "terminal_status",
        "audit_only",
        "halo_depth",
        "halo_overflow",
    ),
)
def test_native_distributed_collective_certificate_and_resource_boundaries(
    scenario: str,
) -> None:
    record = run_on_forced_devices(
        _MODULE + "native_distributed", f"native_{scenario}", (), devices=2, env={}
    )
    _assert_distributed(record)
    if scenario == "halo_depth":
        _assert_halo_depth(record)
    elif scenario == "halo_overflow":
        _assert_halo_overflow(record)


def test_historical_shared_vertex_routes_replay_on_changed_device_count() -> None:
    record = run_on_forced_devices(
        _MODULE + "native_distributed_local", "historical_routes", (), devices=1, env={}
    )
    _assert_historical_routes(record)


def test_one_owner_halo_preserves_local_field_without_peer_packets() -> None:
    record = run_on_forced_devices(
        _MODULE + "native_distributed_local", "one_owner_halo", (), devices=1, env={}
    )
    _assert_one_owner_halo(record)


@pytest.mark.parametrize(
    "scenario",
    ("positive", "cell_boundary", "overflow", "vertex_overflow", "shared_field_conflict"),
)
def test_native_owner_local_semantic_migration(scenario: str) -> None:
    record = run_on_forced_devices(
        _MODULE + "native_distributed_migration",
        f"migration_{scenario}",
        (),
        devices=2,
        env={},
    )
    _assert_migration(record)


@pytest.mark.parametrize("device_count", (2, 4))
def test_collective_complete_family_coarsening(device_count: int) -> None:
    record = run_on_forced_devices(
        _MODULE + "native_distributed_coarsen",
        "complete_family_coarsening",
        (str(device_count),),
        devices=device_count,
        env={},
    )
    _assert_coarsen(record)


@pytest.mark.parametrize("coordinate_kind", ("dyadic", "rounded"))
def test_accepted_predecessor_forest_and_scientific_neighborhood_ancestry(
    coordinate_kind: str, tmp_path: Path,
) -> None:
    record = run_on_forced_devices(
        _MODULE + "native_distributed_proof",
        "predecessor_forest_proof",
        (coordinate_kind, str(tmp_path / "compact-predecessors.npz")),
        devices=2,
        env={},
    )
    _assert_proof(record)


@pytest.mark.parametrize("scenario", ("positive", "forest_overflow"))
def test_complete_forest_migration_and_cross_owner_coarsening(scenario: str) -> None:
    record = run_on_forced_devices(
        _MODULE + "native_distributed_forest", f"forest_{scenario}", (), devices=2, env={}
    )
    _assert_forest(record)


@pytest.fixture(scope="module")
def two_owner_plc_publication() -> tuple[dict[str, Any], ...]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"

    def launch(process_id: int) -> dict[str, Any]:
        return run_on_forced_devices(
            _MODULE + "native_distributed_plc",
            "planar_plc_transfer",
            (),
            devices=1,
            env=gloo_rank_environment(coordinator, process_id),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = tuple(executor.map(launch, range(2)))
    return owners


def test_two_owner_bisection_transfers_exact_planar_plc_associations(
    two_owner_plc_publication: tuple[dict[str, Any], ...],
) -> None:
    owners = two_owner_plc_publication
    for owner in owners:
        _assert_plc(owner["checks"])
    assert [owner["partition"] for owner in owners] == [0, 1]
    for dimension in ("0", "1", "2"):
        first, second = (owner["strata"][dimension] for owner in owners)
        shared = first.keys() & second.keys()
        assert shared and all((first[key] == second[key] for key in shared))
        assert len(first.keys() | second.keys()) == owners[0]["counts"][int(dimension)]
    first, second = (owner["strata"]["1"] for owner in owners)
    assert any((first[key][0] == 1 for key in first.keys() & second.keys()))


def test_owner_local_validity_reuses_proof_without_fetching_global_storage(
    two_owner_plc_publication: tuple[dict[str, Any], ...],
) -> None:
    for owner in two_owner_plc_publication:
        validity = owner["owner_local_validity"]
        assert validity["global_storage_present"]
        assert validity["nonaddressable_global_banks"]
        assert validity["all_certified"]
        assert validity["cache_reused"]
        assert validity["reuse_work_units"] == 0


@pytest.fixture(scope="module")
def _saved_raw_restart(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("raw-forest-restart")
    (directory / "repository").mkdir()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"

    def launch(process_id: int) -> dict[str, Any]:
        return run_on_forced_devices(
            _MODULE + "native_distributed_restart",
            "save_complete_epoch",
            (str(directory),),
            devices=1,
            env=gloo_rank_environment(coordinator, process_id),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = tuple(executor.map(launch, range(2)))
    assert [owner["partition"] for owner in owners] == [0, 1]
    for owner in owners:
        _assert_repack(owner)
    return directory


@pytest.mark.parametrize("target_count", (1, 3))
def test_saved_complete_forest_reassembles_on_changed_device_count(
    _saved_raw_restart: Path, target_count: int
) -> None:
    record = run_on_forced_devices(
        _MODULE + "native_distributed_restart",
        "restart_positive",
        (str(_saved_raw_restart), str(target_count)),
        devices=target_count,
        env={},
    )
    _assert_repack(record)


@pytest.mark.parametrize(
    "fault",
    (
        "duplicate",
        "missing",
        "overflow",
        "shared_conflict",
        "lost_authority",
        "packet_overflow",
    ),
)
def test_changed_owner_count_rejects_malformed_complete_epoch(
    _saved_raw_restart: Path, fault: str
) -> None:
    args = (
        (str(_saved_raw_restart),)
        if fault == "packet_overflow"
        else (str(_saved_raw_restart), "1")
    )
    record = run_on_forced_devices(
        _MODULE + "native_distributed_restart",
        f"restart_{fault}",
        args,
        devices=1,
        env={},
    )
    _assert_repack(record)


@pytest.mark.parametrize(
    "scenario",
    (
        "refine",
        "coarsen",
        "mixed",
        "missing_owner",
        "incomplete_support",
        "contradictory_parent",
        "overlapping_source",
    ),
)
def test_solver_owner_inheritance_from_complete_scientific_forest(scenario: str) -> None:
    record = run_on_forced_devices(
        _MODULE + "native_distributed_solver", f"solver_{scenario}", (), devices=2, env={}
    )
    _assert_solver(record)
