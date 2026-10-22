# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._execution_runtime import ExecutionRuntime
from phydrax.discretization._adaptive_simplex import (
    coarsen_adaptive_simplex_parts,
    refine_adaptive_simplex_parts,
)
from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellMeshStorage
from phydrax.lifecycle._meshing_sources import write_meshing_source_closure
from phydrax.meshing._device_adaptation import (
    commit_partitioned_mesh_restart,
    prepare_partitioned_mesh_adaptation,
    validate_partitioned_mesh_evidence,
)
from phydrax.meshing._restart_distribution import repack_simplex_restart
from phydrax.meshing._result import CollectiveMeshEvidence
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.distributed_meshing.native_distributed import _prepare_source
from tests._support.distributed_meshing.native_distributed_records import (
    _new_checks,
    _record_array,
    _require_instance,
    _require_present,
)


_CONTRACT_KEYS = (
    "distributed_126",
    "distributed_127",
    "distributed_132",
    "distributed_172",
    "distributed_174",
    "distributed_178",
    "distributed_180",
    "distributed_209",
    "distributed_216",
    "distributed_217",
    "distributed_246",
    "distributed_262",
    "distributed_270",
    "distributed_271",
    "distributed_280",
    "distributed_293",
    "distributed_297",
    "distributed_308",
    "distributed_312",
    "distributed_318",
    "distributed_320",
    "distributed_321",
    "distributed_323",
    "distributed_325",
    "distributed_326",
    "distributed_327",
    "distributed_328",
    "distributed_329",
    "distributed_347",
    "distributed_355",
    "distributed_356",
    "distributed_358",
    "distributed_359",
    "distributed_365",
    "distributed_367",
    "distributed_368",
    "distributed_371",
    "distributed_372",
    "distributed_375",
    "distributed_378",
    "distributed_382",
    "distributed_383",
    "distributed_397",
    "distributed_537",
    "distributed_546",
    "distributed_548",
    "distributed_554",
    "distributed_558",
    "distributed_566",
    "distributed_569",
    "distributed_572",
    "distributed_576",
    "distributed_577",
    "distributed_579",
    "distributed_581",
    "distributed_583",
    "distributed_584",
    "distributed_586",
    "distributed_587",
    "distributed_595",
    "distributed_596",
    "distributed_605",
    "distributed_612",
    "distributed_617",
    "distributed_619",
    "distributed_620",
    "distributed_621",
    "distributed_622",
    "distributed_624",
    "distributed_625",
    "distributed_642",
    "distributed_673",
    "distributed_674",
    "distributed_675",
    "distributed_682",
    "distributed_685",
    "distributed_686",
    "distributed_687",
    "distributed_698",
    "distributed_700",
    "distributed_707",
    "distributed_708",
    "distributed_710",
    "distributed_711",
    "distributed_717",
    "distributed_725",
    "distributed_726",
    "distributed_740",
    "distributed_741",
    "distributed_743",
    "distributed_744",
    "distributed_747",
    "distributed_760",
    "distributed_761",
    "distributed_762",
    "distributed_763",
    "distributed_764",
    "distributed_765",
    "distributed_766",
    "distributed_767",
    "distributed_768",
    "distributed_769",
    "distributed_770",
    "distributed_773",
    "distributed_775",
    "distributed_789",
    "distributed_790",
    "distributed_791",
    "distributed_792",
    "distributed_804",
    "distributed_808",
    "distributed_841",
    "distributed_842",
    "distributed_845",
    "distributed_895",
    "distributed_896",
    "distributed_897",
    "distributed_898",
    "distributed_913",
    "distributed_915",
    "distributed_916",
    "distributed_refusal_286",
    "distributed_refusal_389",
    "distributed_refusal_592",
    "distributed_refusal_602",
    "distributed_refusal_679",
    "distributed_refusal_794",
)
M = phx.meshing
D = phx.discretization


def _packed_inverse(
    scenario: str, semantic_owners_flag: str, archive_path: str
) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    semantic_owners = semantic_owners_flag == "1"
    setup = _prepare_source(scenario, checks)
    source, source_attributes, source_zones = (
        setup.source,
        setup.source_attributes,
        setup.source_zones,
    )
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(source.geometry)
    ) as budget:
        original_coefficients = source.geometry.source_coordinates(budget=budget)
    fine = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            M.MarkedMeshAdaptation(),
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=M.BisectionCompatibility.UNIFORM_REFINEMENT,
            ),
        )
    )
    lineage = _require_present(
        _require_instance(fine.hierarchy, M.BisectionHierarchy).uniform_refinement
    )
    checks.setdefault("distributed_126", []).append(
        bool(
            lineage is not None
            and lineage.child_ids.shape[1] == (24 if scenario.endswith("tetra") else 6)
        )
    )
    checks.setdefault("distributed_127", []).append(bool(lineage.source is source))
    fine_ids = np.concatenate(
        tuple((np.asarray(block.global_ids) for block in fine.target.mesh.blocks))
    )
    owners = np.full(fine_ids.shape, -1, dtype=np.int32)
    for rank, children in enumerate(np.asarray(lineage.child_ids)):
        owners[np.isin(fine_ids, children)] = (
            (0 if rank < 2 else 1)
            if scenario in ("packed_partial", "packed_partial_tetra")
            else rank
        )
    checks.setdefault("distributed_132", []).append(bool(np.all(owners >= 0)))
    partition_policy = M.MeshPartitionPolicy(
        M.MeshPartitionKind.PROVIDER, 2, halo_width=2, maximum_imbalance=2.0
    )
    distribution = M.prepare_mesh_distribution(
        M.MeshPart("domain", fine.target), policy=partition_policy, ownership=owners
    )
    policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=D.AdaptiveSimplexPolicy(vertex_capacity=64, cell_capacity=64),
        distribution=distribution,
        partition_policy=partition_policy,
    )
    prepared = M.prepare_adaptive_simplex(
        fine.target,
        policy=policy,
        hierarchy=_require_instance(fine.hierarchy, M.BisectionHierarchy),
    )
    partitioned = M.partition_adaptive_simplex(prepared)
    born = np.setdiff1d(fine.target.mesh.vertex_global_ids, source.mesh.vertex_global_ids)
    protected_scope = M.MeshingScope(
        fine.target.mesh.mesh_id,
        fine.target.mesh.numeric_version,
        M.MeshingEntityKind.MESH,
        0,
        fine.target.mesh.entity_set(0).entity_set_id,
        born[:1],
    )
    protected_policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=policy.device_policy,
        distribution=distribution,
        partition_policy=partition_policy,
        protected_scopes=(protected_scope,),
    )
    protected = prepare_partitioned_mesh_adaptation(
        M.prepare_mesh_adaptation(
            fine.target,
            M.MarkedMeshAdaptation(
                (),
                np.sort(fine_ids),
                hierarchy=_require_instance(fine.hierarchy, M.BisectionHierarchy),
            ),
            policy=protected_policy,
        )
    )
    protected_raw = coarsen_adaptive_simplex_parts(protected.layout, type(protected).execution_parts(protected), protected.states,
    protected.states.mesh.cell_active,).state
    protected_before = tuple(
        (np.array(value, copy=True) for value in jax.tree_util.tree_leaves(protected_raw))
    )
    protected_results = M.commit_partitioned_adaptive_simplex(
        protected, protected_raw
    ).local_results
    protected_banks = dict(
        _require_instance(
            protected_results[0].target.collective_evidence, CollectiveMeshEvidence
        ).logical_arrays
    )
    protected_position = np.flatnonzero(protected_banks["vertex_global_ids"] == born[0])
    checks.setdefault("distributed_172", []).append(bool(protected_position.size == 1))
    fine_position = np.flatnonzero(
        np.asarray(fine.target.mesh.vertex_global_ids) == born[0]
    )
    _record_array(
        checks,
        "distributed_174",
        protected_banks["coordinates"][protected_position[0]],
        np.asarray(fine.target.mesh.coordinates)[fine_position[0]],
    )
    checks.setdefault("distributed_178", []).append(
        bool(
            any(
                (
                    _require_instance(
                        result.hierarchy, M.BisectionHierarchy
                    ).uniform_refinement
                    is not None
                    for result in protected_results
                )
            )
        )
    )
    for value, original in zip(
        jax.tree_util.tree_leaves(protected_raw), protected_before, strict=True
    ):
        _record_array(checks, "distributed_180", value, original)
    if semantic_owners:
        sparse = partitioned.cell_marks(np.asarray(lineage.child_ids)[0, :1])
        split = refine_adaptive_simplex_parts(partitioned.layout, type(partitioned).execution_parts(partitioned), partitioned.states, sparse).state
        split_results = M.commit_partitioned_adaptive_simplex(
            partitioned, split
        ).local_results
        accepted = split_results[0].target
        rank = jnp.arange(2, dtype=jnp.int32)[:, None]
        physical = jnp.where(split.mesh.cell_ids >= 0, rank, -1)
        vertex_owners = jnp.full(split.mesh.vertex_ids.shape, 2, dtype=jnp.int32)
        for owner in range(2):
            present = jnp.any(
                split.mesh.vertex_ids[:, :, None]
                == split.mesh.vertex_ids[owner, None, None, :],
                axis=-1,
            )
            vertex_owners = jnp.minimum(vertex_owners, jnp.where(present, owner, 2))
        vertex_owners = jnp.where(split.mesh.vertex_ids >= 0, vertex_owners, -1)
        proposal = jnp.broadcast_to(rank, split.mesh.cell_ids.shape)
        binary = np.flatnonzero(
            np.all(np.asarray(split.children[0]) >= 0, axis=1)
            & np.all(
                np.asarray(split.mesh.cell_active[0])[
                    np.maximum(np.asarray(split.children[0]), 0)
                ],
                axis=1,
            )
        )
        checks.setdefault("distributed_209", []).append(bool(binary.size > 0))
        second = int(np.asarray(split.children[0])[binary[0], 1])
        proposal = proposal.at[0, second].set(1)
        repack = repack_simplex_restart(
            ExecutionRuntime.current().root_group,
            partitioned.layout,
            split,
            physical,
            vertex_owners,
            proposal,
            source_result=accepted,
        )
        _record_array(checks, "distributed_216", repack.status, 0)
        checks.setdefault("distributed_217", []).append(
            bool(
                np.any(
                    np.asarray(repack.states.mesh.cell_active)
                    & (np.asarray(repack.solver_cell_owners) != np.arange(2)[:, None])
                )
            )
        )
        accepted_distribution = M.prepare_mesh_distribution(
            M.MeshPart("domain", accepted), policy=partition_policy
        )
        accepted_policy = M.MeshAdaptationPolicy(
            M.MeshAdaptationRoute.DEVICE_BISECTION,
            device_policy=policy.device_policy,
            distribution=accepted_distribution,
            partition_policy=partition_policy,
        )
        placed_results = commit_partitioned_mesh_restart(
            repack,
            M.prepare_mesh_adaptation(
                accepted,
                M.MarkedMeshAdaptation(
                    hierarchy=_require_instance(
                        split_results[0].hierarchy, M.BisectionHierarchy
                    )
                ),
                policy=accepted_policy,
            ),
            proof=M.SimplexRestartRepackProof(repack),
        ).local_results
        placed = placed_results[0].target
        placed_distribution = M.prepare_mesh_distribution(
            M.MeshPart("domain", placed), policy=partition_policy
        )
        policy = M.MeshAdaptationPolicy(
            M.MeshAdaptationRoute.DEVICE_BISECTION,
            device_policy=policy.device_policy,
            distribution=placed_distribution,
            partition_policy=partition_policy,
        )
        partitioned = prepare_partitioned_mesh_adaptation(
            M.prepare_mesh_adaptation(
                placed,
                M.MarkedMeshAdaptation(
                    hierarchy=_require_instance(
                        placed_results[0].hierarchy, M.BisectionHierarchy
                    )
                ),
                policy=policy,
            )
        )
        checks.setdefault("distributed_246", []).append(
            bool(
                np.any(
                    np.asarray(partitioned.states.mesh.cell_active)
                    & (
                        np.asarray(partitioned.solver_cell_owners)
                        != np.arange(2)[:, None]
                    )
                )
            )
        )
    coarsen_marks = partitioned.states.mesh.cell_active
    partial = scenario in ("packed_partial", "packed_partial_tetra")
    if partial:
        coarsen_marks = coarsen_marks & (jnp.arange(2)[:, None] == 0)
    compiled = coarsen_adaptive_simplex_parts(partitioned.layout, type(partitioned).execution_parts(partitioned), partitioned.states, coarsen_marks).state
    before = tuple(
        (np.array(value, copy=True) for value in jax.tree_util.tree_leaves(compiled))
    )
    results = M.commit_partitioned_adaptive_simplex(partitioned, compiled).local_results
    evidence = _require_instance(
        results[0].target.collective_evidence, CollectiveMeshEvidence
    )
    checks.setdefault("distributed_262", []).append(
        bool(
            all(
                (
                    retained is submitted
                    for retained, submitted in zip(
                        jax.tree_util.tree_leaves(evidence.compiled_states),
                        jax.tree_util.tree_leaves(compiled),
                        strict=True,
                    )
                )
            )
        )
    )
    checks.setdefault("distributed_270", []).append(
        bool(
            _require_present(evidence.uniform_refinement).lineage_id == lineage.lineage_id
        )
    )
    checks.setdefault("distributed_271", []).append(
        bool(
            all(
                (
                    retained is submitted
                    for retained, submitted in zip(
                        jax.tree_util.tree_leaves(evidence.uniform_refinement),
                        jax.tree_util.tree_leaves(lineage),
                        strict=True,
                    )
                )
            )
        )
    )
    for value, original in zip(jax.tree_util.tree_leaves(compiled), before, strict=True):
        _record_array(checks, "distributed_280", value, original)
    invalid_compiled = eqx.tree_at(
        lambda value: value.counters, compiled, -jnp.ones_like(compiled.counters)
    )
    with pytest.raises(
        (ValueError, M.MeshingFailure),
        match=r"commit_partitioned_adaptive_simplex\.baseline must retain",
    ):
        M.commit_partitioned_adaptive_simplex(partitioned, invalid_compiled)
    checks["distributed_refusal_286"].append(True)
    for value, original in zip(jax.tree_util.tree_leaves(compiled), before, strict=True):
        _record_array(checks, "distributed_293", value, original)
    banks = dict(evidence.logical_arrays)
    original_ids = np.concatenate(
        tuple((np.asarray(block.global_ids) for block in source.mesh.blocks))
    )
    expected_cells = 2 + lineage.child_ids.shape[1] if partial else 2
    checks.setdefault("distributed_297", []).append(
        bool(evidence.global_entity_counts[-1] == expected_cells)
    )
    raw_ids = np.asarray(partitioned.states.mesh.cell_ids)
    raw_owners = np.asarray(partitioned.solver_cell_owners)
    raw_active = np.asarray(partitioned.states.mesh.cell_active)
    raw_children = np.asarray(partitioned.states.children)
    published_owners = np.asarray(banks["cell_owners"])
    published_storage = _require_instance(results[0].target.mesh.storage, CellMeshStorage)
    closure_banks = dict(published_storage.logical_arrays)
    predecessor_ids = np.asarray(closure_banks["closure/source_cell_ids"])
    closure_valid = np.asarray(closure_banks["closure/cell_valid"])
    closure_ids = np.asarray(closure_banks["closure/cell_ids"])
    complete_counts = np.sum(predecessor_ids >= 0, axis=-1)
    assert predecessor_ids.shape[-1] == max(1, int(complete_counts[closure_valid].max()))
    for row in predecessor_ids[closure_valid]:
        selected = row[row >= 0]
        np.testing.assert_array_equal(row[: selected.size], np.sort(selected, kind="stable"))
        np.testing.assert_array_equal(row[selected.size :], -1)
    for parent, siblings in zip(
        np.asarray(lineage.parent_ids), np.asarray(lineage.child_ids), strict=True
    ):
        published = np.flatnonzero(np.asarray(banks["cell_global_ids"]) == parent)
        if published.size:
            support = []
            predecessor_support = []
            pending = [
                tuple(position) for position in np.argwhere(np.isin(raw_ids, siblings))
            ]
            checks.setdefault("distributed_308", []).append(
                bool(len(pending) == siblings.size)
            )
            visited = set()
            while pending:
                owner, slot = pending.pop()
                assert (owner, slot) not in visited
                checks.setdefault("distributed_312", []).append(
                    bool((owner, slot) not in visited)
                )
                visited.add((owner, slot))
                if raw_active[owner, slot]:
                    support.append(raw_owners[owner, slot])
                    predecessor_support.append(raw_ids[owner, slot])
                else:
                    children = raw_children[owner, slot]
                    checks.setdefault("distributed_318", []).append(
                        bool(np.all(children >= 0))
                    )
                    pending.extend(((owner, int(child)) for child in children))
            checks.setdefault("distributed_320", []).append(
                bool(support and np.all(np.asarray(support) >= 0))
            )
            checks.setdefault("distributed_321", []).append(
                bool(published_owners[published[0]] == np.min(support))
            )
            expected_support = np.sort(np.asarray(predecessor_support), kind="stable")
            copies = predecessor_ids[closure_valid & (closure_ids == parent)]
            assert copies.shape[0] > 0
            for copy in copies:
                np.testing.assert_array_equal(copy[: expected_support.size], expected_support)
                np.testing.assert_array_equal(copy[expected_support.size :], -1)
    checks["compact_packed_support"] = {
        "source_width": predecessor_ids.shape[-1],
        "complete_parent_multisets": True,
        "canonical_prefix": True,
    }
    if not partial:
        _record_array(
            checks, "distributed_323", banks["cell_global_ids"][:2], original_ids
        )
        vertex_count = source.mesh.vertex_global_ids.size
        _record_array(
            checks,
            "distributed_325",
            banks["vertex_global_ids"][:vertex_count],
            source.mesh.vertex_global_ids,
        )
        _record_array(
            checks,
            "distributed_326",
            banks["coordinates"][:vertex_count],
            source.mesh.coordinates,
        )
        checks.setdefault("distributed_327", []).append(
            bool(evidence.coordinate_geometry_id == cell_geometry_id(source.geometry))
        )
    checks.setdefault("distributed_328", []).append(
        bool(not np.array_equal(banks["epoch/cell_ids"], compiled.mesh.cell_ids))
    )
    checks.setdefault("distributed_329", []).append(
        bool(
            any(
                (
                    _require_instance(
                        result.hierarchy, M.BisectionHierarchy
                    ).uniform_refinement
                    is not None
                    for result in results
                )
            )
            == partial
        )
    )
    authored = dict(
        zip(
            np.asarray(source_attributes[0].scope.global_entity_ids),
            np.asarray(source_attributes[0].global_values),
            strict=True,
        )
    )
    original_parent = {int(parent): int(parent) for parent in lineage.parent_ids}
    original_parent.update(
        {
            int(child): int(parent)
            for parent, siblings in zip(
                lineage.parent_ids, lineage.child_ids, strict=True
            )
            for child in siblings
        }
    )
    accepted_ids = np.asarray(banks["cell_global_ids"])[:expected_cells]
    expected_markers = np.asarray(
        [authored[original_parent[int(identifier)]] for identifier in accepted_ids]
    )
    serial_elements = {}
    for serial_source in (source, fine.target):
        source_elements, _, _ = serial_source.geometry.resolve(serial_source.mesh)
        for block, element in zip(
            serial_source.mesh.blocks, source_elements, strict=True
        ):
            if block.name in serial_elements:
                checks.setdefault("distributed_347", []).append(
                    bool(serial_elements[block.name] == element.element_id)
                )
            serial_elements[block.name] = element.element_id
    serial_coordinates = {
        int(identifier): np.asarray(fine.target.mesh.coordinates)[index]
        for index, identifier in enumerate(fine.target.mesh.vertex_global_ids)
    }
    for result in results:
        marker = result.target.attributes[0]
        _record_array(
            checks, "distributed_355", marker.scope.global_entity_ids, accepted_ids
        )
        _record_array(checks, "distributed_356", marker.global_values, expected_markers)
        local_rows = np.searchsorted(accepted_ids, np.asarray(marker.scope.entity_ids))
        _record_array(
            checks, "distributed_358", marker.values, expected_markers[local_rows]
        )
        checks.setdefault("distributed_359", []).append(
            bool(
                tuple((zone.name for zone in result.target.zones))
                == tuple((zone.name for zone in source_zones))
            )
        )
        for zone, original_zone in zip(result.target.zones, source_zones, strict=True):
            members = np.asarray(original_zone.scope.global_entity_ids)
            expected = accepted_ids[
                np.isin(
                    np.asarray(
                        [original_parent[int(identifier)] for identifier in accepted_ids]
                    ),
                    members,
                )
            ]
            _record_array(
                checks, "distributed_365", zone.scope.global_entity_ids, expected
            )
        storage = _require_present(result.target.mesh.storage)
        checks.setdefault("distributed_367", []).append(
            bool(isinstance(storage, CellMeshStorage))
        )
        with coordinate_reference_budget(
            queries_upper=(
                cell_geometry_preparation_query_upper(result.target.geometry)
                + 2 * cell_geometry_preparation_query_upper(source.geometry)
            )
        ) as budget:
            checks.setdefault("distributed_368", []).append(
                bool(
                    result.target.geometry.source_coordinates(budget=budget)
                    == source.geometry.source_coordinates(
                        storage.coordinate_global_ids, budget=budget
                    )
                )
            )
            checks.setdefault("distributed_371", []).append(
                bool(lineage.source is source)
            )
            checks.setdefault("distributed_372", []).append(
                bool(
                    source.geometry.source_coordinates(budget=budget)
                    == original_coefficients
                )
            )
        elements, routes, coefficients = result.target.geometry.resolve(
            result.target.mesh
        )
        for block, element, route in zip(
            result.target.mesh.blocks, elements, routes, strict=True
        ):
            checks.setdefault("distributed_375", []).append(
                bool(element.element_id == serial_elements[block.name])
            )
            for row in np.asarray(block.vertices):
                identifiers = np.asarray(result.target.mesh.vertex_global_ids)[row]
                _record_array(
                    checks,
                    "distributed_378",
                    result.target.mesh.coordinates[row],
                    np.stack(
                        [
                            serial_coordinates[int(identifier)]
                            for identifier in identifiers
                        ]
                    ),
                )
        checks.setdefault("distributed_382", []).append(
            bool(_require_present(result.target.execution_evidence) is not None)
        )
        checks.setdefault("distributed_383", []).append(
            bool(
                int(
                    _require_present(
                        result.target.execution_evidence
                    ).externally_charged_work
                )
                > 0
            )
        )
    altered_raw = eqx.tree_at(
        lambda value: value.compiled_states.counters,
        evidence,
        evidence.compiled_states.counters.at[0, 0].add(1),
    )
    with pytest.raises(
        ValueError,
        match="restore_partitioned_mesh_state|run_validate_partitioned_mesh_evidence",
    ):
        validate_partitioned_mesh_evidence(
            _require_present(results[0].target.mesh.storage), altered_raw
        )
    checks["distributed_refusal_389"].append(True)
    limits = ArrayArchiveLimits()
    checks.setdefault("distributed_397", []).append(
        bool(
            (limits.max_members, limits.max_manifest_nesting, limits.max_array_rank)
            == (257, 16, 8)
        )
    )
    receipt = write_meshing_source_closure(
        Path(archive_path),
        (
            results[0].target,
            policy,
            _require_instance(results[0].hierarchy, M.BisectionHierarchy),
        ),
        limits=limits,
    )
    checks["packed_restore"] = {
        "content_id": receipt.content_id,
        "policy_id": policy.policy_id,
        "source_result_id": source.result_id,
        "expected_cells": str(expected_cells),
    }
    return checks


def native_packed_inverse(semantic_owners_flag: str, archive_path: str) -> dict[str, Any]:
    return _packed_inverse("packed_inverse", semantic_owners_flag, archive_path)


def native_packed_inverse_tetra(
    semantic_owners_flag: str, archive_path: str
) -> dict[str, Any]:
    return _packed_inverse("packed_inverse_tetra", semantic_owners_flag, archive_path)


def native_packed_partial(semantic_owners_flag: str, archive_path: str) -> dict[str, Any]:
    return _packed_inverse("packed_partial", semantic_owners_flag, archive_path)


def native_packed_partial_tetra(
    semantic_owners_flag: str, archive_path: str
) -> dict[str, Any]:
    return _packed_inverse("packed_partial_tetra", semantic_owners_flag, archive_path)
