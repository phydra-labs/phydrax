# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax._meshcore import meshcore_runtime_identity, NativeExecutionScope
from phydrax._strict import StrictModule
from phydrax.discretization._adaptive_simplex import (
    AdaptiveSimplexState,
    coarsen_adaptive_simplex_parts,
)
from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
from phydrax.discretization._distributed_field import prepare_owner_local_halo_plans
from phydrax.discretization.fem._distributed import prepare_owner_local_finite_element
from phydrax.discretization.fem._generic import FiniteElementFieldSpec, FiniteElementPlan
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.discretization.finite_volume._unstructured_dynamics import (
    execute_owner_local_finite_volume_advance,
    PreparedUnstructuredFiniteVolumeDynamics,
)
from phydrax.lifecycle._array_artifact import (
    ArrayArtifactProvenance,
    read_typed_array_artifact,
    write_typed_array_artifact,
)
from phydrax.lifecycle._distributed_checkpoint import snapshot_addressable_arrays
from phydrax.linalg._distributed import DistributedKrylovPolicy
from phydrax.meshing._device_adaptation import prepare_partitioned_mesh_adaptation
from phydrax.meshing._distribution import (
    compiled_bisection_checks,
    expand_partitioned_simplex_neighborhood,
)
from phydrax.meshing._result import CollectiveMeshEvidence
from phydrax.meshing.providers._native_publication import (
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from phydrax.meshing.providers._native_route_policy import NativePublicationPolicy
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


class _NativeSource(StrictModule):
    points: np.ndarray
    cells: np.ndarray
    seed: M.CellMeshingResult
    source: M.CellMeshingResult
    source_attributes: tuple[M.MeshAttribute, ...]
    source_zones: tuple[M.MeshZone, ...]


class _NativeEpoch(StrictModule):
    source: M.CellMeshingResult
    partition_policy: M.MeshPartitionPolicy
    distribution: M.MeshDistribution
    policy: M.MeshAdaptationPolicy
    prepared: M.PreparedAdaptiveSimplex
    partitioned: M.PartitionedAdaptiveSimplex
    states: AdaptiveSimplexState


def _prepare_source(scenario: str, checks: dict[str, Any]) -> _NativeSource:
    print(
        "NATIVE_RUNTIME_IDENTITY",
        os.getpid(),
        json.dumps(meshcore_runtime_identity(), sort_keys=True),
        flush=True,
    )
    points = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    if scenario == "rounded_edge":
        points = np.asarray(
            ((0.1, 0.2), (0.4, 0.2), (0.3, 0.7), (0.0, 0.7)), dtype=np.float64
        )
    if scenario == "halo_depth":
        points = np.asarray(
            ((0, 0), (0.5, 0), (1, 0), (1, 1), (0.5, 1), (0, 1)), dtype=np.float64
        )
        cells = np.asarray(((0, 1, 4), (0, 4, 5), (1, 2, 3), (1, 3, 4)), dtype=np.int32)
    if scenario == "packed_inverse":
        points = np.asarray(
            ((0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)), dtype=np.float64
        )
        cells = np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int32)
    if scenario == "packed_inverse_tetra":
        points = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (1.0, 0.5, 0.0),
                (1.0, 0.2, 1.0),
                (1.0, -3.0, -1.0),
            ),
            dtype=np.float64,
        )
        cells = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int32)
    if scenario in ("packed_partial", "packed_partial_tetra"):
        dimension = 3 if scenario.endswith("tetra") else 2
        simplex = np.concatenate(
            (
                np.zeros((1, dimension), dtype=np.float64),
                np.eye(dimension, dtype=np.float64),
            )
        )
        if dimension == 2:
            component = np.asarray(
                ((0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)), dtype=np.float64
            )
            component_cells = np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int32)
        else:
            component = np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (2.0, 0.0, 0.0),
                    (1.0, 0.5, 0.0),
                    (1.0, 0.2, 1.0),
                    (1.0, -3.0, -1.0),
                ),
                dtype=np.float64,
            )
            component_cells = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int32)
        points = np.concatenate((component, simplex + 4.0))
        cells = np.concatenate(
            (
                component_cells,
                np.arange(component.shape[0], points.shape[0], dtype=np.int32)[None],
            )
        )
    mesh = (
        D.CellMesh.from_tetrahedra(points, cells)
        if scenario in ("packed_inverse_tetra", "packed_partial_tetra")
        else D.CellMesh.from_triangles(points, cells)
    )
    seed = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    loop = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int32)
    if scenario == "halo_depth":
        loop = np.asarray(
            ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)), dtype=np.int32
        )
    if scenario == "packed_inverse":
        loop = np.asarray(((0, 3), (3, 1), (1, 2), (2, 0)), dtype=np.int32)
    if scenario == "packed_partial":
        loop = np.asarray(
            ((0, 3), (3, 1), (1, 2), (2, 0), (4, 5), (5, 6), (6, 4)), dtype=np.int32
        )
    if scenario in ("packed_inverse_tetra", "packed_partial_tetra"):
        faces = cells[
            :, np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int32)
        ].reshape((-1, 3))
        _, first, counts = np.unique(
            np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
        )
        loop = faces[first[counts == 1]]
    source_id = "packed-original" if scenario.startswith("packed_") else "unit-square"
    domain = phx.geometry.PiecewiseLinearDomain(
        points,
        loop,
        np.tile(np.asarray((0, -1), dtype=np.int32), (loop.shape[0], 1)),
        ("domain",),
        source_id=source_id,
    )
    source_attributes, source_zones = ((), ())
    if scenario == "scientific_receipts" or scenario.startswith("packed_"):
        cell_entities = mesh.entity_set(mesh.topological_dimension)
        all_cells = M.MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            M.MeshingEntityKind.MESH,
            mesh.topological_dimension,
            cell_entities.entity_set_id,
            np.asarray(cell_entities.entity_ids, dtype=np.int64),
        )
        source_attributes = (
            M.MeshAttribute(
                "marker",
                M.MeshAttributeRole.MARKER,
                all_cells,
                np.asarray(
                    (19, 23, 31)
                    if scenario in ("packed_partial", "packed_partial_tetra")
                    else (19, 23),
                    dtype=np.int64,
                ),
            ),
        )
        source_zones = tuple(
            (
                M.MeshZone(
                    f"cell:{index}",
                    M.MeshZoneRole.USER,
                    M.MeshingScope(
                        mesh.mesh_id,
                        mesh.numeric_version,
                        M.MeshingEntityKind.MESH,
                        mesh.topological_dimension,
                        cell_entities.entity_set_id,
                        np.asarray((identifier,), dtype=np.int64),
                    ),
                )
                for index, identifier in enumerate(
                    np.asarray(cell_entities.entity_ids, dtype=np.int64)
                )
            )
        )
    limits = M.MeshingLimits()
    with (
        NativeExecutionScope(limits.native_limits()) as scope,
        scope.host_workspace() as workspace,
    ):
        source = publish_native_result(
            seed.mesh,
            seed.coordinate_contract,
            M.MeshingComplianceReport(source_id),
            (),
            seed.provider,
            {"kind": "independent-domain-certified-source"},
            NativeCertificationRequest(
                M.MeshCertificationSchedule("volume_plc"),
                source_id,
                "r0",
                limits,
                domain=domain,
                cell_regions=np.zeros(cells.shape[0], dtype=np.int32),
            ),
            policy=NativePublicationPolicy(
                M.CellMeshAuditPolicy(
                    watertight_boundary=M.CellMeshAuditDisposition.REJECT
                ),
                M.MeshingDerivativeMode.FIXED_TOPOLOGY_EXACT,
                (),
                (),
            ),
            organization=NativePublicationOrganization(
                zones=source_zones, attributes=source_attributes
            ),
            execution=NativePublicationExecution(scope, workspace, None, None),
        )
    return _NativeSource(points, cells, seed, source, source_attributes, source_zones)


def _prepare_epoch(
    setup: _NativeSource, scenario: str, checks: dict[str, Any]
) -> _NativeEpoch:
    source, seed = (setup.source, setup.seed)
    if scenario == "audit_only":
        source = seed
    kind = (
        M.MeshPartitionKind.PROVIDER
        if scenario == "halo_depth"
        else M.MeshPartitionKind.MORTON
    )
    partition_policy = M.MeshPartitionPolicy(kind, 2, halo_width=2)
    distribution = M.prepare_mesh_distribution(
        M.MeshPart("domain", source),
        policy=partition_policy,
        ownership=np.asarray((0, 0, 1, 1), dtype=np.int32)
        if scenario == "halo_depth"
        else None,
    )
    if scenario == "algorithm_relabel":
        partition_policy = M.MeshPartitionPolicy(
            M.MeshPartitionKind.HILBERT, 2, halo_width=2
        )
    policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=D.AdaptiveSimplexPolicy(vertex_capacity=64, cell_capacity=64),
        distribution=distribution,
        partition_policy=partition_policy,
        audit_policy=M.CellMeshAuditPolicy(
            watertight_boundary=M.CellMeshAuditDisposition.REJECT
        ),
    )
    prepared = M.prepare_adaptive_simplex(source, policy=policy)
    partitioned = M.partition_adaptive_simplex(prepared)
    marks = partitioned.cell_marks(np.asarray((0,), dtype=np.int64))
    update = D.refine_adaptive_simplex_parts(
        partitioned.layout, partitioned.parts, partitioned.states, marks
    )
    states = update.state
    if scenario == "coarsen_publication":
        states = coarsen_adaptive_simplex_parts(
            partitioned.layout, partitioned.parts, states, states.mesh.cell_active
        ).state
    return _NativeEpoch(
        source, partition_policy, distribution, policy, prepared, partitioned, states
    )


def native_midpoint() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source("midpoint", checks)
    epoch = _prepare_epoch(setup, "midpoint", checks)
    points = setup.points
    partitioned = epoch.partitioned
    states = epoch.states
    issued = np.flatnonzero(np.asarray(states.mesh.vertex_ids[0]) >= points.shape[0])
    issued = issued[np.asarray(states.mesh.vertex_active[0])[issued]]
    coordinates = states.mesh.coordinates.at[0, issued[0], 1].add(0.125)
    states = eqx.tree_at(lambda value: value.mesh.coordinates, states, coordinates)
    verdicts = compiled_bisection_checks(
        partitioned.parts, states, partitioned.states, partitioned.source_exterior
    )
    checks.setdefault("distributed_537", []).append(
        bool(not np.all(np.asarray(verdicts[:, 2])))
    )
    return checks


def native_constraint_history() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source("constraint_history", checks)
    epoch = _prepare_epoch(setup, "constraint_history", checks)
    points = setup.points
    partitioned = epoch.partitioned
    states = epoch.states
    issued = np.flatnonzero(
        (np.asarray(states.mesh.vertex_ids[0]) >= points.shape[0])
        & np.asarray(states.mesh.vertex_active[0])
    )
    endpoints = np.asarray(states.vertex_parents[0, issued[0]], dtype=np.int64)
    code = np.min(endpoints) * partitioned.layout.vertex_capacity + np.max(endpoints)
    protected = partitioned.states.protected_codes.at[0, 0].set(code)
    initial = eqx.tree_at(
        lambda value: value.protected_codes, partitioned.states, protected
    )
    changed = eqx.tree_at(lambda value: value.protected_codes, states, protected)
    checks.setdefault("distributed_546", []).append(
        bool(np.all(np.asarray(changed.status_flags) == 0))
    )
    verdicts = compiled_bisection_checks(
        partitioned.parts, changed, initial, partitioned.source_exterior
    )
    checks.setdefault("distributed_548", []).append(
        bool(not np.all(np.asarray(verdicts[:, 2])))
    )
    return checks


def native_child_coverage() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source("child_coverage", checks)
    epoch = _prepare_epoch(setup, "child_coverage", checks)
    partitioned = epoch.partitioned
    states = epoch.states
    split = np.flatnonzero(np.asarray(states.children[0, :, 0]) >= 0)[0]
    children = states.children.at[0, split, 0].set(states.children[0, split, 1])
    states = eqx.tree_at(lambda value: value.children, states, children)
    verdicts = compiled_bisection_checks(
        partitioned.parts, states, partitioned.states, partitioned.source_exterior
    )
    checks.setdefault("distributed_554", []).append(
        bool(not np.all(np.asarray(verdicts[:, 1])))
    )
    return checks


def native_routes() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source("routes", checks)
    epoch = _prepare_epoch(setup, "routes", checks)
    partitioned = epoch.partitioned
    states = epoch.states
    parts = D.AdaptiveSimplexParts(tuple(jax.devices()), neighbor_pairs=())
    verdicts = compiled_bisection_checks(
        parts, states, partitioned.states, partitioned.source_exterior
    )
    checks.setdefault("distributed_558", []).append(
        bool(not np.all(np.asarray(verdicts[:, 4])))
    )
    return checks


def native_rounded_edge() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source("rounded_edge", checks)
    epoch = _prepare_epoch(setup, "rounded_edge", checks)
    points = setup.points
    partitioned = epoch.partitioned
    states = epoch.states
    results = M.commit_partitioned_adaptive_simplex(partitioned, states).local_results
    checks.setdefault("distributed_566", []).append(bool(len(results) == 2))
    issued = (np.asarray(states.mesh.vertex_ids) >= points.shape[0]) & np.asarray(
        states.mesh.vertex_active
    )
    rounded = np.asarray(states.mesh.coordinates)[issued]
    checks.setdefault("distributed_569", []).append(bool(rounded.size))
    for result in results:
        evidence = _require_instance(
            result.target.collective_evidence, CollectiveMeshEvidence
        )
        checks.setdefault("distributed_572", []).append(
            bool(isinstance(evidence, CollectiveMeshEvidence))
        )
        receipts = dict(evidence.logical_arrays)
        valid = np.asarray(receipts["native/midpoint_valid"])
        signs = np.asarray(receipts["native/midpoint_signs"])
        checks.setdefault("distributed_576", []).append(
            bool(np.array_equal(valid, issued))
        )
        checks.setdefault("distributed_577", []).append(bool(np.all(signs[valid] != 0)))
        geometry = result.target.geometry
        checks.setdefault("distributed_579", []).append(
            bool(geometry.restriction_source is not None)
        )
        for element in geometry.elements:
            checks.setdefault("distributed_581", []).append(
                bool(isinstance(element, BarycentricCellGeometryElement))
            )
            weights = np.asarray(
                _require_instance(
                    element, BarycentricCellGeometryElement
                ).barycentric_weights
            )
            checks.setdefault("distributed_583", []).append(
                bool(np.array_equal(np.round(2.0 * weights), 2.0 * weights))
            )
            _record_array(checks, "distributed_584", np.sum(weights, axis=1), 1.0)
        coefficients = np.asarray(geometry.coordinates)
        checks.setdefault("distributed_586", []).append(
            bool(all((np.any(np.all(points == row, axis=1)) for row in coefficients)))
        )
        checks.setdefault("distributed_587", []).append(
            bool(
                not any((np.any(np.all(coefficients == row, axis=1)) for row in rounded))
            )
        )
    slot = np.flatnonzero(issued[0])[0]
    value = states.mesh.coordinates[0, slot, 1]
    coordinates = states.mesh.coordinates.at[0, slot, 1].set(
        jnp.nextafter(value, jnp.inf)
    )
    perturbed = eqx.tree_at(lambda value: value.mesh.coordinates, states, coordinates)
    with pytest.raises(M.MeshingFailure) as refusal_592:
        M.commit_partitioned_adaptive_simplex(partitioned, perturbed)
    error = refusal_592.value
    checks.setdefault("distributed_595", []).append(
        bool(error.category is M.MeshingFailureCategory.QUALITY_REJECTED)
    )
    checks.setdefault("distributed_596", []).append(
        bool(error.stage == "distributed-reference-restriction")
    )
    checks["distributed_refusal_592"].append(True)
    return checks


def native_terminal_status() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source("terminal_status", checks)
    epoch = _prepare_epoch(setup, "terminal_status", checks)
    partitioned = epoch.partitioned
    states = epoch.states
    clocks = states.clocks.at[1, 2].set(int(D.AdaptiveSimplexStatus.CAPACITY_EXCEEDED))
    states = eqx.tree_at(lambda value: value.clocks, states, clocks)
    with pytest.raises(M.MeshingFailure) as refusal_602:
        M.commit_partitioned_adaptive_simplex(partitioned, states)
    error = refusal_602.value
    checks.setdefault("distributed_605", []).append(
        bool(error.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED)
    )
    checks["distributed_refusal_602"].append(True)
    return checks


def native_audit_only() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source("audit_only", checks)
    epoch = _prepare_epoch(setup, "audit_only", checks)
    seed = setup.seed
    partition_policy = epoch.partition_policy
    policy = epoch.policy
    partitioned = epoch.partitioned
    states = epoch.states
    results = M.commit_partitioned_adaptive_simplex(partitioned, states)
    checks.setdefault("distributed_612", []).append(bool(seed.certification is None))
    for result in results.local_results:
        proof = _require_instance(
            result.target.collective_evidence, CollectiveMeshEvidence
        )
        proof.require_passed()
        embedding = _require_present(proof.source_embedding)
        checks.setdefault("distributed_617", []).append(
            bool(embedding is not None and embedding.status == "certified")
        )
        embedding.binding.require(seed.mesh, seed.geometry)
        checks.setdefault("distributed_619", []).append(
            bool(embedding.validity_certificate_id == seed.audit.validity.certificate_id)
        )
        checks.setdefault("distributed_620", []).append(
            bool(proof.source_audit_id == seed.audit.report_id)
        )
        checks.setdefault("distributed_621", []).append(
            bool(
                proof.source_premise_id
                == _require_present(
                    partitioned.source_witness
                ).scientific_certification_id
            )
        )
        checks.setdefault("distributed_622", []).append(
            bool(proof.source_premise_id != seed.audit.report_id)
        )
    views = tuple(
        (_require_present(result.distribution).target for result in results.local_results)
    )
    checks.setdefault("distributed_624", []).append(
        bool(len({view.distribution_id for view in views}) == 1)
    )
    checks.setdefault("distributed_625", []).append(
        bool(len({result.target.result_id for result in results.local_results}) == 1)
    )
    current = results.local_results[0].target
    continued_policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=policy.device_policy,
        distribution=views[0],
        partition_policy=partition_policy,
    )
    continued = prepare_partitioned_mesh_adaptation(
        M.prepare_mesh_adaptation(
            current, M.MarkedMeshAdaptation(), policy=continued_policy
        )
    )
    back = coarsen_adaptive_simplex_parts(
        continued.layout,
        continued.parts,
        continued.states,
        continued.states.mesh.cell_active,
    )
    restored = M.commit_partitioned_adaptive_simplex(continued, back.state)
    checks.setdefault("distributed_642", []).append(
        bool(restored.identity.target_topology_id == seed.mesh.topology_id)
    )
    for result in restored.local_results:
        result.target.audit.require_passed()
    overlap_points = np.asarray(
        ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.25, 0.25), (1.25, 0.25), (0.25, 1.25)),
        dtype=np.float64,
    )
    overlap_cells = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32)
    overlap = M.certify_cell_mesh(
        D.CellMesh.from_triangles(overlap_points, overlap_cells),
        seed.coordinate_contract,
        audit_policy=M.CellMeshAuditPolicy(
            self_intersection=M.CellMeshAuditDisposition.SKIP
        ),
    )
    overlap.audit.require_passed()
    overlap_distribution = M.prepare_mesh_distribution(
        M.MeshPart("overlap", overlap), policy=partition_policy
    )
    overlap_policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=policy.device_policy,
        distribution=overlap_distribution,
        partition_policy=partition_policy,
    )
    before = np.array(overlap.mesh.coordinates, copy=True)
    with pytest.raises(ValueError, match="meshing_source.embedding must certify"):
        M.partition_adaptive_simplex(
            M.prepare_adaptive_simplex(overlap, policy=overlap_policy)
        )
    checks.setdefault("distributed_673", []).append(
        bool(overlap.collective_evidence is None)
    )
    checks.setdefault("distributed_674", []).append(bool(overlap.certification is None))
    _record_array(checks, "distributed_675", overlap.mesh.coordinates, before)
    return checks


def native_algorithm_relabel() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source("algorithm_relabel", checks)
    epoch = _prepare_epoch(setup, "algorithm_relabel", checks)
    source = epoch.source
    distribution = _require_present(epoch.distribution)
    partitioned = epoch.partitioned
    states = epoch.states
    source_result_id = source.result_id
    source_owners = np.asarray(distribution.partition.cell_owner).copy()
    with pytest.raises(M.MeshingFailure) as refusal_679:
        M.commit_partitioned_adaptive_simplex(partitioned, states)
    error = refusal_679.value
    checks.setdefault("distributed_682", []).append(
        bool(error.category is M.MeshingFailureCategory.LINEAGE_FAILED)
    )
    checks["distributed_refusal_679"].append(True)
    checks.setdefault("distributed_685", []).append(
        bool(source.result_id == source_result_id)
    )
    _record_array(
        checks, "distributed_686", distribution.partition.cell_owner, source_owners
    )
    checks.setdefault("distributed_687", []).append(
        bool(distribution.evidence.kind is M.MeshPartitionKind.MORTON)
    )
    return checks


def native_halo_depth() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source("halo_depth", checks)
    epoch = _prepare_epoch(setup, "halo_depth", checks)
    points = setup.points
    partitioned = epoch.partitioned
    states = epoch.states
    widths = (1, 2, 3)
    counts = []
    for width in widths:
        work = expand_partitioned_simplex_neighborhood(
            partitioned.parts,
            states,
            partitioned.states,
            partitioned.source_exterior,
            halo_width=width,
            cell_capacity=64,
            vertex_capacity=64,
        )
        checks.setdefault("distributed_700", []).append(
            bool(not np.any(np.asarray(work.status)))
        )
        valid = np.asarray(work.cell_valid)
        counts.append(np.sum(valid, axis=1))
        ids = np.asarray(work.source_vertex_ids)
        weights = np.asarray(work.source_weights)
        references = points[np.maximum(ids, 0)]
        rebuilt = np.sum(
            np.where((ids >= 0)[..., None], references * weights[..., None], 0), axis=-2
        )
        _record_array(
            checks,
            "distributed_707",
            np.asarray(work.cell_coordinates)[valid],
            rebuilt[valid],
        )
        _record_array(checks, "distributed_708", np.sum(weights, axis=-1)[valid], 1)
    checks.setdefault("distributed_710", []).append(bool(np.any(counts[1] > counts[0])))
    checks.setdefault("distributed_711", []).append(bool(np.all(counts[2] >= counts[1])))
    return checks


def native_halo_overflow() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source("halo_overflow", checks)
    epoch = _prepare_epoch(setup, "halo_overflow", checks)
    partitioned = epoch.partitioned
    work = expand_partitioned_simplex_neighborhood(
        partitioned.parts,
        epoch.states,
        partitioned.states,
        partitioned.source_exterior,
        halo_width=2,
        cell_capacity=64,
        vertex_capacity=3,
    )
    flags = np.asarray(work.status)
    checks.setdefault("distributed_698", []).append(
        bool(np.all(flags & int(D.AdaptiveSimplexStatus.CAPACITY_EXCEEDED)))
    )
    return checks


def native_publication() -> dict[str, Any]:
    return _publication("publication")


def native_coarsen_publication() -> dict[str, Any]:
    return _publication("coarsen_publication")


def native_scientific_receipts() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source("scientific_receipts", checks)
    epoch = _prepare_epoch(setup, "scientific_receipts", checks)
    source_attributes = setup.source_attributes
    source_zones = setup.source_zones
    partitioned = epoch.partitioned
    states = epoch.states
    original_get = jax.device_get
    transfers = []

    def observed_get(x: Any) -> Any:
        if isinstance(x, AdaptiveSimplexState):
            assert x.mesh.cells.ndim == 2, "The commit transferred all part states"
            checks.setdefault("distributed_717", []).append(bool(x.mesh.cells.ndim == 2))
            transfers.append(x.mesh.cells.shape[0])
        return original_get(x)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax, "device_get", observed_get)
        results = M.commit_partitioned_adaptive_simplex(partitioned, states).local_results
    checks.setdefault("distributed_725", []).append(bool(len(results) == 2))
    checks.setdefault("distributed_726", []).append(
        bool(transfers and max(transfers) == partitioned.layout.cell_capacity)
    )
    authored = dict(
        zip(
            np.asarray(source_attributes[0].scope.global_entity_ids),
            np.asarray(source_attributes[0].global_values),
            strict=True,
        )
    )
    for result in results:
        target = result.target
        banks = dict(
            _require_instance(
                target.collective_evidence, CollectiveMeshEvidence
            ).logical_arrays
        )
        count = _require_present(target.mesh.storage).global_entity_counts[2]
        ids = np.asarray(banks["cell_global_ids"])[:count]
        roots = np.asarray(banks["organization/cell/source_ids"])[:count]
        expected = np.asarray(
            [authored[identifier] for identifier in roots], dtype=np.int64
        )
        marker = target.attributes[0]
        _record_array(
            checks, "distributed_740", np.asarray(marker.scope.global_entity_ids), ids
        )
        _record_array(
            checks, "distributed_741", np.asarray(marker.global_values), expected
        )
        local_rows = np.searchsorted(ids, np.asarray(marker.scope.entity_ids))
        _record_array(
            checks, "distributed_743", np.asarray(marker.values), expected[local_rows]
        )
        checks.setdefault("distributed_744", []).append(
            bool(tuple((zone.name for zone in target.zones)) == ("cell:0", "cell:1"))
        )
        for index, zone in enumerate(target.zones):
            source_ids = np.asarray(source_zones[index].scope.global_entity_ids)
            _record_array(
                checks,
                "distributed_747",
                np.asarray(zone.scope.global_entity_ids),
                ids[np.isin(roots, source_ids)],
            )
    return checks


def _publication(scenario: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    setup = _prepare_source(scenario, checks)
    epoch = _prepare_epoch(setup, scenario, checks)
    source = epoch.source
    prepared = epoch.prepared
    partitioned = epoch.partitioned
    states = epoch.states
    original_get = jax.device_get
    transfers = []

    def observed_get(x: Any) -> Any:
        if isinstance(x, AdaptiveSimplexState):
            assert x.mesh.cells.ndim == 2, "The commit transferred all part states"
            checks.setdefault("distributed_717", []).append(bool(x.mesh.cells.ndim == 2))
            transfers.append(x.mesh.cells.shape[0])
        return original_get(x)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax, "device_get", observed_get)
        results = M.commit_partitioned_adaptive_simplex(partitioned, states).local_results
    checks.setdefault("distributed_725", []).append(bool(len(results) == 2))
    checks.setdefault("distributed_726", []).append(
        bool(transfers and max(transfers) == partitioned.layout.cell_capacity)
    )
    single = D.refine_adaptive_simplex(
        prepared.layout,
        prepared.state,
        prepared.cell_marks(np.asarray((0,), dtype=np.int64)),
    )
    if scenario == "coarsen_publication":
        single = D.coarsen_adaptive_simplex(
            prepared.layout, single.state, single.state.mesh.cell_active
        )
    reference = M.commit_adaptive_simplex(prepared, single.state)
    for result in results:
        target = result.target
        storage = _require_present(target.mesh.storage)
        checks.setdefault("distributed_760", []).append(bool(storage is not None))
        checks.setdefault("distributed_761", []).append(
            bool(target.mesh.topology_id == reference.target.mesh.topology_id)
        )
        checks.setdefault("distributed_762", []).append(
            bool(target.mesh.geometry_id == reference.target.mesh.geometry_id)
        )
        checks.setdefault("distributed_763", []).append(
            bool(
                storage.global_entity_counts
                == ((4, 5, 2) if scenario == "coarsen_publication" else (5, 8, 4))
            )
        )
        checks.setdefault("distributed_764", []).append(
            bool(
                _require_instance(
                    target.collective_evidence, CollectiveMeshEvidence
                ).evidence_id
                == storage.evidence_id
            )
        )
        checks.setdefault("distributed_765", []).append(
            bool(
                np.all(
                    np.asarray(
                        _require_instance(
                            target.collective_evidence, CollectiveMeshEvidence
                        ).global_checks
                    )
                )
            )
        )
        checks.setdefault("distributed_766", []).append(
            bool(_require_present(result.distribution) is not None)
        )
        checks.setdefault("distributed_767", []).append(
            bool(not _require_present(result.distribution).rebalanced)
        )
        checks.setdefault("distributed_768", []).append(
            bool(np.all(np.asarray(storage.local_neighborhood_complete)))
        )
        checks.setdefault("distributed_769", []).append(
            bool(
                _require_present(result.transfer).preserves_linear
                and _require_present(result.transfer).preserves_constants
            )
        )
        checks.setdefault("distributed_770", []).append(
            bool(not _require_present(result.transfer).conservative)
        )
        original = 1 + source.mesh.coordinates[:, 0] + 2 * source.mesh.coordinates[:, 1]
        transferred = _require_present(result.transfer).apply(original)
        _record_array(
            checks,
            "distributed_773",
            transferred,
            1 + target.mesh.coordinates[:, 0] + 2 * target.mesh.coordinates[:, 1],
        )
        owned_cells = np.asarray(storage.entity_global_ids[2])[
            np.asarray(storage.entity_owned[2])
        ]
        _record_array(
            checks,
            "distributed_775",
            np.asarray(
                _require_instance(result.hierarchy, M.BisectionHierarchy).cell_global_ids
            ),
            np.sort(owned_cells),
        )
        artifact = (
            _require_instance(result.hierarchy, M.BisectionHierarchy),
            _require_present(result.lineage),
        )
        provenance = ArrayArtifactProvenance(
            "native-bisection",
            (source.result_id, result.target.result_id),
            (storage.evidence_id,),
            (source.coordinate_contract.spatial_id,),
        )
        structure_ids = {
            "topology": target.mesh.topology_id,
            "coverage": storage.evidence_id,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owner-lineage.zip"
            written = write_typed_array_artifact(
                path,
                artifact,
                artifact_kind="native-bisection-lineage",
                provenance=provenance,
                structure_ids=structure_ids,
            )
            template = jax.tree_util.tree_map(jnp.zeros_like, artifact)
            restored, receipt = read_typed_array_artifact(
                path,
                template,
                artifact_kind="native-bisection-lineage",
                provenance=provenance,
                structure_ids=structure_ids,
            )
            checks.setdefault("distributed_789", []).append(
                bool(receipt.array_digest == written.array_digest)
            )
            _record_array(
                checks,
                "distributed_790",
                restored[0].cell_global_ids,
                np.sort(owned_cells),
            )
            _record_array(
                checks,
                "distributed_791",
                restored[0].record_child_ids,
                _require_instance(
                    result.hierarchy, M.BisectionHierarchy
                ).record_child_ids,
            )
            _record_array(
                checks,
                "distributed_792",
                restored[1].entity_lineage(2).source_global_ids,
                _require_present(result.lineage).entity_lineage(2).source_global_ids,
            )
        with pytest.raises(
            ValueError, match=r"CellMesh\.storage must be explicitly materialized"
        ):
            target.mesh.require_dense("implicit serial consumer")
        checks["distributed_refusal_794"].append(True)
    snapshots = snapshot_addressable_arrays(
        {"arrays": dict(_require_present(results[0].target.mesh.storage).logical_arrays)}
    )
    cell_path = jax.tree_util.keystr(
        (jax.tree_util.DictKey("arrays"), jax.tree_util.DictKey("epoch/cell_ids"))
    )
    cell_shards = [shard for shard in snapshots if shard.array_path == cell_path]
    checks.setdefault("distributed_804", []).append(bool(len(cell_shards) == 2))
    for shard in cell_shards:
        rank = shard.index[0][0]
        held = np.asarray(
            _require_instance(
                results[rank].hierarchy, M.BisectionHierarchy
            ).cell_global_ids
        )
        checks.setdefault("distributed_808", []).append(
            bool(np.all(np.isin(held, shard.payload)))
        )

    def halos_for(degree: int) -> tuple[Any, ...]:
        meshes = tuple((result.target.mesh for result in results))
        capacity = max(
            (
                _require_present(mesh.storage).entity_global_ids[degree].size
                for mesh in meshes
            )
        )
        return prepare_owner_local_halo_plans(
            meshes,
            degree,
            devices=tuple(jax.devices()),
            axis_name="parts",
            message_capacity=capacity,
        )

    vertex_halos = halos_for(0)
    programs = tuple(
        (
            prepare_owner_local_finite_element(
                FiniteElementPlan(
                    result.target.mesh,
                    FiniteElementFieldSpec("u", lagrange_element("triangle", 1)),
                ),
                halo,
                axis_name="parts",
            )
            for result, halo in zip(results, vertex_halos, strict=True)
        )
    )

    def execution_fields(program: Any) -> Any:
        return (
            program.mass,
            program.stiffness,
            program.pairing,
            program.halo,
            program.cell_owned,
            program.collective_metadata,
            program.global_ids,
            program.owned_mask,
            program.local_discretization.mesh.coordinates,
        )

    numeric, static = eqx.partition(execution_fields(programs[0]), eqx.is_array)
    _, structure = jax.tree_util.tree_flatten(numeric)
    leaves = [
        jax.tree_util.tree_leaves(eqx.filter(execution_fields(program), eqx.is_array))
        for program in programs
    ]
    packed = tuple((jnp.stack(values) for values in zip(*leaves, strict=True)))

    def consume(local_leaves: Any) -> Any:
        fields = eqx.combine(
            jax.tree_util.tree_unflatten(structure, local_leaves), static
        )
        program = eqx.tree_at(execution_fields, programs[0], fields)
        coordinates = program.local_discretization.mesh.coordinates
        affine = 1 + coordinates[:, 0] + 2 * coordinates[:, 1]
        owned = jnp.where(program.owned_mask, affine, 0)
        mass = program.operator("mass")
        solved = program.solve_mass(
            mass.mv(owned), DistributedKrylovPolicy(16, relative_tolerance=1e-10)
        )
        return (
            program.operator("stiffness").mv(jnp.where(program.owned_mask, 1.0, 0.0)),
            solved.solve.value,
            solved.accepted,
        )

    constant, values, accepted = jax.pmap(
        consume, axis_name="parts", devices=jax.devices()
    )(packed)
    _record_array(checks, "distributed_841", constant, 0)
    _record_array(checks, "distributed_842", accepted, True)
    for rank, program in enumerate(programs):
        coordinates = program.local_discretization.mesh.coordinates
        _record_array(
            checks,
            "distributed_845",
            values[rank],
            jnp.where(
                program.owned_mask, 1 + coordinates[:, 0] + 2 * coordinates[:, 1], 0
            ),
        )
    system = phx.equations.EulerSystem(2)
    cell_halos = halos_for(2)
    dynamics = []
    for result in results:
        storage = _require_present(result.target.mesh.storage)
        geometry = D.UnstructuredFiniteVolumePlan.from_cell_mesh(
            result.target.mesh,
            component_names=system.component_names,
            neighborhood_complete=storage.local_neighborhood_complete,
            physical_boundary_faces=storage.local_physical_boundary_facets,
            closure_evidence_id=storage.evidence_id,
        ).prepare(owned_face_capacity=8)
        boundaries = D.UnstructuredFiniteVolumeBoundarySet(
            geometry.boundary_patch_names,
            {name: D.SlipWallBoundary() for name in geometry.boundary_patch_names},
        )
        method = D.UnstructuredFiniteVolumeMethodPlan(
            D.PiecewiseConstantReconstruction(), D.RusanovFluxPlan()
        )
        dynamics.append(
            PreparedUnstructuredFiniteVolumeDynamics(system, geometry, method, boundaries)
        )
    initial = []
    for program in dynamics:
        centroids = program.discretization.cell_centers
        primitive = jnp.stack(
            (
                1 + 0.25 * centroids[:, 0],
                jnp.zeros(centroids.shape[0]),
                jnp.zeros(centroids.shape[0]),
                jnp.ones(centroids.shape[0]),
            ),
            axis=1,
        )
        initial.append(system.primitive_to_conserved(primitive))
    seeds = jnp.stack(
        tuple(
            (
                jnp.where(
                    _require_instance(
                        program.discretization, D.UnstructuredFiniteVolumeDiscretization
                    ).cell_owned[:, None],
                    value,
                    2 * value,
                )
                for program, value in zip(dynamics, initial, strict=True)
            )
        )
    )
    devices = tuple(jax.devices())
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "accepted-owner-local-fv",
            tuple(sorted({device.process_index for device in devices})),
            tuple(((device.process_index, device.id) for device in devices)),
            mesh_axes=(("parts", len(devices)),),
        ),
        devices,
    )
    advance = execute_owner_local_finite_volume_advance(
        tuple(zip(dynamics, cell_halos, strict=True)),
        seeds,
        jnp.asarray(0.0),
        jnp.asarray(0.01),
        group,
        axis_name="parts",
    )
    updated = advance.state
    for rank, program in enumerate(dynamics):
        owned = _require_instance(
            program.discretization, D.UnstructuredFiniteVolumeDiscretization
        ).cell_owned
        _record_array(
            checks, "distributed_895", updated[rank][~owned], seeds[rank][~owned]
        )
    _record_array(
        checks, "distributed_896", advance.inventory_after[0], advance.inventory_before[0]
    )
    checks.setdefault("distributed_897", []).append(
        bool(bool(advance.finite) and bool(advance.admissible))
    )
    checks.setdefault("distributed_898", []).append(
        bool(
            np.any(
                np.asarray(
                    updated[0][
                        _require_instance(
                            dynamics[0].discretization,
                            D.UnstructuredFiniteVolumeDiscretization,
                        ).cell_owned,
                        0,
                    ]
                )
                != np.asarray(
                    initial[0][
                        _require_instance(
                            dynamics[0].discretization,
                            D.UnstructuredFiniteVolumeDiscretization,
                        ).cell_owned,
                        0,
                    ]
                )
            )
        )
    )
    L = phx.lifecycle
    result = results[0]
    topology = L.CompositionEntry(
        source,
        entry_id="mesh",
        role="topology",
        owner_id="meshing",
        structure_id=source.mesh.topology_id,
        revision_id=source.result_id,
        semantics_id="unit-square",
    )
    old_values = 1 + source.mesh.coordinates[:, 0] + 2 * source.mesh.coordinates[:, 1]
    field = L.CompositionEntry(
        old_values,
        entry_id="u",
        role="physical-state",
        owner_id="fe",
        structure_id=source.mesh.topology_id,
        revision_id="source-field",
        semantics_id="affine-u",
        dependencies=(topology.binding("structure"),),
    )
    composition = L.Composition((topology, field), boundary_id="accepted-step")
    successor = L.CompositionEntry(
        result.target,
        entry_id="mesh",
        role="topology",
        owner_id="meshing",
        structure_id=result.target.mesh.topology_id,
        revision_id=result.target.result_id,
        semantics_id="unit-square",
    )
    target_values = _require_present(result.transfer).apply(old_values)
    target_field = L.CompositionEntry(
        target_values,
        entry_id="u",
        role="physical-state",
        owner_id="fe",
        structure_id=result.target.mesh.topology_id,
        revision_id=_require_present(result.transfer).transfer_id,
        semantics_id="affine-u",
        dependencies=(successor.binding("structure"),),
    )
    transport = L.CompositionTransport(
        "physical-remap",
        ("u",),
        (target_field,),
        source_structure_ids=(field.structure_id,),
        route_id=_require_present(result.transfer).transfer_id,
        successful=jnp.all(
            _require_instance(
                result.target.collective_evidence, CollectiveMeshEvidence
            ).global_checks
        ),
    )
    rebind = L.CompositionRebind(
        composition, reprepare=(successor,), transports=(transport,)
    )
    rejected = L.commit_composition_rebind(rebind, accepted_boundary=False)
    checks.setdefault("distributed_913", []).append(
        bool(
            not rejected.published
            and rejected.composition.composition_id == composition.composition_id
        )
    )
    receipt = L.commit_composition_rebind(rebind, accepted_boundary=True)
    checks.setdefault("distributed_915", []).append(bool(receipt.published))
    _record_array(
        checks, "distributed_916", receipt.composition.entry("u").value, target_values
    )
    return checks
