# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Distributed initial-generation scenarios with per-rank evidence records."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils

import phydrax as phx
import phydrax.meshing._audit as audit_module
import phydrax.meshing._distributed_generation as generation_module
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.discretization import CellGeometrySpec, CellMesh, CellValidityCertificate
from phydrax.discretization._cell_geometry import CellGeometryStorageProjection
from phydrax.geometry._meshing_domain import MeshingDomain, MeshingSurfacePatch
from phydrax.geometry.brep._patches import PlanePatch
from phydrax.geometry.surface import SurfaceModel
from phydrax.meshing import (
    CellMeshAuditPolicy,
    CellMeshAuditReport,
    CellQualityEvaluation,
    GeometryAssociation,
    MeshAttribute,
    MeshLabel,
    MeshPatch,
    MeshZone,
)
from phydrax.meshing._device_adaptation import (
    repartition_partitioned_mesh_adaptation,
)
from phydrax.meshing._distributed_generation import (
    DistributedSurfaceConstruction,
    execute_distributed_surface_initial,
    PreparedDistributedSurfaceGeneration,
)
from phydrax.meshing._distributed_planar_generation import (
    PlanarTileSchedule,
    prepare_distributed_planar_generation,
)
from phydrax.meshing._distribution import (
    prepare_partitioned_rebalance_proposal,
    replay_partitioned_rebalance_proposal,
)
from phydrax.meshing._distribution_migration import (
    accept_ownerlocal_migration,
    PartitionedEpochMigration,
    prepare_ownerlocal_migration_state,
)
from phydrax.meshing._initial_certification import InitialCollectiveMeshEvidence
from phydrax.meshing._result import (
    CollectiveMeshStorageBinding,
    require_original_meshing_source,
)
from phydrax.meshing.providers._native_planar_domain import (
    source_arrays as planar_source_arrays,
)
from phydrax.meshing.qualification import folded_plates_surface


M = phx.meshing
_ORIGINAL_AUDIT = audit_module.audit_cell_mesh


@contextmanager
def _distributed() -> Iterator[None]:
    # `import phydrax` initializes the runtime from the runner's PHYDRAX_*
    # variables before any backend use; the scenario owns its shutdown.
    try:
        yield
    finally:
        jax.distributed.shutdown()


def _prepare(
    owners: tuple[int, int],
    *,
    small_remote: bool = False,
    overlap: bool = False,
    organization: bool = False,
) -> PreparedDistributedSurfaceGeneration:
    domain = folded_plates_surface().domain
    if overlap:
        second = MeshingSurfacePatch(
            PlanePatch((0, 0, 0), (-1, 0, 0), (0, 1, 0)),
            domain.patches[1].loops,
        )
        domain = MeshingDomain(
            (domain.patches[0], second),
            domain.curves,
            domain.corner_points.shape[0],
            source_id="overlapping-plates",
            source_revision="r1",
        )
    scope = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.arange(len(domain.patches), dtype=np.int64),
    )
    region_controls = ()
    patch_controls = ()
    if organization:
        selected = tuple(
            M.MeshingScope(
                domain.source_id,
                domain.source_revision,
                M.MeshingEntityKind.GEOMETRY,
                2,
                domain.entity_set_id(2),
                np.asarray((index,), dtype=np.int64),
            )
            for index in range(2)
        )
        region_controls = tuple(
            M.RegionControl(
                selected[index],
                name,
                f"material-{name}",
                M.RegionRole.USER,
            )
            for index, name in enumerate(("left", "right"))
        )
        edge_scope = M.MeshingScope(
            domain.source_id,
            domain.source_revision,
            M.MeshingEntityKind.GEOMETRY,
            1,
            domain.entity_set_id(1),
            np.asarray((0,), dtype=np.int64),
        )
        patch_controls = (
            M.PatchControl("shared-interface", edge_scope, ("left", "right")),
        )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            M.UniformSizeControl(scope, 0.35, strength=M.SizeControlStrength.SOFT),
        ),
        region_controls=region_controls,
        patch_controls=patch_controls,
        quality_target=M.MeshQualityTarget(minimum_angle=0.25, hard=True),
    )
    small = small_remote and jax.process_index() == 1
    layout = M.DeviceGenerationLayout(
        vertex_capacity=8 if small else 1000,
        cell_capacity=16 if small else 2000,
        candidate_capacity=8 if small else 64,
        maximum_work_units=2_000_000,
    )
    schedule = M.NativeSurfaceSchedule(quality_angle_degrees=float(np.rad2deg(0.25)))
    return M.prepare_distributed_surface_generation(
        M.NativeSurfaceSource(domain),
        specification,
        schedule,
        layout,
        np.asarray(owners, dtype=np.int32),
        maximum_metadata_bytes=10_000,
    )


def _construction_record(
    generated: DistributedSurfaceConstruction, owners: tuple[int, int]
) -> dict[str, Any]:
    construction = generated.construction
    if construction is None:
        return {
            "empty": True,
            "rank": jax.process_index(),
            "vertex_count": generated.vertex_ids.size,
            "cell_count": generated.cell_ids.size,
        }
    corners = construction.vertices[construction.triangles]
    area = np.sum(
        np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
        )
        / 2
    )
    shared_rows = construction.curve_edges[construction.curve_edge_curves == 0]
    shared = np.unique(shared_rows)
    return {
        "empty": False,
        "triangle_patches": sorted(set(construction.triangle_patches.tolist())),
        "expected_patches": np.flatnonzero(
            np.asarray(owners) == jax.process_index()
        ).tolist(),
        "unresolved": np.asarray(construction.unresolved).tolist(),
        "shared_x": construction.vertices[shared, 0].tolist(),
        "shared_z": construction.vertices[shared, 2].tolist(),
        "shared_length": float(
            np.sum(
                np.linalg.norm(
                    np.diff(construction.vertices[shared_rows], axis=1)[:, 0], axis=1
                )
            )
        ),
        "neighbor_packet_count": generated.neighbor_packet_count,
        "neighbor_transferred_bytes": generated.neighbor_transferred_bytes,
        "shared_curve_ids": generated.shared_curve_ids.tolist(),
        "area": float(area),
        "local_cells": construction.triangles.shape[0],
        "global_cells": generated.global_cell_count,
        "shared_ids": generated.vertex_ids[shared].tolist(),
        "shared_points": construction.vertices[shared].tolist(),
    }


def _publish(prepared: PreparedDistributedSurfaceGeneration) -> M.CellMeshingResult:
    provider = M.NativeMeshingProvider(
        M.NativeMeshingOptions(
            "parametric_surface",
            surface_schedule=prepared.schedule,
        )
    )
    return provider.plan(
        M.NativeSurfaceSource(prepared.compiled.domain),
        prepared.specification,
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.METER),
        initial_partition=prepared,
    ).execute()


def _failure_record(prepared: PreparedDistributedSurfaceGeneration) -> dict[str, Any]:
    try:
        _publish(prepared)
    except M.MeshingFailure as error:
        findings = dict(error.evidence.logical_findings)
        report: dict[str, Any] = {
            "rejected": True,
            "failure_id": error.evidence.evidence_id,
        }
        if "initial/failed_owner_category" in findings:
            raw = findings["initial/failed_owner_category"]
            report["category_fully_addressable"] = raw.is_fully_addressable
            report["local_category"] = int(np.asarray(raw.addressable_shards[0].data)[0])
        if "initial/cross_owner_findings" in findings:
            raw = findings["initial/cross_owner_findings"]
            report["intersection_fully_addressable"] = raw.is_fully_addressable
            records = np.asarray(raw.addressable_shards[0].data)
            report["native_intersection_count"] = int(np.sum(records[:, 2] == 1))
        return report
    return {"rejected": False}


def _consumer_record(
    target: M.CellMeshingResult,
    prepared: PreparedDistributedSurfaceGeneration,
    *,
    organization: bool,
) -> dict[str, Any]:
    domain = prepared.compiled.domain
    provider = M.NativeMeshingProvider(
        M.NativeMeshingOptions("parametric_surface", surface_schedule=prepared.schedule)
    )
    evidence = require_original_meshing_source(target)
    if not isinstance(evidence, InitialCollectiveMeshEvidence):
        raise AssertionError(
            "device_generation.evidence must retain collective source evidence; got an incompatible owner"
        )
    storage = target.mesh.storage
    if storage is None:
        raise AssertionError(
            "device_generation.storage must retain logical storage; got None"
        )
    organization_rows, membership_rows = [], []
    locally_empty = 0
    for degree, records, association in (
        (
            2,
            tuple(patch for patch in target.patches if patch.scope.entity_dimension == 2),
            target.associations[0],
        ),
        (1, target.labels, target.associations[1]),
    ):
        resident = np.asarray(target.mesh.entity_set(degree).entity_ids)
        owned_ids = resident[np.asarray(storage.entity_owned[degree])]
        for record in records:
            source_index = int(record.name.split(":")[1])
            source_id = domain.entity_id(degree, source_index)
            selected = np.asarray(association.target_global_ids)[
                np.asarray(
                    [
                        source == source_id
                        for source in association.source_entities.entity_ids()
                    ],
                    dtype=np.bool_,
                )
            ]
            locally_empty += record.scope.entity_ids.size == 0
            own_members = np.intersect1d(selected, owned_ids)
            packet = np.full((prepared.layout.cell_capacity,), -1, dtype=np.int64)
            packet[: own_members.size] = own_members
            gathered = np.asarray(multihost_utils.process_allgather(packet, tiled=False))
            reference_ids = np.unique(gathered[gathered >= 0])
            membership_rows.append(
                {
                    "local_ids": np.asarray(record.scope.entity_ids).tolist(),
                    "selected_ids": np.sort(selected).tolist(),
                    "global_shape": list(record.scope.global_entity_ids.shape),
                    "reference_shape": list(reference_ids.shape),
                    "global_members_match": bool(
                        jnp.all(
                            record.scope.global_entity_ids == jnp.asarray(reference_ids)
                        )
                    ),
                }
            )
            organization_rows.append(
                (record.name, record.scope.scope_id, reference_ids.size)
            )
    zones, zone_memberships, interface_ids, expected_interface_ids = [], [], [], []
    if organization:
        face_association = target.associations[0]
        for zone in target.zones:
            index = 0 if zone.name == "left" else 1
            selected = np.asarray(face_association.target_global_ids)[
                np.asarray(
                    [
                        source == domain.entity_id(2, index)
                        for source in face_association.source_entities.entity_ids()
                    ],
                    dtype=np.bool_,
                )
            ]
            zone_memberships.append(
                {
                    "local_ids": np.asarray(zone.scope.entity_ids).tolist(),
                    "selected_ids": np.sort(selected).tolist(),
                }
            )
            zones.append(
                (
                    zone.name,
                    zone.material_id,
                    zone.region_role.value if zone.region_role is not None else None,
                    zone.zone_id,
                )
            )
        interface = next(
            patch for patch in target.patches if patch.name == "shared-interface"
        )
        interface_ids = sorted(set(interface.adjacent_zone_ids))
        expected_interface_ids = sorted({zone.zone_id for zone in target.zones})
    D = phx.discretization
    fe = D.FiniteElementPlan(
        target.mesh, D.FiniteElementFieldSpec("u", D.lagrange_element("triangle", 1))
    ).prepare()
    ones = jnp.ones((fe.dof_maps[0].global_dof_count,), dtype=jnp.float64)
    points = fe.dof_maps[0].dof_coordinates
    expected = 1.0 + points[:, 0] + 2.0 * points[:, 1] - points[:, 2]
    solved = phx.linalg.solve(
        phx.linalg.LinearSystem(fe.mass),
        fe.mass.mv(expected),
        policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.PCG(),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1.0e-12, absolute=1.0e-14, max_steps=128
            ),
        ),
    )
    area = jnp.sum(
        jnp.where(storage.entity_owned[2], fe.block_geometries[0][0].measure, 0.0)
    )
    return {
        "target_kind": isinstance(target, M.CellMeshingResult),
        "provider_id": target.provider.provider_id,
        "expected_provider_id": provider.info().provider_id,
        "collective_source_kind": True,
        "domain_id": evidence.premise.source_domain_id,
        "expected_domain_id": domain.domain_id,
        "specification_id": evidence.premise.specification.specification_id,
        "expected_specification_id": prepared.specification.specification_id,
        "epoch_arrays": [
            name for name, _ in evidence.logical_arrays if name.startswith("epoch/")
        ],
        "patch_count": len(target.patches),
        "expected_patch_count": len(domain.patches) + organization,
        "label_count": len(target.labels),
        "expected_label_count": len(domain.curves),
        "memberships": membership_rows,
        "zone_memberships": zone_memberships,
        "interface_ids": interface_ids,
        "expected_interface_ids": expected_interface_ids,
        "stiffness_ones": np.asarray(fe.stiffness.mv(ones)).tolist(),
        "solve_successful": bool(jnp.all(solved.successful)),
        "solved": np.asarray(solved.value).tolist(),
        "expected_solution": np.asarray(expected).tolist(),
        "result_id": target.result_id,
        "owned_area": float(area),
        "solve_error": float(jnp.max(jnp.abs(solved.value - expected))),
        "organization": organization_rows,
        "locally_empty_scopes": locally_empty,
        "zones": zones,
    }


def _empty_owner_record(target: M.CellMeshingResult) -> dict[str, Any]:
    empty = target.mesh.entity_set(2).count == 0
    storage = target.mesh.storage
    if storage is None:
        raise AssertionError(
            "device_generation.storage must retain logical storage on the empty owner; got None"
        )
    report: dict[str, Any] = {
        "target_kind": isinstance(target, M.CellMeshingResult),
        "storage_present": target.mesh.storage is not None,
        "rank": jax.process_index(),
        "empty": empty,
        "result_id": target.result_id,
        "global_cells": storage.global_entity_counts[2],
    }
    if empty:
        target.audit.require_passed()
        report.update(
            coordinate_shape=list(target.mesh.coordinates.shape),
            block_count=len(target.mesh.blocks),
            geometry_shape=list(target.geometry.coordinates.shape),
            entity_counts=[
                entities.count for entities in target.mesh.topology.entity_sets
            ],
            scope_local_counts=[
                record.scope.entity_ids.size
                for record in (*target.patches, *target.labels)
            ],
            scope_global_counts=[
                record.scope.global_entity_ids.size
                for record in (*target.patches, *target.labels)
            ],
            audit_passed=True,
            owned_area=0.0,
        )
    else:
        D = phx.discretization
        fe = D.FiniteElementPlan(
            target.mesh, D.FiniteElementFieldSpec("u", D.lagrange_element("triangle", 1))
        ).prepare()
        report["owned_area"] = float(jnp.sum(fe.block_geometries[0][0].measure))
    multihost_utils.sync_global_devices("initial-empty-consumer")
    return report


def _storage_record(target: M.CellMeshingResult) -> dict[str, Any]:
    evidence, storage = target.collective_evidence, target.mesh.storage
    if evidence is None or storage is None:
        raise AssertionError(
            "device_generation.publication must retain collective evidence and logical storage; got a missing owner"
        )
    projection = storage.geometry_projection
    if projection is None:
        raise AssertionError(
            "device_generation.geometry_projection must retain coefficient ownership; got None"
        )
    copied = tuple((name, value.copy()) for name, value in storage.logical_arrays)
    restored_binding = CollectiveMeshStorageBinding(
        evidence, copied, storage.global_entity_counts
    )
    copied_projection = CellGeometryStorageProjection(
        copied,
        storage.logical_coordinate_geometry_id,
        storage.global_coordinate_count,
        source_elements=dict(projection.source_elements),
    )
    copied_storage = eqx.tree_at(
        lambda value: (value.logical_arrays, value.geometry_projection),
        storage,
        (copied, copied_projection),
    )
    copied_mesh = eqx.tree_at(lambda value: value.storage, target.mesh, copied_storage)
    restored_binding.require_storage(copied_mesh, evidence)
    copied_geometry = copied_storage.restore_geometry()
    copied_geometry.resolve(copied_mesh)
    forged = dict(storage.logical_arrays)
    forged["geometry/coordinates"] = forged["geometry/coordinates"].at[0, 0].add(0.125)
    forged_arrays = tuple(sorted(forged.items()))
    forged_projection = CellGeometryStorageProjection(
        forged_arrays,
        storage.logical_coordinate_geometry_id,
        storage.global_coordinate_count,
        source_elements=dict(projection.source_elements),
    )
    rejected = False
    try:
        CollectiveMeshStorageBinding(
            evidence, forged_arrays, storage.global_entity_counts
        )
    except ValueError:
        rejected = True
    local_rejected = False
    if jax.process_index() == 1:
        forged_storage = eqx.tree_at(
            lambda value: (value.logical_arrays, value.geometry_projection),
            storage,
            (forged_arrays, forged_projection),
        )
        forged_mesh = eqx.tree_at(
            lambda value: value.storage, target.mesh, forged_storage
        )
        forged_geometry = forged_storage.restore_geometry()
        scopes = []
        for record in (*target.patches, *target.zones, *target.labels):
            scope_projection = record.scope.projection
            if scope_projection is None:
                raise AssertionError(
                    "device_generation.scope_projection must retain logical membership; got None"
                )
            scopes.append((scope_projection.membership_name, scope_projection))
        try:
            M.CellMeshingResult(
                forged_mesh,
                forged_geometry,
                target.coordinate_contract,
                target.audit,
                target.quality,
                target.compliance,
                target.trace,
                target.provider,
                target.runtime,
                target.derivative_mode,
                target.provenance,
                patches=target.patches,
                zones=target.zones,
                labels=target.labels,
                collective_evidence=evidence,
                scope_projections=tuple(scopes),
                storage_binding=target.storage_binding,
            )
        except ValueError:
            local_rejected = True
    multihost_utils.sync_global_devices("initial-storage-security")
    return {
        "forged_bank_rejected": rejected,
        "passive_constructor_rejected": local_rejected,
        "independent_buffer_admitted": True,
        "local_cell_count": target.mesh.entity_set(2).count,
        "copied_coordinates": np.asarray(copied_geometry.coordinates).tolist(),
        "target_coordinates": np.asarray(target.geometry.coordinates).tolist(),
    }


def _strict_audit(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    quality: CellQualityEvaluation | None = None,
    /,
    *,
    policy: CellMeshAuditPolicy | None = None,
    prepared_validity: CellValidityCertificate | None = None,
    boundary: SurfaceModel | None = None,
    patches: tuple[MeshPatch, ...] = (),
    associations: tuple[GeometryAssociation, ...] = (),
    attributes: tuple[MeshAttribute, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
) -> CellMeshAuditReport:
    policy = CellMeshAuditPolicy(
        require_complete_association=True, minimum_mean_ratio=0.999999
    )
    return _ORIGINAL_AUDIT(
        mesh,
        geometry,
        quality,
        policy=policy,
        prepared_validity=prepared_validity,
        boundary=boundary,
        patches=patches,
        associations=associations,
        attributes=attributes,
        zones=zones,
        labels=labels,
    )


def _corrupt_source(
    prepared: PreparedDistributedSurfaceGeneration,
    /,
    *,
    minimum_angle: float,
    maximum_normal_angle: float = np.inf,
) -> DistributedSurfaceConstruction:
    generated = execute_distributed_surface_initial(
        prepared, minimum_angle=minimum_angle, maximum_normal_angle=maximum_normal_angle
    )
    original = generated.construction
    if original is None:
        raise AssertionError(
            "device_generation.fault_source must retain the remote construction; got None"
        )
    points = original.vertices.copy()
    points[np.flatnonzero(original.vertex_source_dimensions == 2)[0], 0] += 0.125
    return replace(
        generated,
        construction=eqx.tree_at(lambda value: value.vertices, original, points),
    )


def boundary_projection_record(
    owners: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """Compare closure publication with each local source curve's parameter bank."""
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.meshing._device_adaptation import device_coordinate_budget
    from phydrax.meshing._distributed_generation import _prepare_boundary
    from phydrax.meshing._surface_generation import curve_vertices
    from phydrax.meshing.providers._native_curve import QueryLedger

    prepared = _prepare(owners)
    limits = prepared.specification.limits
    with (
        NativeExecutionScope(limits.native_limits()) as scope,
        device_coordinate_budget(limits, scope=scope) as budget,
    ):
        local, local_limits, boundary, _ = _prepare_boundary(
            prepared, np.inf, scope=scope, budget=budget
        )
    if local is None:
        raise AssertionError(
            "boundary_projection.local must have source patches; got empty"
        )
    with (
        NativeExecutionScope(limits.native_limits()) as scope,
        device_coordinate_budget(limits, scope=scope) as budget,
    ):
        ledger = QueryLedger(local_limits, budget=budget)
        reference = curve_vertices(
            local,
            prepared.schedule,
            ledger,
            scope.operation_started,
            np.inf,
            scope=scope,
            budget=budget,
        )
    np.testing.assert_array_equal(boundary[0], reference[0])
    for bank in (1, 2):
        if set(boundary[bank]) != set(reference[bank]):
            raise AssertionError(
                "boundary_projection.curves must retain source keys; got mismatch"
            )
        for curve in reference[bank]:
            np.testing.assert_array_equal(boundary[bank][curve], reference[bank][curve])
    if boundary[3] != reference[3]:
        raise AssertionError(
            "boundary_projection.corners must retain source IDs; got mismatch"
        )
    return {"identical": True, "boundary_vertices": boundary[0].shape[0]}


def foreign_curve_boundary() -> dict[str, Any]:
    with _distributed():
        return boundary_projection_record((0, 1))


def shared_constraint() -> dict[str, Any]:
    with _distributed():
        prepared = _prepare((0, 1))
        generated = execute_distributed_surface_initial(prepared, minimum_angle=0.25)
        return _construction_record(generated, (0, 1))


def empty_owner() -> dict[str, Any]:
    with _distributed():
        prepared = _prepare((0, 0))
        generated = execute_distributed_surface_initial(prepared, minimum_angle=0.25)
        return _construction_record(generated, (0, 0))


def one_owner_overflow() -> dict[str, Any]:
    with _distributed():
        prepared = _prepare((0, 1), small_remote=True)
        try:
            execute_distributed_surface_initial(prepared, minimum_angle=0.25)
        except M.MeshingFailure as error:
            return {"rejected": True, "category": error.category.value}
        return {"rejected": False}


def accepted_consumer() -> dict[str, Any]:
    with _distributed():
        prepared = _prepare((0, 1))
        return _consumer_record(_publish(prepared), prepared, organization=False)


def accepted_organization() -> dict[str, Any]:
    with _distributed():
        prepared = _prepare((0, 1), organization=True)
        return _consumer_record(_publish(prepared), prepared, organization=True)


def accepted_empty_owner() -> dict[str, Any]:
    with _distributed():
        prepared = _prepare((0, 0))
        target = _publish(prepared)
        report = _empty_owner_record(target)
        provider = M.NativeMeshingProvider(
            M.NativeMeshingOptions(
                "parametric_surface", surface_schedule=prepared.schedule
            )
        )
        report.update(
            provider_id=target.provider.provider_id,
            expected_provider_id=provider.info().provider_id,
        )
        return report


def storage_bank_forgery() -> dict[str, Any]:
    with _distributed():
        return _storage_record(_publish(_prepare((0, 0))))


def remote_source_failure() -> dict[str, Any]:
    with _distributed(), pytest.MonkeyPatch.context() as monkeypatch:
        prepared = _prepare((0, 1))
        if jax.process_index() == 1:
            monkeypatch.setattr(
                generation_module, "execute_distributed_surface_initial", _corrupt_source
            )
        return _failure_record(prepared)


def physical_overlap() -> dict[str, Any]:
    with _distributed():
        return _failure_record(_prepare((0, 1), overlap=True))


def local_audit_failure() -> dict[str, Any]:
    with _distributed(), pytest.MonkeyPatch.context() as monkeypatch:
        prepared = _prepare((0, 1))
        if jax.process_index() == 1:
            monkeypatch.setattr(audit_module, "audit_cell_mesh", _strict_audit)
        return _failure_record(prepared)


def _initial_graph_vertex_fields(ids: jax.Array) -> jax.Array:
    values = ids.astype(jnp.float64)
    return jnp.stack(
        (
            jnp.where(ids % 7 == 0, -0.0, 273.0 + values / 8),
            -values / 16,
            1.0 + values / 32,
            (ids % 3 == 0).astype(jnp.float64),
        ),
        axis=-1,
    )


def _initial_graph_cell_fields(ids: jax.Array) -> jax.Array:
    values = ids.astype(jnp.float64)
    density = 1.0 + values / 8
    first, second = values / 16, -values / 32
    energy = 2.5 + (first**2 + second**2) / (2 * density)
    return jnp.stack((0.5 + values / 4, density, first, second, energy), axis=-1)


def _initial_graph_keys(ids: jax.Array) -> jax.Array:
    keys = jax.vmap(lambda value: jax.random.fold_in(jax.random.key(20261020), value))(
        ids.reshape(-1).astype(jnp.uint32)
    )
    return jax.random.key_data(keys).reshape((*ids.shape, 2))


def _initial_graph_local(value: jax.Array) -> np.ndarray:
    # The existing two-process runner gives each owner exactly one local device.
    return np.asarray(value.addressable_shards[0].data)[0]


def _initial_graph_assert_bytes(actual: Any, expected: Any) -> None:
    left, right = np.asarray(actual), np.asarray(expected)
    assert left.shape == right.shape and left.dtype == right.dtype
    assert left.tobytes() == right.tobytes()


def _initial_graph_source() -> tuple[M.CellMeshingResult, M.PlcAssociationTransfer]:
    # Keep the established planar support source, mesh size and capacities.
    # A finer *tile bank*, not a finer mesh, permits unequal nonempty owners.
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(((2.0, 3.0), (4.0, 3.0), (4.0, 4.0), (2.0, 4.0))),
        ((0, 1, 2, 3),),
        feature_id="initial-planar-support",
    )
    author = M.NativePlanarSource(region, "authored")
    scope = M.MeshingScope(
        author.source_id,
        author.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "initial-planar-support-region",
        np.asarray((0,), dtype=np.int64),
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            M.UniformSizeControl(scope, 0.5, strength=M.SizeControlStrength.SOFT),
        ),
        quality_target=M.MeshQualityTarget(minimum_angle=0.25, hard=True),
    )
    prepared = prepare_distributed_planar_generation(
        author,
        specification,
        PlanarTileSchedule(tile_span_sizes=1),
        M.DeviceGenerationLayout(
            vertex_capacity=1000,
            cell_capacity=2000,
            candidate_capacity=64,
            maximum_work_units=2_000_000,
        ),
    )
    owners = np.zeros(prepared.tile_owners.shape, dtype=np.int32)
    assert owners.size > 2
    owners[-1] = 1
    owners.flags.writeable = False
    # Retain the genuine compiled tile complex and its source-owned native
    # preparation receipt; only its declared process assignment changes.
    prepared = replace(
        prepared,
        tile_owners=owners,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-distributed-planar-generation",
                "tiles": prepared.tiles.compiled_id,
                "layout": (
                    prepared.layout.vertex_capacity,
                    prepared.layout.cell_capacity,
                    prepared.layout.candidate_capacity,
                    prepared.layout.maximum_work_units,
                    prepared.layout.neighborhood_depth,
                ),
                "owners": array_tree_fingerprint(owners),
                "partition_count": prepared.partition_count,
            }
        ),
    )
    source = M.NativeMeshingProvider(
        M.NativeMeshingOptions("planar_constrained_delaunay")
    ).plan(
        author,
        specification,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        initial_partition=prepared,
    ).execute()
    domain = prepared.tiles.region_domain()
    _, edges, _ = planar_source_arrays(author)
    transfer = M.PlcAssociationTransfer(
        domain,
        source.coordinate_contract,
        author.source_revision,
        edge_vertices=edges,
        triangle_vertices=np.empty((0, 3), dtype=np.int64),
        triangle_facets=np.empty((0,), dtype=np.int64),
        facet_regions=domain.facet_regions,
    )
    return source, transfer


@contextmanager
def _observe_live_collective_publication() -> Iterator[dict[str, Any]]:
    """Observe genuine producers and attack only their transient host view."""
    from contextlib import ExitStack
    from copy import copy
    from dataclasses import fields
    from unittest.mock import patch

    from phydrax._meshcore import NativeExecutionScope
    from phydrax.geometry import _collective_domain_coverage as coverage_owner
    from phydrax.meshing import (
        _device_adaptation as device_owner,
        _result as result_owner,
    )

    original_builder = result_owner._build_collective_mesh_evidence
    original_raw = result_owner._prove_collective_raw_source
    original_pack = result_owner._pack_collective_epoch
    original_publication = result_owner._prepare_collective_publication
    original_premises = device_owner._collective_plc_premises
    original_certify = coverage_owner.certify_collective_premises
    original_restore = device_owner.restore_partitioned_mesh_state
    issued: list[tuple[Any, Any, Any, Any, Any]] = []
    packed: list[Any] = []
    counts = dict(raw=0, packed=0, physical=0, plc=0, restore=0)
    refusals: dict[str, bool] = {}
    report: dict[str, Any] = {"counts": counts, "refusals": refusals}

    def observe_raw(partitioned: Any, states: Any, **kwargs: Any) -> Any:
        proof = original_raw(partitioned, states, **kwargs)
        assert partitioned is not None and states is not None
        counts["raw"] += 1
        return proof

    def observe_pack(*args: Any, **kwargs: Any) -> Any:
        value = original_pack(*args, **kwargs)
        packed.append(value.states)
        counts["packed"] += 1
        return value

    def observe_publication(*args: Any, **kwargs: Any) -> Any:
        value = original_publication(*args, **kwargs)
        assert any(args[2].states is state for state in packed)
        counts["physical"] += 1
        return value

    def observe_builder(partitioned: Any, raw: Any, **kwargs: Any) -> Any:
        from phydrax.lifecycle import _meshing_source_records as records

        original_content = records.stored_record_content_id
        scanned: list[Any] = []

        def observe_content(value: Any, **controls: Any) -> str:
            if type(value) is result_owner.CollectiveMeshEvidence:
                assert controls["budget"] is kwargs["budget"]
                assert controls["memoize"] is False
                scanned.append(value)
            return original_content(value, **controls)

        with patch.object(records, "stored_record_content_id", side_effect=observe_content):
            view = original_builder(partitioned, raw, **kwargs)
        report["constructor_content_scans"] = sum(
            owner is view.evidence for owner in scanned
        )
        assert any(view._states is state for state in packed)
        issued.append((view, partitioned, raw, kwargs["scope"], kwargs["budget"]))
        return view

    def observe_restore(*args: Any, **kwargs: Any) -> Any:
        value = original_restore(*args, **kwargs)
        counts["restore"] += 1
        return value

    def observe_certify(*args: Any, **kwargs: Any) -> Any:
        assert any(args[4] is state for state in packed)
        assert any(
            args[5] is view.evidence.initial_states and kwargs["budget"] is budget
            for view, _, _, _, budget in issued
        )
        value = original_certify(*args, **kwargs)
        counts["plc"] += 1
        return value

    def observe_premises(
        partitioned: Any, target: Any, evidence: Any, **kwargs: Any
    ) -> Any:
        view = kwargs["publication"]
        scope, budget = kwargs["scope"], kwargs["budget"]
        storage = target.mesh.storage
        assert storage is not None
        assert any(view is actual for actual, _, _, _, _ in issued)
        states = view.require_states(storage, evidence, scope=scope, budget=budget)
        assert any(states is actual for actual in packed)
        assert evidence.compiled_states is next(
            raw for actual, _, raw, _, _ in issued if actual is view
        )
        epoch = dict(storage.logical_arrays)
        for owner in (states.mesh, states):
            # Static mesh signature and ambient dimension are not epoch banks.
            for member in fields(owner):
                value = getattr(owner, member.name)
                if isinstance(value, jax.Array):
                    assert epoch[f"epoch/{member.name}"] is value

        def refuse(name: str, operation: Any) -> None:
            with pytest.raises((ValueError, TypeError)):
                operation()
            refusals[name] = True

        def require() -> Any:
            return view.require_states(storage, evidence, scope=scope, budget=budget)

        def clone_record(owner: Any, **changes: Any) -> Any:
            from phydrax._model._structure import restore_registered_fields
            from phydrax.lifecycle._meshing_sources import (
                register_meshing_source_artifacts,
            )

            register_meshing_source_artifacts()
            return restore_registered_fields(
                type(owner),
                {
                    member.name: changes.get(member.name, getattr(owner, member.name))
                    for member in fields(owner)
                },
            )

        def rebound(
            name: str, field: str, replacement: Any, *, evidence_arg: Any = evidence
        ) -> None:
            # Only the ordinary private host view is fault-injected. Strict
            # scientific carriers remain untouched; adversaries use clones.
            previous = getattr(view, field)
            try:
                object.__setattr__(view, field, replacement)
                refuse(
                    name,
                    lambda: view.require_states(
                        storage, evidence_arg, scope=scope, budget=budget
                    ),
                )
            finally:
                object.__setattr__(view, field, previous)
            assert require() is states

        def rebound_evidence(name: str, **changes: Any) -> None:
            foreign = clone_record(evidence, **changes)
            rebound(name, "evidence", foreign, evidence_arg=foreign)

        def foreign_storage(
            name: str, *, logical_arrays: tuple[tuple[str, jax.Array], ...]
        ) -> None:
            foreign = eqx.tree_at(
                lambda value: value.logical_arrays, storage, logical_arrays
            )
            refuse(
                name,
                lambda: view.require_states(foreign, evidence, scope=scope, budget=budget),
            )
            assert require() is states

        refuse(
            "foreign_evidence",
            lambda: view.require_states(storage, copy(evidence), scope=scope, budget=budget),
        )
        refuse(
            "foreign_scope",
            lambda: view.require_states(
                storage, evidence, scope=NativeExecutionScope(scope.limits), budget=budget
            ),
        )
        refuse(
            "foreign_budget",
            lambda: view.require_states(storage, evidence, scope=scope, budget=copy(budget)),
        )
        rebound_evidence("layout", layout=copy(evidence.layout))
        rebound_evidence("evidence_epoch", evidence_id=evidence.evidence_id + "-changed")
        prior_storage = partitioned.prepared.adaptation.source.mesh.storage
        assert prior_storage is not None
        assert prior_storage.evidence_id != storage.evidence_id
        refuse(
            "storage_epoch",
            lambda: view.require_states(
                prior_storage, evidence, scope=scope, budget=budget
            ),
        )
        rebound_evidence("stored_content_id", stored_content_id="unissued-content")
        altered = dict(storage.logical_arrays)
        altered["epoch/clocks"] = jnp.copy(altered["epoch/clocks"])
        foreign_storage("storage_array_owner", logical_arrays=tuple(sorted(altered.items())))
        foreign_packed = eqx.tree_at(
            lambda value: value.mesh.coordinates, states, jnp.copy(states.mesh.coordinates)
        )
        rebound("packed_array_owner", "_states", foreign_packed)
        rebound("packed_object", "_states", copy(states))
        foreign_partitioned = eqx.tree_at(
            lambda value: value.layout, partitioned, copy(partitioned.layout)
        )
        rebound("partitioned_layout", "_partitioned", foreign_partitioned)
        foreign_mesh = clone_record(states.mesh, signature_id="foreign-mesh-signature")
        rebound(
            "mesh_signature", "_states",
            eqx.tree_at(lambda value: value.mesh, states, foreign_mesh),
        )
        mesh = states.mesh
        coordinates = (
            mesh.coordinates[..., :2]
            if mesh.ambient_dimension == 3
            else jnp.pad(mesh.coordinates, ((0, 0), (0, 0), (0, 1)))
        )
        foreign_mesh = type(mesh)(
            coordinates, mesh.vertex_ids, mesh.vertex_active, mesh.cells,
            mesh.cell_ids, mesh.cell_active, mesh.facet_neighbors,
        )
        rebound(
            "ambient_dimension", "_states",
            eqx.tree_at(lambda value: value.mesh, states, foreign_mesh),
        )
        with patch.object(
            result_owner.CollectiveMeshEvidence, "_initialize",
            wraps=result_owner.CollectiveMeshEvidence._initialize,
        ):
            refuse("current_constructor", require)
        refuse("direct_view_constructor", lambda: replace(view))
        rebound("empty_owners", "_owners", ())
        rebound("empty_bindings", "_bindings", ())
        foreign_raw = eqx.tree_at(
            lambda value: value.clocks, evidence.compiled_states,
            jnp.copy(evidence.compiled_states.clocks),
        )
        rebound("raw_array_owner", "_raw_states", foreign_raw)
        missing = tuple(
            (name, value) for name, value in storage.logical_arrays if name != "epoch/counters"
        )
        foreign_storage("missing_epoch_bank", logical_arrays=missing)
        refuse(
            "unissued_view",
            lambda: type(view).require_states(
                object.__new__(type(view)), storage, evidence, scope=scope, budget=budget
            ),
        )
        foreign_type = type("ForeignLivePublication", (type(view),), {})
        refuse(
            "subclass_view",
            lambda: type(view).require_states(
                object.__new__(foreign_type), storage, evidence, scope=scope, budget=budget
            ),
        )
        with pytest.raises(M.MeshingFailure) as caught:
            original_premises(
                partitioned, target, evidence, **{**kwargs, "publication": states}
            )
        assert caught.value.category is M.MeshingFailureCategory.LINEAGE_FAILED
        assert caught.value.stage == "distributed-device-commit"
        assert isinstance(caught.value.__cause__, ValueError)
        assert "publication" in str(caught.value.__cause__)
        refusals["arbitrary_caller_data"] = True
        bank = partitioned.prepared.adaptation.policy.association_transfer.source_vertices
        assert isinstance(bank, np.ndarray) and not bank.flags.writeable
        old = bank.flat[0].copy()
        try:
            bank.flags.writeable = True
            bank.flat[0] = old + np.float64(0.125)
            bank.flags.writeable = False
            refuse("readonly_source_bytes", require)
        finally:
            bank.flags.writeable = True
            bank.flat[0] = old
            bank.flags.writeable = False
        assert require() is states
        value = original_premises(partitioned, target, evidence, **kwargs)
        assert value is not None
        for certificate in value:
            assert certificate.status == "certified" and not certificate.findings
        report["same_actual_packed_object"] = True
        return value

    with ExitStack() as stack:
        for owner, name, observer in (
            (result_owner, "_build_collective_mesh_evidence", observe_builder),
            (result_owner, "_prove_collective_raw_source", observe_raw),
            (result_owner, "_pack_collective_epoch", observe_pack),
            (result_owner, "_prepare_collective_publication", observe_publication),
            (device_owner, "_collective_plc_premises", observe_premises),
            (device_owner, "restore_partitioned_mesh_state", observe_restore),
            (coverage_owner, "certify_collective_premises", observe_certify),
        ):
            stack.enter_context(patch.object(owner, name, side_effect=observer))
        yield report
    assert issued
    for view, _, _, scope, budget in issued:
        with pytest.raises(ValueError):
            view.require_states(None, view.evidence, scope=scope, budget=budget)
    refusals["expired_request"] = True


def initial_graph_live_publication() -> dict[str, Any]:
    return initial_graph_migration(_observe_publication=True)


def _collective_migration_record_checks(record: Any, failed_owner: int) -> dict[str, bool]:
    """Reconstruct actual distributed packets and refuse one-rank local corruption."""
    from copy import copy

    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts

    register_meshing_source_artifacts()
    checks: dict[str, bool] = {}
    owners = (
        ("source", record.source, "source_topology_id", "0" * 64),
        ("workset", record.saved_migration_state.workset, "cell_coordinates", np.asarray(0, dtype=np.int32)),
        ("state", record.saved_migration_state, "cell_epochs", np.asarray(0, dtype=np.int64)),
        ("transport", record.transport_evidence, "cell_capacity", 0),
        ("record", record, "migration_id", "0" * 64),
        (
            "source-optional", record.source, "solver_neighborhood",
            record.solver_neighborhood if record.source.solver_neighborhood is None else None,
        ),
        ("state-optional", record.saved_migration_state, "forest", None),
    )
    for name, owner, field, replacement in owners:
        rebuilt = type(owner).from_stored_fields(owner, collective=True)
        assert jax.tree_util.tree_structure(rebuilt) == jax.tree_util.tree_structure(owner)
        assert all(
            first is second for first, second in zip(
                jax.tree_util.tree_leaves(rebuilt), jax.tree_util.tree_leaves(owner),
                strict=True,
            )
        )
        changed = owner
        if jax.process_index() == failed_owner:
            changed = copy(owner)
            object.__setattr__(changed, field, replacement)
        with pytest.raises((ValueError, TypeError)):
            type(owner).from_stored_fields(changed, collective=True)
        recovered = type(owner).from_stored_fields(owner, collective=True)
        assert jax.tree_util.tree_structure(recovered) == jax.tree_util.tree_structure(owner)
        assert all(
            first is second for first, second in zip(
                jax.tree_util.tree_leaves(recovered), jax.tree_util.tree_leaves(owner),
                strict=True,
            )
        )
        checks[name] = True
    assert record._identity(collective=True) == record.migration_id
    return checks


def initial_graph_collective_record_validation(failed_owner: str) -> dict[str, Any]:
    return initial_graph_migration(_collective_record_fault_owner=int(failed_owner))


def initial_graph_migration(
    *, _observe_publication: bool = False,
    _collective_record_fault_owner: int | None = None,
) -> dict[str, Any]:
    with _distributed():
        # A genuine native initial theorem, not an empty adaptive publication.
        # Unequal tile assignment leaves both owners nonempty but forces a
        # balanced GRAPH placement to send actual cells across the interface.
        source, transfer = _initial_graph_source()
        initial = source.collective_evidence
        assert isinstance(initial, InitialCollectiveMeshEvidence)
        assert not any(name.startswith("epoch/") for name, _ in initial.logical_arrays)
        initial.require_passed()
        provider = M.MeshPartitionPolicy(M.MeshPartitionKind.PROVIDER, 2, halo_width=1)
        distribution = M.prepare_mesh_distribution(
            M.MeshPart("domain", source), policy=provider
        )
        graph = M.MeshPartitionPolicy(
            M.MeshPartitionKind.GRAPH, 2, halo_width=1, maximum_imbalance=1.05
        )
        policy = M.MeshAdaptationPolicy(
            M.MeshAdaptationRoute.DEVICE_BISECTION,
            compatibility=M.BisectionCompatibility.UNIFORM_REFINEMENT,
            device_policy=phx.discretization.AdaptiveSimplexPolicy(
                vertex_capacity=512, cell_capacity=512
            ),
            distribution=distribution,
            partition_policy=graph,
            association_transfer=transfer,
        )
        partitioned = M.prepare_partitioned_mesh_adaptation(
            M.prepare_mesh_adaptation(source, M.MarkedMeshAdaptation(), policy=policy)
        )
        states = partitioned.states
        raw_before = [
            _initial_graph_local(value).copy()
            for value in jax.tree_util.tree_leaves(states)
        ]
        initial_owned_cells = int(np.sum(_initial_graph_local(states.mesh.cell_active)))
        mesh = states.mesh
        cell_ids = jnp.where(mesh.cell_active, mesh.cell_ids, -1)
        vertex_ids = jnp.where(mesh.vertex_active, mesh.vertex_ids, -1)
        epochs = jnp.where(mesh.cell_active, 41 + mesh.cell_ids % 5, 0).astype(jnp.int64)
        keys = jax.device_put(_initial_graph_keys(mesh.cell_ids), mesh.cell_ids.sharding)
        counters = jnp.where(mesh.cell_active, 2**54 + mesh.cell_ids, 0).astype(jnp.uint64)
        vertex_history = jnp.where(
            mesh.vertex_ids >= 0, 2**54 + mesh.vertex_ids, -1
        ).astype(jnp.int64)
        bound = prepare_ownerlocal_migration_state(
            partitioned,
            states,
            cell_field_ids=cell_ids,
            cell_fields=_initial_graph_cell_fields(cell_ids),
            vertex_field_ids=vertex_ids,
            vertex_fields=_initial_graph_vertex_fields(vertex_ids),
            cell_epochs=epochs,
            cell_history=(keys, counters),
            vertex_history=(vertex_history,),
        )
        proposal = prepare_partitioned_rebalance_proposal(partitioned, bound, graph)
        assert proposal.graph_evidence is not None and proposal.graph_evidence.accepted
        assert float(proposal.imbalance) <= graph.maximum_imbalance

        # Each corruption must be refused without publishing or modifying the
        # genuine source. These exercise the public migration admission, not
        # a replacement/relaxed adaptive restart proof.
        raw_changed = eqx.tree_at(
            lambda value: value.mesh.coordinates,
            states,
            states.mesh.coordinates + jnp.float64(0.125),
        )
        first_id = jnp.min(jnp.where(mesh.cell_active, mesh.cell_ids, jnp.iinfo(jnp.int64).max))
        lost_cell = eqx.tree_at(
            lambda value: value.mesh.cell_active,
            states,
            mesh.cell_active & (mesh.cell_ids != first_id),
        )
        missing_owner = eqx.tree_at(
            lambda value: value.requested_target_owners,
            proposal,
            jnp.where(bound.workset.cell_valid, -1, proposal.requested_target_owners),
        )
        coordinates_index = next(
            index for index, (name, _) in enumerate(initial.logical_arrays)
            if name == "coordinates"
        )
        changed_source = eqx.tree_at(
            lambda value: value.prepared.adaptation.source.collective_evidence.logical_arrays[
                coordinates_index
            ][1],
            partitioned,
            initial.logical_arrays[coordinates_index][1].at[0, 0].add(0.125),
        )
        refusals = {}
        changed_graph = eqx.tree_at(
            lambda value: value.graph_evidence,
            proposal,
            replace(
                proposal.graph_evidence,
                edge_cut=proposal.graph_evidence.edge_cut + 1,
            ),
        )
        live_parts = type(partitioned).execution_parts(partitioned)
        # Each route must reproduce the same complete genuinely issued evidence,
        # including every array/static measurement and all eleven work counters.
        for execution in (None, live_parts):
            replay_partitioned_rebalance_proposal(
                proposal, axis_name=live_parts.axis_name, parts=execution
            )
            with pytest.raises(ValueError, match="graph_evidence"):
                replay_partitioned_rebalance_proposal(
                    changed_graph, axis_name=live_parts.axis_name, parts=execution
                )
        foreign_parts = phx.discretization.AdaptiveSimplexParts(
            tuple(live_parts.mesh.devices.flat)[::-1],
            axis_name=live_parts.axis_name,
            neighbor_pairs=live_parts.neighbor_pairs,
        )
        with pytest.raises(ValueError, match="sharding"):
            replay_partitioned_rebalance_proposal(
                proposal, axis_name=live_parts.axis_name, parts=foreign_parts
            )
        refusals["physical_placement"] = True
        refusals["graph_measurements"] = True
        for name, original, raw, requested in (
            ("source_mutation", changed_source, states, proposal),
            ("raw_mutation", partitioned, raw_changed, proposal),
            ("lost_cell", partitioned, lost_cell, proposal),
            ("missing_owner", partitioned, states, missing_owner),
        ):
            with pytest.raises((ValueError, M.MeshingFailure)):
                repartition_partitioned_mesh_adaptation(original, raw, bound, requested)
            refusals[name] = True

        moved = repartition_partitioned_mesh_adaptation(partitioned, states, bound, proposal)
        migration = moved.migration
        assert isinstance(migration, PartitionedEpochMigration)
        assert migration.source_partitioned is partitioned
        assert migration.saved_states is states
        assert migration.saved_migration_state is bound
        assert moved.prepared is partitioned.prepared
        assert moved.layout is partitioned.layout
        assert moved.restart_repack is None and moved.restart_proof is None
        accepted = accept_ownerlocal_migration(migration.migration)
        assert bool(jnp.all(migration.migration.evidence.accepted))
        moved_cells = int(jnp.sum(migration.migration.evidence.sent_cells))
        assert moved_cells > 0
        valid = _initial_graph_local(accepted.workset.cell_valid)
        ids = _initial_graph_local(accepted.workset.cell_ids)[valid]
        vertices = _initial_graph_local(accepted.workset.cell_vertices)[valid]
        assert ids.size > 0
        _initial_graph_assert_bytes(
            _initial_graph_local(accepted.cell_fields)[valid],
            _initial_graph_cell_fields(jnp.asarray(ids)),
        )
        _initial_graph_assert_bytes(
            _initial_graph_local(accepted.vertex_fields)[valid],
            _initial_graph_vertex_fields(jnp.asarray(vertices)),
        )
        _initial_graph_assert_bytes(
            _initial_graph_local(accepted.cell_epochs)[valid], 41 + ids % 5
        )
        forest = accepted.forest
        assert forest is not None
        allocated = _initial_graph_local(forest.mesh.cell_ids) >= 0
        forest_ids = _initial_graph_local(forest.mesh.cell_ids)[allocated]
        actual_keys = _initial_graph_local(accepted.cell_history[0])[allocated]
        expected_keys = np.asarray(_initial_graph_keys(jnp.asarray(forest_ids)))
        _initial_graph_assert_bytes(actual_keys, expected_keys)
        _initial_graph_assert_bytes(
            _initial_graph_local(accepted.cell_history[1])[allocated],
            (2**54 + forest_ids).astype(np.uint64),
        )
        for key_data, expected in zip(actual_keys, expected_keys, strict=True):
            _initial_graph_assert_bytes(
                jax.random.bits(jax.random.wrap_key_data(jnp.asarray(key_data)), (8,)),
                jax.random.bits(jax.random.wrap_key_data(jnp.asarray(expected)), (8,)),
            )
        vertex_allocated = _initial_graph_local(forest.mesh.vertex_ids) >= 0
        _initial_graph_assert_bytes(
            _initial_graph_local(accepted.vertex_history[0])[vertex_allocated],
            2**54 + _initial_graph_local(forest.mesh.vertex_ids)[vertex_allocated],
        )
        _initial_graph_assert_bytes(
            _initial_graph_local(migration.states.clocks), _initial_graph_local(states.clocks)
        )
        _initial_graph_assert_bytes(
            _initial_graph_local(migration.states.counters), _initial_graph_local(states.counters)
        )
        if _observe_publication:
            with _observe_live_collective_publication() as live_publication:
                published = M.commit_partitioned_adaptive_simplex(moved, migration.states)
        else:
            live_publication = None
            published = M.commit_partitioned_adaptive_simplex(moved, migration.states)
        assert len(published.local_results) == 1
        result = published.local_results[0]
        assert result.status is M.MeshAdaptationStatus.COMPLETE
        target = result.target
        evidence = target.collective_evidence
        assert isinstance(evidence, M.CollectiveMeshEvidence)
        assert evidence.migration_record is not None
        evidence.require_passed()
        target.audit.require_passed()
        for actual, expected in zip(jax.tree_util.tree_leaves(states), raw_before, strict=True):
            _initial_graph_assert_bytes(_initial_graph_local(actual), expected)
        assert source.collective_evidence is initial
        assert evidence.source is source
        assert evidence.source.collective_evidence is initial
        record = evidence.migration_record
        record.validate_restored()
        assert record.source.partitioned_id == partitioned.partitioned_id
        assert (
            record.source.preparation_layout.signature_id
            == partitioned.prepared.layout.signature_id
        )
        assert record.source.layout.signature_id == partitioned.layout.signature_id
        collective_records = (
            None if _collective_record_fault_owner is None
            else _collective_migration_record_checks(record, _collective_record_fault_owner)
        )
        owned = valid & (
            _initial_graph_local(accepted.workset.cell_owner) == jax.process_index()
        )
        return {
            "initial_evidence": initial.evidence_id,
            "source_result": source.result_id,
            "target_result": target.result_id,
            "migration_record": evidence.migration_record.migration_id,
            "initial_cells": initial.global_entity_counts[-1],
            "initial_owned_cells": initial_owned_cells,
            "moved_cells": moved_cells,
            "published_cells": evidence.global_entity_counts[-1],
            "owned_ids": _initial_graph_local(accepted.workset.cell_ids)[owned].tolist(),
            "proposal_weights": np.asarray(proposal.part_weights).tolist(),
            "imbalance": float(proposal.imbalance),
            "refusals": refusals,
            "bytewise_fields_history_rng_clock": True,
            "source_raw_unchanged": True,
            "live_publication": live_publication,
            "collective_record_refusals": collective_records,
        }
