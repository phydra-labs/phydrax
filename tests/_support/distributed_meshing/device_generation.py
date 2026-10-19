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
from phydrax.meshing._distributed_generation import (
    DistributedSurfaceConstruction,
    execute_distributed_surface_initial,
    PreparedDistributedSurfaceGeneration,
)
from phydrax.meshing._initial_certification import InitialCollectiveMeshEvidence
from phydrax.meshing._result import (
    CollectiveMeshStorageBinding,
    require_original_meshing_source,
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
        "domain_id": evidence.compiled.domain.domain_id,
        "expected_domain_id": domain.domain_id,
        "specification_id": evidence.specification.specification_id,
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
