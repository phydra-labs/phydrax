# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, final, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from examples.boundary_layer_core_mesh import exact_solution
from phydrax import array_tree_fingerprint, canonical_fingerprint, model_array_bytes
from phydrax._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    PeriodicIsometryGroup,
    PeriodicMeshTopology,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_operands import (
    CoordinateOperandCollector,
    PreparedCoordinateOperands,
)
from phydrax.geometry import PiecewiseLinearDomain
from phydrax.lifecycle import (
    MeshingAcceptedEpoch,
    MeshingSourceExecutionControls,
    read_meshing_source_execution_controls,
    write_meshing_source_closure,
)
from phydrax.lifecycle._meshing_sources import meshing_source_owner_reconstruction_scope
from phydrax.meshing import MixedAdaptationEvidence, NativeMeshingPhaseMeasurement
from phydrax.meshing._native_preparation import (
    native_preparation_ledger,
    stage_native_completed_source_issue,
)
from phydrax.meshing.providers._native_publication import (
    bind_native_execution_result,
    NativePublicationExecution,
)
from phydrax.meshing.qualification import (
    hybrid_source_coefficients_restored,
    layer_core_accepted_epoch,
    layer_core_lineage_records,
    layer_core_material_declaration,
    layer_core_restoration_defects,
    LayerCoreAcceptedState,
    LayerCoreLifecyclePlans,
    LayerCoreMaterialDeclaration,
    ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST,
    ORIGINAL_CURVED_PERIODIC_SOURCE_ID,
    ORIGINAL_CURVED_PERIODIC_SOURCE_REVISION,
    ORIGINAL_CURVED_PERIODIC_TRACE_RESIDUALS,
    periodic_layer_source_identity_record,
    prepare_corrected_periodic_layer_source,
    prepare_hybrid_adaptation_policy,
    prepare_hybrid_high_order_map,
    qualify_hybrid_curved_publication,
    qualify_hybrid_high_order_publication,
    qualify_hybrid_reference_cells,
    require_restored_layer_core_result_identity,
    warm_layer_core_initial_state,
    warm_layer_core_step,
)
from phydrax.meshing.qualification._layer_continuation import (
    _layer_core_current_preparation,
    layer_core_continuation_execution,
)
from tools._meshing_cases import (
    _hybrid_qualification_original_box,
    _hybrid_qualification_original_channel,
    _hybrid_qualification_original_control,
    _hybrid_qualification_original_icosphere,
    _hybrid_qualification_original_layer_whole,
    _hybrid_qualification_original_plate,
    _hybrid_qualification_original_wall,
    original_wall_layer_schedule,
)
from tools.layer_core_lifecycle_continuation import main
from tools.meshing_qualification._controls import (
    _hybrid_material_controls,
    _hybrid_qualification_original_controls,
    _hybrid_qualification_original_profile_names,
    _hybrid_qualification_original_wall_profiles,
    _hybrid_qualification_profile_contracts,
    _hybrid_qualification_requirements,
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._physics import _compiled_solver_stages
from tools.meshing_qualification._records import (
    _phase_recordings,
    _quality_summary,
    _result_identity,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _runtime_identity,
    ColdControlPacket,
    ColdWorkerReport,
    Deadline,
    epoch_stage_name,
    peak_resident_bytes,
    PEAK_RESIDENT_SCOPE,
    prepare_cold_worker,
    qualification_failure,
    StageLedger,
)
from tools.meshing_qualification.scenarios.boundary_layer import _layer_evidence
from tools.meshing_qualification.scenarios.cad_curving import _cad_curving_certificate


def _hybrid_original_source_evidence(
    source: phx.discretization.CellMesh
    | phx.geometry.PiecewiseLinearDomain
    | phx.geometry.BRepModel,
) -> dict[str, object]:

    match source:
        case phx.discretization.CellMesh():
            return {
                "source_id": str(source.mesh_id),
                "source_revision": str(source.numeric_version),
            }
        case phx.geometry.PiecewiseLinearDomain():
            return {
                "source_id": str(source.source_id),
                "source_revision": str(source.source_revision),
            }
        case phx.geometry.BRepModel():
            evidence: dict[str, object] = {
                "source_id": str(source.source_id),
                "source_revision": str(source.source_revision),
            }
            geometry = source.geometry
            if geometry is not None:
                evidence["source_vertex_bits_sha256"] = sha256(
                    np.asarray(geometry.vertex_points).tobytes()
                ).hexdigest()
            return evidence
        case _:
            raise TypeError(
                f"qualification.hybrid_source must be a wall mesh, PL domain, or BRep model; got {type(source).__qualname__}"
            )


def _hybrid_mixed_hierarchy(
    adaptation: phx.meshing.MeshAdaptationResult,
) -> phx.meshing.MixedAdaptationHierarchy:

    match adaptation.hierarchy:
        case phx.meshing.MixedAdaptationHierarchy() as hierarchy:
            return hierarchy
        case hierarchy:
            raise qualification_failure(
                "hybrid.mixed_hierarchy",
                "retain the accepted mixed-cell sibling records",
                {"hierarchy": hierarchy},
            )


def _hybrid_rejected_ids(adaptation: phx.meshing.MeshAdaptationResult) -> np.ndarray:

    match adaptation.evidence:
        case (phx.meshing.BisectionEvidence() | MixedAdaptationEvidence()) as evidence:
            return np.asarray(evidence.rejected_coarsening_ids)
        case evidence:
            raise qualification_failure(
                "hybrid.coarsening_evidence",
                "name the rejected native siblings",
                {"evidence": evidence},
            )


def _hybrid_qualification_original_layer_face_points(model: Any, face: int) -> np.ndarray:

    geometry = model.geometry
    if geometry is None:
        raise qualification_failure(
            "_hybrid_qualification_original_layer_face_points",
            ("The native extrusion fixture requires geometry."),
            {"geometry": geometry},
        )
    vertices = sorted(
        {
            vertex
            for loop in geometry.face_loops[face]
            for coedge in loop
            for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
        }
    )
    return np.asarray(geometry.vertex_points, dtype=np.float64)[vertices]


def _hybrid_qualification_original_layer_brep_scope(
    model: Any, dimension: int, entities: tuple[int, ...]
) -> Any:
    domain = phx.geometry.MeshingDomain.from_brep(model)
    definitions = (
        domain.region_source_indices
        if dimension == 3
        else domain.source_indices[dimension]
    )
    selected = set(entities)
    slots = domain.scope_indices(dimension)
    identifiers = tuple(
        slots[row] for row, definition in enumerate(definitions) if definition in selected
    )
    return phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        dimension,
        domain.entity_set_id(dimension),
        np.asarray(identifiers, dtype=np.int64),
    )


def _hybrid_qualification_original_layer_periodic_oracle(
    result: Any,
    identification: Any,
    topology: Any,
    schedule: Any,
    area: float,
    transition_height: float,
    periodicity: str,
) -> None:

    assert result.mesh.periodic_topology is not None
    assert result.cap is not None and result.cap.periodic_topology is not None
    for carrier in (result.mesh, result.cap):
        descriptor = carrier.periodic_topology
        assert isinstance(descriptor.cell, PeriodicIsometryGroup)
        assert descriptor.cell.group_id == identification.group_id
        incidences = [
            incidence.scipy_boundary().toarray()
            for incidence in descriptor.quotient.incidences
        ]
        for lower, upper in zip(incidences, incidences[1:]):
            np.testing.assert_array_equal(lower @ upper, 0.0)
    assert result.closed_cap is (periodicity == "translation")
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses, schedule.thicknesses, rtol=1e-12
    )
    np.testing.assert_allclose(
        phx.discretization.periodic_orbit_measures(result.mesh).total_measure,
        area * (sum(schedule.thicknesses) + transition_height / 3.0),
        rtol=1e-12,
    )
    assert result.source_wall.periodic_topology is not None
    assert (
        result.source_wall.periodic_topology.periodic_topology_id
        == topology.periodic_topology_id
    )


def _hybrid_qualification_original_layer_periodic(
    periodicity: str, quadrilateral: bool
) -> dict[str, Any]:

    if periodicity == "rotation":
        points = np.asarray(
            ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 0.0))
        )
        generators = np.asarray(
            (
                (
                    (0.0, -1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            )
        )
        representatives = np.asarray((0, 1, 1, 0), dtype=np.int64)
        shifts = np.asarray(((0,), (0,), (1,), (1,)), dtype=np.int64)
        area = 0.375
    elif periodicity in ("translation", "partial"):
        points = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        )
        generators = np.repeat(np.eye(4)[None], 2, axis=0)
        generators[0, 0, 3] = 1.0
        generators[1, 1, 3] = 1.0
        representatives = np.zeros(4, dtype=np.int64)
        shifts = points[:, :2].astype(np.int64)
        if periodicity == "partial":
            generators = generators[:1]
            representatives = np.asarray((0, 0, 3, 3), dtype=np.int64)
            shifts = points[:, :1].astype(np.int64)
        area = 1.0
    else:
        raise ValueError(
            f"_hybrid_qualification_original_layer_periodic.periodicity must identify an admitted periodicity mode; got {periodicity!r}."
        )
    rows = np.asarray(
        ((0, 1, 2, 3),) if quadrilateral else ((0, 1, 2), (0, 2, 3)), dtype=np.int64
    )
    block = phx.discretization.CellBlock(
        "wall", "quadrilateral" if quadrilateral else "triangle", rows
    )
    lifted = phx.discretization.CellMesh(points, (block,))
    identification = PeriodicIsometryGroup(generators)
    topology = PeriodicMeshTopology(lifted, identification, representatives, shifts)
    wall = phx.discretization.CellMesh(points, (block,), periodic_topology=topology)
    schedule = original_wall_layer_schedule()
    control = phx.meshing.BoundaryLayerControl(
        _hybrid_qualification_original_layer_whole(wall),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
    )
    result = phx.meshing.prepare_boundary_layers(wall, control)
    transition_height = (
        0.5 * np.mean(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1))
        if quadrilateral
        else 0.0
    )
    _hybrid_qualification_original_layer_periodic_oracle(
        result, identification, topology, schedule, area, transition_height, periodicity
    )
    return dict(source=wall, wall=wall, control=control, layer=result)


def _hybrid_qualification_original_layer_ridge() -> dict[str, Any]:

    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.5, 0.1),
            (1.0, 0.5, 0.1),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        )
    )
    block = phx.discretization.CellBlock(
        "wall",
        "triangle",
        np.asarray(((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)), dtype=np.int64),
    )
    lifted = phx.discretization.CellMesh(points, (block,))
    generator = np.eye(4)[None]
    generator[0, 0, 3] = 1.0
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicIsometryGroup(generator),
        np.asarray((0, 0, 2, 2, 4, 4), dtype=np.int64),
        np.asarray(((0,), (1,), (0,), (1,), (0,), (1,)), dtype=np.int64),
    )
    wall = phx.discretization.CellMesh(points, (block,), periodic_topology=topology)
    schedule = original_wall_layer_schedule()
    control = phx.meshing.BoundaryLayerControl(
        _hybrid_qualification_original_layer_whole(wall),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        feature_angle=0.1,
        corner=phx.meshing.BoundaryLayerCornerPolicy.FAN,
    )
    result = phx.meshing.prepare_boundary_layers(wall, control)
    assert result.evidence.convex_ridge_count == 1
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses, schedule.thicknesses, rtol=1e-12
    )
    assert result.cap is not None and result.cap.periodic_topology is not None
    boundary = result.cap.periodic_topology.quotient.entities(1).subset("boundary")
    assert np.count_nonzero(np.asarray(boundary.mask)) == 2
    assert result.mesh.periodic_topology is not None
    first, second, third = (
        incidence.scipy_boundary().toarray()
        for incidence in result.mesh.periodic_topology.quotient.incidences
    )
    np.testing.assert_array_equal(first @ second, 0.0)
    np.testing.assert_array_equal(second @ third, 0.0)
    return dict(source=wall, wall=wall, control=control, layer=result)


def _hybrid_qualification_original_layer_cad(
    destination: str | Path, remaining_core: bool
) -> dict[str, Any]:
    destination = Path(destination)
    thicknesses = (0.05, 0.10) if remaining_core else (0.25, 0.75)
    source = phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 1.0),
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
        tessellation=phx.geometry.BRepTessellationPolicy(
            linear_deflection=0.05, angular_deflection=0.2
        ),
    )
    walls = tuple(
        face
        for face in range(source.topology.num_faces)
        if np.all(
            _hybrid_qualification_original_layer_face_points(source, face)[:, 2] == 0.0
        )
    )
    schedule = phx.meshing.LayerSchedule(np.asarray(thicknesses, dtype=np.float64))
    request = phx.meshing.BoundaryLayerControl(
        _hybrid_qualification_original_layer_brep_scope(source, 2, walls),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.CAD_EXTRUSION,
        volume_scope=_hybrid_qualification_original_layer_brep_scope(source, 3, (0,)),
    )
    result = phx.meshing.prepare_boundary_layer_extrusion(
        source,
        request,
        destination=destination,
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    published = result.source
    domain = phx.geometry.MeshingDomain.from_brep(published)
    sweep = result.control
    total = sum(thicknesses)
    assert sweep.route is phx.meshing.BoundaryLayerRoute.EXACT_SWEEP
    np.testing.assert_array_equal(sweep.schedule.thicknesses, thicknesses)
    np.testing.assert_allclose(result.slab_volumes, (6.0 * total,), rtol=1e-12)
    assert result.maximum_relative_volume_residual <= 1e-9
    assert len(result.layer_solid_ids) == 1
    assert len(result.core_solid_ids) == (1 if remaining_core else 0)
    assert sweep.volume_scope is not None and sweep.cap_scope is not None
    volume_rows = domain.resolve_indices(
        3, np.asarray(sweep.volume_scope.entity_ids, dtype=np.int64)
    )
    assert (
        tuple(sorted(domain.region_source_indices[row] for row in volume_rows))
        == result.layer_solid_ids
    )
    assert sweep.wall_scope.source_revision == published.source_revision
    assert sweep.cap_scope.source_revision == published.source_revision
    assert published.source_id == str(destination.resolve())
    wall_rows = domain.resolve_indices(
        2, np.asarray(sweep.wall_scope.entity_ids, dtype=np.int64)
    )
    for face in (domain.source_indices[2][row] for row in wall_rows):
        np.testing.assert_allclose(
            _hybrid_qualification_original_layer_face_points(published, face)[:, 2],
            0.0,
            atol=1e-14,
        )
    cap_rows = domain.resolve_indices(
        2, np.asarray(sweep.cap_scope.entity_ids, dtype=np.int64)
    )
    for face in (domain.source_indices[2][row] for row in cap_rows):
        np.testing.assert_allclose(
            _hybrid_qualification_original_layer_face_points(published, face)[:, 2],
            total,
            atol=1e-14,
        )
        assert len(published.topology.face_solids[face]) == (2 if remaining_core else 1)
    if remaining_core:
        core_points = np.concatenate(
            [
                _hybrid_qualification_original_layer_face_points(published, face)
                for face in published.topology.solid_faces[result.core_solid_ids[0]]
            ]
        )
        np.testing.assert_allclose(
            np.min(core_points, axis=0), (0.0, 0.0, total), atol=1e-14
        )
        np.testing.assert_allclose(
            np.max(core_points, axis=0), (2.0, 3.0, 1.0), atol=1e-14
        )
    # This owner is native CAD preparation, not an invented boundary-layer mesh or core mesh.
    return dict(
        source=published,
        original_source=source,
        control=sweep,
        extrusion=result,
        source_domain=domain,
    )


def _hybrid_qualification_original_layer_exact(
    periodic: bool, perturb_cap: bool = False
) -> dict[str, Any]:

    M = phx.meshing
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (1.0, 0.5, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    rows = np.asarray(((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)), dtype=np.int64)
    block = CellBlock("wall", "triangle", rows)
    wall = CellMesh(points, (block,))
    if periodic:
        generator = np.eye(4, dtype=np.float64)[None]
        generator[0, 0, 3] = 1.0
        topology = PeriodicMeshTopology(
            wall,
            PeriodicIsometryGroup(generator),
            np.asarray((0, 0, 2, 2, 4, 4), dtype=np.int64),
            np.asarray(((0,), (1,), (0,), (1,), (0,), (1,)), dtype=np.int64),
        )
        wall = CellMesh(points, (block,), periodic_topology=topology)
    schedule = M.LayerSchedule.geometric(2, 0.01, growth_rate=1.0)
    physical_control = M.BoundaryLayerControl(
        _hybrid_qualification_original_layer_whole(wall),
        schedule,
        route=M.BoundaryLayerRoute.ADVANCING,
        corner=M.BoundaryLayerCornerPolicy.SMOOTH,
    )
    physical = M.prepare_boundary_layers(wall, physical_control)
    actual_points = np.asarray(physical.mesh.coordinates).copy()
    actual_columns = np.asarray(physical.column_index).copy()
    stations = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(np.asarray(schedule.thicknesses, dtype=np.float64)),
        )
    )
    upper = points + stations[-1] * np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if perturb_cap:
        upper[2, 2] = np.nextafter(upper[2, 2], np.inf)
    edges = np.asarray(((0, 1), (1, 3), (3, 5), (5, 4), (4, 2), (2, 0)), dtype=np.int64)
    a, b = edges.T
    sides = np.stack(
        (np.stack((a, b, 6 + b), axis=1), np.stack((a, 6 + b, 6 + a), axis=1)), axis=1
    ).reshape(-1, 3)
    facets = np.concatenate((rows[:, ::-1], rows + 6, sides))
    domain = PiecewiseLinearDomain(
        np.concatenate((points, upper)),
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (facets.shape[0], 1)),
        ("wall-material",),
        source_id="independent-reference-slab",
    )
    scope = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        f"{domain.domain_id}:facets",
        np.arange(4, dtype=np.int64),
    )
    cap_scope = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        scope.entity_set_id,
        np.arange(4, 8, dtype=np.int64),
    )
    volume = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        3,
        f"{domain.domain_id}:regions",
        np.asarray([0], dtype=np.int64),
    )
    association = M.GeometryAssociation(
        M.GeometryAssociationKind.PIECEWISE_LINEAR,
        domain.source_id,
        domain.source_revision,
        wall.entity_set(2).entity_set_id,
        wall.entity_set(2).entity_ids,
        tuple(f"{domain.source_revision}:facet:{row}" for row in range(4)),
        np.zeros(4, dtype=np.float64),
        exact=True,
        source_dimensions=np.full(4, 2, dtype=np.int64),
        source_indices=np.arange(4, dtype=np.int64),
        source_entity_roles=(M.GeometrySourceEntityRole.FACET,) * 4,
        orientations=-np.ones(4, dtype=np.int32),
    )
    control = M.BoundaryLayerControl(
        scope,
        schedule,
        route=M.BoundaryLayerRoute.EXACT_SWEEP,
        volume_scope=volume,
        cap_scope=cap_scope,
    )
    sweep = M.SweepControl(
        M.SweepMapKind.EXTRUSION, schedule, "original-reference-schedule"
    )
    if perturb_cap:
        try:
            M.prepare_boundary_layers(
                wall,
                control,
                sweep_control=sweep,
                wall_association=association,
                source_domain=domain,
            )
        except ValueError as error:
            assert "retain every declared source facet exactly" in str(error)
            return dict(
                source=domain,
                source_domain=domain,
                wall=wall,
                control=control,
                physical=physical,
                sweep_control=sweep,
                wall_association=association,
                rejection=str(error),
            )
        raise AssertionError(
            f"_hybrid_qualification_original_layer_exact.perturb_cap must produce a constructor refusal; got {perturb_cap!r}."
        )
    reference = M.prepare_boundary_layers(
        wall,
        control,
        sweep_control=sweep,
        wall_association=association,
        source_domain=domain,
    )
    expected = points[None] + stations[:, None, None] * np.asarray(
        (0.0, 0.0, 1.0), dtype=np.float64
    )
    np.testing.assert_array_equal(
        np.asarray(reference.mesh.coordinates).view(np.uint64),
        expected.reshape(-1, 3).view(np.uint64),
    )
    np.testing.assert_array_equal(
        np.asarray(physical.mesh.coordinates).view(np.uint64),
        actual_points.view(np.uint64),
    )
    np.testing.assert_array_equal(physical.column_index, actual_columns)
    np.testing.assert_array_equal(reference.layer_index, physical.layer_index)
    np.testing.assert_array_equal(reference.column_index, physical.column_index)
    np.testing.assert_allclose(
        reference.evidence.column_thicknesses,
        np.broadcast_to(np.diff(stations)[:, None], (2, 6)),
        rtol=1e-12,
    )
    assert reference.source_wall is wall and reference.source_domain is domain
    assert reference.control is control and reference.wall_association is association
    assert reference.result_id != physical.result_id
    assert reference.validity.certified_valid_count == 8
    assert reference.cap is not None
    for carrier in (reference.mesh, reference.cap):
        assert (carrier.periodic_topology is not None) == periodic
    qualify_hybrid_reference_cells(reference.mesh)
    return dict(
        source=domain,
        source_domain=domain,
        wall=wall,
        control=control,
        layer=reference,
        physical=physical,
        sweep_control=sweep,
        wall_association=association,
    )


def _hybrid_qualification_original_layer_cad_rejection(
    destination: str | Path, opposite_walls: bool
) -> dict[str, Any]:
    destination = Path(destination)
    thickness, reason = (
        (0.75, "slabs of different walls overlap")
        if opposite_walls
        else (1.25, "leaves its controlled solid")
    )
    source = phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 1.0),
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
    )
    walls = tuple(
        face
        for face in range(source.topology.num_faces)
        if np.all(
            _hybrid_qualification_original_layer_face_points(source, face)[:, 2] == 0.0
        )
        or (
            opposite_walls
            and np.all(
                _hybrid_qualification_original_layer_face_points(source, face)[:, 2]
                == 1.0
            )
        )
    )
    request = phx.meshing.BoundaryLayerControl(
        _hybrid_qualification_original_layer_brep_scope(source, 2, walls),
        phx.meshing.LayerSchedule(np.asarray((thickness,), dtype=np.float64)),
        route=phx.meshing.BoundaryLayerRoute.CAD_EXTRUSION,
        volume_scope=_hybrid_qualification_original_layer_brep_scope(source, 3, (0,)),
    )
    original = source.model_id
    destination.write_bytes(b"accepted artifact")
    try:
        phx.meshing.prepare_boundary_layer_extrusion(
            source, request, destination=destination, overwrite=True
        )
    except phx.meshing.MeshingFailure as error:
        assert reason in str(error)
        assert source.model_id == original
        assert destination.read_bytes() == b"accepted artifact"
        assert not tuple(destination.parent.glob(f".{destination.name}.extrusion-*"))
        return dict(source=source, control=request, rejection=str(error))
    raise AssertionError(
        f"_hybrid_qualification_original_layer_cad_rejection.constructor must refuse the invalid CAD slab; got {'accepted'!r}."
    )


def _hybrid_qualification_original_layer_sweep_build(
    profile: str, destination: str | Path | None = None
) -> dict[str, Any]:
    if profile.startswith("cad-"):
        if destination is None:
            raise ValueError(
                f"_hybrid_qualification_original_layer_sweep_build.destination must be an archive path for native CAD profiles; got {destination!r}."
            )
        if profile == "cad-slab-and-core":
            return _hybrid_qualification_original_layer_cad(destination, True)
        if profile == "cad-entire-solid-slab":
            return _hybrid_qualification_original_layer_cad(destination, False)
        if profile == "cad-outside-controlled-solid":
            return _hybrid_qualification_original_layer_cad_rejection(destination, False)
        if profile == "cad-overlapping-wall-slabs":
            return _hybrid_qualification_original_layer_cad_rejection(destination, True)
    if profile == "periodic-convex-ridge":
        return _hybrid_qualification_original_layer_ridge()
    periodic_profiles = {
        f"periodic-{periodicity}-{kind}": (periodicity, kind == "quadrilateral")
        for periodicity in ("translation", "partial", "rotation")
        for kind in ("triangle", "quadrilateral")
    }
    if profile in periodic_profiles:
        return _hybrid_qualification_original_layer_periodic(*periodic_profiles[profile])
    exact_profiles = {
        f"exact-sweep-{'periodic' if periodic else 'nonperiodic'}-{'perturbed-cap' if perturb else 'accepted'}": (
            periodic,
            perturb,
        )
        for periodic in (False, True)
        for perturb in (False, True)
    }
    if profile in exact_profiles:
        return _hybrid_qualification_original_layer_exact(*exact_profiles[profile])
    raise ValueError(
        f"_hybrid_qualification_original_layer_sweep_build.profile must identify a registered layer sweep profile; got {profile!r}."
    )


def _hybrid_qualification_original_layer_sweep_profile(
    profile: str, destination: str | Path | None = None
) -> dict[str, Any]:

    if profile.startswith("cad-") and destination is None:
        raise ValueError(
            f"_hybrid_qualification_original_layer_sweep_profile.destination must be an archive path for native CAD profiles; got {destination!r}."
        )
    owners = _hybrid_qualification_original_layer_sweep_build(profile, destination)
    control = owners["control"]
    source = owners["source"]
    evidence = {
        "profile": profile,
        "independent_oracles_passed": True,
        "stage_scope": (
            "native-cad-extrusion"
            if "extrusion" in owners
            else "expected-source-rejection"
            if "rejection" in owners
            else "independent-exact-layer"
            if "physical" in owners
            else "physical-advancing-layer"
        ),
        **_hybrid_original_source_evidence(source),
        "route": control.route.name,
        "schedule_thicknesses": np.asarray(control.schedule.thicknesses).tolist(),
        "schedule_bits_sha256": hashlib.sha256(
            np.asarray(control.schedule.thicknesses).tobytes()
        ).hexdigest(),
        "core_mesh_present": False,
    }
    wall = owners.get("wall")
    if wall is not None:
        evidence["wall_coordinate_bits_sha256"] = hashlib.sha256(
            np.asarray(wall.coordinates).tobytes()
        ).hexdigest()
        evidence["wall_connectivity_bits_sha256"] = [
            hashlib.sha256(np.asarray(block.vertices).tobytes()).hexdigest()
            for block in wall.blocks
        ]
        descriptor = wall.periodic_topology
        evidence["periodic_topology_id"] = (
            None if descriptor is None else str(descriptor.periodic_topology_id)
        )
    layer = owners.get("layer")
    if layer is not None:
        evidence["result_id"] = str(layer.result_id)
        evidence["achieved_thicknesses"] = np.asarray(
            layer.evidence.achieved_thicknesses
        ).tolist()
        evidence["cap_present"] = layer.cap is not None
    extrusion = owners.get("extrusion")
    if extrusion is not None:
        evidence["slab_volumes"] = np.asarray(extrusion.slab_volumes).tolist()
        evidence["layer_solid_ids"] = [int(value) for value in extrusion.layer_solid_ids]
        evidence["core_solid_ids"] = [int(value) for value in extrusion.core_solid_ids]
        evidence["maximum_relative_volume_residual"] = float(
            extrusion.maximum_relative_volume_residual
        )
        evidence["source_units"] = "MILLIMETER"
        evidence["tessellation_linear_deflection"] = 0.05
        evidence["tessellation_angular_deflection"] = 0.2
    original_source = owners.get("original_source")
    if original_source is not None:
        evidence["original_source_id"] = str(original_source.source_id)
        evidence["original_source_revision"] = str(original_source.source_revision)
        evidence["original_vertex_bits_sha256"] = hashlib.sha256(
            np.asarray(original_source.geometry.vertex_points).tobytes()
        ).hexdigest()
    if "rejection" in owners:
        evidence["rejection"] = owners["rejection"]
    if "source_domain" in owners:
        owners["domain"] = owners["source_domain"]
    owners["evidence"] = evidence
    return owners


def _hybrid_qualification_original_cell_counts(result: Any) -> Any:
    return dict(result.evidence.cell_counts)


def _hybrid_qualification_original_assert_certified(result: Any) -> None:
    cells = sum((block.cell_count for block in result.mesh.blocks))
    assert result.validity.certified_valid_count == cells
    assert result.evidence.certified_valid_count == cells


def _hybrid_qualification_original_plate5() -> tuple[Any, Any, Any]:
    schedule = original_wall_layer_schedule()
    points, triangles = _hybrid_qualification_original_plate(5)
    wall = _hybrid_qualification_original_wall(points, triangles)
    result = phx.meshing.prepare_boundary_layers(
        wall, (control := _hybrid_qualification_original_control(wall, schedule))
    )
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses, schedule.thicknesses, rtol=1e-12
    )
    np.testing.assert_allclose(result.evidence.achieved_growth_rates, 1.2, rtol=1e-12)
    active = np.asarray(result.evidence.layer_active)
    assert active.dtype == np.bool_
    np.testing.assert_array_equal(active, (True, True, True))
    evidence_values = {
        "requested_thicknesses": result.evidence.requested_thicknesses,
        "achieved_thicknesses": result.evidence.achieved_thicknesses,
        "minimum_thicknesses": result.evidence.minimum_thicknesses,
        "maximum_thicknesses": result.evidence.maximum_thicknesses,
        "achieved_growth_rates": result.evidence.achieved_growth_rates,
        "layer_active": result.evidence.layer_active,
        "column_thicknesses": result.evidence.column_thicknesses,
        "column_active": result.evidence.column_active,
        "active_column_counts": result.evidence.active_column_counts,
        "column_count": result.evidence.column_count,
        "fan_column_count": result.evidence.fan_column_count,
        "corner_patch_count": result.evidence.corner_patch_count,
        "convex_ridge_count": result.evidence.convex_ridge_count,
        "concave_ridge_count": result.evidence.concave_ridge_count,
        "rim_vertex_count": result.evidence.rim_vertex_count,
        "minimum_visibility": result.evidence.minimum_visibility,
        "maximum_stretch": result.evidence.maximum_stretch,
        "unconverged_visibility_count": result.evidence.unconverged_visibility_count,
        "collision_policy": result.evidence.collision_policy,
        "predicted_collision_vertex_count": result.evidence.predicted_collision_vertex_count,
        "detected_collision_count": result.evidence.detected_collision_count,
        "resolution_iterations": result.evidence.resolution_iterations,
        "reduced_vertex_count": result.evidence.reduced_vertex_count,
        "terminated_vertex_count": result.evidence.terminated_vertex_count,
        "merged_vertex_count": result.evidence.merged_vertex_count,
        "minimum_scale": result.evidence.minimum_scale,
        "cell_counts": result.evidence.cell_counts,
        "certified_valid_count": result.evidence.certified_valid_count,
    }
    try:
        phx.meshing.BoundaryLayerEvidence(
            **{**evidence_values, "achieved_thicknesses": np.ones((2,), dtype=np.float64)}
        )
    except ValueError as original_failure:
        assert "invalid shapes" in str(original_failure)
    else:
        raise AssertionError(
            f"_hybrid_qualification_original_plate5.constructor must refuse the declared invalid controls; got {'accepted'!r}."
        )
    assert _hybrid_qualification_original_cell_counts(result) == {
        "prism": 3 * triangles.shape[0]
    }
    heights = np.unique(np.round(np.asarray(result.mesh.coordinates)[:, 2], 12))
    np.testing.assert_allclose(heights, (0.0, 0.01, 0.022, 0.0364))
    _hybrid_qualification_original_assert_certified(result)
    assert not result.closed_cap
    return (wall, control, result)


def _hybrid_qualification_original_quadrilateral_plate3() -> tuple[Any, Any, Any]:

    schedule = original_wall_layer_schedule()
    values = np.linspace(0.0, 1.0, 4)
    x, y = np.meshgrid(values, values, indexing="ij")
    points = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)
    index = np.arange(points.shape[0]).reshape(4, 4)
    quads = np.asarray(
        [
            (index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1])
            for i in range(3)
            for j in range(3)
        ]
    )
    wall = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "quadrilaterals", "quadrilateral", quads, global_ids=np.arange(9)
            ),
        ),
    )
    result = phx.meshing.prepare_boundary_layers(
        wall, (control := _hybrid_qualification_original_control(wall, schedule))
    )
    assert _hybrid_qualification_original_cell_counts(result) == {
        "pyramid": 9,
        "hexahedron": 27,
    }
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses, schedule.thicknesses, rtol=1e-12
    )
    cap = result.cap
    if cap is None:
        raise qualification_failure(
            "_hybrid_qualification_original_quadrilateral_plate3",
            ("Quadrilateral wall omitted its triangular core cap."),
            {"cap": cap},
        )
    assert {block.cell_kind for block in cap.blocks} == {"triangle"}
    layer_index = np.asarray(result.layer_index)
    pyramids = result.mesh.block("pyramids")
    np.testing.assert_array_equal(layer_index[np.asarray(pyramids.global_ids)], 3)
    _hybrid_qualification_original_assert_certified(result)
    return (wall, control, result)


def _hybrid_qualification_original_opposing_sloped_plates4() -> tuple[Any, Any, Any]:
    schedule = original_wall_layer_schedule()
    lower, lower_triangles = _hybrid_qualification_original_plate(4)
    upper, upper_triangles = _hybrid_qualification_original_plate(4, flip=True)
    upper[:, 2] = 0.03 + 0.17 * upper[:, 0]
    wall = _hybrid_qualification_original_wall(
        np.concatenate((lower, upper)),
        np.concatenate((lower_triangles, upper_triangles + lower.shape[0])),
    )
    result = phx.meshing.prepare_boundary_layers(
        wall,
        (
            control := _hybrid_qualification_original_control(
                wall,
                schedule,
                collision=phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY,
            )
        ),
    )
    evidence = result.evidence
    assert 0 < evidence.terminated_vertex_count < 50
    counts = _hybrid_qualification_original_cell_counts(result)
    assert counts["pyramid"] > 0 and counts["tetrahedron"] > 0
    np.testing.assert_allclose(
        evidence.achieved_thicknesses[0],
        0.01,
        rtol=0.0,
        atol=max(1e-12, 1e-06 * abs(0.01)),
    )
    _hybrid_qualification_original_assert_certified(result)
    return (wall, control, result)


def _hybrid_qualification_original_sphere_radius04_subdivision2() -> tuple[Any, Any, Any]:

    schedule = original_wall_layer_schedule()
    points, triangles = _hybrid_qualification_original_icosphere(0.4, 2)
    wall = _hybrid_qualification_original_wall(points, triangles)
    result = phx.meshing.prepare_boundary_layers(
        wall, (control := _hybrid_qualification_original_control(wall, schedule))
    )
    evidence = result.evidence
    np.testing.assert_allclose(
        evidence.minimum_thicknesses, schedule.thicknesses, rtol=1e-06
    )
    np.testing.assert_allclose(
        evidence.maximum_thicknesses, schedule.thicknesses, rtol=1e-06
    )
    np.testing.assert_allclose(evidence.achieved_growth_rates, 1.2, rtol=1e-06)
    assert evidence.fan_column_count == 0
    assert result.closed_cap
    cap = result.cap
    if cap is None:
        raise qualification_failure(
            "_hybrid_qualification_original_sphere_radius04_subdivision2",
            ("Sphere wall omitted its closed core cap."),
            {"cap": cap},
        )
    radii = np.linalg.norm(np.asarray(cap.coordinates), axis=1)
    assert np.all(radii > 0.4)
    _hybrid_qualification_original_assert_certified(result)
    return (wall, control, result)


def _hybrid_qualification_original_outward_box_count3() -> tuple[Any, Any, Any]:

    schedule = original_wall_layer_schedule()
    points, triangles = _hybrid_qualification_original_box(
        (-0.3, -0.3, -0.3), (0.3, 0.3, 0.3), 3, outward=True
    )
    wall = _hybrid_qualification_original_wall(points, triangles)
    result = phx.meshing.prepare_boundary_layers(
        wall, (control := _hybrid_qualification_original_control(wall, schedule))
    )
    evidence = result.evidence
    assert evidence.convex_ridge_count == 12 * 3
    assert evidence.corner_patch_count == 8
    assert evidence.fan_column_count > 0
    counts = _hybrid_qualification_original_cell_counts(result)
    assert (
        counts["hexahedron"] > 0 and counts["tetrahedron"] > 0 and (counts["pyramid"] > 0)
    )
    np.testing.assert_allclose(
        evidence.achieved_thicknesses, schedule.thicknesses, rtol=1e-09
    )
    assert result.closed_cap
    cap = result.cap
    if cap is None:
        raise qualification_failure(
            "_hybrid_qualification_original_outward_box_count3",
            ("Outward box omitted its closed core cap."),
            {"cap": cap},
        )
    assert {block.cell_kind for block in cap.blocks} == {"triangle"}
    _hybrid_qualification_original_assert_certified(result)
    return (wall, control, result)


def _hybrid_qualification_original_concave_box_count3() -> tuple[Any, Any, Any]:
    schedule = original_wall_layer_schedule()
    points, triangles = _hybrid_qualification_original_box(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 3, outward=False
    )
    wall = _hybrid_qualification_original_wall(points, triangles)
    result = phx.meshing.prepare_boundary_layers(
        wall, (control := _hybrid_qualification_original_control(wall, schedule))
    )
    evidence = result.evidence
    assert evidence.concave_ridge_count == 12 * 3
    assert evidence.fan_column_count == 0
    np.testing.assert_allclose(
        evidence.maximum_stretch,
        np.sqrt(3.0),
        rtol=0.0,
        atol=max(1e-12, 1e-06 * abs(np.sqrt(3.0))),
    )
    np.testing.assert_allclose(
        evidence.achieved_thicknesses, schedule.thicknesses, rtol=1e-09
    )
    _hybrid_qualification_original_assert_certified(result)
    return (wall, control, result)


def _hybrid_qualification_original_wedge30_rejection() -> tuple[Any, Any, Any]:
    schedule = original_wall_layer_schedule()
    count = 3
    values = np.linspace(0.0, 1.0, count + 1)
    angle = np.deg2rad(30.0)
    floor = [(r, w, 0.0) for r in values for w in values]
    slope = [
        (r * np.cos(angle), w, r * np.sin(angle)) for r in values[1:] for w in values
    ]
    points = np.asarray(floor + slope)
    first = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)
    second = np.vstack(
        (first[:1], first.size + np.arange(count * (count + 1)).reshape(count, -1))
    )
    triangles = []
    for grid in (first, second):
        for i in range(count):
            for j in range(count):
                a, b, c, d = (
                    grid[i, j],
                    grid[i + 1, j],
                    grid[i + 1, j + 1],
                    grid[i, j + 1],
                )
                triangles.extend(((a, b, c), (a, c, d)))
    triangles = np.asarray(triangles)
    inside = np.asarray((0.5 * np.cos(0.5 * angle), 0.5, 0.5 * np.sin(0.5 * angle)))
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flip = np.sum(normals * (inside - corners.mean(axis=1)), axis=1) < 0.0
    triangles[flip] = triangles[flip][:, ::-1]
    wall = _hybrid_qualification_original_wall(points, triangles)
    try:
        phx.meshing.prepare_boundary_layers(
            wall, (control := _hybrid_qualification_original_control(wall, schedule))
        )
    except phx.meshing.MeshingFailure as original_failure:
        failure = original_failure
    else:
        raise AssertionError(
            f"_hybrid_qualification_original_wedge30_rejection.constructor must refuse the declared invalid controls; got {'accepted'!r}."
        )
    assert failure.category is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    crease = np.flatnonzero(np.isclose(points[:, 0], 0.0) & np.isclose(points[:, 2], 0.0))
    assert set(failure.entity_ids) == {int(value) for value in crease}
    np.testing.assert_allclose(failure.locations, points[crease])
    return (wall, control, failure)


def _hybrid_qualification_original_open_box_floor_count4() -> tuple[Any, Any, Any]:
    schedule = original_wall_layer_schedule()
    points, triangles = _hybrid_qualification_original_box(
        (0.0, 0.0, 0.0), (1.0, 1.0, 0.5), 4, outward=False
    )
    heights = points[triangles][:, :, 2]
    open_box = triangles[~np.all(np.isclose(heights, 0.5), axis=1)]
    wall = _hybrid_qualification_original_wall(points, open_box)
    cells = wall.entity_set(2)
    floor = np.flatnonzero(np.all(np.isclose(points[open_box][:, :, 2], 0.0), axis=1))
    scope = phx.meshing.MeshingScope(
        wall.mesh_id,
        wall.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        np.asarray(cells.entity_ids)[floor],
    )
    control = phx.meshing.BoundaryLayerControl(
        scope, schedule, route=phx.meshing.BoundaryLayerRoute.ADVANCING
    )
    result = phx.meshing.prepare_boundary_layers(wall, control)
    assert result.evidence.rim_vertex_count == 16
    coordinates = np.asarray(result.mesh.coordinates)
    assert np.all((coordinates[:, :2] >= 0.0) & (coordinates[:, :2] <= 1.0))
    on_side = np.count_nonzero(coordinates[:, 0] == 0.0)
    assert on_side == 5 * 4
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses, schedule.thicknesses, rtol=1e-12
    )
    assert not result.closed_cap
    _hybrid_qualification_original_assert_certified(result)
    return (wall, control, result)


def _hybrid_qualification_original_channel_gap005_rejection() -> tuple[Any, Any, Any]:
    schedule = original_wall_layer_schedule()
    wall = _hybrid_qualification_original_channel(0.05)
    try:
        phx.meshing.prepare_boundary_layers(
            wall, (control := _hybrid_qualification_original_control(wall, schedule))
        )
    except phx.meshing.MeshingFailure as original_failure:
        failure = original_failure
    else:
        raise AssertionError(
            f"_hybrid_qualification_original_channel_gap005_rejection.constructor must refuse the declared invalid controls; got {'accepted'!r}."
        )
    assert failure.category is phx.meshing.MeshingFailureCategory.CONTROL_CONFLICT
    assert len(failure.entity_ids) == 50
    assert len(failure.locations) == 50
    return (wall, control, failure)


def _hybrid_qualification_original_channel_gap005_policies() -> tuple[
    tuple[Any, Any, Any], ...
]:
    schedule = original_wall_layer_schedule()
    results = []
    for collision in (
        phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS,
        phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY,
        phx.meshing.BoundaryLayerCollisionPolicy.MERGE,
    ):
        gap = 0.05
        wall = _hybrid_qualification_original_channel(gap)
        result = phx.meshing.prepare_boundary_layers(
            wall,
            (
                control := _hybrid_qualification_original_control(
                    wall, schedule, collision=collision, minimum_thickness_fraction=0.2
                )
            ),
        )
        evidence = result.evidence
        assert evidence.collision_policy is collision
        assert evidence.predicted_collision_vertex_count == 50
        heights = np.asarray(result.mesh.coordinates)[:, 2]
        match collision:
            case phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS:
                assert 0.2 <= evidence.minimum_scale < 1.0
                assert evidence.reduced_vertex_count == 50
                assert np.max(heights[heights < 0.5 * gap]) < 0.5 * gap
                np.testing.assert_allclose(
                    evidence.achieved_growth_rates, 1.2, rtol=1e-09
                )
                assert np.all(np.asarray(evidence.layer_active))
            case phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY:
                assert evidence.terminated_vertex_count == 50
                assert _hybrid_qualification_original_cell_counts(result) == {
                    "prism": 2 * 32
                }
                achieved = np.asarray(evidence.achieved_thicknesses)
                active = np.asarray(evidence.layer_active)
                np.testing.assert_allclose(
                    achieved[0], 0.01, rtol=0.0, atol=max(1e-12, 1e-06 * abs(0.01))
                )
                np.testing.assert_array_equal(active, (True, False, False))
                np.testing.assert_array_equal(np.isnan(achieved), ~active)
                assert np.all(np.isnan(evidence.achieved_growth_rates))
            case phx.meshing.BoundaryLayerCollisionPolicy.MERGE:
                assert evidence.merged_vertex_count == 50
                np.testing.assert_allclose(
                    evidence.minimum_scale,
                    0.5 * gap / schedule.total_thickness,
                    rtol=0.0,
                    atol=max(1e-12, 1e-06 * abs(0.5 * gap / schedule.total_thickness)),
                )
                assert np.any(np.isclose(heights, 0.5 * gap))
                assert result.cap is None
        _hybrid_qualification_original_assert_certified(result)
        results.append((wall, control, result))
    return tuple(results)


def _hybrid_qualification_original_wall_profile(name: str) -> dict[str, Any]:
    """Run only the selected wall/layer constructor and its full oracle."""

    if name not in _hybrid_qualification_original_profile_names:
        raise ValueError(
            f"_hybrid_qualification_original_wall_profile.name must identify a registered wall profile; got {name!r}."
        )

    constructors: dict[str, Callable[[], Any]] = {
        "plate5": _hybrid_qualification_original_plate5,
        "quadrilateral_plate3": _hybrid_qualification_original_quadrilateral_plate3,
        "opposing_sloped_plates4": _hybrid_qualification_original_opposing_sloped_plates4,
        "sphere_radius04_subdivision2": _hybrid_qualification_original_sphere_radius04_subdivision2,
        "outward_box_count3": _hybrid_qualification_original_outward_box_count3,
        "concave_box_count3": _hybrid_qualification_original_concave_box_count3,
        "wedge30_rejection": _hybrid_qualification_original_wedge30_rejection,
        "open_box_floor_count4": _hybrid_qualification_original_open_box_floor_count4,
        "channel_gap005_rejection": _hybrid_qualification_original_channel_gap005_rejection,
        "channel_gap005_policies": _hybrid_qualification_original_channel_gap005_policies,
    }
    outcome: Any = constructors[name]()
    pairs: tuple[tuple[Any, Any, Any], ...] = (
        outcome if name == "channel_gap005_policies" else (outcome,)
    )
    rejected = name.endswith("_rejection")
    owners = []
    receipts = []
    for wall, control, result in pairs:
        receipt = {
            "source_bits_fingerprint": array_tree_fingerprint((wall, control)),
            "wall_mesh_id": str(wall.mesh_id),
            "wall_numeric_version": str(wall.numeric_version),
        }
        if rejected:
            receipt.update(
                category=result.category.name,
                entity_ids=[int(value) for value in result.entity_ids],
                locations=np.asarray(result.locations).tolist(),
            )
            owners.append({"wall": wall, "control": control, "failure": result})
        else:
            receipt.update(
                source_layer_bits_fingerprint=array_tree_fingerprint(
                    (wall, control, result)
                ),
                layer_mesh_id=str(result.mesh.mesh_id),
                cell_counts={
                    str(kind): int(count) for kind, count in result.evidence.cell_counts
                },
                certified_valid_count=int(result.evidence.certified_valid_count),
                closed_cap=bool(result.closed_cap),
                cap_mesh_id=None if result.cap is None else str(result.cap.mesh_id),
            )
            owners.append(
                {"wall": wall, "control": control, "layers": result, "cap": result.cap}
            )
        receipts.append(receipt)
    closed_original = name in {
        "sphere_radius04_subdivision2",
        "outward_box_count3",
        "concave_box_count3",
    }
    evidence = {
        "profile": name,
        "source": "tests/unit/meshing/test_boundary_layer.py scenarios 1-3",
        "original_controls": _hybrid_qualification_original_controls[name],
        "schedule": {"count": 3, "first_thickness": 0.01, "growth_rate": 1.2},
        "stage_scope": "original wall/control rejection"
        if rejected
        else "original wall and certified boundary-layer preparation only",
        "original_oracles_passed": True,
        "core_executed": False,
        "lifecycle_executed": False,
        "receipts": receipts,
        "full_core_boundary_composition": {
            "original_wall_closed": closed_original,
            "admissible_information": "Concave box is an authored closed container; its inward layer cap is the remaining-core boundary. No core mesh is authored by the original scenario."
            if name == "concave_box_count3"
            else "Original open box retains the four side walls and authors a floor-only layer scope; the top at z=0.5 was explicitly removed. Its exact discarded top triangles are recoverable from the original box constructor, but a closed cap/side/top core composition and its orientation were not authored by this oracle."
            if name == "open_box_floor_count4"
            else "Original outward closed wall and closed layer cap describe an obstacle and its outer layer front, not a bounded exterior core."
            if closed_original
            else "Original open sheets or floor-only wall selection do not author a closed remaining-core boundary; merged channel fronts have no free cap.",
            "missing_authoring": "Core meshing request and original independent core/lifecycle oracle."
            if name == "concave_box_count3"
            else "Explicit remaining side-wall trimming, top inclusion, closed core boundary composition, core request, and independent core/lifecycle oracle; no new enclosing geometry is justified."
            if name == "open_box_floor_count4"
            else "Original enclosing outer boundary, core boundary composition, core meshing request, and independent core/lifecycle oracle.",
        },
    }
    evidence["source"] = (
        "tools/meshing_qualification.py::_hybrid_qualification_original_wall_profile"
    )
    evidence["original_lineage"] = (
        "tests/unit/meshing/test_boundary_layer.py scenarios 1-3"
    )
    if name == "concave_box_count3":
        wall, control, layers = pairs[0]
        if layers.cap is None or not layers.closed_cap:
            raise AssertionError(
                f"_hybrid_qualification_original_wall_profile.cap must contain a closed remaining-core cap; got {(layers.cap, layers.closed_cap)!r}."
            )
        if any(block.cell_kind != "triangle" for block in layers.cap.blocks):
            raise AssertionError(
                f"_hybrid_qualification_original_wall_profile.cap_blocks must contain only 'triangle' rows; got {tuple(block.cell_kind for block in layers.cap.blocks)!r}."
            )
        cap_rows = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in layers.cap.blocks]
        )
        core = phx.meshing.PiecewiseLinearComplex(
            np.asarray(layers.cap.coordinates, dtype=np.float64),
            tuple(cap_rows),
            np.arange(cap_rows.shape[0], dtype=np.int64),
            np.tile(np.asarray((0, -1), dtype=np.int64), (cap_rows.shape[0], 1)),
            ("fluid",),
            boundary="fixed",
        )
        core_source = phx.meshing.NativeLayerCoreSource(
            layers,
            core,
            str(wall.mesh_id),
            str(wall.numeric_version),
            vertex_layer_ids=np.asarray(layers.cap_vertices, dtype=np.int64),
            cap_polygon_ids=np.arange(cap_rows.shape[0], dtype=np.int64),
            layer_regions=np.zeros(
                sum(block.cell_count for block in layers.mesh.blocks), dtype=np.int64
            ),
            region_ids=("fluid",),
            core_region_map=np.asarray((0,), dtype=np.int64),
        )
        owners[0]["core_source"] = core_source
        owners[0]["core_complex"] = core
        evidence["core_source_binding_id"] = core_source.binding_id
        evidence["core_source_bits_fingerprint"] = array_tree_fingerprint(core_source)
        evidence["stage_scope"] = (
            "original wall and certified layers with exact unchanged closed-cap core-source composition; no core fill or lifecycle"
        )
        evidence["full_core_boundary_composition"]["missing_authoring"] = (
            "Core fill request and independent core/lifecycle oracle; original wall mesh scope authors no geometry-domain association or fidelity query."
        )
    answer = {"evidence": evidence, "owners": tuple(owners)}
    if len(owners) == 1:
        answer.update(owners[0])
    return answer


def _hybrid_qualification_archived_profile(
    args: QualificationControls,
    record: dict[str, Any],
    *,
    retained_controls: MeshingSourceExecutionControls | None = None,
    expected_rounds: int | None = None,
) -> _HybridContinuationEvidence:
    """Cold-continue the archive's accepted state once, in a remaining-budget process.

    The archive must retain ``expected_rounds`` (default: the frozen
    adaptation rounds) refine/coarsen pairs of accepted epochs.
    """

    densities, material_site = _hybrid_material_controls(
        args.hybrid_profile, args.hybrid_region_densities, args.hybrid_material_site
    )
    rounds = args.adaptation_rounds if expected_rounds is None else expected_rounds

    archive, content_id = args.hybrid_archive, args.hybrid_content_id
    if not archive or not content_id:
        raise ValueError(
            f"_hybrid_qualification_archived_profile.archive must have an archive path and content identity; got {(archive, content_id)!r}."
        )

    def authenticated_controls() -> MeshingSourceExecutionControls:
        if retained_controls is None:
            return read_meshing_source_execution_controls(
                archive, expected_content_id=content_id
            )
        if (
            type(retained_controls) is not MeshingSourceExecutionControls
            or retained_controls.content_id != args.hybrid_content_id
        ):
            raise ValueError(
                f"authenticated_controls.retained_controls must be MeshingSourceExecutionControls bound to the archive content ID; got {retained_controls!r}."
            )
        retained_controls.execution_evidence.require_valid()
        return retained_controls

    controls = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run("authenticated_original_execution_controls", authenticated_controls)
    record["source"] = {
        "source_id": controls.source_id,
        "source_revision": controls.source_revision,
        "source_binding_id": controls.source_binding_id,
        "specification_id": controls.specification_id,
        "archive": str(args.hybrid_archive),
        "content_id": controls.content_id,
        "stage_scope": "original-authenticated-accepted-state-cold-continuation-allfields-FV-history-PDE",
        "controls_only": True,
        "controls_origin": (
            "authenticated-archive"
            if retained_controls is None
            else "retained-live-producer-evidence"
        ),
    }
    remaining = args.timeout - (perf_counter() - record["started"])
    if remaining <= 0.0:
        raise TimeoutError(
            f"_hybrid_qualification_archived_profile.remaining must be positive before the fresh-process consumer; got {remaining!r}."
        )
    # A single-phase consumer: the archive already exists, so prepare and execute
    # run back to back.
    with prepare_cold_worker(
        "tools.meshing_qualification.scenarios.hybrid",
        deadline=Deadline(perf_counter() + remaining),
        ledger=StageLedger(
            record,
            cold_phase="original_fresh_process_accepted_state_allfields_FV_history_PDE",
        ),
    ) as cold_worker:
        cold = cold_worker.execute(
            Path(archive),
            controls.content_id,
            {
                "region_densities": list(densities),
                "material_site": material_site,
                "solver_repeats": args.repeats,
            },
        )
    evidence = cold["evidence"]
    record["cold"] = {
        "returncode": 0,
        "stdout": "\n".join(json.dumps(item) for item in evidence),
        "stderr": "",
        "archive_limits": "DEFAULT_ARRAY_ARCHIVE_LIMITS",
    }
    if not evidence or (
        evidence[0].get("source_id"),
        evidence[0].get("source_revision"),
    ) != (
        controls.source_id,
        controls.source_revision,
    ):
        raise qualification_failure(
            "_hybrid_qualification_archived_profile",
            ("Fresh-process hybrid continuation lost its authenticated source."),
            {"controls": controls, "evidence": evidence},
        )
    cold_stages = evidence[0].get("stages_seconds")
    if (
        not isinstance(cold_stages, dict)
        or not cold_stages
        or any(
            type(name) is not str
            or not name
            or not isinstance(seconds, (int, float))
            or not np.isfinite(seconds)
            or seconds < 0.0
            for name, seconds in cold_stages.items()
        )
    ):
        raise qualification_failure(
            "_hybrid_qualification_archived_profile",
            ("Fresh-process hybrid continuation omitted finite owned phase timings."),
            {"cold_stages": cold_stages},
        )
    record["cold_stages_seconds"] = cold_stages
    for section, output in (
        ("fields", "allfields"),
        ("finite_volume", "finite_volume"),
        ("history", "history"),
    ):
        values = [item[section] for item in evidence if section in item]
        if len(values) != 1:
            raise qualification_failure(
                "_hybrid_qualification_archived_profile",
                (f"Cold hybrid omitted or duplicated its {section} consumer."),
                {"values": values},
            )
        record[output] = values[0]
    original = [item["original_scalar"] for item in evidence if "original_scalar" in item]
    if len(original) > 1 or (args.hybrid_profile == "unit-box" and len(original) != 1):
        raise qualification_failure(
            "_hybrid_qualification_archived_profile",
            "Cold unit-box must independently continue its actual original P1 bank.",
            {"original_scalar": original},
        )
    if original:
        scalar = original[0]
        independent = scalar.get("independent_solve")
        if (
            scalar.get("field") != "original-u"
            or scalar.get("degree") != 1
            or not isinstance(scalar.get("physical"), dict)
            or not isinstance(scalar.get("independent_physical"), dict)
            or not isinstance(independent, dict)
            or not independent.get("compiler", {}).get("warm_samples_seconds")
        ):
            raise qualification_failure(
                "_hybrid_qualification_archived_profile",
                "Cold original P1 continuation omitted its physical target or independent solve.",
                {"original_scalar": scalar},
            )
        record["original_scalar"] = scalar
    declarations = evidence[0].get("material_declaration")
    if (
        not isinstance(declarations, dict)
        or declarations.get("densities") != list(densities)
        or declarations.get("site_id") != material_site
    ):
        raise qualification_failure(
            "_hybrid_qualification_archived_profile",
            ("Cold hybrid changed its declared source-bound material decision."),
            {"args": args, "declarations": declarations},
        )
    record["material_declaration"] = declarations
    restored = evidence[0].get("restored_epoch_ids")
    if (
        not isinstance(restored, list)
        or len(restored) != 2 * rounds + 1
        or any(type(value) is not str or not value for value in restored)
        or not isinstance(evidence[0].get("continued_from"), dict)
        or evidence[0]["continued_from"].get("epoch_id") != restored[-2]
    ):
        raise qualification_failure(
            "_hybrid_qualification_archived_profile",
            (
                f"Cold hybrid must restore {2 * rounds + 1} archived accepted epochs and continue the last refined one."
            ),
            {"restored": restored, "continued_from": evidence[0].get("continued_from")},
        )
    record["restored_epochs"] = evidence[0]["restored_epochs"]
    record["continued_from"] = evidence[0]["continued_from"]
    record["warm_continuation_defects"] = evidence[0]["warm_continuation_defects"]
    record["restoration_defects"] = evidence[0]["restoration_defects"]
    receipts = [item for item in evidence if "execution_evidence" in item]
    if len(receipts) != 1 or receipts[0].get("limits_id") != controls.limits.limits_id:
        raise qualification_failure(
            "_hybrid_qualification_archived_profile",
            ("Cold hybrid omitted ended source-scope evidence."),
            {"controls": controls, "receipts": receipts},
        )
    record["physical_error"] = max(
        stage["pde_defect"] for stage in record["history"]["stages"]
    )
    record["physical_error_scope"] = (
        "maximum-nodal-PDE-defect-of-the-cold-continued-accepted-state"
    )
    if original:
        record["physical_error"] = max(
            record["physical_error"],
            record["original_scalar"]["physical"]["l2_error"],
            record["original_scalar"]["independent_physical"]["l2_error"],
        )
        record["physical_error_scope"] = (
            "maximum-of-original-P1-transferred-and-independent-mapped-physical-L2-and-canonical-history-PDE-defect"
        )
    if (
        not np.isfinite(record["physical_error"])
        or record["physical_error"] > args.target_error
    ):
        raise qualification_failure(
            "_hybrid_qualification_archived_profile",
            ("Continued PDE missed its declared physical error target."),
            {"args": args, "record": record},
        )
    epochs = record["history"].get("epochs")
    source_result = evidence[0].get("source_result")
    if (
        not isinstance(epochs, list)
        or [epoch.get("name") for epoch in epochs] != ["coarsen"]
        or any(
            not isinstance(epoch.get("compiler"), dict)
            or not epoch["compiler"].get("warm_samples_seconds")
            or not isinstance(epoch.get("result"), dict)
            or not isinstance(epoch.get("retained_bytes"), int)
            for epoch in epochs
        )
        or not isinstance(source_result, dict)
    ):
        raise qualification_failure(
            "_hybrid_qualification_archived_profile",
            (
                "Cold hybrid omitted its continued accepted-state solve or source publication."
            ),
            {"epochs": epochs, "source_result": source_result},
        )
    record["cold"]["evidence"] = evidence
    record["source"]["controls_only"] = False
    record["independent_scientific_certification"] = "passed"
    record["status"] = "passed"

    return _HybridContinuationEvidence(
        record["physical_error"], tuple(epochs), source_result
    )


@final
class _HybridContinuationEvidence(phx.StrictModule):
    """Cold continuation error, accepted-state solve epochs and restored source."""

    physical_error: float
    epochs: tuple[dict[str, Any], ...] = eqx.field(static=True)
    source_result: dict[str, Any] = eqx.field(static=True)


def _hybrid_record_cold_epochs(
    record: dict[str, Any], continued: _HybridContinuationEvidence, cycle: int
) -> None:
    """Publish one cold continuation's accepted-state solves as campaign epochs."""

    for epoch in continued.epochs:
        index = len(record["epochs"])
        record["epochs"].append(
            {
                **epoch,
                "epoch": index,
                "name": f"{epoch['name']}_{cycle}",
                "cycle": cycle,
                "continuation_epoch": epoch["epoch"],
                "execution_process": "cold-archive-continuation",
            }
        )
        record["stages_seconds"].update(
            {
                epoch_stage_name(stage, index): seconds
                for stage, seconds in epoch["stages_seconds"].items()
            }
        )


@final
class _HybridOriginalPublication(phx.StrictModule):
    source: phx.meshing.NativeLayerCoreSource
    specification: phx.meshing.VolumeMeshingSpec
    options: phx.meshing.NativeMeshingOptions
    generated: phx.meshing.CellMeshingResult
    coverage: phx.geometry.DomainCoverageCertificate


@final
class _HybridAdaptationCycle(phx.StrictModule):
    refinement: phx.meshing.MeshAdaptationResult
    coarsening: phx.meshing.MeshAdaptationResult


@final
class _HybridWarmLineage(phx.StrictModule):
    """The warm producer's carried accepted state and its shared preparations."""

    plans: LayerCoreLifecyclePlans
    state: LayerCoreAcceptedState
    material: LayerCoreMaterialDeclaration = eqx.field(static=True)


def _hybrid_warm_cycle(
    args: QualificationControls,
    record: dict[str, Any],
    accepted: phx.meshing.CellMeshingResult,
    adaptation: _HybridAdaptationCycle,
    cycle: int,
    /,
    *,
    part_name: str,
    limits: phx.meshing.MeshingLimits,
    carried: _HybridWarmLineage | None,
    original_scalar_values: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    original_scalar_preparations: tuple[Any, Any, Any] | None = None,
    prepared_operands: PreparedCoordinateOperands | None = None,
    transition_operands: tuple[
        PreparedCoordinateOperands | None, PreparedCoordinateOperands | None
    ] = (None, None),
) -> tuple[tuple[MeshingAcceptedEpoch, ...], _HybridWarmLineage]:
    """Continue the accepted state warm through one cycle and checkpoint its epochs.

    The first cycle authors the initial accepted state on ``accepted``; later
    cycles carry the previous cycle's coarsened accepted state, never reseeding.
    """

    densities, material_site = _hybrid_material_controls(
        args.hybrid_profile, args.hybrid_region_densities, args.hybrid_material_site
    )
    ledger = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    )
    if carried is None:
        plans = LayerCoreLifecyclePlans()
        material = layer_core_material_declaration(
            accepted,
            densities=densities,
            site_id=material_site,
        )
        state, epoch, _, _ = ledger.run(
            f"warm_initial_accepted_state:{cycle}",
            lambda: warm_layer_core_initial_state(
                accepted,
                material=material,
                plans=plans,
                limits=limits,
                part_name=part_name,
                solver_repeats=args.repeats,
                current_preparation=None
                if original_scalar_preparations is None
                else original_scalar_preparations[0],
                prepared_operands=prepared_operands,
                original_scalar=(
                    None if original_scalar_values is None else original_scalar_values[0]
                ),
            ),
        )
    else:
        plans, material, state = carried.plans, carried.material, carried.state
        if state.result.result_id != accepted.result_id:
            raise qualification_failure(
                "_hybrid_warm_cycle",
                "The carried accepted state does not belong to this cycle's carrier.",
                {"state": state.result.result_id, "accepted": accepted.result_id},
            )
        if original_scalar_values is not None and (
            state.original_scalar is None
            or not np.array_equal(
                np.asarray(state.original_scalar), np.asarray(original_scalar_values[0])
            )
        ):
            raise qualification_failure(
                "_hybrid_warm_cycle",
                "The original P1 bank must be carried from the preceding accepted epoch without reseeding.",
                {"state_epoch": state.epoch, "cycle": cycle},
            )
        epoch = layer_core_accepted_epoch(state, part_name=part_name, plans=plans)
    refined, refined_epoch, refine_step, _ = ledger.run(
        f"warm_refinement_continuation:{cycle}",
        lambda: warm_layer_core_step(
            state,
            adaptation.refinement,
            material=material,
            plans=plans,
            limits=limits,
            part_name=part_name,
            solver_repeats=args.repeats,
            original_scalar=(
                None if original_scalar_values is None else original_scalar_values[1]
            ),
            current_preparation=None
            if original_scalar_preparations is None
            else original_scalar_preparations[1],
            prepared_operands=transition_operands[0],
        ),
    )
    coarsened, coarsened_epoch, coarsen_step, _ = ledger.run(
        f"warm_coarsening_continuation:{cycle}",
        lambda: warm_layer_core_step(
            refined,
            adaptation.coarsening,
            material=material,
            plans=plans,
            limits=limits,
            part_name=part_name,
            solver_repeats=args.repeats,
            restores=state.result,
            original_scalar=(
                None if original_scalar_values is None else original_scalar_values[2]
            ),
            current_preparation=None
            if original_scalar_preparations is None
            else original_scalar_preparations[2],
            prepared_operands=transition_operands[1],
        ),
    )
    record.setdefault("warm_continuations", []).append(
        {
            "cycle": cycle,
            "steps": [refine_step, coarsen_step],
            "restoration_defects": layer_core_restoration_defects(
                coarsened, epoch, plans=plans
            ),
        }
    )
    return (epoch, refined_epoch, coarsened_epoch), _HybridWarmLineage(
        plans, coarsened, material
    )


def _hybrid_concave_publication(
    args: QualificationControls, record: dict[str, Any]
) -> _HybridOriginalPublication:
    """Generate the fixed-cap core and retain its source and material coverage."""

    densities, material_site = _hybrid_material_controls(
        args.hybrid_profile, args.hybrid_region_densities, args.hybrid_material_site
    )
    owners = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "original_concave_wall_layers_and_cap",
        lambda: _hybrid_qualification_original_wall_profile("concave_box_count3"),
    )
    record["source_profile"] = owners["evidence"]
    source = owners["core_source"]
    limits = phx.meshing.MeshingLimits(maximum_wall_seconds=args.timeout)
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "original-closed-cap-core-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"), allow_mixed=True
            ),
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                args.hybrid_core_size,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        limits=limits,
    )
    options = phx.meshing.NativeMeshingOptions("layer_core")
    plan = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "original_closed_cap_core_plan",
        lambda: phx.meshing.NativeMeshingProvider(options).plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        ),
    )
    generated = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run("original_fixed_cap_core_publication", plan.execute)
    record["generation"] = _result_identity(generated)
    record["core_consumer_controls"] = {
        "specification_id": specification.specification_id,
        "size": args.hybrid_core_size,
        "limits_id": specification.limits.limits_id,
        "source_schedule_unchanged": list(source.layers.control.schedule.thicknesses),
    }
    if generated.certification is None or not generated.certification.passed:
        raise qualification_failure(
            "_hybrid_qualification_concave_lifecycle",
            ("Closed-cap core did not independently certify its source."),
            {"generated": generated},
        )

    record["material_declaration"] = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "original_source_material_declaration",
        lambda: layer_core_material_declaration(
            generated,
            densities=densities,
            site_id=material_site,
        ),
    )
    coverage = generated.certification.coverage
    if not isinstance(coverage, phx.geometry.DomainCoverageCertificate) or any(
        value is None for value in coverage.achieved_region_measures
    ):
        raise qualification_failure(
            "_hybrid_qualification_concave_lifecycle",
            ("Closed container lost its certified material measures."),
            {"coverage": coverage},
        )
    return _HybridOriginalPublication(source, specification, options, generated, coverage)


def _hybrid_concave_high_order_map(
    args: QualificationControls,
    record: dict[str, Any],
    publication: _HybridOriginalPublication,
) -> phx.meshing.CellMeshingResult:
    """Lift the unchanged source map and accept every high-order certificate."""

    source, specification, generated = (
        publication.source,
        publication.specification,
        publication.generated,
    )
    prepared = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "original_map_high_order_entity_node_lift",
        lambda: prepare_hybrid_high_order_map(
            generated,
            degree=args.geometry_order,
            maximum_nodes=32 * args.capacity,
            maximum_entries=1024 * args.capacity,
        ),
    )
    accepted = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "original_high_order_full_source_certification",
        lambda: qualify_hybrid_high_order_publication(
            source, specification, generated, prepared
        ),
    )
    if accepted.certification is None or not accepted.certification.passed:
        raise qualification_failure(
            "_hybrid_qualification_concave_lifecycle",
            ("Source high-order map did not pass full independent certification."),
            {"accepted": accepted},
        )
    record["geometry_decision"] = {
        "degree": args.geometry_order,
        "source_geometry": "unchanged original piecewise source",
        "construction": "family-correct nodal lift accepted by full original-source certificates; no projection or field-degree substitution",
        "acceptance": _result_identity(accepted),
    }
    return accepted


def _hybrid_concave_adaptation_cycle(
    args: QualificationControls,
    record: dict[str, Any],
    publication: _HybridOriginalPublication,
    accepted: phx.meshing.CellMeshingResult,
    cycle: int,
) -> _HybridAdaptationCycle:
    """Refine every physical column once and apply inverse coarsening."""

    source, specification = publication.source, publication.specification
    columns = _hybrid_qualification_columns(accepted)
    policy = prepare_hybrid_adaptation_policy(
        source, specification, accepted, limits=specification.limits
    )
    first = int(np.min(np.asarray(columns.cell_ids)))
    refinement = StageLedger(
        record,
        deadline=Deadline(record["started"] + record["targets"]["wall_seconds"]),
    ).run(
        f"original_physical_layer_refinement:{cycle}",
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                accepted,
                phx.meshing.MarkedMeshAdaptation(
                    np.asarray((first,), dtype=np.int64), layer_columns=columns
                ),
                policy=policy,
                observe_managed_admission=True,
            ),
            observe_managed_admission=True,
        ),
    )
    if refinement.status is not phx.meshing.MeshAdaptationStatus.COMPLETE:
        raise qualification_failure(
            "_hybrid_qualification_concave_lifecycle",
            ("Physical-column refinement omitted requested columns."),
            {"refinement": refinement},
        )
    hierarchy = _hybrid_mixed_hierarchy(refinement)
    active = set(
        np.concatenate(
            [np.asarray(block.global_ids) for block in refinement.target.mesh.blocks]
        ).tolist()
    )
    children = sorted(set(hierarchy.siblings.all_child_ids().tolist()) & active)
    coarsening = StageLedger(
        record,
        deadline=Deadline(record["started"] + record["targets"]["wall_seconds"]),
    ).run(
        f"original_physical_layer_inverse_coarsening:{cycle}",
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                refinement.target,
                phx.meshing.MarkedMeshAdaptation(
                    (),
                    np.asarray(children, dtype=np.int64),
                    hierarchy=refinement.hierarchy,
                ),
                policy=policy,
                observe_managed_admission=True,
            ),
            observe_managed_admission=True,
        ),
    )
    if coarsening.status is not phx.meshing.MeshAdaptationStatus.COMPLETE:
        raise qualification_failure(
            "_hybrid_qualification_concave_lifecycle",
            ("Physical-column inverse coarsening omitted requested columns."),
            {"coarsening": coarsening},
        )
    return _HybridAdaptationCycle(refinement, coarsening)


def _hybrid_concave_archive_continuation(
    args: QualificationControls,
    record: dict[str, Any],
    publication: _HybridOriginalPublication,
    accepted: phx.meshing.CellMeshingResult,
    adaptation: _HybridAdaptationCycle,
    cycle: int,
    carried: _HybridWarmLineage | None,
) -> tuple[_HybridContinuationEvidence, _HybridWarmLineage]:
    """Checkpoint the cycle's warm accepted states and cold-continue them."""

    source, specification, options = (
        publication.source,
        publication.specification,
        publication.options,
    )
    _ledger = StageLedger(record)
    epochs, warm = _hybrid_warm_cycle(
        args,
        record,
        accepted,
        adaptation,
        cycle,
        part_name="original-concave-layer-core",
        limits=specification.limits,
        carried=carried,
    )
    with TemporaryDirectory(prefix="native-original-concave-lineage-") as temporary:
        receipt = StageLedger(
            record,
            deadline=Deadline(record["started"] + record["targets"]["wall_seconds"]),
        ).run(
            f"original_default_durable_lineage:{cycle}",
            lambda: write_meshing_source_closure(
                Path(temporary) / "lifecycle",
                layer_core_lineage_records(source, specification, options, epochs),
            ),
        )
        consumer = QualificationControls.from_record(
            {
                **args.to_record(),
                "hybrid_archive": str(receipt.path),
                "hybrid_content_id": receipt.content_id,
            }
        )
        continuation = {
            "cycle": cycle,
            "started": record["started"],
            "targets": record["targets"],
            "stages_seconds": {},
            "active_phase": "original_default_archive_renewal",
        }
        record["cycles"].append(continuation)
        try:
            continued = _hybrid_qualification_archived_profile(
                consumer, continuation, expected_rounds=1
            )
        except Exception:
            _ledger.inherit_failure(continuation)
            raise
        finally:
            del continuation["started"]
            del continuation["active_phase"]
    return continued, warm


def _hybrid_qualification_concave_lifecycle(
    args: QualificationControls, record: dict[str, Any]
) -> _HybridContinuationEvidence:
    """Fill the closed inward cap and consume lineage for every entity."""

    if (
        args.hybrid_core_size is None
        or not np.isfinite(args.hybrid_core_size)
        or args.hybrid_core_size <= 0.0
    ):
        raise ValueError(
            f"_hybrid_qualification_concave_lifecycle.hybrid_core_size must be finite and positive; got {args.hybrid_core_size!r}."
        )
    if (
        args.adaptation_rounds < 1
        or not np.isfinite(args.target_error)
        or args.target_error <= 0.0
    ):
        raise ValueError(
            f"_hybrid_qualification_concave_lifecycle.bounds must have positive adaptation rounds and finite positive target error; got {(args.adaptation_rounds, args.target_error)!r}."
        )
    publication = _hybrid_concave_publication(args, record)
    coverage = publication.coverage
    np.testing.assert_allclose(
        np.asarray(coverage.achieved_region_measures, dtype=np.float64),
        (1.0,),
        rtol=1.0e-12,
    )
    accepted = _hybrid_concave_high_order_map(args, record, publication)
    record["cycles"] = []
    errors: list[float] = []
    carried: _HybridWarmLineage | None = None
    for cycle in range(args.adaptation_rounds):
        adaptation = _hybrid_concave_adaptation_cycle(
            args, record, publication, accepted, cycle
        )
        continued, carried = _hybrid_concave_archive_continuation(
            args, record, publication, accepted, adaptation, cycle, carried
        )
        _hybrid_record_cold_epochs(record, continued, cycle)
        errors.append(continued.physical_error)
        accepted = adaptation.coarsening.target
    cycles = record["cycles"]
    if not cycles:
        raise qualification_failure(
            "_hybrid_qualification_concave_lifecycle",
            ("Concave lifecycle produced no accepted adaptation cycle."),
            {"cycles": cycles},
        )
    record["physical_error"] = max(errors)
    if record["physical_error"] > args.target_error:
        raise qualification_failure(
            "_hybrid_qualification_concave_lifecycle",
            ("Concave lifecycle missed its frozen physical-error target."),
            {"args": args, "record": record},
        )
    record["independent_scientific_certification"] = "passed"
    record["status"] = "passed"

    return _HybridContinuationEvidence(
        record["physical_error"], tuple(record["epochs"]), record["generation"]
    )


class _HybridOriginalCurvedRefusalOwners(TypedDict):
    evidence: dict[str, str | int | bool | list[str]]


def _hybrid_qualification_original_curved_refusal_profile() -> (
    _HybridOriginalCurvedRefusalOwners
):
    """Bind the immutable source to its nonperiodic trace proof."""

    residuals = tuple(
        Fraction(value) for value in ORIGINAL_CURVED_PERIODIC_TRACE_RESIDUALS
    )
    if any(value == 0 for value in residuals):
        raise qualification_failure(
            "_hybrid_qualification_original_curved_refusal_profile",
            ("The negative trace proof no longer contains its mismatch."),
            {"residuals": residuals},
        )
    evidence: dict[str, str | int | bool | list[str]] = {
        "profile": "original-curved-periodic-refusal",
        "stage_scope": "exact-original-source-periodic-trace-refusal",
        "source_id": ORIGINAL_CURVED_PERIODIC_SOURCE_ID,
        "source_revision": ORIGINAL_CURVED_PERIODIC_SOURCE_REVISION,
        "source_digest": ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST,
        "required_period_x": 1,
        "original_top_trace_x_residuals": list(ORIGINAL_CURVED_PERIODIC_TRACE_RESIDUALS),
        "proof": "exact-rational-source-expression-trace",
        "proof_evidence": "tests/data/meshing/curved_periodic_sources.json",
        "source_bits_preserved": True,
        "rejection": (
            "Exact periodic certification refused the original source for residuals "
            + ", ".join(ORIGINAL_CURVED_PERIODIC_TRACE_RESIDUALS)
        ),
    }
    return {"evidence": evidence}


@final
class _HybridCorrectedPublication(phx.StrictModule):
    authored: phx.meshing.qualification.AuthoredPeriodicLayerSource
    plan: phx.meshing.NativeMeshingPlan
    generated: phx.meshing.CellMeshingResult
    report: phx.meshing.MeshCertificationReport
    mapped_domain: phx.geometry.MappedReferenceDomain


def _hybrid_corrected_source(
    args: QualificationControls, record: dict[str, Any]
) -> phx.meshing.qualification.AuthoredPeriodicLayerSource:
    """Author the periodic-orbit source and retain the immutable refusal identity."""

    limits, _ = _hybrid_qualification_limits(args)
    authored = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "corrected_fundamental_domain_source_authoring",
        lambda: prepare_corrected_periodic_layer_source(limits=limits),
    )
    source = authored.source
    identity = periodic_layer_source_identity_record(authored)
    record["preserved_original_source"] = {
        "source_id": ORIGINAL_CURVED_PERIODIC_SOURCE_ID,
        "source_revision": ORIGINAL_CURVED_PERIODIC_SOURCE_REVISION,
        "source_digest": ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST,
        "exact_residuals": list(ORIGINAL_CURVED_PERIODIC_TRACE_RESIDUALS),
    }
    if (
        source.source_id == ORIGINAL_CURVED_PERIODIC_SOURCE_ID
        or source.source_revision == ORIGINAL_CURVED_PERIODIC_SOURCE_REVISION
        or authored.source_digest == ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST
    ):
        raise qualification_failure(
            "_hybrid_qualification_corrected_curved_lifecycle",
            ("The corrected source reused the immutable source identity."),
            {
                "ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST": ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST,
                "ORIGINAL_CURVED_PERIODIC_SOURCE_ID": ORIGINAL_CURVED_PERIODIC_SOURCE_ID,
                "ORIGINAL_CURVED_PERIODIC_SOURCE_REVISION": ORIGINAL_CURVED_PERIODIC_SOURCE_REVISION,
                "authored": authored,
                "source": source,
            },
        )
    record["source_profile"] = {
        **identity,
        "profile": "curved-periodic-layer-core",
        "stage_scope": (
            "corrected-curved-ADVANCING-layer-core-highorder-AMR-"
            "H1Q2-DG1-Hcurl2-Hdiv2-FV-history-PDE-defaultcold"
        ),
        "source_composition": "fundamental-domain-representatives-plus-exact-unit-translation",
        "original_source_refusal_retained": {
            "source_id": ORIGINAL_CURVED_PERIODIC_SOURCE_ID,
            "exact_residuals": list(ORIGINAL_CURVED_PERIODIC_TRACE_RESIDUALS),
        },
    }
    return authored


def _hybrid_corrected_publication(
    authored: phx.meshing.qualification.AuthoredPeriodicLayerSource,
    record: dict[str, Any],
) -> _HybridCorrectedPublication:
    """Publish the requested hybrid families and verify all required source and periodic certificates."""

    source, specification = authored.source, authored.specification
    provider = phx.meshing.NativeMeshingProvider(authored.options)
    plan = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "corrected_layer_core_plan",
        lambda: provider.plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        ),
    )
    generated = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run("corrected_layer_core_generation", plan.execute)
    report = generated.certification
    if report is None or not report.passed or report.embedding is None:
        raise qualification_failure(
            "_hybrid_qualification_corrected_curved_lifecycle",
            ("The corrected source failed a required native certification check."),
            {"report": report},
        )
    mismatches = [
        finding
        for finding in report.embedding.findings
        if finding.check == "periodic_mapped_trace_mismatch"
    ]
    if mismatches:
        raise qualification_failure(
            "_hybrid_qualification_corrected_curved_lifecycle",
            ("The corrected source retained a nonperiodic source trace."),
            {"mismatches": mismatches},
        )
    if generated.mesh.periodic_topology is None:
        raise qualification_failure(
            "_hybrid_qualification_corrected_curved_lifecycle",
            ("The corrected publication lost its explicit periodic quotient."),
            {"generated": generated},
        )
    if source.mapped_domain is None:
        raise qualification_failure(
            "_hybrid_qualification_corrected_curved_lifecycle",
            ("The corrected publication lost its authored high-order source roots."),
            {"source": source},
        )
    kinds = {block.cell_kind for block in generated.mesh.blocks}
    if kinds != {"prism", "tetrahedron"}:
        raise qualification_failure(
            "_hybrid_qualification_corrected_curved_lifecycle",
            ("The corrected source did not publish the requested hybrid families."),
            {"kinds": kinds},
        )
    record["generation"] = _result_identity(generated)
    record["immutable_cap"] = _hybrid_qualification_cap_identity(generated, source)
    record["curving"] = {
        "geometry_id": generated.audit.geometry_id,
        "mapped_domain_id": source.mapped_domain.domain_id,
        "degree": 2,
        "status": "certified",
        "periodic_trace_status": "certified",
        "periodic_topology_id": generated.mesh.periodic_topology.periodic_topology_id,
    }
    record["request"] = {
        "specification_id": specification.specification_id,
        "limits_id": specification.limits.limits_id,
        "options_id": authored.options.options_id,
        "prepared_id": plan.plan_id,
        "source_authoritative_schedule": list(source.layers.control.schedule.thicknesses),
        "resource_controls_unchanged": True,
    }
    return _HybridCorrectedPublication(
        authored, plan, generated, report, source.mapped_domain
    )


def _hybrid_corrected_adaptation(
    publication: _HybridCorrectedPublication,
    accepted: phx.meshing.CellMeshingResult,
    record: dict[str, Any],
    cycle: int,
) -> _HybridAdaptationCycle:
    """Accept all required mixed refinement and inverse coarsening without changing source coefficients."""

    source, specification, generated = (
        publication.authored.source,
        publication.authored.specification,
        publication.generated,
    )
    columns = _hybrid_qualification_columns(accepted)
    # Later cycles adapt the restored carrier; the source association transfer
    # remains bound to the source's accepted scientific generation request.
    policy = prepare_hybrid_adaptation_policy(
        source, specification, generated, limits=specification.limits
    )
    core_ids = np.sort(
        np.concatenate(
            [
                np.asarray(block.global_ids, dtype=np.int64)
                for block in accepted.mesh.blocks
                if block.cell_kind == "tetrahedron"
            ]
        )
    )
    if not core_ids.size:
        raise qualification_failure(
            "_hybrid_qualification_corrected_curved_lifecycle",
            ("The corrected hybrid source has no native simplex core."),
            {"core_ids": core_ids},
        )
    refinement = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"corrected_mixed_refinement:{cycle}",
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                accepted,
                phx.meshing.MarkedMeshAdaptation(core_ids[:1], layer_columns=columns),
                policy=policy,
                observe_managed_admission=True,
            ),
            observe_managed_admission=True,
        ),
    )
    if (
        refinement.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
        or refinement.hierarchy is None
    ):
        raise qualification_failure(
            "_hybrid_qualification_corrected_curved_lifecycle",
            ("The corrected curved mixed refinement omitted requested cells."),
            {"refinement": refinement},
        )
    hierarchy = _hybrid_mixed_hierarchy(refinement)
    active = set(
        np.concatenate(
            [np.asarray(block.global_ids) for block in refinement.target.mesh.blocks]
        ).tolist()
    )
    children = np.asarray(
        sorted(set(hierarchy.siblings.all_child_ids().tolist()) & active),
        dtype=np.int64,
    )
    coarsening = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"corrected_mixed_inverse_coarsening:{cycle}",
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                refinement.target,
                phx.meshing.MarkedMeshAdaptation(
                    (),
                    children,
                    hierarchy=refinement.hierarchy,
                    layer_columns=_hybrid_qualification_columns(refinement.target),
                ),
                policy=policy,
                observe_managed_admission=True,
            ),
            observe_managed_admission=True,
        ),
    )
    if coarsening.status is not phx.meshing.MeshAdaptationStatus.COMPLETE:
        raise qualification_failure(
            "_hybrid_qualification_corrected_curved_lifecycle",
            ("The corrected curved inverse coarsening omitted requested cells."),
            {"coarsening": coarsening},
        )
    if coarsening.target.mesh.topology_id != generated.mesh.topology_id:
        raise qualification_failure(
            "_hybrid_qualification_corrected_curved_lifecycle",
            ("The corrected lifecycle did not restore its source topology."),
            {"coarsening": coarsening, "generated": generated},
        )
    certification = generated.certification
    if certification is None:
        raise qualification_failure(
            "hybrid.source_certificate",
            "retain the source coordinate request limits",
            {"certification": certification},
        )
    if not hybrid_source_coefficients_restored(
        generated, coarsening.target, limits=certification.request.limits
    ):
        raise qualification_failure(
            "_hybrid_qualification_corrected_curved_lifecycle",
            "The corrected lifecycle changed its authored source coefficients.",
            {"coarsening": coarsening, "generated": generated},
        )
    return _HybridAdaptationCycle(refinement, coarsening)


def _hybrid_corrected_archive_continuation(
    args: QualificationControls,
    record: dict[str, Any],
    publication: _HybridCorrectedPublication,
    accepted: phx.meshing.CellMeshingResult,
    adaptation: _HybridAdaptationCycle,
    cycle: int,
    carried: _HybridWarmLineage | None,
) -> tuple[_HybridContinuationEvidence, _HybridWarmLineage]:
    """Checkpoint the cycle's warm accepted states and cold-continue them.

    The first cycle archives the live generation under its retained producer
    validation; later cycles archive their carried accepted state, which the
    archive owner validates and the consumer authenticates in full.
    """

    authored, plan, generated = (
        publication.authored,
        publication.plan,
        publication.generated,
    )
    source, specification = authored.source, authored.specification
    refinement, coarsening = adaptation.refinement, adaptation.coarsening
    if accepted.certification is None:
        raise qualification_failure(
            "hybrid.accepted_certification",
            "retain the accepted source theorem",
            {"certification": accepted.certification},
        )
    retained = accepted is generated
    _ledger = StageLedger(record)
    epochs, warm = _hybrid_warm_cycle(
        args,
        record,
        accepted,
        adaptation,
        cycle,
        part_name="corrected-curved-periodic-layer-core",
        limits=specification.limits,
        carried=carried,
    )
    with TemporaryDirectory(prefix="native-corrected-curved-lineage-") as temporary:
        receipt = StageLedger(
            record,
            deadline=Deadline(record["started"] + record["targets"]["wall_seconds"]),
        ).run(
            f"corrected_durable_source_archive:{cycle}",
            lambda: write_meshing_source_closure(
                Path(temporary) / "lifecycle",
                layer_core_lineage_records(
                    source, specification, authored.options, epochs
                ),
                retained_layer_core_plan=plan if retained else None,
            ),
        )
        retained_controls = None
        if retained:
            if receipt.validation_id is None or generated.execution_evidence is None:
                raise qualification_failure(
                    "_hybrid_qualification_corrected_curved_lifecycle",
                    ("Corrected lifecycle archive lacks retained producer validation."),
                    {"generated": generated, "receipt": receipt},
                )
            retained_controls = MeshingSourceExecutionControls(
                generated.execution_evidence,
                specification.limits,
                source.source_id,
                source.source_revision,
                specification.specification_id,
                source.binding_id,
                receipt.content_id,
            )
        consumer = QualificationControls.from_record(
            {
                **args.to_record(),
                "hybrid_archive": str(receipt.path),
                "hybrid_content_id": receipt.content_id,
            }
        )
        continuation: dict[str, Any] = {
            "cycle": cycle,
            "started": record["started"],
            "targets": record["targets"],
            "stages_seconds": {},
            "active_phase": f"corrected_cold_continuation:{cycle}",
        }
        record["cycles"].append(continuation)
        try:
            continued = _hybrid_qualification_archived_profile(
                consumer,
                continuation,
                retained_controls=retained_controls,
                expected_rounds=1,
            )
        except Exception:
            _ledger.inherit_failure(continuation)
            raise
        finally:
            del continuation["started"]
            del continuation["active_phase"]
    continuation["archive"] = {
        "content_id": receipt.content_id,
        "source_revision": source.source_revision,
        "source_binding_id": source.binding_id,
        "retained_producer_validation": retained,
        "restored_cold_continuation": True,
    }
    continuation["source"].update(
        {
            "source_digest": authored.source_digest,
            "mapped_domain_id": publication.mapped_domain.domain_id,
            "wall_orbit_ids": list(authored.wall_orbit_ids),
            "coefficient_orbit_ids": list(authored.coefficient_orbit_ids),
        }
    )
    continuation["adaptation"] = {
        "accepted_id": accepted.result_id,
        "refinement_id": refinement.result_id,
        "coarsening_id": coarsening.result_id,
        "restored_topology": True,
        "restored_source_coefficients": True,
    }
    return continued, warm


def _hybrid_qualification_corrected_curved_lifecycle(
    args: QualificationControls,
    record: dict[str, Any],
) -> _HybridContinuationEvidence:
    """Generate, then adapt, archive, and cold-continue every declared cycle."""
    if args.geometry_order != 2:
        raise ValueError(
            f"_hybrid_qualification_corrected_curved_lifecycle.geometry_order must be two for the authored quadratic source; got {args.geometry_order!r}."
        )
    if args.adaptation_rounds < 1:
        raise ValueError(
            f"_hybrid_qualification_corrected_curved_lifecycle.adaptation_rounds must be at least one; got {args.adaptation_rounds!r}."
        )
    authored = _hybrid_corrected_source(args, record)
    publication = _hybrid_corrected_publication(authored, record)
    record["cycles"] = []
    errors: list[float] = []
    accepted = publication.generated
    carried: _HybridWarmLineage | None = None
    for cycle in range(args.adaptation_rounds):
        adaptation = _hybrid_corrected_adaptation(publication, accepted, record, cycle)
        continued, carried = _hybrid_corrected_archive_continuation(
            args, record, publication, accepted, adaptation, cycle, carried
        )
        _hybrid_record_cold_epochs(record, continued, cycle)
        errors.append(continued.physical_error)
        accepted = adaptation.coarsening.target
    record["physical_error"] = max(errors)
    record["physical_error_scope"] = (
        "maximum-nodal-PDE-defect-across-every-cycle-history-transition"
    )
    record["independent_scientific_certification"] = "passed"
    record["status"] = "passed"
    record["implemented_admitted_workflow"] = (
        "corrected-curved-periodic-ADVANCING-layer-core-highorder-"
        "mixed-allfield-history-PDE-cold-continuation"
    )
    return _HybridContinuationEvidence(
        record["physical_error"], tuple(record["epochs"]), record["generation"]
    )


def _hybrid_qualification_original_profile(
    args: QualificationControls,
) -> dict[str, object]:
    """Run source profiles without borrowing the unit-box acceptance law."""

    started = perf_counter()
    record: dict[str, Any] = {
        "scenario": "native-hybrid-lifecycle",
        "profile": args.hybrid_profile,
        "engine": "native",
        "runtime": _runtime_identity(),
        "status": "failed",
        "started": started,
        "active_phase": "original_source_preparation",
        "stages_seconds": {},
        "epochs": [],
        "mandatory_workflow_completion": "incomplete",
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
        "preserved_original_source": {
            "source_id": "curved-periodic-narrow-gap",
            "required_period_x": 1,
            "exact_residuals": [
                "-101121/2361183241434822606848",
                "-6620711477/77371252455336267181195264",
            ],
            "disposition": "immutable-negative-refusal",
        },
        "targets": {"wall_seconds": args.timeout},
        "contract": (
            contract := _hybrid_qualification_profile_contracts()[args.hybrid_profile]
        ),
    }
    source_evidence: dict[str, Any] | None = None
    continued: _HybridContinuationEvidence | None = None
    try:
        if not np.isfinite(args.timeout) or args.timeout <= 0.0:
            raise ValueError(
                f"_hybrid_qualification_original_profile.timeout must be finite and positive; got {args.timeout!r}."
            )
        if args.hybrid_profile == "archived-layer-core" and (
            not args.hybrid_archive or not args.hybrid_content_id
        ):
            record["status"] = "blocked"
            record["missing_prerequisite"] = (
                "Exact authored archive path and content identity."
            )
            return record
        if args.hybrid_profile == "archived-layer-core":
            continued = _hybrid_qualification_archived_profile(args, record)
            record["generation"] = continued.source_result
            _hybrid_record_cold_epochs(record, continued, 0)
        elif args.hybrid_profile == "concave-box-core-lifecycle":
            continued = _hybrid_qualification_concave_lifecycle(args, record)
        elif args.hybrid_profile == "curved-periodic-layer-core":
            continued = _hybrid_qualification_corrected_curved_lifecycle(args, record)
        elif args.hybrid_profile == "original-curved-periodic-refusal":
            owners = StageLedger(
                record,
                deadline=Deadline(record["started"] + record["targets"]["wall_seconds"]),
            ).run(
                "original_curved_periodic_exact_refusal",
                _hybrid_qualification_original_curved_refusal_profile,
            )
            source_evidence = owners["evidence"]
            record["source_profile"] = source_evidence
            record["status"] = "passed"
        elif args.hybrid_profile in _hybrid_qualification_original_wall_profiles():
            owners = StageLedger(
                record,
                deadline=Deadline(record["started"] + record["targets"]["wall_seconds"]),
            ).run(
                "original_wall_layer_oracles",
                lambda: _hybrid_qualification_original_wall_profile(args.hybrid_profile),
            )
            source_evidence = owners["evidence"]
            record["source_profile"] = source_evidence
            record["status"] = "passed"
        else:
            with TemporaryDirectory(prefix="native-original-layer-") as temporary:
                owners = StageLedger(
                    record,
                    deadline=Deadline(
                        record["started"] + record["targets"]["wall_seconds"]
                    ),
                ).run(
                    "original_sweep_source_oracles",
                    lambda: _hybrid_qualification_original_layer_sweep_profile(
                        args.hybrid_profile,
                        destination=Path(temporary) / "original-source.brep",
                    ),
                )
                source_evidence = owners["evidence"]
                record["source_profile"] = source_evidence
                record["status"] = "passed"
    except (
        phx.meshing.MeshingFailure,
        RuntimeError,
        ValueError,
        TypeError,
        AssertionError,
        TimeoutError,
        subprocess.TimeoutExpired,
    ) as failure:
        record_failure(record, failure, record["active_phase"])
    finally:
        record["end_to_end_seconds"] = perf_counter() - started
        record["peak_process_resident_bytes"] = peak_resident_bytes()
        record["process_peak_scope"] = PEAK_RESIDENT_SCOPE
        expected_admission = contract["expected_admission"]
        if expected_admission not in ("positive", "refusal"):
            raise ValueError(
                f"_hybrid_qualification_original_profile.expected_admission must be 'positive' or 'refusal'; got {expected_admission!r}."
            )
        record["expected_admission"] = expected_admission
        if (
            expected_admission == "positive"
            and contract["full_workflow"] is not True
            and record["status"] == "passed"
        ):
            record["case_contract_completion"] = "complete"
            record["independent_scientific_certification"] = "passed"
        if expected_admission == "refusal" and record["status"] == "passed":
            if source_evidence is None:
                raise qualification_failure(
                    "_hybrid_qualification_original_profile.source",
                    "include refusal source evidence",
                    {"source_evidence": source_evidence},
                )
            evidence = source_evidence
            receipts = evidence.get("receipts")
            if not (isinstance(receipts, list) and receipts or "rejection" in evidence):
                raise qualification_failure(
                    "_hybrid_qualification_original_profile",
                    ("Hybrid refusal profile omitted its refusal evidence."),
                    {"evidence": evidence, "receipts": receipts},
                )
            record.update(
                {
                    "status": "expected-refusal",
                    "failure_evidence_id": canonical_fingerprint(evidence),
                    "requested": {"expected_admission": "refusal"},
                    "achieved": {"source_refusal": True},
                    "source_bits_preserved": True,
                }
            )
        if (
            contract["full_workflow"] is True
            and expected_admission == "positive"
            and record["status"] == "passed"
        ):
            physical_error = None if continued is None else continued.physical_error
            if (
                record.get("independent_scientific_certification") != "passed"
                or physical_error is None
                or not np.isfinite(physical_error)
                or physical_error > args.target_error
                or not record["epochs"]
                or not isinstance(record.get("generation"), dict)
            ):
                raise qualification_failure(
                    "_hybrid_qualification_original_profile",
                    (
                        "Full hybrid workflow lacks passing independent physical, "
                        "accepted-epoch or generation publication evidence."
                    ),
                    {"args": args, "physical_error": physical_error, "record": record},
                )
            record["mandatory_workflow_completion"] = "complete"
        if record["end_to_end_seconds"] > args.timeout:
            record["status"] = "timeout"
            record["mandatory_workflow_completion"] = "incomplete"
            if "case_contract_completion" in record:
                record["case_contract_completion"] = "incomplete"
        del record["started"]
        del record["active_phase"]
    return record


def _hybrid_qualification_limits(args: QualificationControls) -> tuple[Any, Any]:
    limits = capacity_meshing_limits(
        args.capacity,
        args.timeout,
        edges=16,
        faces=16,
        cells=8,
        connectivity_entries=128,
        data_bytes=4096,
        work_units=128,
        geometry_queries=128,
        scratch_bytes=8192,
    )
    certificates = phx.geometry.MeshCertificateLimits(
        maximum_candidate_pairs=128 * args.capacity,
        maximum_ray_tests=512 * args.capacity,
        maximum_source_samples=32 * args.capacity,
        maximum_distance_evaluations=1024 * args.capacity,
        maximum_subdivision_pieces=128 * args.capacity,
        maximum_bernstein_nodes=32 * args.capacity,
    )
    return limits, certificates


def _hybrid_qualification_geometry(
    args: QualificationControls,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    limits: Any,
) -> tuple[Any, Any, Any, Any, dict[tuple[int, int], int], Any]:
    """Generate every native CAD boundary face, then select the bottom wall."""

    model = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "native_cad_preparation",
        lambda: phx.geometry.brep_box(
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            coordinate_contract=phx.SpatialCoordinateContract.si(),
            source_id="hybrid-unit-box",
        ),
    )
    domain = phx.geometry.MeshingDomain.from_brep(model)
    geometry = model.geometry
    if geometry is None:
        raise qualification_failure(
            "_hybrid_qualification_geometry",
            ("Hybrid wall source requires native B-Rep source model."),
            {"geometry": geometry},
        )
    vertices = np.asarray(geometry.vertex_points, dtype=np.float64)
    faces: dict[tuple[int, int], int] = {}
    for local, definition in enumerate(domain.source_indices[2]):
        loops = geometry.face_loops[definition]
        members = np.unique(
            [
                vertex
                for loop in loops
                for coedge in loop
                for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
            ]
        )
        for axis in range(3):
            for side in (0, 1):
                if np.all(vertices[members, axis] == side):
                    faces[axis, side] = int(domain.scope_indices(2)[local])
    if set(faces) != {(axis, side) for axis in range(3) for side in (0, 1)}:
        raise qualification_failure(
            "_hybrid_qualification_geometry",
            ("Native unit-box source lost a boundary face."),
            {"faces": faces},
        )
    scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64),
    )
    request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                1.0 / args.resolution,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        limits=limits,
    )
    provider = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("parametric_surface")
    )
    plan = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "wall_source_plan",
        lambda: provider.plan(
            phx.meshing.NativeSurfaceSource(domain),
            request,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
            record_phase=phases.append,
        ),
    )
    boundary = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "native_wall_source_generation", lambda: plan.execute(record_phase=phases.append)
    )
    if boundary.certification is None or boundary.certification.request.source is None:
        raise qualification_failure(
            "_hybrid_qualification_geometry",
            ("Generated native wall lacks the declared CAD source query."),
            {"boundary": boundary},
        )
    query = boundary.certification.request.source
    points = np.asarray(boundary.mesh.coordinates, dtype=np.float64)
    triangles = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in boundary.mesh.blocks]
    )
    triangles = triangles[np.all(np.abs(points[triangles, 2]) <= 1.0e-12, axis=1)]
    rows = np.unique(triangles)
    if rows.size == 0:
        raise qualification_failure(
            "_hybrid_qualification_geometry",
            ("Native boundary generation produced no bottom-wall triangles."),
            {"rows": rows},
        )
    wall_points = points[rows]
    triangles = np.searchsorted(rows, triangles)
    normals = np.cross(
        wall_points[triangles[:, 1]] - wall_points[triangles[:, 0]],
        wall_points[triangles[:, 2]] - wall_points[triangles[:, 0]],
    )
    triangles[normals[:, 2] < 0.0] = triangles[normals[:, 2] < 0.0][:, ::-1]
    wall = phx.discretization.CellMesh(
        wall_points, (phx.discretization.CellBlock("native-wall", "triangle", triangles),)
    )
    record["wall_source"] = {
        "result": _result_identity(boundary),
        "request_id": request.specification_id,
        "source_id": query.source_id,
        "source_revision": query.source_revision,
        "wall_mesh_id": wall.mesh_id,
        "wall_vertices": rows.size,
        "wall_triangles": triangles.shape[0],
        "authority_id": domain.authority_id,
    }
    return model, domain, query, scope, faces, wall


def _hybrid_qualification_core(
    layers: Any,
    domain: Any,
    query: Any,
    scope: Any,
    faces: dict[tuple[int, int], int],
) -> tuple[phx.meshing.NativeLayerCoreSource, phx.discretization.CellMesh]:
    """Build all core facets and cap identities, never a cap-only PLC."""

    if layers.cap is None:
        raise qualification_failure(
            "_hybrid_qualification_core",
            ("The generated unit-box wall must have an open immutable cap."),
            {"layers": layers},
        )
    points = np.asarray(layers.cap.coordinates, dtype=np.float64)
    cap = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in layers.cap.blocks]
    )
    edges = np.concatenate((cap[:, [0, 1]], cap[:, [1, 2]], cap[:, [2, 0]]))
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    rim = edges[counts[inverse] == 1]
    upper = points.copy()
    upper[:, 2] = 1.0
    count = points.shape[0]
    sides: list[tuple[int, int, int]] = []
    side_facets: list[int] = []
    side_keys = ((0, 0), (0, 1), (1, 0), (1, 1))
    for a, b in rim:
        selected = [
            facet
            for facet, (axis, side) in enumerate(side_keys, start=2)
            if np.all(np.abs(points[[a, b], axis] - side) <= 1.0e-12)
        ]
        if len(selected) != 1:
            raise qualification_failure(
                "_hybrid_qualification_core",
                ("Every cap rim edge must belong to one side face."),
                {"selected": selected},
            )
        sides.extend(
            ((int(a), int(b), int(b + count)), (int(a), int(b + count), int(a + count)))
        )
        side_facets.extend((selected[0], selected[0]))
    complex_ = phx.meshing.PiecewiseLinearComplex(
        np.concatenate((points, upper)),
        np.concatenate((cap, cap + count, np.asarray(sides, dtype=np.int64))),
        np.concatenate(
            (
                np.zeros(cap.shape[0], dtype=np.int64),
                np.ones(cap.shape[0], dtype=np.int64),
                np.asarray(side_facets, dtype=np.int64),
            )
        ),
        np.asarray(((0, -1), *((-1, 0),) * 5), dtype=np.int64),
        ("fluid",),
        boundary="fixed",
    )
    source = phx.meshing.NativeLayerCoreSource(
        layers,
        complex_,
        domain.source_id,
        domain.source_revision,
        vertex_layer_ids=np.concatenate(
            (
                np.asarray(layers.cap_vertices, dtype=np.int64),
                np.full(count, -1, dtype=np.int64),
            )
        ),
        cap_polygon_ids=np.arange(cap.shape[0], dtype=np.int64),
        layer_regions=np.zeros(
            sum(block.cell_count for block in layers.mesh.blocks), dtype=np.int64
        ),
        source_boundary_scope=scope,
        source_domain=domain,
        fidelity_source=query,
        core_facet_source_ids=np.asarray(
            [-1, faces[2, 1], *(faces[key] for key in side_keys)], dtype=np.int64
        ),
    )
    return source, layers.cap


def _hybrid_qualification_generation(
    args: QualificationControls,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    limits: Any,
) -> tuple[Any, Any, Any, Any, Any]:

    model, domain, query, scope, faces, wall = _hybrid_qualification_geometry(
        args, record, phases, limits
    )
    projection = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "native_projection_preparation",
        lambda: phx.geometry.prepare_brep_projection(model),
    )
    association_policy = phx.meshing.AssociationPropagationPolicy(
        classification_tolerance=1.0e-8
    )
    wall_vertices = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "wall_cad_association",
        lambda: phx.meshing.associate_mesh_vertices(
            wall, projection, policy=association_policy
        ),
    )
    wall_association = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "wall_cad_face_association",
        lambda: phx.meshing.associate_mesh_entities(
            wall, wall_vertices, projection, 2, policy=association_policy
        ),
    )
    wall_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray([faces[2, 0]], dtype=np.int64),
    )
    schedule = phx.meshing.LayerSchedule.geometric(
        args.layer_count, args.first_thickness, growth_rate=args.layer_growth
    )
    control = phx.meshing.BoundaryLayerControl(
        wall_scope,
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        volume_scope=phx.meshing.MeshingScope(
            domain.source_id,
            domain.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            3,
            domain.entity_set_id(3),
            np.asarray(domain.scope_indices(3), dtype=np.int64),
        ),
        collision=phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
        growth_rate_bounds=(
            args.layer_growth * (1.0 - 1.0e-10),
            args.layer_growth * (1.0 + 1.0e-10),
        ),
    )
    layers = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "native_advancing_layers",
        lambda: phx.meshing.prepare_boundary_layers(
            wall,
            control,
            wall_association=wall_association,
            source_domain=domain,
            policy=phx.meshing.BoundaryLayerPolicy(
                visibility_iterations=128, proximity_samples=16
            ),
        ),
    )
    source, cap_mesh = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "complete_plc_cap_preparation",
        lambda: _hybrid_qualification_core(layers, domain, query, scope, faces),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"), allow_mixed=True
            ),
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                1.0 / args.resolution,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        layer_controls=(control,),
        limits=limits,
    )
    provider = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("layer_core")
    )
    core_facet_ids = source.core_facet_source_ids
    if core_facet_ids is None:
        raise qualification_failure(
            "hybrid.core_facet_source_ids",
            "Retain the core face identities",
            {"core_facet_source_ids": core_facet_ids},
        )
    record["case"] = {
        "case_id": canonical_fingerprint(
            {
                "source": source.binding_id,
                "schedule": schedule.schedule_id,
                "geometry_order": args.geometry_order,
                "limits": limits.limits_id,
            }
        ),
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_binding_id": source.binding_id,
        "cad_authority_id": domain.authority_id,
        "core_complex_id": source.complex.complex_id,
        "cap_id": cap_mesh.mesh_id,
        "material_ids": ["fluid"],
        "independent_domain": "analytic-unit-box",
        "source_geometry": "native-exact-planar-BRep-faces",
        "hard_schedule": list(schedule.thicknesses),
        "layer_control_id": control.control_id,
        "cap_polygon_ids": source.cap_polygon_ids.tolist(),
        "cap_vertex_layer_ids": source.vertex_layer_ids[
            source.vertex_layer_ids >= 0
        ].tolist(),
        "core_facet_original_face_ids": core_facet_ids.tolist(),
        "layer_evidence": _layer_evidence(layers.evidence),
        "layer_validity": _cad_curving_certificate(layers.validity),
    }
    record["request"] = {
        "request_id": specification.specification_id,
        "limits_id": limits.limits_id,
        "native_options_id": provider.options.options_id,
        "required_families": ["prism", "tetrahedron"],
        "immutable_core_boundary": True,
    }
    record["request"]["limits"] = {
        "maximum_vertices": limits.maximum_vertices,
        "maximum_cells": limits.maximum_cells,
        "maximum_edges": limits.maximum_edges,
        "maximum_faces": limits.maximum_faces,
        "maximum_connectivity_entries": limits.maximum_connectivity_entries,
        "maximum_data_bytes": limits.maximum_data_bytes,
        "maximum_work_units": limits.maximum_work_units,
        "maximum_cavity_cells": limits.maximum_cavity_cells,
        "maximum_geometry_queries": limits.maximum_geometry_queries,
        "maximum_scratch_bytes": limits.maximum_scratch_bytes,
        "maximum_wall_seconds": limits.maximum_wall_seconds,
    }
    plan = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "core_geometry_constraint_preparation",
        lambda: provider.plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
            record_phase=phases.append,
        ),
    )
    record["request"]["prepared_id"] = plan.plan_id
    result = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "native_core_generation_publication",
        lambda: plan.execute(record_phase=phases.append),
    )
    record["generation"] = _result_identity(result)
    record["cap_identity"] = _hybrid_qualification_cap_identity(result, source)
    return result, source, projection, query, specification


def _hybrid_qualification_curving(
    result: Any,
    query: Any,
    projection: Any,
    args: QualificationControls,
    limits: Any,
    certificates: Any,
    record: dict[str, Any],
    *,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> Any:
    """Bind high-order curving and every successor to the independent full box.

    Curved publication certifies with the explicit ``certificates`` caps
    unchanged, on tickets of the unit-box request ``limits``.
    """

    vertices = np.asarray(
        phx.discretization.reference_cell_topology("hexahedron").vertices,
        dtype=np.float64,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    triangles = np.asarray(
        [row for a, b, c, d in faces for row in ((a, b, c), (a, c, d))], dtype=np.int64
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        vertices,
        triangles,
        np.tile(np.asarray([0, -1], dtype=np.int64), (12, 1)),
        ("fluid",),
        source_id="independently-declared-hybrid-unit-box",
    )
    regions = np.zeros(
        sum(block.cell_count for block in result.mesh.blocks), dtype=np.int64
    )
    # Curving certifies under the canonical publication audit's validity
    # policy (default 4096 Bernstein nodes, stricter than the curving default
    # 32768, unchanged tolerances) so publication may admit its earned proof.
    validity = phx.discretization.CellValidityPolicy()
    policy = phx.meshing.HighOrderCurvingPolicy(
        degree=args.geometry_order,
        relaxation_rounds=0,
        fidelity_tolerance=1.0e-7,
        maximum_geometry_nodes=32 * args.capacity,
        maximum_tabulation_entries=1024 * args.capacity,
        validity=validity,
    )
    def curve_and_publish() -> Any:
        with (
            NativeExecutionScope(limits.native_limits()) as execution,
            NativeHostStorageWorkspace(execution) as workspace,
        ):
            with execution.batch(0) as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=limits.maximum_work_units,
                    maximum_memory_bytes=limits.maximum_scratch_bytes,
                )
                budget.capture_coordinate_operands = coordinate_operands is not None
                budget.capture_polynomial_operands = (
                    coordinate_operands is not None and coordinate_operands.capture_polynomials
                )
                budget.capture_signed_determinant_operands = (
                    coordinate_operands is not None
                    and coordinate_operands.capture_signed_determinants
                )
            if coordinate_operands is not None:
                coordinate_operands.begin(
                    execution, None,
                    native_preparation_ledger(execution).overlap(execution, None),
                )
            with (
                budget.retained_storage(
                    lambda size: execution.batch(0, bytes_upper=size)
                ),
                meshing_source_owner_reconstruction_scope(budget=budget),
            ):
                with budget.bind_ticket(execution.ticket_view()):
                    association = StageLedger(
                        record,
                        deadline=Deadline(record["started"] + record["targets"]["wall_seconds"]),
                    ).run(
                        "combined_cad_association",
                        lambda: phx.meshing.associate_mesh_vertices(
                            result.mesh,
                            projection,
                            policy=phx.meshing.AssociationPropagationPolicy(
                                classification_tolerance=1.0e-8
                            ),
                            budget=budget,
                        ),
                    )
                    curved = StageLedger(
                        record,
                        deadline=Deadline(record["started"] + record["targets"]["wall_seconds"]),
                    ).run(
                        "high_order_curving_and_independent_certification",
                        lambda: phx.meshing.curve_cell_mesh(
                            result.mesh,
                            association,
                            projection,
                            policy=policy,
                            source=query,
                            domain=domain,
                            cell_regions=regions,
                            certificate_limits=certificates,
                            budget=budget,
                        ),
                    )
                    record["curving"] = {
                        "status": curved.status.value,
                        "policy_id": policy.policy_id,
                        "accepted": curved.evidence.accepted,
                        "certification_failures": list(curved.evidence.certification_failures),
                        "coordinate_map_certificate": _cad_curving_certificate(
                            curved.evidence.certificate
                        ),
                        "certificate_limits_id": certificates.limits_id,
                        "publication_limits_id": limits.limits_id,
                        "maximum_geometry_nodes": policy.maximum_geometry_nodes,
                        "maximum_tabulation_entries": policy.maximum_tabulation_entries,
                        "validity_policy": {
                            "policy_id": validity.policy_id,
                            "maximum_bernstein_nodes": validity.maximum_bernstein_nodes,
                            "origin": "canonical-publication-default",
                            "certificate_policy_id": curved.evidence.certificate.policy_id,
                        },
                        "maximum_source_residual": curved.evidence.maximum_residual,
                        "geometry_order": args.geometry_order,
                        "domain_id": domain.domain_id,
                        "source_query_id": query.source_id,
                        "source_query_revision": query.source_revision,
                        "nonlinear_curved_wall_qualification": "not-claimed",
                    }
                    if (
                        curved.status is not phx.meshing.HighOrderCurvingStatus.CURVED
                        or not curved.evidence.accepted
                    ):
                        raise qualification_failure(
                            "_hybrid_qualification_curving",
                            "Native hybrid high-order curving failed a source certificate check.",
                            {"curved": curved},
                        )
                    inputs = phx.meshing.MeshCertificationInputs(
                        result.mesh,
                        result.geometry,
                        phx.meshing.MeshCertificationSchedule("volume_layer"),
                        domain=domain,
                        cell_regions=regions,
                        source=query,
                        fidelity_tolerance=policy.fidelity_tolerance,
                        limits=certificates,
                        budget=budget,
                    )
                published = StageLedger(
                    record,
                    deadline=Deadline(record["started"] + record["targets"]["wall_seconds"]),
                ).run(
                    "curved_canonical_publication",
                    lambda: qualify_hybrid_curved_publication(
                        result,
                        curved.geometry,
                        association,
                        inputs,
                        limits=limits,
                        audit_policy=phx.meshing.CellMeshAuditPolicy(),
                        curving=curved.evidence,
                        execution=NativePublicationExecution(
                            execution, workspace, None, budget
                        ),
                    ),
                )
                if coordinate_operands is not None:
                    with budget.bind_ticket(execution.ticket_view()):
                        coordinate_operands.capture(
                            published.mesh, published.geometry, budget, execution,
                            coordinate_contract=published.coordinate_contract,
                        )
                publication_owner = (
                    published.result_id if result.execution_evidence is None
                    else result.execution_evidence.owner_id
                )
                with budget.bind_ticket(execution.ticket_view()):
                    issue = stage_native_completed_source_issue(
                        execution, published, owner_id=publication_owner, budget=budget
                    )
        if execution.evidence is None:
            raise RuntimeError("hybrid.curving must retain its complete request evidence")
        bound = bind_native_execution_result(
            published,
            execution.evidence,
            limits,
            execution.operation_started,
            owner_id=publication_owner,
            history_evidence=()
            if result.execution_evidence is None else (result.execution_evidence,),
        )
        issue.complete(bound)
        if coordinate_operands is not None:
            if bound.execution_evidence is None:
                raise RuntimeError("Hybrid operands require their ended publication receipt.")
            coordinate_operands.complete(bound.execution_evidence)
        return bound

    return StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run("curving_and_publication_complete_request", curve_and_publish)


def hybrid_solver_field_templates(
    cell_kind: str, /
) -> tuple[tuple[str, phx.discretization.FiniteElementFieldSpec], ...]:
    """Return the scalar linear diffusion declaration for one hybrid cell family."""
    return (
        (
            "solve:intensive",
            phx.discretization.FiniteElementFieldSpec(
                "u", phx.discretization.lagrange_element(cell_kind, 1)
            ),
        ),
    )


def _hybrid_qualification_solver(
    result: Any, *, budget: CoordinateEnclosureBudget | None = None
) -> tuple[Any, Any, Any]:

    templates = {
        cell_kind: hybrid_solver_field_templates(cell_kind)[0][1]
        for cell_kind in {block.cell_kind for block in result.mesh.blocks}
    }
    template = next(iter(templates.values()))
    field = phx.discretization.FiniteElementFieldSpec(
        template.name,
        {
            block.name: templates[block.cell_kind].elements[0]
            for block in result.mesh.blocks
        },
        component_shape=template.component_shape,
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare(budget=budget)
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    form = phx.equations.FiniteElementForm(
        "qualified-hybrid-diffusion", "u", (phx.equations.DiffusionAction("u", 1.0),)
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, dirichlet_values=exact_solution, budget=budget
    )
    return space, problem, constraint.free_dofs


def _hybrid_qualification_physical(
    result: Any, space: Any, values: Any
) -> dict[str, float]:
    """Integrate mixed mapped fields against the independent analytic patch."""

    squared = 0.0
    integral = 0.0
    volume = 0.0
    for geometry, dofs in zip(
        space.evaluate_geometry("u", result.geometry.coordinates),
        space.dof_maps[0].cell_dofs,
        strict=True,
    ):
        local = np.asarray(values)[np.asarray(dofs)]
        evaluated = local @ np.asarray(geometry.basis_values).T
        weights = np.asarray(geometry.physical_weights, dtype=np.float64)
        exact = np.asarray(exact_solution(geometry.physical_points))
        squared += float(np.sum(weights * (evaluated - exact) ** 2))
        integral += float(np.sum(weights * evaluated))
        volume += float(np.sum(weights))
    return {
        "l2_error": float(np.sqrt(max(0.0, squared))),
        "state_integral": integral,
        "mapped_volume": volume,
        "analytic_volume_defect": volume - 1.0,
        "nodal_maximum_error": float(
            np.max(
                np.abs(
                    np.asarray(values)
                    - np.asarray(exact_solution(space.dof_maps[0].dof_coordinates))
                )
            )
        ),
    }


def _hybrid_qualification_acceptance(result: Any, source: Any) -> dict[str, object]:

    certificate = result.certification
    if certificate is None or not certificate.passed:
        raise qualification_failure(
            "_hybrid_qualification_acceptance",
            ("Hybrid result lacks certification of every declared source obligation."),
            {"certificate": certificate},
        )
    coverage, fidelity, embedding = (
        certificate.coverage,
        certificate.fidelity,
        certificate.embedding,
    )
    if (
        not isinstance(coverage, phx.geometry.DomainCoverageCertificate)
        or fidelity is None
        or embedding is None
    ):
        raise qualification_failure(
            "_hybrid_qualification_acceptance",
            ("Hybrid result lacks independent coverage, fidelity, or global embedding."),
            {"coverage": coverage, "embedding": embedding, "fidelity": fidelity},
        )
    if (
        coverage.status != "certified"
        or fidelity.status != "certified"
        or embedding.status != "certified"
    ):
        raise qualification_failure(
            "_hybrid_qualification_acceptance",
            ("Hybrid source certificates are unresolved."),
            {"coverage": coverage, "embedding": embedding, "fidelity": fidelity},
        )
    if (certificate.request.source_id, certificate.request.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        raise qualification_failure(
            "_hybrid_qualification_acceptance",
            ("Hybrid successor lost CAD source authority."),
            {"certificate": certificate, "source": source},
        )
    if (
        not result.audit.validity.all_certified
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise qualification_failure(
            "_hybrid_qualification_acceptance",
            ("Hybrid mapped-cell or compliance certificate failed."),
            {"result": result},
        )
    kinds = {block.cell_kind for block in result.mesh.blocks}
    if kinds != {"prism", "tetrahedron"}:
        raise qualification_failure(
            "_hybrid_qualification_acceptance",
            ("Hybrid lifecycle substituted or lost a required mixed cell family."),
            {"kinds": kinds},
        )
    if (
        coverage.region_ids != ("fluid",)
        or coverage.covered_source_facet_count != coverage.source_facet_count
    ):
        raise qualification_failure(
            "_hybrid_qualification_acceptance",
            ("Hybrid source material or coverage of every boundary patch changed."),
            {"coverage": coverage},
        )
    markers = {attribute.name: attribute for attribute in result.attributes}
    layer = markers["layer_index"]
    by_id = dict(
        zip(
            np.asarray(layer.scope.entity_ids).tolist(),
            np.asarray(layer.values).tolist(),
            strict=True,
        )
    )
    thicknesses = source.layers.control.schedule.thicknesses
    starts = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(thicknesses)))
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    defects = []
    for block in result.mesh.blocks:
        for identifier, corners in zip(
            np.asarray(block.global_ids),
            np.asarray(block.vertices),
            strict=True,
        ):
            interval = by_id[int(identifier)]
            if interval >= 0:
                heights = points[corners, 2]
                defects.extend(
                    (
                        abs(float(np.min(heights)) - starts[interval]),
                        abs(float(np.max(heights)) - starts[interval + 1]),
                    )
                )
    if not defects or max(defects) > 1.0e-10:
        raise qualification_failure(
            "_hybrid_qualification_acceptance",
            ("Hybrid mixed adaptation changed a hard layer interval."),
            {"defects": defects},
        )
    return {
        "passed": True,
        "coverage_id": coverage.certificate_id,
        "source_fidelity_id": fidelity.certificate_id,
        "embedding_id": embedding.certificate_id,
        "source_id": certificate.request.source_id,
        "source_revision": certificate.request.source_revision,
        "domain_id": coverage.domain_id,
        "covered_source_facets": coverage.covered_source_facet_count,
        "source_facets": coverage.source_facet_count,
        "material_ids": list(coverage.region_ids),
        "requested_region_measures": list(coverage.requested_region_measures),
        "achieved_region_measures": list(coverage.achieved_region_measures),
        "hard_schedule_maximum_defect": float(max(defects)),
        "map_certificate": _cad_curving_certificate(result.audit.validity),
        "quality": _quality_summary(result.quality),
    }


def _hybrid_qualification_columns(result: Any) -> Any:
    attributes = {attribute.name: attribute for attribute in result.attributes}
    layer, column = attributes["layer_index"], attributes["layer_column"]
    columns = dict(
        zip(
            np.asarray(column.scope.entity_ids).tolist(),
            np.asarray(column.values).tolist(),
            strict=True,
        )
    )
    ids = np.asarray(layer.scope.entity_ids, dtype=np.int64)
    intervals = np.asarray(layer.values, dtype=np.int64)
    active = intervals >= 0
    return phx.meshing.MixedLayerColumns(
        ids[active],
        np.asarray(
            [columns[int(identifier)] for identifier in ids[active]], dtype=np.int64
        ),
        intervals[active],
        hard_first_thickness=True,
        axial_refinement=False,
    )


def _hybrid_qualification_cap_identity(result: Any, source: Any) -> dict[str, object]:
    """Recount the shared interface with row identities, not coordinate welding."""

    layers = source.layers
    rows = np.asarray(layers.cap_vertices, dtype=np.int64)
    if not np.array_equal(
        np.asarray(result.mesh.coordinates)[rows],
        np.asarray(layers.cap.coordinates),
    ):
        raise qualification_failure(
            "_hybrid_qualification_cap_identity",
            ("Native core publication moved an immutable layer-cap vertex."),
            {"layers": layers, "result": result, "rows": rows},
        )
    cap = rows[
        np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in layers.cap.blocks]
        )
    ]
    incidence: dict[tuple[int, ...], list[str]] = {}
    for block in result.mesh.blocks:
        facets = phx.discretization.reference_cell_topology(block.cell_kind).entities[2]
        for cell in np.asarray(block.vertices, dtype=np.int64):
            for facet in facets:
                key = tuple(sorted(cell[np.asarray(facet, dtype=np.int64)].tolist()))
                incidence.setdefault(key, []).append(block.cell_kind)
    if any(
        sorted(incidence.get(tuple(sorted(face.tolist())), []))
        != ["prism", "tetrahedron"]
        for face in cap
    ):
        raise qualification_failure(
            "_hybrid_qualification_cap_identity",
            ("An immutable cap face lacks exactly one prism and one core tetrahedron."),
            {"cap": cap, "incidence": incidence},
        )
    return {
        "cap_mesh_id": layers.cap.mesh_id,
        "shared_vertex_rows": rows.tolist(),
        "bitwise_coordinates_retained": True,
        "shared_cap_faces": cap.shape[0],
        "incidence": "one-prism-and-one-tetrahedron",
        "identity_semantics": "explicit-layer-row-maps-without-tolerance-welding",
    }


def _hybrid_qualification_lineage(adaptation: Any) -> dict[str, object]:
    """Check each entity dimension; prism vertex count is not dimension."""

    lineage = adaptation.lineage
    source, target = adaptation.source.mesh, adaptation.target.mesh
    if lineage is None:
        raise qualification_failure(
            "_hybrid_qualification_lineage",
            "retain lineage for every entity dimension",
            {"lineage": lineage},
        )
    return lineage.validate_complete(source, target).to_record()


def _hybrid_qualification_state_id(result: Any, values: Any) -> str:

    return canonical_fingerprint(
        {
            "result": result.result_id,
            "topology": result.mesh.topology_id,
            "arrays": array_tree_fingerprint(
                (
                    result.mesh.coordinates,
                    result.geometry.coordinates,
                    result.mesh.vertex_global_ids,
                    tuple(block.vertices for block in result.mesh.blocks),
                    tuple(block.global_ids for block in result.mesh.blocks),
                    values,
                )
            ),
        }
    )


def _hybrid_qualification_rejection(
    adapted: Any,
    values: Any,
    policy: Any,
    record: dict[str, Any],
    cycle: int,
) -> None:
    """Retain every accepted state role when a sibling request omits children."""

    if adapted.hierarchy is None or len(adapted.hierarchy.siblings) == 0:
        raise qualification_failure(
            "_hybrid_qualification_rejection",
            ("Hybrid refinement omitted sibling ancestry records."),
            {"adapted": adapted},
        )
    child = adapted.hierarchy.siblings[0].child_ids[0]
    source = adapted.target
    before = _hybrid_qualification_state_id(source, values)
    request = phx.meshing.MarkedMeshAdaptation(
        (),
        np.asarray([child], dtype=np.int64),
        hierarchy=adapted.hierarchy,
        layer_columns=_hybrid_qualification_columns(source),
    )
    prepared = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"rejection_prepare_cycle_{cycle}",
        lambda: phx.meshing.prepare_mesh_adaptation(source, request, policy=policy),
    )
    refused = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"rejection_execute_cycle_{cycle}",
        lambda: phx.meshing.execute_mesh_adaptation(prepared),
    )
    if refused.target.result_id != source.result_id or refused.transition is not None:
        raise qualification_failure(
            "_hybrid_qualification_rejection",
            ("Incomplete mixed sibling coarsening published a changed target."),
            {"refused": refused, "source": source},
        )
    rejected_ids = _hybrid_rejected_ids(refused)
    if child not in rejected_ids:
        raise qualification_failure(
            "_hybrid_qualification_rejection",
            ("Incomplete mixed sibling refusal omitted the rejected child."),
            {"child": child, "refused": refused},
        )
    after = _hybrid_qualification_state_id(source, values)
    if before != after:
        raise qualification_failure(
            "_hybrid_qualification_rejection",
            ("Refused mixed coarsening changed accepted geometry, identities, or state."),
            {"after": after, "before": before},
        )
    record["rejection_retention"].append(
        {
            "cycle": cycle,
            "request_id": request.request_id,
            "status": refused.status.value,
            "rejected_coarsening_ids": np.asarray(rejected_ids).tolist(),
            "accepted_result_id": source.result_id,
            "before_state_id": before,
            "after_state_id": after,
            "exact_retention": True,
        }
    )


def _hybrid_scalar_request[T](
    result: Any,
    limits: Any,
    current_preparation: Any,
    call: Callable[[CoordinateEnclosureBudget], T],
    /,
) -> tuple[T, Any]:
    previous = _layer_core_current_preparation(result, current_preparation)

    def execute(execution: NativeExecutionScope) -> T:
        with execution.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with budget.activate(), meshing_source_owner_reconstruction_scope(budget=budget):
                return call(budget)

    return layer_core_continuation_execution(limits, previous, execute)


def _hybrid_qualification_transition(
    result: Any,
    space: Any,
    values: Any,
    request: Any,
    policy: Any,
    source: Any,
    record: dict[str, Any],
    label: str,
    coarsening: bool,
    *,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    prepared = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"{label}_prepare",
        lambda: phx.meshing.prepare_mesh_adaptation(
            result, request, policy=policy, observe_managed_admission=True
        ),
    )
    adaptation = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"{label}_execute",
        lambda: phx.meshing.execute_mesh_adaptation(
            prepared,
            coordinate_operands=coordinate_operands,
            observe_managed_admission=True,
        ),
    )
    if (
        adaptation.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
        or adaptation.transition is None
        or adaptation.geometry_transition is None
    ):
        raise qualification_failure(
            "_hybrid_qualification_transition",
            "Hybrid mixed adaptation did not complete every requested cell with a geometry-preserving transition.",
            {"adaptation": adaptation},
        )
    if not adaptation.geometry_transition.evidence.exact:
        raise qualification_failure(
            "_hybrid_qualification_transition",
            "Hybrid mixed refinement/coarsening changed the coordinate map.",
            {"adaptation": adaptation},
        )
    payload, current = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"{label}_original_scalar_request",
        lambda: _hybrid_scalar_request(
            adaptation.target,
            policy.limits,
            None,
            lambda budget: _hybrid_scalar_transition(
                result, space, values, request, source, record, label, coarsening,
                prepared=prepared, adaptation=adaptation, budget=budget,
            ),
        ),
    )
    return *payload, current


def _hybrid_scalar_transition(
    result: Any,
    space: Any,
    values: Any,
    request: Any,
    source: Any,
    record: dict[str, Any],
    label: str,
    coarsening: bool,
    *,
    prepared: Any,
    adaptation: Any,
    budget: CoordinateEnclosureBudget,
) -> tuple[Any, Any, Any, Any, Any]:

    acceptance = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"{label}_source_recertification",
        lambda: _hybrid_qualification_acceptance(adaptation.target, source),
    )
    target_solver = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"{label}_target_solver_preparation",
        lambda: _hybrid_qualification_solver(adaptation.target, budget=budget),
    )
    target_space, problem, free = target_solver
    transfer = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"{label}_field_transfer_preparation",
        lambda: phx.discretization.prepare_nested_field_transfer(
            space,
            target_space,
            None if coarsening else adaptation.parent_cells,
            field_name="u",
            parent_reference_vertices=None
            if coarsening
            else adaptation.parent_reference_vertices,
            parent_reference_homogeneous=None
            if coarsening
            else adaptation.parent_reference_homogeneous,
            source_geometry=result.geometry,
            target_geometry=adaptation.target.geometry,
            geometry_transition=adaptation.geometry_transition,
            budget=budget,
        ),
    )
    if not transfer.evidence.passed:
        raise qualification_failure(
            "_hybrid_qualification_transition",
            ("Hybrid scalar H1 transfer certification failed."),
            {"transfer": transfer},
        )
    carried = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"{label}_actual_scalar_state_transfer",
        lambda: jax.block_until_ready(transfer.transfer.apply(values)),
    )

    probe = transfer.transfer.apply(exact_solution(space.dof_maps[0].dof_coordinates))
    linear_defect = float(
        jnp.max(jnp.abs(probe - exact_solution(target_space.dof_maps[0].dof_coordinates)))
    )
    constant_defect = float(
        jnp.max(jnp.abs(transfer.transfer.apply(jnp.ones_like(values)) - 1.0))
    )
    before = _hybrid_qualification_physical(result, space, values)
    after = _hybrid_qualification_physical(adaptation.target, target_space, carried)
    inventory_defect = after["state_integral"] - before["state_integral"]
    if max(linear_defect, constant_defect, abs(inventory_defect)) > 1.0e-9:
        raise qualification_failure(
            "_hybrid_qualification_transition",
            ("Hybrid scalar patch transfer changed the field or its physical inventory."),
            {
                "constant_defect": constant_defect,
                "inventory_defect": inventory_defect,
                "linear_defect": linear_defect,
            },
        )
    record["transitions"].append(
        {
            "operation": label,
            "request_id": request.request_id,
            "prepared_id": prepared.prepared_id,
            "status": adaptation.status.value,
            "source": _result_identity(result),
            "target": _result_identity(adaptation.target),
            "acceptance": acceptance,
            "geometry_transition_id": adaptation.geometry_transition.transition_id,
            "exact_geometry_transfer": adaptation.geometry_transition.evidence.exact,
            "lineage": _hybrid_qualification_lineage(adaptation),
            "field": "u",
            "field_semantics": transfer.semantics,
            "constant_reproduction_defect": constant_defect,
            "linear_reproduction_defect": linear_defect,
            "source_state_integral": before["state_integral"],
            "target_state_integral": after["state_integral"],
            "state_inventory_defect": inventory_defect,
            "transferred_state_error": after["l2_error"],
            "retained_bytes": model_array_bytes((adaptation, transfer)),
            "general_coarsening_conservation": "not-claimed-scalar-patch-only",
        }
    )
    return adaptation, target_space, problem, free, carried


def _hybrid_qualification_epoch(
    result: Any,
    space: Any,
    problem: Any,
    free: Any,
    carried: Any,
    args: QualificationControls,
    record: dict[str, Any],
    epoch: int,
    name: str,
    *,
    limits: Any,
    current_preparation: Any,
) -> tuple[jax.Array, float, Any]:
    (values, error), current = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"{name}_original_scalar_epoch_request",
        lambda: _hybrid_scalar_request(
            result,
            limits,
            current_preparation,
            lambda budget: _hybrid_scalar_epoch_values(
                result, space, problem, free, carried, args, record, epoch, name,
                budget=budget,
            ),
        ),
    )
    record["epochs"][-1]["original_scalar_execution_evidence"] = current.to_record()
    return values, error, current


def _hybrid_scalar_epoch_values(
    result: Any,
    space: Any,
    problem: Any,
    free: Any,
    carried: Any,
    args: QualificationControls,
    record: dict[str, Any],
    epoch: int,
    name: str,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[jax.Array, float]:

    space.require_exact_plan(
        space.construction_plan, numeric_version=space.numeric_version, budget=budget
    )
    seed = (
        jnp.zeros((free.size,), dtype=result.mesh.coordinates.dtype)
        if carried is None
        else jnp.asarray(carried)[free]
    )
    solved = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"{name}_solver_execution",
        lambda: _compiled_solver_stages(
            problem,
            seed,
            args.repeats,
            policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU()),
        ),
    )
    values, compiler, timings = solved
    record["stages_seconds"].update(
        {epoch_stage_name(key, epoch): value for key, value in timings.items()}
    )
    physical = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        f"{name}_physical_error",
        lambda: _hybrid_qualification_physical(result, space, values),
    )
    if abs(physical["analytic_volume_defect"]) > 1.0e-10:
        raise qualification_failure(
            "_hybrid_qualification_epoch",
            ("Hybrid mapped volume differs from the independent analytic unit box."),
            {"physical": physical},
        )
    record["epochs"].append(
        {
            "epoch": epoch,
            "name": name,
            "result": _result_identity(result),
            "physical": physical,
            "dofs": space.dof_maps[0].global_dof_count,
            "free_dofs": free.size,
            "compiler": compiler,
            "seed_origin": "zero"
            if carried is None
            else "actual-transferred-scalar-H1-state",
            "retained_bytes": model_array_bytes(
                (result, space, problem, values, carried)
            ),
        }
    )
    record["physical_error"] = physical["l2_error"]
    if (
        physical["l2_error"] <= args.target_error
        and record["time_to_target_error_seconds"] is None
    ):
        record["time_to_target_error_seconds"] = perf_counter() - record["started"]
    return values, physical["l2_error"]


def _hybrid_unit_box_archive_continuation(
    args: QualificationControls,
    record: dict[str, Any],
    source: phx.meshing.NativeLayerCoreSource,
    specification: phx.meshing.VolumeMeshingSpec,
    epochs: tuple[MeshingAcceptedEpoch, ...],
) -> _HybridContinuationEvidence:
    """Restore the entire carried history and replay its final coarsening cold."""

    continuation: dict[str, Any] = {
        "started": record["started"],
        "targets": record["targets"],
        "stages_seconds": {},
        "active_phase": "unit_box_default_archive_renewal",
    }
    record["cold_continuation"] = continuation
    ledger = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    )
    try:
        with TemporaryDirectory(prefix="native-unit-box-lineage-") as temporary:
            receipt = ledger.run(
                "unit_box_default_durable_lineage",
                lambda: write_meshing_source_closure(
                    Path(temporary) / "lifecycle",
                    layer_core_lineage_records(
                        source,
                        specification,
                        phx.meshing.NativeMeshingOptions("layer_core"),
                        epochs,
                    ),
                ),
            )
            consumer = QualificationControls.from_record(
                {
                    **args.to_record(),
                    "hybrid_archive": str(receipt.path),
                    "hybrid_content_id": receipt.content_id,
                }
            )
            continued = _hybrid_qualification_archived_profile(
                consumer, continuation, expected_rounds=args.adaptation_rounds
            )
        restored = continuation["restored_epochs"]
        if [epoch["epoch_id"] for epoch in restored] != [
            epoch.epoch_id for epoch in epochs
        ]:
            raise qualification_failure(
                "_hybrid_unit_box_archive_continuation",
                "Cold unit-box restoration must retain every warm accepted epoch.",
                {"restored": restored, "warm": [epoch.epoch_id for epoch in epochs]},
            )
        record["accepted_epochs"] = restored
        for section in (
            "allfields", "finite_volume", "history", "material_declaration",
            "original_scalar",
        ):
            record[section] = continuation[section]
        return continued
    except Exception:
        ledger.inherit_failure(continuation)
        raise
    finally:
        del continuation["started"]
        del continuation["active_phase"]


def _hybrid_qualification_epochs(
    result: Any,
    source: Any,
    projection: Any,
    specification: Any,
    args: QualificationControls,
    record: dict[str, Any],
    *,
    prepared_operands: PreparedCoordinateOperands | None = None,
) -> float:

    (space, problem, free), current = StageLedger(
        record, deadline=Deadline(record["started"] + record["targets"]["wall_seconds"])
    ).run(
        "initial_solver_preparation",
        lambda: _hybrid_scalar_request(
            result,
            specification.limits,
            None,
            lambda budget: _hybrid_qualification_solver(result, budget=budget),
        ),
    )
    values, physical_error, current = _hybrid_qualification_epoch(
        result, space, problem, free, None, args, record, 0, "initial",
        limits=specification.limits, current_preparation=current,
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
        limits=specification.limits,
        association_transfer=phx.meshing.BRepAssociationTransfer(
            projection,
            policy=phx.meshing.AssociationPropagationPolicy(
                classification_tolerance=1.0e-8
            ),
        ),
    )
    accepted_epochs: tuple[MeshingAcceptedEpoch, ...] = ()
    warm: _HybridWarmLineage | None = None
    for cycle in range(args.adaptation_rounds):
        root, root_values = result, values
        root_preparation = current
        refinement_operands = CoordinateOperandCollector()
        coarsening_operands = CoordinateOperandCollector()
        core_ids = np.concatenate(
            [
                np.asarray(block.global_ids, dtype=np.int64)
                for block in result.mesh.blocks
                if block.cell_kind == "tetrahedron"
            ]
        )
        request = phx.meshing.MarkedMeshAdaptation(
            core_ids, layer_columns=_hybrid_qualification_columns(result)
        )
        adaptation, space, problem, free, carried, current = _hybrid_qualification_transition(
            result,
            space,
            values,
            request,
            policy,
            source,
            record,
            f"refine_{cycle}",
            False,
            coordinate_operands=refinement_operands,
        )
        result = adaptation.target
        values, physical_error, current = _hybrid_qualification_epoch(
            result,
            space,
            problem,
            free,
            carried,
            args,
            record,
            2 * cycle + 1,
            f"refine_{cycle}",
            limits=specification.limits,
            current_preparation=current,
        )
        refinement, refined_values = adaptation, values
        refined_preparation = current
        children = np.unique(adaptation.hierarchy.siblings.all_child_ids())
        request = phx.meshing.MarkedMeshAdaptation(
            (),
            children,
            hierarchy=adaptation.hierarchy,
            layer_columns=_hybrid_qualification_columns(result),
        )
        adaptation, space, problem, free, carried, current = _hybrid_qualification_transition(
            result,
            space,
            values,
            request,
            policy,
            source,
            record,
            f"coarsen_{cycle}",
            True,
            coordinate_operands=coarsening_operands,
        )
        result = adaptation.target
        values, physical_error, current = _hybrid_qualification_epoch(
            result,
            space,
            problem,
            free,
            carried,
            args,
            record,
            2 * cycle + 2,
            f"coarsen_{cycle}",
            limits=specification.limits,
            current_preparation=current,
        )
        cycle_epochs, warm = _hybrid_warm_cycle(
            args,
            record,
            root,
            _HybridAdaptationCycle(refinement, adaptation),
            cycle,
            part_name="unit-box-layer-core",
            limits=specification.limits,
            carried=warm,
            original_scalar_values=(root_values, refined_values, values),
            original_scalar_preparations=(root_preparation, refined_preparation, current),
            prepared_operands=prepared_operands if cycle == 0 else None,
            transition_operands=(
                refinement_operands.for_result(refinement.target.mesh, refinement.target.geometry),
                coarsening_operands.for_result(adaptation.target.mesh, adaptation.target.geometry),
            ),
        )
        accepted_epochs += cycle_epochs if cycle == 0 else cycle_epochs[1:]
        _hybrid_qualification_rejection(
            refinement,
            (refined_values, cycle_epochs[1].fields),
            policy,
            record,
            cycle,
        )
        if result.mesh.topology_id != root.mesh.topology_id:
            raise qualification_failure(
                "_hybrid_qualification_epochs",
                ("Coarsening every hybrid sibling did not restore the source topology."),
                {"result": result, "root": root},
            )
        require_restored_layer_core_result_identity(root, result)
        columns = _hybrid_qualification_columns(result)
        original_columns = _hybrid_qualification_columns(root)
        if columns.columns_id != original_columns.columns_id:
            raise qualification_failure(
                "_hybrid_qualification_epochs",
                "Inverse coarsening must restore the authored physical layer columns.",
                {"columns": columns, "original_columns": original_columns},
            )
        record["invariants"].append(
            {
                "cycle": cycle,
                "complete_refine_coarsen": True,
                "restored_topology_id": result.mesh.topology_id,
                "restored_coordinate_map_bitwise": True,
                "restored_columns_id": columns.columns_id,
                "final_source_acceptance": _hybrid_qualification_acceptance(
                    result, source
                ),
            }
        )
    continued = _hybrid_unit_box_archive_continuation(
        args, record, source, specification, accepted_epochs
    )
    return max(physical_error, continued.physical_error)


def _hybrid_qualification_lifecycle(args: QualificationControls) -> dict[str, object]:
    """Callable native generated wall → curved mixed epochs → transferred-state PDE."""

    densities, material_site = _hybrid_material_controls(
        args.hybrid_profile, args.hybrid_region_densities, args.hybrid_material_site
    )
    if args.hybrid_region_densities is None or args.hybrid_material_site is None:
        args = replace(
            args, hybrid_region_densities=densities, hybrid_material_site=material_site
        )
    if args.hybrid_profile != "unit-box":
        return _hybrid_qualification_original_profile(args)
    counts = (
        args.resolution,
        args.capacity,
        args.layer_count,
        args.repeats,
        args.adaptation_rounds,
    )
    if (
        any(value < 1 for value in counts)
        or args.resolution < 2
        or args.geometry_order not in (2, 3, 4, 6, 10)
        or not all(
            np.isfinite(value) and value > 0.0
            for value in (
                args.first_thickness,
                args.layer_growth,
                args.timeout,
                args.target_error,
            )
        )
        or args.layer_growth < 1.0
    ):
        raise ValueError(
            f"_hybrid_qualification_lifecycle.bounds must have resolution at least two, positive counts and finite positive thickness, growth, timeout and error, growth at least one, and geometry order in [2, 3, 4, 6, 10]; got {(counts, args.resolution, args.geometry_order, args.first_thickness, args.layer_growth, args.timeout, args.target_error)!r}."
        )
    schedule = phx.meshing.LayerSchedule.geometric(
        args.layer_count, args.first_thickness, growth_rate=args.layer_growth
    )
    if sum(schedule.thicknesses) >= 1.0:
        raise ValueError(
            f"_hybrid_qualification_lifecycle.thicknesses must sum to less than one to leave positive core height; got {schedule.thicknesses!r}."
        )
    started = perf_counter()
    phases: list[NativeMeshingPhaseMeasurement] = []
    record: dict[str, Any] = {
        "scenario": "native-hybrid-lifecycle",
        "engine": "native",
        "status": "failed",
        "runtime": _runtime_identity(),
        "started": started,
        "active_phase": "source_preparation",
        "expected_admission": (
            contract := _hybrid_qualification_profile_contracts()["unit-box"]
        )["expected_admission"],
        "contract": contract,
        "targets": {"l2_error": args.target_error, "wall_seconds": args.timeout},
        "control_capacities": {
            "resolution": args.resolution,
            "entity_capacity": args.capacity,
            "layer_count": args.layer_count,
            "first_thickness": args.first_thickness,
            "layer_growth": args.layer_growth,
            "geometry_order": args.geometry_order,
            "adaptation_cycles": args.adaptation_rounds,
            "warm_repeats": args.repeats,
        },
        "manufactured_solution": {
            "exact": "x+y+z",
            "forcing": 0.0,
            "diffusion": 1.0,
            "norm": "mixed-mapped-physical-quadrature-L2-versus-analytic-patch",
            "declared_fields": [
                "H1", "DG", "Hcurl", "Hdiv", "u", "damage-history", "state",
                "original-u",
            ],
        },
        "stages_seconds": {},
        "epochs": [],
        "transitions": [],
        "invariants": [],
        "rejection_retention": [],
        "time_to_target_error_seconds": None,
        "physical_error": None,
        "target_error_reached": False,
        "independent_scientific_certification": "unassessed",
        "case_contract_completion": "incomplete",
        "mandatory_workflow_completion": "incomplete",
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
        "nonclaims": _hybrid_qualification_requirements()["nonclaims"],
        "deadline_scope": "native-work-bounds-and-between-measured-phase-boundaries",
    }
    try:
        limits, certificates = _hybrid_qualification_limits(args)
        result, source, projection, query, specification = (
            _hybrid_qualification_generation(args, record, phases, limits)
        )
        record["generation_acceptance"] = _hybrid_qualification_acceptance(result, source)
        coordinate_operands = CoordinateOperandCollector(
            capture_signed_determinants=True
        )
        result = _hybrid_qualification_curving(
            result, query, projection, args, limits, certificates, record,
            coordinate_operands=coordinate_operands,
        )
        record["curved_acceptance"] = _hybrid_qualification_acceptance(result, source)
        physical_error = _hybrid_qualification_epochs(
            result, source, projection, specification, args, record,
            prepared_operands=coordinate_operands.for_result(result.mesh, result.geometry),
        )
        record["target_error_reached"] = physical_error <= args.target_error
        record["status"] = "passed" if record["target_error_reached"] else "failed"
        if record["target_error_reached"]:
            record["independent_scientific_certification"] = "passed"
        record["implemented_admitted_workflow"] = (
            "native-wall-layer-core-curving-allfields-FV-history-accepted-state-cold-continuation-diffusion"
        )
    except (
        phx.meshing.MeshingFailure,
        RuntimeError,
        ValueError,
        TypeError,
        TimeoutError,
    ) as failure:
        record_failure(record, failure, record["active_phase"])
    finally:
        record["native_phases"] = _phase_recordings(phases)
        record["end_to_end_seconds"] = perf_counter() - started
        record["peak_process_resident_bytes"] = peak_resident_bytes()
        record["process_peak_scope"] = PEAK_RESIDENT_SCOPE
        record["device_memory"] = [
            {
                "device_id": device.id,
                "platform": device.platform,
                "kind": device.device_kind,
                "status": "unavailable" if statistics is None else "backend-reported",
                "peak_bytes": None
                if statistics is None
                else statistics.get("peak_bytes_in_use"),
                "statistics": statistics,
            }
            for device in jax.local_devices()
            for statistics in (device.memory_stats(),)
        ]
        if record["end_to_end_seconds"] > args.timeout:
            record["status"] = "timeout"
        # The routine contract certifies this case only; a full-workflow
        # contract additionally completes the mandatory workflow.
        if record["status"] == "passed":
            record["case_contract_completion"] = "complete"
            if contract["full_workflow"] is True:
                record["mandatory_workflow_completion"] = "complete"
        del record["active_phase"]
        del record["started"]
    return record


def cold_worker(packet: ColdControlPacket) -> ColdWorkerReport:
    """Convert the layer continuation's line records to one cold worker report."""

    controls = packet.controls
    sys.argv = [
        "tools.layer_core_lifecycle_continuation",
        str(packet.archive_path),
        packet.content_id,
        "--region-densities",
        *(str(float(value)) for value in controls["region_densities"]),
        "--material-site",
        str(controls["material_site"]),
        "--solver-repeats",
        str(int(controls["solver_repeats"])),
    ]
    stream = StringIO()
    with redirect_stdout(stream):
        main()
    evidence = [
        json.loads(line) for line in stream.getvalue().splitlines() if line.strip()
    ]
    stages = evidence[0]["stages_seconds"]
    return ColdWorkerReport(
        record={"evidence": evidence},
        read_seconds=stages["archive_member_authentication"]
        + stages["execution_control_restoration"]
        + stages["scientific_owner_restoration"],
    )


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _hybrid_qualification_lifecycle(args)
