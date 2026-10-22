# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, final

import numpy as np
from jax import Array

import phydrax as phx
from phydrax import model_array_bytes
from phydrax.meshing import meshcore_available
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._records import _quality_summary, _result_identity
from tools.meshing_qualification._runtime import (
    _contract,
    _measured,
    _missing_dependency,
    _runtime_identity,
    qualification_failure,
    StageMeasurement,
)
from tools.meshing_qualification.scenarios.boundary_layer import (
    _layer_certificate_counts,
    _layer_enclosed_volume,
    _layer_entity_rows,
    _layer_quality,
    _layer_wall_mesh,
)


@final
class _GmshLayerPublication(phx.StrictModule):
    control: phx.meshing.BoundaryLayerControl
    plan: phx.meshing.GmshMeshingPlan
    result: phx.meshing.CellMeshingResult
    certificate: phx.discretization.CellValidityCertificate
    planning: StageMeasurement
    execution: StageMeasurement
    certification: StageMeasurement


@final
class _GmshLayerMeasurements(phx.StrictModule):
    patches: dict[str, np.ndarray]
    zones: dict[str, Array]
    conforming: bool | np.bool_
    domain_volume: float
    boundary_volume: float
    layer_volume: float
    layer_enclosed: float
    domain_residual: float
    layer_residual: float
    before: phx.meshing.CellQualityReport
    achieved: dict[str, float]


def _gmsh_layer_publication(
    provider: phx.meshing.GmshProvider,
    source: phx.geometry.BRepModel,
    route: phx.meshing.BoundaryLayerRoute,
    corner: phx.meshing.BoundaryLayerCornerPolicy,
    schedule: phx.meshing.LayerSchedule,
) -> _GmshLayerPublication:
    """Execute the layered request and retain its independent validity certificate."""
    points = np.asarray(source.mesh_vertices)
    face_ids = np.asarray(source.triangle_face_ids)
    inner = np.all(np.linalg.norm(points[np.asarray(source.mesh_faces)], axis=2) < 0.5, 1)
    wall_faces = tuple(source.face_ids[index] for index in np.unique(face_ids[inner]))
    whole = provider.whole_scope(source, 3)
    control = phx.meshing.BoundaryLayerControl(
        provider.entity_scope(source, wall_faces),
        schedule,
        route=route,
        volume_scope=whole,
        corner=corner,
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allowed_transitions=("pyramid", "hexahedron"),
                allow_mixed=True,
            ),
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(whole, 0.25, maximum_growth_rate=10.0),
        ),
        layer_controls=(control,),
    )
    prepared, planning = _measured(lambda: provider.plan(source, specification))
    match prepared:
        case phx.meshing.GmshMeshingPlan() as plan:
            result, execution = _measured(plan.execute)
        case plan:
            raise qualification_failure(
                "gmsh.boundary_layer_plan",
                "retain the requested BRep volume meshing plan",
                {"plan": plan, "specification": specification},
            )

    certificate, certification = _measured(
        lambda: phx.discretization.certify_cell_geometry_validity(result.mesh)
    )
    return _GmshLayerPublication(
        control, plan, result, certificate, planning, execution, certification
    )


def _gmsh_layer_measurements(
    publication: _GmshLayerPublication,
) -> _GmshLayerMeasurements:
    """Measure conforming zones and independent oriented domain and layer volumes."""
    result, control = publication.result, publication.control
    mesh = result.mesh
    match mesh.connectivity:
        case phx.discretization.PolyhedralConnectivity() as connectivity:
            owners = np.asarray(connectivity.face_owner, dtype=np.int64)
        case connectivity:
            raise qualification_failure(
                "gmsh.boundary_layer_connectivity",
                "publish packed polyhedral face incidence",
                {"mesh": mesh, "connectivity": connectivity},
            )
    neighbors = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    face_entities = mesh.entity_set(2).entity_ids
    patches = {
        patch.name: _layer_entity_rows(face_entities, patch.scope.entity_ids)
        for patch in result.patches
    }

    zones = {zone.name: zone.scope.entity_ids for zone in result.zones}
    if set(zones) != {"boundary-layer", "core"} or set(patches) != {
        "wall",
        "layer-core-interface",
        "outer",
    }:
        raise qualification_failure(
            "_gmsh_layered_route",
            ("Layered Gmsh result lost its zones or patches."),
            {"patches": patches, "zones": zones},
        )
    cell_entities = mesh.entity_set(3).entity_ids
    zone_of = np.full((cell_entities.shape[0],), -1, dtype=np.int8)
    zone_of[_layer_entity_rows(cell_entities, zones["boundary-layer"])] = 0
    zone_of[_layer_entity_rows(cell_entities, zones["core"])] = 1

    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    entry_cells = np.repeat(np.arange(zone_of.size), np.diff(cell_face_offsets))
    entry_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    entry_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)

    def triangles(faces: np.ndarray, zones: tuple[int, ...]) -> np.ndarray:
        """Triangles of ``faces`` oriented out of their incident cells in ``zones``."""
        selected = np.isin(entry_faces, faces) & np.isin(zone_of[entry_cells], zones)
        rows = entry_faces[selected]
        if rows.size != faces.size or np.any(offsets[rows + 1] - offsets[rows] != 3):
            raise qualification_failure(
                "triangles",
                ("Layered Gmsh wall and interface faces must be triangles."),
                {"faces": faces, "offsets": offsets, "rows": rows},
            )
        vertices = values[offsets[rows][:, None] + np.arange(3)]
        return np.where(entry_signs[selected][:, None] > 0.0, vertices, vertices[:, ::-1])

    interface = patches["layer-core-interface"]
    boundary = np.flatnonzero(neighbors < 0)
    conforming = (
        np.all(zone_of >= 0)
        and np.array_equal(
            np.sort(
                np.stack((zone_of[owners[interface]], zone_of[neighbors[interface]]), 1),
                1,
            ),
            np.tile((0, 1), (interface.size, 1)),
        )
        and np.array_equal(
            np.sort(np.concatenate((patches["wall"], patches["outer"]))), boundary
        )
    )
    coordinates = np.asarray(mesh.coordinates)

    evaluation = result.quality.evaluation
    measures = np.asarray(evaluation.measures)
    measure_rows = _layer_entity_rows(evaluation.cell_global_ids, cell_entities)
    cell_measures = measures[measure_rows]
    domain_volume = float(np.sum(cell_measures))
    boundary_volume = _layer_enclosed_volume(
        coordinates,
        triangles(np.concatenate((patches["wall"], patches["outer"])), (0, 1)),
    )
    wall_triangles = triangles(patches["wall"], (0,))
    layer_volume = float(np.sum(cell_measures[zone_of == 0]))
    layer_enclosed = _layer_enclosed_volume(
        coordinates, np.concatenate((wall_triangles, triangles(interface, (0,))))
    )
    domain_residual = abs(domain_volume - boundary_volume) / boundary_volume
    layer_residual = abs(layer_volume - layer_enclosed) / layer_enclosed
    wall_vertices, wall_inverse = np.unique(wall_triangles, return_inverse=True)
    before = _layer_quality(
        _layer_wall_mesh(coordinates[wall_vertices], wall_inverse.reshape((-1, 3)))
    )
    key = f"layer:{control.control_id}:"
    achieved = {
        name.removeprefix(key): value
        for name, value in result.compliance.achieved
        if name.startswith(key)
        or name in ("fixed_boundary_bitwise", "layer_interface_compliance")
    }
    return _GmshLayerMeasurements(
        patches,
        zones,
        conforming,
        domain_volume,
        boundary_volume,
        layer_volume,
        layer_enclosed,
        domain_residual,
        layer_residual,
        before,
        achieved,
    )


@final
class _GmshLayerCompliance(phx.StrictModule):
    thickness_error: float
    growth_error: float


def _gmsh_layer_compliance(
    publication: _GmshLayerPublication,
    measured: _GmshLayerMeasurements,
    route: phx.meshing.BoundaryLayerRoute,
    schedule: phx.meshing.LayerSchedule,
    tolerance: float,
) -> _GmshLayerCompliance:
    """Accept provider audit, layer compliance, independent validity, and volume conservation together."""
    result, mesh, certificate = (
        publication.result,
        publication.result.mesh,
        publication.certificate,
    )
    achieved, conforming = measured.achieved, measured.conforming
    domain_residual, layer_residual = measured.domain_residual, measured.layer_residual
    requested_values = np.asarray(schedule.thicknesses, dtype=np.float64)
    thickness_error = float(
        np.max(
            np.abs(
                np.asarray(
                    [
                        achieved[f"thickness:{index}"]
                        for index in range(schedule.layer_count)
                    ]
                )
                - requested_values
            )
            / requested_values
        )
    )
    growth_error = float(
        np.max(
            np.abs(
                np.asarray(
                    [
                        achieved[f"growth:{index}"]
                        for index in range(1, schedule.layer_count)
                    ]
                )
                - np.asarray(schedule.growth_rates)
            )
        )
    )
    cells = sum(block.cell_count for block in mesh.blocks)
    if (
        not result.audit.passed
        or not result.compliance.passed
        or not all(association.complete for association in result.associations)
        or not conforming
        or achieved["fixed_boundary_bitwise"] != 1.0
        or achieved["layer_interface_compliance"] != 1.0
        or achieved["layer_count"] != schedule.layer_count
        or thickness_error > tolerance
        or growth_error > tolerance * schedule.growth_rates[0]
        or certificate.certified_valid_count != cells
        or result.quality.sampled_invalid_count != 0
        or domain_residual > 1.0e-10
        or layer_residual > 1.0e-10
    ):
        raise qualification_failure(
            "_gmsh_layered_route",
            (f"Gmsh {route.value} layers failed audit or compliance."),
            {
                "achieved": achieved,
                "cells": cells,
                "certificate": certificate,
                "conforming": conforming,
                "domain_residual": domain_residual,
                "growth_error": growth_error,
                "layer_residual": layer_residual,
                "result": result,
                "schedule": schedule,
                "thickness_error": thickness_error,
                "tolerance": tolerance,
            },
        )
    return _GmshLayerCompliance(thickness_error, growth_error)


def _gmsh_layered_route(
    provider: phx.meshing.GmshProvider,
    source: phx.geometry.BRepModel,
    route: phx.meshing.BoundaryLayerRoute,
    corner: phx.meshing.BoundaryLayerCornerPolicy,
    schedule: phx.meshing.LayerSchedule,
    tolerance: float,
    /,
) -> dict[str, Any]:
    """Plan, execute, and independently audit one layered Gmsh volume route."""
    publication = _gmsh_layer_publication(provider, source, route, corner, schedule)
    result, mesh = publication.result, publication.result.mesh
    control, plan, certificate = (
        publication.control,
        publication.plan,
        publication.certificate,
    )
    planning, execution, certification = (
        publication.planning,
        publication.execution,
        publication.certification,
    )
    measured = _gmsh_layer_measurements(publication)
    patches, zones, conforming = measured.patches, measured.zones, measured.conforming
    domain_volume, boundary_volume = measured.domain_volume, measured.boundary_volume
    layer_volume, layer_enclosed = measured.layer_volume, measured.layer_enclosed
    domain_residual, layer_residual = measured.domain_residual, measured.layer_residual
    before, achieved = measured.before, measured.achieved
    compliance = _gmsh_layer_compliance(publication, measured, route, schedule, tolerance)
    thickness_error, growth_error = compliance.thickness_error, compliance.growth_error
    return {
        "plan_id": plan.plan_id,
        "control_id": control.control_id,
        "target": _result_identity(result),
        "provider": {
            "version": result.runtime.actual_version,
            "runtime_id": result.runtime.runtime_id,
            "deterministic": result.runtime.deterministic,
        },
        "stages": {
            stage.stage.value: stage.status.value for stage in result.trace.stages
        },
        "cell_counts": {block.cell_kind: block.cell_count for block in mesh.blocks},
        "zone_cells": {name: ids.shape[0] for name, ids in zones.items()},
        "patch_faces": {name: rows.size for name, rows in patches.items()},
        "organization_conforming": bool(conforming),
        "compliance": {
            **achieved,
            "maximum_relative_thickness_error": thickness_error,
            "maximum_growth_rate_error": growth_error,
            "tolerance": tolerance,
        },
        "certificate": _layer_certificate_counts(certificate),
        "quality": {
            "before": _quality_summary(before),
            "after": _quality_summary(result.quality),
        },
        "conservation": {
            "domain_cell_volume": domain_volume,
            "domain_boundary_enclosed_volume": boundary_volume,
            "domain_relative_residual": domain_residual,
            "layer_cell_volume": layer_volume,
            "layer_boundary_enclosed_volume": layer_enclosed,
            "layer_relative_residual": layer_residual,
        },
        "resources": {
            "plan": planning,
            "execute": execution,
            "independent_certification": certification,
            "retained_mesh_bytes": model_array_bytes(mesh),
        },
    }


def qualify_gmsh_boundary_layer() -> dict[str, object]:

    # Optional provider imports are admitted only for this fixture workflow.
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Pnt

    if find_spec("gmsh") is None:
        return _missing_dependency(
            "gmsh",
            "Install the optional provider: pip install 'phydrax[meshing-gmsh]'.",
        )
    if find_spec("OCP") is None:
        return _missing_dependency(
            "OCP", "Install OpenCascade bindings (cadquery-ocp) for B-Rep persistence."
        )
    if not meshcore_available():
        return _missing_dependency(
            "meshcore",
            "Exact layer certification requires the native meshcore library; build "
            "it and set PHYDRAX_MESHCORE_LIBRARY to the shared library path.",
        )

    schedule = phx.meshing.LayerSchedule.geometric(3, 0.02, growth_rate=1.25)
    provider = phx.meshing.GmshProvider()
    with TemporaryDirectory(prefix="phydrax-gmsh-boundary-layer-") as temporary:
        box = BRepPrimAPI_MakeBox(gp_Pnt(-1.0, -1.0, -1.0), 2.0, 2.0, 2.0).Shape()
        ball = BRepPrimAPI_MakeSphere(gp_Pnt(0.0, 0.0, 0.0), 0.4).Shape()
        source = phx.interchange.persist_occt_shape(
            BRepAlgoAPI_Cut(box, ball).Shape(),
            Path(temporary) / "ball-in-box.brep",
            coordinate_contract=_contract(),
            linear_deflection=0.01,
            angular_deflection=0.1,
        )
        routes = {
            "advancing": _gmsh_layered_route(
                provider,
                source,
                phx.meshing.BoundaryLayerRoute.ADVANCING,
                phx.meshing.BoundaryLayerCornerPolicy.FAN,
                schedule,
                1.0e-6,
            ),
            "provider": _gmsh_layered_route(
                provider,
                source,
                phx.meshing.BoundaryLayerRoute.PROVIDER,
                phx.meshing.BoundaryLayerCornerPolicy.SMOOTH,
                schedule,
                2.0e-2,
            ),
        }
    sections = ("quality", "conservation", "resources")
    return {
        "status": "passed",
        "source": {
            "model_id": source.model_id,
            "source_id": source.report.source_id,
            "source_revision": source.report.source_revision,
            "faces": len(source.face_ids),
            "schedule_id": schedule.schedule_id,
        },
        "target": {name: record["target"] for name, record in routes.items()},
        "runtime": {
            **_runtime_identity(),
            "provider_version": routes["advancing"]["provider"]["version"],
        },
        "quality": {
            "before": {
                name: record["quality"]["before"] for name, record in routes.items()
            },
            "after": {
                name: record["quality"]["after"] for name, record in routes.items()
            },
        },
        "conservation": {name: record["conservation"] for name, record in routes.items()},
        "resources": {name: record["resources"] for name, record in routes.items()},
        "routes": {
            name: {key: value for key, value in record.items() if key not in sections}
            for name, record in routes.items()
        },
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_gmsh_boundary_layer()
