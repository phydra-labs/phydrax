# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Any

import numpy as np

import phydrax as phx
from phydrax import model_array_bytes
from phydrax.meshing import meshcore_available
from tools._meshing_cases import original_wall_layer_schedule
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._records import _quality_summary
from tools.meshing_qualification._runtime import (
    _measured,
    _missing_dependency,
    _runtime_identity,
    qualification_failure,
)


def _layer_wall_mesh(points: np.ndarray, triangles: np.ndarray, /) -> Any:
    return phx.discretization.CellMesh(
        np.asarray(points, dtype=np.float64),
        (
            phx.discretization.CellBlock(
                "wall",
                "triangle",
                np.asarray(triangles, dtype=np.int64),
                global_ids=np.arange(triangles.shape[0], dtype=np.int64),
            ),
        ),
    )


def _layer_box_wall(half_width: float, count: int, /) -> Any:
    """Outward-oriented structured triangulation of a centered box surface."""
    grid = np.linspace(-half_width, half_width, count + 1)
    u, v = (value.ravel() for value in np.meshgrid(grid, grid, indexing="ij"))
    lower = (np.arange(count)[:, None] * (count + 1) + np.arange(count)).ravel()
    quads = np.stack((lower, lower + count + 1, lower + count + 2, lower + 1), axis=1)
    local = np.concatenate((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]))
    points = []
    triangles = []
    for axis in range(3):
        tangent = [value for value in range(3) if value != axis]
        for bound in (-half_width, half_width):
            face = np.empty((u.size, 3), dtype=np.float64)
            face[:, axis] = bound
            face[:, tangent[0]] = u
            face[:, tangent[1]] = v
            triangles.append(local + len(points) * u.size)
            points.append(face)
    coordinates, inverse = np.unique(
        np.round(np.concatenate(points), 12), axis=0, return_inverse=True
    )
    triangles = inverse.reshape(-1)[np.concatenate(triangles)]
    corners = coordinates[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = np.sum(normals * corners.mean(axis=1), axis=1) < 0.0
    triangles[inward] = triangles[inward][:, ::-1]
    return _layer_wall_mesh(coordinates, triangles)


def _layer_plate(count: int, height: float, /, *, flip: bool) -> Any:
    """Unit-square plate at ``height`` growing along +z (or -z when flipped)."""
    grid = np.linspace(0.0, 1.0, count + 1)
    x, y = (value.ravel() for value in np.meshgrid(grid, grid, indexing="ij"))
    points = np.stack((x, y, np.full(x.size, height)), axis=1)
    lower = (np.arange(count)[:, None] * (count + 1) + np.arange(count)).ravel()
    quads = np.stack((lower, lower + count + 1, lower + count + 2, lower + 1), axis=1)
    triangles = np.concatenate((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]))
    return points, triangles[:, ::-1] if flip else triangles


def _advancing_layer_control(wall: Any, schedule: Any, /, **options: Any) -> Any:
    cells = wall.entity_set(2)
    return phx.meshing.BoundaryLayerControl(
        phx.meshing.MeshingScope(
            wall.mesh_id,
            wall.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            2,
            cells.entity_set_id,
            cells.entity_ids,
        ),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        **options,
    )


def _layer_enclosed_volume(points: np.ndarray, triangles: np.ndarray, /) -> float:
    """Divergence-theorem volume enclosed by outward-oriented triangles."""
    corners = points[triangles]
    return float(np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2])) / 6.0)


def _layer_block_rows(mesh: Any, /) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )


def _layer_entity_rows(identifiers: Any, requested: Any, /) -> np.ndarray:
    """Rows of the unique ``identifiers`` holding every ``requested`` ID."""

    values = np.asarray(identifiers, dtype=np.int64)
    wanted = np.asarray(requested, dtype=np.int64)
    order = np.argsort(values, kind="stable")
    rows = order[np.searchsorted(values[order], wanted).clip(0, values.size - 1)]
    if not np.array_equal(values[rows], wanted):
        raise qualification_failure(
            "_layer_entity_rows",
            ("Requested entity IDs are absent from the mesh."),
            {"rows": rows, "values": values, "wanted": wanted},
        )
    return rows


def _layer_evidence(evidence: Any, /) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "requested_thicknesses": list(evidence.requested_thicknesses),
        "measured_thicknesses": np.asarray(evidence.achieved_thicknesses).tolist(),
        "minimum_thicknesses": np.asarray(evidence.minimum_thicknesses).tolist(),
        "maximum_thicknesses": np.asarray(evidence.maximum_thicknesses).tolist(),
        "measured_growth_rates": np.asarray(evidence.achieved_growth_rates).tolist(),
        "layer_active": np.asarray(evidence.layer_active).tolist(),
        "columns": evidence.column_count,
        "fan_columns": evidence.fan_column_count,
        "corner_patches": evidence.corner_patch_count,
        "convex_ridges": evidence.convex_ridge_count,
        "concave_ridges": evidence.concave_ridge_count,
        "rim_vertices": evidence.rim_vertex_count,
        "minimum_visibility": evidence.minimum_visibility,
        "maximum_stretch": evidence.maximum_stretch,
        "unconverged_visibility": evidence.unconverged_visibility_count,
        "collision_policy": evidence.collision_policy.value,
        "predicted_collision_vertices": evidence.predicted_collision_vertex_count,
        "detected_collisions": evidence.detected_collision_count,
        "resolution_iterations": evidence.resolution_iterations,
        "reduced_vertices": evidence.reduced_vertex_count,
        "terminated_vertices": evidence.terminated_vertex_count,
        "merged_vertices": evidence.merged_vertex_count,
        "minimum_scale": evidence.minimum_scale,
        "cell_counts": dict(evidence.cell_counts),
        "certified_valid_cells": evidence.certified_valid_count,
    }


def _layer_certificate_counts(certificate: Any, /) -> dict[str, object]:
    return {
        "certificate_id": certificate.certificate_id,
        "cells": certificate.status.shape[0],
        "certified_valid": certificate.certified_valid_count,
        "invalid": certificate.invalid_count,
        "unresolved": certificate.unresolved_count,
    }


def _layer_quality(mesh: Any, /) -> Any:
    return phx.meshing.summarize_cell_quality(phx.meshing.evaluate_cell_quality(mesh))


def qualify_boundary_layer() -> dict[str, object]:

    if not meshcore_available():
        return _missing_dependency(
            "meshcore",
            "Exact layer certification requires the native meshcore library; build "
            "it and set PHYDRAX_MESHCORE_LIBRARY to the shared library path.",
        )
    schedule = original_wall_layer_schedule()
    wall = _layer_box_wall(0.3, 2)
    control = _advancing_layer_control(
        wall,
        schedule,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
        corner=phx.meshing.BoundaryLayerCornerPolicy.FAN,
    )
    layers, prepare = _measured(
        lambda: phx.meshing.prepare_boundary_layers(wall, control)
    )
    before, wall_quality = _measured(lambda: _layer_quality(wall))

    after, layer_quality = _measured(lambda: _layer_quality(layers.mesh))
    certificate, certification = _measured(
        lambda: phx.discretization.certify_cell_geometry_validity(layers.mesh)
    )

    cap = layers.cap
    if cap is None:
        raise qualification_failure(
            "qualify_boundary_layer",
            ("Closed boundary-layer qualification did not publish its cap mesh."),
            {"cap": cap},
        )
    evidence = layers.evidence

    cells = sum(block.cell_count for block in layers.mesh.blocks)
    requested = np.asarray(schedule.thicknesses, dtype=np.float64)
    measured = np.stack(
        (
            np.asarray(evidence.achieved_thicknesses),
            np.asarray(evidence.minimum_thicknesses),
            np.asarray(evidence.maximum_thicknesses),
        )
    )
    thickness_error = float(np.max(np.abs(measured - requested) / requested))
    growth_error = float(np.max(np.abs(np.asarray(evidence.achieved_growth_rates) - 1.2)))

    layer_volume = float(np.sum(np.asarray(after.evaluation.measures)))
    enclosed = _layer_enclosed_volume(
        np.asarray(cap.coordinates),
        _layer_block_rows(cap),
    ) - _layer_enclosed_volume(np.asarray(wall.coordinates), _layer_block_rows(wall))
    volume_residual = abs(layer_volume - enclosed) / enclosed
    if (
        not layers.closed_cap
        or layers.validity.certified_valid_count != cells
        or certificate.certified_valid_count != cells
        or evidence.convex_ridge_count != 24
        or evidence.corner_patch_count != 8
        or evidence.detected_collision_count != 0
        or thickness_error > 1.0e-9
        or growth_error > 1.0e-9
        or after.sampled_invalid_count != 0
        or volume_residual > 1.0e-10
    ):
        raise qualification_failure(
            "qualify_boundary_layer",
            ("Native layers missed the schedule, certification, or volume closure."),
            {
                "after": after,
                "cells": cells,
                "certificate": certificate,
                "evidence": evidence,
                "growth_error": growth_error,
                "layers": layers,
                "thickness_error": thickness_error,
                "volume_residual": volume_residual,
            },
        )

    gap = 0.05
    lower, lower_triangles = _layer_plate(4, 0.0, flip=False)
    upper, upper_triangles = _layer_plate(4, gap, flip=True)
    channel = _layer_wall_mesh(
        np.concatenate((lower, upper)),
        np.concatenate((lower_triangles, upper_triangles + lower.shape[0])),
    )
    reduction = _advancing_layer_control(
        channel,
        schedule,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS,
        corner=phx.meshing.BoundaryLayerCornerPolicy.FAN,
        minimum_thickness_fraction=0.2,
    )
    reduced, reduce = _measured(
        lambda: phx.meshing.prepare_boundary_layers(channel, reduction)
    )
    reduced_certificate, reduced_certification = _measured(
        lambda: phx.discretization.certify_cell_geometry_validity(reduced.mesh)
    )
    # The cap faces away from its layers: lower-front faces point along +z.

    reduced_cap = reduced.cap
    if reduced_cap is None:
        raise qualification_failure(
            "qualify_boundary_layer",
            ("Reduced boundary layers lost their cap."),
            {"reduced_cap": reduced_cap},
        )
    cap_points = np.asarray(reduced_cap.coordinates)

    cap_corners = cap_points[_layer_block_rows(reduced_cap)]
    upward = (
        np.cross(
            cap_corners[:, 1] - cap_corners[:, 0], cap_corners[:, 2] - cap_corners[:, 0]
        )[:, 2]
        > 0.0
    )
    clearance = float(
        np.min(cap_corners[~upward][..., 2]) - np.max(cap_corners[upward][..., 2])
    )

    reduced_evidence = reduced.evidence

    reduced_cells = sum(block.cell_count for block in reduced.mesh.blocks)
    if (
        reduced_evidence.predicted_collision_vertex_count != 50
        or reduced_evidence.reduced_vertex_count != 50
        or not 0.2 <= reduced_evidence.minimum_scale < 1.0
        or clearance <= 0.0
        or np.max(np.abs(np.asarray(reduced_evidence.achieved_growth_rates) - 1.2))
        > 1.0e-9
        or reduced_certificate.certified_valid_count != reduced_cells
    ):
        raise qualification_failure(
            "qualify_boundary_layer",
            ("REDUCE_THICKNESS did not resolve the narrow-gap collision."),
            {
                "clearance": clearance,
                "reduced_cells": reduced_cells,
                "reduced_certificate": reduced_certificate,
                "reduced_evidence": reduced_evidence,
            },
        )
    return {
        "status": "passed",
        "source": {
            "wall_mesh_id": wall.mesh_id,
            "topology_id": wall.topology_id,
            "vertices": wall.coordinates.shape[0],
            "cells": wall.blocks[0].cell_count,
            "control_id": control.control_id,
            "schedule_id": schedule.schedule_id,
        },
        "target": {
            "result_id": layers.result_id,
            "mesh_id": layers.mesh.mesh_id,
            "topology_id": layers.mesh.topology_id,
            "vertices": layers.mesh.coordinates.shape[0],
            "cells": cells,
            "cap_mesh_id": cap.mesh_id,
            "closed_cap": layers.closed_cap,
        },
        "runtime": _runtime_identity(),
        "quality": {
            "before": _quality_summary(before),
            "after": _quality_summary(after),
        },
        "conservation": {
            "layer_cell_volume": layer_volume,
            "cap_minus_wall_enclosed_volume": enclosed,
            "relative_volume_residual": volume_residual,
        },
        "certificates": {
            "layer_generation": _layer_certificate_counts(layers.validity),
            "independent": _layer_certificate_counts(certificate),
        },
        "boundary_layer": {
            **_layer_evidence(evidence),
            "maximum_relative_thickness_error": thickness_error,
            "maximum_growth_rate_error": growth_error,
        },
        "collision": {
            "gap": gap,
            "wall_faces": channel.blocks[0].cell_count,
            "result_id": reduced.result_id,
            "front_clearance": clearance,
            "certificate": _layer_certificate_counts(reduced_certificate),
            **_layer_evidence(reduced_evidence),
        },
        "resources": {
            "prepare_boundary_layers": prepare,
            "wall_quality": wall_quality,
            "layer_quality": layer_quality,
            "independent_certification": certification,
            "collision_prepare_boundary_layers": reduce,
            "collision_certification": reduced_certification,
            "retained_boundary_layer_bytes": model_array_bytes(layers),
        },
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_boundary_layer()
