#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Grow certified boundary layers around a box, then fill the core with Gmsh.

The wall is the closed, outward-oriented triangle surface of a small box, so
every edge of the box is a convex ridge and every box corner meets three
ridges. `prepare_boundary_layers` grows the native ADVANCING layers of an
explicit geometric `LayerSchedule` with an explicit collision policy
(`REDUCE_THICKNESS`) and corner policy (`FAN`): columns split at the convex
ridges into fan columns, the corners close with corner patches, and every
layer cell is certified by Bernstein validity and exact-predicate intersection
tests. The measured layer thicknesses are exact distances to the wall.

`GmshProvider.fill_boundary_layer_core` then tetrahedralizes the core between
the closed layer cap and an outer box while keeping both surfaces fixed
bitwise. The merged mesh carries `boundary-layer` and `core` zones and
`wall`, `layer-core-interface`, and `outer` patches; it is audited and its
geometry is certified independently with `certify_cell_geometry_validity`.

Run with the optional Gmsh dependency installed (`pip install
'phydrax[meshing-gmsh]'`) and the native meshcore library available:

    python examples/boundary_layer_core_mesh.py
"""

import json

import numpy as np

import phydrax as phx


def box_surface(half_width: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Outward-oriented structured triangulation of a centered box surface."""
    grid = np.linspace(-half_width, half_width, count + 1)
    u, v = (value.ravel() for value in np.meshgrid(grid, grid, indexing="ij"))
    index = np.arange((count + 1) ** 2).reshape((count + 1, count + 1))
    lower = index[:-1, :-1].ravel()
    quads = np.stack(
        (lower, lower + count + 1, lower + count + 2, lower + 1), axis=1
    ).astype(np.int64)
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
    return coordinates, triangles


points, triangles = box_surface(0.3, 2)
wall = phx.discretization.CellMesh(
    points,
    (
        phx.discretization.CellBlock(
            "wall",
            "triangle",
            triangles,
            global_ids=np.arange(triangles.shape[0], dtype=np.int64),
        ),
    ),
)
wall_cells = wall.entity_set(2)
schedule = phx.meshing.LayerSchedule.geometric(3, 0.01, growth_rate=1.2)
control = phx.meshing.BoundaryLayerControl(
    phx.meshing.MeshingScope(
        wall.mesh_id,
        wall.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        wall_cells.entity_set_id,
        wall_cells.entity_ids,
    ),
    schedule,
    route=phx.meshing.BoundaryLayerRoute.ADVANCING,
    collision=phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS,
    corner=phx.meshing.BoundaryLayerCornerPolicy.FAN,
    minimum_thickness_fraction=0.5,
)
layers = phx.meshing.prepare_boundary_layers(wall, control)
evidence = layers.evidence
layer_cells = sum(block.cell_count for block in layers.mesh.blocks)
requested = np.asarray(schedule.thicknesses, dtype=np.float64)
thickness_error = float(
    np.max(np.abs(np.asarray(evidence.achieved_thicknesses) - requested) / requested)
)
if (
    not layers.closed_cap
    or layers.validity.certified_valid_count != layer_cells
    or evidence.convex_ridge_count == 0
    or evidence.corner_patch_count != 8
    or evidence.reduced_vertex_count != 0
    or thickness_error > 1e-9
):
    raise RuntimeError("Native boundary layers did not realize the certified schedule.")

outer_points = np.stack(
    np.meshgrid((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), indexing="ij"), axis=-1
).reshape((-1, 3))
outer = phx.geometry.surface.SurfaceModel.from_triangles(
    outer_points,
    np.asarray(
        (
            (0, 1, 3),
            (0, 3, 2),
            (4, 6, 7),
            (4, 7, 5),
            (0, 4, 5),
            (0, 5, 1),
            (2, 3, 7),
            (2, 7, 6),
            (0, 2, 6),
            (0, 6, 4),
            (1, 5, 7),
            (1, 7, 3),
        ),
        dtype=np.int64,
    ),
    phx.geometry.surface.SurfaceMetadata(
        source_id="outer-box",
        source_revision="0",
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
        provenance=("examples/boundary_layer_core_mesh.py",),
    ),
    repair_orientation=True,
)
result = phx.meshing.GmshProvider().fill_boundary_layer_core(
    layers, outer, maximum_size=0.5
)
certificate = phx.discretization.certify_cell_geometry_validity(result.mesh)
cells = sum(block.cell_count for block in result.mesh.blocks)
zones = {zone.name: zone.scope.entity_ids.shape[0] for zone in result.zones}
patches = {patch.name: patch.scope.entity_ids.shape[0] for patch in result.patches}
layer_points = np.asarray(layers.mesh.coordinates)
fixed_bitwise = np.array_equal(
    np.asarray(result.mesh.coordinates)[: layer_points.shape[0]], layer_points
)
if (
    not result.audit.passed
    or not result.compliance.passed
    or set(zones) != {"boundary-layer", "core"}
    or zones["boundary-layer"] != layer_cells
    or set(patches) != {"wall", "layer-core-interface", "outer"}
    or patches["wall"] != triangles.shape[0]
    or not fixed_bitwise
    or certificate.certified_valid_count != cells
    or result.quality.minimum_scaled_jacobian <= 0.0
):
    raise RuntimeError("The merged boundary-layer/core mesh failed audit or validity.")
print(
    json.dumps(
        {
            "wall_triangles": triangles.shape[0],
            "layers": {
                "result_id": layers.result_id,
                "requested_thicknesses": list(schedule.thicknesses),
                "measured_thicknesses": np.asarray(
                    evidence.achieved_thicknesses
                ).tolist(),
                "minimum_thicknesses": np.asarray(evidence.minimum_thicknesses).tolist(),
                "maximum_thicknesses": np.asarray(evidence.maximum_thicknesses).tolist(),
                "measured_growth_rates": np.asarray(
                    evidence.achieved_growth_rates
                ).tolist(),
                "layer_active": np.asarray(evidence.layer_active).tolist(),
                "maximum_relative_thickness_error": thickness_error,
                "columns": evidence.column_count,
                "fan_columns": evidence.fan_column_count,
                "corner_patches": evidence.corner_patch_count,
                "convex_ridges": evidence.convex_ridge_count,
                "collision_policy": evidence.collision_policy.value,
                "detected_collisions": evidence.detected_collision_count,
                "cell_counts": dict(evidence.cell_counts),
                "certified_valid_cells": layers.validity.certified_valid_count,
            },
            "merged": {
                "result_id": result.result_id,
                "provider_version": result.runtime.actual_version,
                "cell_counts": {
                    block.cell_kind: block.cell_count for block in result.mesh.blocks
                },
                "zone_cells": zones,
                "patch_faces": patches,
                "fixed_boundary_bitwise": fixed_bitwise,
                "minimum_scaled_jacobian": result.quality.minimum_scaled_jacobian,
                "minimum_mean_ratio": result.quality.minimum_mean_ratio,
                "certified_valid_cells": certificate.certified_valid_count,
                "invalid_cells": certificate.invalid_count,
                "unresolved_cells": certificate.unresolved_count,
                "audit_passed": result.audit.passed,
            },
        },
        indent=2,
    )
)
