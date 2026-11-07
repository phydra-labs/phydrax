#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Grow certified boundary layers around a box, then fill the core natively.

The wall is the closed, outward-oriented triangle surface of a small box, so
every edge of the box is a convex ridge and every box corner meets three
ridges. `prepare_boundary_layers` grows the native ADVANCING layers of an
explicit geometric `LayerSchedule` with an explicit collision policy
(`REDUCE_THICKNESS`) and corner policy (`FAN`): columns split at the convex
ridges into fan columns, the corners close with corner patches, and every
layer cell is certified by Bernstein validity and exact-predicate intersection
tests. The measured layer thicknesses are exact distances to the wall.

The native layer/core source binds the closed cap and the outer box by exact
source identity. The requested core size 0.5 is a mesher target, not a hard
edge-length ceiling: the fixed outer triangles already have longer edges.
The original comparison did not prescribe radius-edge or dihedral bounds, so
the native schedule explicitly omits those shape aims. The result reports
measured quality, retains fixed surfaces bitwise, and is independently audited
and certified. All coordinates and sizes use the original millimeter contract.

Run with the native meshcore library available:

    python examples/boundary_layer_core_mesh.py
"""

import json

import numpy as np

import phydrax as phx
from phydrax.geometry._source_identity import source_entity_id
from phydrax.meshing.providers import NativeLayerCoreSource


def box_surface(half_width: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Outward-oriented structured triangulation of a centered box surface."""
    grid = np.linspace(-half_width, half_width, count + 1)
    vertex_rows: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for axis in range(3):
        tangent = [value for value in range(3) if value != axis]
        for side in (0, count):
            face = np.empty((count + 1, count + 1), dtype=np.int64)
            for u in range(count + 1):
                for v in range(count + 1):
                    lattice = [0, 0, 0]
                    lattice[axis] = side
                    lattice[tangent[0]] = u
                    lattice[tangent[1]] = v
                    key = (lattice[0], lattice[1], lattice[2])
                    if key not in vertex_rows:
                        vertex_rows[key] = len(points)
                        points.append(
                            (
                                float(grid[key[0]]),
                                float(grid[key[1]]),
                                float(grid[key[2]]),
                            )
                        )
                    face[u, v] = vertex_rows[key]
            for u in range(count):
                for v in range(count):
                    a, b = face[u, v], face[u + 1, v]
                    c, d = face[u + 1, v + 1], face[u, v + 1]
                    triangles.extend(((a, b, c), (a, c, d)))
    coordinates = np.asarray(points, dtype=np.float64)
    cells = np.asarray(triangles, dtype=np.int64)
    corners = coordinates[cells]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = np.sum(normals * corners.mean(axis=1), axis=1) < 0.0
    cells[inward] = cells[inward][:, ::-1]
    return coordinates, cells


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
policy = phx.meshing.BoundaryLayerPolicy()
layers = phx.meshing.prepare_boundary_layers(wall, control, policy=policy)
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
cap = layers.cap
if cap is None:
    raise RuntimeError("The certified boundary layers have no core cap.")

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
contract = phx.SpatialCoordinateContract(phx.units.MILLIMETER)
source = NativeLayerCoreSource.fixed_core(
    layers,
    outer,
    contract,
    "example:box-boundary-layer-core",
    "0",
    region_id="fluid",
    region_entity_id=0,
    region_occurrence_path=(),
    wall_occurrence_path=("wall",),
    outer_occurrence_path=("outer",),
    layer_control=control,
    layer_policy=policy,
)
scope = phx.meshing.MeshingScope.whole(source, 2)
layer_kinds = tuple(
    sorted({block.cell_kind for block in layers.mesh.blocks} - {"tetrahedron"})
)
specification = phx.meshing.VolumeMeshingSpec(
    phx.meshing.CellMeshingTarget(
        3,
        3,
        phx.meshing.CellFamilyPolicy(
            required=("tetrahedron",),
            allowed_transitions=layer_kinds,
            allow_mixed=True,
        ),
    ),
    scope,
    phx.meshing.VolumeFillStrategy.SIMPLEX,
    size_controls=(
        phx.meshing.UniformSizeControl(
            scope, 0.5, strength=phx.meshing.SizeControlStrength.SOFT
        ),
    ),
    maximum_deviation=0.0,
)
volume_schedule = phx.meshing.NativeVolumeSchedule(
    radius_edge_bound=None, minimum_dihedral_degrees=0.0
)
result = (
    phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("layer_core", volume_schedule=volume_schedule)
    )
    .plan(source, specification, coordinate_contract=contract)
    .execute()
)
certificate = phx.discretization.certify_cell_geometry_validity(result.mesh)
cells = sum(block.cell_count for block in result.mesh.blocks)
zones = {zone.name: zone.scope.entity_ids.shape[0] for zone in result.zones}
labels = {label.name: label.scope.entity_ids.shape[0] for label in result.labels}
patches = {patch.name: patch.scope.entity_ids.shape[0] for patch in result.patches}
layer_points = np.asarray(layers.mesh.coordinates)
core_points = np.asarray(source.complex.vertices)[np.asarray(source.vertex_layer_ids) < 0]
outer_bitwise = np.array_equal(
    core_points.view(np.uint64),
    np.asarray(outer.mesh.coordinates, dtype=np.float64).view(np.uint64),
)
expected_fixed = np.concatenate((layer_points, core_points))
fixed_bitwise = np.array_equal(
    np.asarray(result.mesh.coordinates)[: expected_fixed.shape[0]].view(np.uint64),
    expected_fixed.view(np.uint64),
)
cap_count = sum(block.cell_count for block in cap.blocks)
cap_keys = {
    (
        source_entity_id(
            source.source_id,
            source.source_revision,
            "face",
            source.complex.facet_ids[int(facet)],
            occurrence_path=source.complex.facet_occurrence_paths[int(facet)],
        ),
        source.complex.facet_occurrence_paths[int(facet)],
    )
    for facet in np.asarray(source.complex.polygon_facets, dtype=np.int64)[
        np.asarray(source.cap_polygon_ids, dtype=np.int64)
    ]
}
domain = source.source_domain
if not isinstance(domain, phx.geometry.MeshingDomain):
    raise RuntimeError("The native layer source lost its material authority.")
fluid_id = domain.entity_id(3, 0)
cap_rows = {
    (int(target), entity, path, sides)
    for association in result.associations
    if association.source_id == source.source_id
    and association.target_entity_set_id == result.mesh.entity_set(2).entity_set_id
    and association.exact
    for target, entity, path, sides in zip(
        np.asarray(association.target_global_ids, dtype=np.int64),
        association.source_entity_ids,
        association.source_occurrence_paths,
        association.source_region_sides,
        strict=True,
    )
    if (entity, path) in cap_keys
}
cap_targets = {target for target, _, _, _ in cap_rows}
cap_hits = {(entity, path) for _, entity, path, _ in cap_rows}
interface_targets = {
    int(identifier)
    for patch in result.patches
    if patch.name == "layer-core-interface"
    for identifier in np.asarray(patch.scope.entity_ids, dtype=np.int64)
}
outer_keys = set(
    zip(
        source.core_facet_source_ids[cap_count:],
        source.core_facet_source_occurrence_paths[cap_count:],
        strict=True,
    )
)
outer_rows = {
    (int(target), entity, path)
    for association in result.associations
    if association.source_id == source.source_id
    and association.target_entity_set_id == result.mesh.entity_set(2).entity_set_id
    and association.exact
    for target, entity, path in zip(
        np.asarray(association.target_global_ids, dtype=np.int64),
        association.source_entity_ids,
        association.source_occurrence_paths,
        strict=True,
    )
    if (entity, path) in outer_keys
}
outer_targets = {target for target, _, _ in outer_rows}
outer_hits = {(entity, path) for _, entity, path in outer_rows}
outer_face_count = sum(block.cell_count for block in outer.mesh.blocks)
acceptance = result.certification
achieved_core_maximum = dict(result.compliance.achieved)["core_maximum_edge_length"]
if (
    not result.audit.passed
    or not result.compliance.passed
    or acceptance is None
    or not acceptance.passed
    or labels.get("boundary-layer") != layer_cells
    or labels.get("core", 0) <= 0
    or patches.get("wall") != triangles.shape[0]
    or patches.get("layer-core-interface") != cap_count
    or len(cap_keys) != cap_count
    or cap_hits != cap_keys
    or cap_targets != interface_targets
    or any(sides != (fluid_id, fluid_id) for _, _, _, sides in cap_rows)
    or len(outer_keys) != outer_face_count
    or outer_hits != outer_keys
    or len(outer_targets) != outer_face_count
    or not outer_bitwise
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
                "label_cells": labels,
                "patch_faces": patches,
                "core_size_target": 0.5,
                "achieved_core_maximum_edge_length": achieved_core_maximum,
                "fixed_boundary_bitwise": fixed_bitwise,
                "outer_source_vertices_bitwise": outer_bitwise,
                "outer_source_faces_preserved": len(outer_targets),
                "cap_source_faces_preserved": len(cap_targets),
                "cap_material_sides": [fluid_id, fluid_id],
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
