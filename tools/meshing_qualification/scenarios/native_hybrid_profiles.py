from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._periodic_exact import identification_id
from phydrax.geometry.brep import BRepTessellationPolicy
from phydrax.meshing._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._boundary_layer import BoundaryLayerMesh
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import NativeLayerCoreSource


_WORKFLOW = "hybrid-layers"
_PREFIX = "native-hybrid-lifecycle:"
_LAYER_TOLERANCE = 1.0e-9
_VOLUME_TOLERANCE = 1.0e-10
_GROWTH_RATE = 1.2


class _ProfileFailure(RuntimeError):
    def __init__(
        self, stage: str, message: str, evidence: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.evidence = {} if evidence is None else evidence


def _migration_profiles() -> tuple[str, ...]:
    migration = Path(__file__).resolve().parents[1] / "native_case_migration.json"
    record = json.loads(migration.read_text(encoding="utf-8"))
    for workflow in record["workflow_contracts"]:
        if workflow["workflow"] == _WORKFLOW:
            profiles = tuple(str(value) for value in workflow["profiles"])
            if len(profiles) != 25 or len(set(profiles)) != 25:
                raise ValueError(
                    "The recovered hybrid-layer inventory must contain 25 unique profiles."
                )
            return profiles
    raise KeyError(f"{_WORKFLOW!r} is absent from the recovered case migration.")


def _profile_name(profile_id: str) -> str:
    if not profile_id.startswith(_PREFIX):
        raise ValueError(f"Unexpected hybrid profile identity {profile_id!r}.")
    return profile_id[len(_PREFIX) :]


def _wall(points: np.ndarray, triangles: np.ndarray, /, *, name: str = "wall") -> Any:
    return phx.discretization.CellMesh(
        np.asarray(points, dtype=np.float64),
        (
            phx.discretization.CellBlock(
                name,
                "triangle",
                np.asarray(triangles, dtype=np.int64),
                global_ids=np.arange(len(triangles), dtype=np.int64),
            ),
        ),
    )


def _whole_wall_scope(mesh: Any) -> Any:
    cells = mesh.entity_set(2)
    return phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        cells.entity_ids,
    )


def _control(mesh: Any, schedule: Any, /, **options: Any) -> Any:
    return phx.meshing.BoundaryLayerControl(
        _whole_wall_scope(mesh),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        **options,
    )


def _plate(
    count: int, height: float = 0.0, /, *, flip: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    values = np.linspace(0.0, 1.0, count + 1)
    x, y = np.meshgrid(values, values, indexing="ij")
    points = np.stack((x.ravel(), y.ravel(), np.full(x.size, height)), axis=1)
    index = np.arange(points.shape[0]).reshape(count + 1, count + 1)
    triangles: list[tuple[int, int, int]] = []
    for i in range(count):
        for j in range(count):
            a, b, c, d = (
                int(index[i, j]),
                int(index[i + 1, j]),
                int(index[i + 1, j + 1]),
                int(index[i, j + 1]),
            )
            triangles.extend(((a, b, c), (a, c, d)))
    rows = np.asarray(triangles, dtype=np.int64)
    return points, rows[:, ::-1] if flip else rows


def _orient_closed_surface(
    points: np.ndarray,
    triangles: np.ndarray,
    center: np.ndarray,
    /,
    *,
    outward: bool,
) -> np.ndarray:
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    away = np.sum(normals * (corners.mean(axis=1) - center), axis=1) > 0.0
    flip = away != outward
    rows = triangles.copy()
    rows[flip] = rows[flip][:, ::-1]
    return rows


def _box(
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    count: int,
    /,
    *,
    outward: bool,
) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, count + 1)
    points: list[tuple[float, float, float]] = []
    point_rows: dict[tuple[float, float, float], int] = {}
    triangles: list[tuple[int, int, int]] = []

    def vertex(coordinate: np.ndarray) -> int:
        key: tuple[float, float, float] = (
            float(coordinate[0]),
            float(coordinate[1]),
            float(coordinate[2]),
        )
        row = point_rows.get(key)
        if row is None:
            row = len(points)
            point_rows[key] = row
            points.append(key)
        return row

    for axis in range(3):
        tangent = tuple(value for value in range(3) if value != axis)
        for bound in (low[axis], high[axis]):
            for i in range(count):
                for j in range(count):
                    quad: list[int] = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        coordinate = np.zeros(3, dtype=np.float64)
                        coordinate[axis] = bound
                        coordinate[tangent[0]] = (
                            low[tangent[0]]
                            + (high[tangent[0]] - low[tangent[0]]) * grid[i + di]
                        )
                        coordinate[tangent[1]] = (
                            low[tangent[1]]
                            + (high[tangent[1]] - low[tangent[1]]) * grid[j + dj]
                        )
                        quad.append(vertex(coordinate))
                    a, b, c, d = quad
                    triangles.extend(((a, b, c), (a, c, d)))
    coordinates = np.asarray(points, dtype=np.float64)
    rows = _orient_closed_surface(
        coordinates,
        np.asarray(triangles, dtype=np.int64),
        0.5 * (low + high),
        outward=outward,
    )
    return coordinates, rows


def _icosphere(radius: float, subdivisions: int, /) -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + 5.0**0.5) / 2.0
    points = [
        np.asarray(value, dtype=np.float64)
        for value in (
            (-1, t, 0),
            (1, t, 0),
            (-1, -t, 0),
            (1, -t, 0),
            (0, -1, t),
            (0, 1, t),
            (0, -1, -t),
            (0, 1, -t),
            (t, 0, -1),
            (t, 0, 1),
            (-t, 0, -1),
            (-t, 0, 1),
        )
    ]
    points = [value / np.linalg.norm(value) for value in points]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    for _ in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoints:
                point = points[a] + points[b]
                points.append(point / np.linalg.norm(point))
                midpoints[key] = len(points) - 1
            return midpoints[key]

        refined: list[tuple[int, int, int]] = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = refined
    coordinates = radius * np.asarray(points, dtype=np.float64)
    return coordinates, _orient_closed_surface(
        coordinates, np.asarray(faces, dtype=np.int64), np.zeros(3), outward=True
    )


def _periodic_wall(
    periodicity: str, /, *, quadrilateral: bool
) -> tuple[Any, dict[str, object]]:
    if periodicity == "rotation":
        points = np.asarray(
            ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 0.0)),
            dtype=np.float64,
        )
        generators = np.asarray(
            (
                (
                    (0.0, -1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            ),
            dtype=np.float64,
        )
        representatives = np.asarray((0, 1, 1, 0), dtype=np.int64)
        shifts = np.asarray(((0,), (0,), (1,), (1,)), dtype=np.int64)
        area = 0.375
    else:
        points = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float64,
        )
        generators = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        generators[0, 0, 3] = 1.0
        generators[1, 1, 3] = 1.0
        representatives = np.zeros(4, dtype=np.int64)
        shifts = points[:, :2].astype(np.int64)
        area = 1.0
        if periodicity == "partial":
            generators = generators[:1]
            representatives = np.asarray((0, 0, 3, 3), dtype=np.int64)
            shifts = points[:, :1].astype(np.int64)
    rows = np.asarray(
        ((0, 1, 2, 3),) if quadrilateral else ((0, 1, 2), (0, 2, 3)),
        dtype=np.int64,
    )
    kind = "quadrilateral" if quadrilateral else "triangle"
    block = phx.discretization.CellBlock(
        "wall", kind, rows, global_ids=np.arange(len(rows))
    )
    lifted = phx.discretization.CellMesh(points, (block,))
    group = phx.discretization.PeriodicIsometryGroup(generators)
    topology = phx.discretization.PeriodicMeshTopology(
        lifted, group, representatives, shifts
    )
    wall = phx.discretization.CellMesh(points, (block,), periodic_topology=topology)
    transition_height = (
        0.5 * np.mean(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1))
        if quadrilateral
        else 0.0
    )
    return wall, {
        "periodicity": periodicity,
        "quadrilateral": quadrilateral,
        "expected_area": area,
        "transition_height": transition_height,
        "periodic_topology_id": topology.periodic_topology_id,
        "group_id": group.group_id,
    }


def _ridge_wall() -> tuple[Any, dict[str, object]]:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.5, 0.1),
            (1.0, 0.5, 0.1),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    rows = np.asarray(((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)), dtype=np.int64)
    block = phx.discretization.CellBlock(
        "wall", "triangle", rows, global_ids=np.arange(4)
    )
    lifted = phx.discretization.CellMesh(points, (block,))
    generator = np.eye(4, dtype=np.float64)[None]
    generator[0, 0, 3] = 1.0
    topology = phx.discretization.PeriodicMeshTopology(
        lifted,
        phx.discretization.PeriodicIsometryGroup(generator),
        np.asarray((0, 0, 2, 2, 4, 4), dtype=np.int64),
        np.asarray(((0,), (1,), (0,), (1,), (0,), (1,)), dtype=np.int64),
    )
    wall = phx.discretization.CellMesh(points, (block,), periodic_topology=topology)
    return wall, {
        "periodicity": "translation-x-with-convex-ridge",
        "periodic_topology_id": topology.periodic_topology_id,
        "group_id": identification_id(topology.cell),
    }


def _curved_periodic_wall() -> tuple[Any, dict[str, object]]:
    # The archived/corrected source owns a quadratic profile map; its wall carrier
    # consists of the four exact seam representatives used by that source.
    representatives = np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    period = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    points = np.concatenate((representatives, representatives + period))
    rows = np.asarray(((0, 2, 3), (0, 3, 1)), dtype=np.int64)
    block = phx.discretization.CellBlock(
        "fundamental-profile-strips", "triangle", rows, global_ids=np.asarray((100, 101))
    )
    lifted = phx.discretization.CellMesh(points, (block,), vertex_global_ids=np.arange(4))
    topology = phx.discretization.PeriodicMeshTopology(
        lifted,
        phx.discretization.PeriodicIsometryGroup(
            np.asarray(
                (
                    (
                        (1.0, 0.0, 0.0, 1.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (0.0, 0.0, 0.0, 1.0),
                    ),
                ),
                dtype=np.float64,
            )
        ),
        np.asarray((0, 1, 0, 1), dtype=np.int64),
        np.asarray(((0,), (0,), (1,), (1,)), dtype=np.int64),
    )
    wall = phx.discretization.CellMesh(
        points, (block,), vertex_global_ids=np.arange(4), periodic_topology=topology
    )
    return wall, {
        "period": period.tolist(),
        "profile_control_points": [[0.0, 0.0], [0.5, 0.1875], [1.0, 0.0]],
        "periodic_topology_id": topology.periodic_topology_id,
    }


def _count_cells(mesh: Any) -> int:
    return sum(int(block.cell_count) for block in mesh.blocks)


def _block_rows(mesh: Any) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks], axis=0
    )


def _boundary_volume(mesh: Any) -> float:
    outward_faces = {
        "tetrahedron": ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
        "pyramid": ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
        "prism": ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
        "hexahedron": (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ),
    }
    faces: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for block in mesh.blocks:
        if block.cell_kind not in outward_faces:
            raise ValueError(f"Cannot close volume for cell family {block.cell_kind!r}.")
        for raw_row in np.asarray(block.vertices, dtype=np.int64):
            row = tuple(int(value) for value in raw_row)
            for local_face in outward_faces[block.cell_kind]:
                oriented = tuple(row[index] for index in local_face)
                faces.setdefault(tuple(sorted(oriented)), []).append(oriented)
    total = 0.0
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    for incidents in faces.values():
        if len(incidents) == 1:
            face = incidents[0]
            for index in range(1, len(face) - 1):
                a, b, c = points[[face[0], face[index], face[index + 1]]]
                total += float(np.dot(a, np.cross(b, c)) / 6.0)
        elif len(incidents) != 2:
            raise ValueError("Layer volume boundary has non-manifold face incidence.")
    return total


def _layer_record(
    wall: Any,
    control: Any,
    result: BoundaryLayerMesh,
    /,
    *,
    expected_growth: float | None = _GROWTH_RATE,
    expected_all_thicknesses: bool = True,
    closed_cap: bool | None = None,
    source_face_ids: Any | None = None,
) -> dict[str, object]:
    evidence = result.evidence
    cells = _count_cells(result.mesh)
    independent = phx.discretization.certify_cell_geometry_validity(result.mesh)
    quality = phx.meshing.summarize_cell_quality(
        phx.meshing.evaluate_cell_quality(result.mesh)
    )
    if (
        result.validity.certified_valid_count != cells
        or independent.certified_valid_count != cells
    ):
        raise RuntimeError("A source wall produced an uncertified or invalid layer cell.")
    if (
        independent.invalid_count
        or independent.unresolved_count
        or quality.sampled_invalid_count
    ):
        raise RuntimeError(
            "Independent layer validity found an invalid or unresolved cell."
        )
    if evidence.collision_policy is not control.collision:
        raise RuntimeError("The layer evidence changed the authored collision policy.")
    requested = np.asarray(control.schedule.thicknesses, dtype=np.float64)
    achieved = np.asarray(evidence.achieved_thicknesses, dtype=np.float64)
    active = np.asarray(evidence.layer_active, dtype=np.bool_)
    if achieved.shape != requested.shape or active.shape != requested.shape:
        raise RuntimeError(
            "The layer producer returned malformed thickness or activity evidence."
        )
    if expected_all_thicknesses:
        measured = np.stack(
            (
                achieved,
                np.asarray(evidence.minimum_thicknesses, dtype=np.float64),
                np.asarray(evidence.maximum_thicknesses, dtype=np.float64),
            )
        )
        thickness_error = float(np.max(np.abs(measured - requested) / requested))
        if thickness_error > _LAYER_TOLERANCE:
            raise RuntimeError(
                f"Physical layer schedule residual {thickness_error:.3e} exceeds tolerance."
            )
    else:
        valid_rows = active & np.isfinite(achieved)
        thickness_error = float(
            np.max(
                np.abs(achieved[valid_rows] - requested[valid_rows])
                / requested[valid_rows],
                initial=0.0,
            )
        )
        first = float(achieved[0])
        if not active[0] or not np.isfinite(first):
            raise RuntimeError("The profile lost its physical first-layer thickness.")
        if control.collision in (
            phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
            phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY,
        ):
            first_error = abs(first - requested[0]) / requested[0]
            if first_error > _LAYER_TOLERANCE:
                raise RuntimeError(
                    "The profile's physical first-layer thickness differs from its authored value."
                )
        elif (
            control.collision is phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS
        ):
            lower = requested[0] * control.minimum_thickness_fraction
            if first < lower * (1.0 - _LAYER_TOLERANCE) or first > requested[0] * (
                1.0 + _LAYER_TOLERANCE
            ):
                raise RuntimeError(
                    "REDUCE_THICKNESS produced a first layer outside its physical thickness bounds."
                )
    rates = np.asarray(evidence.achieved_growth_rates, dtype=np.float64)
    expected_rates = np.asarray(control.schedule.growth_rates, dtype=np.float64)
    if expected_growth is not None and rates.size:
        finite_rates = rates[np.isfinite(rates)]
        if (
            finite_rates.size
            and np.max(np.abs(finite_rates - expected_growth)) > _LAYER_TOLERANCE
        ):
            raise RuntimeError(
                "Measured growth rates do not match the profile's physical schedule."
            )
        if (
            expected_rates.size
            and np.max(np.abs(expected_rates - expected_growth)) > _LAYER_TOLERANCE
        ):
            raise RuntimeError(
                "The authored profile schedule differs from its required growth rate."
            )
    source_faces = (
        {
            int(value)
            for value in np.asarray(
                control.wall_scope.entity_ids, dtype=np.int64
            ).tolist()
        }
        if source_face_ids is None
        else {
            int(value) for value in np.asarray(source_face_ids, dtype=np.int64).tolist()
        }
    )
    front_sources = {
        int(value)
        for value in np.asarray(
            result.front_row_source_global_ids, dtype=np.int64
        ).tolist()
        if int(value) >= 0
    }
    source_covered = source_faces <= front_sources
    if (
        control.collision is phx.meshing.BoundaryLayerCollisionPolicy.FAIL
        and not source_covered
    ):
        raise RuntimeError("The accepted wall front lost source-face ancestry.")
    cap_cells = 0 if result.cap is None else _count_cells(result.cap)
    if closed_cap is not None and result.closed_cap is not closed_cap:
        raise RuntimeError(
            "The boundary-layer cap closure differs from the source profile."
        )
    measured_volume = float(
        np.sum(np.asarray(quality.evaluation.measures, dtype=np.float64))
    )
    enclosed_volume = float(_boundary_volume(result.mesh))
    volume_residual = abs(measured_volume - enclosed_volume) / max(
        abs(enclosed_volume), np.finfo(np.float64).tiny
    )
    if volume_residual > _VOLUME_TOLERANCE:
        raise RuntimeError(
            f"Layer volume closure residual {volume_residual:.3e} exceeds tolerance."
        )
    source_cap_coverage = None
    if result.cap is not None:
        cap_vertices = np.asarray(result.cap_vertices, dtype=np.int64)
        if cap_vertices.shape != (result.cap.coordinates.shape[0],):
            raise RuntimeError(
                "The output cap lost its exact vertex-to-layer identity map."
            )
        source_cap_coverage = {
            "cap_cell_count": cap_cells,
            "cap_vertex_count": int(cap_vertices.size),
            "cap_layer_vertex_ids": cap_vertices.tolist(),
            "wall_source_face_ids": sorted(source_faces),
            "front_source_face_ids": sorted(front_sources),
            "wall_source_faces_covered": source_covered,
        }
    return {
        "stage": "prepare_boundary_layers",
        "wall_mesh_id": wall.mesh_id,
        "wall_numeric_version": wall.numeric_version,
        "control_id": control.control_id,
        "schedule_id": control.schedule.schedule_id,
        "route": control.route.value,
        "collision_policy": evidence.collision_policy.value,
        "requested_thicknesses": list(control.schedule.thicknesses),
        "measured_thicknesses": achieved.tolist(),
        "minimum_thicknesses": np.asarray(evidence.minimum_thicknesses).tolist(),
        "maximum_thicknesses": np.asarray(evidence.maximum_thicknesses).tolist(),
        "measured_growth_rates": rates.tolist(),
        "layer_active": active.tolist(),
        "relative_thickness_error": thickness_error,
        "certified_valid_cells": int(result.validity.certified_valid_count),
        "independent_valid_cells": int(independent.certified_valid_count),
        "cell_counts": dict(evidence.cell_counts),
        "predicted_collision_vertices": int(evidence.predicted_collision_vertex_count),
        "detected_collisions": int(evidence.detected_collision_count),
        "reduced_vertices": int(evidence.reduced_vertex_count),
        "terminated_vertices": int(evidence.terminated_vertex_count),
        "merged_vertices": int(evidence.merged_vertex_count),
        "closed_cap": bool(result.closed_cap),
        "cap_mesh_id": None if result.cap is None else result.cap.mesh_id,
        "source_cap_coverage": source_cap_coverage,
        "volume_closure": {
            "cell_measure": measured_volume,
            "oriented_boundary_measure": enclosed_volume,
            "relative_residual": volume_residual,
            "tolerance": _VOLUME_TOLERANCE,
        },
        "independent_certificate_id": independent.certificate_id,
        "sampled_invalid_cells": int(quality.sampled_invalid_count),
    }


def _make_cap_core_source(
    layers: BoundaryLayerMesh, /, *, region: str = "fluid"
) -> NativeLayerCoreSource:
    cap = layers.cap
    if cap is None or not layers.closed_cap:
        raise ValueError("This source profile did not author a closed fixed core cap.")
    cap_rows = _block_rows(cap)
    cell_count = sum(block.cell_count for block in layers.mesh.blocks)
    source_id = str(layers.source_wall.mesh_id)
    revision = str(layers.source_wall.numeric_version)
    vertices = np.asarray(cap.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(cap.vertex_global_ids, dtype=np.int64)
    complex_ = phx.meshing.PiecewiseLinearComplex(
        vertices,
        tuple(tuple(int(value) for value in row) for row in cap_rows),
        np.arange(cap_rows.shape[0], dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (cap_rows.shape[0], 1)),
        (region,),
        vertex_ids=tuple(f"{source_id}:cap-vertex:{int(value)}" for value in vertex_ids),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(
            f"{source_id}:cap-facet:{row}" for row in range(cap_rows.shape[0])
        ),
        facet_occurrence_paths=((),) * cap_rows.shape[0],
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    return NativeLayerCoreSource(
        layers,
        complex_,
        source_id,
        revision,
        vertex_layer_ids=np.asarray(layers.cap_vertices, dtype=np.int64),
        cap_polygon_ids=np.arange(cap_rows.shape[0], dtype=np.int64),
        layer_regions=np.zeros(cell_count, dtype=np.int64),
        region_entity_ids=np.asarray(complex_.region_entity_ids, dtype=np.int64),
        region_occurrence_paths=complex_.region_occurrence_paths,
        region_ids=(region,),
        core_region_map=np.asarray((0,), dtype=np.int64),
        core_facet_source_ids=tuple(complex_.facet_ids),
        core_facet_source_occurrence_paths=((),) * cap_rows.shape[0],
    )


def _core_specification(source: NativeLayerCoreSource, /, *, target_size: float) -> Any:
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    return phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("tetrahedron",), preferred=("prism",), allow_mixed=True
            ),
            geometry_order=2,
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                target_size,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        region_seeds=(
            phx.meshing.RegionSeed(
                np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
                source.region_ids[0],
                "physical-fluid-core",
                phx.meshing.RegionRole.SOLID,
            ),
        ),
    )


def _mimetic_affine_field(mesh: Any, /) -> dict[str, object]:
    from phydrax.discretization.finite_volume import (
        HybridDiffusionBoundary,
        HybridMimeticDiffusion,
        UnstructuredFiniteVolumePlan,
    )

    discretization = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare()
    boundary_faces = np.flatnonzero(np.asarray(discretization.neighbor_cells) < 0)
    centers = np.asarray(discretization.face_centers, dtype=np.float64)
    boundary_values = {
        int(face): float(
            1.0 + 0.2 * centers[face, 0] - 0.3 * centers[face, 1] + 0.4 * centers[face, 2]
        )
        for face in boundary_faces.tolist()
    }
    boundary = HybridDiffusionBoundary(discretization, dirichlet=boundary_values)
    system = HybridMimeticDiffusion(discretization)
    solved = system.solve(1.0, boundary)
    count = int(discretization.cell_volumes.shape[0])
    values = np.asarray(solved.value[:count], dtype=np.float64)
    expected = (
        1.0
        + 0.2 * np.asarray(discretization.cell_centers)[:, 0]
        - 0.3 * np.asarray(discretization.cell_centers)[:, 1]
        + 0.4 * np.asarray(discretization.cell_centers)[:, 2]
    )
    maximum_error = float(np.max(np.abs(values - expected), initial=0.0))
    normalized = float(
        np.max(
            np.asarray(solved.diagnostics.relative_residual, dtype=np.float64),
            initial=0.0,
        )
    )
    if (
        not np.all(np.asarray(solved.successful, dtype=np.bool_))
        or maximum_error > 2.0e-8
        or normalized > 2.0e-8
    ):
        raise RuntimeError("The mimetic physical affine-field oracle failed.")
    return {
        "consumer": "HybridMimeticDiffusion",
        "cell_unknowns": count,
        "face_unknowns": int(discretization.face_measures.shape[0]),
        "successful": bool(np.all(np.asarray(solved.successful, dtype=np.bool_))),
        "exact_field": "u=1+0.2x-0.3y+0.4z",
        "maximum_affine_error": maximum_error,
        "normalized_residual": normalized,
        "tolerance": 2.0e-8,
    }


def _run_wall_profile(name: str, /) -> dict[str, object]:
    schedule = phx.meshing.LayerSchedule.geometric(3, 0.01, growth_rate=_GROWTH_RATE)
    metadata: dict[str, object] = {
        "source_profile": name,
        "source_revision": "original-wall-fixture",
    }
    expected_closed: bool | None = None
    expected_thicknesses = True
    growth: float | None = _GROWTH_RATE
    if name == "plate5":
        points, rows = _plate(5)
        wall = _wall(points, rows)
        expected_closed = False
    elif name == "quadrilateral_plate3":
        values = np.linspace(0.0, 1.0, 4)
        x, y = np.meshgrid(values, values, indexing="ij")
        points = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)
        index = np.arange(points.shape[0]).reshape(4, 4)
        quads = np.asarray(
            [
                (index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1])
                for i in range(3)
                for j in range(3)
            ],
            dtype=np.int64,
        )
        wall = phx.discretization.CellMesh(
            points,
            (
                phx.discretization.CellBlock(
                    "quadrilaterals", "quadrilateral", quads, global_ids=np.arange(9)
                ),
            ),
        )
        expected_closed = False
        metadata["physical_input"] = "9 authored unit-square quadrilateral faces"
    elif name == "opposing_sloped_plates4":
        lower, lower_rows = _plate(4)
        upper, upper_rows = _plate(4, flip=True)
        upper[:, 2] = 0.03 + 0.17 * upper[:, 0]
        wall = _wall(
            np.concatenate((lower, upper)),
            np.concatenate((lower_rows, upper_rows + lower.shape[0])),
        )
        control = _control(
            wall,
            schedule,
            collision=phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY,
        )
        expected_closed = False
        expected_thicknesses = False
        metadata["physical_input"] = "opposed 4x4 plate grids; upper wall z=0.03+0.17*x"
        return _complete_wall_attempt(
            name,
            wall,
            control,
            schedule,
            metadata,
            expected_closed,
            expected_thicknesses,
            growth,
        )
    elif name == "sphere_radius04_subdivision2":
        points, rows = _icosphere(0.4, 2)
        wall = _wall(points, rows)
        expected_closed = True
        metadata["physical_input"] = "outward icosphere, radius 0.4, subdivision level 2"
    elif name == "outward_box_count3":
        points, rows = _box((-0.3, -0.3, -0.3), (0.3, 0.3, 0.3), 3, outward=True)
        wall = _wall(points, rows)
        expected_closed = True
        metadata["physical_input"] = "outward box [-0.3,0.3]^3, 3 subdivisions per edge"
    elif name in (
        "concave_box_count3",
        "concave-box-core-lifecycle",
        "concave-box-core-lifecycle:larger",
    ):
        points, rows = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 3, outward=False)
        wall = _wall(points, rows)
        expected_closed = True
        metadata["physical_input"] = "inward-oriented unit box, 3 subdivisions per edge"
    elif name == "open_box_floor_count4":
        points, rows = _box((0.0, 0.0, 0.0), (1.0, 1.0, 0.5), 4, outward=False)
        heights = points[rows][:, :, 2]
        floor_only_input = rows[~np.all(np.isclose(heights, 0.5), axis=1)]
        wall = _wall(points, floor_only_input)
        cells = wall.entity_set(2)
        floor = np.flatnonzero(
            np.all(np.isclose(points[floor_only_input][:, :, 2], 0.0), axis=1)
        )
        control = phx.meshing.BoundaryLayerControl(
            phx.meshing.MeshingScope(
                wall.mesh_id,
                wall.numeric_version,
                phx.meshing.MeshingEntityKind.MESH,
                2,
                cells.entity_set_id,
                np.asarray(cells.entity_ids, dtype=np.int64)[floor],
            ),
            schedule,
            route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        )
        expected_closed = False
        metadata["physical_input"] = (
            "open box [0,1]x[0,1]x[0,0.5], top removed, floor-only layer scope"
        )
        return _complete_wall_attempt(
            name, wall, control, schedule, metadata, expected_closed, True, growth
        )
    elif name == "periodic-convex-ridge":
        wall, metadata = _ridge_wall()
        metadata["source_profile"] = name
        metadata["physical_input"] = (
            "periodic x-translation wall with one authored convex ridge"
        )
        control = _control(wall, schedule, feature_angle=0.1)
        return _complete_wall_attempt(
            name, wall, control, schedule, metadata, False, True, growth
        )
    elif name.startswith("periodic-"):
        _, periodicity, kind = name.split("-", 2)
        wall, metadata = _periodic_wall(
            periodicity, quadrilateral=kind == "quadrilateral"
        )
        metadata["source_profile"] = name
        metadata["physical_input"] = (
            f"{periodicity} periodic {'quadrilateral' if kind == 'quadrilateral' else 'two-triangle'} fundamental patch"
        )
        expected_closed = periodicity == "translation"
    elif name.startswith("curved-periodic-layer-core"):
        wall, metadata = _curved_periodic_wall()
        metadata["source_profile"] = name
        metadata["physical_input"] = (
            "exact-x-orbit fundamental wall carrier; source quadratic profile points retained"
        )
        schedule = phx.meshing.LayerSchedule.geometric(2, 0.01, growth_rate=1.0)
        growth = 1.0
        control = _control(
            wall,
            schedule,
            corner=phx.meshing.BoundaryLayerCornerPolicy.SMOOTH,
            feature_angle=0.1,
            growth_rate_bounds=(0.9999999999, 1.0000000001),
        )
        wall_attempt = _complete_wall_attempt(
            name,
            wall,
            control,
            schedule,
            metadata,
            False,
            True,
            growth,
            periodic_area=1.0,
        )
        source_factory = getattr(
            phx.meshing, "prepare_corrected_periodic_layer_source", None
        )
        if source_factory is None:
            raise _ProfileFailure(
                "curved-periodic-source-map",
                "The current package exposes no corrected periodic layer/core source authoring API; exact coefficient-orbit, mapped-domain, and source-chart authority cannot be reconstructed from the wall carrier alone.",
                {
                    "wall_attempt": wall_attempt,
                    "source_factory_api": "phx.meshing.prepare_corrected_periodic_layer_source",
                },
            )
        authored = source_factory()
        raise _ProfileFailure(
            "curved-periodic-layer-core",
            "The source producer was reached but its current typed source result has no scenario adapter for the native matrix.",
            {"wall_attempt": wall_attempt, "authored_source_id": authored.source_id},
        )
    else:
        raise ValueError(f"No original wall constructor is registered for {name!r}.")
    if name == "plate5":
        control = _control(wall, schedule)
        metadata["physical_input"] = (
            "unit-square triangular plate, 5 subdivisions per edge"
        )
    elif name == "quadrilateral_plate3":
        control = _control(wall, schedule)
    elif name in (
        "sphere_radius04_subdivision2",
        "outward_box_count3",
        "concave_box_count3",
        "concave-box-core-lifecycle",
        "concave-box-core-lifecycle:larger",
    ):
        control = _control(wall, schedule)
    elif name.startswith("periodic-"):
        control = _control(wall, schedule)
    else:
        raise ValueError(
            f"The physical source route for {name!r} did not bind a control."
        )
    return _complete_wall_attempt(
        name,
        wall,
        control,
        schedule,
        metadata,
        expected_closed,
        expected_thicknesses,
        growth,
    )


def _complete_wall_attempt(
    name: str,
    wall: Any,
    control: Any,
    schedule: Any,
    metadata: dict[str, object],
    expected_closed: bool | None,
    expected_all_thicknesses: bool,
    growth: float | None,
    /,
    *,
    periodic_area: float | None = None,
) -> dict[str, object]:
    layers = phx.meshing.prepare_boundary_layers(wall, control)
    layer = _layer_record(
        wall,
        control,
        layers,
        expected_growth=growth,
        expected_all_thicknesses=expected_all_thicknesses,
        closed_cap=expected_closed,
    )
    periodic = wall.periodic_topology
    if periodic is not None:
        layer_periodic = layers.mesh.periodic_topology
        source_periodic = layers.source_wall.periodic_topology
        cap_periodic = None if layers.cap is None else layers.cap.periodic_topology
        if layer_periodic is None or source_periodic is None:
            raise RuntimeError("The layer mesh lost the authored periodic identity.")
        group_id = identification_id(periodic.cell)
        if (
            identification_id(layer_periodic.cell) != group_id
            or source_periodic.periodic_topology_id != periodic.periodic_topology_id
        ):
            raise RuntimeError("The layer mesh lost the authored periodic identity.")
        if layers.cap is not None and (
            cap_periodic is None or identification_id(cap_periodic.cell) != group_id
        ):
            raise RuntimeError("The exact cap lost periodic group correspondence.")
        periodic_evidence: dict[str, object] = {
            "source_topology_id": periodic.periodic_topology_id,
            "layer_topology_id": layer_periodic.periodic_topology_id,
            "cap_topology_id": (
                None if cap_periodic is None else cap_periodic.periodic_topology_id
            ),
            "group_id": group_id,
        }
        periodic_carriers = [layer_periodic]
        if cap_periodic is not None:
            periodic_carriers.append(cap_periodic)
        for carrier in periodic_carriers:
            incidences = tuple(
                incidence.scipy_boundary().toarray()
                for incidence in carrier.quotient.incidences
            )
            if any(
                not np.array_equal(
                    left @ right,
                    np.zeros((left.shape[0], right.shape[1]), dtype=np.int64),
                )
                for left, right in zip(incidences[:-1], incidences[1:], strict=True)
            ):
                raise RuntimeError(
                    "Periodic quotient incidence failed the exact chain-complex identity."
                )
        if periodic_area is None:
            area_value = metadata.get("expected_area")
            if isinstance(area_value, (int, float)):
                periodic_area = float(area_value)
        if periodic_area is not None:
            transition_value = metadata.get("transition_height", 0.0)
            if not isinstance(transition_value, (int, float)):
                raise TypeError("Periodic transition-height evidence must be numeric.")
            expected_volume = periodic_area * (
                schedule.total_thickness + float(transition_value) / 3.0
            )
            volume_record = layer.get("volume_closure")
            if not isinstance(volume_record, dict):
                raise TypeError("Layer volume-closure evidence is malformed.")
            measured_value = volume_record.get("cell_measure")
            if not isinstance(measured_value, (int, float)):
                raise TypeError("Layer cell measure is not numeric.")
            measured_volume = float(measured_value)
            if not np.isclose(
                measured_volume, expected_volume, rtol=_VOLUME_TOLERANCE, atol=1.0e-14
            ):
                raise RuntimeError(
                    "The periodic quotient layer volume differs from the authored patch measure."
                )
            periodic_evidence["expected_quotient_volume"] = expected_volume
        layer["periodic_identity"] = periodic_evidence
    cell_counts = dict(layers.evidence.cell_counts)
    if name == "plate5" and cell_counts != {"prism": 150}:
        raise RuntimeError(
            "The five-subdivision triangle plate lost its exact prism inventory."
        )
    if name == "quadrilateral_plate3" and cell_counts != {"pyramid": 9, "hexahedron": 27}:
        raise RuntimeError(
            "The quadrilateral plate lost its exact prism/hexahedron transition inventory."
        )
    if name == "opposing_sloped_plates4":
        if not 0 < layers.evidence.terminated_vertex_count < 50:
            raise RuntimeError(
                "The sloped opposing plates did not retain local termination evidence."
            )
        if cell_counts.get("pyramid", 0) <= 0 or cell_counts.get("tetrahedron", 0) <= 0:
            raise RuntimeError(
                "The sloped opposing plates lost their collision transition families."
            )
    if name == "periodic-convex-ridge" and layers.evidence.convex_ridge_count != 1:
        raise RuntimeError(
            "The periodic convex-ridge source did not retain its one ridge."
        )
    if name == "open_box_floor_count4" and layers.evidence.rim_vertex_count != 16:
        raise RuntimeError("The floor-only open box lost its 16 authored rim vertices.")
    if name == "sphere_radius04_subdivision2" and layers.evidence.fan_column_count != 0:
        raise RuntimeError(
            "The spherical source unexpectedly generated ridge fan columns."
        )
    if name == "outward_box_count3" and any(
        cell_counts.get(kind, 0) <= 0 for kind in ("hexahedron", "tetrahedron", "pyramid")
    ):
        raise RuntimeError(
            "The outward box lost one of its certified transition cell families."
        )
    if name in (
        "concave_box_count3",
        "concave-box-core-lifecycle",
        "concave-box-core-lifecycle:larger",
    ) and not np.isclose(
        layers.evidence.maximum_stretch, np.sqrt(3.0), rtol=0.0, atol=1.0e-6
    ):
        raise RuntimeError(
            "The inward box did not retain its authored concave-corner stretch."
        )
    if name == "open_box_floor_count4":
        coordinates = np.asarray(layers.mesh.coordinates, dtype=np.float64)
        if np.any(coordinates[:, :2] < 0.0) or np.any(coordinates[:, :2] > 1.0):
            raise RuntimeError("The open-box floor layer escaped its source side bounds.")
        if np.count_nonzero(coordinates[:, 0] == 0.0) != 20:
            raise RuntimeError(
                "The open-box floor layer lost exact side-rim column locations."
            )
    if name == "sphere_radius04_subdivision2" and layers.cap is not None:
        radii = np.linalg.norm(
            np.asarray(layers.cap.coordinates, dtype=np.float64), axis=1
        )
        if not np.all(radii > 0.4):
            raise RuntimeError(
                "The sphere cap did not retain its outward physical radius."
            )
        layer["cap_radius_range"] = [float(np.min(radii)), float(np.max(radii))]
    if name == "outward_box_count3":
        if (
            layers.evidence.convex_ridge_count != 36
            or layers.evidence.corner_patch_count != 8
        ):
            raise RuntimeError("The outward-box ridge/corner source inventory changed.")
    if name in (
        "concave_box_count3",
        "concave-box-core-lifecycle",
        "concave-box-core-lifecycle:larger",
    ):
        if (
            layers.evidence.concave_ridge_count != 36
            or layers.evidence.fan_column_count != 0
        ):
            raise RuntimeError("The inward-box concave-ridge source inventory changed.")
        source = _make_cap_core_source(layers)
        cap_count = _count_cells(layers.cap)
        if (
            source.cap_polygon_ids.size != cap_count
            or np.unique(source.cap_polygon_ids).size != cap_count
        ):
            raise RuntimeError(
                "The core PLC does not cover every fixed layer-cap facet exactly once."
            )
        layer["fixed_core_source"] = {
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "binding_id": source.binding_id,
            "core_region_ids": list(source.region_ids),
            "cap_polygon_count": int(source.cap_polygon_ids.size),
            "cap_source_facet_count": len(source.core_facet_source_ids),
            "cap_source_coverage": "all closed cap facets mapped one-to-one",
        }
        if name.startswith("concave-box-core-lifecycle"):
            target_size = 0.05 if name.endswith(":larger") else 0.1
            specification = _core_specification(source, target_size=target_size)
            options = NativeMeshingOptions("layer_core")
            if options.volume_schedule is None:
                raise RuntimeError(
                    "The current layer-core route lacks its source-matched volume schedule."
                )
            try:
                provider = phx.meshing.NativeMeshingProvider(options)
                plan = provider.plan(
                    source,
                    specification,
                    coordinate_contract=SpatialCoordinateContract.si(),
                )
                generated = plan.execute()
            except Exception as error:
                raise _ProfileFailure(
                    "native-layer-core-consumer",
                    f"The original concave wall and fixed cap were attempted through the native layer-core producer, but the current source/consumer contract refused them: {type(error).__name__}: {error}",
                    {
                        "wall_layer": layer,
                        "fixed_core_source": layer["fixed_core_source"],
                        "specification_id": specification.specification_id,
                    },
                ) from error
            certificate = generated.certification
            if certificate is None or not certificate.passed:
                raise _ProfileFailure(
                    "native-layer-core-certification",
                    "The current native layer-core route did not publish independent source coverage certification.",
                    {"wall_layer": layer, "generated_result_id": generated.result_id},
                )
            field = _mimetic_affine_field(generated.mesh)
            core_measure = float(
                np.sum(
                    np.asarray(generated.quality.evaluation.measures, dtype=np.float64)
                )
            )
            layer["native_layer_core"] = {
                "result_id": generated.result_id,
                "mesh_id": generated.mesh.mesh_id,
                "certification_passed": certificate.passed,
                "region_ids": list(source.region_ids),
                "cell_measure": core_measure,
                "physical_field": field,
                "layer_core_interface_patch_present": any(
                    patch.name == "layer-core-interface" for patch in generated.patches
                ),
            }
            curving_evidence = generated.layer_curving_evidence
            embedding = certificate.embedding
            fidelity = certificate.fidelity
            if (
                curving_evidence is None
                or embedding.status != "certified"
                or fidelity is None
                or fidelity.status != "certified"
            ):
                raise _ProfileFailure(
                    "layer-core-curving-source-map",
                    "The current layer-core route did not accept its required high-order source map, validity, embedding, and fidelity evidence.",
                    {
                        "wall_layer": layer,
                        "native_layer_core_result_id": generated.result_id,
                    },
                )
            layer["curved_validity"] = {
                "geometry_order": specification.target.geometry_order,
                "source_evidence_id": curving_evidence.evidence_id,
                "fidelity_status": fidelity.status,
                "embedding_status": embedding.status,
                "minimum_scaled_jacobian": curving_evidence.minimum_scaled_jacobian,
                "maximum_distortion": curving_evidence.maximum_distortion,
            }
    return {"status": "passed", **metadata, "wall_layer": layer}


def _run_channel_policies() -> dict[str, object]:
    lower, lower_rows = _plate(4)
    upper, upper_rows = _plate(4, 0.05, flip=True)
    points = np.concatenate((lower, upper))
    rows = np.concatenate((lower_rows, upper_rows + lower.shape[0]))
    wall = _wall(points, rows)
    schedule = phx.meshing.LayerSchedule.geometric(3, 0.01, growth_rate=_GROWTH_RATE)
    attempts: list[dict[str, object]] = []
    failures: list[str] = []
    for policy in (
        phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS,
        phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY,
        phx.meshing.BoundaryLayerCollisionPolicy.MERGE,
    ):
        case_name = policy.value
        try:
            control = _control(
                wall,
                schedule,
                collision=policy,
                minimum_thickness_fraction=0.2,
            )
            layers = phx.meshing.prepare_boundary_layers(wall, control)
            if policy is phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS:
                record = _layer_record(
                    wall, control, layers, expected_all_thicknesses=False
                )
                evidence = layers.evidence
                if (
                    not 0.2 <= evidence.minimum_scale < 1.0
                    or evidence.reduced_vertex_count != 50
                    or not np.all(np.asarray(evidence.layer_active))
                ):
                    raise RuntimeError(
                        "REDUCE_THICKNESS missed its actual narrow-gap scale and active-layer oracle."
                    )
            elif policy is phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY:
                record = _layer_record(
                    wall,
                    control,
                    layers,
                    expected_all_thicknesses=False,
                    expected_growth=None,
                )
                evidence = layers.evidence
                if evidence.terminated_vertex_count != 50 or not np.array_equal(
                    np.asarray(evidence.layer_active), (True, False, False)
                ):
                    raise RuntimeError(
                        "TERMINATE_LOCALLY missed its narrow-gap vertex/active-layer oracle."
                    )
            else:
                record = _layer_record(
                    wall,
                    control,
                    layers,
                    expected_all_thicknesses=False,
                    expected_growth=None,
                )
                evidence = layers.evidence
                expected_scale = 0.5 * 0.05 / schedule.total_thickness
                if (
                    evidence.merged_vertex_count != 50
                    or not np.isclose(
                        evidence.minimum_scale, expected_scale, rtol=0.0, atol=1.0e-6
                    )
                    or layers.cap is not None
                ):
                    raise RuntimeError(
                        "MERGE missed its physical gap closure and merged-front oracle."
                    )
                expected_first = schedule.thicknesses[0] * expected_scale
                if not np.isclose(
                    float(evidence.achieved_thicknesses[0]),
                    expected_first,
                    rtol=_LAYER_TOLERANCE,
                    atol=1.0e-14,
                ):
                    raise RuntimeError(
                        "MERGE did not report the physically scaled first-layer thickness."
                    )
                rates = np.asarray(evidence.achieved_growth_rates, dtype=np.float64)
                if (
                    rates.shape != (2,)
                    or not np.all(np.isfinite(rates))
                    or np.max(np.abs(rates - schedule.growth_rates)) > _LAYER_TOLERANCE
                ):
                    raise RuntimeError(
                        "MERGE lost the physical growth schedule before the shared midsurface."
                    )
                heights = np.asarray(layers.mesh.coordinates, dtype=np.float64)[:, 2]
                if not np.any(np.isclose(heights, 0.025, rtol=0.0, atol=1.0e-12)):
                    raise RuntimeError(
                        "MERGE did not realize the exact shared channel midsurface."
                    )
            record["policy_case"] = case_name
            record["expected_front_clearance"] = (
                0.0 if policy is phx.meshing.BoundaryLayerCollisionPolicy.MERGE else 0.05
            )
            attempts.append({"status": "passed", **record})
        except Exception as error:
            failures.append(case_name)
            attempts.append(
                {
                    "status": "failed",
                    "policy_case": case_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    if failures:
        raise _ProfileFailure(
            "channel-gap-collision-policies",
            f"The 0.05 physical channel was attempted for all three original collision policies; failing policy cases: {failures}.",
            {
                "physical_gap": 0.05,
                "wall_cells": _count_cells(wall),
                "policy_attempts": attempts,
            },
        )
    return {
        "status": "passed",
        "source_profile": "channel_gap005_policies",
        "physical_input": "opposing 4x4 unit-square triangle plates separated by exactly 0.05",
        "physical_gap": 0.05,
        "policy_attempts": attempts,
    }


def _brep_scope(model: Any, dimension: int, identifiers: tuple[int, ...], /) -> Any:
    return phx.meshing.MeshingScope(
        model.source_id,
        model.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        dimension,
        f"{model.source_revision}:brep:{dimension}",
        np.asarray(identifiers, dtype=np.int64),
    )


def _face_vertex_rows(model: Any, face: int, /) -> np.ndarray:
    geometry = model.geometry
    if geometry is None:
        raise ValueError("The native BRep source omitted its exact geometry owner.")
    return np.asarray(
        sorted(
            {
                int(vertex)
                for loop in geometry.face_loops[face]
                for coedge in loop
                for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
            }
        ),
        dtype=np.int64,
    )


def _run_cad_extrusion(name: str, /) -> dict[str, object]:
    remaining_core = name == "cad-slab-and-core"
    thicknesses = (0.05, 0.10) if remaining_core else (0.25, 0.75)
    model = phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 1.0),
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.05, angular_deflection=0.2
        ),
    )
    geometry = model.geometry
    if geometry is None:
        raise RuntimeError("The native CAD slab source has no retained BRep geometry.")
    walls = tuple(
        face
        for face in range(model.topology.num_faces)
        if np.all(
            np.asarray(geometry.vertex_points, dtype=np.float64)[
                _face_vertex_rows(model, face), 2
            ]
            == 0.0
        )
    )
    control = phx.meshing.BoundaryLayerControl(
        _brep_scope(model, 2, walls),
        phx.meshing.LayerSchedule(thicknesses),
        route=phx.meshing.BoundaryLayerRoute.CAD_EXTRUSION,
        volume_scope=_brep_scope(model, 3, (0,)),
    )
    try:
        extrusion = phx.meshing.prepare_boundary_layer_extrusion(model, control)
    except Exception as error:
        raise _ProfileFailure(
            "native-cad-exact-partition",
            f"The original CAD box and authored slab schedule were attempted through prepare_boundary_layer_extrusion; current source/consumer refusal: {type(error).__name__}: {error}",
            {
                "source_id": model.source_id,
                "source_revision": model.source_revision,
                "source_units": "MILLIMETER",
                "wall_face_ids": list(walls),
                "requested_thicknesses": list(thicknesses),
                "schedule_id": control.schedule.schedule_id,
            },
        ) from error
    expected = 6.0 * sum(thicknesses)
    volumes = np.asarray(extrusion.slab_volumes, dtype=np.float64)
    residual = abs(float(np.sum(volumes)) - expected) / expected
    if (
        residual > 1.0e-10
        or extrusion.maximum_relative_volume_residual > _VOLUME_TOLERANCE
    ):
        raise RuntimeError(
            "The CAD slab producer failed independent exact slab-volume closure."
        )
    if extrusion.control.route is not phx.meshing.BoundaryLayerRoute.EXACT_SWEEP:
        raise RuntimeError(
            "The native CAD extrusion did not produce an exact-sweep consumer control."
        )
    if len(extrusion.layer_solid_ids) != 1 or len(extrusion.core_solid_ids) != int(
        remaining_core
    ):
        raise RuntimeError(
            "The exact CAD partition lost its authored layer/core solid coverage."
        )
    wall_scope = extrusion.control.wall_scope
    cap_scope = extrusion.control.cap_scope
    volume_scope = extrusion.control.volume_scope
    if cap_scope is None or volume_scope is None:
        raise RuntimeError("The CAD extrusion lost its exact wall/cap/volume scopes.")
    return {
        "status": "passed",
        "source_profile": name,
        "producer_route": "prepare_boundary_layer_extrusion -> native BRep partition -> EXACT_SWEEP control",
        "physical_input": "native planar BRep box (2,3,1) millimeters; bottom wall face",
        "source_id": extrusion.source.source_id,
        "source_revision": extrusion.source.source_revision,
        "original_source_id": model.source_id,
        "original_source_revision": model.source_revision,
        "source_units": "MILLIMETER",
        "requested_thicknesses": list(thicknesses),
        "measured_slab_volumes": volumes.tolist(),
        "expected_total_slab_volume": expected,
        "relative_volume_closure_residual": residual,
        "partition_relative_volume_residual": extrusion.maximum_relative_volume_residual,
        "layer_solid_ids": list(extrusion.layer_solid_ids),
        "core_solid_ids": list(extrusion.core_solid_ids),
        "wall_scope_ids": np.asarray(wall_scope.entity_ids, dtype=np.int64).tolist(),
        "cap_scope_ids": np.asarray(cap_scope.entity_ids, dtype=np.int64).tolist(),
        "volume_scope_ids": np.asarray(volume_scope.entity_ids, dtype=np.int64).tolist(),
    }


def _run_exact_sweep(name: str, /) -> dict[str, object]:
    periodic = "periodic" in name
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
    wall = _wall(points, rows)
    if periodic:
        generator = np.eye(4, dtype=np.float64)[None]
        generator[0, 0, 3] = 1.0
        topology = phx.discretization.PeriodicMeshTopology(
            wall,
            phx.discretization.PeriodicIsometryGroup(generator),
            np.asarray((0, 0, 2, 2, 4, 4), dtype=np.int64),
            np.asarray(((0,), (1,), (0,), (1,), (0,), (1,)), dtype=np.int64),
        )
        wall = phx.discretization.CellMesh(
            points, wall.blocks, periodic_topology=topology
        )
    periodic_source_id: str | None = None
    if periodic:
        source_topology = wall.periodic_topology
        if source_topology is None:
            raise ValueError("The authored exact-sweep source lost its periodic topology.")
        periodic_source_id = source_topology.periodic_topology_id
    schedule = phx.meshing.LayerSchedule.geometric(2, 0.01, growth_rate=1.0)
    stations = np.concatenate(
        (np.zeros(1), np.cumsum(np.asarray(schedule.thicknesses, dtype=np.float64)))
    )
    upper = points + stations[-1] * np.asarray((0.0, 0.0, 1.0))
    edges = np.asarray(((0, 1), (1, 3), (3, 5), (5, 4), (4, 2), (2, 0)), dtype=np.int64)
    a, b = edges.T
    sides = np.stack(
        (np.stack((a, b, 6 + b), axis=1), np.stack((a, 6 + b, 6 + a), axis=1)),
        axis=1,
    ).reshape(-1, 3)
    facets = np.concatenate((rows[:, ::-1], rows + 6, sides))
    domain = phx.geometry.PiecewiseLinearDomain(
        np.concatenate((points, upper)),
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (facets.shape[0], 1)),
        ("wall-material",),
        source_id=f"{name}:exact-reference-slab",
    )
    wall_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        f"{domain.domain_id}:facets",
        np.arange(rows.shape[0], dtype=np.int64),
    )
    cap_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        f"{domain.domain_id}:facets",
        np.arange(rows.shape[0], rows.shape[0] * 2, dtype=np.int64),
    )
    volume_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        f"{domain.domain_id}:regions",
        np.asarray((0,), dtype=np.int64),
    )
    entity_ids = np.asarray(wall.entity_set(2).entity_ids, dtype=np.int64)
    association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        domain.source_id,
        domain.source_revision,
        wall.entity_set(2).entity_set_id,
        entity_ids,
        tuple(f"{domain.source_revision}:facet:{row}" for row in range(rows.shape[0])),
        np.zeros(rows.shape[0]),
        exact=True,
        source_dimensions=np.full(rows.shape[0], 2, dtype=np.int64),
        source_indices=np.arange(rows.shape[0], dtype=np.int64),
        source_roles=(GeometrySourceEntityRole.FACET,) * rows.shape[0],
        orientations=-np.ones(rows.shape[0], dtype=np.int32),
    )
    control = phx.meshing.BoundaryLayerControl(
        wall_scope,
        schedule,
        route=phx.meshing.BoundaryLayerRoute.EXACT_SWEEP,
        volume_scope=volume_scope,
        cap_scope=cap_scope,
    )
    sweep = phx.meshing.SweepControl(
        phx.meshing.SweepMapKind.EXTRUSION, schedule, "original-reference-schedule"
    )
    try:
        phx.meshing.prepare_boundary_layers(
            wall,
            control,
            wall_association=association,
        )
    except Exception as error:
        raise _ProfileFailure(
            "exact-sweep-producer-route",
            f"The exact physical sweep source and boundary association reached the current prepare_boundary_layers API, which refused its exact-sweep control: {type(error).__name__}: {error}",
            {
                "source_profile": name,
                "wall_mesh_id": wall.mesh_id,
                "source_domain_id": domain.domain_id,
                "wall_association_id": association.association_id,
                "control_id": control.control_id,
                "sweep_control_id": sweep.control_id,
                "schedule_thicknesses": list(schedule.thicknesses),
                "periodic": periodic,
                "periodic_source_topology_id": periodic_source_id,
            },
        ) from error
    raise _ProfileFailure(
        "exact-sweep-producer-oracle",
        "The current boundary-layer producer unexpectedly returned without a native exact-sweep acceptance record.",
        {
            "source_profile": name,
            "source_domain_id": domain.domain_id,
            "wall_association_id": association.association_id,
            "control_id": control.control_id,
            "sweep_control_id": sweep.control_id,
        },
    )


def _run_unit_box() -> dict[str, object]:
    model = phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="hybrid-unit-box",
    )
    domain = phx.geometry.MeshingDomain.from_brep(model)
    geometry = model.geometry
    if geometry is None:
        raise RuntimeError("The exact unit-box BRep omitted its source geometry.")
    face_rows = tuple(int(value) for value in domain.source_indices[2])
    bottom = tuple(
        row
        for row, face in enumerate(face_rows)
        if np.all(
            np.asarray(geometry.vertex_points, dtype=np.float64)[
                _face_vertex_rows(model, face), 2
            ]
            == 0.0
        )
    )
    if len(bottom) != 1:
        raise RuntimeError("The exact unit-box BRep did not resolve one bottom face.")
    bottom_row = bottom[0]
    surface_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64),
    )
    surface_source = domain
    surface_specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        surface_scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                surface_scope, 0.125, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    surface_provider = phx.meshing.NativeMeshingProvider(
        NativeMeshingOptions("parametric_surface")
    )
    surface_plan = surface_provider.plan(
        surface_source,
        surface_specification,
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    surface_result = surface_plan.execute()
    surface_mesh = surface_result.mesh
    triangle_blocks = tuple(
        block for block in surface_mesh.blocks if block.cell_kind == "triangle"
    )
    if not triangle_blocks or len(triangle_blocks) != len(surface_mesh.blocks):
        raise RuntimeError(
            "The exact native unit-box source did not publish only triangular faces."
        )
    all_triangles = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in triangle_blocks], axis=0
    )
    all_cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in triangle_blocks],
        axis=0,
    )
    surface_points = np.asarray(surface_mesh.coordinates, dtype=np.float64)
    bottom_mask = np.all(np.abs(surface_points[all_triangles, 2]) <= 1.0e-12, axis=1)
    selected_triangles = all_triangles[bottom_mask]
    selected_cell_ids = all_cell_ids[bottom_mask]
    wall_vertex_rows = np.unique(selected_triangles.reshape(-1))
    if not wall_vertex_rows.size or not selected_cell_ids.size:
        raise RuntimeError(
            "The native unit-box surface route produced no exact bottom-wall triangles."
        )
    wall_points = surface_points[wall_vertex_rows]
    wall_triangles = np.searchsorted(wall_vertex_rows, selected_triangles)
    normals = np.cross(
        wall_points[wall_triangles[:, 1]] - wall_points[wall_triangles[:, 0]],
        wall_points[wall_triangles[:, 2]] - wall_points[wall_triangles[:, 0]],
    )
    flip = normals[:, 2] < 0.0
    wall_triangles[flip] = wall_triangles[flip][:, ::-1]
    wall = phx.discretization.CellMesh(
        wall_points,
        (
            phx.discretization.CellBlock(
                "native-unit-box-wall",
                "triangle",
                wall_triangles,
                global_ids=selected_cell_ids,
            ),
        ),
        vertex_global_ids=np.asarray(surface_mesh.vertex_global_ids, dtype=np.int64)[
            wall_vertex_rows
        ],
    )
    projection = phx.geometry.prepare_brep_projection(model)
    association_policy = phx.meshing.AssociationPropagationPolicy(
        classification_tolerance=1.0e-8
    )
    wall_vertex_association = phx.meshing.associate_mesh_vertices(
        wall, projection, policy=association_policy
    )
    association = phx.meshing.associate_mesh_entities(
        wall, wall_vertex_association, projection, 2, policy=association_policy
    )
    whole = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64),
    )
    wall_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray((domain.scope_indices(2)[bottom_row],), dtype=np.int64),
    )
    volume_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        domain.entity_set_id(3),
        np.asarray(domain.scope_indices(3), dtype=np.int64),
    )
    schedule = phx.meshing.LayerSchedule.geometric(2, 0.1, growth_rate=1.0)
    control = phx.meshing.BoundaryLayerControl(
        wall_scope,
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        volume_scope=volume_scope,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
    )
    layers = phx.meshing.prepare_boundary_layers(
        wall,
        control,
        wall_association=association,
    )
    layer_record = _layer_record(
        wall,
        control,
        layers,
        closed_cap=False,
        expected_growth=1.0,
        source_face_ids=wall.entity_set(2).entity_ids,
    )
    if layers.cap is None:
        raise _ProfileFailure(
            "unit-box-cap",
            "The source-bound bottom-wall layers produced no fixed cap.",
            {"wall_layer": layer_record},
        )
    cap_rows = _block_rows(layers.cap)
    cap_points = np.asarray(layers.cap.coordinates, dtype=np.float64)
    upper_points = cap_points.copy()
    upper_points[:, 2] = 1.0
    count = cap_points.shape[0]
    edge_rows = np.concatenate(
        (cap_rows[:, (0, 1)], cap_rows[:, (1, 2)], cap_rows[:, (2, 0)])
    )
    _, edge_inverse, edge_counts = np.unique(
        np.sort(edge_rows, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    rim = edge_rows[edge_counts[edge_inverse] == 1]
    side_rows: list[tuple[int, int, int]] = []
    side_sources: list[int] = []
    for first, second in rim:
        endpoints = cap_points[[first, second]]
        candidates = [
            row
            for row, face in enumerate(face_rows)
            if row != bottom_row
            and any(
                np.all(
                    np.isclose(
                        np.asarray(geometry.vertex_points)[
                            _face_vertex_rows(model, face), axis
                        ],
                        value,
                    )
                )
                for axis, value in ((0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0))
                if np.all(np.isclose(endpoints[:, axis], value))
            )
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "A unit-box fixed cap rim edge lost its exact BRep side-face owner."
            )
        side_rows.extend(
            (
                (int(first), int(second), int(second + count)),
                (int(first), int(second + count), int(first + count)),
            )
        )
        side_sources.extend((candidates[0], candidates[0]))
    top_row = next(
        row
        for row, face in enumerate(face_rows)
        if row != bottom_row
        and np.all(
            np.asarray(geometry.vertex_points, dtype=np.float64)[
                _face_vertex_rows(model, face), 2
            ]
            == 1.0
        )
    )
    points = np.concatenate((cap_points, upper_points))
    top = cap_rows + count
    facets = np.concatenate((cap_rows, top, np.asarray(side_rows, dtype=np.int64)))
    face_mapping: list[int | None] = [None] * cap_rows.shape[0]
    face_mapping.extend([top_row] * top.shape[0])
    face_mapping.extend(side_sources)
    source_face_ids = tuple(
        None if row is None else domain.entity_id(2, row) for row in face_mapping
    )
    source_paths = tuple(
        None if row is None else tuple(domain.source_occurrences[2][row])
        for row in face_mapping
    )
    core_complex = phx.meshing.PiecewiseLinearComplex(
        points,
        tuple(tuple(int(value) for value in row) for row in facets),
        np.arange(facets.shape[0], dtype=np.int64),
        np.asarray(((0, -1), *((-1, 0),) * (facets.shape[0] - 1)), dtype=np.int64),
        ("fluid",),
        vertex_ids=tuple(
            f"{model.source_id}:layer-core-vertex:{row}" for row in range(points.shape[0])
        ),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=tuple(
            f"{model.source_id}:layer-core-facet:{row}" for row in range(facets.shape[0])
        ),
        facet_occurrence_paths=((),) * facets.shape[0],
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    source = NativeLayerCoreSource(
        layers,
        core_complex,
        domain.source_id,
        domain.source_revision,
        vertex_layer_ids=np.concatenate(
            (
                np.asarray(layers.cap_vertices, dtype=np.int64),
                np.full(count, -1, dtype=np.int64),
            )
        ),
        cap_polygon_ids=np.arange(cap_rows.shape[0], dtype=np.int64),
        layer_regions=np.zeros(_count_cells(layers.mesh), dtype=np.int64),
        region_entity_ids=np.asarray(core_complex.region_entity_ids, dtype=np.int64),
        region_occurrence_paths=core_complex.region_occurrence_paths,
        region_ids=("fluid",),
        core_region_map=np.asarray((0,), dtype=np.int64),
        source_boundary_scope=whole,
        source_domain=domain,
        fidelity_source=phx.geometry.MeshingDomainBoundarySource(
            domain, tuple(range(len(domain.patches)))
        ),
        core_facet_source_ids=source_face_ids,
        core_facet_source_occurrence_paths=source_paths,
    )
    family = {block.cell_kind for block in layers.mesh.blocks} | {"tetrahedron"}
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=tuple(sorted(family)),
                preferred=tuple(sorted(family)),
                allow_mixed=True,
            ),
            geometry_order=2,
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                whole, 0.125, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        region_seeds=(
            phx.meshing.RegionSeed(
                np.asarray((0.5, 0.5, 0.5)),
                "fluid",
                "unit-box-fluid",
                phx.meshing.RegionRole.SOLID,
            ),
        ),
        layer_controls=(control,),
    )
    try:
        provider = phx.meshing.NativeMeshingProvider(NativeMeshingOptions("layer_core"))
        plan = provider.plan(
            source, specification, coordinate_contract=SpatialCoordinateContract.si()
        )
        generated = plan.execute()
    except Exception as error:
        raise _ProfileFailure(
            "unit-box-native-layer-core",
            f"The exact native unit-box source, bottom wall, fixed cap, and remaining BRep faces reached the native layer-core consumer, which refused the source map: {type(error).__name__}: {error}",
            {
                "wall_layer": layer_record,
                "source_binding_id": source.binding_id,
                "source_facet_count": len(source.core_facet_source_ids),
                "wall_source_result_id": surface_result.result_id,
                "wall_association_id": association.association_id,
            },
        ) from error
    certificate = generated.certification
    if certificate is None or not certificate.passed:
        raise _ProfileFailure(
            "unit-box-source-coverage",
            "Native unit-box layer/core publication did not independently certify its source domain.",
            {"wall_layer": layer_record, "result_id": generated.result_id},
        )
    curving_evidence = generated.layer_curving_evidence
    if (
        curving_evidence is None
        or certificate.embedding.status != "certified"
        or certificate.fidelity is None
        or certificate.fidelity.status != "certified"
    ):
        raise _ProfileFailure(
            "unit-box-curved-source-map",
            "The current unit-box layer/core route did not accept exact high-order source-map, embedding, and fidelity evidence.",
            {"wall_layer": layer_record, "result_id": generated.result_id},
        )
    measured_volume = float(
        np.sum(np.asarray(generated.quality.evaluation.measures, dtype=np.float64))
    )
    volume_residual = abs(measured_volume - 1.0)
    if volume_residual > _VOLUME_TOLERANCE:
        raise _ProfileFailure(
            "unit-box-volume-closure",
            "The native layer/core mesh did not close to the exact unit-box volume.",
            {
                "measured_volume": measured_volume,
                "expected_volume": 1.0,
                "absolute_residual": volume_residual,
            },
        )
    field = _mimetic_affine_field(generated.mesh)
    return {
        "status": "passed",
        "source_profile": "unit-box",
        "physical_input": "native BRep unit cube with source-bound bottom triangle wall and two 0.1 advancing layers",
        "producer_route": "native BRep source -> prepare_boundary_layers -> NativeLayerCoreSource -> native layer_core provider",
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_binding_id": source.binding_id,
        "core_facet_sources": list(source.core_facet_source_ids),
        "cap_polygon_count": int(source.cap_polygon_ids.size),
        "result_id": generated.result_id,
        "certification_passed": certificate.passed,
        "wall_source_result_id": surface_result.result_id,
        "wall_association_id": association.association_id,
        "whole_source_face_coverage": {
            "core_source_facets": sum(
                value is not None for value in source.core_facet_source_ids
            ),
            "fixed_cap_facets": int(source.cap_polygon_ids.size),
            "source_boundary_scope_ids": whole.entity_ids.tolist(),
        },
        "volume_closure": {
            "measured_volume": measured_volume,
            "expected_volume": 1.0,
            "absolute_residual": volume_residual,
            "tolerance": _VOLUME_TOLERANCE,
        },
        "physical_field": field,
        "curved_validity": {
            "geometry_order": specification.target.geometry_order,
            "source_evidence_id": curving_evidence.evidence_id,
            "fidelity_status": certificate.fidelity.status,
            "embedding_status": certificate.embedding.status,
            "minimum_scaled_jacobian": curving_evidence.minimum_scaled_jacobian,
            "maximum_distortion": curving_evidence.maximum_distortion,
        },
        "wall_layer": layer_record,
    }


def _run_profile(profile_id: str, /) -> dict[str, object]:
    name = _profile_name(profile_id)
    if name == "archived-layer-core":
        raise _ProfileFailure(
            "archive-source-load",
            "The original profile requires the authenticated layer-core archive path and exact archive content ID; the migrated profile request provides neither and the no-argument current scenario cannot truthfully restore or substitute that accepted state.",
            {
                "profile_identity": profile_id,
                "source_required": "authenticated accepted layer-core archive",
                "archive_path": None,
                "content_id": None,
            },
        )
    if name in ("cad-entire-solid-slab", "cad-slab-and-core"):
        return _run_cad_extrusion(name)
    if name in ("exact-sweep-nonperiodic-accepted", "exact-sweep-periodic-accepted"):
        return _run_exact_sweep(name)
    if name == "channel_gap005_policies":
        return _run_channel_policies()
    if name == "unit-box":
        return _run_unit_box()
    return _run_wall_profile(name)


def native_hybrid_layer_profile_matrix() -> dict[str, object]:
    """Attempt every recovered hybrid layer profile using its original physical input."""
    profile_ids = _migration_profiles()
    profiles: dict[str, dict[str, object]] = {}
    for profile_id in profile_ids:
        try:
            record = _run_profile(profile_id)
            if record.get("status") != "passed":
                raise RuntimeError(
                    "The profile attempt returned without a passing source/oracle record."
                )
        except _ProfileFailure as error:
            record = {
                "status": "failed",
                "stage": error.stage,
                "error_type": type(error).__name__,
                "error": str(error),
                **error.evidence,
            }
        except Exception as error:
            record = {
                "status": "failed",
                "stage": "profile-attempt",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        profiles[profile_id] = {"profile_id": profile_id, **record}
    passed = sum(value["status"] == "passed" for value in profiles.values())
    failed = len(profiles) - passed
    return {
        "status": "passed" if failed == 0 else "failed",
        "workflow": _WORKFLOW,
        "source_registry": "tools/meshing_qualification/native_case_migration.json#workflow_contracts[hybrid-layers]",
        "attempted_profile_count": len(profiles),
        "passed_profile_count": passed,
        "failed_profile_count": failed,
        "profile_attempts": profiles,
        "tolerances": {
            "relative_layer_thickness": _LAYER_TOLERANCE,
            "growth_rate_error": _LAYER_TOLERANCE,
            "volume_closure_relative": _VOLUME_TOLERANCE,
            "geometry_certificate": "all active layer cells independently certified valid",
        },
    }
