from __future__ import annotations

import json
import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, cast, Literal, NamedTuple

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._periodic_exact import (
    identification_id,
    PeriodicIsometryGroup,
)
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep import (
    brep_box,
    BRepTessellationPolicy,
    prepare_brep_projection,
)
from phydrax.meshing._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._boundary_layer import BoundaryLayerMesh
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import NativeLayerCoreSource


_WORKFLOW = "hybrid-layers"
_PREFIX = "native-hybrid-lifecycle:"
_LAYER_TOLERANCE = 1.0e-9
_VOLUME_TOLERANCE = 1.0e-10
_GROWTH_RATE = 1.2
# Declared symbolic work (ordering, elimination tree, counts) of every
# harmonic-history Cholesky analysis: the refined archived-layer-core operator
# (238,387 P2 DOFs) needs 1,111,891,428 under nested dissection, so 2**33
# leaves ~7.7x.
_SYMBOLIC_WORK_BUDGET = 1 << 33
# One declared execution budget serves every harmonic-history solve and Hodge
# admission in each profile. The unit-box refined target's untwisted order-2
# complex refused at 4 GiB: numeric admission required 6,804,653,553 bytes.
# Round that measured requirement upward to the next 256 MiB; later Hodges
# remain subject to the same declared limit.
_UNIT_BOX_SOLVE_BUDGET = 26 << 28
_LAYER_CORE_SOLVE_BUDGET = 4 << 30
# MAC-20261107T1759Z: declared byte budget of every layer/core cycle archive,
# writer and cold reader alike. The cold reader materializes the archive's
# logical aggregate, so writer and reader share this declared archive budget.
_LAYER_CORE_ARCHIVE_BUDGET = 4 << 30


class _ProfileFailure(RuntimeError):
    def __init__(
        self, stage: str, message: str, evidence: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.evidence = {} if evidence is None else evidence


class _LayerCoreLifecycle(NamedTuple):
    """Original per-profile material, manufactured-field and cycle controls.

    ``solve_memory_bytes`` is the profile's declared execution-memory budget of
    every harmonic-history solve and factorization, writer and cold reader
    alike (MAC-20261107T1649Z); it enters plan_id and is never host-derived.
    ``archive_bytes`` is its declared cycle-archive byte budget
    (MAC-20261107T1759Z), recorded with each cycle and checked by the reader.
    """

    densities: tuple[float, ...]
    site_id: str
    harmonic: tuple[float, float, float]
    cycles: int
    semantics: str
    density_semantics: str
    solve_memory_bytes: int
    archive_bytes: int


# Material densities/sites are the original campaign controls; nonperiodic
# profiles keep the original x+y+z patch, the x-periodic ridge its y+z patch.
_LAYER_CORE_LIFECYCLES = {
    "archived-layer-core": _LayerCoreLifecycle(
        (2.0, 3.0),
        "two-material-density",
        (0.0, 1.0, 1.0),
        1,
        "original-periodic-ridge",
        "two-material-cell-average-density",
        _LAYER_CORE_SOLVE_BUDGET,
        _LAYER_CORE_ARCHIVE_BUDGET,
    ),
    "concave-box-core-lifecycle": _LayerCoreLifecycle(
        (2.0,),
        "fluid-density",
        (1.0, 1.0, 1.0),
        1,
        "original-concave-closed-cap",
        "fluid-cell-average-density",
        _LAYER_CORE_SOLVE_BUDGET,
        _LAYER_CORE_ARCHIVE_BUDGET,
    ),
    "concave-box-core-lifecycle:larger": _LayerCoreLifecycle(
        (2.0,),
        "fluid-density",
        (1.0, 1.0, 1.0),
        2,
        "original-concave-closed-cap",
        "fluid-cell-average-density",
        _LAYER_CORE_SOLVE_BUDGET,
        _LAYER_CORE_ARCHIVE_BUDGET,
    ),
    "unit-box": _LayerCoreLifecycle(
        (2.0,),
        "fluid-density",
        (1.0, 1.0, 1.0),
        1,
        "original-unit-box",
        "fluid-cell-average-density",
        _UNIT_BOX_SOLVE_BUDGET,
        _LAYER_CORE_ARCHIVE_BUDGET,
    ),
}


def _layer_core_archive_limits(budget: int, /) -> ArrayArchiveLimits:
    """Cycle-archive admission limits from one declared byte budget.

    Container, aggregate and member bytes are the budget. Element and axis
    counts use the same number, since each element takes at least one byte.
    Manifest bytes are budget / 64, central-directory bytes budget / 1024.
    Member-count and nesting limits stay the codec's model limits.
    """
    from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS

    return replace(
        MODEL_ARCHIVE_LIMITS,
        max_container_bytes=budget,
        max_aggregate_bytes=budget,
        max_member_bytes=budget,
        max_manifest_bytes=budget // 64,
        max_central_directory_bytes=budget // 1024,
        max_axis_length=budget,
        max_array_elements=budget,
        max_total_array_elements=budget,
    )


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
    group = PeriodicIsometryGroup(generators)
    topology = PeriodicMeshTopology(lifted, group, representatives, shifts)
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
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicIsometryGroup(generator),
        np.asarray((0, 0, 2, 2, 4, 4), dtype=np.int64),
        np.asarray(((0,), (1,), (0,), (1,), (0,), (1,)), dtype=np.int64),
    )
    wall = phx.discretization.CellMesh(points, (block,), periodic_topology=topology)
    return wall, {
        "periodicity": "translation-x-with-convex-ridge",
        "periodic_topology_id": topology.periodic_topology_id,
        "group_id": identification_id(topology.cell),
    }


def _archived_ridge_source(
    *,
    cap_refinement: Literal["fixed", "conforming"] = "fixed",
    noncap_refinement: Literal["fixed", "conforming"] = "fixed",
) -> tuple[NativeLayerCoreSource, Any, Any]:
    """Reconstruct the original two-material ridge and fixed 0.03 core offset."""
    wall, _ = _ridge_wall()
    schedule = phx.meshing.LayerSchedule.geometric(2, 0.01, growth_rate=1.0)
    control = _control(
        wall,
        schedule,
        feature_angle=0.1,
        corner=phx.meshing.BoundaryLayerCornerPolicy.SMOOTH,
    )
    policy = phx.meshing.BoundaryLayerPolicy()
    layers = phx.meshing.prepare_boundary_layers(wall, control, policy=policy)
    _complete_wall_attempt(
        "archived-layer-core",
        wall,
        control,
        schedule,
        {"source_profile": "archived-layer-core"},
        False,
        True,
        1.0,
        prepared_layers=layers,
    )
    if (
        not np.isfinite(layers.evidence.minimum_visibility)
        or layers.evidence.minimum_visibility <= 0.0
        or layers.evidence.detected_collision_count != 0
    ):
        raise ValueError("The original ridge layers lack collision-free visible columns.")
    cap = layers.cap
    if cap is None or cap.periodic_topology is None:
        raise ValueError("The original ridge source requires its periodic cap.")
    cap_rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in cap.blocks]
    )
    cap_points = np.asarray(cap.coordinates, dtype=np.float64)
    count = len(cap_points)
    upper = cap_points + np.asarray((0.0, 0.0, 0.03), dtype=np.float64)
    roots = np.asarray(cap.periodic_topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(cap.periodic_topology.vertex_shifts, dtype=np.int64)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row in cast(list[list[int]], cap_rows.tolist()):
        for a, b in zip(row, row[1:] + row[:1], strict=True):
            first, second = int(a), int(b)
            edges.setdefault((min(first, second), max(first, second)), []).append(
                (first, second)
            )
    polygons = cap_rows.tolist() + (cap_rows + count).tolist()
    seam_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for incidents in edges.values():
        if len(incidents) != 1:
            continue
        a, b = incidents[0]
        forward = (int(roots[a]), a) <= (int(roots[b]), b)
        first, second = (a, b) if forward else (b, a)
        triangles = (
            (first, second, second + count),
            (first, second + count, first + count),
        )
        start = len(polygons)
        polygons.extend(
            triangles if forward else tuple(tuple(reversed(row)) for row in triangles)
        )
        if roots[a] != roots[b] and shifts[a, 0] == shifts[b, 0]:
            seam_key = cast(
                tuple[int, int], tuple(sorted((int(roots[a]), int(roots[b]))))
            )
            seam_edges.setdefault(seam_key, []).append((start, start + 1))
    if any(len(copies) != 2 for copies in seam_edges.values()):
        raise ValueError("The original ridge core has unmatched periodic seam edges.")
    pairs = np.asarray(
        [
            (copies[0][triangle], copies[1][triangle])
            for copies in seam_edges.values()
            for triangle in range(2)
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    if pairs.size == 0:
        raise ValueError("The original ridge core lost its periodic seam pairs.")
    cap_count = len(cap_rows)
    incidence = np.tile(np.asarray((-1, 0), dtype=np.int64), (len(polygons), 1))
    incidence[:cap_count] = (0, -1)
    complex_ = PiecewiseLinearComplex(
        np.concatenate((cap_points, upper)),
        tuple(np.asarray(row, dtype=np.int64) for row in polygons),
        np.arange(len(polygons), dtype=np.int64),
        incidence,
        ("core-material",),
        vertex_ids=tuple(f"ridge-core-vertex:{row}" for row in range(2 * count)),
        vertex_occurrence_paths=((),) * (2 * count),
        facet_ids=tuple(f"ridge-core-facet:{row}" for row in range(len(polygons))),
        facet_occurrence_paths=((),) * len(polygons),
        segments=np.empty((0, 2), dtype=np.int64),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    revision = canonical_fingerprint(
        {
            "kind": "periodic-layered-gap",
            "coordinate_contract": SpatialCoordinateContract.si().spatial_id,
            "wall_coordinates": phx.array_tree_fingerprint(
                np.asarray(wall.coordinates, dtype=np.float64)
            ),
            "wall_triangles": phx.array_tree_fingerprint(
                np.asarray(wall.blocks[0].vertices, dtype=np.int64)
            ),
            "layer_thicknesses": schedule.thicknesses,
            "layer_control": control.control_id,
            "layer_policy": policy.policy_id,
            "regions": ("wall-material", "core-material"),
            "period": (1.0, 0.0, 0.0),
            "core_offset": 0.03,
        }
    )
    from phydrax.meshing import _layer_core_topology as topology

    vertex_map = np.concatenate(
        (
            np.asarray(layers.cap_vertices, dtype=np.int64),
            np.full(count, -1, dtype=np.int64),
        )
    )
    cap_ids = np.arange(cap_count, dtype=np.int64)
    layer_regions = np.zeros(_count_cells(layers.mesh), dtype=np.int64)
    mapping, cap_ids, polygon_rows, points = topology.prepare_identity(
        layers, complex_, vertex_map, cap_ids
    )
    domain, _ = topology.combined_domain(
        layers,
        complex_,
        layer_regions,
        mapping,
        cap_ids,
        polygon_rows,
        points,
        "periodic-layered-gap",
        ("wall-material", "core-material"),
        np.asarray((1,), dtype=np.int64),
        ((), ()),
    )
    source_domain, query = topology.declared_charts(domain, revision)
    source_facets = np.asarray(domain.facets, dtype=np.int64)
    source_sides = np.asarray(domain.facet_regions, dtype=np.int64)
    source_rows = {
        tuple(sorted(int(value) for value in facet)): row
        for row, facet in enumerate(cast(list[list[int]], source_facets.tolist()))
    }
    source_ids: list[str | None] = []
    source_paths: list[tuple[str, ...] | None] = []
    cap_set = set(cap_ids.tolist())
    for index, polygon in enumerate(polygon_rows):
        vertices = tuple(int(value) for value in mapping[polygon])
        row = source_rows.get(tuple(sorted(vertices)))
        if row is None or topology.cycle(vertices) != topology.cycle(
            tuple(int(value) for value in source_facets[row])
        ):
            raise ValueError(
                "A declared ridge core facet changed its source orientation."
            )
        expected = (0, 1) if index in cap_set else (1, -1)
        if tuple(int(value) for value in source_sides[row]) != expected:
            raise ValueError("A declared ridge core facet changed its material sides.")
        source_ids.append(source_domain.entity_id(2, row))
        source_paths.append(tuple(source_domain.source_occurrences[2][row]))
    source = NativeLayerCoreSource(
        layers,
        complex_,
        "periodic-layered-gap",
        revision,
        vertex_layer_ids=vertex_map,
        cap_polygon_ids=cap_ids,
        cap_refinement=cap_refinement,
        noncap_refinement=noncap_refinement,
        layer_control=control
        if "conforming" in (cap_refinement, noncap_refinement)
        else None,
        layer_policy=policy
        if "conforming" in (cap_refinement, noncap_refinement)
        else None,
        layer_regions=layer_regions,
        region_entity_ids=np.arange(2, dtype=np.int64),
        region_occurrence_paths=((), ()),
        region_ids=("wall-material", "core-material"),
        core_region_map=np.asarray((1,), dtype=np.int64),
        source_domain=source_domain,
        fidelity_source=query,
        maximum_deviation=0.0,
        core_facet_source_ids=tuple(source_ids),
        core_facet_source_occurrence_paths=tuple(source_paths),
        core_vertex_representatives=np.concatenate((roots, roots + count)),
        core_vertex_shifts=np.concatenate((shifts, shifts)),
        core_seam_polygon_pairs=pairs,
    )
    return source, control, policy


def _periodic_layer_core_specification(
    source: NativeLayerCoreSource, /, *, tolerance: float = 1.0e-10
) -> Any:
    """Retain fixed core, material and periodic source controls."""
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    pairs = source.core_seam_polygon_pairs
    shifts = source.core_vertex_shifts
    if pairs is None or shifts is None:
        raise ValueError("The original ridge source lost its periodic seam ancestry.")
    rows = np.asarray(source.complex.polygon_vertices, dtype=np.int64).reshape(-1, 3)
    facet_ids = np.asarray(source.complex.facet_entity_ids, dtype=np.int64)[
        np.asarray(source.complex.polygon_facets, dtype=np.int64)
    ]
    first, second = np.asarray(pairs[:, 0]), np.asarray(pairs[:, 1])
    reverse = shifts[rows[first, 0], 0] > shifts[rows[second, 0], 0]
    first, second = np.where(reverse, second, first), np.where(reverse, first, second)
    order = np.argsort(facet_ids[second], kind="stable")
    first_ids, second_ids = facet_ids[first[order]], facet_ids[second[order]]

    def face_scope(ids: np.ndarray) -> Any:
        return phx.meshing.MeshingScope(
            source.source_id,
            source.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            source.scope_entity_set_id(2),
            ids,
        )

    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = 1.0
    periodic = phx.meshing.PeriodicConstraint(
        face_scope(first_ids),
        face_scope(second_ids),
        transform,
        tolerance=tolerance,
    )
    regions = tuple(
        phx.meshing.RegionControl(
            phx.meshing.MeshingScope(
                source.source_id,
                source.source_revision,
                phx.meshing.MeshingEntityKind.GEOMETRY,
                3,
                source.scope_entity_set_id(3),
                np.asarray((source.region_entity_ids[index],), dtype=np.int64),
            ),
            name,
            f"material:{index}",
            phx.meshing.RegionRole.FLUID,
        )
        for index, name in enumerate(source.region_ids)
    )
    cap_scope = face_scope(facet_ids[np.asarray(source.cap_polygon_ids, dtype=np.int64)])
    return phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allowed_transitions=("hexahedron", "pyramid"),
                allow_mixed=True,
            ),
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 2.0, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        region_controls=regions,
        patch_controls=(
            phx.meshing.PatchControl("material-contact", cap_scope, source.region_ids),
        ),
        periodic_constraints=(periodic,),
        maximum_deviation=0.0,
    )


def _certified_block_regions(result: Any, /) -> np.ndarray:
    certificate = result.certification
    if (
        certificate is None
        or not certificate.passed
        or certificate.inputs.cell_regions is None
    ):
        raise ValueError("Certified cell-region ancestry is required.")
    labels = np.asarray(certificate.inputs.cell_regions, dtype=np.int64)
    block_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )
    certified_ids = np.asarray(certificate.inputs.cell_global_ids, dtype=np.int64)
    if (
        labels.shape != block_ids.shape
        or np.unique(block_ids).size != block_ids.size
        or not np.array_equal(certified_ids, block_ids)
    ):
        raise ValueError(
            "Certified region rows must bind the exact block-order cell bank."
        )
    return labels


def _layer_core_region_inventory(
    regions: NDArray[np.int64],
    content: NDArray[np.float64],
    region_count: int,
    /,
) -> NDArray[np.float64]:
    """Sum physical content without cell-count-dependent ledger roundoff."""
    return np.asarray(
        [math.fsum(content[regions == region]) for region in range(region_count)],
        dtype=np.float64,
    )


def _layer_core_density(
    result: Any, controls: _LayerCoreLifecycle, /
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Initialize the original FV density on certified cell regions."""
    from phydrax.discretization.finite_volume import UnstructuredFiniteVolumePlan

    region_count = len(controls.densities)
    labels = _certified_block_regions(result)
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )
    if labels.shape != cell_ids.shape or np.any((labels < 0) | (labels >= region_count)):
        raise ValueError("Certified material labels must cover the original regions.")
    label_by_id = dict(zip(cell_ids.tolist(), labels.tolist(), strict=True))
    if result.geometry is None:
        raise ValueError("Original FV density requires its declared coordinate geometry.")
    volume = UnstructuredFiniteVolumePlan.from_cell_mesh(
        result.mesh,
        coordinate_spec=result.geometry,
        embedding=result.certification.embedding,
        measure_relative_tolerance=1.0e-13 / 16.0,
    ).prepare()
    fv_ids = np.asarray(volume.cell_global_ids, dtype=np.int64)
    try:
        regions = np.asarray(
            [label_by_id[int(identifier)] for identifier in fv_ids], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("The FV carrier lost a certified cell identity.") from error
    density = np.asarray(controls.densities, dtype=np.float64)[regions]
    measures = np.asarray(volume.cell_volumes, dtype=np.float64)
    if measures.shape != density.shape or np.any(measures <= 0.0):
        raise ValueError("Original FV density requires positive accepted cell measures.")
    inventories = _layer_core_region_inventory(regions, measures * density, region_count)
    return volume, density, inventories


def _layer_core_initial_history(
    result: Any,
    volume: Any,
    density: np.ndarray,
    controls: _LayerCoreLifecycle,
    /,
) -> tuple[Any, Any, float]:
    """Solve the original harmonic state and commit its material/damage inputs."""
    accepted, space, defect, _ = _layer_core_harmonic_history(
        result, volume, density, controls, ()
    )
    return accepted, space, defect


# Declared per-chunk geometry/Hodge-assembly workset of the de Rham complexes:
# 512 MiB is the workset validated on the unit-box target (38,696 cells;
# geometry setup peak 7.95 GiB). It only changes chunking, never values.
_LAYER_CORE_HODGE_WORKSET = 512 << 20


def _layer_core_hodge_resources(
    controls: _LayerCoreLifecycle, /
) -> phx.linalg.SolveResourcePolicy:
    """The profile's declared budget of every de Rham Hodge admission and solve."""
    budget = controls.solve_memory_bytes
    return phx.linalg.SolveResourcePolicy(
        factorization_bytes=budget,
        workspace_bytes=_LAYER_CORE_HODGE_WORKSET,
        krylov_basis_bytes=budget,
        preconditioner_bytes=budget,
        recycling_state_bytes=budget,
    )


def _harmonic_history_policy(budget: int, /) -> phx.linalg.LinearSolvePolicy:
    """PCG on the exact supernodal Cholesky factor under one declared byte budget.

    Every solve and factorization budget is the profile's declared execution
    budget: all fields are static, so they enter plan_id, and the writer and
    the cold reader prepare under the same declaration. The factor's
    structural nonzeros are bounded by the same bytes (one float64 each).
    Nested dissection is the ordering. Its separator capacity follows the
    declared factor nonzeros, so the unit-box target's 3,690-vertex separator
    is admitted. On the refined archived-layer-core operator (238,387 P2 DOFs)
    it needs 70,816,866 factor nonzeros (1.10 GB), 6.3e10 flops and a 14.1 s
    symbolic analysis. Minimum degree needs 88,052,378 (1.25 GB), 1.0e11 flops
    and 21.0 s.
    """
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.PCG(),
        tolerance=phx.linalg.TolerancePolicy(relative=1.0e-12, absolute=1.0e-14),
        preconditioning=phx.linalg.PreconditioningPolicy(
            phx.linalg.SparseFactorizationPreconditionerBuilder(
                phx.linalg.SparseFactorizationPolicy(
                    "cholesky",
                    ordering="nested-dissection",
                    max_factor_nnz=budget // np.dtype(np.float64).itemsize,
                    max_factor_bytes=budget,
                    max_symbolic_work=_SYMBOLIC_WORK_BUDGET,
                )
            )
        ),
        resources=phx.linalg.SolveResourcePolicy(
            factorization_bytes=budget,
            workspace_bytes=budget,
            krylov_basis_bytes=budget,
            preconditioner_bytes=budget,
            recycling_state_bytes=budget,
        ),
    )


def _layer_core_harmonic_history(
    result: Any,
    volume: Any,
    density: np.ndarray,
    controls: _LayerCoreLifecycle,
    offsets: tuple[float, ...],
    /,
) -> tuple[Any, Any, float, tuple[Any, ...]]:
    """Solve the harmonic state and its affine Dirichlet ``offsets`` on one operator.

    The space, the compiled problem and its sparse operator are prepared once; the
    accepted state (offset 0) and every offset state differ only in the Dirichlet
    lift, so one sparse LU factor solves them as one multi-RHS block. The Q2
    diffusion form is SPD, but its assembled entries are symmetric only to
    rounding (transposed pairs differ by an ulp), and a Cholesky factor reads one
    entry per symmetric pair; LU factors the assembled operator itself. A direct
    solve leaves no iterative stopping error between the solution and the 1e-9
    affine oracle. Returns the accepted state, its space, the largest
    affine-oracle defect and the expanded ``u`` values of each offset.
    """
    a, b, c = (float(value) for value in controls.harmonic)
    elements = {
        block.name: phx.discretization.lagrange_element(block.cell_kind, 2)
        for block in result.mesh.blocks
    }
    fields = (
        phx.discretization.FiniteElementFieldSpec("u", elements),
        phx.discretization.FiniteElementFieldSpec("damage-history", elements),
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, fields, coordinate_spec=result.geometry
    ).prepare()

    def affine(offset: float) -> Any:
        return lambda points: (
            a * points[..., 0] + b * points[..., 1] + c * points[..., 2] + offset
        )

    constraint = phx.discretization.dirichlet_constraint(space, "u")
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "layer-core-harmonic-diffusion",
            "u",
            (phx.equations.DiffusionAction("u", 1.0),),
            # The Dirichlet-constrained P2 Laplacian is symmetric positive definite.
            properties=phx.linalg.OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={"positive_definite": "asserted"},
            ),
        ),
        space,
        constraint=constraint,
        dirichlet_values=affine(0.0),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(realization="sparse"),
    )
    system, rhs = problem.linear_system(storage="sparse")
    if not isinstance(rhs, jnp.ndarray):
        raise TypeError(
            "The harmonic history operator requires an array right-hand side."
        )
    contexts = (
        None,
        *(
            phx.equations.FiniteElementExecutionContext(
                space.default_runtime, lift=constraint.lift(affine(offset))
            )
            for offset in offsets
        ),
    )
    solved = phx.linalg.solve(
        system,
        jnp.stack(
            (rhs, *(problem.right_hand_side(context) for context in contexts[1:])),
            axis=-1,
        ),
        policy=_harmonic_history_policy(controls.solve_memory_bytes),
        rhs_layout=phx.linalg.RHSLayout((len(contexts),)),
    )
    if not bool(np.all(np.asarray(solved.successful))):
        raise ValueError("The original layer/core harmonic direct solve failed.")
    states = tuple(solved.value[:, index] for index in range(len(contexts)))
    del solved, rhs
    points = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    oracle = a * points[:, 0] + b * points[:, 1] + c * points[:, 2]
    solutions = []
    defect = 0.0
    for state, offset, context in zip(states, (0.0, *offsets), contexts, strict=True):
        expanded = problem.expand(state, context)
        column_defect = float(
            np.max(np.abs(np.asarray(expanded, dtype=np.float64) - (oracle + offset)))
        )
        if not np.isfinite(column_defect) or column_defect > 1.0e-9:
            raise ValueError(
                "The original layer/core harmonic solution failed its affine oracle."
            )
        defect = max(defect, column_defect)
        solutions.append(expanded)
    del states
    values = solutions[0]
    damage = space.project(
        "damage-history",
        lambda point, _: 0.5 + 0.125 * point[..., 1] + 0.1 * point[..., 2],
    )
    damage_values = np.asarray(damage, dtype=np.float64)
    if not np.all(np.isfinite(damage_values)) or np.any(
        (damage_values < 0.0) | (damage_values > 1.0)
    ):
        raise ValueError(
            "The original layer/core damage field is outside its physical bounds."
        )
    fv_ids = np.asarray(volume.cell_global_ids, dtype=np.int64)
    density_by_id = dict(zip(fv_ids.tolist(), density.tolist(), strict=True))
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )
    try:
        material_density = np.asarray(
            [density_by_id[int(identifier)] for identifier in cell_ids],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ValueError(
            "The original material state lost a finite-volume cell."
        ) from error
    material = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId(controls.site_id),
                "physical-cell-average-density",
                material_density,
            ),
        )
    )
    layouts = tuple(
        phx.solver.FiniteElementAcceptedFieldLayout(
            field,
            transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.NESTED_FE,
        )
        for field in fields
    )
    accepted = phx.solver.FiniteElementAcceptedState(
        (values, damage),
        0.0,
        0,
        result.mesh.topology_id,
        space.prepared_id,
        problem.compilation_id,
        field_layouts=layouts,
        materials=material,
    )
    return accepted, space, defect, tuple(solutions[1:])


def _layer_core_history_transition(
    adaptation: Any,
    accepted: Any,
    source_space: Any,
    controls: _LayerCoreLifecycle,
    /,
) -> tuple[Any, Any, dict[str, object]]:
    """Transfer and atomically certify the original affine history/material state."""
    from phydrax.discretization.fem._topology_transfer import nested_nodal_transfer
    from phydrax.discretization.finite_volume import (
        prepare_unstructured_conservative_remap,
    )
    from phydrax.geometry._supermesh import CommonRefinementPolicy

    if adaptation.transition is None or adaptation.transition.geometry_transition is None:
        raise ValueError("History continuation requires the exact geometry transition.")
    source_volume, _, _ = _layer_core_density(adaptation.source, controls)
    target_volume, target_declared_density, _ = _layer_core_density(
        adaptation.target, controls
    )
    expected, target_space, _, (changed_values,) = _layer_core_harmonic_history(
        adaptation.target, target_volume, target_declared_density, controls, (0.1,)
    )
    source_certificate = adaptation.source.certification
    target_certificate = adaptation.target.certification
    if (
        source_certificate is None
        or target_certificate is None
        or source_certificate.inputs.cell_regions is None
        or target_certificate.inputs.cell_regions is None
        or accepted.materials is None
    ):
        raise ValueError(
            "History continuation requires retained material-region authority."
        )
    source_ids = np.concatenate(
        [
            np.asarray(block.global_ids, dtype=np.int64)
            for block in adaptation.source.mesh.blocks
        ]
    )
    target_ids = np.concatenate(
        [
            np.asarray(block.global_ids, dtype=np.int64)
            for block in adaptation.target.mesh.blocks
        ]
    )
    source_fv_ids = np.asarray(source_volume.cell_global_ids, dtype=np.int64)
    target_fv_ids = np.asarray(target_volume.cell_global_ids, dtype=np.int64)
    if not np.array_equal(source_ids, source_fv_ids) or not np.array_equal(
        target_ids, target_fv_ids
    ):
        raise ValueError(
            "Original FV material rows must match the accepted cell identities."
        )
    source_regions = _certified_block_regions(adaptation.source)
    target_regions = _certified_block_regions(adaptation.target)
    old_material = accepted.materials.state(controls.site_id)
    old_density = np.asarray(old_material.committed, dtype=np.float64)
    if old_density.shape != source_regions.shape:
        raise ValueError("The committed density lost its source cell layout.")
    remap = prepare_unstructured_conservative_remap(
        source_volume,
        target_volume,
        provenance=f"{controls.semantics}-layer-core-history",
        policy=CommonRefinementPolicy(coverage_tolerance=1.0e-13),
        geometry_transition=adaptation.transition.geometry_transition,
        source_embedding=source_certificate.embedding,
        target_embedding=target_certificate.embedding,
    )
    if not remap.succeeded or remap.plan is None:
        raise ValueError("The original material overlap was not fully certified.")
    overlap = remap.refinement
    if np.any(
        source_regions[np.asarray(overlap.source_cells, dtype=np.int64)]
        != target_regions[np.asarray(overlap.target_cells, dtype=np.int64)]
    ):
        raise ValueError("The physical FV remap crosses an original material interface.")
    carried = np.asarray(remap.plan.apply(old_density), dtype=np.float64)
    source_measures = np.asarray(source_volume.cell_volumes, dtype=np.float64)
    target_measures = np.asarray(target_volume.cell_volumes, dtype=np.float64)
    region_count = len(controls.densities)
    source_inventory = _layer_core_region_inventory(
        source_regions, source_measures * old_density, region_count
    )
    target_inventory = _layer_core_region_inventory(
        target_regions, target_measures * carried, region_count
    )
    inventory_defect = float(np.max(np.abs(target_inventory - source_inventory)))
    arbitrary_content = (
        1.0
        + (4.0 - 1.0) * np.random.default_rng(53).random(size=(len(source_measures), 1))
    ) * source_measures[:, None]
    arbitrary_defect = float(
        abs(
            np.sum(np.asarray(remap.plan.apply_content(arbitrary_content)))
            - np.sum(arbitrary_content)
        )
    )
    content_scale = max(1.0, float(np.sum(np.abs(source_measures * old_density))))
    if (
        not np.all(np.isfinite(carried))
        or max(inventory_defect, arbitrary_defect) > 1.0e-13 * content_scale
        or np.max(np.abs(carried - target_declared_density)) > 1.0e-12
    ):
        raise ValueError("Original material density or per-region inventory changed.")
    geometry_transition = adaptation.transition.geometry_transition

    def transfer_field(field: Any, layout: Any, context: Any, args: Any) -> Any:
        del context, args
        operator = nested_nodal_transfer(
            source_space,
            target_space,
            geometry_transition,
            layout.name,
            semantics="interpolate",
        )
        return operator.apply(field)

    def transfer_material(materials: Any, lineage: Any, args: Any) -> Any:
        del lineage, args
        original = materials.state(controls.site_id)
        if not np.array_equal(np.asarray(original.committed), old_density):
            raise ValueError(
                "The material callback did not receive the accepted source bank."
            )
        return phx.equations.MaterialTransaction(
            (
                phx.equations.MaterialState(
                    original.site_id,
                    original.model_id,
                    carried,
                    state_version=original.state_version + 1,
                ),
            )
        )

    def certification(expected_values: Any) -> Any:
        def certify(
            mesh: Any, fields: Any, materials: Any, lineage: Any, args: Any
        ) -> bool:
            del mesh, lineage, args
            if materials is None:
                return False
            solution = np.asarray(fields[0], dtype=np.float64)
            damage = np.asarray(fields[1], dtype=np.float64)
            committed = np.asarray(
                materials.state(controls.site_id).committed, dtype=np.float64
            )
            return bool(
                np.all(np.isfinite(solution))
                and np.all(np.isfinite(damage))
                and np.all((damage >= 0.0) & (damage <= 1.0))
                and np.max(np.abs(solution - np.asarray(expected_values))) <= 1.0e-9
                and np.max(np.abs(damage - np.asarray(expected.fields[1]))) <= 1.0e-9
                and np.max(np.abs(committed - carried)) <= 1.0e-12
            )

        return certify

    def transaction(expected_values: Any) -> Any:
        return phx.solver.FiniteElementTopologyTransaction(
            certification(expected_values),
            nested_field_transfer=transfer_field,
            material_transfer=transfer_material,
        )

    refused = transaction(changed_values).execute(
        accepted, adaptation.source.mesh, adaptation
    )
    if (
        bool(np.asarray(refused.committed))
        or refused.state is not accepted
        or (refused.mesh is not adaptation.source.mesh)
    ):
        raise ValueError("A changed PDE must roll back the original accepted state.")
    committed = transaction(expected.fields[0]).execute(
        accepted, adaptation.source.mesh, adaptation
    )
    if not bool(np.asarray(committed.committed)) or committed.state.materials is None:
        raise ValueError("The original history/material transaction did not commit.")
    return (
        committed.state,
        target_space,
        {
            "changed_pde_rollback": True,
            "density_remap_id": remap.plan.plan_id,
            "region_inventory_defect": inventory_defect,
            "arbitrary_content_defect": arbitrary_defect,
        },
    )


def _write_layer_core_cycle_archive(
    path: Path,
    controls: _LayerCoreLifecycle,
    source: Any,
    specification: Any,
    options: Any,
    initial: Any,
    prepared: Any,
    refinement: Any,
    initial_history: Any,
    refined_history: Any,
    initial_fields: Any,
    refined_fields: Any,
    initial_density: np.ndarray,
    refined_density: np.ndarray,
    /,
) -> str:
    """Publish the actual accepted refinement phase and return its content ID."""
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        _layer_core_accepted_lifecycle_recipes,
        layer_core_authoring_recipe,
        model_archive_content_id,
        write_model_artifact,
    )

    if (
        initial.certification is None
        or initial.certification.inputs.fidelity_witness is None
    ):
        raise ValueError(
            "The archived initial result must retain its full source witness."
        )
    if prepared.source.result_id != initial.result_id:
        raise ValueError(
            "The archived refinement must start at its accepted cycle source."
        )
    if source.layer_control is None or source.layer_policy is None:
        raise ValueError("The archived layer/core source lost its authoring controls.")
    authoring = layer_core_authoring_recipe(
        source, source.layer_control, source.layer_policy
    )
    initial_recipe, adaptation, refined_recipe = _layer_core_accepted_lifecycle_recipes(
        initial, prepared, refinement, source, specification
    )
    source_entry = CompositionEntry(
        authoring,
        entry_id="source",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=source.layers.mesh.topology_id,
        revision_id=source.binding_id,
        semantics_id=f"{controls.semantics}-authority",
    )
    specification_entry = CompositionEntry(
        specification,
        entry_id="specification",
        role="model-state",
        owner_id="phydrax.meshing",
        structure_id=source.binding_id,
        revision_id=specification.specification_id,
        semantics_id="original-layer-core-request",
        dependencies=(source_entry.binding("revision"),),
    )
    options_entry = CompositionEntry(
        options,
        entry_id="options",
        role="model-state",
        owner_id="phydrax.meshing",
        structure_id=source.binding_id,
        revision_id=options.options_id,
        semantics_id="original-native-meshing-options",
        dependencies=(source_entry.binding("revision"),),
    )
    initial_result_entry = CompositionEntry(
        initial_recipe,
        entry_id="result/initial",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=initial.mesh.topology_id,
        revision_id=initial.result_id,
        semantics_id="retained-original-accepted-layer-core",
        dependencies=(
            source_entry.binding("revision"),
            specification_entry.binding("revision"),
            options_entry.binding("revision"),
        ),
    )
    adaptation_entry = CompositionEntry(
        adaptation,
        entry_id="refinement",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=initial.mesh.topology_id,
        revision_id=refinement.result_id,
        semantics_id=(
            "periodic-mixed-refinement"
            if initial.mesh.periodic_topology is not None
            else "mixed-refinement"
        ),
        dependencies=(initial_result_entry.binding("revision"),),
    )
    refined_result_entry = CompositionEntry(
        refined_recipe,
        entry_id="result/refined",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=refinement.target.mesh.topology_id,
        revision_id=refinement.target.result_id,
        semantics_id="retained-refined-accepted-layer-core",
        dependencies=(adaptation_entry.binding("revision"),),
    )

    def state_entry(
        name: str,
        value: Any,
        topology_id: str,
        revision_id: str,
        semantics: str,
        dependency: Any,
        /,
    ) -> Any:
        return CompositionEntry(
            value,
            entry_id=name,
            role="physical-state",
            owner_id="phydrax.solver",
            structure_id=topology_id,
            revision_id=revision_id,
            semantics_id=semantics,
            dependencies=(dependency.binding("revision"),),
        )

    history0 = state_entry(
        "history/initial",
        initial_history,
        initial.mesh.topology_id,
        initial_history.accepted_id,
        "harmonic-damage-material-state",
        initial_result_entry,
    )
    history1 = state_entry(
        "history/refined",
        refined_history,
        refinement.target.mesh.topology_id,
        refined_history.accepted_id,
        "harmonic-damage-material-state",
        refined_result_entry,
    )
    fields0 = state_entry(
        "fields/initial",
        initial_fields,
        initial_fields.topology_id,
        canonical_fingerprint(phx.array_tree_fingerprint(initial_fields.fields)),
        "H1-DG-Hcurl-Hdiv-original-banks",
        initial_result_entry,
    )
    fields1 = state_entry(
        "fields/refined",
        refined_fields,
        refined_fields.topology_id,
        canonical_fingerprint(phx.array_tree_fingerprint(refined_fields.fields)),
        "H1-DG-Hcurl-Hdiv-original-banks",
        refined_result_entry,
    )
    fv0 = state_entry(
        "density/initial",
        initial_density,
        initial.mesh.topology_id,
        canonical_fingerprint(phx.array_tree_fingerprint(initial_density)),
        controls.density_semantics,
        initial_result_entry,
    )
    fv1 = state_entry(
        "density/refined",
        refined_density,
        refinement.target.mesh.topology_id,
        canonical_fingerprint(phx.array_tree_fingerprint(refined_density)),
        controls.density_semantics,
        refined_result_entry,
    )
    composition = Composition(
        (
            source_entry,
            specification_entry,
            options_entry,
            initial_result_entry,
            adaptation_entry,
            refined_result_entry,
            history0,
            history1,
            fields0,
            fields1,
            fv0,
            fv1,
        ),
        boundary_id=f"{controls.semantics}-refined-accepted-state",
    )
    limits = _layer_core_archive_limits(controls.archive_bytes)
    write_model_artifact(path, composition, limits=limits)
    return model_archive_content_id(path, limits=limits)


def _write_mapped_refinement_archive(
    path: Path,
    source: Any,
    specification: Any,
    options: Any,
    initial: Any,
    prepared: Any,
    refinement: Any,
    initial_space: Any,
    initial_fields: dict[str, np.ndarray],
    refined_space: Any,
    refined_fields: dict[str, np.ndarray],
    /,
) -> str:
    """Write the actual mapped accepted endpoints, transition, and FE banks."""
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        _layer_core_accepted_lifecycle_recipes,
        finite_element_field_banks_recipe,
        layer_core_authoring_recipe,
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        write_model_artifact,
    )

    names = ("H1", "DG", "Hcurl", "Hdiv")
    if prepared.source.result_id != initial.result_id:
        raise ValueError("The retained mapped transition has another accepted parent.")

    def banks(result: Any, space: Any, values: dict[str, np.ndarray]) -> Any:
        fields = _layer_field_plan(result).fields
        layouts = tuple(
            phx.solver.FiniteElementAcceptedFieldLayout(
                field,
                periodic_quotient=False,
                transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.NESTED_FE,
            )
            for field in fields
        )
        return finite_element_field_banks_recipe(
            space, layouts, tuple(values[name] for name in names)
        )

    initial_banks = banks(initial, initial_space, initial_fields)
    refined_banks = banks(refinement.target, refined_space, refined_fields)
    initial_recipe, adaptation_recipe, refined_recipe = (
        _layer_core_accepted_lifecycle_recipes(
            initial, prepared, refinement, source, specification
        )
    )
    source_entry = CompositionEntry(
        layer_core_authoring_recipe(source, source.layer_control, source.layer_policy),
        entry_id="source",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=source.layers.mesh.topology_id,
        revision_id=source.binding_id,
        semantics_id="original-mapped-layer-core-source",
    )
    specification_entry = CompositionEntry(
        specification,
        entry_id="specification",
        role="model-state",
        owner_id="phydrax.meshing",
        structure_id=source.binding_id,
        revision_id=specification.specification_id,
        semantics_id="original-mapped-controls",
        dependencies=(source_entry.binding("revision"),),
    )
    options_entry = CompositionEntry(
        options,
        entry_id="options",
        role="model-state",
        owner_id="phydrax.meshing",
        structure_id=source.binding_id,
        revision_id=options.options_id,
        semantics_id="original-mapped-native-options",
        dependencies=(source_entry.binding("revision"),),
    )
    initial_entry = CompositionEntry(
        initial_recipe,
        entry_id="result/initial",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=initial.mesh.topology_id,
        revision_id=initial.result_id,
        semantics_id="actual-accepted-mapped-parent",
        dependencies=(
            source_entry.binding("revision"),
            specification_entry.binding("revision"),
            options_entry.binding("revision"),
        ),
    )
    refinement_entry = CompositionEntry(
        adaptation_recipe,
        entry_id="refinement",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=initial.mesh.topology_id,
        revision_id=refinement.result_id,
        semantics_id="actual-mapped-mixed-refinement",
        dependencies=(initial_entry.binding("revision"),),
    )
    refined_entry = CompositionEntry(
        refined_recipe,
        entry_id="result/refined",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=refinement.target.mesh.topology_id,
        revision_id=refinement.target.result_id,
        semantics_id="actual-accepted-mapped-child",
        dependencies=(refinement_entry.binding("revision"),),
    )
    field_entries = tuple(
        CompositionEntry(
            value,
            entry_id=name,
            role="physical-state",
            owner_id="phydrax.solver",
            structure_id=value.topology_id,
            revision_id=canonical_fingerprint(phx.array_tree_fingerprint(value.fields)),
            semantics_id="mapped-H1-DG-Hcurl-Hdiv-banks",
            dependencies=(owner.binding("revision"),),
        )
        for name, value, owner in (
            ("fields/initial", initial_banks, initial_entry),
            ("fields/refined", refined_banks, refined_entry),
        )
    )
    write_model_artifact(
        path,
        Composition(
            (
                source_entry,
                specification_entry,
                options_entry,
                initial_entry,
                refinement_entry,
                refined_entry,
                *field_entries,
            ),
            boundary_id="original-mapped-layer-refinement-state",
        ),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    return model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS)


def _resume_mapped_refinement(path: Path, content_id: str, /) -> dict[str, object]:
    """Independently restore actual mapped state and continue inverse physics."""
    from phydrax.lifecycle import Composition
    from phydrax.meshing._model_codec import (
        _restore_layer_core_accepted_lifecycle,
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_finite_element_field_banks,
        restore_layer_core_authoring,
        restore_mixed_adaptation_policy,
        restore_model_archive_value,
    )

    archive = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
        expected_content_id=content_id,
    )
    specification = phx.meshing.VolumeMeshingSpec.from_stored_fields(
        archive.value("specification")
    )
    source = restore_layer_core_authoring(
        archive.value("source"), specification=specification
    )
    recipe = archive.value("refinement")
    initial, _, refined, adapter = _restore_layer_core_accepted_lifecycle(
        archive.value("result/initial"),
        archive.value("result/refined"),
        recipe,
        source,
        specification,
    )
    policy = restore_mixed_adaptation_policy(recipe.policy, adapter)
    names = ("H1", "DG", "Hcurl", "Hdiv")

    def field_space(result: Any, bank: Any, *, scalar_only: bool = False) -> Any:
        layouts = bank.field_layouts[:2] if scalar_only else bank.field_layouts
        return phx.discretization.FiniteElementPlan(
            result.mesh,
            tuple(layout.rebuild_field_spec() for layout in layouts),
            coordinate_spec=result.geometry,
        ).prepare()

    initial_recipe = archive.value("fields/initial")
    refined_recipe = archive.value("fields/refined")
    if any(
        tuple(layout.name for layout in bank.field_layouts) != names
        for bank in (initial_recipe, refined_recipe)
    ):
        raise ValueError("Cold mapped field layouts changed their original semantics.")
    initial_space = field_space(initial, initial_recipe)
    refined_space = field_space(refined.target, refined_recipe)
    _, initial_values = restore_finite_element_field_banks(initial_recipe, initial_space)
    _, refined_values = restore_finite_element_field_banks(refined_recipe, refined_space)
    initial_fields = _layer_form_fields(initial, initial_space)
    refined_fields = _layer_form_fields(refined.target, refined_space)
    del initial_space, refined_space
    _, scalar_forward, forward_scalars = _layer_scalar_transition(
        refined,
        field_space(initial, initial_recipe, scalar_only=True),
        {name: np.asarray(initial_values[row]) for row, name in enumerate(names[:2])},
        target_space=field_space(refined.target, refined_recipe, scalar_only=True),
    )
    del _
    form_forward, forward_forms = _layer_form_transition(
        refined,
        initial_fields,
        refined_fields,
        {
            name: np.asarray(initial_values[row])
            for row, name in enumerate(names)
            if row >= 2
        },
        # Mapped wall profiles declare no run envelope yet: library defaults.
        hodge_resources=None,
    )
    forward = {**scalar_forward, **form_forward}
    if any(
        not np.allclose(
            forward[name], np.asarray(refined_values[row]), rtol=2e-11, atol=2e-11
        )
        for row, name in enumerate(names)
    ):
        raise ValueError(
            "Cold mapped forward fields differ from the actual accepted banks."
        )
    del initial_fields, refined_fields
    prepared, inverse = _periodic_mixed_coarsening(initial, refined, policy)
    _, scalar_back, backward_scalars = _layer_scalar_transition(
        inverse,
        field_space(refined.target, refined_recipe, scalar_only=True),
        {name: np.asarray(refined_values[row]) for row, name in enumerate(names[:2])},
        target_space=field_space(inverse.target, initial_recipe, scalar_only=True),
    )
    del _
    form_back: dict[str, np.ndarray] = {}
    backward_forms: dict[str, object] = {}
    source_fields = tuple(
        layout.rebuild_field_spec() for layout in refined_recipe.field_layouts
    )
    target_fields = tuple(
        layout.rebuild_field_spec() for layout in initial_recipe.field_layouts
    )
    for name in ("Hcurl", "Hdiv"):
        # Both original declarations are retained; only this named field is prepared.
        values, evidence = _layer_form_transition(
            inverse,
            source_fields,
            target_fields,
            {name: np.asarray(refined_values[names.index(name)])},
            fields=(name,),
            hodge_resources=None,
        )
        form_back.update(values)
        backward_forms.update(evidence)
        del values, evidence
    restored = {**scalar_back, **form_back}
    if any(
        not np.allclose(
            restored[name], np.asarray(initial_values[row]), rtol=2e-11, atol=2e-11
        )
        for row, name in enumerate(names)
    ):
        raise ValueError(
            "Cold mapped coarsening did not restore the actual original fields."
        )
    return {
        "refinement_id": refined.result_id,
        "inverse_request_id": prepared.request.request_id,
        "inverse_result_id": inverse.result_id,
        "forward_scalar_transfer": forward_scalars,
        "forward_compatible_transfer": forward_forms,
        "backward_scalar_transfer": backward_scalars,
        "backward_compatible_transfer": backward_forms,
    }


def _accepted_layer_columns(result: Any, /) -> Any:
    """Recover authored layer columns from the accepted result's own attributes."""
    selected = [
        attribute
        for attribute in result.attributes
        if attribute.name in ("layer_index", "layer_column")
    ]
    attributes = {attribute.name: attribute for attribute in selected}
    if len(selected) != 2 or set(attributes) != {"layer_index", "layer_column"}:
        raise ValueError("Accepted layer/core state must retain both column attributes.")
    interval = attributes["layer_index"]
    column = attributes["layer_column"]
    ids = np.asarray(interval.scope.entity_ids, dtype=np.int64)
    if not np.array_equal(ids, np.asarray(column.scope.entity_ids, dtype=np.int64)):
        raise ValueError("Layer interval and column attributes must share cell ancestry.")
    intervals = np.asarray(interval.values)
    columns = np.asarray(column.values)
    if (
        intervals.dtype.kind not in "iu"
        or columns.dtype.kind not in "iu"
        or intervals.shape != ids.shape
        or columns.shape != ids.shape
    ):
        raise ValueError("Layer attributes must contain one integer per accepted cell.")
    active = intervals >= 0
    if np.any(columns[active] < 0) or np.any(columns[~active] != -1):
        raise ValueError("Layer column markers must agree with the layer/core partition.")
    return phx.meshing.MixedLayerColumns(
        ids[active],
        columns[active],
        intervals[active],
        hard_first_thickness=True,
        axial_refinement=False,
        allow_schedule_change=False,
    )


def _periodic_mixed_refinement(
    accepted: Any,
    columns: Any,
    transfer: Any,
    hierarchy: Any | None,
    /,
    *,
    mark_layer: bool = False,
) -> tuple[Any, Any, Any]:
    """Prepare and execute the original source-bound mixed refinement."""
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
        association_transfer=transfer,
    )
    if mark_layer:
        marked = np.sort(np.asarray(columns.cell_ids, dtype=np.int64))
    else:
        core_blocks = [
            np.asarray(block.global_ids, dtype=np.int64)
            for block in accepted.mesh.blocks
            if block.cell_kind in ("tetrahedron", "simplex:3")
        ]
        if not core_blocks:
            raise ValueError("The original periodic layer/core requires a simplex core.")
        marked = np.sort(np.concatenate(core_blocks))
    if not marked.size:
        raise ValueError("The original refinement requires its authored marked cell.")
    request = phx.meshing.MarkedMeshAdaptation(
        marked[:1],
        hierarchy=hierarchy,
        layer_columns=columns if hierarchy is None else None,
    )
    prepared = phx.meshing.prepare_mesh_adaptation(accepted, request, policy=policy)
    refinement = phx.meshing.execute_mesh_adaptation(prepared)
    if (
        refinement.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
        or not isinstance(refinement.hierarchy, phx.meshing.MixedAdaptationHierarchy)
        or refinement.transition is None
        or refinement.transition.geometry_transition is None
        or refinement.transition.geometry_transition.evidence.kind != "nested_restriction"
        or refinement.transition.geometry_transition.evidence.approximation_bound != 0.0
    ):
        raise ValueError(
            "The original periodic mixed refinement was not exact and complete."
        )
    return prepared, refinement, policy


def _periodic_mixed_coarsening(
    accepted: Any, refinement: Any, policy: Any, /
) -> tuple[Any, Any]:
    """Prepare and execute the complete original inverse sibling coarsening."""
    if not isinstance(refinement.hierarchy, phx.meshing.MixedAdaptationHierarchy):
        raise ValueError("Mixed inverse coarsening requires the retained hierarchy.")
    active = {
        int(identifier)
        for block in refinement.target.mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    children = np.asarray(
        sorted(
            child
            for sibling in refinement.hierarchy.siblings
            for child in sibling.child_ids
            if child in active
        ),
        dtype=np.int64,
    )
    if not children.size:
        raise ValueError("The original mixed refinement lost active sibling ancestry.")
    prepared = phx.meshing.prepare_mesh_adaptation(
        refinement.target,
        phx.meshing.MarkedMeshAdaptation((), children, hierarchy=refinement.hierarchy),
        policy=policy,
    )
    coarsening = phx.meshing.execute_mesh_adaptation(prepared)
    if (
        coarsening.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
        or not isinstance(coarsening.hierarchy, phx.meshing.MixedAdaptationHierarchy)
        or coarsening.transition is None
        or coarsening.transition.geometry_transition is None
        or coarsening.transition.geometry_transition.coarsening_partition is None
        or coarsening.transition.geometry_transition.evidence.approximation_bound != 0.0
        or coarsening.target.mesh.topology_id != accepted.mesh.topology_id
    ):
        raise ValueError(
            "The original periodic inverse coarsening was not exact and complete."
        )
    old_periodic = accepted.mesh.periodic_topology
    restored_periodic = coarsening.target.mesh.periodic_topology
    if (old_periodic is None) != (restored_periodic is None) or (
        old_periodic is not None
        and restored_periodic is not None
        and old_periodic.periodic_topology_id != restored_periodic.periodic_topology_id
    ):
        raise ValueError(
            "Inverse coarsening did not restore the original quotient identity."
        )
    return prepared, coarsening


def _layer_scalar_space(result: Any, /) -> Any:
    from phydrax.discretization.fem import (
        discontinuous_element,
        FiniteElementFieldSpec,
        FiniteElementPlan,
        lagrange_element,
    )

    h1 = {
        block.name: lagrange_element(block.cell_kind, 2) for block in result.mesh.blocks
    }
    dg = {
        block.name: discontinuous_element(block.cell_kind, 1)
        for block in result.mesh.blocks
    }
    return FiniteElementPlan(
        result.mesh,
        (FiniteElementFieldSpec("H1", h1), FiniteElementFieldSpec("DG", dg)),
        coordinate_spec=result.geometry,
    ).prepare()


def _layer_scalar_transition(
    adaptation: Any,
    source_space: Any,
    source_banks: dict[str, np.ndarray],
    /,
    *,
    target_space: Any | None = None,
) -> tuple[Any, dict[str, np.ndarray], dict[str, object]]:
    from phydrax.discretization.fem._topology_transfer import nested_nodal_transfer

    if adaptation.transition is None or adaptation.transition.geometry_transition is None:
        raise ValueError("Scalar continuation needs its exact bound geometry transition.")
    source_certificate = adaptation.source.certification
    target_certificate = adaptation.target.certification
    if (
        source_certificate is None
        or target_certificate is None
        or not source_certificate.passed
        or not target_certificate.passed
        or source_certificate.inputs.cell_regions is None
        or target_certificate.inputs.cell_regions is None
    ):
        raise ValueError("Scalar continuation requires certified material regions.")
    if target_space is None:
        target_space = _layer_scalar_space(adaptation.target)
    transition = adaptation.transition.geometry_transition
    h1 = nested_nodal_transfer(
        source_space, target_space, transition, "H1", semantics="interpolate"
    )
    dg = nested_nodal_transfer(
        source_space,
        target_space,
        transition,
        "DG",
        semantics="density",
        source_cell_classes=_certified_block_regions(adaptation.source),
        target_cell_classes=_certified_block_regions(adaptation.target),
    )
    if not dg.conservative:
        raise ValueError("The original DG material transfer must certify content.")
    banks = {
        "H1": np.asarray(h1.apply(source_banks["H1"]), dtype=np.float64),
        "DG": np.asarray(dg.apply(source_banks["DG"]), dtype=np.float64),
    }
    points = np.asarray(target_space.dof_maps[0].dof_coordinates, dtype=np.float64)
    h1_defect = float(np.max(np.abs(banks["H1"] - (1.0 + points[:, 1] + points[:, 2]))))
    if not np.isfinite(h1_defect) or h1_defect > 1.0e-9:
        raise ValueError("The transferred H1 field lost its original affine oracle.")
    return (
        target_space,
        banks,
        {
            "h1_transfer_id": h1.transfer_id,
            "dg_transfer_id": dg.transfer_id,
            "dg_material_conservative": bool(dg.conservative),
            "h1_affine_defect": h1_defect,
        },
    )


def _layer_field_plan(result: Any, /) -> Any:
    """Retain the original mixed Q2, DG1 and compatible-form declarations."""
    from phydrax.discretization.fem import (
        discontinuous_element,
        FiniteElementFieldSpec,
        FiniteElementPlan,
        form_element,
        lagrange_element,
    )

    kinds = ("H1", "DG", "Hcurl", "Hdiv")
    fields = []
    for name in kinds:
        elements = {}
        for block in result.mesh.blocks:
            cell = block.cell_kind
            if name == "H1":
                element = lagrange_element(cell, 2)
            elif name == "DG":
                element = discontinuous_element(cell, 1)
            else:
                family = (
                    "prism-trimmed"
                    if cell == "prism"
                    else "tensor-trimmed"
                    if cell in ("quadrilateral", "hexahedron")
                    else "trimmed"
                )
                element = form_element(
                    cell,
                    1 if name == "Hcurl" else 2,
                    2,
                    family=family,
                    twist="untwisted" if name == "Hcurl" else "twisted",
                    proxy="circulation" if name == "Hcurl" else "flux",
                )
            elements[block.name] = element
        fields.append(FiniteElementFieldSpec(name, elements))
    return FiniteElementPlan(result.mesh, tuple(fields), coordinate_spec=result.geometry)


def _layer_initial_field_banks(result: Any, /) -> tuple[Any, Any, dict[str, np.ndarray]]:
    """Author the original Q2, DG1 and degree-two compatible coefficient banks."""
    kinds = ("H1", "DG", "Hcurl", "Hdiv")
    plan = _layer_field_plan(result)
    space = plan.prepare()
    banks: dict[str, np.ndarray] = {}
    for index, name in enumerate(kinds):
        dof_map = space.dof_maps[index]
        if name in ("H1", "DG"):
            coordinates = np.asarray(dof_map.dof_coordinates, dtype=np.float64)
            banks[name] = 1.0 + coordinates[:, 1] + coordinates[:, 2]
        else:
            banks[name] = np.random.default_rng(19).standard_normal(
                size=(dof_map.global_dof_count, 2)
            )[:, 0]
    return plan, space, banks


def _layer_form_cells(
    space: Any, field_index: int, /
) -> list[tuple[int, Any, np.ndarray, np.ndarray]]:
    rows = []
    for block, element, routes, transforms in zip(
        space.mesh.blocks,
        space.elements[field_index],
        space.dof_maps[field_index].cell_dofs,
        space.dof_maps[field_index].cell_transforms,
        strict=True,
    ):
        for identifier, route, transform in zip(
            np.asarray(block.global_ids, dtype=np.int64),
            np.asarray(routes, dtype=np.int64),
            np.asarray(transforms, dtype=np.float64),
            strict=True,
        ):
            rows.append((int(identifier), element, route, transform))
    return rows


def _layer_divergence_content(
    space: Any, field_index: int, values: np.ndarray, /
) -> np.ndarray:
    """Integrate H(div) divergence by the canonical exact-degree reference rule."""
    from phydrax.discretization.fem._generic import _degree_aware_reference_rule

    payload = np.asarray(values, dtype=np.float64)
    if payload.ndim == 1:
        payload = payload[:, None]
    total = np.zeros(payload.shape[1:], dtype=np.float64)
    rules: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, element, route, transform in _layer_form_cells(space, field_index):
        if element.element_id not in rules:
            points, weights = _degree_aware_reference_rule(element.cell_kind, 8)
            divergence = np.trace(
                np.asarray(element.tabulate(points)[1]), axis1=-2, axis2=-1
            )
            rules[element.element_id] = (np.asarray(weights), divergence)
        weights, divergence = rules[element.element_id]
        total += np.einsum("q,qn,np->p", weights, divergence, transform @ payload[route])
    return total


def _layer_circulation_reference_basis(
    element: Any,
    points: np.ndarray,
    cache: dict[tuple[int, tuple[int, ...], str, bytes], tuple[Any, np.ndarray]],
    /,
) -> np.ndarray:
    """Reuse one exact tabulation only for its actual element and point bytes."""
    key = (id(element), points.shape, points.dtype.str, points.tobytes())
    cached = cache.get(key)
    if cached is None:
        basis = np.asarray(element.tabulate(points)[0], dtype=np.float64)
        cache[key] = (element, basis)
        return basis
    owner, basis = cached
    if owner is not element:
        raise RuntimeError("Circulation basis owner changed identity.")
    return basis


def _layer_edge_circulation_defect(
    fine: Any,
    coarse: Any,
    fine_index: int,
    coarse_index: int,
    fine_values: np.ndarray,
    coarse_values: np.ndarray,
    partition: Any,
    /,
) -> float:
    """Compare every coarse edge circulation with its exact fine-edge partition."""
    from phydrax.discretization._nested_reference_types import (
        exact_reference_cell_vertices,
    )
    from phydrax.discretization.fem._generic import _degree_aware_reference_rule

    joins: dict[int, tuple[int, tuple[tuple[Fraction, ...], ...]]] = {}
    for witness in partition.maps:
        for child, parent, vertices in zip(
            np.asarray(witness.target_cell_ids, dtype=np.int64),
            np.asarray(witness.source_cell_ids, dtype=np.int64),
            exact_reference_cell_vertices(witness),
            strict=True,
        ):
            if int(child) in joins:
                raise ValueError("A fine cell has multiple coarsening chart parents.")
            joins[int(child)] = (int(parent), vertices)
    fine_payload = np.asarray(fine_values, dtype=np.float64)
    coarse_payload = np.asarray(coarse_values, dtype=np.float64)
    if fine_payload.ndim == 1:
        fine_payload = fine_payload[:, None]
        coarse_payload = coarse_payload[:, None]
    children: dict[int, list[tuple[Any, np.ndarray, np.ndarray, Any]]] = {}
    for identifier, element, route, transform in _layer_form_cells(fine, fine_index):
        if identifier in joins:
            parent, vertices = joins[identifier]
        else:
            parent = identifier
            vertices = tuple(
                tuple(Fraction.from_float(float(value)) for value in row)
                for row in phx.discretization.reference_cell_topology(
                    element.cell_kind
                ).vertices
            )
        children.setdefault(parent, []).append((element, route, transform, vertices))
    nodes, weights = _degree_aware_reference_rule("interval", 8)
    parameters = np.asarray(nodes, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64)
    basis_cache: dict[
        tuple[int, tuple[int, ...], str, bytes], tuple[Any, np.ndarray]
    ] = {}

    def circulation(
        element: Any, coefficients: np.ndarray, first: Any, last: Any, /
    ) -> np.ndarray:
        start = np.asarray(first, dtype=np.float64)
        tangent = np.asarray(last, dtype=np.float64) - start
        points = start + parameters[:, None] * tangent
        basis = _layer_circulation_reference_basis(element, points, basis_cache)
        return np.einsum("q,qnc,np,c->p", weights, basis, coefficients, tangent)

    defect = 0.0
    for identifier, element, route, transform in _layer_form_cells(coarse, coarse_index):
        topology = phx.discretization.reference_cell_topology(element.cell_kind)
        vertices = tuple(
            tuple(Fraction.from_float(float(value)) for value in row)
            for row in topology.vertices
        )
        if identifier not in children:
            raise ValueError("A coarse edge has no retained fine-cell ancestry.")
        for first, last in topology.entities[1]:
            start, end = vertices[first], vertices[last]
            direction = tuple(b - a for a, b in zip(start, end, strict=True))
            axis = next(index for index, value in enumerate(direction) if value != 0)
            intervals: dict[
                tuple[Fraction, Fraction], tuple[Any, np.ndarray, Any, Any, int]
            ] = {}
            for child_element, child_route, child_transform, image in children[
                identifier
            ]:
                child_topology = phx.discretization.reference_cell_topology(
                    child_element.cell_kind
                )
                for a, b in child_topology.entities[1]:
                    p, q = image[a], image[b]
                    t0 = (p[axis] - start[axis]) / direction[axis]
                    t1 = (q[axis] - start[axis]) / direction[axis]
                    if not (0 <= t0 <= 1 and 0 <= t1 <= 1 and t0 != t1):
                        continue
                    if any(
                        p[index] != start[index] + t0 * direction[index]
                        or q[index] != start[index] + t1 * direction[index]
                        for index in range(len(start))
                    ):
                        continue
                    key = (min(t0, t1), max(t0, t1))
                    current = (
                        child_element,
                        child_transform @ fine_payload[child_route],
                        child_topology.vertices[a],
                        child_topology.vertices[b],
                        1 if t1 > t0 else -1,
                    )
                    previous = intervals.get(key)
                    if previous is None:
                        intervals[key] = current
                    else:
                        left = previous[4] * circulation(
                            previous[0], previous[1], previous[2], previous[3]
                        )
                        right = current[4] * circulation(
                            current[0], current[1], current[2], current[3]
                        )
                        if not np.allclose(left, right, rtol=0.0, atol=3.0e-11):
                            raise ValueError(
                                "Shared fine-edge circulation is inconsistent."
                            )
            position = Fraction(0)
            expected = np.zeros(fine_payload.shape[1:], dtype=np.float64)
            for (lower, upper), (child_element, coefficients, a, b, sign) in sorted(
                intervals.items()
            ):
                if lower != position or upper <= lower:
                    raise ValueError(
                        "Fine source edges do not form one exact coarse edge."
                    )
                expected += sign * circulation(child_element, coefficients, a, b)
                position = upper
            if position != 1:
                raise ValueError("Fine source edges do not cover the entire coarse edge.")
            actual = circulation(element, transform @ coarse_payload[route], start, end)
            defect = max(defect, float(np.max(np.abs(actual - expected))))
    return defect


def _layer_form_fields(result: Any, space: Any, /) -> tuple[Any, ...]:
    """Retain original named declarations after full canonical DOF comparison."""
    from phydrax.discretization.fem import FiniteElementFieldSpec
    from phydrax.geometry._mesh_certificates import _same_tree_content

    coordinate_elements, coordinate_dofs, coordinate_values = result.geometry.resolve(
        result.mesh
    )

    if (
        not _same_tree_content(space.mesh, result.mesh)
        or space.default_runtime.geometry_layout_id != result.geometry.geometry_layout_id
        or not _same_tree_content(space.coordinate_elements, coordinate_elements)
        or not _same_tree_content(space.coordinate_dofs, coordinate_dofs)
        or not np.array_equal(
            np.asarray(space.default_runtime.coordinates), np.asarray(coordinate_values)
        )
    ):
        raise ValueError(
            "Compatible field layout does not bind its actual mesh and geometry."
        )
    names = tuple(field.name for field in space.field_spaces)
    if len(set(names)) != len(names) or any(
        name not in names for name in ("Hcurl", "Hdiv")
    ):
        raise ValueError(
            "Compatible field declarations must name Hcurl and Hdiv exactly once."
        )
    result_fields = []
    for name in ("Hcurl", "Hdiv"):
        index = names.index(name)
        elements = space.elements[index]
        dofs = space.dof_maps[index]
        spec = FiniteElementFieldSpec(
            name,
            {
                block.name: element
                for block, element in zip(space.mesh.blocks, elements, strict=True)
            },
            component_shape=dofs.component_shape,
        )
        canonical = phx.discretization.FiniteElementDofMap(
            result.mesh,
            spec.resolve(result.mesh),
            component_shape=spec.component_shape,
            coordinate_spec=result.geometry,
        )
        if not _same_tree_content(canonical, dofs):
            raise ValueError(f"{name} field changed its original DOF layout.")
        result_fields.append(spec)
        del canonical
    return tuple(result_fields)


def _layer_form_action(operator: Any, values: Any, /) -> Any:
    return operator.mv(values)


def _layer_form_transpose(operator: Any, primal: Any, dual: Any, /) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.linear_transpose(operator.mv, jnp.zeros_like(primal))(dual)[0]


def _layer_form_transition(
    adaptation: Any,
    source_fields: Any,
    target_fields: Any,
    banks: dict[str, np.ndarray],
    /,
    *,
    hodge_resources: phx.linalg.SolveResourcePolicy | None,
    fields: tuple[str, ...] = ("Hcurl", "Hdiv"),
    on_field: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Move the original lifted H(curl)/H(div) banks and prove periodic descent.

    ``hodge_resources`` is the profile's declared budget of both complexes'
    Hodge admission factorizations, assembly workset and Riesz solves; ``None``
    keeps the documented library defaults (unbudgeted callers).
    """
    import jax.numpy as jnp

    from phydrax._owned_programs import run_owned_program
    from phydrax.discretization.fem import FiniteElementFieldSpec
    from phydrax.geometry._mesh_certificates import _same_tree_content

    if adaptation.transition is None or adaptation.transition.geometry_transition is None:
        raise ValueError("Compatible fields require their exact geometry transition.")
    degrees = {"Hcurl": 1, "Hdiv": 2}
    if (
        not fields
        or len(set(fields)) != len(fields)
        or any(name not in degrees for name in fields)
    ):
        raise ValueError("Compatible field phases must name distinct Hcurl/Hdiv fields.")
    declarations = []
    for values in (source_fields, target_fields):
        values = tuple(values)
        if any(not isinstance(value, FiniteElementFieldSpec) for value in values):
            raise TypeError(
                "Compatible fields require actual finite-element declarations."
            )
        by_name = {value.name: value for value in values}
        if len(by_name) != len(values) or any(name not in by_name for name in degrees):
            raise ValueError(
                "Compatible declarations must name Hcurl and Hdiv exactly once."
            )
        declarations.append(by_name)
    source_by_name, target_by_name = declarations
    transition = adaptation.transition.geometry_transition
    output: dict[str, np.ndarray] = {}
    records: dict[str, object] = {}

    def run_field(name: str, degree: int, /) -> tuple[np.ndarray, dict[str, object]]:
        twist = "untwisted" if name == "Hcurl" else "twisted"
        source_spec, target_spec = source_by_name[name], target_by_name[name]
        source_complex = phx.discretization.FiniteElementDeRhamComplex(
            adaptation.source.mesh,
            family="trimmed",
            order=2,
            twist=twist,
            coordinate_spec=adaptation.source.geometry,
            hodge_resources=hodge_resources,
        )
        target_complex = phx.discretization.FiniteElementDeRhamComplex(
            adaptation.target.mesh,
            family="trimmed",
            order=2,
            twist=twist,
            coordinate_spec=adaptation.target.geometry,
            hodge_resources=hodge_resources,
        )
        for endpoint, spec, complex_ in (
            (adaptation.source, source_spec, source_complex),
            (adaptation.target, target_spec, target_complex),
        ):
            elements = spec.resolve(endpoint.mesh)
            actual = phx.discretization.FiniteElementDofMap(
                endpoint.mesh,
                elements,
                component_shape=spec.component_shape,
                coordinate_spec=endpoint.geometry,
            )
            expected = complex_.discretization.dof_maps[degree]
            if not _same_tree_content(actual, expected) or not _same_tree_content(
                elements, complex_.discretization.elements[degree]
            ):
                raise ValueError(f"{name} field changed its canonical moment layout.")
            del actual, expected, elements
        # The route follows the meshes' declared periodicity: a periodic pair
        # descends to its quotients, a non-periodic pair uses the nested
        # h/Piola transfer itself.
        source_periodic = adaptation.source.mesh.periodic_topology is not None
        if source_periodic != (adaptation.target.mesh.periodic_topology is not None):
            raise ValueError("Form transfer endpoints must agree on periodicity.")
        if source_periodic:
            periodic = source_complex.transfer_periodic_quotient(
                target_complex, geometry_transition=transition
            )
            route = "periodic-quotient"
            transferred = periodic.lifted_transfer_result
            commuting = np.asarray(periodic.commuting_defects, dtype=np.float64)
            bounds = np.asarray(periodic.commuting_error_bounds, dtype=np.float64)
            transfer_id = periodic.transfer_id
            del periodic
        else:
            transferred = source_complex.transfer(
                target_complex, geometry_transition=transition
            )
            route = "nested"
            commuting = np.asarray(transferred.commuting_defects, dtype=np.float64)
            bounds = np.asarray(transferred.commuting_error_bounds, dtype=np.float64)
            transfer_id = transferred.transfer_id
        action = transferred.complex_map.map(degree)
        evidence_id = transferred.evidence_id
        # All complex, Hodge, descent and commuting checks have completed.
        # Release their owners before the selected action's consumers.
        del transferred, source_complex, target_complex, complex_
        source_values = np.asarray(banks[name], dtype=np.float64)
        target_values = np.asarray(
            action.mv(jnp.asarray(source_values)), dtype=np.float64
        )
        compiled_values = np.asarray(
            run_owned_program(
                action,
                "layer-form-action",
                _layer_form_action,
                jnp.asarray(source_values),
            )
        )
        if not np.allclose(compiled_values, target_values, rtol=2.0e-13, atol=2.0e-13):
            raise ValueError(f"{name} compiled transfer changed the actual coefficients.")
        dual = np.random.default_rng(29).standard_normal(size=target_values.shape)
        transpose = run_owned_program(
            action,
            "layer-form-transpose",
            _layer_form_transpose,
            jnp.asarray(source_values),
            jnp.asarray(dual),
        )
        forward_pairing = float(np.vdot(target_values, dual))
        reverse_pairing = float(np.vdot(source_values, np.asarray(transpose)))
        adjoint_defect = abs(forward_pairing - reverse_pairing)
        if not np.isclose(forward_pairing, reverse_pairing, rtol=2.0e-11, atol=2.0e-11):
            raise ValueError(f"{name} compatible transfer failed its adjoint identity.")
        # The actual commuting values were retained before releasing the owners.
        if (
            not np.all(np.isfinite(target_values))
            or not np.all(np.isfinite(commuting))
            or not np.all(np.isfinite(bounds))
            or np.max(commuting + bounds, initial=0.0) > 3.0e-11
        ):
            raise ValueError(f"{name} compatible transfer did not commute.")
        physical_defect = None
        partition = transition.coarsening_partition

        def physical_space(endpoint: Any, spec: Any) -> Any:
            space = phx.discretization.FiniteElementPlan(
                endpoint.mesh, (spec,), coordinate_spec=endpoint.geometry
            ).prepare()
            elements = spec.resolve(endpoint.mesh)
            canonical = phx.discretization.FiniteElementDofMap(
                endpoint.mesh,
                elements,
                component_shape=spec.component_shape,
                coordinate_spec=endpoint.geometry,
            )
            if not _same_tree_content(
                space.dof_maps[0], canonical
            ) or not _same_tree_content(space.elements[0], elements):
                raise ValueError(f"{name} physical check changed its actual DOF layout.")
            return space

        if name == "Hdiv":
            probe = (
                source_values
                if partition is None
                else np.random.default_rng(47).standard_normal(
                    size=(source_values.size, 2)
                )
            )
            mapped_probe = np.asarray(
                action.mv(jnp.asarray(probe))
                if probe.ndim == 1
                else action.mv_block(jnp.asarray(probe))
            )
            source_physical = physical_space(adaptation.source, source_spec)
            target_physical = physical_space(adaptation.target, target_spec)
            physical_defect = float(
                np.max(
                    np.abs(
                        _layer_divergence_content(target_physical, 0, mapped_probe)
                        - _layer_divergence_content(source_physical, 0, probe)
                    )
                )
            )
        elif partition is not None:
            probe = np.random.default_rng(47).standard_normal(
                size=(source_values.size, 2)
            )
            mapped_probe = np.asarray(action.mv_block(jnp.asarray(probe)))
            source_physical = physical_space(adaptation.source, source_spec)
            target_physical = physical_space(adaptation.target, target_spec)
            physical_defect = _layer_edge_circulation_defect(
                source_physical,
                target_physical,
                0,
                0,
                probe,
                mapped_probe,
                partition,
            )
        if physical_defect is not None and (
            not np.isfinite(physical_defect) or physical_defect > 3.0e-11
        ):
            raise ValueError(f"{name} physical compatible content changed.")
        record = {
            "field_space": "lifted-canonical-dofs"
            if source_periodic
            else "canonical-dofs",
            "route": route,
            "transfer_id": transfer_id,
            "evidence_id": evidence_id,
            "adjoint_defect": adjoint_defect,
            "physical_defect": physical_defect,
            "compiled_defect": float(np.max(np.abs(compiled_values - target_values))),
            "maximum_commuting_bound": float(np.max(commuting + bounds, initial=0.0)),
        }
        return target_values, record

    for name in fields:
        degree = degrees[name]
        output[name], records[name] = run_field(name, degree)
        if on_field is not None:
            on_field(name, output[name], records[name])
    return output, records


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
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicIsometryGroup(
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


def _curved_periodic_profile_geometry(wall: Any, /) -> tuple[Any, Any, tuple[str, ...]]:
    """Author the original quadratic profile and its translated upper graph."""
    element = phx.discretization.lagrange_element("triangle", 2)
    profile = phx.discretization.CellMesh(
        wall.coordinates, wall.blocks, vertex_global_ids=wall.vertex_global_ids
    )
    space = phx.discretization.FiniteElementPlan(
        profile, phx.discretization.FiniteElementFieldSpec("coordinates", element)
    ).prepare()
    route = np.asarray(space.dof_maps[0].cell_dofs[0], dtype=np.int64)
    parameters = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    lower = np.column_stack(
        (
            parameters[:, 0],
            parameters[:, 1],
            2.0 * (1.0 - parameters[:, 1]) * parameters[:, 1] * (3.0 / 16.0),
        )
    )
    orbit_ids = [f"corrected-profile-coefficient:{row}" for row in range(len(lower))]
    period = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    targets = np.flatnonzero(parameters[:, 0] == 1.0)
    if targets.size != 3:
        raise ValueError("The quadratic profile must retain all three seam coefficients.")
    for target in targets:
        matches = np.flatnonzero(
            (parameters[:, 0] == 0.0) & (parameters[:, 1] == parameters[target, 1])
        )
        if matches.size != 1:
            raise ValueError(
                "Each periodic profile coefficient needs one x=0 representative."
            )
        representative = int(matches[0])
        lower[target] = lower[representative] + period
        orbit = f"corrected-profile-x-orbit:{parameters[target, 1].hex()}"
        orbit_ids[representative] = orbit
        orbit_ids[target] = orbit
    upper = lower + np.asarray((0.0, 0.0, 1.0 / 16.0), dtype=np.float64)
    elements = {wall.blocks[0].name: element}
    routes = {wall.blocks[0].name: route}
    return (
        phx.discretization.CellGeometrySpec(elements, routes, lower),
        phx.discretization.CellGeometrySpec(elements, routes, upper),
        tuple(orbit_ids),
    )


def _curved_periodic_root_geometry(
    wall: Any, layers: BoundaryLayerMesh, /
) -> tuple[Any, Any, tuple[str, ...]]:
    from phydrax.meshing._layer_curving import prepare_layer_column_geometry

    cap = layers.cap
    periodic = layers.mesh.periodic_topology
    if cap is None or cap.periodic_topology is None or periodic is None:
        raise ValueError("The original layer and cap must retain periodic ancestry.")
    cap_vertices = np.asarray(layers.cap_vertices, dtype=np.int64)
    source_ids = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)[cap_vertices]
    wall_ids = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    if len(np.unique(source_ids)) != len(wall_ids) or set(source_ids) != set(wall_ids):
        raise ValueError("Each cap vertex must retain one original wall vertex.")
    wall_rows = {int(identifier): row for row, identifier in enumerate(wall_ids)}
    profile_points = np.asarray(wall.coordinates, dtype=np.float64)[
        [wall_rows[int(identifier)] for identifier in source_ids]
    ]
    cap_rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in cap.blocks]
    )
    profile = phx.discretization.CellMesh(
        profile_points,
        (
            phx.discretization.CellBlock(
                "fundamental-profile",
                "triangle",
                cap_rows,
                global_ids=np.arange(400, 400 + len(cap_rows), dtype=np.int64),
            ),
        ),
        vertex_global_ids=source_ids,
    )
    lower, upper, orbits = _curved_periodic_profile_geometry(profile)
    layer_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    upper_ids = int(np.max(layer_ids)) + 1 + np.arange(len(cap_vertices), dtype=np.int64)
    layer_cell_ids = np.asarray(layers.mesh.entity_set(3).entity_ids, dtype=np.int64)
    core_ids = int(np.max(layer_cell_ids)) + 1 + np.arange(len(cap_rows), dtype=np.int64)
    upper_reference = profile_points.copy()
    upper_reference[:, 2] = 1.0 / 16.0
    layer_count = len(layer_ids)
    core_rows = np.concatenate((cap_vertices[cap_rows], layer_count + cap_rows), axis=1)
    plain = phx.discretization.CellMesh(
        np.concatenate((np.asarray(layers.mesh.coordinates), upper_reference)),
        (
            *layers.mesh.blocks,
            phx.discretization.CellBlock(
                "declared-core-roots", "prism", core_rows, global_ids=core_ids
            ),
        ),
        vertex_global_ids=np.concatenate((layer_ids, upper_ids)),
    )
    cap_periodic = cap.periodic_topology
    topology = PeriodicMeshTopology(
        plain,
        periodic.cell,
        np.concatenate(
            (
                np.asarray(periodic.vertex_representatives, dtype=np.int64),
                layer_count
                + np.asarray(cap_periodic.vertex_representatives, dtype=np.int64),
            )
        ),
        np.concatenate(
            (
                np.asarray(periodic.vertex_shifts, dtype=np.int64),
                np.asarray(cap_periodic.vertex_shifts, dtype=np.int64),
            )
        ),
    )
    root = phx.discretization.CellMesh(
        plain.coordinates,
        plain.blocks,
        vertex_global_ids=plain.vertex_global_ids,
        periodic_topology=topology,
    )
    geometry = prepare_layer_column_geometry(
        root, layers, profile, lower, upper, upper_ids
    )
    return root, geometry, orbits


def _curved_periodic_reference_core(
    wall: Any, layers: BoundaryLayerMesh, root: Any, /
) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Describe the original fixed cap, outer facets, and periodic seam orbits."""
    from phydrax.meshing import _layer_core_topology as topology

    cap = layers.cap
    if cap is None or cap.periodic_topology is None:
        raise ValueError("The reference core requires its periodic cap.")
    count = cap.coordinates.shape[0]
    cap_rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in cap.blocks]
    )
    cap_layer_rows = np.asarray(layers.cap_vertices, dtype=np.int64)
    source_ids = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)[
        cap_layer_rows
    ]
    wall_ids = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    wall_rows = {int(identifier): row for row, identifier in enumerate(wall_ids)}
    profile = np.asarray(wall.coordinates, dtype=np.float64)[
        [wall_rows[int(identifier)] for identifier in source_ids]
    ]
    upper = profile.copy()
    upper[:, 2] = 1.0 / 16.0
    core = phx.discretization.CellMesh(
        np.concatenate((np.asarray(cap.coordinates), upper)),
        (
            phx.discretization.CellBlock(
                "reference-core",
                "prism",
                np.concatenate((cap_rows, cap_rows + count), axis=1),
            ),
        ),
    )
    roots = np.asarray(cap.periodic_topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(cap.periodic_topology.vertex_shifts, dtype=np.int64)
    image = np.arange(2 * count, dtype=np.int64)
    for vertex in np.flatnonzero(shifts[:, 0] == 1):
        image[roots[vertex]] = vertex
        image[count + roots[vertex]] = count + vertex
    polygons = [tuple(int(value) for value in row) for row in cap_rows]
    incidence = [(0, -1)] * len(polygons)
    cap_keys = {tuple(sorted(row.tolist())) for row in cap_rows}
    seam_source: list[int] = []
    seam_target: dict[tuple[int, ...], int] = {}
    internal: list[tuple[int, ...]] = []
    core_faces = topology.faces(core, np.zeros(len(cap_rows), dtype=np.int64))
    for key, incidents in core_faces.items():
        if len(incidents) == 2:
            internal.extend(topology.triangles(incidents[0].vertices))
            continue
        if key in cap_keys:
            continue
        face = incidents[0].vertices
        if all(vertex >= count for vertex in face):
            role = "upper"
            triangles = topology.triangles(face)
        else:
            feet = np.asarray(sorted({int(vertex) % count for vertex in face}))
            points = profile[feet]
            roles = [
                name
                for name, axis, value in (
                    ("x0", 0, 0.0),
                    ("x1", 0, 1.0),
                    ("y0", 1, 0.0),
                    ("y1", 1, 1.0),
                )
                if feet.size == 2 and np.all(points[:, axis] == value)
            ]
            if len(roles) != 1:
                raise ValueError("A core side lacks a unique original boundary role.")
            role = roles[0]
            a, b = (int(value) for value in feet)
            if (int(roots[a]), a) > (int(roots[b]), b):
                a, b = b, a
            triangles = ((a, b, count + b), (a, count + b, count + a))
            if topology.cycle((a, b, count + b, count + a)) != topology.cycle(face):
                triangles = tuple(tuple(reversed(row)) for row in triangles)
        for triangle in triangles:
            row = len(polygons)
            polygons.append(tuple(int(value) for value in triangle))
            incidence.append((-1, 0))
            if role == "x0":
                seam_source.append(row)
            elif role == "x1":
                seam_target[tuple(sorted(triangle))] = row
    pairs = np.asarray(
        [
            (row, seam_target[tuple(sorted(image[list(polygons[row])].tolist()))])
            for row in seam_source
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    polygons.extend(internal)
    incidence.extend((0, 0) for _ in internal)
    complex_ = PiecewiseLinearComplex(
        np.asarray(core.coordinates, dtype=np.float64),
        tuple(np.asarray(row, dtype=np.int64) for row in polygons),
        np.arange(len(polygons), dtype=np.int64),
        np.asarray(incidence, dtype=np.int64),
        ("core-material",),
        vertex_ids=tuple(f"periodic-core-vertex:{row}" for row in range(2 * count)),
        vertex_occurrence_paths=((),) * (2 * count),
        facet_ids=tuple(f"periodic-core-facet:{row}" for row in range(len(polygons))),
        facet_occurrence_paths=((),) * len(polygons),
        segments=np.empty((0, 2), dtype=np.int64),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    vertex_layer_ids = np.concatenate(
        (cap_layer_rows, np.full(count, -1, dtype=np.int64))
    )
    cap_ids = np.arange(len(cap_rows), dtype=np.int64)
    mapping, cap_ids, polygon_rows, points = topology.prepare_identity(
        layers, complex_, vertex_layer_ids, cap_ids
    )
    domain, _ = topology.combined_domain(
        layers,
        complex_,
        np.zeros(_count_cells(layers.mesh), dtype=np.int64),
        mapping,
        cap_ids,
        polygon_rows,
        points,
        "curved-periodic-narrow-gap-exact-x-orbits",
        ("wall-material", "core-material"),
        np.asarray((1,), dtype=np.int64),
        ((), ()),
    )
    if not np.array_equal(
        np.asarray(root.coordinates, dtype=np.float64),
        np.asarray(domain.vertices, dtype=np.float64),
    ):
        raise ValueError("The core and layer reference root registry changed.")
    return (
        complex_,
        domain,
        vertex_layer_ids,
        cap_ids,
        np.concatenate((roots, count + roots)),
        np.concatenate((shifts, shifts)),
        pairs,
    )


def _curved_periodic_full_authority(
    wall: Any,
    schedule: Any,
    layers: BoundaryLayerMesh,
    complex_: Any,
    vertex_map: np.ndarray,
    cap_ids: np.ndarray,
    mapped: Any,
    /,
) -> tuple[Any, Any, tuple[str | None, ...], tuple[tuple[str, ...] | None, ...]]:
    """Author the full source independently from its wall and column schedule."""
    from phydrax.meshing import _layer_core_topology as topology
    from phydrax.meshing.providers._native_sources import NativePlcSource

    root = mapped.reference_mesh
    root_ids = np.asarray(root.vertex_global_ids, dtype=np.int64)
    layer_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    layer_count = len(layer_ids)
    if not np.array_equal(root_ids[:layer_count], layer_ids):
        raise ValueError("Mapped roots changed the original layer vertex registry.")
    wall_ids = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    wall_points = np.asarray(wall.coordinates, dtype=np.float64)
    wall_rows = {int(identifier): row for row, identifier in enumerate(wall_ids)}
    predecessors: dict[int, int] = {}
    for block in layers.mesh.blocks:
        if block.cell_kind != "prism":
            raise ValueError("The original source has directed prism columns.")
        for row in np.asarray(block.vertices, dtype=np.int64):
            for first, last in zip(row[:3], row[3:], strict=True):
                previous = predecessors.setdefault(int(last), int(first))
                if previous != int(first):
                    raise ValueError("A layer vertex has conflicting source ancestry.")
    depths = {int(row): 0 for row in np.asarray(layers.wall_vertices) if row >= 0}
    for row in range(layer_count):
        chain: list[int] = []
        current = row
        while current not in depths:
            if current in chain or current not in predecessors:
                raise ValueError("A layer vertex lost its authored column ancestry.")
            chain.append(current)
            current = predecessors[current]
        for child in reversed(chain):
            depths[child] = depths[current] + 1
            current = child
    source_ids = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)
    thicknesses = tuple(float(value) for value in schedule.thicknesses)
    heights = (0.0, *np.cumsum(thicknesses).tolist())
    if thicknesses != (0.01, 0.01):
        raise ValueError("The original periodic source requires its two 0.01 layers.")
    points = np.empty((len(root_ids), 3), dtype=np.float64)
    vertex_keys: list[str] = [""] * len(root_ids)
    by_column: dict[tuple[int, int], int] = {}
    for row, source_id in enumerate(source_ids.tolist()):
        depth = depths[row]
        if source_id not in wall_rows or depth >= len(heights):
            raise ValueError("A source column escaped its original wall or schedule.")
        key = (source_id, depth)
        if key in by_column:
            raise ValueError("The original source column has duplicate level vertices.")
        by_column[key] = row
        points[row] = wall_points[wall_rows[source_id]] + (0.0, 0.0, heights[depth])
        vertex_keys[row] = f"wall:{source_id}:level:{depth}"
    cap_rows = np.asarray(layers.cap_vertices, dtype=np.int64)
    if len(root_ids) != layer_count + len(cap_rows):
        raise ValueError("The original source upper-root registry is incomplete.")
    for offset, cap_row in enumerate(cap_rows):
        source_id = int(source_ids[cap_row])
        row = layer_count + offset
        key = (source_id, len(heights))
        if key in by_column or depths[int(cap_row)] != len(thicknesses):
            raise ValueError("An upper root has inconsistent cap ancestry.")
        by_column[key] = row
        points[row] = wall_points[wall_rows[source_id]] + (0.0, 0.0, 1.0 / 16.0)
        vertex_keys[row] = f"wall:{source_id}:upper"
    if not np.array_equal(
        points.view(np.uint64),
        np.asarray(root.coordinates, dtype=np.float64).view(np.uint64),
    ):
        raise ValueError(
            "Mapped root vertices differ from the independently authored source."
        )
    published_cells = {
        tuple(int(value) for value in row): int(identifier)
        for block in root.blocks
        for identifier, row in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        )
    }
    triangles = np.concatenate(
        [wall_ids[np.asarray(block.vertices, dtype=np.int64)] for block in wall.blocks]
    )
    cells: list[tuple[int, ...]] = []
    cell_ids: list[int] = []
    cell_regions: list[int] = []
    for level in range(len(heights)):
        for triangle in triangles:
            row = tuple(
                by_column[(int(identifier), depth)]
                for depth in (level, level + 1)
                for identifier in triangle
            )
            if row not in published_cells:
                raise ValueError(
                    "A root prism differs from the authored wall/schedule product."
                )
            cells.append(row)
            cell_ids.append(published_cells[row])
            cell_regions.append(0 if level < len(thicknesses) else 1)
    if len(cells) != len(published_cells):
        raise ValueError("Mapped roots contain cells absent from the original source.")
    authored_root = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "authored-source-columns",
                "prism",
                np.asarray(cells, dtype=np.int64),
                global_ids=np.asarray(cell_ids, dtype=np.int64),
            ),
        ),
        vertex_global_ids=root_ids,
    )
    cap = layers.cap
    if cap is None or cap.periodic_topology is None:
        raise ValueError("The authored source needs its periodic cap orbits.")
    cap_roots = np.asarray(cap.periodic_topology.vertex_representatives, dtype=np.int64)
    cap_by_source = {
        int(source_ids[int(row)]): index for index, row in enumerate(cap_rows)
    }
    face_table = topology.faces(authored_root, np.asarray(cell_regions, dtype=np.int64))
    facets: list[tuple[int, int, int]] = []
    facet_sides: list[tuple[int, int]] = []
    for incidents in face_table.values():
        first = incidents[0]
        other = -1 if len(incidents) == 1 else incidents[1].region
        if first.region == other:
            continue
        if first.region == 1 and other == -1 and len(first.vertices) == 4:
            bottoms = [int(row) for row in first.vertices if int(row) < layer_count]
            if len(bottoms) != 2:
                raise ValueError("A core side must join two authored cap columns.")
            bottoms.sort(
                key=lambda row: (
                    int(cap_roots[cap_by_source[int(source_ids[row])]]),
                    cap_by_source[int(source_ids[row])],
                )
            )
            a, b = bottoms
            upper_a = by_column[(int(source_ids[a]), len(heights))]
            upper_b = by_column[(int(source_ids[b]), len(heights))]
            side_triangles = ((a, b, upper_b), (a, upper_b, upper_a))
            if topology.cycle((a, b, upper_b, upper_a)) != topology.cycle(first.vertices):
                side_triangles = tuple(tuple(reversed(row)) for row in side_triangles)
        else:
            side_triangles = topology.triangles(first.vertices)
        for triangle in side_triangles:
            facets.append(cast(tuple[int, int, int], triangle))
            facet_sides.append((other, first.region))
    facet_ids = tuple(
        "physical-facet:" + ":".join(sorted(vertex_keys[row] for row in triangle))
        for triangle in facets
    )
    full = PiecewiseLinearComplex(
        points,
        tuple(np.asarray(row, dtype=np.int64) for row in facets),
        np.arange(len(facets), dtype=np.int64),
        np.asarray(facet_sides, dtype=np.int64),
        ("wall-material", "core-material"),
        vertex_ids=tuple(vertex_keys),
        vertex_occurrence_paths=((),) * len(points),
        facet_ids=facet_ids,
        facet_occurrence_paths=((),) * len(facets),
        segments=np.empty((0, 2), dtype=np.int64),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
        boundary="fixed",
    )
    authority = NativePlcSource(
        full,
        mapped.source_id,
        mapped.source_revision,
        mapped_domain=mapped,
        plc_vertex_root_global_ids=root_ids,
    )
    query = authority.boundary_query()
    mapping, _, polygon_rows, _ = topology.prepare_identity(
        layers, complex_, vertex_map, cap_ids
    )
    full_rows = {
        tuple(sorted(int(value) for value in facet)): row
        for row, facet in enumerate(facets)
    }
    identities: list[str | None] = []
    occurrences: list[tuple[str, ...] | None] = []
    for polygon in polygon_rows:
        row = full_rows.get(tuple(sorted(mapping[polygon].tolist())))
        identities.append(None if row is None else facet_ids[row])
        occurrences.append(None if row is None else ())
    return authority, query, tuple(identities), tuple(occurrences)


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
        result.mesh, phx.meshing.evaluate_cell_quality(result.mesh)
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
    layers: BoundaryLayerMesh, control: Any, /, *, region: str = "fluid"
) -> NativeLayerCoreSource:
    cap = layers.cap
    if cap is None or not layers.closed_cap:
        raise ValueError("This source profile did not author a closed fixed core cap.")
    source_id = str(layers.source_wall.mesh_id)
    revision = str(layers.source_wall.numeric_version)
    from phydrax.meshing._layer_core_topology import closed_fixed_core_source
    from phydrax.meshing._plc_source import _numeric_entity_ids

    return closed_fixed_core_source(
        layers,
        None,
        SpatialCoordinateContract.si(),
        source_id,
        revision,
        region_id=region,
        region_entity_id=int(_numeric_entity_ids("plc-region", (region,), ((),))[0]),
        region_occurrence_path=(),
        wall_occurrence_path=(),
        outer_occurrence_path=None,
        layer_control=control,
        layer_policy=phx.meshing.BoundaryLayerPolicy(),
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


def _mimetic_affine_field(mesh: Any, budget: int, /) -> dict[str, object]:
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
    solved = system.solve(
        1.0,
        boundary,
        policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.GMRES(restart=40, stagnation_iterations=40),
            tolerance=phx.linalg.TolerancePolicy(relative=1.0e-12, absolute=1.0e-14),
            resources=phx.linalg.SolveResourcePolicy(
                factorization_bytes=budget,
                workspace_bytes=budget,
                krylov_basis_bytes=budget,
                preconditioner_bytes=budget,
                recycling_state_bytes=budget,
            ),
        ),
    )
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
        # The hybrid solve carries one trace unknown per facet, not per face.
        "facet_unknowns": int(system.facet_count),
        "successful": bool(np.all(np.asarray(solved.successful, dtype=np.bool_))),
        "exact_field": "u=1+0.2x-0.3y+0.4z",
        "maximum_affine_error": maximum_error,
        "normalized_residual": normalized,
        "tolerance": 2.0e-8,
    }


def _run_wall_profile(
    name: str, /, *, mapped_archive_dir: Path | None = None
) -> dict[str, object]:
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
        if mapped_archive_dir is None:
            raise ValueError("Mapped lifecycle requires its independent cold supervisor.")
        wall, metadata = _curved_periodic_wall()
        metadata["source_profile"] = name
        metadata["physical_input"] = (
            "exact-x-orbit fundamental wall carrier; source quadratic profile points retained"
        )
        metadata["original_resolution_scale"] = 2 if name.endswith(":larger") else 1
        metadata["minimum_adaptation_rounds"] = 2 if name.endswith(":larger") else 0
        schedule = phx.meshing.LayerSchedule.geometric(2, 0.01, growth_rate=1.0)
        growth = 1.0
        control = _control(
            wall,
            schedule,
            corner=phx.meshing.BoundaryLayerCornerPolicy.SMOOTH,
            feature_angle=0.1,
            growth_rate_bounds=(0.9999999999, 1.001),
        )
        layer_policy = phx.meshing.BoundaryLayerPolicy()
        layers = phx.meshing.prepare_boundary_layers(wall, control, policy=layer_policy)
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
            prepared_layers=layers,
        )
        root, geometry, coefficient_orbits = _curved_periodic_root_geometry(wall, layers)
        from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain

        complex_, domain, vertex_map, cap_ids, roots, shifts, seams = (
            _curved_periodic_reference_core(wall, layers, root)
        )
        revision = canonical_fingerprint(
            {
                "kind": "curved-periodic-layer-core",
                "coordinate_contract": SpatialCoordinateContract.si().spatial_id,
                "wall_coordinates": phx.array_tree_fingerprint(
                    np.asarray(wall.coordinates, dtype=np.float64)
                ),
                "wall_triangles": phx.array_tree_fingerprint(
                    np.asarray(wall.blocks[0].vertices, dtype=np.int64)
                ),
                "layer_thicknesses": schedule.thicknesses,
                "profile_height": 3.0 / 16.0,
                "profile_gap": 1.0 / 16.0,
                "period": (1.0, 0.0, 0.0),
                "regions": ("wall-material", "core-material"),
            }
        )
        mapped = MappedReferenceDomain(
            domain,
            root,
            geometry,
            np.concatenate(
                (
                    np.zeros(_count_cells(layers.mesh), dtype=np.int64),
                    np.ones(len(cap_ids), dtype=np.int64),
                )
            ),
            source_id=domain.source_id,
            source_revision=revision,
            reference_domain_vertex_root_ids=np.asarray(root.vertex_global_ids),
            coordinate_contract=SpatialCoordinateContract.si(),
        )
        authority, query, source_ids, source_paths = _curved_periodic_full_authority(
            wall, schedule, layers, complex_, vertex_map, cap_ids, mapped
        )
        reference = NativeLayerCoreSource(
            layers,
            complex_,
            domain.source_id,
            revision,
            vertex_layer_ids=vertex_map,
            cap_polygon_ids=cap_ids,
            cap_refinement="conforming",
            noncap_refinement="conforming",
            layer_control=control,
            layer_policy=layer_policy,
            layer_regions=np.zeros(_count_cells(layers.mesh), dtype=np.int64),
            region_entity_ids=np.arange(2, dtype=np.int64),
            region_occurrence_paths=((), ()),
            region_ids=domain.region_ids,
            core_region_map=np.asarray((1,), dtype=np.int64),
            core_facet_source_ids=source_ids,
            core_facet_source_occurrence_paths=source_paths,
            core_vertex_representatives=roots,
            core_vertex_shifts=shifts,
            core_seam_polygon_pairs=seams,
        )
        source = NativeLayerCoreSource(
            layers,
            complex_,
            domain.source_id,
            revision,
            vertex_layer_ids=vertex_map,
            cap_polygon_ids=cap_ids,
            cap_refinement="conforming",
            noncap_refinement="conforming",
            layer_control=control,
            layer_policy=layer_policy,
            layer_regions=np.zeros(_count_cells(layers.mesh), dtype=np.int64),
            region_entity_ids=np.arange(2, dtype=np.int64),
            region_occurrence_paths=((), ()),
            region_ids=domain.region_ids,
            core_region_map=np.asarray((1,), dtype=np.int64),
            core_facet_source_ids=source_ids,
            core_facet_source_occurrence_paths=source_paths,
            core_vertex_representatives=roots,
            core_vertex_shifts=shifts,
            core_seam_polygon_pairs=seams,
            fidelity_source=query,
            maximum_deviation=0.0,
            reference_source=reference,
            mapped_domain=mapped,
        )
        from phydrax.meshing.providers._native_sources import (
            _mapped_layer_core_facet_relation,
        )

        core_facet_relation = _mapped_layer_core_facet_relation(source, authority)
        base = _periodic_layer_core_specification(source, tolerance=0.0)
        from phydrax.meshing._layer_core_controls import mapped_wall_source_scope

        volume_scope = phx.meshing.MeshingScope(
            source.source_id,
            source.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            3,
            source.scope_entity_set_id(3),
            source.scope_entity_ids(3),
        )
        layer_volume_scope = phx.meshing.MeshingScope(
            source.source_id,
            source.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            3,
            source.scope_entity_set_id(3),
            np.asarray(source.region_entity_ids, dtype=np.int64)[
                np.unique(np.asarray(source.layer_regions, dtype=np.int64))
            ],
        )
        source_control = phx.meshing.BoundaryLayerControl(
            mapped_wall_source_scope(source),
            control.schedule,
            route=control.route,
            volume_scope=layer_volume_scope,
            collision=control.collision,
            corner=control.corner,
            feature_angle=control.feature_angle,
            minimum_thickness_fraction=control.minimum_thickness_fraction,
            growth_rate_bounds=control.growth_rate_bounds,
            maximum_corner_stretch=control.maximum_corner_stretch,
            smoothing_iterations=control.smoothing_iterations,
            core_maximum_size=control.core_maximum_size,
        )
        specification = phx.meshing.VolumeMeshingSpec(
            phx.meshing.CellMeshingTarget(
                3, 3, base.target.cell_families, geometry_order=2
            ),
            volume_scope,
            base.fill_strategy,
            size_controls=(
                phx.meshing.UniformSizeControl(
                    volume_scope, 2.0, strength=phx.meshing.SizeControlStrength.SOFT
                ),
            ),
            region_controls=base.region_controls,
            layer_controls=(source_control,),
            periodic_constraints=base.periodic_constraints,
            maximum_deviation=0.0,
        )
        options = NativeMeshingOptions("layer_core")
        generation_plan = phx.meshing.NativeMeshingProvider(options).plan(
            source, specification, coordinate_contract=SpatialCoordinateContract.si()
        )
        result = generation_plan.execute()
        certificate = result.certification
        if (
            certificate is None
            or not certificate.passed
            or not result.audit.passed
            or not result.compliance.passed
            or certificate.embedding.status != "certified"
            or certificate.fidelity is None
            or certificate.fidelity.status != "certified"
        ):
            raise _ProfileFailure(
                "curved-periodic-certification",
                "The original mapped periodic layer/core was not independently certified.",
                {"source_binding_id": source.binding_id, "result_id": result.result_id},
            )
        periodic = result.mesh.periodic_topology
        if periodic is None or identification_id(periodic.cell) != identification_id(
            wall.periodic_topology.cell
        ):
            raise ValueError("The accepted core lost the original periodic group.")
        incidences = tuple(
            incidence.scipy_boundary().toarray()
            for incidence in periodic.quotient.incidences
        )
        if any(
            np.any(left @ right)
            for left, right in zip(incidences[:-1], incidences[1:], strict=True)
        ):
            raise ValueError("The accepted periodic quotient is not a chain complex.")
        measured_volume = float(
            np.sum(np.asarray(result.quality.evaluation.measures, dtype=np.float64))
        )
        if abs(measured_volume - 1.0 / 16.0) > _VOLUME_TOLERANCE / 16.0:
            raise _ProfileFailure(
                "curved-periodic-volume",
                "The mapped layer/core does not close to its original 1/16 gap volume.",
                {"measured_volume": measured_volume, "expected_volume": 1.0 / 16.0},
            )
        columns = _accepted_layer_columns(result)
        from phydrax.meshing._layer_core_association_transfer import (
            NativeLayerCoreAssociationTransfer,
        )

        association_transfer = NativeLayerCoreAssociationTransfer(source, specification)
        _, form_space, initial_fields = _layer_initial_field_banks(result)
        form_banks = {name: initial_fields[name] for name in ("Hcurl", "Hdiv")}
        source_form_fields = _layer_form_fields(result, form_space)
        del form_space
        scalar_space = _layer_scalar_space(result)
        scalar_banks = {}
        for index, field_name in enumerate(("H1", "DG")):
            coordinates = np.asarray(
                scalar_space.dof_maps[index].dof_coordinates, dtype=np.float64
            )
            scalar_banks[field_name] = 1.0 + coordinates[:, 1] + coordinates[:, 2]
        accepted = result
        hierarchy = None
        cycles = []
        for cycle_index in range(2 if name.endswith(":larger") else 1):
            prepared_refinement, refinement, mixed_policy = _periodic_mixed_refinement(
                accepted, columns, association_transfer, hierarchy
            )
            refined_space, refined_banks, forward_transfer = _layer_scalar_transition(
                refinement, scalar_space, scalar_banks
            )
            target_form_fields = _layer_field_plan(refinement.target).fields
            del scalar_space, refined_space
            refined_forms, forward_forms = _layer_form_transition(
                refinement,
                source_form_fields,
                target_form_fields,
                form_banks,
                hodge_resources=None,
            )
            form_space = _layer_field_plan(accepted).prepare()
            refined_form_space = _layer_field_plan(refinement.target).prepare()
            scalar_space = _layer_scalar_space(accepted)
            refined_space = _layer_scalar_space(refinement.target)
            for scalar, combined in (
                (scalar_space, form_space),
                (refined_space, refined_form_space),
            ):
                if any(
                    scalar.dof_maps[index].dof_map_id
                    != combined.dof_maps[index].dof_map_id
                    for index in range(2)
                ):
                    raise ValueError(
                        "Mapped scalar banks do not bind the retained combined FE layouts."
                    )
            del scalar, combined, scalar_space, refined_space
            archive_path = mapped_archive_dir / f"mapped-refinement-{cycle_index}.npz"
            archive_id = _write_mapped_refinement_archive(
                archive_path,
                source,
                specification,
                options,
                accepted,
                prepared_refinement,
                refinement,
                form_space,
                {**scalar_banks, **form_banks},
                refined_form_space,
                {**refined_banks, **refined_forms},
            )
            del form_space, refined_form_space
            refined_space = _layer_scalar_space(refinement.target)
            prepared_coarsening, coarsening = _periodic_mixed_coarsening(
                accepted, refinement, mixed_policy
            )
            restored_space, restored_banks, backward_transfer = _layer_scalar_transition(
                coarsening, refined_space, refined_banks
            )
            inverse_form_fields = _layer_field_plan(coarsening.target).fields
            del refined_space, restored_space
            restored_forms, backward_forms = _layer_form_transition(
                coarsening,
                target_form_fields,
                inverse_form_fields,
                refined_forms,
                hodge_resources=None,
            )
            for field_name in ("H1", "DG"):
                if not np.allclose(
                    restored_banks[field_name],
                    scalar_banks[field_name],
                    rtol=2.0e-11,
                    atol=2.0e-11,
                ):
                    raise ValueError(
                        f"Inverse coarsening did not restore the original {field_name} field."
                    )
            for field_name in ("Hcurl", "Hdiv"):
                if not np.allclose(
                    restored_forms[field_name],
                    form_banks[field_name],
                    rtol=2.0e-11,
                    atol=2.0e-11,
                ):
                    raise ValueError(
                        f"Inverse coarsening did not restore the original {field_name} field."
                    )
            cycles.append(
                {
                    "refinement_request_id": prepared_refinement.request.request_id,
                    "coarsening_request_id": prepared_coarsening.request.request_id,
                    "refinement_id": refinement.result_id,
                    "coarsening_id": coarsening.result_id,
                    "archive_path": str(archive_path),
                    "archive_content_id": archive_id,
                    "forward_scalar_transfer": forward_transfer,
                    "backward_scalar_transfer": backward_transfer,
                    "forward_compatible_transfer": forward_forms,
                    "backward_compatible_transfer": backward_forms,
                }
            )
            accepted = coarsening.target
            hierarchy = coarsening.hierarchy
            scalar_space, scalar_banks = _layer_scalar_space(accepted), restored_banks
            source_form_fields, form_banks = inverse_form_fields, restored_forms
        return {
            "status": "writer-complete",
            "wall_attempt": wall_attempt,
            "reference_root_topology_id": root.topology_id,
            "root_geometry_layout_id": geometry.geometry_layout_id,
            "mapped_domain_id": mapped.domain_id,
            "reference_source_binding_id": reference.binding_id,
            "source_binding_id": source.binding_id,
            "result_id": result.result_id,
            "layer_columns_id": columns.columns_id,
            "topology_cycles": cycles,
            "core_facet_relation": list(core_facet_relation),
            "measured_volume": measured_volume,
            "coefficient_orbits": list(coefficient_orbits),
            "profile_gap": 1.0 / 16.0,
        }
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
    prepared_layers: BoundaryLayerMesh | None = None,
) -> dict[str, object]:
    layers = (
        phx.meshing.prepare_boundary_layers(wall, control)
        if prepared_layers is None
        else prepared_layers
    )
    periodic = wall.periodic_topology
    layer = _layer_record(
        wall,
        control,
        layers,
        expected_growth=growth,
        expected_all_thicknesses=expected_all_thicknesses,
        closed_cap=None if periodic is not None and expected_closed else expected_closed,
    )
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
        if expected_closed:
            if layers.cap is None or cap_periodic is None or layers.closed_cap:
                raise RuntimeError("The periodic cap must be open in its lifted carrier.")
            from phydrax.geometry._periodic_embedding import _quotient_faces_pair_oriented

            if _quotient_faces_pair_oriented(layers.cap, cap_periodic) != "closed":
                raise RuntimeError(
                    "The periodic cap quotient is not an oriented closed surface."
                )
            periodic_evidence["cap_quotient_closed"] = True
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
        source = _make_cap_core_source(layers, control)
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
            # Original hybrid_core_size=0.1 for both rows; :larger adds cycles only.
            target_size = 0.1
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
            field = _mimetic_affine_field(
                generated.mesh, _LAYER_CORE_LIFECYCLES[name].solve_memory_bytes
            )
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
            coverage = certificate.coverage
            measures = (
                None if coverage is None else tuple(coverage.achieved_region_measures)
            )
            if (
                measures is None
                or len(measures) != 1
                or measures[0] is None
                or abs(float(measures[0]) - 1.0) > 1.0e-12
                or abs(core_measure - 1.0) > _VOLUME_TOLERANCE
            ):
                raise _ProfileFailure(
                    "layer-core-material-coverage",
                    "The closed inward-box container did not certify its unit material measure.",
                    {
                        "wall_layer": layer,
                        "achieved_region_measures": None
                        if measures is None
                        else [
                            None if value is None else float(value) for value in measures
                        ],
                        "cell_measure": core_measure,
                    },
                )
            layer["material_coverage"] = {
                "achieved_region_measures": [float(measures[0])],
                "expected_region_measures": [1.0],
                "relative_tolerance": 1.0e-12,
            }
            layer["lifecycle"] = _run_layer_core_lifecycle(
                name, source, specification, options, generated
            )
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
    model = brep_box(
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
    published = extrusion.source
    total = sum(thicknesses)
    if not np.array_equal(
        np.asarray(extrusion.control.schedule.thicknesses, dtype=np.float64),
        np.asarray(thicknesses, dtype=np.float64),
    ):
        raise RuntimeError("The CAD extrusion changed the authored slab schedule bits.")
    for scope in (wall_scope, cap_scope, volume_scope):
        if (scope.source_id, scope.source_revision) != (
            published.source_id,
            published.source_revision,
        ):
            raise RuntimeError(
                "A CAD sweep scope is not bound to the published slab source."
            )
    if (
        tuple(np.asarray(volume_scope.entity_ids, dtype=np.int64).tolist())
        != extrusion.layer_solid_ids
    ):
        raise RuntimeError(
            "The CAD sweep volume scope differs from the exact layer solids."
        )
    published_geometry = published.geometry
    if published_geometry is None:
        raise RuntimeError("The published CAD slab source has no retained BRep geometry.")
    published_points = np.asarray(published_geometry.vertex_points, dtype=np.float64)
    wall_faces = np.asarray(wall_scope.entity_ids, dtype=np.int64).tolist()
    cap_faces = np.asarray(cap_scope.entity_ids, dtype=np.int64).tolist()
    for face in wall_faces:
        if np.any(published_points[_face_vertex_rows(published, face), 2] != 0.0):
            raise RuntimeError("A published CAD wall face left the authored z=0 wall.")
    cap_solid_counts = []
    for face in cap_faces:
        if np.any(
            np.abs(published_points[_face_vertex_rows(published, face), 2] - total)
            > 1.0e-14
        ):
            raise RuntimeError("A published CAD cap face is not at the slab thickness.")
        cap_solid_counts.append(len(published.topology.face_solids[face]))
    if cap_solid_counts != [2 if remaining_core else 1] * len(cap_faces):
        raise RuntimeError("The published CAD cap has the wrong layer/core incidence.")
    core_bounds: list[list[float]] | None = None
    if remaining_core:
        core_points = np.concatenate(
            [
                published_points[_face_vertex_rows(published, face)]
                for face in published.topology.solid_faces[extrusion.core_solid_ids[0]]
            ]
        )
        lower = np.min(core_points, axis=0)
        upper = np.max(core_points, axis=0)
        if np.any(np.abs(lower - (0.0, 0.0, total)) > 1.0e-14) or np.any(
            np.abs(upper - (2.0, 3.0, 1.0)) > 1.0e-14
        ):
            raise RuntimeError(
                "The retained CAD core differs from the authored remainder."
            )
        core_bounds = [lower.tolist(), upper.tolist()]
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
        "scope": "native CAD preparation only, as in the original profile; no layer or core mesh is claimed",
        "independent_oracles": [
            "schedule bits unchanged",
            "wall/cap/volume scopes bound to published source revision",
            "volume scope equals exact layer solids",
            "published wall faces on z=0",
            "published cap faces on z=total thickness",
            "cap layer/core solid incidence",
            "retained core bounds" if remaining_core else "no retained core",
        ],
        "cap_solid_incidence": cap_solid_counts,
        "core_bounds": core_bounds,
    }


def _run_exact_sweep(name: str, /) -> dict[str, object]:
    periodic = name == "exact-sweep-periodic-accepted"
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
        topology = PeriodicMeshTopology(
            wall,
            PeriodicIsometryGroup(generator),
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
            raise ValueError(
                "The authored exact-sweep source lost its periodic topology."
            )
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
    from phydrax.geometry._piecewise_linear_boundary import (
        PiecewiseLinearBoundarySource,
        PiecewiseLinearDomainBoundaryOwner,
    )
    from phydrax.meshing._sweep import SweepControl, SweepMapKind
    from phydrax.meshing.providers._native_sources import NativeSweepSource

    owner = PiecewiseLinearDomainBoundaryOwner(domain)
    face_ids = np.asarray(owner.scope_entity_ids_by_mesh_row[2], dtype=np.int64)
    wall_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        owner.scope_entity_set_id(2),
        face_ids[: rows.shape[0]],
    )
    cap_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        owner.scope_entity_set_id(2),
        face_ids[rows.shape[0] : rows.shape[0] * 2],
    )
    volume_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        owner.scope_entity_set_id(3),
        owner.scope_entity_ids(3),
    )
    entity_ids = np.asarray(wall.entity_set(2).entity_ids, dtype=np.int64)
    association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        domain.source_id,
        domain.source_revision,
        wall.entity_set(2).entity_set_id,
        entity_ids,
        tuple(
            owner.source_entity_key_for_scope(2, int(value))[0]
            for value in face_ids[: rows.shape[0]]
        ),
        np.zeros(rows.shape[0]),
        exact=True,
        source_dimensions=np.full(rows.shape[0], 2, dtype=np.int64),
        source_indices=np.arange(rows.shape[0], dtype=np.int64),
        source_roles=(GeometrySourceEntityRole.FACET,) * rows.shape[0],
        source_region_sides=(("wall-material", None),) * rows.shape[0],
        orientations=-np.ones(rows.shape[0], dtype=np.int32),
    )
    control = phx.meshing.BoundaryLayerControl(
        wall_scope,
        schedule,
        route=phx.meshing.BoundaryLayerRoute.EXACT_SWEEP,
        volume_scope=volume_scope,
        cap_scope=cap_scope,
    )
    sweep = SweepControl(SweepMapKind.EXTRUSION, schedule, "original-reference-schedule")
    source = NativeSweepSource(
        wall,
        sweep,
        domain.source_id,
        domain.source_revision,
        fidelity_source=PiecewiseLinearBoundarySource(owner, 2),
        maximum_deviation=0.0,
        domain=domain,
        domain_vertex_map=np.asarray(
            [
                (int(vertex_id), station)
                for station in (0, schedule.layer_count)
                for vertex_id in np.asarray(wall.vertex_global_ids, dtype=np.int64)
            ],
            dtype=np.int64,
        ),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("prism",))
        ),
        phx.meshing.MeshingScope.whole(owner, 2),
        phx.meshing.VolumeFillStrategy.SWEEP,
        size_controls=(),
        layer_controls=(control,),
        maximum_deviation=0.0,
    )
    result = (
        phx.meshing.NativeMeshingProvider(NativeMeshingOptions("sweep"))
        .plan(source, specification, coordinate_contract=SpatialCoordinateContract.si())
        .execute()
    )
    if (
        result.certification is None
        or not result.certification.passed
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise ValueError("The original exact sweep did not pass source certification.")
    coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
    expected = np.stack(
        tuple(points + station * np.asarray((0.0, 0.0, 1.0)) for station in stations)
    )
    if not np.array_equal(coordinates.reshape(expected.shape), expected):
        raise ValueError("The exact sweep did not retain the authored physical stations.")
    if sum(block.cell_count for block in result.mesh.blocks) != 8 or any(
        block.cell_kind != "prism" for block in result.mesh.blocks
    ):
        raise ValueError("The exact sweep lost its four columns and two layers.")
    measured_volume = float(np.sum(np.asarray(result.quality.evaluation.measures)))
    if not np.isclose(measured_volume, stations[-1], rtol=_VOLUME_TOLERANCE, atol=1e-14):
        raise ValueError(
            "The exact sweep did not close the original physical slab volume."
        )
    target_periodic = result.mesh.periodic_topology
    if periodic:
        wall_periodic = wall.periodic_topology
        if (
            target_periodic is None
            or wall_periodic is None
            or identification_id(target_periodic.cell)
            != identification_id(wall_periodic.cell)
        ):
            raise ValueError("The exact sweep lost its original periodic group.")
        incidences = tuple(
            item.scipy_boundary().toarray()
            for item in target_periodic.quotient.incidences
        )
        if any(
            np.any(left @ right)
            for left, right in zip(incidences[:-1], incidences[1:], strict=True)
        ):
            raise ValueError("The exact sweep periodic quotient is not a chain complex.")
    elif target_periodic is not None:
        raise ValueError(
            "The nonperiodic sweep unexpectedly acquired a periodic quotient."
        )
    return {
        "status": "passed",
        "source_profile": name,
        "source_domain_id": domain.domain_id,
        "source_binding_id": source.binding_id,
        "wall_association_id": association.association_id,
        "control_id": control.control_id,
        "sweep_control_id": sweep.control_id,
        "specification_id": specification.specification_id,
        "result_id": result.result_id,
        "schedule_thicknesses": list(schedule.thicknesses),
        "measured_volume": measured_volume,
        "periodic_source_topology_id": periodic_source_id,
        "periodic_target_topology_id": None
        if target_periodic is None
        else target_periodic.periodic_topology_id,
    }


def _unit_box_source() -> tuple[
    NativeLayerCoreSource, Any, NativeMeshingOptions, dict[str, Any]
]:
    """Build the original source and request before native layer/core execution."""
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="hybrid-unit-box",
    )
    domain = MeshingDomain.from_brep(model)
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
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
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
    from phydrax.meshing._layer_core_brep import restrict_wall_charts

    if surface_result.surface_source is None:
        raise ValueError("The original CAD wall requires its accepted source charts.")
    wall_charts = restrict_wall_charts(surface_result.surface_source, wall)
    projection = prepare_brep_projection(model)
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
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
    )
    bottom_key = (
        domain.entity_id(2, bottom_row),
        tuple(domain.source_occurrences[2][bottom_row]),
    )
    bottom_ids = np.asarray(
        [
            int(identifier)
            for identifier in domain.scope_entity_ids(2)
            if domain.source_entity_key_for_scope(2, int(identifier)) == bottom_key
        ],
        dtype=np.int64,
    )
    if bottom_ids.size != 1:
        raise ValueError("The original bottom BRep face has no unique source scope ID.")
    wall_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        bottom_ids,
    )
    volume_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        domain.scope_entity_set_id(3),
        domain.scope_entity_ids(3),
    )
    schedule = phx.meshing.LayerSchedule.geometric(2, 0.1, growth_rate=1.0)
    control = phx.meshing.BoundaryLayerControl(
        wall_scope,
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        volume_scope=volume_scope,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
    )
    realized_control = phx.meshing.BoundaryLayerControl(
        _whole_wall_scope(wall),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
    )
    layer_policy = phx.meshing.BoundaryLayerPolicy()
    layers = phx.meshing.prepare_boundary_layers(
        wall,
        realized_control,
        wall_association=wall_vertex_association,
        policy=layer_policy,
    )
    layer_record = _layer_record(
        wall,
        realized_control,
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
    core_complex = PiecewiseLinearComplex(
        points,
        tuple(tuple(int(value) for value in row) for row in facets),
        np.arange(facets.shape[0], dtype=np.int64),
        np.asarray(
            ((0, -1),) * cap_rows.shape[0]
            + ((-1, 0),) * (facets.shape[0] - cap_rows.shape[0]),
            dtype=np.int64,
        ),
        ("fluid",),
        vertex_ids=tuple(
            f"{model.source_id}:layer-core-vertex:{row}" for row in range(points.shape[0])
        ),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=tuple(
            f"{model.source_id}:layer-core-facet:{row}" for row in range(facets.shape[0])
        ),
        facet_occurrence_paths=((),) * facets.shape[0],
        segments=np.empty((0, 2), dtype=np.int64),
        segment_ids=(),
        segment_occurrence_paths=(),
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
        region_entity_ids=np.asarray(domain.region_source_indices, dtype=np.int64),
        region_occurrence_paths=domain.region_source_occurrences,
        region_ids=("fluid",),
        core_region_map=np.asarray((0,), dtype=np.int64),
        source_boundary_scope=whole,
        source_domain=domain,
        source_wall_charts=wall_charts,
        source_wall_projection=projection,
        source_wall_association_policy=association_policy,
        source_boundary_charts=surface_result.surface_source,
        source_boundary_schedule=surface_plan.route_options.surface_schedule,
        fidelity_source=surface_result.surface_source.boundary_source,
        noncap_refinement="conforming",
        layer_control=realized_control,
        layer_policy=layer_policy,
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
    return (
        source,
        specification,
        NativeMeshingOptions("layer_core"),
        {
            "layer_record": layer_record,
            "wall_result_id": surface_result.result_id,
            "wall_association_id": association.association_id,
            "whole": whole,
        },
    )


def _run_unit_box() -> dict[str, object]:
    source, specification, options, context = _unit_box_source()
    layer_record = context["layer_record"]
    whole = context["whole"]
    try:
        provider = phx.meshing.NativeMeshingProvider(options)
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
                "wall_source_result_id": context["wall_result_id"],
                "wall_association_id": context["wall_association_id"],
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
    field = _mimetic_affine_field(
        generated.mesh, _LAYER_CORE_LIFECYCLES["unit-box"].solve_memory_bytes
    )
    lifecycle = _run_layer_core_lifecycle(
        "unit-box", source, specification, options, generated
    )
    return {
        "status": "passed",
        "lifecycle": lifecycle,
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
        "wall_source_result_id": context["wall_result_id"],
        "wall_association_id": context["wall_association_id"],
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


def _resume_layer_core_cycle(
    path: Path, content_id: str, profile: str, archive_bytes: str, /
) -> dict[str, object]:
    """Authenticate and continue one accepted layer/core cycle in a fresh process."""
    from phydrax._array_archive import ArrayArchiveCorruptionError
    from phydrax._model._structure import FiniteElementAcceptedStateArchiveRecipe
    from phydrax.lifecycle import Composition
    from phydrax.meshing._model_codec import (
        _restore_layer_core_accepted_lifecycle,
        read_model_artifact,
        restore_finite_element_field_banks,
        restore_layer_core_authoring,
        restore_mixed_adaptation_policy,
        restore_model_archive_value,
    )

    controls = _LAYER_CORE_LIFECYCLES[profile]
    # The cold reader admits the archive under its profile's own declaration,
    # which must be the archive budget the writer recorded.
    if controls.archive_bytes != int(archive_bytes):
        raise ValueError(
            "Cold reader's declared archive budget differs from the writer's."
        )
    limits = _layer_core_archive_limits(controls.archive_bytes)
    if not content_id:
        raise ValueError("Cold layer/core continuation requires its trusted content ID.")
    changed_id = ("0" if content_id[0] != "0" else "1") + content_id[1:]
    try:
        read_model_artifact(
            path,
            expected_type=Composition,
            restore=restore_model_archive_value,
            limits=limits,
            expected_content_id=changed_id,
        )
    except ArrayArchiveCorruptionError:
        pass
    else:
        raise ValueError("The layer/core archive accepted an unauthenticated content ID.")

    archive = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=limits,
        expected_content_id=content_id,
    )
    specification = phx.meshing.VolumeMeshingSpec.from_stored_fields(
        archive.value("specification")
    )
    # Charted (BRep) authoring replays its source atlas against the actual request.
    source = restore_layer_core_authoring(
        archive.value("source"), specification=specification
    )
    # One operation-local root proof binds both endpoints, the transition and
    # the association transfer to the same restored request owner.
    recipe = archive.value("refinement")
    initial, _, refined, adapter = _restore_layer_core_accepted_lifecycle(
        archive.value("result/initial"),
        archive.value("result/refined"),
        recipe,
        source,
        specification,
    )
    policy = restore_mixed_adaptation_policy(recipe.policy, adapter)

    def field_space(result: Any, bank: Any, *, scalar_only: bool = False) -> Any:
        layouts = bank.field_layouts[:2] if scalar_only else bank.field_layouts
        return phx.discretization.FiniteElementPlan(
            result.mesh,
            tuple(layout.rebuild_field_spec() for layout in layouts),
            coordinate_spec=result.geometry,
        ).prepare()

    initial_recipe = archive.value("fields/initial")
    refined_recipe = archive.value("fields/refined")
    initial_space = field_space(initial, initial_recipe)
    refined_space = field_space(refined.target, refined_recipe)
    _, initial_values = restore_finite_element_field_banks(initial_recipe, initial_space)
    _, refined_values = restore_finite_element_field_banks(refined_recipe, refined_space)
    initial_form_fields = _layer_form_fields(initial, initial_space)
    refined_form_fields = _layer_form_fields(refined.target, refined_space)
    del initial_space, refined_space
    names = ("H1", "DG", "Hcurl", "Hdiv")
    if (
        tuple(layout.name for layout in initial_recipe.field_layouts) != names
        or tuple(layout.name for layout in refined_recipe.field_layouts) != names
    ):
        raise ValueError(
            "Archived layer/core field layouts changed their original semantics."
        )

    def restore_history(value: Any) -> Any:
        if type(value) is FiniteElementAcceptedStateArchiveRecipe:
            return phx.solver.FiniteElementAcceptedState.from_archive_recipe(value)
        if type(value) is phx.solver.FiniteElementAcceptedState:
            return phx.solver.FiniteElementAcceptedState.from_stored_fields(value)
        raise TypeError("Archived layer/core history has no accepted-state owner.")

    initial_history = restore_history(archive.value("history/initial"))
    history = restore_history(archive.value("history/refined"))
    initial_volume, initial_density, _ = _layer_core_density(initial, controls)
    refined_volume, refined_density, _ = _layer_core_density(refined.target, controls)
    expected_initial, initial_history_space, _ = _layer_core_initial_history(
        initial, initial_volume, initial_density, controls
    )
    expected_refined, refined_history_space, _ = _layer_core_initial_history(
        refined.target, refined_volume, refined_density, controls
    )
    for saved, expected, result, density, name in (
        (initial_history, expected_initial, initial, initial_density, "initial"),
        (history, expected_refined, refined.target, refined_density, "refined"),
    ):
        # Donor c56614 _layer_continuation.py: committed material equals the
        # checkpointed density bitwise (L2685-2690); carried densities match
        # the declaration within the 1e-12 inventory tolerance (L3500-3513).
        archived_density = np.asarray(archive.value(f"density/{name}"))
        if (
            saved.topology_id != result.mesh.topology_id
            or saved.materials is None
            or not np.array_equal(
                np.asarray(saved.materials.state(controls.site_id).committed),
                archived_density,
            )
            or archived_density.shape != density.shape
            or np.max(np.abs(archived_density - density), initial=0.0) > 1.0e-12
            or any(
                not np.allclose(
                    np.asarray(actual),
                    np.asarray(expected_value),
                    rtol=1.0e-9,
                    atol=1.0e-9,
                )
                for actual, expected_value in zip(
                    saved.fields, expected.fields, strict=True
                )
            )
        ):
            raise ValueError("Archived accepted history lost its physical source or PDE.")
    replayed_history, _, forward_history = _layer_core_history_transition(
        refined, initial_history, initial_history_space, controls
    )
    if replayed_history.accepted_id != history.accepted_id:
        raise ValueError("Cold physical replay changed the saved refined history.")
    _, replayed_scalars, forward_scalars = _layer_scalar_transition(
        refined,
        field_space(initial, initial_recipe, scalar_only=True),
        {name: np.asarray(initial_values[row]) for row, name in enumerate(names[:2])},
        target_space=field_space(refined.target, refined_recipe, scalar_only=True),
    )
    del _
    replayed_forms, forward_forms = _layer_form_transition(
        refined,
        initial_form_fields,
        refined_form_fields,
        {
            name: np.asarray(initial_values[row])
            for row, name in enumerate(names)
            if row >= 2
        },
        hodge_resources=_layer_core_hodge_resources(controls),
    )
    replayed = {**replayed_scalars, **replayed_forms}
    if any(
        not np.allclose(
            replayed[name], np.asarray(refined_values[row]), rtol=2.0e-11, atol=2.0e-11
        )
        for row, name in enumerate(names)
    ):
        raise ValueError("Cold field-specific forward transfer differs from the archive.")
    prepared, inverse = _periodic_mixed_coarsening(initial, refined, policy)
    restored_history, _, backward_history = _layer_core_history_transition(
        inverse, history, refined_history_space, controls
    )
    _, restored_scalars, backward_scalars = _layer_scalar_transition(
        inverse,
        field_space(refined.target, refined_recipe, scalar_only=True),
        {name: np.asarray(refined_values[row]) for row, name in enumerate(names[:2])},
        target_space=field_space(inverse.target, initial_recipe, scalar_only=True),
    )
    del _
    restored_form_fields = tuple(
        layout.rebuild_field_spec() for layout in initial_recipe.field_layouts
    )
    restored_forms, backward_forms = _layer_form_transition(
        inverse,
        refined_form_fields,
        restored_form_fields,
        {
            name: np.asarray(refined_values[row])
            for row, name in enumerate(names)
            if row >= 2
        },
        hodge_resources=_layer_core_hodge_resources(controls),
    )
    restored = {**restored_scalars, **restored_forms}
    if any(
        not np.allclose(
            restored[name], np.asarray(initial_values[row]), rtol=2.0e-11, atol=2.0e-11
        )
        for row, name in enumerate(names)
    ) or any(
        not np.allclose(
            np.asarray(actual), np.asarray(expected), rtol=1.0e-9, atol=1.0e-9
        )
        for actual, expected in zip(
            restored_history.fields, initial_history.fields, strict=True
        )
    ):
        raise ValueError(
            "Cold inverse coarsening changed the original fields or history."
        )
    return {
        "inverse_request_id": prepared.request.request_id,
        "inverse_result_id": inverse.result_id,
        "forward_history": forward_history,
        "backward_history": backward_history,
        "forward_scalar_transfer": forward_scalars,
        "backward_scalar_transfer": backward_scalars,
        "forward_compatible_transfer": forward_forms,
        "backward_compatible_transfer": backward_forms,
    }


def _layer_field_bank_recipe(
    space: Any, fields: Any, banks: dict[str, np.ndarray], /
) -> Any:
    """Bind the original H1, DG, Hcurl and Hdiv banks to their accepted layout."""
    from phydrax.meshing._model_codec import finite_element_field_banks_recipe

    layouts = tuple(
        phx.solver.FiniteElementAcceptedFieldLayout(
            field,
            periodic_quotient=False,
            transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.NESTED_FE,
        )
        for field in fields
    )
    return finite_element_field_banks_recipe(
        space, layouts, tuple(banks[name] for name in ("H1", "DG", "Hcurl", "Hdiv"))
    )


def _layer_cycle_field_inputs(
    result: Any, plan: Any, banks: dict[str, np.ndarray], space: Any | None, /
) -> tuple[Any, tuple[Any, ...]]:
    """Archive one cycle source's banks and retain only its form-field operands."""
    if space is None:
        space = plan.prepare()
    return (
        _layer_field_bank_recipe(space, plan.fields, banks),
        _layer_form_fields(result, space),
    )


def _run_layer_core_lifecycle(
    profile: str, source: Any, specification: Any, options: Any, generated: Any, /
) -> dict[str, object]:
    """Refine, archive and cold-continue every original accepted cycle."""
    import subprocess
    import sys
    import tempfile

    from phydrax.meshing._layer_core_association_transfer import (
        NativeLayerCoreAssociationTransfer,
    )

    controls = _LAYER_CORE_LIFECYCLES[profile]
    columns = _accepted_layer_columns(generated)
    volume, density, inventories = _layer_core_density(generated, controls)
    history, history_space, harmonic_defect = _layer_core_initial_history(
        generated, volume, density, controls
    )
    del volume, density
    field_plan, initial_space, banks = _layer_initial_field_banks(generated)
    space: Any | None = initial_space
    del initial_space
    original_banks = dict(banks)
    adapter = NativeLayerCoreAssociationTransfer(source, specification)
    directory = Path(tempfile.mkdtemp(prefix=f"phydrax-{controls.semantics}-"))
    accepted, hierarchy = generated, None
    cycles: list[dict[str, object]] = []
    for cycle in range(controls.cycles):
        if history.materials is None:
            raise ValueError("The accepted layer/core history lost its materials.")
        field_recipe, source_form_fields = _layer_cycle_field_inputs(
            accepted, field_plan, banks, space
        )
        space = None
        prepared, refined, policy = _periodic_mixed_refinement(
            accepted, columns, adapter, hierarchy, mark_layer=True
        )
        refined_history, refined_history_space, forward_history = (
            _layer_core_history_transition(refined, history, history_space, controls)
        )
        refined_scalar_space, refined_scalars, forward_scalars = _layer_scalar_transition(
            refined,
            _layer_scalar_space(accepted),
            {name: banks[name] for name in ("H1", "DG")},
        )
        del refined_scalar_space
        refined_plan = _layer_field_plan(refined.target)
        refined_forms, forward_forms = _layer_form_transition(
            refined,
            source_form_fields,
            refined_plan.fields,
            {name: banks[name] for name in ("Hcurl", "Hdiv")},
            hodge_resources=_layer_core_hodge_resources(controls),
        )
        del source_form_fields
        refined_space = refined_plan.prepare()
        refined_banks = {**refined_scalars, **refined_forms}
        refined_recipe = _layer_field_bank_recipe(
            refined_space, refined_plan.fields, refined_banks
        )
        if refined_history.materials is None:
            raise ValueError("Refined accepted history lost its committed materials.")
        path = directory / f"cycle-{cycle}.phx"
        content_id = _write_layer_core_cycle_archive(
            path,
            controls,
            source,
            specification,
            options,
            accepted,
            prepared,
            refined,
            history,
            refined_history,
            field_recipe,
            refined_recipe,
            np.asarray(
                history.materials.state(controls.site_id).committed, dtype=np.float64
            ),
            np.asarray(
                refined_history.materials.state(controls.site_id).committed,
                dtype=np.float64,
            ),
        )
        # These archive workspaces have no later use, including subsequent cycles.
        del field_recipe, refined_recipe, refined_space
        completed = subprocess.run(
            (
                sys.executable,
                "-c",
                "import json,sys; from pathlib import Path; "
                "from tools.meshing_qualification.scenarios.native_hybrid_profiles "
                "import _resume_layer_core_cycle; "
                "print(json.dumps(_resume_layer_core_cycle("
                "Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4])))",
                str(path),
                content_id,
                profile,
                str(controls.archive_bytes),
            ),
            cwd=str(Path(__file__).resolve().parents[3]),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise ValueError(
                f"Authenticated cold layer/core continuation failed: {completed.stderr}"
            )
        cold = json.loads(completed.stdout)
        cycles.append(
            {
                "cycle": cycle,
                "refinement_request_id": prepared.request.request_id,
                "refinement_id": refined.result_id,
                "archive_path": str(path),
                "archive_content_id": content_id,
                "archive_budget_bytes": controls.archive_bytes,
                "forward_history": forward_history,
                "forward_scalar_transfer": forward_scalars,
                "forward_compatible_transfer": forward_forms,
                "cold_continuation": cold,
            }
        )
        if cycle + 1 == controls.cycles:
            break
        _, coarsening = _periodic_mixed_coarsening(accepted, refined, policy)
        history, history_space, _ = _layer_core_history_transition(
            coarsening, refined_history, refined_history_space, controls
        )
        _, restored_scalars, _ = _layer_scalar_transition(
            coarsening,
            _layer_scalar_space(refined.target),
            {name: refined_banks[name] for name in ("H1", "DG")},
        )
        field_plan = _layer_field_plan(coarsening.target)
        restored_forms, _ = _layer_form_transition(
            coarsening,
            refined_plan.fields,
            field_plan.fields,
            {name: refined_banks[name] for name in ("Hcurl", "Hdiv")},
            hodge_resources=_layer_core_hodge_resources(controls),
        )
        banks = {**restored_scalars, **restored_forms}
        if any(
            not np.allclose(banks[name], original_banks[name], rtol=2.0e-11, atol=2.0e-11)
            for name in original_banks
        ):
            raise ValueError("Warm inverse coarsening changed the original field banks.")
        del refined_plan
        accepted, hierarchy = coarsening.target, coarsening.hierarchy
    return {
        "material_site": controls.site_id,
        "region_densities": list(controls.densities),
        "manufactured_harmonic": list(controls.harmonic),
        "material_inventories": inventories.tolist(),
        "harmonic_defect": harmonic_defect,
        "adaptation_cycles": controls.cycles,
        "cycles": cycles,
    }


def _run_mapped_cold_supervisor(name: str, /) -> dict[str, object]:
    """Wait for the real writer to exit before restoring and continuing in a new process."""
    import subprocess
    import sys
    import tempfile

    directory = Path(tempfile.mkdtemp(prefix="phydrax-original-mapped-layer-"))
    writer = subprocess.run(
        (
            sys.executable,
            "-c",
            "import json,sys; from pathlib import Path; "
            "from tools.meshing_qualification.scenarios.native_hybrid_profiles "
            "import _run_wall_profile; "
            "print(json.dumps(_run_wall_profile(sys.argv[1], "
            "mapped_archive_dir=Path(sys.argv[2]))))",
            name,
            str(directory),
        ),
        cwd=str(Path(__file__).resolve().parents[3]),
        capture_output=True,
        text=True,
        check=False,
    )
    if writer.returncode:
        raise _ProfileFailure(
            "curved-periodic-writer",
            "The original mapped writer did not complete its accepted refinement cycles.",
            {
                "returncode": writer.returncode,
                "stdout": writer.stdout,
                "stderr": writer.stderr,
            },
        )
    output = json.loads(writer.stdout.strip().splitlines()[-1])
    if output.get("status") != "writer-complete":
        raise ValueError("Mapped writer did not retain its actual accepted state.")
    cold_cycles = []
    for cycle in output["topology_cycles"]:
        reader = subprocess.run(
            (
                sys.executable,
                "-c",
                "import json,sys; from pathlib import Path; "
                "from tools.meshing_qualification.scenarios.native_hybrid_profiles "
                "import _resume_mapped_refinement; "
                "print(json.dumps(_resume_mapped_refinement(Path(sys.argv[1]), sys.argv[2])))",
                cycle["archive_path"],
                cycle["archive_content_id"],
            ),
            cwd=str(Path(__file__).resolve().parents[3]),
            capture_output=True,
            text=True,
            check=False,
        )
        if reader.returncode:
            raise _ProfileFailure(
                "curved-periodic-cold-continuation",
                "Fresh mapped accepted-state restore or physical continuation failed.",
                {
                    "returncode": reader.returncode,
                    "stdout": reader.stdout,
                    "stderr": reader.stderr,
                },
            )
        cold = json.loads(reader.stdout.strip().splitlines()[-1])
        if (
            cold["refinement_id"] != cycle["refinement_id"]
            or cold["inverse_request_id"] != cycle["coarsening_request_id"]
            or cold["inverse_result_id"] != cycle["coarsening_id"]
        ):
            raise ValueError(
                "Cold mapped continuation changed its actual topology transition."
            )
        cold_cycles.append(cold)
    output["status"] = "passed"
    output["cold_cycles"] = cold_cycles
    return output


def _run_profile(profile_id: str, /) -> dict[str, object]:
    name = _profile_name(profile_id)
    if name == "archived-layer-core":
        source, control, layer_policy = _archived_ridge_source(
            cap_refinement="conforming"
        )
        from phydrax.meshing._model_codec import layer_core_authoring_recipe

        authoring_recipe = layer_core_authoring_recipe(source, control, layer_policy)
        schedule = control.schedule
        wall_attempt = _complete_wall_attempt(
            name,
            source.layers.source_wall,
            control,
            schedule,
            {"source_profile": name},
            False,
            True,
            1.0,
            prepared_layers=source.layers,
        )
        specification = _periodic_layer_core_specification(source)
        options = NativeMeshingOptions("layer_core")
        generation_plan = phx.meshing.NativeMeshingProvider(options).plan(
            source, specification, coordinate_contract=SpatialCoordinateContract.si()
        )
        generated = generation_plan.execute()
        if generated.certification is None or not generated.certification.passed:
            raise _ProfileFailure(
                "archive-initial-certification",
                "The original periodic ridge generation did not pass source certification.",
                {
                    "source_binding_id": source.binding_id,
                    "result_id": generated.result_id,
                },
            )
        return {
            "status": "passed",
            "profile_identity": profile_id,
            "source_binding_id": source.binding_id,
            "authoring_binding_id": authoring_recipe.expected_binding_id,
            "wall_layer": wall_attempt,
            "generation_result_id": generated.result_id,
            "lifecycle": _run_layer_core_lifecycle(
                name, source, specification, options, generated
            ),
        }
    if name in ("cad-entire-solid-slab", "cad-slab-and-core"):
        return _run_cad_extrusion(name)
    if name in ("exact-sweep-nonperiodic-accepted", "exact-sweep-periodic-accepted"):
        return _run_exact_sweep(name)
    if name == "channel_gap005_policies":
        return _run_channel_policies()
    if name == "unit-box":
        return _run_unit_box()
    if name.startswith("curved-periodic-layer-core"):
        return _run_mapped_cold_supervisor(name)
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
