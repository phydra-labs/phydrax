# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Any, NotRequired, TypedDict

import numpy as np
from scipy.spatial import ConvexHull

import phydrax as phx
from phydrax import canonical_fingerprint
from phydrax.meshing import meshcore_available
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._records import _quality_summary, _result_identity
from tools.meshing_qualification._runtime import (
    _contract,
    _measured,
    _missing_dependency,
    _runtime_identity,
    qualification_failure,
    StageLedger,
)


class _SegmentConformityRecord(TypedDict):
    segments: int
    carried_edges: int
    uncovered_segments: int
    off_segment_endpoints: int
    maximum_relative_length_defect: float


class _ConstrainedRecord(TypedDict):
    status: str
    route: str
    predicate_mode: str
    requested_minimum_angle_degrees: float
    minimum_angle_degrees: float
    unrefined_minimum_angle_degrees: float
    certified_minimum_angle_degrees: float
    requested_maximum_area: float
    maximum_area: float
    relative_area_residual: float
    positive_orientations: int
    triangles: int
    unrefined_triangles: int
    steiner_points: int
    conformity: _SegmentConformityRecord
    unrefined_conformity: _SegmentConformityRecord
    limited_status: str
    limited_steiner_points: int
    evidence_id: str
    repeat_equal: bool
    resources: NotRequired[dict[str, dict[str, float]]]
    stage_measurements: dict[str, dict[str, float | int | str]]


def _simplex_measures(points: np.ndarray, simplices: np.ndarray, /) -> np.ndarray:
    """Signed planar area/volume; positive area for immersed triangles."""
    edges = points[simplices[:, 1:]] - points[simplices[:, :1]]
    if points.shape[1] == 2:
        return 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    if simplices.shape[1] == 3:
        return 0.5 * np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
    return np.sum(edges[:, 0] * np.cross(edges[:, 1], edges[:, 2]), axis=1) / 6.0


def _delaunay_record(points: np.ndarray, /) -> dict[str, object]:
    """Exact orientation, empty-ball, hull-coverage, and repeat checks of one set."""

    measurements: dict[str, object] = {}
    _ledger = StageLedger(measurements)

    exact = phx.geometry.PredicateMode.EXACT
    triangulation, stage = _ledger.run(
        "DelaunayTriangulation",
        lambda: _measured(lambda: phx.geometry.DelaunayTriangulation(points)),
    )

    simplices = triangulation.simplices
    corners = tuple(points[simplices[:, index]] for index in range(simplices.shape[1]))
    # Every (simplex, point) pair.
    balls = tuple(corner[:, None, :] for corner in corners)
    if points.shape[1] == 2:
        orientation = phx.geometry.orient2d(*corners, mode=exact)
        inside = phx.geometry.incircle(
            balls[0], balls[1], balls[2], points[None], mode=exact
        )
    else:
        orientation = phx.geometry.orient3d(*corners, mode=exact)
        inside = phx.geometry.insphere(
            balls[0], balls[1], balls[2], balls[3], points[None], mode=exact
        )
    hull = ConvexHull(points).volume
    measure = float(np.sum(_simplex_measures(points, simplices)))
    repeated = phx.geometry.DelaunayTriangulation(points)

    evidence = triangulation.evidence
    record = {
        "points": points.shape[0],
        "simplices": simplices.shape[0],
        "simplices_id": canonical_fingerprint(simplices),
        "evidence_id": evidence.evidence_id,
        "route": evidence.route,
        "status": evidence.status,
        "predicate_mode": evidence.predicate_mode.value,
        "positive_orientations": int(np.count_nonzero(orientation.signs == 1)),
        "in_ball_checks": inside.signs.size,
        "empty_ball_violations": int(np.count_nonzero(inside.signs > 0)),
        # Zero signs beyond each simplex's own vertices are cocircular/cospherical ties.
        "cospherical_ties": int(np.count_nonzero(inside.signs == 0)) - simplices.size,
        "hull_measure": hull,
        "measure": measure,
        "relative_hull_residual": abs(measure - hull) / hull,
        "repeat_equal": bool(
            np.array_equal(repeated.simplices, simplices)
            and repeated.evidence.evidence_id == evidence.evidence_id
        ),
        "resources": stage,
    }
    record["stage_measurements"] = measurements["stage_measurements"]
    if (
        not (np.all(orientation.certain) and np.all(inside.certain))
        or record["positive_orientations"] != simplices.shape[0]
        or record["empty_ball_violations"]
        or record["relative_hull_residual"] > 1.0e-12
        or not record["repeat_equal"]
        or evidence.status != "ok"
    ):
        raise qualification_failure(
            "_delaunay_record",
            (f"{evidence.route} verification failed: {record}"),
            {
                "evidence": evidence,
                "inside": inside,
                "orientation": orientation,
                "record": record,
                "simplices": simplices,
            },
        )
    return record


def _duplicate_record(rng: np.random.Generator, dimension: int, /) -> dict[str, object]:
    """Duplicated points map to their first copy and never become vertices or cells."""

    count, copies = 48, 8
    base = rng.random((count, dimension))
    chosen = np.sort(rng.choice(count, size=copies, replace=False))
    points = np.concatenate((base, base[chosen]))
    triangulation = phx.geometry.DelaunayTriangulation(points)
    unique = phx.geometry.DelaunayTriangulation(base)
    voronoi = phx.geometry.VoronoiDiagram(
        points,
        box_lower=np.zeros((dimension,), dtype=np.float64),
        box_upper=np.ones((dimension,), dtype=np.float64),
    )
    measures = voronoi.cells.measures
    record = {
        "input_points": points.shape[0],
        "duplicate_count": triangulation.evidence.duplicate_count,
        "vertex_count": triangulation.evidence.vertex_count,
        "vertex_map_to_first_copy": bool(
            np.array_equal(
                triangulation.vertex_map, np.concatenate((np.arange(count), chosen))
            )
        ),
        "duplicates_in_simplices": int(
            np.count_nonzero(triangulation.simplices >= count)
        ),
        "simplices_equal_unique_input": bool(
            np.array_equal(triangulation.simplices, unique.simplices)
        ),
        "voronoi_duplicate_count": voronoi.evidence.duplicate_count,
        "voronoi_duplicate_measure": float(np.sum(measures[count:])),
        "voronoi_measure_sum": float(np.sum(measures)),
    }
    if (
        record["duplicate_count"] != copies
        or record["voronoi_duplicate_count"] != copies
        or not record["vertex_map_to_first_copy"]
        or record["duplicates_in_simplices"]
        or not record["simplices_equal_unique_input"]
        or record["voronoi_duplicate_measure"] != 0.0
        or abs(record["voronoi_measure_sum"] - 1.0) > 1.0e-12
    ):
        raise qualification_failure(
            "_duplicate_record",
            (f"{dimension}D duplicate handling failed: {record}"),
            {"copies": copies, "record": record},
        )
    return record


def _segment_conformity(
    triangulation: phx.geometry.ConstrainedDelaunayTriangulation,
    segments: np.ndarray,
    /,
) -> _SegmentConformityRecord:
    """Measure collinearity and length coverage of the edges carrying each segment."""
    points, triangles = triangulation.points, triangulation.triangles
    cells, corners = np.nonzero(triangulation.segment_ids >= 0)
    edges = np.sort(
        np.stack(
            (triangles[cells, (corners + 1) % 3], triangles[cells, (corners + 2) % 3]),
            axis=1,
        ),
        axis=1,
    )
    carried = np.unique(
        np.column_stack((triangulation.segment_ids[cells, corners], edges)), axis=0
    )
    owner, first, second = carried.T
    start, stop = points[segments[owner, 0]], points[segments[owner, 1]]
    on_line = phx.geometry.orient2d(
        np.concatenate((start, start)),
        np.concatenate((stop, stop)),
        np.concatenate((points[first], points[second])),
        mode=phx.geometry.PredicateMode.EXACT,
    )
    lengths = np.linalg.norm(points[second] - points[first], axis=1)
    covered = np.bincount(owner, weights=lengths, minlength=segments.shape[0])
    expected = np.linalg.norm(points[segments[:, 1]] - points[segments[:, 0]], axis=1)
    return {
        "segments": segments.shape[0],
        "carried_edges": carried.shape[0],
        "uncovered_segments": int(np.count_nonzero(covered == 0.0)),
        "off_segment_endpoints": int(np.count_nonzero(on_line.signs != 0)),
        "maximum_relative_length_defect": float(
            np.max(np.abs(covered - expected) / expected)
        ),
    }


def _constrained_record() -> tuple[
    _ConstrainedRecord, phx.meshing.CellMeshingResult, phx.meshing.CellMeshingResult
]:
    """Plate with a hole and an interior crack: CDT, refinement, and certification."""

    stage_measurements: dict[str, dict[str, float | int | str]] = {}
    measurements: dict[str, object] = {"stage_measurements": stage_measurements}
    _ledger = StageLedger(measurements)

    points = np.asarray(
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.75, 0.75),
            (1.25, 0.75),
            (1.25, 1.25),
            (0.75, 1.25),
            (0.25, 0.4),
            (1.75, 0.4),
        )
    )
    loop = np.arange(4, dtype=np.int32)
    ring = np.stack((loop, (loop + 1) % 4), axis=1)
    segments = np.concatenate((ring, 4 + ring, np.asarray(((8, 9),), dtype=np.int32)))
    holes = np.asarray(((1.0, 1.0),))
    min_angle, max_area = 28.0, 0.02

    def triangulate(
        max_steiner: int, refine: bool
    ) -> phx.geometry.ConstrainedDelaunayTriangulation:
        return phx.geometry.ConstrainedDelaunayTriangulation(
            points,
            segments,
            holes=holes,
            min_angle=min_angle if refine else 0.0,
            max_area=max_area if refine else np.inf,
            max_steiner=max_steiner,
        )

    unrefined, unrefined_stage = _ledger.run(
        "triangulate", lambda: _measured(lambda: triangulate(0, False))
    )
    refined, refined_stage = _ledger.run(
        "triangulate", lambda: _measured(lambda: triangulate(4000, True))
    )
    limited = triangulate(1, True)
    repeated = triangulate(4000, True)

    areas = _simplex_measures(refined.points, refined.triangles)
    orientation = phx.geometry.orient2d(
        *(refined.points[refined.triangles[:, index]] for index in range(3)),
        mode=phx.geometry.PredicateMode.EXACT,
    )

    def certify(
        triangulation: phx.geometry.ConstrainedDelaunayTriangulation,
    ) -> phx.meshing.CellMeshingResult:
        return phx.meshing.certify_cell_mesh(
            phx.discretization.CellMesh.from_triangles(
                triangulation.points, triangulation.triangles
            ),
            _contract(),
        )

    before, before_stage = _ledger.run(
        "certify", lambda: _measured(lambda: certify(unrefined))
    )
    after, after_stage = _ledger.run(
        "certify", lambda: _measured(lambda: certify(refined))
    )
    plate_area = 4.0 - 0.25
    record: _ConstrainedRecord = {
        "status": refined.evidence.status,
        "route": refined.evidence.route,
        "predicate_mode": refined.evidence.predicate_mode.value,
        "requested_minimum_angle_degrees": min_angle,
        "minimum_angle_degrees": refined.evidence.minimum_angle_degrees,
        "unrefined_minimum_angle_degrees": unrefined.evidence.minimum_angle_degrees,
        "certified_minimum_angle_degrees": float(np.degrees(after.quality.minimum_angle)),
        "requested_maximum_area": max_area,
        "maximum_area": float(np.max(areas)),
        "relative_area_residual": abs(float(np.sum(areas)) - plate_area) / plate_area,
        "positive_orientations": int(np.count_nonzero(orientation.signs == 1)),
        "triangles": refined.triangles.shape[0],
        "unrefined_triangles": unrefined.triangles.shape[0],
        "steiner_points": refined.evidence.steiner_count,
        "conformity": _segment_conformity(refined, segments),
        "unrefined_conformity": _segment_conformity(unrefined, segments),
        "limited_status": limited.evidence.status,
        "limited_steiner_points": limited.evidence.steiner_count,
        "evidence_id": refined.evidence.evidence_id,
        "repeat_equal": bool(
            repeated.evidence.evidence_id == refined.evidence.evidence_id
            and np.array_equal(repeated.points, refined.points)
            and np.array_equal(repeated.triangles, refined.triangles)
        ),
        "resources": {
            "unrefined": unrefined_stage,
            "refined": refined_stage,
            "certify_unrefined": before_stage,
            "certify_refined": after_stage,
        },
        "stage_measurements": stage_measurements,
    }
    conformity = (record["conformity"], record["unrefined_conformity"])
    if (
        record["status"] != "ok"
        or record["minimum_angle_degrees"] < min_angle
        or record["maximum_area"] > max_area
        or record["relative_area_residual"] > 1.0e-12
        or record["positive_orientations"] != record["triangles"]
        or any(
            item["uncovered_segments"]
            or item["off_segment_endpoints"]
            or item["maximum_relative_length_defect"] > 1.0e-12
            for item in conformity
        )
        or record["limited_status"] != "refinement_limit"
        or record["limited_steiner_points"] > 1
        or not record["repeat_equal"]
        or not (before.audit.passed and after.audit.passed)
        or record["certified_minimum_angle_degrees"] < min_angle * (1.0 - 1.0e-12)
    ):
        raise qualification_failure(
            "_constrained_record",
            (f"Constrained Delaunay qualification failed: {record}"),
            {
                "after": after,
                "before": before,
                "conformity": conformity,
                "max_area": max_area,
                "min_angle": min_angle,
                "record": record,
            },
        )
    return record, before, after


def _reciprocal_faces(cells: phx.geometry.DiagramCells, /) -> bool:
    """Every interior face (owner, neighbor) has the opposite face (neighbor, owner)."""
    owners = np.repeat(np.arange(cells.cell_count), np.diff(cells.cell_face_offsets))
    interior = cells.face_labels >= 0
    forward = np.unique(
        np.stack((owners[interior], cells.face_labels[interior]), axis=1), axis=0
    )
    return bool(np.array_equal(forward, np.unique(forward[:, ::-1], axis=0)))


def _diagram_record(
    diagram: Any, repeated: Any, lower: np.ndarray, upper: np.ndarray, stage: dict, /
) -> dict[str, object]:
    """Box coverage, reciprocity, centroid, and repeat checks of one diagram."""

    cells, evidence = diagram.cells, diagram.evidence
    box = float(np.prod(upper - lower))
    total = float(np.sum(cells.measures))
    occupied = cells.measures > 0.0
    centroids = cells.centroids[occupied]
    record = {
        "route": evidence.route,
        "predicate_mode": evidence.predicate_mode.value,
        "evidence_id": evidence.evidence_id,
        "measures_id": canonical_fingerprint(cells.measures),
        "cells": cells.cell_count,
        "empty_cells": int(np.count_nonzero(~occupied)),
        "redundant_generators": evidence.redundant_count,
        "dual_simplices": evidence.simplex_count,
        "box_measure": box,
        "measure_sum": total,
        "relative_box_residual": abs(total - box) / box,
        "reciprocal_faces": _reciprocal_faces(cells),
        "centroids_inside_box": bool(np.all((centroids >= lower) & (centroids <= upper))),
        "repeat_equal": bool(
            repeated.evidence.evidence_id == evidence.evidence_id
            and np.array_equal(repeated.cells.vertices, cells.vertices)
        ),
        "resources": stage,
    }
    if (
        record["relative_box_residual"] > 1.0e-12
        or np.any(cells.measures < 0.0)
        or not record["reciprocal_faces"]
        or not record["centroids_inside_box"]
        or not record["repeat_equal"]
    ):
        raise qualification_failure(
            "_diagram_record",
            (f"{evidence.route} qualification failed: {record}"),
            {"cells": cells, "record": record},
        )
    return record


def _diagram_pair(
    rng: np.random.Generator, dimension: int, count: int, /
) -> dict[str, dict[str, object]]:
    """Voronoi and weighted power diagrams of one generator set in a box."""

    lower = np.zeros((dimension,), dtype=np.float64)
    upper = np.asarray((1.0, 2.0, 1.0)[:dimension], dtype=np.float64)
    generators = lower + (upper - lower) * rng.random((count, dimension))
    weights = 0.01 * rng.random(count)
    voronoi, voronoi_stage = _measured(
        lambda: phx.geometry.VoronoiDiagram(generators, box_lower=lower, box_upper=upper)
    )
    power, power_stage = _measured(
        lambda: phx.geometry.PowerDiagram(
            generators, weights, box_lower=lower, box_upper=upper
        )
    )
    unweighted = phx.geometry.PowerDiagram(
        generators,
        np.zeros((count,), dtype=np.float64),
        box_lower=lower,
        box_upper=upper,
    )
    voronoi_record = _diagram_record(
        voronoi,
        phx.geometry.VoronoiDiagram(generators, box_lower=lower, box_upper=upper),
        lower,
        upper,
        voronoi_stage,
    )

    deviation = float(np.max(np.abs(unweighted.cells.measures - voronoi.cells.measures)))
    voronoi_record["unweighted_power_maximum_deviation"] = deviation

    if deviation > 1.0e-10 * float(np.max(voronoi.cells.measures)):
        raise qualification_failure(
            "_diagram_pair",
            ("Unweighted power cells differ from the Voronoi cells."),
            {"deviation": deviation, "voronoi": voronoi},
        )
    return {
        f"voronoi_{dimension}d": voronoi_record,
        f"power_{dimension}d": _diagram_record(
            power,
            phx.geometry.PowerDiagram(
                generators, weights, box_lower=lower, box_upper=upper
            ),
            lower,
            upper,
            power_stage,
        ),
    }


def qualify_triangulation() -> dict[str, object]:

    if not meshcore_available():
        return _missing_dependency(
            "meshcore",
            "Exact triangulations and diagrams require the native meshcore library; "
            "install phydrax[meshcore] or set PHYDRAX_MESHCORE_LIBRARY to the shared "
            "library path.",
        )
    rng = np.random.default_rng(20260927)
    lattice_2d = np.stack(np.meshgrid(*(np.arange(6.0),) * 2, indexing="ij"), axis=-1)
    lattice_3d = np.stack(np.meshgrid(*(np.arange(4.0),) * 3, indexing="ij"), axis=-1)
    delaunay = {
        "random_2d": _delaunay_record(rng.random((200, 2))),
        "lattice_2d": _delaunay_record(lattice_2d.reshape((-1, 2))),
        "random_3d": _delaunay_record(rng.random((60, 3))),
        "lattice_3d": _delaunay_record(lattice_3d.reshape((-1, 3))),
    }
    duplicates = {
        f"{dimension}d": _duplicate_record(rng, dimension) for dimension in (2, 3)
    }
    constrained, unrefined, refined = _constrained_record()
    diagrams = {**_diagram_pair(rng, 2, 64), **_diagram_pair(rng, 3, 40)}
    # A zero-weight generator inside the heavy triangle is redundant: no cell.
    hidden = phx.geometry.PowerDiagram(
        np.asarray(((0.2, 0.2), (0.8, 0.2), (0.5, 0.8), (0.5, 0.4))),
        np.asarray((1.0, 1.0, 1.0, 0.0)),
        box_lower=(0.0, 0.0),
        box_upper=(1.0, 1.0),
    )
    dual_vertex_map = hidden.dual_vertex_map
    if dual_vertex_map is None:
        raise qualification_failure(
            "qualify_triangulation",
            (
                "Redundant-generator qualification requires its regular-triangulation vertex map."
            ),
            {"dual_vertex_map": dual_vertex_map},
        )
    redundant = {
        "redundant_count": hidden.evidence.redundant_count,
        "dual_vertex_map": dual_vertex_map.tolist(),
        "redundant_measure": float(hidden.cells.measures[3]),
        "measure_sum": float(np.sum(hidden.cells.measures)),
    }
    if (
        redundant["redundant_count"] != 1
        or redundant["dual_vertex_map"][3] != -1
        or redundant["redundant_measure"] != 0.0
        or abs(redundant["measure_sum"] - 1.0) > 1.0e-12
    ):
        raise qualification_failure(
            "qualify_triangulation",
            (f"Redundant power generator was not hidden: {redundant}"),
            {"redundant": redundant},
        )
    resources = {
        **{
            f"delaunay_{name}": record.pop("resources")
            for name, record in delaunay.items()
        },
        **{
            f"constrained_{name}": stage
            for name, stage in constrained.pop("resources").items()
        },
        **{name: record.pop("resources") for name, record in diagrams.items()},
    }
    return {
        "status": "passed",
        "source": {
            "constrained_unrefined": _result_identity(unrefined),
            "delaunay_simplices": {
                name: record["simplices_id"] for name, record in delaunay.items()
            },
        },
        "target": {
            "constrained_refined": _result_identity(refined),
            "evidence": {
                **{name: record["evidence_id"] for name, record in delaunay.items()},
                "constrained": constrained["evidence_id"],
                **{name: record["evidence_id"] for name, record in diagrams.items()},
            },
        },
        "runtime": _runtime_identity(),
        "quality": {
            "before": _quality_summary(unrefined.quality),
            "after": _quality_summary(refined.quality),
        },
        "conservation": {
            "delaunay_relative_hull_residual": {
                name: record["relative_hull_residual"]
                for name, record in delaunay.items()
            },
            "constrained_relative_area_residual": constrained["relative_area_residual"],
            "diagram_relative_box_residual": {
                name: record["relative_box_residual"] for name, record in diagrams.items()
            },
        },
        "determinism": {
            **{name: record["repeat_equal"] for name, record in delaunay.items()},
            "constrained": constrained["repeat_equal"],
            **{name: record["repeat_equal"] for name, record in diagrams.items()},
        },
        "resources": resources,
        "delaunay": delaunay,
        "duplicates": duplicates,
        "constrained": constrained,
        "diagrams": diagrams,
        "redundant_power_generator": redundant,
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_triangulation()
