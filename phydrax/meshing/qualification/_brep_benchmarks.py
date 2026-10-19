# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""BRep benchmark corpus and comparisons with caller-supplied sampling."""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from math import sqrt
from typing import assert_never, Callable, Literal, Protocol
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._resource_ticket import HostBatchTicket
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...geometry.brep import _measure as measure, _query_membership as membership
from ...geometry.brep._affine_query import QueryBroadphase
from ...geometry.brep._constructors import (
    assemble_brep_model,
    brep_box,
    brep_revolution,
    brep_sphere,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from ...geometry.brep._exact import square_root_interval
from ...geometry.brep._measure import BRepMassProperties
from ...geometry.brep._model import BRepModel
from ...geometry.brep._patches import BSplineCurve, BSplineSurfacePatch
from ...geometry.brep._query import prepare_brep_query, PreparedBRepQuery
from ...interchange._step_repair import audit_shells, StepRepairPolicy
from ...typing import checked, parse


type _BRepCorpusRoute = Literal["sphere", "revolution", "affine", "nurbs"]


@checked
def benchmark_box(*, tessellation: BRepTessellationPolicy | None = None) -> BRepModel:
    """Return the SI-coordinate 1-by-2-by-3 box used by BRep benchmarks."""
    return brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=tessellation,
    )


def rational_ellipse_profile(
    center: tuple[float, float], a: float, b: float
) -> ProfileLoop:
    """Return a closed rational quadratic ellipse with quarter-period knots."""
    cx, cy = center
    weight = sqrt(0.5)
    curve = BSplineCurve(
        [
            [cx + a, cy],
            [cx + a, cy + b],
            [cx, cy + b],
            [cx - a, cy + b],
            [cx - a, cy],
            [cx - a, cy - b],
            [cx, cy - b],
            [cx + a, cy - b],
            [cx + a, cy],
        ],
        [1.0, weight, 1.0, weight, 1.0, weight, 1.0, weight, 1.0],
        [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
        2,
    )
    return ProfileLoop(((cx + a, cy),), (curve,))


@checked
def revolved_ellipse(
    tessellation: BRepTessellationPolicy | None = None,
) -> BRepModel:
    """Return the radius-three revolution of a 1-by-0.5 rational ellipse."""
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        rational_ellipse_profile((3.0, 0.0), 1.0, 0.5),
    )
    return brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="revolved-rational-ellipse",
        tessellation=tessellation,
    )


def trimmed_polynomial_nurbs_box() -> BRepModel:
    """Return the benchmark box with trimmed bilinear polynomial NURBS faces."""
    source = benchmark_box()
    geometry = source.require_geometry()
    patches = []
    bounds = np.rint(np.asarray(source.parameter_bounds))
    for face, plane in enumerate(source.patches):
        box = bounds[face]
        uv = np.asarray(
            (
                ((box[0, 0], box[0, 1]), (box[0, 0], box[1, 1])),
                ((box[1, 0], box[0, 1]), (box[1, 0], box[1, 1])),
            ),
            dtype=np.float64,
        )
        controls = np.asarray(plane.evaluate(jnp.asarray(uv)))
        patches.append(
            BSplineSurfacePatch(
                controls,
                np.ones((2, 2), dtype=np.float64),
                np.asarray((box[0, 0],) * 2 + (box[1, 0],) * 2),
                np.asarray((box[0, 1],) * 2 + (box[1, 1],) * 2),
                1,
                1,
            )
        )
    name = "trimmed-nurbs-box"
    return assemble_brep_model(
        geometry,
        tuple(patches),
        bounds,
        np.asarray(source.orientation),
        ("face",) * len(patches),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id=name,
        source_format="native",
        source_digest=sha256(name.encode()).hexdigest(),
        import_policy_id=sha256(b"certified-property-test").hexdigest(),
    )


def corpus_route_query(route: str) -> PreparedBRepQuery:
    """Prepare the sphere, rational revolution, affine, or NURBS corpus query."""
    selected_route = parse(route, _BRepCorpusRoute, "corpus_route_query.route")
    lazy = BRepTessellationPolicy(realize=False)
    match selected_route:
        case "nurbs":
            model = trimmed_polynomial_nurbs_box()
        case "revolution":
            model = revolved_ellipse(lazy)
        case "sphere":
            model = brep_sphere(
                1.0,
                coordinate_contract=SpatialCoordinateContract.si(),
                tessellation=lazy,
            )
        case "affine":
            model = benchmark_box(tessellation=lazy)
        case _:
            assert_never(selected_route)
    return prepare_brep_query(model)


@checked
def corpus_route_points(query: PreparedBRepQuery, route: str) -> np.ndarray:
    """Return inside/outside points and a classifier-width boundary band."""
    selected_route = parse(route, _BRepCorpusRoute, "corpus_route_points.route")
    match selected_route:
        case "sphere":
            inside, outside, boundary = (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        case "revolution":
            inside, outside, boundary = (3.0, 0.0, 0.0), (5.0, 0.0, 0.0), (4.0, 0.0, 0.0)
        case "affine" | "nurbs":
            inside, outside, boundary = (0.5, 1.0, 1.5), (2.0, 1.0, 1.5), (1.0, 1.0, 1.5)
        case _:
            assert_never(selected_route)
    offsets = np.asarray((-2.0, -0.5, 0.0, 0.5, 2.0), dtype=np.float64)
    band = np.tile(np.asarray(boundary, dtype=np.float64), (7, 1))
    band[:5, 0] += offsets * query.tolerances.classifier
    band[5, 0] = np.nextafter(boundary[0], -np.inf)
    band[6, 0] = np.nextafter(boundary[0], np.inf)
    return np.concatenate((np.asarray((inside, outside), dtype=np.float64), band))


class BenchmarkSampleMeasure(Protocol):
    """Collect synchronized samples without installing a profiler in clean runs."""

    def __call__[T](self, call: Callable[[], T], /) -> tuple[T, dict[str, object]]: ...


def _rational_face_filter(
    point: np.ndarray,
    boxes: np.ndarray,
    faces: tuple[int, ...],
    face_indices: dict[int, int],
    broadphase: QueryBroadphase,
    classifier: float,
    /,
) -> tuple[tuple[int, ...], float, float, float]:
    del face_indices, broadphase
    bounds = []
    delta = tuple(Fraction(float(value)) for value in point)
    for box in boxes:
        first, last = (tuple(Fraction(float(value)) for value in row) for row in box)
        gap = tuple(
            max(Fraction(0), a - q, q - b)
            for a, b, q in zip(first, last, delta, strict=True)
        )
        far = tuple(
            max(abs(a - q), abs(b - q))
            for a, b, q in zip(first, last, delta, strict=True)
        )
        bounds.append(
            (
                square_root_interval(sum((value * value for value in gap), Fraction(0)))[
                    0
                ],
                square_root_interval(sum((value * value for value in far), Fraction(0)))[
                    1
                ],
            )
        )
    distances = np.asarray(bounds, dtype=np.float64)
    near = distances[:, 0] <= classifier
    return (
        tuple(faces[index] for index in np.flatnonzero(near)),
        float(np.min(distances[:, 0])),
        float(np.min(distances[:, 1])),
        float(np.min(distances[~near, 0], initial=np.inf)),
    )


def _decision_payload(result: membership.BRepContainmentResult) -> dict[str, object]:
    return {
        name: np.asarray(value).tolist()
        for name, value in (
            ("inside", result.inside),
            ("status", result.status),
            ("unresolved", result.unresolved),
            ("resource_exhausted", result.resource_exhausted),
            ("query_operations", result.query_operations),
        )
    }


def qualify_brep_containment(
    query: PreparedBRepQuery,
    points: Array,
    /,
    *,
    sampling: BenchmarkSampleMeasure,
) -> dict[str, object]:
    """Compare the directed filter with rational bounds on one fixed request."""

    def contains() -> membership.BRepContainmentResult:
        return query.contains(points)

    with patch.object(membership, "_containment_face_filter", _rational_face_filter):
        baseline, baseline_samples = sampling(contains)
    reduced, reduced_samples = sampling(contains)
    before, after = _decision_payload(baseline), _decision_payload(reduced)
    if before != after:
        raise ValueError(
            "BRepBenchmark.decisions must preserve the corpus payload; got a changed decision."
        )
    return {
        "model_id": query.model.model_id,
        "request_id": canonical_fingerprint(
            {
                "kind": "certified-brep-containment",
                "model_id": query.model.model_id,
                "policy_id": query.policy.policy_id,
                "solid": 0,
                "points": np.asarray(points),
                "classifier": query.tolerances.classifier,
                "distance": query.tolerances.distance,
            }
        ),
        "query_count": points.shape[0],
        "decisions_identical": True,
        "decision_payload": after,
        "baseline": baseline_samples,
        "boundary_filter": reduced_samples,
    }


def _mass_call(model: BRepModel, subdivisions: int, /) -> tuple[BRepMassProperties, int]:
    with HostBatchTicket(100_000_000, bytes_upper=1_000_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000_000, maximum_memory_bytes=1_000_000_000
        )
        with budget.activate():
            result = measure.brep_mass_properties(
                model,
                model.require_geometry(),
                subdivisions=subdivisions,
                coordinate_budget=budget,
            )
            return result, budget.work_units


def qualify_brep_moments(
    model: BRepModel,
    subdivisions: int,
    /,
    *,
    sampling: BenchmarkSampleMeasure,
) -> dict[str, object]:
    """Compare certified volume evidence and charged work at unchanged capacity."""

    def moments() -> tuple[BRepMassProperties, int]:
        return _mass_call(model, subdivisions)

    with patch.object(
        measure, "_face_moment_enclosures", measure._face_cell_moment_enclosures
    ):
        (baseline, baseline_work), baseline_samples = sampling(moments)
    (reduced, reduced_work), reduced_samples = sampling(moments)
    before, after = (
        np.asarray(baseline.volume_enclosures),
        np.asarray(reduced.volume_enclosures),
    )
    return {
        "model_id": model.model_id,
        "request_id": canonical_fingerprint(
            {
                "kind": "certified-brep-moments",
                "model_id": model.model_id,
                "subdivisions": subdivisions,
            }
        ),
        "constant_cells": baseline_samples,
        "boundary_integrals": reduced_samples,
        "counts": {
            "constant_cell_work_units": baseline_work,
            "boundary_work_units": reduced_work,
        },
        "constant_cell_volume_enclosures": before.tolist(),
        "boundary_volume_enclosures": after.tolist(),
        "constant_cell_volume_evidence_id": canonical_fingerprint(before),
        "boundary_volume_evidence_id": canonical_fingerprint(after),
        "volume_widths_no_larger": bool(
            np.all(after[:, 1] - after[:, 0] <= before[:, 1] - before[:, 0])
        ),
    }


def qualify_brep_orientation_audit(
    model: BRepModel,
    /,
    *,
    sampling: BenchmarkSampleMeasure,
) -> dict[str, object]:
    """Compare default STEP orientation outcomes using the signed-moment owner."""
    policy = StepRepairPolicy()
    labels = tuple(
        str(shell) for shell in range(len(model.require_geometry().shell_faces))
    )

    def audit() -> tuple[BRepModel, tuple[object, ...], tuple[str, ...]]:
        return audit_shells(model, policy, labels, None)

    with patch.object(
        measure, "_face_moment_enclosures", measure._face_cell_moment_enclosures
    ):
        baseline, baseline_samples = sampling(audit)
    reduced, reduced_samples = sampling(audit)
    if baseline[0].model_id != reduced[0].model_id or baseline[1:] != reduced[1:]:
        raise ValueError(
            "BRepBenchmark.orientation must preserve model and audit evidence; got a changed outcome."
        )
    return {
        "model_id": model.model_id,
        "request_id": canonical_fingerprint(
            {
                "kind": "step-default-orientation-audit",
                "model_id": model.model_id,
                "subdivisions": policy.subdivisions,
            }
        ),
        "default_subdivisions": policy.subdivisions,
        "constant_cells": baseline_samples,
        "boundary_integrals": reduced_samples,
        "orientation_assumptions": list(reduced[2]),
        "outcomes_identical": True,
    }
