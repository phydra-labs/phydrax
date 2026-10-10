#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded source-complete ray parity for host CAD solid containment."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..._bvh import prepare_bvh
from .._atlas import CurveTrimLoop, PolygonTrimLoop, TrimDomain
from ._affine_query import QueryBroadphase
from ._constructors import brep_trim_domain
from ._intersection import intersect_curve_region, ParametricIntersectionPolicy
from ._intersection_curve import CurveRange, SurfaceRegion
from ._model import BRepModel
from ._patches import LineCurve


@dataclass(frozen=True, slots=True)
class SolidContainmentCertificate:
    """Ray-parity evidence; ``ray_start`` is the declared search start (<= 0).

    Searching from ``ray_start`` makes a root at the query point interior to
    the ray range, so it can be classified rather than left at a range wall.
    """

    inside: bool
    complete: bool
    crossings: int
    boxes_processed: int
    unresolved_faces: tuple[int, ...]
    ray_start: float
    owner_operations: int
    resource_exhausted: bool


def _trim_box_decision(
    domain: TrimDomain,
    box: np.ndarray,
    /,
    *,
    maximum_depth: int,
) -> tuple[bool, bool, int]:
    """A connected box disjoint from all exact boundary covers has constant winding."""
    operations = 0
    for loop in domain.loops:
        if isinstance(loop, CurveTrimLoop):
            operations += loop.arc_lower.shape[0]
            if np.any(
                np.all((loop.arc_upper >= box[0]) & (loop.arc_lower <= box[1]), axis=1)
            ):
                return False, False, operations
            operations += loop.junction_lower.shape[0]
            if np.any(
                np.all(
                    (loop.junction_upper >= box[0]) & (loop.junction_lower <= box[1]),
                    axis=1,
                )
            ):
                return False, False, operations
        elif isinstance(loop, PolygonTrimLoop):
            operations += loop.vertices.shape[0]
            first, second = loop.vertices, np.roll(loop.vertices, -1, axis=0)
            lower, upper = np.minimum(first, second), np.maximum(first, second)
            if np.any(np.all((upper >= box[0]) & (lower <= box[1]), axis=1)):
                return False, False, operations
        else:
            raise TypeError("Containment requires a canonical native trim loop.")
    decision = domain.classify(
        (0.5 * (box[0] + box[1]))[None], maximum_depth=maximum_depth
    )
    for loop in domain.loops:
        operations += (
            2 * loop.arc_lower.shape[0] + loop.junction_lower.shape[0]
            if isinstance(loop, CurveTrimLoop)
            else loop.vertices.shape[0]
        )
    return (
        bool(decision.inside[0]),
        bool(decision.resolved[0]),
        operations + int(decision.refinements[0]),
    )


def certify_solid_containment(
    model: BRepModel,
    point: np.ndarray,
    solid: int,
    /,
    *,
    maximum_boxes: int,
    maximum_depth: int,
    relative_resolution: float = 1.0e-10,
    maximum_operations: int | None = None,
    source_boxes: np.ndarray | None = None,
    trim_domains: tuple[TrimDomain, ...] | None = None,
    broadphase: QueryBroadphase | None = None,
) -> SolidContainmentCertificate:
    """Count every transverse source intersection on one proved admissible ray.

    Rays through an edge, a pole, tangent/coincident components or unresolved
    root boxes do not establish parity. Finite alternate directions only seek
    an admissible certified ray; exhausting them remains explicitly unresolved.
    The search starts at ``-2**-20 * extent`` so a supporting surface through
    the query point gives an interior root: roots certified at negative ray
    parameters are ignored, a root inside a face trim whose enclosure holds 0
    means the point is on the boundary (unresolved), and only roots certified
    at positive parameters inside a face trim are crossings.
    """
    if model.geometry is None or not 0 <= solid < model.topology.num_solids:
        raise ValueError("Solid containment requires an authoritative model solid.")
    query = np.asarray(point, dtype=np.float64)
    if (
        query.shape != (3,)
        or not np.all(np.isfinite(query))
        or maximum_boxes < 1
        or maximum_depth < 0
    ):
        raise ValueError(
            "Containment requires a finite point and valid resource budgets."
        )
    if maximum_operations is not None and (
        type(maximum_operations) is not int or maximum_operations < 0
    ):
        raise ValueError("Containment operation allowance must be a nonnegative integer.")
    faces = model.topology.solid_faces[solid]
    parameter_boxes = np.asarray(model.parameter_bounds)
    if source_boxes is None:
        source_boxes = np.asarray(
            [model.patches[face].bounding_box(parameter_boxes[face]) for face in faces]
        )
    elif source_boxes.shape != (len(faces), 2, 3):
        raise ValueError(
            "Prepared containment source boxes must align with the authoritative solid faces."
        )
    if maximum_operations == 0:
        return SolidContainmentCertificate(False, False, 0, 0, faces, 0.0, 0, True)
    lower, upper = np.min(source_boxes[:, 0], axis=0), np.max(source_boxes[:, 1], axis=0)
    if np.any(query < lower) or np.any(query > upper):
        return SolidContainmentCertificate(False, True, 0, 0, (), 0.0, 1, False)
    extent = float(np.max(upper - lower))
    if not np.isfinite(extent) or extent <= 0:
        return SolidContainmentCertificate(False, False, 0, 0, faces, 0.0, 1, False)
    if trim_domains is None:
        domains = {
            face: brep_trim_domain(
                model.geometry, face, model.patches[face], tolerance=0.01
            )
            for face in faces
        }
    else:
        if len(trim_domains) != len(faces):
            raise ValueError(
                "Prepared containment trims must align with the authoritative solid faces."
            )
        domains = dict(zip(faces, trim_domains, strict=True))
    if broadphase is None:
        broadphase = QueryBroadphase.prepare(
            prepare_bvh(source_boxes[:, 0], source_boxes[:, 1])
        )
        source_indices = faces
    else:
        source_indices = tuple(range(len(model.patches)))
    ray_start = -float(np.ldexp(extent, -20))
    processed = 0
    owner_operations = 1
    exhausted = False
    pending: set[int] = set()
    for direction_values in (
        (1.0, 0.317, 0.613),
        (0.487, 1.0, 0.293),
        (0.211, 0.719, 1.0),
    ):
        direction = np.asarray(direction_values, dtype=np.float64)
        axis = int(np.argmax(direction))
        last = float(
            np.nextafter((upper[axis] - query[axis] + extent) / direction[axis], np.inf)
        )
        ray = CurveRange(LineCurve(query, direction), ray_start, last)
        crossings: list[tuple[float, float]] = []
        pending = set()
        if (
            maximum_operations is not None
            and owner_operations + broadphase.lower.shape[0] > maximum_operations
        ):
            pending.update(faces)
            exhausted = True
            break
        work = [0]
        start = tuple(
            Fraction(float(value)) + Fraction(ray_start) * Fraction(float(component))
            for value, component in zip(query, direction, strict=True)
        )
        candidates = {
            source_indices[index]
            for index in broadphase.ray(
                start, tuple(Fraction(float(value)) for value in direction), work
            )
        }
        owner_operations += work[0]
        for face in faces:
            if face not in candidates:
                continue
            if maximum_operations is not None:
                remaining = min(
                    maximum_boxes - processed, maximum_operations - owner_operations - 1
                )
            else:
                remaining = maximum_boxes - processed
            owner_operations += 1 if remaining > 0 else 0
            if remaining < 1:
                pending.add(face)
                exhausted = True
                continue
            intersection = intersect_curve_region(
                ray,
                SurfaceRegion(model.patches[face], parameter_boxes[face]),
                policy=ParametricIntersectionPolicy(
                    maximum_boxes=remaining, relative_resolution=relative_resolution
                ),
            )
            processed += intersection.work.boxes_processed
            owner_operations += intersection.work.boxes_processed
            if not intersection.complete or intersection.coincident:
                pending.add(face)
                continue
            for root in intersection.points:
                start, end = (
                    float(root.parameter_lower[0]),
                    float(root.parameter_upper[0]),
                )
                if end < 0.0:
                    # Certified behind the query point: not on the parity ray.
                    continue
                if root.kind != "transversal" or root.certificate != "krawczyk":
                    pending.add(face)
                    continue
                box = np.stack((root.parameter_lower[1:], root.parameter_upper[1:]))
                required = sum(
                    3 * loop.arc_lower.shape[0] + 2 * loop.junction_lower.shape[0]
                    if isinstance(loop, CurveTrimLoop)
                    else 2 * loop.vertices.shape[0]
                    for loop in domains[face].loops
                )
                if (
                    maximum_operations is not None
                    and owner_operations + required > maximum_operations
                ):
                    pending.add(face)
                    exhausted = True
                    continue
                inside, complete, trim_work = _trim_box_decision(
                    domains[face], box, maximum_depth=maximum_depth
                )
                owner_operations += trim_work
                if not complete:
                    pending.add(face)
                    continue
                if inside:
                    # An in-trim root whose enclosure holds 0 is the query point
                    # on the boundary; it never establishes parity.
                    if start <= 0.0 or end >= last:
                        pending.add(face)
                    else:
                        crossings.append((start, end))
        crossings.sort()
        if any(
            first[1] >= second[0]
            for first, second in zip(crossings[:-1], crossings[1:], strict=True)
        ):
            pending.update(faces)
        if not pending:
            return SolidContainmentCertificate(
                bool(len(crossings) % 2),
                True,
                len(crossings),
                processed,
                (),
                ray_start,
                owner_operations,
                False,
            )
        if processed >= maximum_boxes:
            break
    return SolidContainmentCertificate(
        False,
        False,
        len(crossings),
        processed,
        tuple(sorted(pending)),
        ray_start,
        owner_operations,
        exhausted or processed >= maximum_boxes,
    )


__all__ = ["SolidContainmentCertificate", "certify_solid_containment"]
