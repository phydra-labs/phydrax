#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded ray-parity evidence for host CAD solid containment."""

from __future__ import annotations

from typing import assert_never

import equinox as eqx
import numpy as np

from ..._bvh import prepare_bvh
from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, ConvertibleToArray
from .._atlas import CurveTrimLoop, DynamicCurveTrimLoop, PolygonTrimLoop, TrimDomain
from .._rational import rational_vector
from ._affine_query import (
    AffineFace,
    certify_affine_parity,
    prepare_affine_faces,
    QueryBroadphase,
)
from ._constructors import brep_trim_domain
from ._coordinate_request import coordinate_source_request
from ._intersection import (
    CurveIntersectionResult,
    intersect_curve_region,
    ParametricIntersectionPolicy,
)
from ._intersection_curve import CurveRange, SurfaceRegion
from ._model import BRepModel
from ._patches import LineCurve
from ._request_preparations import TrimRequestPreparations


class SolidContainmentCertificate(StrictModule):
    """Return parity and work evidence for a solid query.

    Curved-carrier searches begin at ``ray_start <= 0`` so a boundary root is
    interior to the search range. Boundary points never establish parity.
    """

    inside: bool = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    crossings: int = eqx.field(static=True)
    boxes_processed: int = eqx.field(static=True)
    unresolved_faces: tuple[int, ...] = eqx.field(static=True)
    ray_start: float = eqx.field(static=True)
    owner_operations: int = eqx.field(static=True)
    resource_exhausted: bool = eqx.field(static=True)


class _ContainmentPreparation(StrictModule):
    """Retain validated host inputs and the source-face index mapping."""

    query: np.ndarray
    faces: tuple[int, ...]
    parameter_boxes: np.ndarray
    domains: tuple[TrimDomain, ...]
    broadphase: QueryBroadphase
    source_indices: tuple[int, ...]
    upper: np.ndarray
    extent: float


class _AffineContainmentPreparation(StrictModule):
    """Retain admitted affine faces without constructing curved trim covers."""

    query: np.ndarray
    faces: tuple[int, ...]
    affine: tuple[AffineFace, ...]


class _RayEvidence(StrictModule):
    """Retain one ray's accepted roots and cumulative work."""

    crossings: tuple[tuple[float, float], ...]
    pending: tuple[int, ...]
    processed: int
    operations: int
    exhausted: bool
    admitted: bool


class _TrimmedRoots(StrictModule):
    """Retain trim decisions for one face's isolated roots."""

    crossings: tuple[tuple[float, float], ...]
    unresolved: bool
    operations: int
    exhausted: bool


def _trim_work(domain: TrimDomain, /) -> int:
    return sum(
        3 * loop.arc_lower.shape[0] + 2 * loop.junction_lower.shape[0]
        if isinstance(loop, (CurveTrimLoop, DynamicCurveTrimLoop))
        else 2 * loop.vertices.shape[0]
        for loop in domain.loops
    )


def _trim_box_decision(
    domain: TrimDomain,
    box: np.ndarray,
    /,
    *,
    maximum_depth: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> tuple[bool, bool, int]:
    """Classify a connected box disjoint from every trim boundary cover."""
    operations = 0
    for loop in domain.loops:
        match loop:
            case CurveTrimLoop():
                operations += loop.arc_lower.shape[0]
                if np.any(
                    np.all(
                        (loop.arc_upper >= box[0]) & (loop.arc_lower <= box[1]), axis=1
                    )
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
            case PolygonTrimLoop():
                operations += loop.vertices.shape[0]
                first, second = loop.vertices, np.roll(loop.vertices, -1, axis=0)
                lower, upper = np.minimum(first, second), np.maximum(first, second)
                if np.any(np.all((upper >= box[0]) & (lower <= box[1]), axis=1)):
                    return False, False, operations
            case _:
                raise TypeError(
                    f"containment.trim_loop must be a curve or polygon loop; got {type(loop).__name__}"
                )
    decision = domain.classify(
        (0.5 * (box[0] + box[1]))[None],
        maximum_depth=maximum_depth,
        preparations=preparations,
        budget=budget,
    )
    for loop in domain.loops:
        operations += (
            2 * loop.arc_lower.shape[0] + loop.junction_lower.shape[0]
            if isinstance(loop, (CurveTrimLoop, DynamicCurveTrimLoop))
            else loop.vertices.shape[0]
        )
    return (
        bool(decision.inside[0]),
        bool(decision.resolved[0]),
        operations + int(decision.refinements[0]),
    )


def _prepare_containment(
    model: BRepModel,
    point: ConvertibleToArray,
    solid: int,
    *,
    maximum_boxes: int,
    maximum_depth: int,
    maximum_operations: int | None,
    source_boxes: np.ndarray | None,
    trim_domains: tuple[TrimDomain, ...] | None,
    broadphase: QueryBroadphase | None,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> (
    _ContainmentPreparation | _AffineContainmentPreparation | SolidContainmentCertificate
):
    if model.geometry is None or not 0 <= solid < model.topology.num_solids:
        raise ValueError(
            f"containment.solid must index a model with geometry; got {solid}"
        )
    query = np.asarray(point, dtype=np.float64)
    if (
        query.shape != (3,)
        or not np.all(np.isfinite(query))
        or maximum_boxes < 1
        or maximum_depth < 0
    ):
        raise ValueError(
            "containment.point_and_limits must specify a finite (3,) point, positive box limit "
            f"and nonnegative depth; got shape={query.shape}, boxes={maximum_boxes}, depth={maximum_depth}"
        )
    if maximum_operations is not None and (
        type(maximum_operations) is not int or maximum_operations < 0
    ):
        raise ValueError(
            "containment.maximum_operations must be a nonnegative integer or None; "
            f"got {maximum_operations!r}"
        )
    faces = model.topology.solid_faces[solid]
    parameter_boxes = np.asarray(model.parameter_bounds)
    if source_boxes is None:
        source_boxes = np.asarray(
            [
                model.patches[face].bounding_box(
                    parameter_boxes[face], preparations=preparations, budget=budget
                )
                for face in faces
            ]
        )
    elif source_boxes.shape != (len(faces), 2, 3):
        raise ValueError(
            f"containment.source_boxes must have shape {(len(faces), 2, 3)}; got {source_boxes.shape}"
        )
    if maximum_operations == 0:
        return SolidContainmentCertificate(
            inside=False,
            complete=False,
            crossings=0,
            boxes_processed=0,
            unresolved_faces=faces,
            ray_start=0.0,
            owner_operations=0,
            resource_exhausted=True,
        )
    lower, upper = np.min(source_boxes[:, 0], axis=0), np.max(source_boxes[:, 1], axis=0)
    if np.any(query < lower) or np.any(query > upper):
        return SolidContainmentCertificate(
            inside=False,
            complete=True,
            crossings=0,
            boxes_processed=0,
            unresolved_faces=(),
            ray_start=0.0,
            owner_operations=1,
            resource_exhausted=False,
        )
    extent = float(np.max(upper - lower))
    if not np.isfinite(extent) or extent <= 0:
        return SolidContainmentCertificate(
            inside=False,
            complete=False,
            crossings=0,
            boxes_processed=0,
            unresolved_faces=faces,
            ray_start=0.0,
            owner_operations=1,
            resource_exhausted=False,
        )
    if trim_domains is not None and len(trim_domains) != len(faces):
        raise ValueError(
            f"containment.trim_domains must contain {len(faces)} face domains; got {len(trim_domains)}"
        )
    affine = prepare_affine_faces(model, faces, budget=budget)
    if affine is not None:
        return _AffineContainmentPreparation(query=query, faces=faces, affine=affine)
    if trim_domains is None:
        domains = tuple(
            brep_trim_domain(
                model.geometry,
                face,
                model.patches[face],
                tolerance=0.01,
                budget=budget,
                preparations=preparations,
            )
            for face in faces
        )
    else:
        domains = trim_domains
    source_indices = faces if broadphase is None else tuple(range(len(model.patches)))
    if broadphase is None:
        broadphase = QueryBroadphase.prepare(
            prepare_bvh(source_boxes[:, 0], source_boxes[:, 1])
        )
    return _ContainmentPreparation(
        query=query,
        faces=faces,
        parameter_boxes=parameter_boxes,
        domains=domains,
        broadphase=broadphase,
        source_indices=source_indices,
        upper=upper,
        extent=extent,
    )


def _classify_face_roots(
    intersection: CurveIntersectionResult,
    domain: TrimDomain,
    last: float,
    operations: int,
    *,
    maximum_operations: int | None,
    maximum_depth: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> _TrimmedRoots:
    crossings: list[tuple[float, float]] = []
    unresolved = False
    exhausted = False
    for root in intersection.points:
        start, end = float(root.parameter_lower[0]), float(root.parameter_upper[0])
        if end < 0.0:
            continue
        if root.kind != "crossing" or root.certificate != "krawczyk":
            unresolved = True
            continue
        box = np.stack((root.parameter_lower[1:], root.parameter_upper[1:]))
        required = _trim_work(domain)
        if maximum_operations is not None and operations + required > maximum_operations:
            unresolved = True
            exhausted = True
            continue
        inside, complete, trim_work = _trim_box_decision(
            domain,
            box,
            maximum_depth=maximum_depth,
            preparations=preparations,
            budget=budget,
        )
        operations += trim_work
        if not complete:
            unresolved = True
        elif inside:
            # A root enclosing zero is a boundary query, not a crossing.
            if start <= 0.0 or end >= last:
                unresolved = True
            else:
                crossings.append((start, end))
    return _TrimmedRoots(
        crossings=tuple(crossings),
        unresolved=unresolved,
        operations=operations,
        exhausted=exhausted,
    )


def _curved_ray(
    model: BRepModel,
    prepared: _ContainmentPreparation,
    direction: np.ndarray,
    previous: _RayEvidence,
    ray_start: float,
    *,
    maximum_boxes: int,
    maximum_operations: int | None,
    maximum_depth: int,
    relative_resolution: float,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> _RayEvidence:
    broadphase = prepared.broadphase
    axis = int(np.argmax(direction))
    last = float(
        np.nextafter(
            (prepared.upper[axis] - prepared.query[axis] + prepared.extent)
            / direction[axis],
            np.inf,
        )
    )
    ray = CurveRange(LineCurve(prepared.query, direction), ray_start, last)
    crossings: list[tuple[float, float]] = []
    pending: set[int] = set()
    processed, operations, exhausted = (
        previous.processed,
        previous.operations,
        previous.exhausted,
    )
    if (
        maximum_operations is not None
        and operations + broadphase.lower.shape[0] > maximum_operations
    ):
        return _RayEvidence(
            crossings=(),
            pending=tuple(sorted(prepared.faces)),
            processed=processed,
            operations=operations,
            exhausted=True,
            admitted=False,
        )
    work = [0]
    query, vector = rational_vector(prepared.query), rational_vector(direction)
    offset = rational_vector((ray_start,))[0]
    start = tuple(
        value + offset * component for value, component in zip(query, vector, strict=True)
    )
    candidates = {
        prepared.source_indices[index] for index in broadphase.ray(start, vector, work)
    }
    operations += work[0]
    for face, domain in zip(prepared.faces, prepared.domains, strict=True):
        if face not in candidates:
            continue
        remaining = maximum_boxes - processed
        if maximum_operations is not None:
            remaining = min(remaining, maximum_operations - operations - 1)
        operations += 1 if remaining > 0 else 0
        if remaining < 1:
            pending.add(face)
            exhausted = True
            continue
        intersection = intersect_curve_region(
            ray,
            SurfaceRegion(
                model.patches[face],
                prepared.parameter_boxes[face],
                budget=budget,
                preparations=preparations,
            ),
            policy=ParametricIntersectionPolicy(
                maximum_boxes=remaining, relative_resolution=relative_resolution
            ),
            preparations=preparations,
            coordinate_budget=budget,
        )
        processed += intersection.work.boxes_processed
        operations += intersection.work.boxes_processed
        if not intersection.complete or intersection.coincident:
            pending.add(face)
            continue
        roots = _classify_face_roots(
            intersection,
            domain,
            last,
            operations,
            maximum_operations=maximum_operations,
            maximum_depth=maximum_depth,
            preparations=preparations,
            budget=budget,
        )
        operations = roots.operations
        exhausted |= roots.exhausted
        crossings.extend(roots.crossings)
        if roots.unresolved:
            pending.add(face)
    crossings.sort()
    if any(
        first[1] >= second[0]
        for first, second in zip(crossings[:-1], crossings[1:], strict=True)
    ):
        pending.update(prepared.faces)
    return _RayEvidence(
        crossings=tuple(crossings),
        pending=tuple(sorted(pending)),
        processed=processed,
        operations=operations,
        exhausted=exhausted,
        admitted=True,
    )


def _certify_prepared(
    model: BRepModel,
    preparation: _ContainmentPreparation
    | _AffineContainmentPreparation
    | SolidContainmentCertificate,
    *,
    maximum_boxes: int,
    maximum_depth: int,
    relative_resolution: float,
    maximum_operations: int | None,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> SolidContainmentCertificate:
    match preparation:
        case SolidContainmentCertificate():
            return preparation
        case _AffineContainmentPreparation():
            parity = certify_affine_parity(
                preparation.query,
                preparation.affine,
                preparation.faces,
                maximum_operations=(
                    None if maximum_operations is None else maximum_operations - 1
                ),
            )
            return SolidContainmentCertificate(
                inside=parity.inside,
                complete=parity.complete,
                crossings=parity.crossings,
                boxes_processed=0,
                unresolved_faces=tuple(sorted(parity.unresolved_faces)),
                ray_start=0.0,
                owner_operations=1 + parity.operations,
                resource_exhausted=parity.resource_exhausted,
            )
        case _ContainmentPreparation():
            prepared = preparation
        case _:
            assert_never(preparation)
    ray_start = -float(np.ldexp(prepared.extent, -20))
    evidence = _RayEvidence(
        crossings=(),
        pending=(),
        processed=0,
        operations=1,
        exhausted=False,
        admitted=True,
    )
    for values in ((1.0, 0.317, 0.613), (0.487, 1.0, 0.293), (0.211, 0.719, 1.0)):
        evidence = _curved_ray(
            model,
            prepared,
            np.asarray(values, dtype=np.float64),
            evidence,
            ray_start,
            maximum_boxes=maximum_boxes,
            maximum_operations=maximum_operations,
            maximum_depth=maximum_depth,
            relative_resolution=relative_resolution,
            preparations=preparations,
            budget=budget,
        )
        if not evidence.pending or evidence.processed >= maximum_boxes:
            break
        if not evidence.admitted:
            break
    complete = not evidence.pending
    return SolidContainmentCertificate(
        inside=bool(len(evidence.crossings) % 2) if complete else False,
        complete=complete,
        crossings=len(evidence.crossings),
        boxes_processed=evidence.processed,
        unresolved_faces=evidence.pending,
        ray_start=ray_start,
        owner_operations=evidence.operations,
        resource_exhausted=False
        if complete
        else evidence.exhausted or evidence.processed >= maximum_boxes,
    )


@checked
def certify_solid_containment(
    model: BRepModel,
    point: ConvertibleToArray,
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
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> SolidContainmentCertificate:
    """Certify solid parity, refusing boundary queries and unresolved curved roots.

    Planar faces with affine polygon loops use rational half-open winding and a
    direction outside finitely many tangent and edge-plane exclusions. These
    queries are unresolved only on the boundary or on resource exhaustion.
    Root-defined and non-affine trims retain source-intersection certification.

    Curved faces retain three alternate ray directions. A negative search start
    makes a root on the query's supporting plane interior to the search range;
    only strictly positive in-trim roots count toward parity. Work and unresolved
    faces are retained when the declared allowance is exhausted.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request((model, point, solid)) as request_budget:
            prepared = _prepare_containment(
                model,
                point,
                solid,
                maximum_boxes=maximum_boxes,
                maximum_depth=maximum_depth,
                maximum_operations=maximum_operations,
                source_boxes=source_boxes,
                trim_domains=trim_domains,
                broadphase=broadphase,
                preparations=preparations,
                budget=request_budget,
            )
            return _certify_prepared(
                model,
                prepared,
                maximum_boxes=maximum_boxes,
                maximum_depth=maximum_depth,
                relative_resolution=relative_resolution,
                maximum_operations=maximum_operations,
                preparations=preparations,
                budget=request_budget,
            )
    prepared = _prepare_containment(
        model,
        point,
        solid,
        maximum_boxes=maximum_boxes,
        maximum_depth=maximum_depth,
        maximum_operations=maximum_operations,
        source_boxes=source_boxes,
        trim_domains=trim_domains,
        broadphase=broadphase,
        preparations=preparations,
        budget=budget,
    )
    return _certify_prepared(
        model,
        prepared,
        maximum_boxes=maximum_boxes,
        maximum_depth=maximum_depth,
        relative_resolution=relative_resolution,
        maximum_operations=maximum_operations,
        preparations=preparations,
        budget=budget,
    )


__all__ = ["SolidContainmentCertificate", "certify_solid_containment"]
