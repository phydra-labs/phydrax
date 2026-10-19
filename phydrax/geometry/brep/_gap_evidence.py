#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical-length bounds for topology/carrier agreement, separate from identity.

Trim gaps refer to the spatial lift of a p-curve, not distances between unlike
parameter spaces. Paired-coedge bounds compare the two spatial lifts through
their common edge carrier. No geometric coordinates or declared limits change.
"""

from __future__ import annotations

from typing import assert_never, Literal

import equinox as eqx
import numpy as np

from ..._strict import StrictModule
from ..._validation import finite_real_scalar
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, parse
from ._coordinate_request import coordinate_source_request
from ._intersection_curve import IntersectionCurve, IntersectionPCurve
from ._model import BRepCurve, BRepEntityId, BRepGeometry, BRepModel, BRepPCurve
from ._patches import AbstractCurve
from ._request_preparations import TrimRequestPreparations
from ._source_correspondence import norm_upper


type BRepGapKind = Literal[
    "vertex-edge", "vertex-face", "vertex-trim", "edge-edge", "edge-face", "edge-trim"
]


class BRepGapLimitError(ValueError):
    """A declared gap limit was exceeded by an unchanged certified upper bound."""

    @checked
    def __init__(
        self,
        entity: BRepEntityId,
        partner: BRepEntityId,
        kind: BRepGapKind,
        measured: float,
        limit: float,
    ) -> None:
        self.entity = entity
        self.partner = partner
        self.kind = parse(kind, BRepGapKind, "BRepGapLimitError.kind")
        self.measured = measured
        self.limit = limit
        super().__init__(
            f"BRepGapEvidence.{kind} must not exceed limit={limit}; got bound={measured}, entity={entity}"
        )


class BRepGapEvidence(StrictModule):
    """Batched certified gap rows keyed by source entities and oriented use indices.

    Bounds have model length units. Infinite bounds explicitly refuse every
    finite caller limit. Row order is deterministic; evidence is not model data.
    """

    entities: tuple[BRepEntityId, ...] = eqx.field(static=True)
    partners: tuple[BRepEntityId, ...] = eqx.field(static=True)
    kinds: tuple[BRepGapKind, ...] = eqx.field(static=True)
    use_indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    upper_bounds: np.ndarray
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        entities: tuple[BRepEntityId, ...],
        partners: tuple[BRepEntityId, ...],
        kinds: tuple[BRepGapKind, ...],
        use_indices: tuple[tuple[int, int], ...],
        upper_bounds: np.ndarray,
        model_id: str,
    ) -> None:
        bounds = np.array(upper_bounds, dtype=np.float64, copy=True)
        count = len(entities)
        if (
            len(partners) != count
            or len(kinds) != count
            or len(use_indices) != count
            or bounds.shape != (count,)
        ):
            raise ValueError(
                f"BRepGapEvidence.rows must align; got count={count}, bounds={bounds.shape}"
            )
        if np.any(np.isnan(bounds)) or np.any(bounds < 0.0):
            raise ValueError(
                "BRepGapEvidence.upper_bounds must be nonnegative without NaN; got invalid bounds"
            )
        bounds.setflags(write=False)
        self.entities = entities
        self.partners = partners
        self.kinds = tuple(
            parse(kind, BRepGapKind, "BRepGapEvidence.kind") for kind in kinds
        )
        self.use_indices = use_indices
        self.upper_bounds = bounds
        self.model_id = model_id

    @checked
    def bound(self, entity: BRepEntityId, /, *, kind: BRepGapKind | None = None) -> float:
        """Return the largest published bound for one entity; absent evidence refuses."""
        if kind is not None:
            kind = parse(kind, BRepGapKind, "BRepGapEvidence.kind")
        selected = [
            index
            for index, item in enumerate(self.entities)
            if (item == entity or self.partners[index] == entity)
            and (kind is None or self.kinds[index] == kind)
        ]
        if not selected:
            raise ValueError(
                f"BRepGapEvidence.entity must have measured evidence; got {entity}"
            )
        return float(np.max(self.upper_bounds[selected]))

    def require_limit(self, limit: float, /) -> None:
        """Refuse the first deterministic row whose certified bound exceeds a limit."""
        limit = finite_real_scalar(limit, "BRepGapEvidence.limit")
        if limit < 0.0:
            raise ValueError(f"BRepGapEvidence.limit must be nonnegative; got {limit}")
        failed = np.flatnonzero(self.upper_bounds > limit)
        if failed.size:
            index = int(failed[0])
            raise BRepGapLimitError(
                self.entities[index],
                self.partners[index],
                self.kinds[index],
                float(self.upper_bounds[index]),
                limit,
            )


def _box_gap(first: np.ndarray, second: np.ndarray) -> float:
    # Identical singleton source enclosures establish point equality without
    # numerical subtraction or a norm's outward-rounding allowance.
    if np.array_equal(first[0], first[1]) and np.array_equal(first, second):
        return 0.0
    lower = np.nextafter(first[0] - second[1], -np.inf)
    upper = np.nextafter(first[1] - second[0], np.inf)
    return norm_upper(np.maximum(np.abs(lower), np.abs(upper)))


def _carrier_box(
    curve: BRepCurve | BRepPCurve,
    first: float,
    last: float,
    geometry: BRepGeometry,
    edge: int,
    coedge: int | None,
    endpoint: int,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    match curve:
        case IntersectionCurve() | IntersectionPCurve():
            roots = (
                geometry.edge_endpoint_roots[edge]
                if coedge is None
                else geometry.coedge_endpoint_roots[coedge]
            )
            root = roots[endpoint]
            return curve.bounding_box(
                first,
                last,
                endpoint_roots=(root, root),
                preparations=preparations,
                budget=budget,
            )
        case AbstractCurve():
            return curve.bounding_box(
                first, last, preparations=preparations, budget=budget
            )
        case _:
            assert_never(curve)


def _endpoint_bounds(
    model: BRepModel,
    geometry: BRepGeometry,
    points: np.ndarray,
    edge: int,
    endpoint: int,
    coedge: int | None,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    coedge_faces: tuple[int, ...],
) -> float:
    vertex = geometry.edge_vertices[edge][endpoint]
    root = geometry.vertex_roots[vertex]
    vertex_box = (
        np.stack((points[vertex], points[vertex]))
        if root is None
        else root.point_enclosure(preparations=preparations, budget=budget)
    )
    parameter_box = geometry.edge_parameter_enclosure(
        edge, preparations=preparations, budget=budget
    )
    first, last = (float(value) for value in parameter_box[endpoint])
    if coedge is not None:
        definition = geometry.coedge_endpoint_roots[coedge][endpoint]
        if definition is not None:
            first, last = definition.parameter_enclosure(
                preparations=preparations, budget=budget
            )
    if coedge is None:
        curve_index = geometry.edge_curves[edge]
        if curve_index == -1:
            return 0.0
        target = _carrier_box(
            geometry.curves[curve_index],
            first,
            last,
            geometry,
            edge,
            None,
            endpoint=endpoint,
            preparations=preparations,
            budget=budget,
        )
    else:
        uv = _carrier_box(
            geometry.pcurves[coedge],
            first,
            last,
            geometry,
            edge,
            coedge,
            endpoint=endpoint,
            preparations=preparations,
            budget=budget,
        )
        face = coedge_faces[coedge]
        target = model.patches[face].bounding_box(
            uv, preparations=preparations, budget=budget
        )
    return _box_gap(vertex_box, target)


@checked
def certify_brep_gaps(
    model: BRepModel,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
    preparations: TrimRequestPreparations | None = None,
) -> BRepGapEvidence:
    """Publish six kinds of continuously certified carrier agreement in length units.

    Whole-edge bounds reuse the independently certified curve/surface owner,
    including its coupled approximation and Bernstein/analytic interval routes.
    Endpoint bounds use closed source enclosures, not sampled residuals. Work
    and storage are admitted by the request's coordinate batch ticket.
    """
    geometry = model.require_geometry()
    preparations = TrimRequestPreparations() if preparations is None else preparations
    coedge_faces = geometry.coedge_faces
    edge_uses: list[list[int]] = [[] for _ in geometry.edge_curves]
    for coedge, edge in enumerate(geometry.coedge_edges):
        edge_uses[edge].append(coedge)
    count = (
        2 * len(geometry.edge_curves)
        + 6 * len(geometry.coedge_edges)
        + sum(len(uses) * (len(uses) - 1) // 2 for uses in edge_uses)
    )
    with coordinate_source_request((model,), budget=budget, rows=count) as admitted:
        with admitted.measurement():
            admitted.reserve(
                count,
                count * 1024
                + (
                    model.topology.num_vertices
                    + model.topology.num_edges
                    + model.topology.num_faces
                )
                * 512,
            )
            points = np.asarray(geometry.vertex_points)
            correspondence = model.certified_coedge_bounds(
                budget=admitted, preparations=preparations
            )
            edge_ids, vertex_ids, face_ids = (
                model.edge_ids,
                model.vertex_ids,
                model.face_ids,
            )
            entities: list[BRepEntityId] = []
            partners: list[BRepEntityId] = []
            kinds: list[BRepGapKind] = []
            use_indices: list[tuple[int, int]] = []
            bounds = np.empty((count,), dtype=np.float64)

            def append(
                entity: BRepEntityId,
                partner: BRepEntityId,
                kind: BRepGapKind,
                use: tuple[int, int],
                upper: float,
            ) -> None:
                bounds[len(entities)] = upper
                entities.append(entity)
                partners.append(partner)
                kinds.append(kind)
                use_indices.append(use)

            for edge, vertices in enumerate(geometry.edge_vertices):
                edge_id = edge_ids[edge]
                for endpoint, vertex in enumerate(vertices):
                    bound = _endpoint_bounds(
                        model,
                        geometry,
                        points,
                        edge,
                        endpoint,
                        None,
                        preparations=preparations,
                        budget=admitted,
                        coedge_faces=coedge_faces,
                    )
                    append(
                        vertex_ids[vertex], edge_id, "vertex-edge", (endpoint, -1), bound
                    )
                for coedge in edge_uses[edge]:
                    face_id = face_ids[coedge_faces[coedge]]
                    append(
                        edge_id,
                        face_id,
                        "edge-face",
                        (coedge, -1),
                        float(correspondence[coedge]),
                    )
                    append(
                        edge_id,
                        face_id,
                        "edge-trim",
                        (coedge, -1),
                        float(correspondence[coedge]),
                    )
                    for endpoint, vertex in enumerate(vertices):
                        bound = _endpoint_bounds(
                            model,
                            geometry,
                            points,
                            edge,
                            endpoint,
                            coedge,
                            preparations=preparations,
                            budget=admitted,
                            coedge_faces=coedge_faces,
                        )
                        append(
                            vertex_ids[vertex],
                            face_id,
                            "vertex-face",
                            (coedge, endpoint),
                            bound,
                        )
                        append(
                            vertex_ids[vertex],
                            face_id,
                            "vertex-trim",
                            (coedge, endpoint),
                            bound,
                        )
                for position, first in enumerate(edge_uses[edge]):
                    for second in edge_uses[edge][position + 1 :]:
                        total = correspondence[first] + correspondence[second]
                        # Two proved zero residuals share the same edge carrier.
                        upper = (
                            0.0 if total == 0.0 else float(np.nextafter(total, np.inf))
                        )
                        append(edge_id, edge_id, "edge-edge", (first, second), upper)
            return BRepGapEvidence(
                entities=tuple(entities),
                partners=tuple(partners),
                kinds=tuple(kinds),
                use_indices=tuple(use_indices),
                upper_bounds=bounds,
                model_id=model.model_id,
            )


__all__ = ["BRepGapEvidence", "BRepGapKind", "BRepGapLimitError", "certify_brep_gaps"]
