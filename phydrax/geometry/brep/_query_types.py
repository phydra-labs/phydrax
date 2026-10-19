#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared BRep query carrier metadata and batch hit records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import final

import equinox as eqx
import numpy as np
from jax import Array

from ..._bvh import PackedBVH
from ..._resource_ticket import BatchTicket
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import nonnegative_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from .._atlas import TrimDomain
from .._interval_enclosure import (
    interval_add,
    interval_multiply,
    PreparedIntervalFunction,
)
from ._coordinate_request import coordinate_source_request
from ._interval_source import evaluate_interval
from ._model import BRepCurve, BRepPCurve
from ._patches import AbstractSurfacePatch
from ._request_preparations import TrimRequestPreparations


@final
class Tolerances(StrictModule, NonTrainableState):
    """Policy metadata and dynamic, translation-independent source length."""

    distance: float = eqx.field(static=True)
    parametric: float = eqx.field(static=True)
    classifier: float = eqx.field(static=True)
    scale: Array
    newton_steps: int = eqx.field(static=True)
    seeds: int = eqx.field(static=True)


class EdgeData(StrictModule):
    curve: BRepCurve | None
    start_point: Array
    end_point: Array
    seed_parameters: Array
    seed_points: Array
    interval_boxes: Array
    discovery_bvh: PackedBVH
    first: float | Array
    last: float | Array
    closed: bool = eqx.field(static=True)


class FaceData(StrictModule):
    patch: AbstractSurfacePatch
    seed_uv: Array
    seed_points: Array
    trim_domain: TrimDomain
    interval_boxes: Array
    discovery_bvh: PackedBVH
    parameter_boxes: Array
    pole_points: Array
    pole_parameters: Array
    pcurves: tuple[BRepPCurve, ...]
    carrier_center: Array
    carrier_projection: Array
    carrier_radii: Array
    lower: tuple[float, float] | Array
    upper: tuple[float, float] | Array
    periods: tuple[float, float] = eqx.field(static=True)
    orientation: float = eqx.field(static=True)
    coedge_edges: tuple[int, ...] = eqx.field(static=True)
    seam_coedges: tuple[bool, ...] = eqx.field(static=True)
    full_rectangle: bool = eqx.field(static=True)


@dataclass(frozen=True, slots=True, eq=False)
class FaceCertificate:
    face: int
    piece_index: int
    lower: tuple[float, float]
    upper: tuple[float, float]
    value: PreparedIntervalFunction
    gradient: PreparedIntervalFunction
    hessian: PreparedIntervalFunction

    def bound(
        self,
        box: np.ndarray,
        point: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> float:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    box,
                    point,
                )
            ) as budget:
                return self.bound(box, point, budget=budget, preparations=preparations)
        middle = 0.5 * (box[0] + box[1])
        center = np.concatenate((middle, point))[None]
        lower_value, _ = evaluate_interval(
            self.value, center, center, budget=budget, preparations=preparations
        )
        lower = np.concatenate((box[0], point))[None]
        upper = np.concatenate((box[1], point))[None]
        gradient = self.gradient.evaluate(lower, upper)
        offset = (
            np.nextafter(box[0] - middle, -np.inf),
            np.nextafter(box[1] - middle, np.inf),
        )
        raw_terms = interval_multiply((gradient[0][0], gradient[1][0]), offset)
        terms = (
            np.asarray(raw_terms[0], dtype=np.float64),
            np.asarray(raw_terms[1], dtype=np.float64),
        )
        total = (lower_value[0], lower_value[0])
        for axis in range(2):
            total = interval_add(total, (terms[0][axis], terms[1][axis]))
        return float(np.nextafter(np.sqrt(max(0.0, float(total[0]))), -np.inf))


@final
class QuadricHit(StrictModule):
    """Quadric hit batch."""

    point: Array
    parameter: Array
    parameter_bound: Array
    point_lower: Array
    point_upper: Array
    point_error: Array
    unique: Array
    pole: Array
    normal: Array


@final
class EdgeHit(StrictModule):
    """Edge hit batch."""

    point: Array
    parameter: Array
    distance: Array
    status: Array
    tangent: Array
    lower_bound: Array
    unresolved: Array
    parameter_bound: Array
    proposal_status: Array


@final
class FaceHit(StrictModule):
    """Face hit batch."""

    point: Array
    parameter: Array
    distance: Array
    status: Array
    normal: Array
    du: Array
    dv: Array
    stationarity: Array
    lower_bound: Array
    unresolved: Array
    parameter_bound: Array
    edge_index: Array
    edge_parameter: Array
    multiplicity: Array
    point_error: Array


@final
class LowerEntityHit(StrictModule):
    """LowerEntity hit batch."""

    point: Array
    distance: Array
    parameter: Array
    tangent: Array
    status: Array
    lower_bound: Array
    occurrence: Array


@final
class ProjectionHit(StrictModule):
    """Projection result fields shared by every entity dimension."""

    point: Array
    distance: Array
    status: Array
    occurrence: Array


class BRepQueryResourceError(RuntimeError):
    """A query refuses owner work before consuming an insufficient allowance."""

    def __init__(self, resource: str, requested: int, remaining: int, /) -> None:
        self.resource, self.requested, self.remaining = resource, requested, remaining
        super().__init__(
            f"BRepQueryBudget.{resource} must have allowance for {requested}; got remaining allowance {remaining}"
        )


@dataclass(slots=True)
class BRepQueryBudget:
    """Shared host allowance for point, carrier, trim, BVH and interval queries.

    Counts executed source operations, not Newton steps or arithmetic primitives.
    Admission reserves the worst-case cover for all points and sources before work;
    only executed operations are consumed. Source preparation is not a point
    query and does not consume candidate-pair allowance.
    """

    maximum_operations: int
    maximum_points: int | None = None
    maximum_scratch_bytes: int | None = None
    ticket: BatchTicket | None = field(default=None, kw_only=True)
    coordinate_budget: CoordinateEnclosureBudget | None = field(
        default=None, kw_only=True
    )
    operations: int = field(init=False, default=0)
    points: int = field(init=False, default=0)
    subdivisions: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.maximum_operations = nonnegative_integer(
            self.maximum_operations, "maximum_operations"
        )
        if self.maximum_points is not None:
            self.maximum_points = nonnegative_integer(
                self.maximum_points, "maximum_points"
            )
        if self.maximum_scratch_bytes is not None:
            self.maximum_scratch_bytes = nonnegative_integer(
                self.maximum_scratch_bytes, "maximum_scratch_bytes"
            )

    def admit(self, operations: int, points: int, scratch_bytes: int, /) -> None:
        operations = nonnegative_integer(operations, "operations")
        points = nonnegative_integer(points, "points")
        scratch_bytes = nonnegative_integer(scratch_bytes, "scratch_bytes")
        for resource, requested, remaining in (
            ("operations", operations, self.maximum_operations - self.operations),
            (
                "points",
                points,
                None
                if self.maximum_points is None
                else self.maximum_points - self.points,
            ),
            ("scratch_bytes", scratch_bytes, self.maximum_scratch_bytes),
        ):
            if remaining is not None and requested > remaining:
                raise BRepQueryResourceError(resource, requested, remaining)

    def consume(
        self, operations: int, /, *, points: int = 0, subdivisions: int = 0
    ) -> None:
        operations = nonnegative_integer(operations, "operations")
        points = nonnegative_integer(points, "points")
        subdivisions = nonnegative_integer(subdivisions, "subdivisions")
        self.admit(operations, points, 0)
        if self.ticket is not None:
            if operations > self.ticket.remaining_work_units:
                self.ticket.refuse(operations, queries_upper=points)
            if points > self.ticket.remaining_geometry_queries:
                self.ticket.refuse(operations, queries_upper=points)
            self.ticket.consume(operations, geometry_queries=points)
        self.operations += operations
        self.points += points
        self.subdivisions += subdivisions
