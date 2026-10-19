#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded certified curve/curve, curve/surface and surface/surface intersection.

Every route subdivides the coupled parameter box with rigorous interval
enclosures of the defining equations (`_interval_enclosure`), excludes boxes
whose enclosure omits zero, and certifies existence and uniqueness of regular
roots with a Krawczyk inclusion preconditioned by the native small linear
solve. Local corrections use `VectorLocalRootPlan`. Tangent contacts are
certified through a regular foot-point/parallel-normal system and reported with
a gap bound; coincident components use a normal-distance and projection-coverage
test against the codimension-one partner. Surface/surface branches start from
certified boundary points and from turning points of a fixed generic linear
functional (every closed interior loop has one) and are continued with a
predictor/corrector whose every step is certified as a parametric Krawczyk
chart. Boxes that are neither excluded, certified, coincident nor tangent within
the work budget are reported as unresolved, so completeness is explicit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from fractions import Fraction, Fraction as F
from itertools import combinations
from typing import Literal, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.geometry.brep._intersection_curve as intersection_curve_owner
import phydrax.geometry.brep._placed as placed_owner
import phydrax.geometry.brep._source_correspondence as source_correspondence_owner

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import nonnegative_integer, positive_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...linalg import (
    inverse_small_linear,
    orthonormal_frame,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ...linalg._exact import solve_exact
from ...nonlinear import VectorLocalRootPlan
from ...typing import (
    checked,
    ConvertibleToArray,
    Dim,
    HostBool,
    HostFloat64,
    parse,
    Scope,
)
from .._atlas import AbstractTrimCurve
from .._interval_enclosure import (
    interval_add,
    interval_divide,
    interval_dot,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._rational import (
    fraction_down,
    fraction_up,
    rational_cross,
    rational_dot,
    rational_vector,
    RationalVector,
)
from ._carrier_pieces import SurfacePiece
from ._contact_indicatrix import contact_indicatrix
from ._coordinate_request import coordinate_source_request
from ._intersection_kernel import (
    approximate_inverses,
    interval_magnitude,
    interval_times_interval,
    krawczyk,
    KrawczykEvaluator,
    point_times_interval,
    round_interval,
)
from ._interval_source import (
    evaluate_interval,
)
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BernsteinSurfacePiece,
    BSplineCurve,
    CircleCurve,
    closure_tangent_continuous,
    coefficient_enclosures,
    CylinderPatch,
    ExtrusionSurface,
    LineCurve,
    OffsetSurface,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
)
from ._request_preparations import TrimRequestPreparations
from ._trim_helpers import period_offset, period_scalar_bounds, PeriodOffset


ParametricIntersectionKind: TypeAlias = Literal[
    "crossing", "tangent_point", "tangent_curve", "singular_point"
]
ParametricIntersectionCertificate: TypeAlias = Literal[
    "krawczyk", "tangency-krawczyk", "proximity-krawczyk", "analytic-contact"
]
UnresolvedIntersectionReason: TypeAlias = Literal[
    "budget", "resolution", "singular", "continuation"
]


class _SourceCurveSurfaceLift(Protocol):
    @property
    def curve(self) -> AbstractCurve | intersection_curve_owner.IntersectionCurve: ...

    @property
    def pcurve(self) -> intersection_curve_owner.CurveEvaluator: ...

    @property
    def surface(self) -> intersection_curve_owner.SurfaceRegion: ...

    @property
    def first(self) -> float: ...

    @property
    def last(self) -> float: ...


class ParameterDim(Dim):
    """Coupled intersection parameters of one event."""


class AmbientDim(Dim):
    """Ambient coordinates of an intersection event."""


class CellDim(Dim):
    """Parameter cells of one coincident component."""


# ------------------------------------------------------------------ policy


class ParametricIntersectionPolicy(StrictModule):
    """Work budget and relative tolerances of bounded intersection.

    Lengths are relative to the diagonal of the conservative bounding box of the
    two operands. ``maximum_boxes`` bounds every subdivided box across all
    subsystems of one call; exhausting it reports the remaining boxes unresolved.
    ``relative_junction`` is the parameter radius, relative to the domain widths,
    of the neighborhood summarized by a singular surface contact; the branches
    leaving it are isolated on its boundary.
    """

    maximum_boxes: int = eqx.field(static=True)
    relative_resolution: float = eqx.field(static=True)
    relative_probe: float = eqx.field(static=True)
    relative_coincidence: float = eqx.field(static=True)
    relative_tangency: float = eqx.field(static=True)
    relative_step: float = eqx.field(static=True)
    relative_minimum_step: float = eqx.field(static=True)
    maximum_turn: float = eqx.field(static=True)
    maximum_march_steps: int = eqx.field(static=True)
    batch: int = eqx.field(static=True)
    relative_junction: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_boxes: int = 200_000,
        relative_resolution: float = 1.0e-10,
        relative_probe: float = 1.0e-2,
        relative_coincidence: float = 1.0e-11,
        relative_tangency: float = 1.0e-8,
        relative_step: float = 0.05,
        relative_minimum_step: float = 1.0e-7,
        maximum_turn: float = 0.35,
        maximum_march_steps: int = 20_000,
        batch: int = 128,
        relative_junction: float = 1.0e-3,
    ) -> None:
        for name, value in (
            ("maximum_boxes", maximum_boxes),
            ("maximum_march_steps", maximum_march_steps),
            ("batch", batch),
        ):
            positive_integer(value, name)
        for name, value in (
            ("relative_resolution", relative_resolution),
            ("relative_probe", relative_probe),
            ("relative_coincidence", relative_coincidence),
            ("relative_tangency", relative_tangency),
            ("relative_step", relative_step),
            ("relative_minimum_step", relative_minimum_step),
            ("maximum_turn", maximum_turn),
            ("relative_junction", relative_junction),
        ):
            if not math.isfinite(value) or value <= 0.0:
                field = f"ParametricIntersectionPolicy.{name}"
                raise ValueError(f"{field} must be finite and positive; got {value}")
        if not relative_resolution < relative_probe:
            raise ValueError(
                "ParametricIntersectionPolicy.relative_resolution must be smaller than relative_probe; got invalid ordering"
            )
        if not relative_minimum_step < relative_step:
            raise ValueError(
                "ParametricIntersectionPolicy.relative_minimum_step must be smaller than relative_step; got invalid ordering"
            )
        self.maximum_boxes = maximum_boxes
        self.relative_resolution = float(relative_resolution)
        self.relative_probe = float(relative_probe)
        self.relative_coincidence = float(relative_coincidence)
        self.relative_tangency = float(relative_tangency)
        self.relative_step = float(relative_step)
        self.relative_minimum_step = float(relative_minimum_step)
        self.maximum_turn = float(maximum_turn)
        self.maximum_march_steps = maximum_march_steps
        self.batch = batch
        self.relative_junction = float(relative_junction)


# ----------------------------------------------------------------- results


class ParametricIntersectionWork(StrictModule):
    """Resource evidence of one intersection call."""

    boxes_processed: int = eqx.field(static=True)
    boxes_excluded: int = eqx.field(static=True)
    boxes_certified: int = eqx.field(static=True)
    boxes_coincident: int = eqx.field(static=True)
    boxes_unresolved: int = eqx.field(static=True)
    march_steps: int = eqx.field(static=True)
    charts_certified: int = eqx.field(static=True)
    charts_uncertified: int = eqx.field(static=True)
    budget_exhausted: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        boxes_processed: int,
        boxes_excluded: int,
        boxes_certified: int,
        boxes_coincident: int,
        boxes_unresolved: int,
        march_steps: int,
        charts_certified: int,
        charts_uncertified: int,
        budget_exhausted: bool,
    ) -> None:
        self.boxes_processed = boxes_processed
        self.boxes_excluded = boxes_excluded
        self.boxes_certified = boxes_certified
        self.boxes_coincident = boxes_coincident
        self.boxes_unresolved = boxes_unresolved
        self.march_steps = march_steps
        self.charts_certified = charts_certified
        self.charts_uncertified = charts_uncertified
        self.budget_exhausted = budget_exhausted


class TrimIntersectionRoot(StrictModule):
    """Enclose a UV junction by its source curves and unique interval root box.

    Numerical parameters and points remain bounded realizations of that root.
    """

    __strict_contract__ = True

    first: AbstractTrimCurve
    second: AbstractTrimCurve
    parameter_lower: HostFloat64[Literal[2]]
    parameter_upper: HostFloat64[Literal[2]]
    parameters: HostFloat64[Literal[2]]
    preconditioner: HostFloat64[Literal[2], Literal[2]]
    contraction: float = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        first: AbstractTrimCurve,
        second: AbstractTrimCurve,
        /,
        *,
        parameter_lower: ConvertibleToArray,
        parameter_upper: ConvertibleToArray,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                first,
                second,
                parameter_lower,
                parameter_upper,
            ),
            budget=budget,
        ) as budget:
            lower = parse(
                np.asarray(parameter_lower, dtype=np.float64),
                HostFloat64[Literal[2]],
                "parameter_lower",
            )
            upper = parse(
                np.asarray(parameter_upper, dtype=np.float64),
                HostFloat64[Literal[2]],
                "parameter_upper",
            )
            if (
                not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower >= upper)
            ):
                raise ValueError(
                    "TrimIntersectionRoot.parameter_box must be finite and nonempty; got invalid bounds"
                )
            prepared = prepare(
                _CurvePairSystem(first, second), 2, 1, preparations=preparations
            )
            inclusion = krawczyk(
                prepared,
                lower[None],
                upper[None],
                budget=budget,
                preparations=preparations,
            )
            for factor in (32.0, 256.0, 2048.0):
                if bool(inclusion.certified[0]):
                    break
                radius = factor * _UNIT * (1 + np.maximum(np.abs(lower), np.abs(upper)))
                lower = np.nextafter(lower - radius, -np.inf)
                upper = np.nextafter(upper + radius, np.inf)
                inclusion = krawczyk(
                    prepared,
                    lower[None],
                    upper[None],
                    budget=budget,
                    preparations=preparations,
                )
            if not bool(inclusion.certified[0]):
                raise ValueError(
                    "TrimIntersectionRoot.parameter_box must certify a regular unique root; got unresolved inclusion"
                )
            roots, _, _, _ = prepared.newton(
                jnp.asarray((0.5 * (lower + upper))[None]),
                VectorLocalRootPlan(
                    2, maximum_steps=40, tolerance=1e-13, plan_id="trim-junction"
                ),
            )
            self.first, self.second = first, second
            self.parameter_lower, self.parameter_upper = lower, upper
            self.parameters = np.clip(np.asarray(roots)[0], lower, upper)
            self.preconditioner = inclusion.preconditioner[0]
            self.contraction = float(inclusion.contraction[0])
            self.root_id = canonical_fingerprint(
                {
                    "kind": "trim-intersection-root",
                    "curves": array_tree_fingerprint((first, second)),
                    "box": array_tree_fingerprint((lower, upper)),
                }
            )

    def parameter_enclosure(
        self,
        /,
        *,
        maximum_steps: int = 16,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.parameter_enclosure(
                    maximum_steps=maximum_steps, preparations=preparations, budget=budget
                )
        maximum_steps = nonnegative_integer(maximum_steps, "maximum_steps")
        return _trim_root_parameter_enclosure(
            self, maximum_steps, preparations=preparations, budget=budget
        )

    def point_enclosure(
        self,
        /,
        *,
        maximum_steps: int = 16,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.point_enclosure(
                    maximum_steps=maximum_steps, preparations=preparations, budget=budget
                )
        parameters = self.parameter_enclosure(
            maximum_steps=maximum_steps, preparations=preparations, budget=budget
        )
        first = np.stack(
            intersection_curve_owner.trim_source_jet(
                self.first,
                float(parameters[0, 0]),
                float(parameters[1, 0]),
                0,
                preparations=preparations,
                budget=budget,
            )
        )
        second = np.stack(
            intersection_curve_owner.trim_source_jet(
                self.second,
                float(parameters[0, 1]),
                float(parameters[1, 1]),
                0,
                preparations=preparations,
                budget=budget,
            )
        )
        lower, upper = np.maximum(first[0], second[0]), np.minimum(first[1], second[1])
        if np.any(lower > upper):
            raise ValueError(
                "TrimIntersectionRoot.point_enclosure must intersect both source images; got disjoint enclosures"
            )
        return np.stack((lower, upper))

    def evaluate(self) -> tuple[np.ndarray, float, bool]:
        with coordinate_source_request((self,)) as budget:
            point = np.asarray(self.first.evaluate(jnp.asarray(self.parameters[0])))
            box = self.point_enclosure(budget=budget)
            bound = float(np.max(np.nextafter(np.abs(box - point), np.inf)))
            return point, bound, True


@contextmanager
def original_trim_root_preparation_scope(
    curves: tuple[AbstractTrimCurve, ...],
    /,
    *,
    preparations: TrimRequestPreparations,
) -> Iterator[TrimRequestPreparations]:
    """Register trim roots in the explicitly borrowed request mapping."""
    sources = jax.tree_util.tree_leaves(
        curves, is_leaf=lambda value: isinstance(value, TrimIntersectionRoot)
    )
    for source in sources:
        if isinstance(source, TrimIntersectionRoot):
            preparations.trim_roots.setdefault(id(source), (source, None, {}))
    yield preparations


def _trim_root_parameter_enclosure(
    root: TrimIntersectionRoot,
    maximum_steps: int,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    owners = None if preparations is None else preparations.trim_roots
    entry = None if owners is None else owners.get(id(root))
    if entry is not None and entry[0] is root:
        cached = entry[2].get(maximum_steps)
        if cached is not None:
            return cached.copy()
        prepared = entry[1]
    else:
        prepared = None
    if prepared is None:
        prepared = prepare(
            _CurvePairSystem(root.first, root.second), 2, 1, preparations=preparations
        )
        if owners is not None and entry is not None and entry[0] is root:
            entry = (root, prepared, entry[2])
            owners[id(root)] = entry
    lower, upper = _contract(
        prepared,
        root.parameter_lower[None],
        root.parameter_upper[None],
        maximum_steps=maximum_steps,
        budget=budget,
        preparations=preparations,
    )
    result = np.stack((lower[0], upper[0]))
    if entry is not None and entry[0] is root:
        entry[2][maximum_steps] = result
    return result.copy()


class CurveSurfaceIntersectionRoot(StrictModule):
    """A source-backed spatial junction on a curve and another surface."""

    __strict_contract__ = True

    curve: AbstractCurve | intersection_curve_owner.IntersectionCurve
    surface: intersection_curve_owner.SurfaceRegion
    parameter_lower: HostFloat64[Literal[3]]
    parameter_upper: HostFloat64[Literal[3]]
    parameters: HostFloat64[Literal[3]]
    preconditioner: HostFloat64[Literal[3], Literal[3]]
    contraction: float = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        curve: AbstractCurve | intersection_curve_owner.IntersectionCurve,
        surface: intersection_curve_owner.SurfaceRegion,
        /,
        *,
        parameter_lower: ConvertibleToArray,
        parameter_upper: ConvertibleToArray,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                curve,
                surface,
                parameter_lower,
                parameter_upper,
            ),
            budget=budget,
        ) as budget:
            if curve.ambient_dimension != 3:
                raise TypeError(
                    f"CurveSurfaceIntersectionRoot.curve must have dimension three; got {curve.ambient_dimension}"
                )
            lower = parse(
                np.asarray(parameter_lower, dtype=np.float64),
                HostFloat64[Literal[3]],
                "parameter_lower",
            )
            upper = parse(
                np.asarray(parameter_upper, dtype=np.float64),
                HostFloat64[Literal[3]],
                "parameter_upper",
            )
            if (
                not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower >= upper)
            ):
                raise ValueError(
                    "CurveSurfaceIntersectionRoot.parameter_box must be finite and nonempty; got invalid bounds"
                )
            if isinstance(curve, intersection_curve_owner.IntersectionCurve):
                evaluator = curve
            else:
                pieces = intersection_curve_owner.curve_pieces(
                    intersection_curve_owner.CurveRange(
                        curve, float(lower[0]), float(upper[0])
                    ),
                    budget=budget,
                )
                if len(pieces) != 1:
                    raise ValueError(
                        "CurveSurfaceIntersectionRoot.parameter_box must lie in one curve span; got multiple spans"
                    )
                evaluator = pieces[0].evaluator
            # The span is selected by the root's source box: a root on a
            # region wall (shared with a neighboring face) belongs to the patch law.
            spans = intersection_curve_owner.surface_pieces_for_box(
                surface.patch, np.stack((lower[1:], upper[1:])), budget=budget
            )
            if len(spans) != 1:
                raise ValueError(
                    "CurveSurfaceIntersectionRoot.parameter_box must lie in one surface span; got multiple spans"
                )
            prepared = prepare(
                CurveSurfaceSystem(evaluator, spans[0].evaluator),
                3,
                1,
                preparations=preparations,
            )
            inclusion = krawczyk(
                prepared,
                lower[None],
                upper[None],
                budget=budget,
                preparations=preparations,
            )
            # Contracted event boxes can meet their outward rounding floor and no
            # longer pass a STRICT re-inclusion on restoration. Re-isolate in a
            # slightly larger source box; only a fresh inclusion admits the root.
            for factor in (32.0, 256.0, 2048.0):
                if bool(inclusion.certified[0]):
                    break
                radius = factor * _UNIT * (1 + np.maximum(np.abs(lower), np.abs(upper)))
                lower = np.nextafter(lower - radius, -np.inf)
                upper = np.nextafter(upper + radius, np.inf)
                inclusion = krawczyk(
                    prepared,
                    lower[None],
                    upper[None],
                    budget=budget,
                    preparations=preparations,
                )
            if not bool(inclusion.certified[0]):
                raise ValueError(
                    "CurveSurfaceIntersectionRoot.parameter_box must certify a regular unique root; got unresolved inclusion"
                )
            roots, _, _, _ = prepared.newton(
                jnp.asarray((0.5 * (lower + upper))[None]),
                VectorLocalRootPlan(
                    3, maximum_steps=40, tolerance=1e-13, plan_id="spatial-junction"
                ),
            )
            self.curve, self.surface = curve, surface
            self.parameter_lower, self.parameter_upper = lower, upper
            self.parameters = np.clip(np.asarray(roots)[0], lower, upper)
            self.preconditioner = inclusion.preconditioner[0]
            self.contraction = float(inclusion.contraction[0])
            self.root_id = canonical_fingerprint(
                {
                    "kind": "curve-surface-intersection-root",
                    "sources": array_tree_fingerprint((curve, surface)),
                    "box": array_tree_fingerprint((lower, upper)),
                }
            )

    def parameter_enclosure(
        self,
        /,
        *,
        maximum_steps: int = 16,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.parameter_enclosure(
                    maximum_steps=maximum_steps, preparations=preparations, budget=budget
                )
        prepared = _curve_surface_root_preparation(
            self, preparations=preparations, budget=budget
        )
        lower, upper = _contract(
            prepared,
            self.parameter_lower[None],
            self.parameter_upper[None],
            maximum_steps=maximum_steps,
            budget=budget,
            preparations=preparations,
        )
        return np.stack((lower[0], upper[0]))

    def point_enclosure(
        self,
        /,
        *,
        maximum_steps: int = 16,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.point_enclosure(
                    maximum_steps=maximum_steps, preparations=preparations, budget=budget
                )
        parameters = self.parameter_enclosure(
            maximum_steps=maximum_steps, preparations=preparations, budget=budget
        )
        curve_box = np.asarray(
            self.curve.bounding_box(
                float(parameters[0, 0]),
                float(parameters[1, 0]),
                preparations=preparations,
                budget=budget,
            )
        )
        surface_box = np.asarray(
            self.surface.patch.bounding_box(
                parameters[:, 1:], preparations=preparations, budget=budget
            )
        )
        lower, upper = (
            np.maximum(curve_box[0], surface_box[0]),
            np.minimum(curve_box[1], surface_box[1]),
        )
        if np.any(lower > upper):
            raise ValueError(
                "CurveSurfaceIntersectionRoot.point_enclosure must intersect both source images; got disjoint enclosures"
            )
        return np.stack((lower, upper))

    def evaluate(self) -> tuple[np.ndarray, float, bool]:
        with coordinate_source_request((self,)) as budget:
            if isinstance(self.curve, intersection_curve_owner.IntersectionCurve):
                point = np.asarray(
                    self.curve.evaluate(
                        jnp.asarray(self.parameters[0]), budget=budget
                    ).point
                )
            else:
                point = np.asarray(self.curve.evaluate(jnp.asarray(self.parameters[0])))
            return (
                point,
                float(
                    np.max(
                        np.nextafter(
                            np.abs(self.point_enclosure(budget=budget) - point), np.inf
                        )
                    )
                ),
                True,
            )


@contextmanager
def original_curve_surface_root_preparation(
    endpoints: tuple[RootEndpoint | None, ...],
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
) -> Iterator[TrimRequestPreparations]:
    """Register spatial roots in one explicitly owned request mapping."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    sources = jax.tree_util.tree_leaves(
        endpoints,
        is_leaf=lambda value: isinstance(value, CurveSurfaceIntersectionRoot),
    )
    request = TrimRequestPreparations() if preparations is None else preparations
    for source in sources:
        if isinstance(source, CurveSurfaceIntersectionRoot):
            request.curve_surface_roots.setdefault(id(source), (source, None))
    yield request


def _curve_surface_root_preparation(
    root: CurveSurfaceIntersectionRoot,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> Prepared:
    owners = None if preparations is None else preparations.curve_surface_roots
    entry = None if owners is None else owners.get(id(root))
    if entry is not None and entry[0] is root and entry[1] is not None:
        return entry[1]
    curve = root.curve
    if not isinstance(curve, intersection_curve_owner.IntersectionCurve):
        curve = intersection_curve_owner.curve_pieces(
            intersection_curve_owner.CurveRange(
                curve,
                float(root.parameter_lower[0]),
                float(root.parameter_upper[0]),
            ),
            budget=budget,
        )[0].evaluator
    surface = intersection_curve_owner.surface_pieces_for_box(
        root.surface.patch,
        np.stack((root.parameter_lower[1:], root.parameter_upper[1:])),
        budget=budget,
    )[0].evaluator
    prepared = prepare(
        CurveSurfaceSystem(curve, surface), 3, 1, preparations=preparations
    )
    if owners is not None and entry is not None and entry[0] is root:
        owners[id(root)] = (root, prepared)
    return prepared


class IntersectionCurvePointRoot(StrictModule):
    """Define a source graph point at one fixed curve parameter atom."""

    __strict_contract__ = True

    curve: intersection_curve_owner.IntersectionCurve
    parameter: float = eqx.field(static=True)
    parameters: HostFloat64[Literal[1]]
    parameter_lower: HostFloat64[Literal[1]]
    parameter_upper: HostFloat64[Literal[1]]
    root_id: str = eqx.field(static=True)
    point_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, curve: intersection_curve_owner.IntersectionCurve, parameter: float, /
    ) -> None:
        if not curve.fully_certified:
            raise ValueError(
                "IntersectionCurvePointRoot.curve must have a certified shared-node atlas; got uncertified charts"
            )
        if not math.isfinite(parameter) or not 0 <= parameter <= curve.num_charts:
            raise ValueError(
                "IntersectionCurvePointRoot.parameter must be finite and lie inside the curve domain; got an invalid parameter"
            )
        self.curve, self.parameter = curve, float(parameter)
        self.parameters = self.parameter_lower = self.parameter_upper = np.asarray(
            [self.parameter], dtype=np.float64
        )
        self.root_id = canonical_fingerprint(
            {
                "kind": "intersection-curve-point-root",
                "curve": curve.branch_id,
                "parameter": self.parameter.hex(),
            }
        )
        phase = 0.0 if curve.closed and parameter == curve.num_charts else self.parameter
        self.point_id = canonical_fingerprint(
            {
                "kind": "intersection-curve-point",
                "curve": curve.branch_id,
                "phase": phase.hex(),
            }
        )

    def parameter_enclosure(
        self,
        /,
        *,
        maximum_steps: int = 16,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.parameter_enclosure(
                    maximum_steps=maximum_steps, preparations=preparations, budget=budget
                )
        if maximum_steps < 0:
            raise ValueError(
                "root.maximum_steps must be nonnegative; got a negative count"
            )
        return np.stack((self.parameter_lower, self.parameter_upper))

    def point_enclosure(
        self,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.point_enclosure(preparations=preparations, budget=budget)
        return self.curve.bounding_box(
            self.parameter, self.parameter, preparations=preparations, budget=budget
        )

    def evaluate(self) -> tuple[np.ndarray, float, bool]:
        with coordinate_source_request((self,)) as budget:
            point = np.asarray(
                self.curve.evaluate(jnp.asarray(self.parameter), budget=budget).point
            )
            return (
                point,
                float(
                    np.max(
                        np.nextafter(
                            np.abs(self.point_enclosure(budget=budget) - point), np.inf
                        )
                    )
                ),
                True,
            )


class TripleSurfaceIntersectionRoot(StrictModule):
    """One regular joint source root of three surfaces, with six UV parameters."""

    __strict_contract__ = True

    first: intersection_curve_owner.SurfaceRegion
    second: intersection_curve_owner.SurfaceRegion
    third: intersection_curve_owner.SurfaceRegion
    parameter_lower: HostFloat64[Literal[6]]
    parameter_upper: HostFloat64[Literal[6]]
    parameters: HostFloat64[Literal[6]]
    preconditioner: HostFloat64[Literal[6], Literal[6]]
    contraction: float = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        first: intersection_curve_owner.SurfaceRegion,
        second: intersection_curve_owner.SurfaceRegion,
        third: intersection_curve_owner.SurfaceRegion,
        /,
        *,
        parameter_lower: ConvertibleToArray,
        parameter_upper: ConvertibleToArray,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                first,
                second,
                third,
                parameter_lower,
                parameter_upper,
            ),
            budget=budget,
        ) as budget:
            lower = parse(
                np.asarray(parameter_lower, dtype=np.float64),
                HostFloat64[Literal[6]],
                "parameter_lower",
            )
            upper = parse(
                np.asarray(parameter_upper, dtype=np.float64),
                HostFloat64[Literal[6]],
                "parameter_upper",
            )
            if (
                not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower >= upper)
            ):
                raise ValueError(
                    "TripleSurfaceIntersectionRoot.parameter_box must be finite and nonempty with six parameters; got invalid bounds"
                )
            evaluators = []
            for index, region in enumerate((first, second, third)):
                lo, hi = (
                    lower[2 * index : 2 * index + 2],
                    upper[2 * index : 2 * index + 2],
                )
                pieces = [
                    piece
                    for piece in intersection_curve_owner.surface_pieces(
                        region, budget=budget
                    )
                    if np.all(lo >= piece.lower) and np.all(hi <= piece.upper)
                ]
                if len(pieces) != 1:
                    raise ValueError(
                        "TripleSurfaceIntersectionRoot.parameter_box must lie in one span per surface; got multiple spans"
                    )
                evaluators.append(pieces[0].evaluator)
            prepared = prepare(
                TripleSurfaceSystem(*evaluators), 6, 1, preparations=preparations
            )
            inclusion = krawczyk(
                prepared,
                lower[None],
                upper[None],
                budget=budget,
                preparations=preparations,
            )
            if not bool(inclusion.certified[0]):
                raise ValueError(
                    "TripleSurfaceIntersectionRoot.parameter_box must certify a regular unique root; got unresolved inclusion"
                )
            roots, _, _, _ = prepared.newton(
                jnp.asarray((0.5 * (lower + upper))[None]),
                VectorLocalRootPlan(
                    6,
                    maximum_steps=40,
                    tolerance=1e-13,
                    plan_id="triple-surface-junction",
                ),
            )
            self.first, self.second, self.third = first, second, third
            self.parameter_lower, self.parameter_upper = lower, upper
            self.parameters = np.clip(np.asarray(roots)[0], lower, upper)
            self.preconditioner = inclusion.preconditioner[0]
            self.contraction = float(inclusion.contraction[0])
            self.root_id = canonical_fingerprint(
                {
                    "kind": "triple-surface-intersection-root",
                    "sources": array_tree_fingerprint((first, second, third)),
                    "box": array_tree_fingerprint((lower, upper)),
                }
            )

    def parameter_enclosure(
        self,
        /,
        *,
        maximum_steps: int = 16,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.parameter_enclosure(
                    maximum_steps=maximum_steps, preparations=preparations, budget=budget
                )
        evaluators = []
        for index, region in enumerate((self.first, self.second, self.third)):
            lower = self.parameter_lower[2 * index : 2 * index + 2]
            upper = self.parameter_upper[2 * index : 2 * index + 2]
            evaluators.append(
                next(
                    piece.evaluator
                    for piece in intersection_curve_owner.surface_pieces(
                        region, budget=budget
                    )
                    if np.all(lower >= piece.lower) and np.all(upper <= piece.upper)
                )
            )
        prepared = prepare(
            TripleSurfaceSystem(*evaluators), 6, 1, preparations=preparations
        )
        lower, upper = _contract(
            prepared,
            self.parameter_lower[None],
            self.parameter_upper[None],
            maximum_steps=maximum_steps,
            budget=budget,
            preparations=preparations,
        )
        return np.stack((lower[0], upper[0]))

    def point_enclosure(
        self,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.point_enclosure(preparations=preparations, budget=budget)
        boxes = []
        for index, region in enumerate((self.first, self.second, self.third)):
            boxes.append(
                np.asarray(
                    region.patch.bounding_box(
                        np.stack(
                            (
                                self.parameter_lower[2 * index : 2 * index + 2],
                                self.parameter_upper[2 * index : 2 * index + 2],
                            )
                        ),
                        preparations=preparations,
                        budget=budget,
                    )
                )
            )
        boxes_ = np.asarray(boxes)
        lower, upper = np.max(boxes_[:, 0], axis=0), np.min(boxes_[:, 1], axis=0)
        if np.any(lower > upper):
            raise ValueError(
                "TripleSurfaceIntersectionRoot.point_enclosure must intersect all source images; got disjoint enclosures"
            )
        return np.stack((lower, upper))

    def evaluate(self) -> tuple[np.ndarray, float, bool]:
        with coordinate_source_request((self,)) as budget:
            point = np.asarray(
                self.first.patch.evaluate(jnp.asarray(self.parameters[:2]))
            )
            return (
                point,
                float(
                    np.max(
                        np.nextafter(
                            np.abs(self.point_enclosure(budget=budget) - point), np.inf
                        )
                    )
                ),
                True,
            )

    def certifies_root(
        self,
        root: TrimIntersectionRoot
        | CurveSurfaceIntersectionRoot
        | TripleSurfaceIntersectionRoot,
        /,
        *,
        source_edge_lifts: Sequence[_SourceCurveSurfaceLift] = (),
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        """Prove equation identity and full source-parameter image inclusion.

        Source edges use construction identities or supplied whole-edge lifts;
        point-box overlap and numerical gap alone do not establish identity.
        """
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    root,
                )
            ) as budget:
                return self.certifies_root(
                    root,
                    source_edge_lifts=source_edge_lifts,
                    preparations=preparations,
                    budget=budget,
                )
        box = joint_root_parameter_image(
            root,
            (self.first, self.second, self.third),
            source_edge_lifts=source_edge_lifts,
            joint_box=np.stack((self.parameter_lower, self.parameter_upper)),
            preparations=preparations,
            budget=budget,
        )
        return box is not None and bool(
            np.all(box[0] >= self.parameter_lower)
            and np.all(box[1] <= self.parameter_upper)
        )


def _surface_definition_key(region: intersection_curve_owner.SurfaceRegion, /) -> str:
    return canonical_fingerprint(intersection_curve_owner.encode_geometry(region.patch))


@checked
def same_geometry(
    first: intersection_curve_owner.GeometryDefinition,
    second: intersection_curve_owner.GeometryDefinition,
    /,
) -> bool:
    return canonical_fingerprint(
        intersection_curve_owner.encode_geometry(first)
    ) == canonical_fingerprint(intersection_curve_owner.encode_geometry(second))


def _analytic_isoline_curve(curve: SurfaceIsoparametricCurve, /) -> AbstractCurve | None:
    """Specialize an analytic isoline while retaining its parameter atom."""

    patch = curve.surface
    if isinstance(patch, CylinderPatch):
        if curve.fixed_axis == 1:
            values = tuple(
                F(float(a)) + F(curve.fixed_value) * F(float(b))
                for a, b in zip(
                    np.asarray(patch.origin), np.asarray(patch.axis), strict=True
                )
            )
            if all(F(float(value)) == value for value in values):
                return CircleCurve(
                    np.asarray(tuple(float(value) for value in values)),
                    patch.first_axis,
                    patch.second_axis,
                    patch.radius,
                )
        elif curve.fixed_value == 0.0:
            values = tuple(
                F(float(a)) + F(float(patch.radius)) * F(float(b))
                for a, b in zip(
                    np.asarray(patch.origin), np.asarray(patch.first_axis), strict=True
                )
            )
            if all(F(float(value)) == value for value in values):
                return LineCurve(
                    np.asarray(tuple(float(value) for value in values)), patch.axis
                )
    if isinstance(patch, SpherePatch) and curve.fixed_value == 0.0:
        return CircleCurve(
            patch.center,
            patch.first_axis,
            patch.axis if curve.fixed_axis == 0 else patch.second_axis,
            patch.radius,
        )
    if (
        isinstance(patch, ExtrusionSurface)
        and curve.fixed_axis == 1
        and curve.fixed_value == 0.0
    ):
        return patch.curve
    if (
        isinstance(patch, RevolutionSurface)
        and curve.fixed_axis == 0
        and curve.fixed_value == 0.0
    ):
        return patch.curve
    return None


def _common_source_pose(
    curve: AbstractCurve,
    region: intersection_curve_owner.SurfaceRegion,
    /,
) -> tuple[AbstractCurve, intersection_curve_owner.SurfaceRegion] | None:
    """Source definitions of a placed edge and placed support under one pose.

    ``R c(t) + t0 == R s(p(t)) + t0`` iff ``c(t) == s(p(t))`` for the proved
    identical invertible authored pose, so lifts are decided on the sources.
    """
    patch = region.patch
    if (
        isinstance(curve, placed_owner.PlacedCurve)
        and isinstance(curve.definition, AbstractCurve)
        and isinstance(patch, placed_owner.PlacedSurface)
        and placed_owner.same_source_pose(curve, patch)
    ):
        return curve.definition, intersection_curve_owner.SurfaceRegion(
            patch.definition, region.parameter_box
        )
    return None


type _AffineCoefficients = tuple[tuple[Fraction, ...], tuple[Fraction, ...]]


def _isoline_source_lift(
    curve: SurfaceIsoparametricCurve,
    pcurve: AbstractCurve,
    region: intersection_curve_owner.SurfaceRegion,
    coefficients: _AffineCoefficients | None,
    first: float,
    last: float,
    /,
) -> bool:
    surface = region.patch
    if same_geometry(curve.surface, surface) and same_geometry(curve.p_curve(), pcurve):
        return True
    # Offset normals agree across a swept profile closure only at a certified
    # tangent-continuous join, not merely at matching profile positions.
    base = surface.base if isinstance(surface, OffsetSurface) else surface
    expected = intersection_curve_owner.affine_curve_coefficients(
        curve.p_curve(), np.zeros(2, dtype=np.float64)
    )
    fixed, free = curve.fixed_axis, 1 - curve.fixed_axis
    if (
        same_geometry(curve.surface, surface)
        and isinstance(base, (RevolutionSurface, ExtrusionSurface))
        and isinstance(base.curve, BSplineCurve)
        and coefficients is not None
        and expected is not None
        and coefficients[1] == expected[1]
        and coefficients[0][free] == expected[0][free]
    ):
        lower, upper = base.curve.parameter_domain
        if {coefficients[0][fixed], expected[0][fixed]} == {
            Fraction(lower),
            Fraction(upper),
        } and closure_tangent_continuous(
            base, fixed, lower, upper, tangent=isinstance(surface, OffsetSurface)
        ):
            return True
    analytic = _analytic_isoline_curve(curve)
    return analytic is not None and exact_source_lift(
        analytic, pcurve, region, first, last
    )


def _cylinder_affine_lift(
    curve: AbstractCurve, surface: CylinderPatch, coefficients: _AffineCoefficients, /
) -> bool:
    constant, derivative = coefficients
    if constant[0] != 0 or derivative[0] != 0:
        return False
    actual = intersection_curve_owner.affine_curve_coefficients(
        curve, np.zeros(3, dtype=np.float64)
    )
    if actual is None:
        return False
    radius = Fraction(float(surface.radius))
    origin = tuple(
        Fraction(float(a))
        + radius * Fraction(float(b))
        + constant[1] * Fraction(float(c))
        for a, b, c in zip(
            np.asarray(surface.origin),
            np.asarray(surface.first_axis),
            np.asarray(surface.axis),
            strict=True,
        )
    )
    tangent = tuple(
        derivative[1] * value for value in rational_vector(np.asarray(surface.axis))
    )
    return actual == (origin, tangent)


def _sphere_circle_lift(
    curve: CircleCurve, surface: SpherePatch, coefficients: _AffineCoefficients, /
) -> bool:
    return (
        coefficients[0] == (Fraction(0), Fraction(0))
        and coefficients[1][0] == 0
        and abs(coefficients[1][1]) == 1
        and np.array_equal(np.asarray(curve.center), np.asarray(surface.center))
        and np.array_equal(np.asarray(curve.radius), np.asarray(surface.radius))
        and np.array_equal(np.asarray(curve.first_axis), np.asarray(surface.first_axis))
        and np.array_equal(
            np.asarray(curve.second_axis),
            int(coefficients[1][1]) * np.asarray(surface.axis),
        )
    )


def _analytic_source_lift(
    curve: AbstractCurve,
    surface: AbstractSurfacePatch,
    coefficients: _AffineCoefficients,
    /,
) -> bool:
    constant, derivative = coefficients
    match surface:
        case ExtrusionSurface():
            return (
                constant == (Fraction(0), Fraction(0))
                and derivative == (Fraction(1), Fraction(0))
                and same_geometry(curve, surface.curve)
            )
        case RevolutionSurface():
            return (
                constant == (Fraction(0), Fraction(0))
                and derivative == (Fraction(0), Fraction(1))
                and same_geometry(curve, surface.curve)
            )
        case CylinderPatch():
            return _cylinder_affine_lift(curve, surface, coefficients)
        case SpherePatch() if isinstance(curve, CircleCurve):
            return _sphere_circle_lift(curve, surface, coefficients)
        case _:
            return False


@checked
def exact_source_lift(
    curve: AbstractCurve,
    pcurve: intersection_curve_owner.CurveEvaluator,
    region: intersection_curve_owner.SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    """Prove a whole-interval carrier construction identity."""
    if not isinstance(pcurve, AbstractCurve):
        return False
    common = _common_source_pose(curve, region)
    if common is not None:
        return exact_source_lift(common[0], pcurve, common[1], first, last)
    curve.validate_query_range(first, last)
    pcurve.validate_query_range(first, last)
    source = intersection_curve_owner.pcurve_periodic_source(pcurve, region.patch)
    if source is None or not isinstance(source[0], AbstractCurve):
        return False
    pcurve = source[0]
    surface = region.patch
    coefficients = intersection_curve_owner.affine_curve_coefficients(
        pcurve, np.zeros(2, dtype=np.float64)
    )
    if isinstance(curve, SurfaceIsoparametricCurve):
        return _isoline_source_lift(curve, pcurve, region, coefficients, first, last)
    if coefficients is not None and _analytic_source_lift(curve, surface, coefficients):
        return True
    if source_correspondence_owner.circle_correspondence(curve, pcurve, surface) == 0.0:
        return True
    return (
        isinstance(surface, PlanePatch)
        and source_correspondence_owner.plane_correspondence(
            curve, pcurve, surface, first, last, np.zeros(3, dtype=np.float64)
        )
        == 0.0
    )


def _cylinder_line_pcurve(
    curve: LineCurve,
    region: intersection_curve_owner.SurfaceRegion,
    patch: CylinderPatch,
    /,
) -> AbstractCurve | None:
    axis = rational_vector(np.asarray(patch.axis))
    nonzero = next((index for index, value in enumerate(axis) if value), None)
    if nonzero is None:
        return None
    delta = tuple(
        F(float(a)) - F(float(b)) - F(float(patch.radius)) * F(float(c))
        for a, b, c in zip(
            np.asarray(curve.origin),
            np.asarray(patch.origin),
            np.asarray(patch.first_axis),
            strict=True,
        )
    )
    tangent = rational_vector(np.asarray(curve.direction))
    offset, scale = delta[nonzero] / axis[nonzero], tangent[nonzero] / axis[nonzero]
    if not scale or F(float(offset)) != offset or F(float(scale)) != scale:
        return None
    candidate = LineCurve(
        np.asarray((0.0, float(offset))), np.asarray((0.0, float(scale)))
    )
    return candidate if exact_source_lift(curve, candidate, region, 0.0, 1.0) else None


def _cylinder_circle_pcurve(
    curve: CircleCurve,
    region: intersection_curve_owner.SurfaceRegion,
    patch: CylinderPatch,
    /,
) -> AbstractCurve | None:
    delta = tuple(
        F(float(a)) - F(float(b))
        for a, b in zip(np.asarray(curve.center), np.asarray(patch.origin), strict=True)
    )
    direction = rational_vector(np.asarray(patch.axis))
    axis = next((i for i, value in enumerate(direction) if value), None)
    if axis is None:
        return None
    height = delta[axis] / direction[axis]
    if not all(a == height * b for a, b in zip(delta, direction, strict=True)):
        return None
    if F(float(height)) != height:
        return None
    candidate = LineCurve(np.asarray((0.0, float(height))), np.asarray((1.0, 0.0)))
    return candidate if exact_source_lift(curve, candidate, region, 0.0, 1.0) else None


def _plane_curve_coordinates(
    vector: np.ndarray,
    translated: bool,
    patch: PlanePatch,
    axes: tuple[RationalVector, RationalVector],
    gram: list[list[Fraction]],
    /,
) -> np.ndarray | None:
    values = tuple(
        F(float(value)) - (F(float(origin)) if translated else F(0))
        for value, origin in zip(vector, np.asarray(patch.origin), strict=True)
    )
    dots = [rational_dot(axis, values) for axis in axes]
    solution = solve_exact(gram, [[dots[0]], [dots[1]]])
    uv = (solution[0][0], solution[1][0])
    if any(F(float(value)) != value for value in uv):
        return None
    return np.asarray(tuple(float(value) for value in uv))


def _plane_native_pcurve(
    curve: LineCurve | CircleCurve, patch: PlanePatch, /
) -> AbstractCurve | None:
    axes = (
        rational_vector(np.asarray(patch.first_axis)),
        rational_vector(np.asarray(patch.second_axis)),
    )
    gram = [[rational_dot(x, y) for y in axes] for x in axes]
    if not gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]:
        return None
    origin = _plane_curve_coordinates(
        np.asarray(curve.origin if isinstance(curve, LineCurve) else curve.center),
        True,
        patch,
        axes,
        gram,
    )
    first = _plane_curve_coordinates(
        np.asarray(curve.direction if isinstance(curve, LineCurve) else curve.first_axis),
        False,
        patch,
        axes,
        gram,
    )
    if origin is None or first is None or not np.any(first):
        return None
    if isinstance(curve, LineCurve):
        return LineCurve(origin, first)
    second = _plane_curve_coordinates(
        np.asarray(curve.second_axis), False, patch, axes, gram
    )
    return (
        CircleCurve(origin, first, second, curve.radius)
        if second is not None and np.any(second)
        else None
    )


@checked
def native_edge_pcurve(
    curve: AbstractCurve, region: intersection_curve_owner.SurfaceRegion, /
) -> AbstractCurve | None:
    """Recover native support constructions while retaining the source parameter."""
    common = _common_source_pose(curve, region)
    if common is not None:
        return native_edge_pcurve(common[0], common[1])
    patch = region.patch
    match curve, patch:
        case SurfaceIsoparametricCurve(), _:
            if same_geometry(curve.surface, patch):
                return curve.p_curve()
            analytic = _analytic_isoline_curve(curve)
            return None if analytic is None else native_edge_pcurve(analytic, region)
        case _, ExtrusionSurface() if same_geometry(curve, patch.curve):
            return LineCurve(np.zeros(2, dtype=np.float64), np.asarray((1.0, 0.0)))
        case _, RevolutionSurface() if same_geometry(curve, patch.curve):
            return LineCurve(np.zeros(2, dtype=np.float64), np.asarray((0.0, 1.0)))
        case CircleCurve(), SpherePatch():
            for sign in (1.0, -1.0):
                candidate = LineCurve(
                    np.zeros(2, dtype=np.float64), np.asarray((0.0, sign))
                )
                if exact_source_lift(curve, candidate, region, 0.0, 1.0):
                    return candidate
            return None
        case LineCurve(), CylinderPatch():
            return _cylinder_line_pcurve(curve, region, patch)
        case CircleCurve(), CylinderPatch():
            return _cylinder_circle_pcurve(curve, region, patch)
        case ((LineCurve() | CircleCurve()), PlanePatch()):
            return _plane_native_pcurve(curve, patch)
        case _:
            return None


@checked
def source_pcurve_image(
    curve: intersection_curve_owner.CurveEvaluator,
    first: float,
    last: float,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Preserve constant chart walls in affine source-coordinate images."""

    if isinstance(curve, intersection_curve_owner.PeriodicPCurve):
        source = intersection_curve_owner.pcurve_periodic_source(curve, curve.patch)
        if source is None:
            raise ValueError(
                "source_pcurve_image.curve must have a periodic source factorization; got unsupported source"
            )
        base, shifts = source
        image = source_pcurve_image(
            base, first, last, preparations=preparations, budget=budget
        )
        if any(shifts):
            offset = intersection_curve_owner.PeriodicPCurve(
                base, curve.patch, shifts
            ).period_offset_bounds()
            image = np.stack(interval_add((image[0], image[1]), (offset[0], offset[1])))
        return image
    coefficients = (
        intersection_curve_owner.affine_curve_coefficients(
            curve, np.zeros(2, dtype=np.float64)
        )
        if isinstance(curve, AbstractCurve)
        else None
    )
    if coefficients is None:
        return np.asarray(
            curve.enclosure(first, last, preparations=preparations, budget=budget)
            if isinstance(curve, AbstractTrimCurve)
            else curve.bounding_box(first, last, preparations=preparations, budget=budget)
        )
    values = []
    for constant, derivative in zip(*coefficients, strict=True):
        endpoints = (constant + derivative * F(first), constant + derivative * F(last))
        lo, hi = min(endpoints), max(endpoints)
        lower, upper = float(lo), float(hi)
        if F(lower) > lo:
            lower = np.nextafter(lower, -np.inf)
        if F(upper) < hi:
            upper = np.nextafter(upper, np.inf)
        values.append((lower, upper))
    return np.asarray(values).T


def _joint_coordinate_image(
    region: intersection_curve_owner.SurfaceRegion,
    regions: tuple[intersection_curve_owner.SurfaceRegion, ...],
    box: np.ndarray,
    /,
) -> np.ndarray:
    """Intersect joint components with rational plane-support equations."""

    result = box.copy()
    patch = region.patch
    if not isinstance(patch, (CylinderPatch, PlanePatch)):
        return result

    for supporting in regions:
        plane = supporting.patch
        if not isinstance(plane, PlanePatch):
            continue
        a = rational_vector(np.asarray(plane.first_axis))
        b = rational_vector(np.asarray(plane.second_axis))
        normal = rational_cross(a, b)
        delta = tuple(
            a - b
            for a, b in zip(
                rational_vector(np.asarray(patch.origin)),
                rational_vector(np.asarray(plane.origin)),
                strict=True,
            )
        )
        constant = rational_dot(normal, delta)
        if isinstance(patch, CylinderPatch):
            if rational_dot(
                normal, rational_vector(np.asarray(patch.first_axis))
            ) or rational_dot(normal, rational_vector(np.asarray(patch.second_axis))):
                continue
            axis, coefficient = (
                1,
                rational_dot(normal, rational_vector(np.asarray(patch.axis))),
            )
        else:
            coefficients = (
                rational_dot(normal, rational_vector(np.asarray(patch.first_axis))),
                rational_dot(normal, rational_vector(np.asarray(patch.second_axis))),
            )
            active = [index for index, value in enumerate(coefficients) if value]
            if len(active) != 1:
                continue
            axis = active[0]
            coefficient = coefficients[axis]
        if not coefficient:
            continue
        exact = -constant / coefficient
        lower, upper = fraction_down(exact), fraction_up(exact)
        result[0, axis] = max(result[0, axis], lower)
        result[1, axis] = min(result[1, axis], upper)
        if result[0, axis] > result[1, axis]:
            raise ValueError(
                "joint_root.parameters must intersect the support equation; got disjoint enclosures"
            )
    return result


def _curve_plane_coordinate_image(
    curve: AbstractCurve,
    region: intersection_curve_owner.SurfaceRegion,
    box: np.ndarray,
    /,
) -> np.ndarray:
    """Return constant planar UV components implied by the spatial source edge."""

    if isinstance(curve, SurfaceIsoparametricCurve):
        analytic = _analytic_isoline_curve(curve)
        return (
            box
            if analytic is None
            else _curve_plane_coordinate_image(analytic, region, box)
        )
    plane = region.patch
    if not isinstance(plane, PlanePatch) or not isinstance(
        curve, (LineCurve, CircleCurve)
    ):
        return box
    axes = (
        rational_vector(np.asarray(plane.first_axis)),
        rational_vector(np.asarray(plane.second_axis)),
    )
    gram = [[rational_dot(x, y) for y in axes] for x in axes]
    if not gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]:
        return box
    dual = tuple(tuple(row) for row in solve_exact(gram, [list(axis) for axis in axes]))
    origin = curve.origin if isinstance(curve, LineCurve) else curve.center
    relative = tuple(
        F(float(a)) - F(float(b))
        for a, b in zip(np.asarray(origin), np.asarray(plane.origin), strict=True)
    )
    varying = (
        (curve.direction,)
        if isinstance(curve, LineCurve)
        else (curve.first_axis, curve.second_axis)
    )
    result = box.copy()
    for coordinate in range(2):
        if any(
            rational_dot(dual[coordinate], rational_vector(np.asarray(vector)))
            for vector in varying
        ):
            continue
        exact = rational_dot(dual[coordinate], relative)
        lower, upper = fraction_down(exact), fraction_up(exact)
        result[0, coordinate] = max(result[0, coordinate], lower)
        result[1, coordinate] = min(result[1, coordinate], upper)
        if result[0, coordinate] > result[1, coordinate]:
            raise ValueError(
                "spatial_root.parameters must intersect planar source coordinates; got disjoint enclosures"
            )
    return result


def _pcurve_source_chart(
    curve: intersection_curve_owner.CurveEvaluator,
    /,
) -> tuple[
    intersection_curve_owner.CurveEvaluator,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    """Return the source chart, rational affine map and symbolic periods."""

    matrix = np.asarray(
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))), dtype=object
    )
    offset = np.asarray((Fraction(0), Fraction(0)), dtype=object)
    periods = np.asarray((Fraction(0), Fraction(0)), dtype=object)
    patches = []
    while isinstance(
        curve,
        (
            intersection_curve_owner.CurveTrimSegment,
            intersection_curve_owner.AffinePCurve,
            intersection_curve_owner.PeriodicPCurve,
        ),
    ):
        match curve:
            case intersection_curve_owner.AffinePCurve():
                offset = offset + matrix @ np.asarray(curve.offset, dtype=object)
                matrix = matrix @ np.asarray(curve.matrix, dtype=object)
                curve = curve.curve
            case intersection_curve_owner.PeriodicPCurve():
                periods = periods + matrix @ np.asarray(curve.period_shifts, dtype=object)
                patches.append(
                    canonical_fingerprint(
                        intersection_curve_owner.encode_geometry(curve.patch)
                    )
                )
                curve = curve.source_curve
            case intersection_curve_owner.CurveTrimSegment():
                curve = curve.curve
    return curve, matrix, offset, periods, tuple(patches)


def _pcurve_parameter_image(
    curve: intersection_curve_owner.CurveEvaluator,
    lower: float,
    upper: float,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> (
    tuple[
        intersection_curve_owner.SurfaceRegion,
        intersection_curve_owner.SurfaceRegion,
        np.ndarray,
    ]
    | None
):
    """Return source coordinates behind a bounded p-curve root parameter."""

    source_first: float | None = None
    source_last: float | None = None
    while isinstance(
        curve,
        (
            intersection_curve_owner.CurveTrimSegment,
            intersection_curve_owner.AffinePCurve,
            intersection_curve_owner.PeriodicPCurve,
        ),
    ):
        if isinstance(curve, intersection_curve_owner.CurveTrimSegment):
            source_first, source_last = curve.first, curve.last
            interval = curve.carrier_parameter_enclosure(
                lower,
                upper,
                source_extension=True,
                preparations=preparations,
                budget=budget,
            )
            lower, upper = float(interval[0]), float(interval[1])
        curve = (
            curve.source_curve
            if isinstance(curve, intersection_curve_owner.PeriodicPCurve)
            else curve.curve
        )
    if not isinstance(curve, intersection_curve_owner.IntersectionPCurve):
        return None
    if curve.reversed:
        total = interval_add(
            (np.asarray(curve.first), np.asarray(curve.first)),
            (np.asarray(curve.last), np.asarray(curve.last)),
        )
        image = interval_subtract(total, (np.asarray(lower), np.asarray(upper)))
        lower, upper = float(image[0]), float(image[1])

    head, tail = intersection_curve_owner.pcurve_chart_support(
        curve,
        curve.first if source_first is None else source_first,
        curve.last if source_last is None else source_last,
    )
    boxes = curve.curve.parameter_enclosures(
        lower,
        upper,
        minimum_chart=head,
        maximum_chart=tail,
        source_extension=True,
        preparations=preparations,
        budget=budget,
    )
    box = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
    return curve.curve.first, curve.curve.second, box


def _image_in_joint_box(
    image: np.ndarray, index: int, joint_box: np.ndarray | None, /
) -> bool:
    return joint_box is None or bool(
        np.all(image[0] >= joint_box[0, 2 * index : 2 * index + 2])
        and np.all(image[1] <= joint_box[1, 2 * index : 2 * index + 2])
    )


class _RootContributions(StrictModule):
    images: dict[str, list[np.ndarray]]


def _native_curve_root_images(
    root: CurveSurfaceIntersectionRoot,
    curve: AbstractCurve,
    parameters: np.ndarray,
    regions: tuple[
        intersection_curve_owner.SurfaceRegion,
        intersection_curve_owner.SurfaceRegion,
        intersection_curve_owner.SurfaceRegion,
    ],
    lifts: Sequence[_SourceCurveSurfaceLift],
    joint_box: np.ndarray | None,
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> _RootContributions:
    contributions: dict[str, list[np.ndarray]] = {}
    first, last = float(parameters[0, 0]), float(parameters[1, 0])
    for index, region in enumerate(regions):
        key = _surface_definition_key(region)
        if key == _surface_definition_key(root.surface):
            continue
        images: list[np.ndarray] = []
        pcurve = native_edge_pcurve(curve, region)
        if pcurve is not None and exact_source_lift(curve, pcurve, region, first, last):
            images.append(
                source_pcurve_image(
                    pcurve, first, last, preparations=preparations, budget=budget
                )
            )
        for lift in lifts:
            if (
                _surface_definition_key(lift.surface) == key
                and same_geometry(lift.curve, curve)
                and lift.first <= first <= last <= lift.last
                and exact_source_lift(curve, lift.pcurve, region, lift.first, lift.last)
            ):
                images.append(
                    source_pcurve_image(
                        lift.pcurve, first, last, preparations=preparations, budget=budget
                    )
                )
        for image in images:
            # Declared UV sheets are alternatives, not intersectable copies.
            if _image_in_joint_box(image, index, joint_box):
                contributions.setdefault(key, []).append(image)
    return _RootContributions(images=contributions)


def _curve_surface_root_images(
    root: CurveSurfaceIntersectionRoot,
    parameters: np.ndarray,
    regions: tuple[
        intersection_curve_owner.SurfaceRegion,
        intersection_curve_owner.SurfaceRegion,
        intersection_curve_owner.SurfaceRegion,
    ],
    lifts: Sequence[_SourceCurveSurfaceLift],
    joint_box: np.ndarray | None,
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> _RootContributions:
    match root.curve:
        case intersection_curve_owner.IntersectionCurve() as curve:
            boxes = curve.parameter_enclosures(
                float(parameters[0, 0]),
                float(parameters[1, 0]),
                source_extension=True,
                preparations=preparations,
                budget=budget,
            )
            coupled = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
            contributions: dict[str, list[np.ndarray]] = {}
            for region, image in (
                (curve.first, coupled[:, :2]),
                (curve.second, coupled[:, 2:]),
            ):
                contributions.setdefault(_surface_definition_key(region), []).append(
                    image
                )
        case AbstractCurve() as curve:
            phase = _native_curve_root_images(
                root,
                curve,
                parameters,
                regions,
                lifts,
                joint_box,
                preparations=preparations,
                budget=budget,
            )
            contributions = phase.images
        case _:
            raise TypeError(
                f"root.curve must be a supported carrier; got {type(root.curve).__name__}"
            )
    contributions.setdefault(_surface_definition_key(root.surface), []).append(
        parameters[:, 1:]
    )
    return _RootContributions(images=contributions)


def _intersection_trim_root_images(
    root: TrimIntersectionRoot,
    parameters: np.ndarray,
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> _RootContributions | None:
    images: list[
        tuple[
            intersection_curve_owner.SurfaceRegion,
            intersection_curve_owner.SurfaceRegion,
            np.ndarray,
        ]
    ] = []
    anchors: list[str] = []
    maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index, source in enumerate((root.first, root.second)):
        image = _pcurve_parameter_image(
            source,
            float(parameters[0, index]),
            float(parameters[1, index]),
            preparations=preparations,
            budget=budget,
        )
        if image is None:
            return None
        first, second, box = image
        underlying, matrix, offset, periods, patches = _pcurve_source_chart(source)
        if not isinstance(underlying, intersection_curve_owner.IntersectionPCurve):
            return None
        maps.append((matrix, offset, periods))
        anchor = first if underlying.side == "first" else second
        if any(patch != _surface_definition_key(anchor) for patch in patches):
            return None
        anchors.append(_surface_definition_key(anchor))
        images.append((first, second, box))
    if anchors[0] != anchors[1] or not all(
        np.array_equal(first, second)
        for first, second in zip(maps[0], maps[1], strict=True)
    ):
        return None
    contributions: dict[str, list[np.ndarray]] = {}
    for first, second, box in images:
        for region, image in ((first, box[:, :2]), (second, box[:, 2:])):
            contributions.setdefault(_surface_definition_key(region), []).append(image)
    if (
        np.array_equal(maps[0][0], np.eye(2, dtype=np.int64))
        and not np.any(maps[0][1])
        and not np.any(maps[0][2])
    ):
        contributions[anchors[0]] = [
            root.point_enclosure(preparations=preparations, budget=budget)
        ]
    return _RootContributions(images=contributions)


@checked
def joint_root_parameter_image(
    root: TrimIntersectionRoot
    | CurveSurfaceIntersectionRoot
    | TripleSurfaceIntersectionRoot,
    regions: tuple[
        intersection_curve_owner.SurfaceRegion,
        intersection_curve_owner.SurfaceRegion,
        intersection_curve_owner.SurfaceRegion,
    ],
    /,
    *,
    source_edge_lifts: Sequence[_SourceCurveSurfaceLift] = (),
    joint_box: np.ndarray | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray | None:
    """Enclose joint parameters using carrier identities and parameter inclusion."""
    parameters = root.parameter_enclosure(preparations=preparations, budget=budget)
    match root:
        case TripleSurfaceIntersectionRoot():
            contributions: dict[str, list[np.ndarray]] = {}
            for index, region in enumerate((root.first, root.second, root.third)):
                contributions.setdefault(_surface_definition_key(region), []).append(
                    parameters[:, 2 * index : 2 * index + 2]
                )
        case CurveSurfaceIntersectionRoot():
            phase = _curve_surface_root_images(
                root,
                parameters,
                regions,
                source_edge_lifts,
                joint_box,
                preparations=preparations,
                budget=budget,
            )
            contributions = phase.images
        case TrimIntersectionRoot():
            native = _native_trim_root_contributions(
                root,
                regions,
                source_edge_lifts,
                joint_box,
                preparations=preparations,
                budget=budget,
            )
            if native is not None:
                contributions = native
            else:
                phase = _intersection_trim_root_images(
                    root, parameters, preparations=preparations, budget=budget
                )
                if phase is None:
                    return None
                contributions = phase.images
        case _:
            raise TypeError(
                f"joint_root.root must be a supported junction; got {type(root).__name__}"
            )
    return _intersect_contributions(
        contributions, [_surface_definition_key(region) for region in regions]
    )


def _intersect_contributions(
    contributions: dict[str, list[np.ndarray]], keys: list[str], /
) -> np.ndarray | None:
    if len(set(keys)) != 3 or set(keys) != set(contributions):
        return None
    result = []
    for key in keys:
        boxes = np.asarray(contributions[key])
        lower, upper = np.max(boxes[:, 0], axis=0), np.min(boxes[:, 1], axis=0)
        if np.any(lower > upper):
            return None
        result.append(np.stack((lower, upper)))
    return np.concatenate(result, axis=1)


class _NativeTrimOperands(StrictModule):
    carriers: tuple[tuple[AbstractCurve, float, float], ...]


def _native_trim_operands(
    root: TrimIntersectionRoot,
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> _NativeTrimOperands | None:
    operands: list[tuple[AbstractCurve, float, float]] = []
    for operand in ("first", "second"):
        endpoint = TrimRootEndpoint(root, operand)
        carrier = endpoint.carrier
        if not isinstance(carrier, AbstractCurve) or carrier.ambient_dimension != 2:
            return None
        if isinstance(
            _pcurve_source_chart(carrier)[0], intersection_curve_owner.IntersectionPCurve
        ):
            return None
        lower, upper = endpoint.parameter_enclosure(
            preparations=preparations, budget=budget
        )
        operands.append((carrier, lower, upper))
    return _NativeTrimOperands(carriers=tuple(operands))


def _trim_face_edges(
    phase: _NativeTrimOperands,
    region: intersection_curve_owner.SurfaceRegion,
    key: str,
    lifts: Sequence[_SourceCurveSurfaceLift],
    /,
) -> tuple[AbstractCurve, ...]:
    edges: list[AbstractCurve] = []
    for carrier, lower, upper in phase.carriers:
        edge = next(
            (
                lift.curve
                for lift in lifts
                if isinstance(lift.curve, AbstractCurve)
                and _surface_definition_key(lift.surface) == key
                and same_geometry(lift.pcurve, carrier)
                and exact_source_lift(lift.curve, carrier, region, lower, upper)
            ),
            None,
        )
        if edge is None:
            break
        edges.append(edge)
    return tuple(edges)


def _native_trim_face_images(
    face: int,
    edges: tuple[AbstractCurve, ...],
    operands: _NativeTrimOperands,
    uv: np.ndarray,
    regions: tuple[
        intersection_curve_owner.SurfaceRegion,
        intersection_curve_owner.SurfaceRegion,
        intersection_curve_owner.SurfaceRegion,
    ],
    keys: list[str],
    lifts: Sequence[_SourceCurveSurfaceLift],
    joint_box: np.ndarray | None,
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> _RootContributions:
    contributions: dict[str, list[np.ndarray]] = {keys[face]: [uv]}
    for edge, (_, lower, upper) in zip(edges, operands.carriers, strict=True):
        for index, other in enumerate(regions):
            if index == face:
                continue
            candidates = [
                lift.pcurve
                for lift in lifts
                if isinstance(lift.pcurve, AbstractCurve)
                and _surface_definition_key(lift.surface) == keys[index]
                and same_geometry(lift.curve, edge)
            ]
            native = native_edge_pcurve(edge, other)
            if native is not None:
                candidates.append(native)
            for pcurve in candidates:
                if not exact_source_lift(edge, pcurve, other, lower, upper):
                    continue
                image = source_pcurve_image(
                    pcurve, lower, upper, preparations=preparations, budget=budget
                )
                if _image_in_joint_box(image, index, joint_box):
                    contributions.setdefault(keys[index], []).append(image)
    return _RootContributions(images=contributions)


def _native_trim_root_contributions(
    root: TrimIntersectionRoot,
    regions: tuple[
        intersection_curve_owner.SurfaceRegion,
        intersection_curve_owner.SurfaceRegion,
        intersection_curve_owner.SurfaceRegion,
    ],
    lifts: Sequence[_SourceCurveSurfaceLift],
    joint_box: np.ndarray | None,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> dict[str, list[np.ndarray]] | None:
    """Enclose a native UV root by proving its source edges on each support."""
    operands = _native_trim_operands(root, preparations=preparations, budget=budget)
    if operands is None:
        return None
    keys = [_surface_definition_key(region) for region in regions]
    if len(set(keys)) != 3:
        return None
    uv = root.point_enclosure(preparations=preparations, budget=budget)
    for face, region in enumerate(regions):
        if not _image_in_joint_box(uv, face, joint_box):
            continue
        edges = _trim_face_edges(operands, region, keys[face], lifts)
        if len(edges) != 2:
            continue
        phase = _native_trim_face_images(
            face,
            edges,
            operands,
            uv,
            regions,
            keys,
            lifts,
            joint_box,
            preparations=preparations,
            budget=budget,
        )
        box = _intersect_contributions(phase.images, keys)
        if box is not None and (
            joint_box is None
            or bool(np.all(box[0] >= joint_box[0]) and np.all(box[1] <= joint_box[1]))
        ):
            return phase.images
    return None


class TrimRootEndpoint(StrictModule):
    """Define a source root's scalar atom, optionally lifted to its p-curve."""

    root: TrimIntersectionRoot | CurveSurfaceIntersectionRoot | IntersectionCurvePointRoot
    operand: intersection_curve_owner.IntersectionCurveSide = eqx.field(static=True)
    affine_parameter_offset: float = eqx.field(static=True)
    affine_parameter_scale: float = eqx.field(static=True)
    affine_transforms: tuple[tuple[float, float], ...] = eqx.field(static=True)
    source_pcurve: AbstractCurve | intersection_curve_owner.IntersectionPCurve | None
    source_surface: intersection_curve_owner.SurfaceRegion | None

    @checked
    def __init__(
        self,
        root: TrimIntersectionRoot
        | CurveSurfaceIntersectionRoot
        | IntersectionCurvePointRoot,
        operand: intersection_curve_owner.IntersectionCurveSide,
        /,
        *,
        affine_parameter_offset: float = 0.0,
        affine_parameter_scale: float = 1.0,
        affine_transforms: tuple[tuple[float, float], ...] = (),
        source_pcurve: AbstractCurve
        | intersection_curve_owner.IntersectionPCurve
        | None = None,
        source_surface: intersection_curve_owner.SurfaceRegion | None = None,
    ) -> None:
        operand = parse(
            operand,
            intersection_curve_owner.IntersectionCurveSide,
            "TrimRootEndpoint.operand",
        )
        if (
            isinstance(root, (CurveSurfaceIntersectionRoot, IntersectionCurvePointRoot))
            and operand != "first"
        ):
            raise ValueError(
                "TrimRootEndpoint.operand must select first for a spatial root; got another operand"
            )
        if (
            not math.isfinite(affine_parameter_offset)
            or not math.isfinite(affine_parameter_scale)
            or affine_parameter_scale == 0
        ):
            raise ValueError(
                "TrimRootEndpoint.affine_expression must be finite with nonzero scale; got invalid coefficients"
            )
        if (source_pcurve is None) != (source_surface is None):
            raise ValueError(
                "TrimRootEndpoint.source_lift must contain both a p-curve and a surface; got an incomplete lift"
            )
        if source_pcurve is not None and source_surface is not None:
            if not isinstance(root, CurveSurfaceIntersectionRoot):
                raise ValueError(
                    "TrimRootEndpoint.root must be a curve-surface root for a p-curve lift; got another root kind"
                )
            if source_pcurve.ambient_dimension != 2:
                raise ValueError(
                    "TrimRootEndpoint.source_pcurve must have dimension two; got another dimension"
                )
            if isinstance(root.curve, intersection_curve_owner.IntersectionCurve):
                generating = (
                    (
                        source_pcurve.curve.first
                        if source_pcurve.side == "first"
                        else source_pcurve.curve.second
                    )
                    if isinstance(
                        source_pcurve, intersection_curve_owner.IntersectionPCurve
                    )
                    else None
                )
                valid = (
                    isinstance(source_pcurve, intersection_curve_owner.IntersectionPCurve)
                    and source_pcurve.curve.branch_id == root.curve.branch_id
                    and generating is not None
                    and _surface_definition_key(generating)
                    == _surface_definition_key(source_surface)
                )
            else:
                valid = exact_source_lift(
                    root.curve,
                    source_pcurve,
                    source_surface,
                    float(root.parameter_lower[0]),
                    float(root.parameter_upper[0]),
                )
            if not valid:
                raise ValueError(
                    "TrimRootEndpoint.source_pcurve must represent the source carrier on its support; got unproved construction identity"
                )
        self.source_pcurve, self.source_surface = source_pcurve, source_surface
        self.root, self.operand = root, operand
        self.affine_parameter_offset = float(affine_parameter_offset)
        self.affine_parameter_scale = float(affine_parameter_scale)
        if any(
            not math.isfinite(scale) or not math.isfinite(offset) or scale == 0
            for scale, offset in affine_transforms
        ):
            raise ValueError(
                "TrimRootEndpoint.affine_transforms must be finite and nonsingular; got invalid coefficients"
            )
        self.affine_transforms = tuple(
            (float(scale), float(offset)) for scale, offset in affine_transforms
        )

    @property
    def root_id(self) -> str:
        return self.root.root_id

    def _expression(
        self,
        *,
        maximum_steps: int = 16,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve,
        tuple[np.ndarray, np.ndarray],
    ]:
        if maximum_steps < 0:
            raise ValueError(
                "root.maximum_steps must be nonnegative; got a negative count"
            )
        if isinstance(self.root, IntersectionCurvePointRoot):
            value = Fraction(self.root.parameter) * Fraction(
                self.affine_parameter_scale
            ) + Fraction(self.affine_parameter_offset)
            for scale, offset in self.affine_transforms:
                value = value * Fraction(scale) + Fraction(offset)
            represented = float(value)
            lower = (
                represented
                if Fraction(represented) <= value
                else np.nextafter(represented, -np.inf)
            )
            upper = (
                represented
                if Fraction(represented) >= value
                else np.nextafter(represented, np.inf)
            )
            return self.root.curve, (np.asarray(lower), np.asarray(upper))
        index = 0 if self.operand == "first" else 1
        curve = (
            self.root.curve
            if isinstance(
                self.root, (CurveSurfaceIntersectionRoot, IntersectionCurvePointRoot)
            )
            else (self.root.first if index == 0 else self.root.second)
        )
        if self.source_pcurve is not None:
            curve = self.source_pcurve
        parameters = self.root.parameter_enclosure(
            maximum_steps=maximum_steps, preparations=preparations, budget=budget
        )
        interval = (np.asarray(parameters[0, index]), np.asarray(parameters[1, index]))

        operations: list[
            intersection_curve_owner.AffinePCurve
            | intersection_curve_owner.PeriodicPCurve
        ] = []
        while isinstance(
            curve,
            (
                intersection_curve_owner.CurveTrimSegment,
                intersection_curve_owner.AffinePCurve,
                intersection_curve_owner.PeriodicPCurve,
            ),
        ):
            match curve:
                case intersection_curve_owner.CurveTrimSegment():
                    interval = curve.carrier_parameter_enclosure(
                        float(interval[0]),
                        float(interval[1]),
                        source_extension=True,
                        preparations=preparations,
                        budget=budget,
                    )
                    curve = curve.curve
                case intersection_curve_owner.PeriodicPCurve():
                    operations.append(curve)
                    curve = curve.source_curve
                case intersection_curve_owner.AffinePCurve():
                    operations.append(curve)
                    curve = curve.curve
        if (
            isinstance(curve, intersection_curve_owner.IntersectionPCurve)
            and curve.reversed
        ):
            constant = interval_add(
                (np.asarray(curve.first), np.asarray(curve.first)),
                (np.asarray(curve.last), np.asarray(curve.last)),
            )
            interval = interval_subtract(constant, interval)
            curve = intersection_curve_owner.IntersectionPCurve(
                curve.curve, curve.side, first=curve.first, last=curve.last
            )
        interval = interval_add(
            interval_multiply(
                interval,
                (
                    np.asarray(self.affine_parameter_scale),
                    np.asarray(self.affine_parameter_scale),
                ),
            ),
            (
                np.asarray(self.affine_parameter_offset),
                np.asarray(self.affine_parameter_offset),
            ),
        )
        for scale, offset in self.affine_transforms:
            interval = interval_add(
                interval_multiply(interval, (np.asarray(scale), np.asarray(scale))),
                (np.asarray(offset), np.asarray(offset)),
            )
        for operation in reversed(operations):
            if not isinstance(curve, (AbstractCurve, AbstractTrimCurve)):
                raise TypeError(
                    "TrimRootEndpoint.carrier must support the UV operation; got unsupported carrier"
                )
            if isinstance(operation, intersection_curve_owner.AffinePCurve):
                curve = intersection_curve_owner.AffinePCurve(
                    curve, operation.matrix, operation.offset
                )
            else:
                curve = intersection_curve_owner.PeriodicPCurve(
                    curve, operation.patch, operation.period_shifts
                )
        return curve, (
            np.asarray(interval[0], dtype=np.float64),
            np.asarray(interval[1], dtype=np.float64),
        )

    @property
    def carrier(
        self,
    ) -> (
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve
    ):
        preparations = TrimRequestPreparations()
        with coordinate_source_request((self,)) as budget:
            return self._expression(
                maximum_steps=0, preparations=preparations, budget=budget
            )[0]

    @property
    def parameter(self) -> float:
        preparations = TrimRequestPreparations()
        with coordinate_source_request((self,)) as budget:
            lower, upper = self.parameter_enclosure(
                preparations=preparations, budget=budget
            )
            return 0.5 * (lower + upper)

    def parameter_enclosure(
        self,
        /,
        *,
        maximum_steps: int = 16,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[float, float]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.parameter_enclosure(
                    maximum_steps=maximum_steps, preparations=preparations, budget=budget
                )
        _, bounds = self._expression(
            maximum_steps=maximum_steps, preparations=preparations, budget=budget
        )
        return float(bounds[0]), float(bounds[1])

    def affine(self, scale: float, offset: float, /) -> TrimRootEndpoint:
        return TrimRootEndpoint(
            self.root,
            self.operand,
            affine_parameter_offset=self.affine_parameter_offset,
            affine_parameter_scale=self.affine_parameter_scale,
            affine_transforms=(*self.affine_transforms, (float(scale), float(offset))),
            source_pcurve=self.source_pcurve,
            source_surface=self.source_surface,
        )


class BranchRootEndpoint(StrictModule):
    """Transport a spatial or joint source root into a chart scalar expression.

    The expression is ``chart + (q_axis - node_start)/(node_end-node_start)``.
    Both node coordinates remain source-root expressions, not floating chart
    representatives. Period shifts are explicit source gauge operations.
    """

    root: CurveSurfaceIntersectionRoot | TripleSurfaceIntersectionRoot
    curve: intersection_curve_owner.IntersectionCurve
    chart: int = eqx.field(static=True)
    source_pcurve: AbstractCurve | None
    source_side: intersection_curve_owner.IntersectionCurveSide | None = eqx.field(
        static=True
    )
    source_first: float | None = eqx.field(static=True)
    source_last: float | None = eqx.field(static=True)
    periodic_shifts: tuple[int, int, int, int] = eqx.field(static=True)
    affine_transforms: tuple[tuple[float, float], ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        root: CurveSurfaceIntersectionRoot | TripleSurfaceIntersectionRoot,
        curve: intersection_curve_owner.IntersectionCurve,
        chart: int,
        /,
        *,
        source_pcurve: AbstractCurve | None = None,
        source_side: intersection_curve_owner.IntersectionCurveSide | None = None,
        source_first: float | None = None,
        source_last: float | None = None,
        periodic_shifts: tuple[int, int, int, int] = (0, 0, 0, 0),
        affine_transforms: tuple[tuple[float, float], ...] = (),
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                root,
                curve,
                chart,
                source_pcurve,
                source_side,
                source_first,
                source_last,
                periodic_shifts,
                affine_transforms,
            ),
            budget=budget,
        ) as budget:
            if not curve.fully_certified:
                raise ValueError(
                    "BranchRootEndpoint.curve must have certified charts; got an uncertified atlas"
                )
            if type(chart) is not int or not 0 <= chart < curve.num_charts:
                raise ValueError(
                    "BranchRootEndpoint.chart must select a source chart; got an out-of-range index"
                )
            if len(periodic_shifts) != 4 or any(
                type(value) is not int for value in periodic_shifts
            ):
                raise TypeError(
                    "BranchRootEndpoint.periodic_shifts must contain four integers; got invalid coefficients"
                )
            if any(
                not math.isfinite(scale) or not math.isfinite(offset) or scale == 0
                for scale, offset in affine_transforms
            ):
                raise ValueError(
                    "BranchRootEndpoint.affine_transforms must be finite and nonsingular; got invalid coefficients"
                )
            self.root, self.curve, self.chart = root, curve, chart
            self.source_pcurve, self.source_side = source_pcurve, source_side
            self.source_first, self.source_last = source_first, source_last
            self.periodic_shifts = tuple(periodic_shifts)
            self.affine_transforms = tuple(
                (float(scale), float(offset)) for scale, offset in affine_transforms
            )
            match root:
                case TripleSurfaceIntersectionRoot():
                    if any(
                        value is not None
                        for value in (
                            source_pcurve,
                            source_side,
                            source_first,
                            source_last,
                        )
                    ):
                        raise ValueError(
                            "BranchRootEndpoint.source_lift must be absent for a joint root; got an additional lift"
                        )
                case CurveSurfaceIntersectionRoot(
                    curve=intersection_curve_owner.IntersectionCurve() as source_branch
                ):
                    if (
                        source_pcurve is not None
                        or source_side is not None
                        or source_first is not None
                        or source_last is not None
                    ):
                        raise ValueError(
                            "BranchRootEndpoint.source_pcurve must be absent for a coupled branch; got an additional UV image"
                        )
                    if {
                        _surface_definition_key(source_branch.first),
                        _surface_definition_key(source_branch.second),
                    } != {
                        _surface_definition_key(curve.first),
                        _surface_definition_key(curve.second),
                    }:
                        raise ValueError(
                            "BranchRootEndpoint.root.curve must generate the target surfaces; got different carrier definitions"
                        )
                case CurveSurfaceIntersectionRoot(curve=AbstractCurve() as source_curve):
                    if source_side is None or source_pcurve is None:
                        raise TypeError(
                            "BranchRootEndpoint.source_lift must describe the generating support; got an incomplete lift"
                        )
                    source_side = parse(
                        source_side,
                        intersection_curve_owner.IntersectionCurveSide,
                        "BranchRootEndpoint.source_side",
                    )
                    source = curve.first if source_side == "first" else curve.second
                    opposite = curve.second if source_side == "first" else curve.first
                    if _surface_definition_key(root.surface) != _surface_definition_key(
                        opposite
                    ):
                        raise ValueError(
                            "BranchRootEndpoint.root.surface must match the opposite generating surface; got different carrier definitions"
                        )
                    first = (
                        float(root.parameter_lower[0])
                        if source_first is None
                        else float(source_first)
                    )
                    last = (
                        float(root.parameter_upper[0])
                        if source_last is None
                        else float(source_last)
                    )
                    if (
                        not math.isfinite(first)
                        or not math.isfinite(last)
                        or not first < last
                        or not first
                        <= root.parameter_lower[0]
                        <= root.parameter_upper[0]
                        <= last
                    ):
                        raise ValueError(
                            "BranchRootEndpoint.source_range must cover the root parameter box; got uncovered parameters"
                        )
                    if not exact_source_lift(
                        source_curve, source_pcurve, source, first, last
                    ):
                        raise ValueError(
                            "BranchRootEndpoint.source_pcurve must represent the carrier on its support; got unproved construction identity"
                        )
                    self.source_first, self.source_last = first, last
                case _:
                    raise TypeError(
                        f"BranchRootEndpoint.root must have a supported carrier; got {type(root).__name__}"
                    )
            image = self.source_image(
                maximum_steps=16, budget=budget, preparations=preparations
            )
            if not np.all(image[0] >= curve.box_lower[chart]) or not np.all(
                image[1] <= curve.box_upper[chart]
            ):
                raise ValueError(
                    "BranchRootEndpoint.source_image must lie inside the target chart box; got uncovered parameters"
                )
            self.parameter_enclosure(
                maximum_steps=0, budget=budget, preparations=preparations
            )

    @property
    def root_id(self) -> str:
        return self.root.root_id

    @property
    def carrier(self) -> intersection_curve_owner.IntersectionCurve:
        return self.curve

    def native_source_lift(
        self,
    ) -> tuple[AbstractCurve, intersection_curve_owner.IntersectionCurveSide]:
        """Return the native edge lift after checking its required source metadata."""
        pcurve, side = self.source_pcurve, self.source_side
        if pcurve is None or side is None:
            raise ValueError(
                "BranchRootEndpoint.source_lift must describe the spatial edge; got an incomplete lift"
            )
        return pcurve, side

    def source_image(
        self,
        /,
        *,
        maximum_steps: int,
        include_periodic_shifts: bool = True,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> np.ndarray:
        parameters = self.root.parameter_enclosure(
            maximum_steps=maximum_steps, preparations=preparations, budget=budget
        )
        match self.root:
            case TripleSurfaceIntersectionRoot():
                regions = (self.root.first, self.root.second, self.root.third)
                columns = []
                for target in (self.curve.first, self.curve.second):
                    matches = [
                        index
                        for index, region in enumerate(regions)
                        if _surface_definition_key(region)
                        == _surface_definition_key(target)
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "BranchRootEndpoint.root must have a unique generating patch definition; got ambiguous support"
                        )
                    index = matches[0]
                    columns.append(
                        _joint_coordinate_image(
                            target, regions, parameters[:, 2 * index : 2 * index + 2]
                        )
                    )
                image = np.concatenate(columns, axis=1)
            case CurveSurfaceIntersectionRoot(
                curve=intersection_curve_owner.IntersectionCurve() as source_branch
            ):
                boxes = source_branch.parameter_enclosures(
                    float(parameters[0, 0]),
                    float(parameters[1, 0]),
                    preparations=preparations,
                    budget=budget,
                )
                coupled = np.stack(
                    (np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0))
                )
                columns = []
                for target in (self.curve.first, self.curve.second):
                    offset = (
                        0
                        if _surface_definition_key(target)
                        == _surface_definition_key(source_branch.first)
                        else 2
                    )
                    columns.append(coupled[:, offset : offset + 2])
                image = np.concatenate(columns, axis=1)
            case CurveSurfaceIntersectionRoot(curve=AbstractCurve() as source_curve):
                pcurve, side = self.native_source_lift()
                patch = (
                    self.curve.first.patch if side == "first" else self.curve.second.patch
                )
                factorization = intersection_curve_owner.pcurve_periodic_source(
                    pcurve, patch
                )
                if factorization is None:
                    raise ValueError(
                        "BranchRootEndpoint.source_pcurve must have an integer periodic gauge; got unsupported factorization"
                    )
                source = source_pcurve_image(
                    factorization[0],
                    float(parameters[0, 0]),
                    float(parameters[1, 0]),
                    preparations=preparations,
                    budget=budget,
                )
                opposite = _curve_plane_coordinate_image(
                    source_curve, self.root.surface, parameters[:, 1:]
                )
                image = np.concatenate(
                    (source, opposite) if side == "first" else (opposite, source),
                    axis=1,
                )
            case _:
                raise TypeError(
                    f"BranchRootEndpoint.root must have a supported carrier; got {type(self.root).__name__}"
                )
        gauges = self.source_gauges()
        lower, upper = self.curve.shift_parameter_bounds(gauges)
        if include_periodic_shifts and np.any(gauges):
            image = np.stack(interval_add((image[0], image[1]), (lower, upper)))
        return image

    def source_gauges(self) -> np.ndarray:
        """Compose explicit source-sheet and branch gauges before realization."""

        shifts = np.asarray(self.periodic_shifts, dtype=np.int64).copy()
        if isinstance(self.root, CurveSurfaceIntersectionRoot) and not isinstance(
            self.root.curve, intersection_curve_owner.IntersectionCurve
        ):
            pcurve, side = self.native_source_lift()
            patch = self.curve.first.patch if side == "first" else self.curve.second.patch
            source = intersection_curve_owner.pcurve_periodic_source(pcurve, patch)
            if source is None:
                raise ValueError(
                    "BranchRootEndpoint.source_pcurve must have an integer periodic gauge; got unsupported factorization"
                )
            offset = 0 if side == "first" else 2
            shifts[offset : offset + 2] += np.asarray(source[1], dtype=np.int64)
        return shifts

    def certifies_point_root(
        self,
        point_root: IntersectionCurvePointRoot,
        /,
        *,
        maximum_steps: int = 16,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        """Prove that a spatial source root defines this shared-node point atom.

        The source must satisfy the node's coordinate constraint and lie inside
        its unique source-system enclosure.
        """
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    point_root,
                )
            ) as budget:
                return self.certifies_point_root(
                    point_root,
                    maximum_steps=maximum_steps,
                    preparations=preparations,
                    budget=budget,
                )
        if (
            not isinstance(point_root, IntersectionCurvePointRoot)
            or point_root.curve.branch_id != self.curve.branch_id
        ):
            return False
        if not point_root.parameter.is_integer():
            return False
        target = int(point_root.parameter)
        reference = int(self.curve.node_references[target])
        image = self.source_image(
            maximum_steps=maximum_steps,
            include_periodic_shifts=False,
            preparations=preparations,
            budget=budget,
        )
        for node in (self.chart, self.chart + 1):
            if int(self.curve.node_references[node]) != reference:
                continue
            shifts = self.curve.node_period_shifts[node].astype(np.int64).copy()
            if node == self.chart and self.chart:
                for axis, period in enumerate(
                    (*self.curve.first.patch.periods, *self.curve.second.patch.periods)
                ):
                    if period is not None:
                        shifts[axis] += int(
                            round(
                                self.curve.transition_shifts[self.chart - 1, axis]
                                / period
                            )
                        )
            difference = self.source_gauges() - shifts
            transported = image
            if np.any(difference):
                lower, upper = self.curve.shift_parameter_bounds(difference)
                transported = np.stack(interval_add((image[0], image[1]), (lower, upper)))
            axis = int(self.curve.node_axes[reference])
            value = self.curve.node_values[reference]
            if transported[0, axis] != value or transported[1, axis] != value:
                continue
            witness = max(0, reference - 1)
            if np.all(transported[0] >= self.curve.box_lower[witness]) and np.all(
                transported[1] <= self.curve.box_upper[witness]
            ):
                return True
        return False

    def parameter_enclosure(
        self,
        /,
        *,
        maximum_steps: int = 16,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[float, float]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.parameter_enclosure(
                    maximum_steps=maximum_steps, preparations=preparations, budget=budget
                )
        maximum_steps = nonnegative_integer(maximum_steps, "maximum_steps")
        image = self.source_image(
            maximum_steps=maximum_steps, preparations=preparations, budget=budget
        )
        axis = int(self.curve.chart_axes[self.chart])
        start = (
            self.curve.chart_start_lower[self.chart, axis],
            self.curve.chart_start_upper[self.chart, axis],
        )
        end = (
            self.curve.chart_end_lower[self.chart, axis],
            self.curve.chart_end_upper[self.chart, axis],
        )
        advance_lower, advance_upper = interval_subtract(end, start)
        advance = (
            np.asarray(advance_lower, dtype=np.float64),
            np.asarray(advance_upper, dtype=np.float64),
        )
        if advance[0] <= 0 <= advance[1]:
            raise ValueError(
                "BranchRootEndpoint.chart must have separated endpoint coordinates; got an interval containing zero advance"
            )
        scalar = interval_add(
            interval_divide(
                interval_subtract((image[0, axis], image[1, axis]), start), advance
            ),
            (np.asarray(float(self.chart)), np.asarray(float(self.chart))),
        )
        for scale, offset in self.affine_transforms:
            scalar = interval_add(
                interval_multiply(scalar, (np.asarray(scale), np.asarray(scale))),
                (np.asarray(offset), np.asarray(offset)),
            )
        return float(scalar[0]), float(scalar[1])

    @property
    def parameter(self) -> float:
        preparations = TrimRequestPreparations()
        with coordinate_source_request((self,)) as budget:
            lower, upper = self.parameter_enclosure(
                preparations=preparations, budget=budget
            )
            return 0.5 * (lower + upper)

    def affine(self, scale: float, offset: float, /) -> BranchRootEndpoint:
        return BranchRootEndpoint(
            self.root,
            self.curve,
            self.chart,
            source_pcurve=self.source_pcurve,
            source_side=self.source_side,
            source_first=self.source_first,
            source_last=self.source_last,
            periodic_shifts=self.periodic_shifts,
            affine_transforms=(*self.affine_transforms, (float(scale), float(offset))),
        )


def _within_derived_native_period(
    curve: intersection_curve_owner.CurveEvaluator
    | intersection_curve_owner.IntersectionCurve,
    value: Fraction | PeriodOffset,
    /,
) -> bool | None:
    """Test inclusion in a carrier domain defined by one symbolic period.

    A derived isoline with no authored ``parameter_range`` along a proved
    native 2*pi surface axis has domain ``[0, 2*pi]``; its binary
    domain endpoint is merely that period's representative. ``None`` means the
    carrier has an authored domain, whose binary bounds are never read as tau.
    """

    source = curve.definition if isinstance(curve, placed_owner.PlacedCurve) else curve
    if (
        not isinstance(source, SurfaceIsoparametricCurve)
        or source.parameter_range is not None
        or source.native_period_symbol() != "two_pi"
    ):
        return None
    rational, turns = (
        (value, Fraction(0))
        if isinstance(value, Fraction)
        else (value.rational, value.turns)
    )

    def nonnegative(scalar_rational: Fraction, scalar_turns: Fraction) -> bool:
        if not scalar_turns:
            return scalar_rational >= 0
        return (
            float(period_scalar_bounds(period_offset(scalar_rational, scalar_turns))[0])
            >= 0.0
        )

    return nonnegative(rational, turns) and nonnegative(-rational, 1 - turns)


class NativePeriodEndpoint(StrictModule):
    """Authored source parameter in Q + Q*(2*pi), not a floating root.

    ``patch`` and ``axis`` prove a surface period when supplied; otherwise the
    source curve must have a proved mathematical period. Numerical
    representatives never change a separately authored binary64 endpoint.
    """

    curve: (
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve
    )
    patch: AbstractSurfacePatch | None
    axis: Literal[0, 1] | None = eqx.field(static=True)
    rational: Fraction = eqx.field(static=True)
    turns: Fraction = eqx.field(static=True)

    @checked
    def __init__(
        self,
        curve: intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve,
        patch: AbstractSurfacePatch | None = None,
        axis: Literal[0, 1] | None = None,
        /,
        *,
        rational: Fraction | int = 0,
        turns: Fraction | int = 1,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                curve,
                patch,
                axis,
                rational,
                turns,
            ),
            budget=budget,
        ) as budget:
            if patch is None:
                if axis is not None:
                    raise ValueError(
                        "NativePeriodEndpoint.axis must be absent for a curve-owned period; got a surface axis"
                    )
                if (
                    not isinstance(
                        curve, (AbstractCurve, intersection_curve_owner.IntersectionCurve)
                    )
                    or curve.native_period_symbol() != "two_pi"
                ):
                    raise ValueError(
                        "NativePeriodEndpoint.curve must have a proved mathematical period; got unsupported carrier"
                    )
            else:
                if type(axis) is not int or axis not in (0, 1):
                    raise TypeError(
                        "NativePeriodEndpoint.surface_coordinate must select a patch and axis; got invalid selection"
                    )
                if patch.native_period_symbols()[axis] != "two_pi":
                    raise ValueError(
                        "NativePeriodEndpoint.axis must have a proved mathematical period; got unsupported coordinate"
                    )
            if any(
                not isinstance(value, (Fraction, int)) or isinstance(value, bool)
                for value in (rational, turns)
            ):
                raise TypeError(
                    "NativePeriodEndpoint.coefficients must be rationals or integers; got invalid coefficients"
                )
            self.curve, self.patch, self.axis = curve, patch, axis
            self.rational, self.turns = Fraction(rational), Fraction(turns)
            lower, upper = self.parameter_enclosure(
                budget=budget, preparations=preparations
            )
            if not (math.isfinite(lower) and math.isfinite(upper)):
                raise ValueError(
                    "NativePeriodEndpoint.parameter_enclosure must be finite; got unbounded values"
                )
            domain = (
                (0.0, float(curve.num_charts))
                if isinstance(curve, intersection_curve_owner.IntersectionCurve)
                else curve.parameter_domain
            )
            derived = _within_derived_native_period(curve, self.exact_parameter)
            if derived is False or (
                derived is None
                and domain is not None
                and not domain[0] <= lower <= upper <= domain[1]
            ):
                raise ValueError(
                    "NativePeriodEndpoint.parameter must lie within the carrier domain; got an out-of-domain enclosure"
                )

    @property
    def carrier(
        self,
    ) -> (
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve
    ):
        return self.curve

    @property
    def exact_parameter(self) -> Fraction | PeriodOffset:

        return period_offset(self.rational, self.turns)

    @property
    def parameter(self) -> float:
        return float(self.rational + self.turns * Fraction(math.tau))

    @property
    def root_id(self) -> str:
        return canonical_fingerprint(intersection_curve_owner.encode_geometry(self))

    @property
    def affine_transforms(self) -> tuple[tuple[float, float], ...]:
        return ()

    def parameter_enclosure(
        self,
        /,
        *,
        maximum_steps: int = 16,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[float, float]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.parameter_enclosure(
                    maximum_steps=maximum_steps, preparations=preparations, budget=budget
                )

        maximum_steps = nonnegative_integer(maximum_steps, "maximum_steps")
        lower, upper = period_scalar_bounds(self.exact_parameter)
        return float(lower), float(upper)

    def affine(self, scale: float, offset: float, /) -> NativePeriodEndpoint:
        if not math.isfinite(scale) or not math.isfinite(offset) or scale == 0:
            raise ValueError(
                "NativePeriodEndpoint.affine_expression must be finite with nonzero scale; got invalid coefficients"
            )
        return NativePeriodEndpoint(
            self.curve,
            self.patch,
            self.axis,
            rational=self.rational * Fraction(scale) + Fraction(offset),
            turns=self.turns * Fraction(scale),
        )


RootEndpoint: TypeAlias = TrimRootEndpoint | BranchRootEndpoint | NativePeriodEndpoint


class ParametricIntersectionPoint(StrictModule):
    """One isolated intersection event with its parameter enclosure.

    ``parameters`` concatenates the operands' parameters (first operand first).
    ``crossing`` events are unique regular roots certified inside the parameter
    enclosure. ``tangent_point`` denotes isolated contact, ``tangent_curve`` a
    certified contact continuum, and ``singular_point`` a branch junction.
    Degenerate or uncertified contact neighborhoods are reported as unresolved.
    """

    __strict_contract__ = True

    kind: ParametricIntersectionKind = eqx.field(static=True)
    certificate: ParametricIntersectionCertificate = eqx.field(static=True)
    parameters: HostFloat64[ParameterDim]
    parameter_lower: HostFloat64[ParameterDim]
    parameter_upper: HostFloat64[ParameterDim]
    point: HostFloat64[AmbientDim]
    gap_bound: float = eqx.field(static=True)
    condition_estimate: float = eqx.field(static=True)
    branch_count: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        kind: ParametricIntersectionKind,
        certificate: ParametricIntersectionCertificate,
        parameters: np.ndarray,
        parameter_lower: np.ndarray,
        parameter_upper: np.ndarray,
        point: np.ndarray,
        gap_bound: float,
        condition_estimate: float,
        branch_count: int = -1,
    ) -> None:
        scope = Scope()
        if branch_count < -1:
            raise ValueError(
                f"ParametricIntersectionPoint.branch_count must be nonnegative or -1 (unresolved); got {branch_count}"
            )
        self.kind = parse(kind, ParametricIntersectionKind, "kind")
        self.certificate = parse(
            certificate, ParametricIntersectionCertificate, "certificate"
        )
        self.parameters = parse(
            np.asarray(parameters, dtype=np.float64),
            HostFloat64[ParameterDim],
            "parameters",
            scope=scope,
        )
        self.parameter_lower = parse(
            np.asarray(parameter_lower, dtype=np.float64),
            HostFloat64[ParameterDim],
            "parameter_lower",
            scope=scope,
        )
        self.parameter_upper = parse(
            np.asarray(parameter_upper, dtype=np.float64),
            HostFloat64[ParameterDim],
            "parameter_upper",
            scope=scope,
        )
        self.point = parse(
            np.asarray(point, dtype=np.float64), HostFloat64[AmbientDim], "point"
        )
        self.gap_bound = float(gap_bound)
        self.condition_estimate = float(condition_estimate)
        self.branch_count = branch_count


class CoincidentParameterRegion(StrictModule):
    """Connected cells where the operands agree within ``distance_bound``.

    Cells hold coupled parameter boxes. ``partial`` cells meet the boundary of the
    coincident set: only part of the first operand's cell is covered.
    """

    __strict_contract__ = True

    cell_lower: HostFloat64[CellDim, ParameterDim]
    cell_upper: HostFloat64[CellDim, ParameterDim]
    partial: HostBool[CellDim]
    distance_bound: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        cell_lower: np.ndarray,
        cell_upper: np.ndarray,
        partial: np.ndarray,
        distance_bound: float,
    ) -> None:
        scope = Scope()
        self.cell_lower = parse(
            np.asarray(cell_lower, dtype=np.float64),
            HostFloat64[CellDim, ParameterDim],
            "cell_lower",
            scope=scope,
        )
        self.cell_upper = parse(
            np.asarray(cell_upper, dtype=np.float64),
            HostFloat64[CellDim, ParameterDim],
            "cell_upper",
            scope=scope,
        )
        self.partial = parse(
            np.asarray(partial, dtype=np.bool_),
            HostBool[CellDim],
            "partial",
            scope=scope,
        )
        self.distance_bound = float(distance_bound)


class UnresolvedParameterRegion(StrictModule):
    """A parameter box whose intersection status was not decided."""

    __strict_contract__ = True

    lower: HostFloat64[ParameterDim]
    upper: HostFloat64[ParameterDim]
    reason: UnresolvedIntersectionReason = eqx.field(static=True)

    def __init__(
        self, lower: np.ndarray, upper: np.ndarray, reason: UnresolvedIntersectionReason
    ) -> None:
        scope = Scope()
        self.lower = parse(
            np.asarray(lower, dtype=np.float64),
            HostFloat64[ParameterDim],
            "lower",
            scope=scope,
        )
        self.upper = parse(
            np.asarray(upper, dtype=np.float64),
            HostFloat64[ParameterDim],
            "upper",
            scope=scope,
        )
        self.reason = parse(reason, UnresolvedIntersectionReason, "reason")


class CurveIntersectionResult(StrictModule):
    """Isolated events of a curve/curve or curve/surface intersection."""

    points: tuple[ParametricIntersectionPoint, ...]
    coincident: tuple[CoincidentParameterRegion, ...]
    unresolved: tuple[UnresolvedParameterRegion, ...]
    work: ParametricIntersectionWork

    def __init__(
        self,
        points: Sequence[ParametricIntersectionPoint],
        coincident: Sequence[CoincidentParameterRegion],
        unresolved: Sequence[UnresolvedParameterRegion],
        work: ParametricIntersectionWork,
    ) -> None:
        self.points = tuple(points)
        self.coincident = tuple(coincident)
        self.unresolved = tuple(unresolved)
        self.work = work

    @property
    def complete(self) -> bool:
        """Subdivision finished; partial coincidence cells remain incomplete."""
        return (
            not self.unresolved
            and not self.work.budget_exhausted
            and not any(np.any(region.partial) for region in self.coincident)
        )


class SurfaceIntersectionResult(StrictModule):
    """Branches, isolated contacts and coincident regions of two surfaces."""

    curves: tuple[intersection_curve_owner.IntersectionCurve, ...]
    points: tuple[ParametricIntersectionPoint, ...]
    coincident: tuple[CoincidentParameterRegion, ...]
    unresolved: tuple[UnresolvedParameterRegion, ...]
    work: ParametricIntersectionWork

    def __init__(
        self,
        curves: Sequence[intersection_curve_owner.IntersectionCurve],
        points: Sequence[ParametricIntersectionPoint],
        coincident: Sequence[CoincidentParameterRegion],
        unresolved: Sequence[UnresolvedParameterRegion],
        work: ParametricIntersectionWork,
    ) -> None:
        self.curves = tuple(curves)
        self.points = tuple(points)
        self.coincident = tuple(coincident)
        self.unresolved = tuple(unresolved)
        self.work = work

    @property
    def complete(self) -> bool:
        """Discovery finished and every branch is covered by certified charts."""
        return (
            not self.unresolved
            and not self.work.budget_exhausted
            and not any(np.any(region.partial) for region in self.coincident)
            and all(
                curve.fully_certified
                and curve.start_kind != "uncertified"
                and curve.end_kind != "uncertified"
                for curve in self.curves
            )
        )


# ------------------------------------------------------------ equation systems


def _curve_value(
    curve: intersection_curve_owner.CurveEvaluator
    | intersection_curve_owner.IntersectionCurve,
    parameter: Array,
    /,
) -> Array:
    if isinstance(curve, intersection_curve_owner.IntersectionCurve):
        return curve.evaluate(parameter).point
    return curve.evaluate(parameter)


def _curve_derivative(
    curve: intersection_curve_owner.CurveEvaluator
    | intersection_curve_owner.IntersectionCurve,
    parameter: Array,
    /,
) -> Array:
    return jax.jacfwd(lambda value: _curve_value(curve, value))(parameter)


def _surface_derivative(surface: AbstractSurfacePatch, parameters: Array, /) -> Array:
    return jax.jacfwd(surface.evaluate)(parameters)


class _CurvePairSystem(StrictModule):
    first: (
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve
    )
    second: (
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve
    )

    def residual(self, parameters: Array, /) -> Array:
        return _curve_value(self.first, parameters[0]) - _curve_value(
            self.second, parameters[1]
        )


class _CurveProximitySystem(StrictModule):
    """Stationarity of ``|c1(s) - c2(t)|^2`` for space curves."""

    first: (
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve
    )
    second: (
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve
    )

    def residual(self, parameters: Array, /) -> Array:
        gap = _curve_value(self.first, parameters[0]) - _curve_value(
            self.second, parameters[1]
        )
        return jnp.stack(
            (
                jnp.dot(gap, _curve_derivative(self.first, parameters[0])),
                jnp.dot(gap, _curve_derivative(self.second, parameters[1])),
            )
        )


class _PlanarCurveTangencySystem(StrictModule):
    """Foot point of ``c1(s)`` on ``c2`` with parallel tangents."""

    first: (
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve
    )
    second: (
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve
    )

    def residual(self, parameters: Array, /) -> Array:
        gap = _curve_value(self.first, parameters[0]) - _curve_value(
            self.second, parameters[1]
        )
        first_tangent = _curve_derivative(self.first, parameters[0])
        second_tangent = _curve_derivative(self.second, parameters[1])
        return jnp.stack(
            (
                jnp.dot(gap, second_tangent),
                first_tangent[0] * second_tangent[1]
                - first_tangent[1] * second_tangent[0],
            )
        )


class TripleSurfaceSystem(StrictModule):
    first: AbstractSurfacePatch
    second: AbstractSurfacePatch
    third: AbstractSurfacePatch

    def residual(self, parameters: Array, /) -> Array:
        point = self.first.evaluate(parameters[:2])
        return jnp.concatenate(
            (
                point - self.second.evaluate(parameters[2:4]),
                point - self.third.evaluate(parameters[4:]),
            )
        )


class CurveSurfaceSystem(StrictModule):
    curve: (
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve
    )
    surface: AbstractSurfacePatch

    def residual(self, parameters: Array, /) -> Array:
        if isinstance(self.curve, intersection_curve_owner.IntersectionCurve):
            point = self.curve.evaluate(parameters[0]).point
        else:
            point = self.curve.evaluate(parameters[0])
        return point - self.surface.evaluate(parameters[1:])


class _CurveSurfaceTangencySystem(StrictModule):
    """Foot point of ``c(t)`` on the surface with the curve tangent in its plane."""

    curve: (
        intersection_curve_owner.CurveEvaluator
        | intersection_curve_owner.IntersectionCurve
    )
    surface: AbstractSurfacePatch

    def residual(self, parameters: Array, /) -> Array:
        gap = _curve_value(self.curve, parameters[0]) - self.surface.evaluate(
            parameters[1:]
        )
        frame = _surface_derivative(self.surface, parameters[1:])
        normal = jnp.cross(frame[:, 0], frame[:, 1])
        tangent = _curve_derivative(self.curve, parameters[0])
        return jnp.stack(
            (
                jnp.dot(gap, frame[:, 0]),
                jnp.dot(gap, frame[:, 1]),
                jnp.dot(tangent, normal),
            )
        )


class _SurfaceTangencySystem(StrictModule):
    """Foot point of ``S1(p)`` on ``S2`` with parallel normals."""

    pair: intersection_curve_owner.SurfacePairSystem

    def residual(self, parameters: Array, /) -> Array:
        gap = self.pair.residual(parameters)
        first = _surface_derivative(self.pair.first, parameters[:2])
        second = _surface_derivative(self.pair.second, parameters[2:])
        normal = jnp.cross(first[:, 0], first[:, 1])
        return jnp.stack(
            (
                jnp.dot(gap, second[:, 0]),
                jnp.dot(gap, second[:, 1]),
                jnp.dot(normal, second[:, 0]),
                jnp.dot(normal, second[:, 1]),
            )
        )


# A fixed generic direction for turning points. Every closed branch has an
# extremum of this linear functional; a branch along which it is constant must
# be orthogonal to it, which is non-generic for these incommensurate weights.
_TURNING_DIRECTION = np.asarray(
    (1.0, math.sqrt(2.0) - 0.5, math.sqrt(3.0) - 1.0, math.pi - 2.7), dtype=np.float64
)


class _TurningSystem(StrictModule):
    """Branch points where the tangent is orthogonal to a scaled generic direction.

    With ``J = [S1_u, S1_v, -S2_u, -S2_v]`` the branch tangent is the cofactor
    vector of ``J`` and ``a . t = det([a; J])``, expanded into triple products of
    the surface frames and normals.
    """

    pair: intersection_curve_owner.SurfacePairSystem
    direction: Array

    def residual(self, parameters: Array, /) -> Array:
        first = _surface_derivative(self.pair.first, parameters[:2])
        second = _surface_derivative(self.pair.second, parameters[2:])
        first_normal = jnp.cross(first[:, 0], first[:, 1])
        second_normal = jnp.cross(second[:, 0], second[:, 1])
        weights = self.direction
        functional = (
            weights[0] * jnp.dot(second_normal, first[:, 1])
            - weights[1] * jnp.dot(second_normal, first[:, 0])
            - weights[2] * jnp.dot(first_normal, second[:, 1])
            + weights[3] * jnp.dot(first_normal, second[:, 0])
        )
        return jnp.concatenate((self.pair.residual(parameters), functional[None]))


class _FaceSystem(StrictModule):
    """A surface pair restricted to one face of the coupled parameter box."""

    pair: intersection_curve_owner.SurfacePairSystem
    axis: int = eqx.field(static=True)
    value: Array

    def residual(self, parameters: Array, /) -> Array:
        fixed = jnp.reshape(self.value, (1,)).astype(parameters.dtype)
        return self.pair.residual(
            jnp.concatenate((parameters[: self.axis], fixed, parameters[self.axis :]))
        )


class _CorrectorSystem(StrictModule):
    """Branch equations plus the hyperplane through a predictor."""

    pair: intersection_curve_owner.SurfacePairSystem
    normal: Array
    anchor: Array

    def residual(self, parameters: Array, /) -> Array:
        return jnp.concatenate(
            (
                self.pair.residual(parameters),
                jnp.dot(self.normal, parameters - self.anchor)[None],
            )
        )


type _System = (
    _CurvePairSystem
    | _CurveProximitySystem
    | _PlanarCurveTangencySystem
    | CurveSurfaceSystem
    | _CurveSurfaceTangencySystem
    | _SurfaceTangencySystem
    | _TurningSystem
    | _FaceSystem
    | _CorrectorSystem
    | intersection_curve_owner.SurfacePairSystem
    | TripleSurfaceSystem
    | _JunctionSystem
)


# ----------------------------------------------------- compiled point kernels


@eqx.filter_jit
def _point_values(system: _System, points: Array) -> Array:
    return jax.vmap(system.residual)(points)


@eqx.filter_jit
def _point_jacobians(system: _System, points: Array) -> Array:
    return jax.vmap(jax.jacfwd(system.residual))(points)


def _newton_values(
    system: _System, points: Array, plan: VectorLocalRootPlan
) -> tuple[Array, Array, Array, Array]:
    def one(point: Array) -> tuple[Array, Array, Array, Array]:
        root, diagnostics = plan.solve_with_diagnostics(system.residual, point)
        if point.shape[0] <= 4:
            jacobian = jax.jacfwd(system.residual)(root)
            inverse = inverse_small_linear(SmallLinearSolvePlan(point.shape[0]), jacobian)
            condition = jnp.max(inverse.condition_estimate)
        else:
            condition = jnp.max(diagnostics.condition_estimate)
        return (
            root,
            diagnostics.converged,
            diagnostics.residual_norm,
            condition,
        )

    return jax.vmap(one)(points)


_newton = eqx.filter_jit(_newton_values)


@eqx.filter_jit
def _branch_tangent(
    pair: intersection_curve_owner.SurfacePairSystem, point: Array, reference: Array
) -> Array:
    jacobian = jax.jacfwd(pair.residual)(point)
    matrix = jnp.concatenate((jacobian, reference[None]), axis=0)
    right = jnp.zeros((4,), dtype=point.dtype).at[3].set(1.0)
    tangent = solve_small_linear(SmallLinearSolvePlan(4), matrix, right).value
    return tangent / jnp.linalg.norm(tangent)


@eqx.filter_jit
def _march_step(
    pair: intersection_curve_owner.SurfacePairSystem,
    point: Array,
    tangent: Array,
    step: Array,
    plan: VectorLocalRootPlan,
) -> tuple[Array, Array, Array, Array]:
    frame = jax.jacfwd(pair.first.evaluate)(point[:2])
    speed = jnp.linalg.norm(frame @ tangent[:2]) + jnp.linalg.norm(
        jax.jacfwd(pair.second.evaluate)(point[2:]) @ tangent[2:]
    )
    predictor = point + (2.0 * step / speed) * tangent
    corrector = _CorrectorSystem(pair, tangent, predictor)
    root, diagnostics = plan.solve_with_diagnostics(corrector.residual, predictor)
    jacobian = jax.jacfwd(pair.residual)(root)
    matrix = jnp.concatenate((jacobian, tangent[None]), axis=0)
    right = jnp.zeros((4,), dtype=point.dtype).at[3].set(1.0)
    following = solve_small_linear(SmallLinearSolvePlan(4), matrix, right).value
    following = following / jnp.linalg.norm(following)
    return root, following, predictor, diagnostics.converged


def _padded_call(
    kernel: Callable[[np.ndarray], tuple[Array, ...] | Array],
    points: np.ndarray,
    batch: int,
    /,
) -> list[np.ndarray]:
    """Evaluate a compiled batch kernel at a fixed padded batch size."""
    count = points.shape[0]
    outputs: list[list[np.ndarray]] = []
    for start in range(0, count, batch):
        chunk = points[start : start + batch]
        padding = batch - chunk.shape[0]
        padded = np.concatenate((chunk, np.repeat(chunk[:1], padding, axis=0)))
        result = kernel(padded)
        values = result if isinstance(result, tuple) else (result,)
        outputs.append([np.asarray(value)[: chunk.shape[0]] for value in values])
    return [np.concatenate(parts) for parts in zip(*outputs, strict=True)]


# --------------------------------------------------------- interval algebra


_UNIT = np.finfo(np.float64).eps


def _contains_zero(lower: np.ndarray, upper: np.ndarray, /) -> np.ndarray:
    return np.all((lower <= 0.0) & (upper >= 0.0), axis=-1)


@dataclass
class _Budget:
    maximum: int
    processed: int = 0
    excluded: int = 0
    certified: int = 0
    coincident: int = 0
    unresolved: int = 0
    march_steps: int = 0
    charts_certified: int = 0
    charts_uncertified: int = 0
    exhausted: bool = False

    def work(self) -> ParametricIntersectionWork:
        return ParametricIntersectionWork(
            boxes_processed=self.processed,
            boxes_excluded=self.excluded,
            boxes_certified=self.certified,
            boxes_coincident=self.coincident,
            boxes_unresolved=self.unresolved,
            march_steps=self.march_steps,
            charts_certified=self.charts_certified,
            charts_uncertified=self.charts_uncertified,
            budget_exhausted=self.exhausted,
        )


@dataclass(frozen=True)
class _CurveCapabilityIntervals:
    """Certified trim values/jets, rather than a traced numerical chart solve."""

    system: _CurvePairSystem | _CurveProximitySystem | _PlanarCurveTangencySystem
    jacobian: bool

    @property
    def output_shape(self) -> tuple[int, ...]:
        if isinstance(self.system, _CurvePairSystem):
            dimension = self.system.first.ambient_dimension
            return (dimension, 2) if self.jacobian else (dimension,)
        return (2, 2) if self.jacobian else (2,)

    def evaluate_scope(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        def jet(
            curve: intersection_curve_owner.CurveEvaluator
            | intersection_curve_owner.IntersectionCurve,
            first: float,
            last: float,
            order: int,
            *,
            preparations: TrimRequestPreparations | None = preparations,
            budget: CoordinateEnclosureBudget = budget,
        ) -> tuple[np.ndarray, np.ndarray]:
            if preparations is None:
                preparations = TrimRequestPreparations()
            if isinstance(curve, intersection_curve_owner.IntersectionCurve):
                domain = curve.parameter_domain
                if first < domain[0] or last > domain[1]:
                    return np.full(3, -np.inf), np.full(3, np.inf)
                if order == 0:
                    box = curve.bounding_box(
                        first, last, preparations=preparations, budget=budget
                    )
                    return box[0], box[1]
                return curve.derivative_bounds(
                    first, last, order=order, preparations=preparations, budget=budget
                )
            if isinstance(curve, AbstractTrimCurve):
                return intersection_curve_owner.trim_source_jet(
                    curve, first, last, order, preparations=preparations, budget=budget
                )
            derivative = curve.evaluate
            for _ in range(order):
                derivative = jax.jacfwd(derivative)
            prepared = prepare_interval_function(
                lambda parameter: derivative(parameter[0]),
                1,
                batch_capacity=1,
                constant_bounds=coefficient_enclosures(curve),
            )
            lo, hi = evaluate_interval(
                prepared,
                np.asarray([[first]]),
                np.asarray([[last]]),
                budget=budget,
                preparations=preparations,
            )
            return lo[0], hi[0]

        def determinant(
            a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray]:
            bounds = interval_subtract(
                interval_multiply((a[0][0], a[1][0]), (b[0][1], b[1][1])),
                interval_multiply((a[0][1], a[1][1]), (b[0][0], b[1][0])),
            )
            return (
                np.asarray(bounds[0], dtype=np.float64),
                np.asarray(bounds[1], dtype=np.float64),
            )

        lower_outputs, upper_outputs = [], []
        for lo, hi in zip(lower, upper, strict=True):
            a = jet(
                self.system.first,
                float(lo[0]),
                float(hi[0]),
                0,
                budget=budget,
                preparations=preparations,
            )
            b = jet(
                self.system.second,
                float(lo[1]),
                float(hi[1]),
                0,
                budget=budget,
                preparations=preparations,
            )
            gap = interval_subtract(a, b)
            if isinstance(self.system, _CurvePairSystem) and not self.jacobian:
                result = gap
            else:
                da = jet(
                    self.system.first,
                    float(lo[0]),
                    float(hi[0]),
                    1,
                    budget=budget,
                    preparations=preparations,
                )
                db = jet(
                    self.system.second,
                    float(lo[1]),
                    float(hi[1]),
                    1,
                    budget=budget,
                    preparations=preparations,
                )
                match self.system:
                    case _CurvePairSystem():
                        result = (
                            np.stack((da[0], -db[1]), axis=1),
                            np.stack((da[1], -db[0]), axis=1),
                        )
                    case _CurveProximitySystem():
                        if not self.jacobian:
                            first_value, second_value = (
                                interval_dot(gap, da),
                                interval_dot(gap, db),
                            )
                            result = (
                                np.asarray((first_value[0], second_value[0])),
                                np.asarray((first_value[1], second_value[1])),
                            )
                        else:
                            dda = jet(
                                self.system.first,
                                float(lo[0]),
                                float(hi[0]),
                                2,
                                budget=budget,
                                preparations=preparations,
                            )
                            ddb = jet(
                                self.system.second,
                                float(lo[1]),
                                float(hi[1]),
                                2,
                                budget=budget,
                                preparations=preparations,
                            )
                            aa = interval_add(
                                interval_dot(da, da), interval_dot(gap, dda)
                            )
                            ab = tuple(
                                np.asarray(value, dtype=np.float64)
                                for value in interval_dot(da, db)
                            )
                            bb = tuple(
                                np.asarray(value, dtype=np.float64)
                                for value in interval_dot(db, db)
                            )
                            bb = interval_add((-bb[1], -bb[0]), interval_dot(gap, ddb))
                            result = (
                                np.asarray(((aa[0], -ab[1]), (ab[0], bb[0]))),
                                np.asarray(((aa[1], -ab[0]), (ab[1], bb[1]))),
                            )
                    case _PlanarCurveTangencySystem() if not self.jacobian:
                        value_a, value_b = interval_dot(gap, db), determinant(da, db)
                        result = (
                            np.asarray((value_a[0], value_b[0])),
                            np.asarray((value_a[1], value_b[1])),
                        )
                    case _PlanarCurveTangencySystem():
                        dda = jet(
                            self.system.first,
                            float(lo[0]),
                            float(hi[0]),
                            2,
                            budget=budget,
                            preparations=preparations,
                        )
                        ddb = jet(
                            self.system.second,
                            float(lo[1]),
                            float(hi[1]),
                            2,
                            budget=budget,
                            preparations=preparations,
                        )
                        aa = interval_dot(da, db)
                        bb = tuple(
                            np.asarray(value, dtype=np.float64)
                            for value in interval_dot(db, db)
                        )
                        ab = interval_add((-bb[1], -bb[0]), interval_dot(gap, ddb))
                        ba, bb = determinant(dda, db), determinant(da, ddb)
                        result = (
                            np.asarray(((aa[0], ab[0]), (ba[0], bb[0]))),
                            np.asarray(((aa[1], ab[1]), (ba[1], bb[1]))),
                        )
            lower_outputs.append(result[0])
            upper_outputs.append(result[1])
        return np.asarray(lower_outputs), np.asarray(upper_outputs)


def _vector_interval_dot(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    product = tuple(
        np.asarray(value, dtype=np.float64) for value in interval_multiply(first, second)
    )
    return (
        np.asarray(
            np.nextafter(
                np.sum(product[0]) - 8 * _UNIT * np.sum(np.abs(product[0])), -np.inf
            )
        ),
        np.asarray(
            np.nextafter(
                np.sum(product[1]) + 8 * _UNIT * np.sum(np.abs(product[1])), np.inf
            )
        ),
    )


def _vector_interval_cross(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    components = []
    for a, b in ((1, 2), (2, 0), (0, 1)):
        components.append(
            interval_subtract(
                interval_multiply(
                    (np.asarray(first[0][a]), np.asarray(first[1][a])),
                    (np.asarray(second[0][b]), np.asarray(second[1][b])),
                ),
                interval_multiply(
                    (np.asarray(first[0][b]), np.asarray(first[1][b])),
                    (np.asarray(second[0][a]), np.asarray(second[1][a])),
                ),
            )
        )
    return np.asarray([component[0] for component in components]), np.asarray(
        [component[1] for component in components]
    )


@dataclass(frozen=True)
class _IntersectionSurfaceIntervals:
    """Compose spatial source branch bounds with a surface evaluator."""

    system: CurveSurfaceSystem | _CurveSurfaceTangencySystem
    jacobian: bool

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (3, 3) if self.jacobian else (3,)

    def evaluate_scope(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        curve = self.system.curve
        if not isinstance(curve, intersection_curve_owner.IntersectionCurve):
            raise TypeError(
                "source_intervals.curve must be an intersection branch; got another carrier"
            )
        if isinstance(self.system, _CurveSurfaceTangencySystem):
            return self._tangency_enclosures(
                lower, upper, budget=budget, preparations=preparations
            )
        surface_function = (
            jax.jacfwd(self.system.surface.evaluate)
            if self.jacobian
            else self.system.surface.evaluate
        )
        prepared = prepare_interval_function(
            surface_function,
            2,
            batch_capacity=1,
            constant_bounds=coefficient_enclosures(self.system),
        )
        lows, highs = [], []
        for lo, hi in zip(lower, upper, strict=True):
            domain = curve.parameter_domain
            if lo[0] < domain[0] or hi[0] > domain[1]:
                lows.append(np.full(self.output_shape, -np.inf))
                highs.append(np.full(self.output_shape, np.inf))
                continue
            surface_lo, surface_hi = evaluate_interval(
                prepared,
                lo[None, 1:],
                hi[None, 1:],
                budget=budget,
                preparations=preparations,
            )
            if self.jacobian:
                first, last = float(lo[0]), float(hi[0])
                curve_lo, curve_hi = curve.derivative_bounds(
                    first, last, budget=budget, preparations=preparations
                )
                lows.append(np.concatenate((curve_lo[:, None], -surface_hi[0]), axis=1))
                highs.append(np.concatenate((curve_hi[:, None], -surface_lo[0]), axis=1))
            else:
                box = curve.bounding_box(
                    float(lo[0]),
                    float(hi[0]),
                    budget=budget,
                    preparations=preparations,
                )
                value_lo, value_hi = interval_subtract(
                    (box[0], box[1]), (surface_lo[0], surface_hi[0])
                )
                lows.append(value_lo)
                highs.append(value_hi)
        return np.asarray(lows), np.asarray(highs)

    def _tangency_enclosures(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Differentiate gap/frame/tangent equations using source jets."""
        curve = self.system.curve
        if not isinstance(curve, intersection_curve_owner.IntersectionCurve):
            raise TypeError(
                "source_tangency.curve must be an intersection branch; got another carrier"
            )
        surface = self.system.surface
        coefficients = coefficient_enclosures(self.system)
        values = prepare_interval_function(
            surface.evaluate, 2, batch_capacity=1, constant_bounds=coefficients
        )
        frames = prepare_interval_function(
            jax.jacfwd(surface.evaluate),
            2,
            batch_capacity=1,
            constant_bounds=coefficients,
        )
        hessians = (
            prepare_interval_function(
                jax.jacfwd(jax.jacfwd(surface.evaluate)),
                2,
                batch_capacity=1,
                constant_bounds=coefficients,
            )
            if self.jacobian
            else None
        )
        lows, highs = [], []
        for lo, hi in zip(lower, upper, strict=True):
            domain = curve.parameter_domain
            if lo[0] < domain[0] or hi[0] > domain[1]:
                lows.append(np.full(self.output_shape, -np.inf))
                highs.append(np.full(self.output_shape, np.inf))
                continue
            first, last = float(lo[0]), float(hi[0])
            box = curve.bounding_box(
                first, last, preparations=preparations, budget=budget
            )
            surface_lo, surface_hi = evaluate_interval(
                values,
                lo[None, 1:],
                hi[None, 1:],
                budget=budget,
                preparations=preparations,
            )
            gap_lower, gap_upper = interval_subtract(
                (box[0], box[1]), (surface_lo[0], surface_hi[0])
            )
            gap = (
                np.asarray(gap_lower, dtype=np.float64),
                np.asarray(gap_upper, dtype=np.float64),
            )
            frame_lo, frame_hi = evaluate_interval(
                frames,
                lo[None, 1:],
                hi[None, 1:],
                budget=budget,
                preparations=preparations,
            )
            u = (frame_lo[0, :, 0], frame_hi[0, :, 0])
            v = (frame_lo[0, :, 1], frame_hi[0, :, 1])
            normal = _vector_interval_cross(u, v)
            tangent = curve.derivative_bounds(
                first, last, preparations=preparations, budget=budget
            )
            if not self.jacobian:
                rows = (
                    _vector_interval_dot(gap, u),
                    _vector_interval_dot(gap, v),
                    _vector_interval_dot(tangent, normal),
                )
                lows.append(np.asarray([row[0] for row in rows]))
                highs.append(np.asarray([row[1] for row in rows]))
                continue
            if hessians is None:
                raise TypeError(
                    "source_tangency.surface must provide Hessian enclosures; got missing derivative evidence"
                )
            hessian_lo, hessian_hi = evaluate_interval(
                hessians,
                lo[None, 1:],
                hi[None, 1:],
                budget=budget,
                preparations=preparations,
            )
            curvature = curve.derivative_bounds(
                first, last, order=2, preparations=preparations, budget=budget
            )
            matrix_lo, matrix_hi = np.empty((3, 3)), np.empty((3, 3))
            t_rows = (
                _vector_interval_dot(tangent, u),
                _vector_interval_dot(tangent, v),
                _vector_interval_dot(curvature, normal),
            )
            for row, bounds in enumerate(t_rows):
                matrix_lo[row, 0], matrix_hi[row, 0] = bounds
            for column, frame in enumerate((u, v)):
                for row, other in enumerate((u, v)):
                    metric = _vector_interval_dot(frame, other)
                    hessian = (
                        hessian_lo[0, :, row, column],
                        hessian_hi[0, :, row, column],
                    )
                    bounds = interval_add(
                        (-metric[1], -metric[0]), _vector_interval_dot(gap, hessian)
                    )
                    matrix_lo[row, column + 1], matrix_hi[row, column + 1] = bounds
                u_derivative = (hessian_lo[0, :, 0, column], hessian_hi[0, :, 0, column])
                v_derivative = (hessian_lo[0, :, 1, column], hessian_hi[0, :, 1, column])
                normal_lower, normal_upper = interval_add(
                    _vector_interval_cross(u_derivative, v),
                    _vector_interval_cross(u, v_derivative),
                )
                normal_derivative = (
                    np.asarray(normal_lower, dtype=np.float64),
                    np.asarray(normal_upper, dtype=np.float64),
                )
                bounds = _vector_interval_dot(tangent, normal_derivative)
                matrix_lo[2, column + 1], matrix_hi[2, column + 1] = bounds
            lows.append(matrix_lo)
            highs.append(matrix_hi)
        return np.asarray(lows), np.asarray(highs)


@dataclass(frozen=True)
class Prepared:
    """Interval and point evaluation of one system."""

    system: _System
    value: (
        PreparedIntervalFunction
        | _CurveCapabilityIntervals
        | _IntersectionSurfaceIntervals
    )
    jacobian: (
        PreparedIntervalFunction
        | _CurveCapabilityIntervals
        | _IntersectionSurfaceIntervals
    )
    dimension: int
    batch: int
    source_values: Callable[[Array], Array] | None = None
    source_jacobians: Callable[[Array], Array] | None = None
    source_newton: (
        Callable[[Array, VectorLocalRootPlan], tuple[Array, Array, Array, Array]] | None
    ) = None

    def point_values(self, points: np.ndarray, /) -> np.ndarray:
        (values,) = _padded_call(
            lambda chunk: (
                self.source_values(jnp.asarray(chunk))
                if self.source_values is not None
                else _point_values(self.system, jnp.asarray(chunk))
            ),
            points,
            self.batch,
        )
        return values

    def point_jacobians(
        self,
        points: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> np.ndarray:
        (values,) = _padded_call(
            lambda chunk: (
                self.source_jacobians(jnp.asarray(chunk))
                if self.source_jacobians is not None
                else _point_jacobians(self.system, jnp.asarray(chunk))
            ),
            points,
            self.batch,
        )
        return values

    def newton(
        self, points: Array, plan: VectorLocalRootPlan, /
    ) -> tuple[Array, Array, Array, Array]:
        if self.source_newton is not None:
            return self.source_newton(points, plan)
        return _newton(self.system, points, plan)


class _AxialRevolutionEvaluator(AbstractSurfacePatch):
    """Evaluate an on-axis circle sweep with its authored angular parameter."""

    source: RevolutionSurface
    profile: CircleCurve
    radial_first: Array
    radial_second: Array
    tangent_first: Array
    tangent_second: Array
    axial_first: Array
    axial_second: Array

    def __init__(self, source: RevolutionSurface, /) -> None:
        if not isinstance(source.curve, CircleCurve):
            raise TypeError(
                "axial_revolution.source.curve must be a circle; got unsupported profile"
            )
        axis = np.asarray(source.axis_direction)
        active = np.flatnonzero(axis)
        if active.size != 1 or abs(axis[active[0]]) != 1.0:
            raise ValueError(
                "axial_revolution.source.axis_direction must be a signed coordinate axis; got another axis"
            )
        index = int(active[0])
        if any(
            np.asarray(source.curve.center)[i] != np.asarray(source.axis_origin)[i]
            for i in range(3)
            if i != index
        ):
            raise ValueError(
                "axial_revolution.source.curve.center must lie on the rotation axis; got an off-axis center"
            )
        self.source, self.profile = source, source.curve
        first, second = (
            np.asarray(source.curve.first_axis),
            np.asarray(source.curve.second_axis),
        )
        radial_first, radial_second = first.copy(), second.copy()
        radial_first[index] = radial_second[index] = 0.0
        axial_first, axial_second = (
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
        axial_first[index], axial_second[index] = first[index], second[index]
        self.radial_first, self.radial_second = (
            jnp.asarray(radial_first),
            jnp.asarray(radial_second),
        )
        self.tangent_first = jnp.asarray(np.cross(axis, radial_first))
        self.tangent_second = jnp.asarray(np.cross(axis, radial_second))
        self.axial_first, self.axial_second = (
            jnp.asarray(axial_first),
            jnp.asarray(axial_second),
        )

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.source.periods

    def evaluate(self, parameters: Array, /) -> Array:
        longitude, angle = parameters[..., 0], parameters[..., 1]
        cosine, sine = jnp.cos(angle)[..., None], jnp.sin(angle)[..., None]
        radial = cosine * self.radial_first + sine * self.radial_second
        tangent = cosine * self.tangent_first + sine * self.tangent_second
        axial = cosine * self.axial_first + sine * self.axial_second
        return self.profile.center + self.profile.radius * (
            jnp.cos(longitude)[..., None] * radial
            + jnp.sin(longitude)[..., None] * tangent
            + axial
        )

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        return self.source.bounding_box(
            parameter_box, preparations=preparations, budget=budget
        )

    def validate_parameter_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.validate_parameter_box(parameter_box, budget=budget)
        return self.source.validate_parameter_box(parameter_box, budget=budget)

    def degenerate_isolines(
        self, parameter_box: ConvertibleToArray, /
    ) -> tuple[tuple[int, float], ...]:
        return self.source.degenerate_isolines(parameter_box)


def _canonical_surface_evaluator(
    surface: AbstractSurfacePatch, /
) -> AbstractSurfacePatch:
    """Lower a proved Rodrigues law at its authored (u, source-t) coordinates.

    Evaluator simplification preserves regions, topology, source records,
    parameter domains and period declarations.
    """

    if not isinstance(surface, RevolutionSurface) or not isinstance(
        surface.curve, CircleCurve
    ):
        return surface
    profile = surface.curve
    axis = rational_vector(np.asarray(surface.axis_direction))
    active = [index for index, value in enumerate(axis) if value]
    if len(active) != 1 or abs(axis[active[0]]) != 1:
        return surface
    index = active[0]
    delta = tuple(
        F(float(a)) - F(float(b))
        for a, b in zip(
            np.asarray(profile.center), np.asarray(surface.axis_origin), strict=True
        )
    )
    if any(value for i, value in enumerate(delta) if i != index):
        return surface
    first = rational_vector(np.asarray(profile.first_axis))
    second = rational_vector(np.asarray(profile.second_axis))
    opposite = tuple(-value for value in axis)
    if (
        (second == axis or second == opposite)
        and sum((value * value for value in first), F(0)) == 1
        and first[index] == 0
    ):
        tangent = np.cross(
            np.asarray(surface.axis_direction), np.asarray(profile.first_axis)
        )
        return SpherePatch(
            profile.center,
            profile.first_axis,
            tangent,
            profile.second_axis,
            profile.radius,
        )
    # Preserve the authored angle and each binary64 coefficient.
    # Axial terms are collected before evaluating u, removing the artificial
    # dependency in cos(u)*axial + (1-cos(u))*axial exactly.
    return _AxialRevolutionEvaluator(surface)


def _pose_cancelled_pair(
    system: intersection_curve_owner.SurfacePairSystem, /
) -> tuple[intersection_curve_owner.SurfacePairSystem, np.ndarray] | None:
    """Cancel a common pose in a placed plane/plane equation.

    ``(R f(p) + t) - (R g(q) + t) = R (f(p) - g(q))`` holds in the authored
    operation tree, and ``same_source_pose`` proves one identical ``R, t``
    with positive determinant, so both equations have one zero set. Only the
    equations are prepared in the shared source frame: regions, curves, roots
    and point maps keep the placed carriers. The returned rotation
    maps source-frame residual bounds back to world bounds.
    """
    first, second = system.first, system.second
    if not (
        isinstance(first, placed_owner.PlacedSurface)
        and isinstance(second, placed_owner.PlacedSurface)
    ):
        return None
    if not (
        isinstance(first.definition, PlanePatch)
        and isinstance(second.definition, PlanePatch)
    ):
        return None
    if not placed_owner.same_source_pose(first, second):
        return None
    return intersection_curve_owner.SurfacePairSystem(
        first.definition, second.definition
    ), np.asarray(first.rotation)


def _source_frame_scales(scales: _Scales, rotation: np.ndarray, /) -> _Scales:
    """World residual tolerances, rounded inward through the pose operator bound."""

    norm = source_correspondence_owner.placed_deviation_bound(rotation, 1.0)
    return replace(
        scales,
        coincidence=float(np.nextafter(scales.coincidence / norm, 0.0)),
        tangency=float(np.nextafter(scales.tangency / norm, 0.0)),
    )


def prepare(
    system: _System,
    dimension: int,
    batch: int,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
) -> Prepared:
    request = TrimRequestPreparations() if preparations is None else preparations
    match system:
        case intersection_curve_owner.SurfacePairSystem():
            cancelled = _pose_cancelled_pair(system)
            if cancelled is not None:
                system = cancelled[0]
            first, second = (
                _canonical_surface_evaluator(system.first),
                _canonical_surface_evaluator(system.second),
            )
            if first is not system.first or second is not system.second:
                system = intersection_curve_owner.SurfacePairSystem(first, second)
        case CurveSurfaceSystem():
            surface = _canonical_surface_evaluator(system.surface)
            if surface is not system.surface:
                system = CurveSurfaceSystem(system.curve, surface)
        case _CurveSurfaceTangencySystem():
            surface = _canonical_surface_evaluator(system.surface)
            if surface is not system.surface:
                system = _CurveSurfaceTangencySystem(system.curve, surface)
        case TripleSurfaceSystem():
            first = _canonical_surface_evaluator(system.first)
            second = _canonical_surface_evaluator(system.second)
            third = _canonical_surface_evaluator(system.third)
            if (
                first is not system.first
                or second is not system.second
                or third is not system.third
            ):
                system = TripleSurfaceSystem(first, second, third)
    pair_key = None
    source_key = None
    if isinstance(system, intersection_curve_owner.SurfacePairSystem):
        pair_key = (canonical_fingerprint(array_tree_fingerprint(system)), batch)
        cached = request.pair_programs.get(pair_key)
    else:
        source_key = (
            str(jax.tree_util.tree_structure(system)),
            canonical_fingerprint(array_tree_fingerprint(system)),
            dimension,
            batch,
        )
        cached = request.source_programs.get(source_key)
    if cached is not None:
        return cached
    capability_system = (
        isinstance(system, (CurveSurfaceSystem, _CurveSurfaceTangencySystem))
        and isinstance(system.curve, intersection_curve_owner.IntersectionCurve)
    ) or (
        isinstance(
            system, (_CurvePairSystem, _CurveProximitySystem, _PlanarCurveTangencySystem)
        )
        and (
            isinstance(
                system.first,
                (AbstractTrimCurve, intersection_curve_owner.IntersectionCurve),
            )
            or isinstance(
                system.second,
                (AbstractTrimCurve, intersection_curve_owner.IntersectionCurve),
            )
        )
    )
    if capability_system:
        # Close over source definitions before JAX tracing. Passing them as a
        # dynamic module tree would trace host chart/span identities as arrays.
        value: _IntersectionSurfaceIntervals | _CurveCapabilityIntervals
        jacobian: _IntersectionSurfaceIntervals | _CurveCapabilityIntervals
        match system:
            case CurveSurfaceSystem() | _CurveSurfaceTangencySystem():
                value = _IntersectionSurfaceIntervals(system, False)
                jacobian = _IntersectionSurfaceIntervals(system, True)
            case (
                _CurvePairSystem()
                | _CurveProximitySystem()
                | _PlanarCurveTangencySystem()
            ):
                value = _CurveCapabilityIntervals(system, False)
                jacobian = _CurveCapabilityIntervals(system, True)
            case _:
                raise TypeError(
                    "intersection_preparation.system must support interval evaluation; got an unsupported system"
                )
        prepared = Prepared(
            system,
            value,
            jacobian,
            dimension,
            batch,
            jax.jit(jax.vmap(system.residual)),
            jax.jit(jax.vmap(jax.jacfwd(system.residual))),
            eqx.filter_jit(lambda points, plan: _newton_values(system, points, plan)),
        )
    else:
        coefficients = coefficient_enclosures(system)
        prepared = Prepared(
            system,
            prepare_interval_function(
                system.residual,
                dimension,
                batch_capacity=batch,
                constant_bounds=coefficients,
            ),
            prepare_interval_function(
                jax.jacfwd(system.residual),
                dimension,
                batch_capacity=batch,
                constant_bounds=coefficients,
            ),
            dimension,
            batch,
        )
    if source_key is not None:
        if len(request.source_programs) >= 128:
            del request.source_programs[next(iter(request.source_programs))]
        request.source_programs[source_key] = prepared
    if pair_key is not None:
        if len(request.pair_programs) >= 128:
            del request.pair_programs[next(iter(request.pair_programs))]
        request.pair_programs[pair_key] = prepared
    return prepared


# ------------------------------------------------------------ root isolation


@dataclass(frozen=True)
class _Layout:
    """Which unknowns belong to which operand of ``F = A(x_A) - B(x_B)``."""

    first: tuple[int, ...]
    second: tuple[int, ...]
    ambient: int


@dataclass
class _Isolated:
    roots: list[ParametricIntersectionPoint] = field(default_factory=list)
    root_boxes: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    contacts: list[ParametricIntersectionPoint] = field(default_factory=list)
    contact_boxes: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    coincident: list[tuple[np.ndarray, np.ndarray, bool]] = field(default_factory=list)
    coincidence_bound: float = 0.0
    unresolved: list[UnresolvedParameterRegion] = field(default_factory=list)


@dataclass(frozen=True)
class _Problem:
    """A square system with its operand layout and optional auxiliary systems.

    ``excluded`` holds boxes (problem coordinates) already accounted by a
    coincident component; ``junction`` is the scaled radius vector of the
    neighborhood summarized by a certified singular contact.
    """

    system: Prepared
    point: Callable[[np.ndarray], np.ndarray]
    pair: Prepared | None
    layout: _Layout | None
    tangency: Prepared | None
    proximity: bool
    excluded: tuple[np.ndarray, np.ndarray] | None = None
    junction: np.ndarray | None = None


def _image_size(
    jacobian_lower: np.ndarray, jacobian_upper: np.ndarray, widths: np.ndarray, /
) -> np.ndarray:
    return np.max(
        np.sum(
            interval_magnitude(jacobian_lower, jacobian_upper) * widths[:, None, :],
            axis=2,
        ),
        axis=1,
    )


def _inside_any(
    lower: np.ndarray, upper: np.ndarray, boxes: list[tuple[np.ndarray, np.ndarray]], /
) -> np.ndarray:
    inside = np.zeros((lower.shape[0],), dtype=np.bool_)
    for box_lower, box_upper in boxes:
        inside |= np.all((lower >= box_lower) & (upper <= box_upper), axis=1)
    return inside


def _inside_cells(
    lower: np.ndarray, upper: np.ndarray, cells: tuple[np.ndarray, np.ndarray], /
) -> np.ndarray:
    return np.any(
        np.all(
            (lower[:, None, :] >= cells[0][None]) & (upper[:, None, :] <= cells[1][None]),
            axis=2,
        ),
        axis=1,
    )


def _coincidence(
    problem: _Problem,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
    /,
    anchor: np.ndarray | None = None,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normal-distance and projection coverage against an affine partner.

    A linear projection is not a continuum coincidence test for a curved
    target. Such candidates remain in subdivision rather than silently using
    its tangent plane as the represented surface.
    """
    pair = problem.pair
    layout = problem.layout
    count = lower.shape[0]
    none = np.zeros((count,), dtype=np.bool_)
    if pair is None or layout is None:
        return none, none, np.full((count,), np.inf)
    ambient = layout.ambient
    if len(layout.second) == ambient - 1:
        target, source, sign = layout.second, layout.first, 1.0
    elif len(layout.first) == ambient - 1:
        target, source, sign = layout.first, layout.second, -1.0
    else:
        return none, none, np.full((count,), np.inf)
    target_ = np.asarray(target)
    system = pair.system
    match system:
        case CurveSurfaceSystem():
            target_evaluator = system.surface
        case (
            _CurvePairSystem()
            | _CurveProximitySystem()
            | intersection_curve_owner.SurfacePairSystem()
        ):
            target_evaluator = system.second if sign > 0 else system.first
        case _FaceSystem():
            target_evaluator = system.pair.second if sign > 0 else system.pair.first
        case _:
            return none, none, np.full((count,), np.inf)
    if not isinstance(target_evaluator, (LineCurve, PlanePatch)):
        return none, none, np.full((count,), np.inf)
    center = 0.5 * (lower + upper) if anchor is None else anchor
    probe_lower = lower.copy()
    probe_upper = upper.copy()
    probe_lower[:, target_] = center[:, target_]
    probe_upper[:, target_] = center[:, target_]
    gap_lower, gap_upper = evaluate_interval(
        pair.value, probe_lower, probe_upper, budget=budget, preparations=preparations
    )
    if sign < 0.0:
        gap_lower, gap_upper = -gap_upper, -gap_lower
    jacobian = pair.point_jacobians(center, budget=budget, preparations=preparations)
    frame = -sign * jacobian[:, :, target_]
    tangents = jacobian[:, :, np.asarray(source)]
    if ambient == 2:
        normal = np.stack((-frame[:, 1, 0], frame[:, 0, 0]), axis=1)
    else:
        normal = np.cross(frame[:, :, 0], frame[:, :, 1])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    regular = length[:, 0] > 0.0
    normal = normal / np.where(length > 0.0, length, 1.0)
    distance_lower, distance_upper = point_times_interval(
        normal[:, None, :], gap_lower[..., None], gap_upper[..., None]
    )
    distance = interval_magnitude(distance_lower, distance_upper)[:, 0, 0]
    gram = np.swapaxes(frame, 1, 2) @ frame
    projector = approximate_inverses(gram) @ np.swapaxes(frame, 1, 2)
    step_lower, step_upper = point_times_interval(
        projector, gap_lower[..., None], gap_upper[..., None]
    )
    projected_lower = center[:, target_] + step_lower[..., 0]
    projected_upper = center[:, target_] + step_upper[..., 0]
    # A coincident source is tangent to the partner: a transversal crossing near
    # a tiny box can pass the distance test but never this alignment test.
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    alignment = np.max(
        np.abs((normal[:, None, :] @ tangents)[:, 0, :])
        / np.where(tangent_lengths > 0.0, tangent_lengths, 1.0),
        axis=1,
    )
    close = regular & (distance <= tolerance) & (alignment <= 1.0e-8)
    covered = np.all(
        (projected_lower >= lower[:, target_]) & (projected_upper <= upper[:, target_]),
        axis=1,
    )
    meets = np.all(
        (projected_upper >= lower[:, target_]) & (projected_lower <= upper[:, target_]),
        axis=1,
    )
    return close & covered, close & meets & ~covered, distance


def _record_root(
    problem: _Problem,
    isolated: _Isolated,
    point: np.ndarray,
    condition: float,
    unique: tuple[np.ndarray, np.ndarray],
    tight: tuple[np.ndarray, np.ndarray],
    tolerance: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> None:
    """Record a certified root unless its uniqueness box already holds one."""
    for existing_lower, existing_upper in isolated.root_boxes:
        if np.all((point >= existing_lower) & (point <= existing_upper)):
            return
    isolated.root_boxes.append(unique)
    pair = problem.pair if problem.pair is not None else problem.system
    value_lower, value_upper = evaluate_interval(
        pair.value,
        tight[0][None],
        tight[1][None],
        budget=budget,
        preparations=preparations,
    )
    gap = float(np.linalg.norm(interval_magnitude(value_lower, value_upper)[0]))
    if problem.proximity and gap > tolerance:
        # A certified critical point of the operand distance with a positive gap
        # is not an intersection; its box is still accounted as decided.
        return
    isolated.roots.append(
        ParametricIntersectionPoint(
            kind="crossing",
            certificate="proximity-krawczyk" if problem.proximity else "krawczyk",
            parameters=np.clip(point, tight[0], tight[1]),
            parameter_lower=tight[0],
            parameter_upper=tight[1],
            point=problem.point(np.clip(point, tight[0], tight[1])),
            gap_bound=gap,
            condition_estimate=condition,
        )
    )


def _contract(
    prepared: KrawczykEvaluator,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
    *,
    maximum_steps: int = 8,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate Krawczyk on certified boxes; every iterate still holds the root."""
    maximum_steps = nonnegative_integer(maximum_steps, "maximum_steps")
    for _ in range(maximum_steps):
        result = krawczyk(
            prepared, lower, upper, budget=budget, preparations=preparations
        )
        shrink = result.certified[:, None]
        new_lower = np.where(shrink, np.maximum(lower, result.lower), lower)
        new_upper = np.where(shrink, np.minimum(upper, result.upper), upper)
        if np.array_equal(new_lower, lower) and np.array_equal(new_upper, upper):
            break
        lower, upper = new_lower, new_upper
    return lower, upper


def _certify_near(
    prepared: Prepared,
    points: np.ndarray,
    widths: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """epsilon-inflation Krawczyk around converged Newton points.

    Returns the certified mask, the uniqueness boxes and contracted enclosures.
    """
    count = points.shape[0]
    certified = np.zeros((count,), dtype=np.bool_)
    lower = points.copy()
    upper = points.copy()
    seed = krawczyk(prepared, points, points, budget=budget, preparations=preparations)
    seed_radius = np.maximum(np.abs(seed.lower - points), np.abs(seed.upper - points))
    seed_radius = np.where(
        np.isfinite(seed_radius), np.nextafter(2.0 * seed_radius, np.inf), 0.0
    )
    for exponent in (1, 2, 4, 6, 8):
        pending = ~certified
        if not np.any(pending):
            break
        radius = widths[pending] * 10.0 ** (-exponent) + 64.0 * _UNIT * (
            1.0 + np.abs(points[pending])
        )
        # Source node enclosures can be wider than a contracted leaf's
        # numerical coordinates. Inflate by the interval Newton image, not by
        # a geometry tolerance; strict Krawczyk inclusion remains mandatory.
        radius = np.maximum(radius, seed_radius[pending])
        trial_lower = points[pending] - radius
        trial_upper = points[pending] + radius
        result = krawczyk(
            prepared, trial_lower, trial_upper, budget=budget, preparations=preparations
        )
        rows = np.flatnonzero(pending)[result.certified]
        certified[rows] = True
        lower[rows] = trial_lower[result.certified]
        upper[rows] = trial_upper[result.certified]
    if not np.any(certified):
        empty = (lower[certified], upper[certified])
        return certified, empty, empty
    tight = _contract(
        prepared,
        lower[certified],
        upper[certified],
        budget=budget,
        preparations=preparations,
    )
    return certified, (lower[certified], upper[certified]), tight


def _tangency_probe(
    problem: _Problem,
    isolated: _Isolated,
    centers: np.ndarray,
    widths: np.ndarray,
    tolerance: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> None:
    tangency = problem.tangency
    if tangency is None:
        return
    plan = VectorLocalRootPlan(
        tangency.dimension,
        maximum_steps=60,
        tolerance=1.0e-14,
        plan_id="intersection-tangency",
    )
    roots, converged, _, condition = _padded_call(
        lambda chunk: tangency.newton(jnp.asarray(chunk), plan),
        centers,
        tangency.batch,
    )
    usable = converged & np.all(np.isfinite(roots), axis=1)
    usable &= np.all(np.abs(roots - centers) <= 4.0 * widths + 1.0e-300, axis=1)
    if not np.any(usable):
        return
    certified, unique, tight = _certify_near(
        tangency, roots[usable], widths[usable], budget=budget, preparations=preparations
    )
    for point, estimate, unique_lower, unique_upper, tight_lower, tight_upper in zip(
        roots[usable][certified],
        condition[usable][certified],
        *unique,
        *tight,
        strict=True,
    ):
        if any(
            np.all((point >= low) & (point <= high))
            for low, high in isolated.contact_boxes
        ):
            continue
        pair = problem.pair if problem.pair is not None else problem.system
        gap_lower, gap_upper = evaluate_interval(
            pair.value,
            tight_lower[None],
            tight_upper[None],
            budget=budget,
            preparations=preparations,
        )
        gap = float(np.linalg.norm(interval_magnitude(gap_lower, gap_upper)[0]))
        if gap > tolerance:
            # A distance-critical point with a positive gap proves nothing about
            # nearby roots of the operand equations; subdivision continues.
            continue
        isolated.contact_boxes.append((unique_lower, unique_upper))
        representative = np.clip(point, tight_lower, tight_upper)
        if problem.junction is not None:
            # The certified contact summarizes its junction neighborhood; the
            # branches leaving it are isolated separately on its boundary.
            isolated.root_boxes.append(
                (representative - problem.junction, representative + problem.junction)
            )
        isolated.contacts.append(
            ParametricIntersectionPoint(
                kind="singular_point"
                if _operand_degenerate(
                    problem, point, budget=budget, preparations=preparations
                )
                else "tangent_point",
                certificate="tangency-krawczyk",
                parameters=representative,
                parameter_lower=tight_lower,
                parameter_upper=tight_upper,
                point=problem.point(representative),
                gap_bound=gap,
                condition_estimate=float(estimate),
            )
        )


def _operand_degenerate(
    problem: _Problem,
    point: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> bool:
    """Whether an operand's parametrization loses rank at ``point``."""
    pair = problem.pair if problem.pair is not None else problem.system
    layout = problem.layout
    if layout is None:
        return False
    jacobian = pair.point_jacobians(
        point[None], budget=budget, preparations=preparations
    )[0]
    scale = max(float(np.max(np.abs(jacobian))), np.finfo(np.float64).tiny)
    for columns in (layout.first, layout.second):
        block = jacobian[:, np.asarray(columns)]
        singular_values = np.asarray(
            orthonormal_frame(jnp.asarray(block)).singular_values
        )
        if np.min(singular_values) <= 1.0e-10 * scale:
            return True
    return False


@dataclass(frozen=True)
class _Scales:
    resolution: float
    probe: float
    coincidence: float
    tangency: float


def _isolate(
    problem: _Problem,
    lower: np.ndarray,
    upper: np.ndarray,
    scales: _Scales,
    budget: _Budget,
    batch: int,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> _Isolated:
    """Bounded subdivision with exclusion, Krawczyk, probes and explicit leftovers."""
    isolated = _Isolated()
    prepared = problem.system
    dimension = prepared.dimension
    newton_plan = VectorLocalRootPlan(
        dimension, maximum_steps=40, tolerance=1.0e-14, plan_id="intersection-root"
    )
    queue_lower = [np.asarray(lower, dtype=np.float64)[None]]
    queue_upper = [np.asarray(upper, dtype=np.float64)[None]]
    parameter_widths = queue_upper[0][0] - queue_lower[0][0]
    while queue_lower:
        pending_lower = np.concatenate(queue_lower)
        pending_upper = np.concatenate(queue_upper)
        queue_lower, queue_upper = [], []
        for start in range(0, pending_lower.shape[0], batch):
            box_lower = pending_lower[start : start + batch]
            box_upper = pending_upper[start : start + batch]
            if budget.exhausted or budget.processed + box_lower.shape[0] > budget.maximum:
                budget.exhausted = True
                for low, high in zip(box_lower, box_upper, strict=True):
                    isolated.unresolved.append(
                        UnresolvedParameterRegion(low, high, "budget")
                    )
                budget.unresolved += box_lower.shape[0]
                continue
            budget.processed += box_lower.shape[0]
            keep = ~_inside_any(box_lower, box_upper, isolated.root_boxes)
            budget.excluded += int(np.sum(~keep))
            if problem.excluded is not None and problem.excluded[0].shape[0]:
                accounted = _inside_cells(box_lower, box_upper, problem.excluded) & keep
                budget.coincident += int(np.sum(accounted))
                keep &= ~accounted
            box_lower, box_upper = box_lower[keep], box_upper[keep]
            if box_lower.shape[0] == 0:
                continue
            split_lower, split_upper = _process(
                problem,
                isolated,
                box_lower,
                box_upper,
                scales,
                budget,
                newton_plan,
                parameter_widths,
                coordinate_budget=coordinate_budget,
                preparations=preparations,
            )
            queue_lower.append(split_lower)
            queue_upper.append(split_upper)
        queue_lower = [part for part in queue_lower if part.shape[0]]
        queue_upper = [part for part in queue_upper if part.shape[0]]
    return isolated


def _process(
    problem: _Problem,
    isolated: _Isolated,
    lower: np.ndarray,
    upper: np.ndarray,
    scales: _Scales,
    budget: _Budget,
    newton_plan: VectorLocalRootPlan,
    parameter_widths: np.ndarray,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = problem.system
    value_lower, value_upper = evaluate_interval(
        prepared.value, lower, upper, budget=coordinate_budget, preparations=preparations
    )
    alive = _contains_zero(value_lower, value_upper)
    if problem.pair is not None and problem.proximity:
        gap_lower, gap_upper = evaluate_interval(
            problem.pair.value,
            lower,
            upper,
            budget=coordinate_budget,
            preparations=preparations,
        )
        alive &= np.all(
            (gap_lower <= scales.tangency) & (gap_upper >= -scales.tangency), axis=1
        )
    jacobian_lower, jacobian_upper = evaluate_interval(
        prepared.jacobian,
        lower,
        upper,
        budget=coordinate_budget,
        preparations=preparations,
    )
    center = 0.5 * (lower + upper)
    center_lower, center_upper = evaluate_interval(
        prepared.value,
        center,
        center,
        budget=coordinate_budget,
        preparations=preparations,
    )
    spread_lower, spread_upper = interval_times_interval(
        jacobian_lower,
        jacobian_upper,
        (lower - center)[..., None],
        (upper - center)[..., None],
    )
    centered_lower, centered_upper = round_interval(
        center_lower + spread_lower[..., 0], center_upper + spread_upper[..., 0]
    )
    alive &= _contains_zero(centered_lower, centered_upper)
    budget.excluded += int(np.sum(~alive))
    lower, upper = lower[alive], upper[alive]
    jacobian_lower, jacobian_upper = jacobian_lower[alive], jacobian_upper[alive]
    if lower.shape[0] == 0:
        return lower, upper
    widths = upper - lower
    # Box sizes are measured in the operand-gap rows (ambient length), not in the
    # auxiliary turning/tangency rows whose units differ.
    gap_rows = _gap_rows(problem)
    size = _image_size(jacobian_lower[:, :gap_rows], jacobian_upper[:, :gap_rows], widths)
    krawczyk_bank = krawczyk(
        prepared, lower, upper, budget=coordinate_budget, preparations=preparations
    )
    budget.excluded += int(np.sum(krawczyk_bank.excluded))
    certified_rows = np.flatnonzero(krawczyk_bank.certified)
    budget.certified += certified_rows.shape[0]
    if certified_rows.shape[0]:
        unique = (lower[certified_rows], upper[certified_rows])
        tight = _contract(
            prepared,
            krawczyk_bank.lower[certified_rows],
            krawczyk_bank.upper[certified_rows],
            budget=coordinate_budget,
            preparations=preparations,
        )
        points, conditions = _refine(problem, 0.5 * (tight[0] + tight[1]), newton_plan)
        for index in range(certified_rows.shape[0]):
            _record_root(
                problem,
                isolated,
                points[index],
                float(conditions[index]),
                (unique[0][index], unique[1][index]),
                (tight[0][index], tight[1][index]),
                scales.tangency,
                budget=coordinate_budget,
                preparations=preparations,
            )
    undecided = ~krawczyk_bank.certified & ~krawczyk_bank.excluded
    # Krawczyk contraction: intersect with the image when it shrinks the box.
    contracted_lower = np.maximum(lower, krawczyk_bank.lower)
    contracted_upper = np.minimum(upper, krawczyk_bank.upper)
    lower = np.where(undecided[:, None], contracted_lower, lower)[undecided]
    upper = np.where(undecided[:, None], contracted_upper, upper)[undecided]
    size = size[undecided]
    if lower.shape[0] == 0:
        return lower, upper
    full, partial, distance = _coincidence(
        problem,
        lower,
        upper,
        scales.coincidence,
        budget=coordinate_budget,
        preparations=preparations,
    )
    # Boxes far below the probe scale can meet the distance test at a tangency;
    # a coincident component is claimed only on boxes of at least probe size.
    claimable = size > 0.25 * scales.probe
    full &= claimable
    partial &= claimable & (size <= scales.probe)
    for row in np.flatnonzero(full | partial):
        isolated.coincident.append((lower[row], upper[row], bool(partial[row])))
        isolated.coincidence_bound = max(isolated.coincidence_bound, float(distance[row]))
        budget.coincident += 1
    remaining = ~(full | partial)
    lower, upper, size = lower[remaining], upper[remaining], size[remaining]
    small = size <= scales.probe
    if np.any(small):
        _probe(
            problem,
            isolated,
            lower[small],
            upper[small],
            newton_plan,
            scales,
            budget=coordinate_budget,
            preparations=preparations,
        )
        keep = ~_inside_any(lower, upper, isolated.root_boxes)
        lower, upper, size = lower[keep], upper[keep], size[keep]
    leaf = size <= scales.resolution
    for low, high in zip(lower[leaf], upper[leaf], strict=True):
        isolated.unresolved.append(UnresolvedParameterRegion(low, high, "resolution"))
    budget.unresolved += int(np.sum(leaf))
    lower, upper = lower[~leaf], upper[~leaf]
    if lower.shape[0] == 0:
        return lower, upper
    jacobian_lower, jacobian_upper = evaluate_interval(
        prepared.jacobian,
        lower,
        upper,
        budget=coordinate_budget,
        preparations=preparations,
    )
    gap_rows = _gap_rows(problem)
    contribution = np.max(
        interval_magnitude(jacobian_lower[:, :gap_rows], jacobian_upper[:, :gap_rows]),
        axis=1,
    ) * (upper - lower)
    axis = np.argmax(contribution, axis=1)
    # Unbounded rational interval jets carry no sensitivity ordering. Always
    # selecting the first infinite column starves the denominator's other source
    # coordinates. Bisect the widest remaining normalized uncertain coordinate.
    uncertain = ~np.isfinite(contribution)
    relative_widths = np.divide(
        upper - lower,
        parameter_widths,
        out=np.zeros_like(lower),
        where=parameter_widths > 0,
    )
    axis = np.where(
        np.any(uncertain, axis=1),
        np.argmax(np.where(uncertain, relative_widths, -np.inf), axis=1),
        axis,
    )
    rows = np.arange(lower.shape[0])
    middle = 0.5 * (lower[rows, axis] + upper[rows, axis])
    left_upper = upper.copy()
    left_upper[rows, axis] = middle
    right_lower = lower.copy()
    right_lower[rows, axis] = middle
    return np.concatenate((lower, right_lower)), np.concatenate((left_upper, upper))


def _gap_rows(problem: _Problem, /) -> int:
    rows = problem.system.value.output_shape[0]
    if problem.pair is None:
        return rows
    return min(rows, problem.pair.value.output_shape[0])


def _refine(
    problem: _Problem, points: np.ndarray, plan: VectorLocalRootPlan, /
) -> tuple[np.ndarray, np.ndarray]:
    """Newton-polish representatives inside certified enclosures."""
    roots, converged, _, condition = _padded_call(
        lambda chunk: problem.system.newton(jnp.asarray(chunk), plan),
        points,
        problem.system.batch,
    )
    usable = (converged & np.all(np.isfinite(roots), axis=1))[:, None]
    return np.where(usable, roots, points), condition


def _probe(
    problem: _Problem,
    isolated: _Isolated,
    lower: np.ndarray,
    upper: np.ndarray,
    plan: VectorLocalRootPlan,
    scales: _Scales,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> None:
    """Newton plus epsilon-inflation certification in small undecided boxes."""
    center = 0.5 * (lower + upper)
    widths = upper - lower
    roots, converged, _, condition = _padded_call(
        lambda chunk: problem.system.newton(jnp.asarray(chunk), plan),
        center,
        problem.system.batch,
    )
    usable = converged & np.all(np.isfinite(roots), axis=1)
    # Newton's residual stopping criterion can move a few ulps beyond an
    # already contracted endpoint box. Keep its seed near the undecided leaf;
    # only the subsequent source-system inclusion can certify an event.
    roots = np.clip(roots, lower, upper)
    regular = np.zeros_like(usable)
    if np.any(usable):
        certified, unique, tight = _certify_near(
            problem.system,
            roots[usable],
            widths[usable],
            budget=budget,
            preparations=preparations,
        )
        regular[np.flatnonzero(usable)[certified]] = True
        if np.any(certified):
            midpoint = 0.5 * (tight[0] + tight[1])
            refined, refined_condition = _refine(problem, midpoint, plan)
            inside = np.all(
                (refined >= tight[0]) & (refined <= tight[1]),
                axis=1,
            )
            representatives = np.where(inside[:, None], refined, midpoint)
            for index, (point, estimate) in enumerate(
                zip(representatives, refined_condition, strict=True)
            ):
                _record_root(
                    problem,
                    isolated,
                    point,
                    float(estimate),
                    (unique[0][index], unique[1][index]),
                    (tight[0][index], tight[1][index]),
                    scales.tangency,
                    budget=budget,
                    preparations=preparations,
                )
    # Newton failed or its limit is not a certifiable regular root: probe for a
    # certified tangential contact instead.
    singular = ~regular
    if np.any(singular):
        _tangency_probe(
            problem,
            isolated,
            center[singular],
            widths[singular],
            scales.tangency,
            budget=budget,
            preparations=preparations,
        )


# --------------------------------------------------------- operand helpers


def _scale(boxes: Sequence[np.ndarray], /) -> float:
    stacked = np.stack(boxes)
    lower = np.min(stacked[:, 0], axis=0)
    upper = np.max(stacked[:, 1], axis=0)
    return max(float(np.linalg.norm(upper - lower)), np.finfo(np.float64).tiny)


def _scales(policy: ParametricIntersectionPolicy, scale: float, /) -> _Scales:
    return _Scales(
        resolution=policy.relative_resolution * scale,
        probe=policy.relative_probe * scale,
        coincidence=policy.relative_coincidence * scale,
        tangency=policy.relative_tangency * scale,
    )


def _dedupe_points(
    points: list[ParametricIntersectionPoint], periods: Sequence[float | None], /
) -> list[ParametricIntersectionPoint]:
    """Remove events found twice on identified faces (seams, span boundaries)."""
    kept: list[ParametricIntersectionPoint] = []
    for point in points:
        duplicate = False
        for existing in kept:
            difference = np.abs(point.parameters - existing.parameters)
            for axis, period in enumerate(periods):
                if period is not None:
                    difference[axis] = min(
                        difference[axis], abs(difference[axis] - period)
                    )
            widths = np.maximum(
                point.parameter_upper - point.parameter_lower,
                existing.parameter_upper - existing.parameter_lower,
            )
            if np.all(difference <= widths + 1.0e-12 * (1.0 + np.abs(point.parameters))):
                duplicate = True
                break
        if not duplicate:
            kept.append(point)
    return kept


def _components(
    cells: list[tuple[np.ndarray, np.ndarray, bool]], bound: float, /
) -> list[CoincidentParameterRegion]:
    """Group coincident cells into connected components by closed-box contact."""
    count = len(cells)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(count):
        for second in range(first + 1, count):
            if np.all(cells[first][0] <= cells[second][1]) and np.all(
                cells[second][0] <= cells[first][1]
            ):
                parent[find(first)] = find(second)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return [
        CoincidentParameterRegion(
            cell_lower=np.stack([cells[index][0] for index in members]),
            cell_upper=np.stack([cells[index][1] for index in members]),
            partial=np.asarray([cells[index][2] for index in members], dtype=np.bool_),
            distance_bound=bound,
        )
        for members in sorted(groups.values())
    ]


# ------------------------------------------------------------ curve routes


@eqx.filter_jit
def _evaluate_curves(
    evaluator: intersection_curve_owner.CurveEvaluator, parameters: Array
) -> Array:
    return jax.vmap(evaluator.evaluate)(parameters)


def _curve_point_map(
    curve: intersection_curve_owner.CurveEvaluator
    | intersection_curve_owner.IntersectionCurve,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(curve, intersection_curve_owner.IntersectionCurve):
        return lambda parameters: np.asarray(
            curve.evaluate(
                jnp.asarray(parameters[0]), budget=budget, preparations=preparations
            ).point
        )
    if isinstance(curve, AbstractTrimCurve):
        # Chart identities are host preparation, not dynamic JIT arguments.
        return lambda parameters: np.asarray(
            curve.evaluate(
                jnp.asarray(parameters[0]), budget=budget, preparations=preparations
            )
        )
    return lambda parameters: np.asarray(
        _evaluate_curves(curve, jnp.asarray(parameters[:1]))
    )[0]


@checked
def intersect_trim_curves(
    first: AbstractTrimCurve,
    second: AbstractTrimCurve,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> CurveIntersectionResult:
    """Return bounded trim/trim events in the operand parameter intervals."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                first,
                second,
            )
        ) as budget:
            return intersect_trim_curves(
                first, second, policy=policy, preparations=preparations, budget=budget
            )
    return intersect_curve_ranges(
        intersection_curve_owner.CurveRange(first),
        intersection_curve_owner.CurveRange(second),
        policy=policy,
        preparations=preparations,
        coordinate_budget=budget,
    )


def _exact_unit_axes(axes: Sequence[np.ndarray]) -> bool:

    vectors = [rational_vector(axis) for axis in axes]
    return all(
        rational_dot(vectors[i], vectors[j]) == (1 if i == j else 0)
        for i in range(len(vectors))
        for j in range(len(vectors))
    )


def _analytic_point(
    parameters: np.ndarray,
    point: np.ndarray,
    gap: float,
    kind: ParametricIntersectionKind = "tangent_point",
) -> ParametricIntersectionPoint:
    radius = 32 * _UNIT * (1 + np.abs(parameters))
    return ParametricIntersectionPoint(
        kind=kind,
        certificate="analytic-contact",
        parameters=parameters,
        parameter_lower=np.nextafter(parameters - radius, -np.inf),
        parameter_upper=np.nextafter(parameters + radius, np.inf),
        point=point,
        gap_bound=gap,
        condition_estimate=math.inf,
        branch_count=0,
    )


def _analytic_line_circle(
    first: intersection_curve_owner.CurveRange,
    second: intersection_curve_owner.CurveRange,
) -> CurveIntersectionResult | None:

    match first.curve, second.curve:
        case LineCurve(), CircleCurve():
            line_range, circle_range, reversed_ = first, second, False
        case CircleCurve(), LineCurve():
            line_range, circle_range, reversed_ = second, first, True
        case _:
            return None
    line, circle = line_range.curve, circle_range.curve
    if not isinstance(line, LineCurve) or not isinstance(circle, CircleCurve):
        return None
    if (
        line.ambient_dimension != 2
        or not circle_range.periodic
        or not _exact_unit_axes(
            (np.asarray(circle.first_axis), np.asarray(circle.second_axis))
        )
    ):
        return None
    origin, center, direction = [
        [F(float(value)) for value in np.asarray(vector)]
        for vector in (line.origin, circle.center, line.direction)
    ]
    normal = (-direction[1], direction[0])
    norm_squared = sum(value * value for value in normal)
    distance = sum(n * (c - o) for n, c, o in zip(normal, center, origin, strict=True))
    radius = F(float(circle.radius))
    if distance * distance != radius * radius * norm_squared:
        return None
    exact_point = [
        c - distance * n / norm_squared for c, n in zip(center, normal, strict=True)
    ]
    parameter = (
        sum((p - o) * d for p, o, d in zip(exact_point, origin, direction, strict=True))
        / norm_squared
    )
    budget = _Budget(1)
    budget.processed = 1
    if not F(line_range.first) <= parameter <= F(line_range.last):
        budget.excluded = 1
        return CurveIntersectionResult((), (), (), budget.work())
    local = [(p - c) / radius for p, c in zip(exact_point, center, strict=True)]
    a, b = (
        [F(float(value)) for value in np.asarray(axis)]
        for axis in (circle.first_axis, circle.second_axis)
    )
    angle = math.atan2(
        float(sum(x * y for x, y in zip(local, b, strict=True))),
        float(sum(x * y for x, y in zip(local, a, strict=True))),
    )
    while angle < circle_range.first:
        angle += 2 * math.pi
    parameters = np.asarray(
        (angle, float(parameter)) if reversed_ else (float(parameter), angle)
    )
    point = np.asarray([float(value) for value in exact_point])
    gap = 256 * _UNIT * (1 + np.linalg.norm(point) + float(circle.radius))
    budget.certified = 1
    return CurveIntersectionResult(
        (_analytic_point(parameters, point, gap),), (), (), budget.work()
    )


def _analytic_circle_circle(
    first: intersection_curve_owner.CurveRange,
    second: intersection_curve_owner.CurveRange,
    /,
) -> CurveIntersectionResult | None:
    """Prove a tangent contact of two full-period coplanar circles."""

    if (
        not isinstance(first.curve, CircleCurve)
        or not isinstance(second.curve, CircleCurve)
        or first.curve.ambient_dimension != 3
        or second.curve.ambient_dimension != 3
        or not first.periodic
        or not second.periodic
    ):
        return None
    a, b = first.curve, second.curve
    axes = tuple(
        np.asarray(axis)
        for curve in (a, b)
        for axis in (curve.first_axis, curve.second_axis)
    )
    if not _exact_unit_axes(axes[:2]) or not _exact_unit_axes(axes[2:]):
        return None
    exact_axes = tuple(rational_vector(axis) for axis in axes)
    normal = rational_cross(exact_axes[0], exact_axes[1])
    centers = tuple(rational_vector(np.asarray(curve.center)) for curve in (a, b))
    delta = tuple(y - x for x, y in zip(*centers, strict=True))
    if rational_dot(normal, delta) != 0 or any(
        rational_dot(normal, axis) != 0 for axis in exact_axes[2:]
    ):
        return None
    length_squared = rational_dot(delta, delta)
    radii = F(float(a.radius)), F(float(b.radius))
    if length_squared == 0 or length_squared not in (
        (radii[0] + radii[1]) ** 2,
        (radii[0] - radii[1]) ** 2,
    ):
        return None
    factor = (length_squared + radii[0] * radii[0] - radii[1] * radii[1]) / (
        2 * length_squared
    )
    exact_point = tuple(
        center + factor * offset for center, offset in zip(centers[0], delta, strict=True)
    )
    parameters = []
    for curve_range, curve, center, radius, curve_axes in (
        (first, a, centers[0], radii[0], exact_axes[:2]),
        (second, b, centers[1], radii[1], exact_axes[2:]),
    ):
        local = tuple(
            (point - origin) / radius
            for point, origin in zip(exact_point, center, strict=True)
        )
        coordinates = tuple(
            sum(x * y for x, y in zip(local, axis, strict=True)) for axis in curve_axes
        )
        angle = math.atan2(float(coordinates[1]), float(coordinates[0]))
        while angle < curve_range.first:
            angle += math.tau
        while angle > curve_range.last:
            angle -= math.tau
        if not curve_range.first <= angle <= curve_range.last:
            return None
        parameters.append(angle)
    point = np.asarray(tuple(float(value) for value in exact_point))
    gap = 256 * _UNIT * (1 + np.linalg.norm(point) + float(radii[0]) + float(radii[1]))
    budget = _Budget(1)
    budget.processed = budget.certified = 1
    return CurveIntersectionResult(
        (_analytic_point(np.asarray(parameters), point, gap),),
        (),
        (),
        budget.work(),
    )


def _analytic_sphere_contact(
    first: intersection_curve_owner.SurfaceRegion,
    second: intersection_curve_owner.SurfaceRegion,
) -> SurfaceIntersectionResult | None:

    a, b = first.patch, second.patch
    if not isinstance(a, SpherePatch) or not isinstance(b, SpherePatch):
        return None
    for region, patch in ((first, a), (second, b)):
        if (
            not region.periodic[0]
            or region.lower[1] > -math.pi / 2
            or region.upper[1] < math.pi / 2
        ):
            return None
        if not _exact_unit_axes(
            (
                np.asarray(patch.first_axis),
                np.asarray(patch.second_axis),
                np.asarray(patch.axis),
            )
        ):
            return None
    center_a, center_b = (
        [F(float(value)) for value in np.asarray(center)]
        for center in (a.center, b.center)
    )
    delta = [y - x for x, y in zip(center_a, center_b, strict=True)]
    length_squared = sum(value * value for value in delta)
    ra, rb = F(float(a.radius)), F(float(b.radius))
    if length_squared == 0 or length_squared not in ((ra + rb) ** 2, (ra - rb) ** 2):
        return None
    factor = (length_squared + ra * ra - rb * rb) / (2 * length_squared)
    exact_point = [x + factor * d for x, d in zip(center_a, delta, strict=True)]
    parameters = []
    for patch, center, radius in ((a, center_a, ra), (b, center_b, rb)):
        local = [(p - c) / radius for p, c in zip(exact_point, center, strict=True)]
        coordinates = [
            sum(x * F(float(y)) for x, y in zip(local, np.asarray(axis), strict=True))
            for axis in (patch.first_axis, patch.second_axis, patch.axis)
        ]
        longitude = math.atan2(float(coordinates[1]), float(coordinates[0])) % (
            2 * math.pi
        )
        latitude = math.atan2(
            float(coordinates[2]),
            math.hypot(float(coordinates[0]), float(coordinates[1])),
        )
        parameters.extend((longitude, latitude))
    point = np.asarray([float(value) for value in exact_point])
    gap = 256 * _UNIT * (1 + np.linalg.norm(point) + float(a.radius) + float(b.radius))
    budget = _Budget(1)
    budget.processed = budget.certified = 1
    event = _analytic_point(np.asarray(parameters), point, gap, "tangent_point")
    return SurfaceIntersectionResult((), (event,), (), (), budget.work())


def _analytic_plane_sphere_contact(
    first: intersection_curve_owner.SurfaceRegion,
    second: intersection_curve_owner.SurfaceRegion,
) -> SurfaceIntersectionResult | None:
    """Prove represented quadric support equality at a tangent contact."""

    match first.patch, second.patch:
        case PlanePatch(), SpherePatch():
            plane_region, sphere_region, reverse = first, second, False
        case SpherePatch(), PlanePatch():
            plane_region, sphere_region, reverse = second, first, True
        case _:
            return None
    plane, sphere = plane_region.patch, sphere_region.patch
    if not isinstance(plane, PlanePatch) or not isinstance(sphere, SpherePatch):
        return None
    if (
        not sphere_region.periodic[0]
        or sphere_region.lower[1] > -math.pi / 2
        or sphere_region.upper[1] < math.pi / 2
    ):
        return None
    a, b = [
        [F(float(value)) for value in np.asarray(axis)]
        for axis in (plane.first_axis, plane.second_axis)
    ]
    n = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    if sum(value * value for value in n) == 0:
        return None
    origin, center = [
        [F(float(value)) for value in np.asarray(point)]
        for point in (plane.origin, sphere.center)
    ]
    axes = [
        [F(float(value)) for value in np.asarray(axis)]
        for axis in (sphere.first_axis, sphere.second_axis, sphere.axis)
    ]
    radius = F(float(sphere.radius))
    d = sum(x * (y - z) for x, y, z in zip(n, center, origin, strict=True))
    weights = [radius * sum(x * y for x, y in zip(n, axis, strict=True)) for axis in axes]
    extent = sum(value * value for value in weights)
    if extent == 0 or d * d != extent:
        return None
    unit = [-d * value / extent for value in weights]
    point = [
        center[j] + radius * sum(axes[i][j] * unit[i] for i in range(3)) for j in range(3)
    ]
    delta = [x - y for x, y in zip(point, origin, strict=True)]
    aa, bb, ab = (
        sum(x * x for x in a),
        sum(x * x for x in b),
        sum(x * y for x, y in zip(a, b, strict=True)),
    )
    determinant = aa * bb - ab * ab
    da, db = (
        sum(x * y for x, y in zip(delta, a, strict=True)),
        sum(x * y for x, y in zip(delta, b, strict=True)),
    )
    uv = ((bb * da - ab * db) / determinant, (aa * db - ab * da) / determinant)
    work = _Budget(1)
    work.processed = 1
    if any(
        not F(plane_region.lower[i]) <= uv[i] <= F(plane_region.upper[i])
        for i in range(2)
    ):
        work.excluded = 1
        return SurfaceIntersectionResult((), (), (), (), work.work())
    longitude = math.atan2(float(unit[1]), float(unit[0])) % (2 * math.pi)
    latitude = math.atan2(float(unit[2]), math.hypot(float(unit[0]), float(unit[1])))
    plane_parameters, sphere_parameters = (
        [float(value) for value in uv],
        [longitude, latitude],
    )
    parameters = np.asarray(
        sphere_parameters + plane_parameters
        if reverse
        else plane_parameters + sphere_parameters
    )
    world = np.asarray([float(value) for value in point])
    gap = 512 * _UNIT * (1 + np.linalg.norm(world) + float(sphere.radius))
    event = _analytic_point(parameters, world, gap, "tangent_point")
    work.certified = 1
    return SurfaceIntersectionResult((), (event,), (), (), work.work())


@checked
def intersect_curve_ranges(
    first: intersection_curve_owner.CurveRange,
    second: intersection_curve_owner.CurveRange,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
    preparations: TrimRequestPreparations | None = None,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> CurveIntersectionResult:
    """Intersect curves of equal dimension (planar p-curves or space curves).

    Planar events are certified roots of ``c1(s) = c2(t)``. Space curves meet
    only non-generically, so their events are certified distance-critical points
    whose gap is at most the policy's tangency tolerance.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if coordinate_budget is None:
        with coordinate_source_request(
            (
                first,
                second,
            )
        ) as coordinate_budget:
            return intersect_curve_ranges(
                first,
                second,
                policy=policy,
                preparations=preparations,
                coordinate_budget=coordinate_budget,
            )
    policy_ = ParametricIntersectionPolicy() if policy is None else policy
    analytic = _analytic_line_circle(first, second)
    if analytic is not None:
        return analytic
    analytic = _analytic_circle_circle(first, second)
    if analytic is not None:
        return analytic
    ambient = first.ambient_dimension
    if second.ambient_dimension != ambient or ambient not in (2, 3):
        raise ValueError(
            "intersect_curve_ranges.operands must share dimension two or three; got incompatible dimensions"
        )
    first_pieces = intersection_curve_owner.curve_pieces(first, budget=coordinate_budget)
    second_pieces = intersection_curve_owner.curve_pieces(
        second, budget=coordinate_budget
    )
    boxes = [
        np.asarray(
            first.curve.bounding_box(
                first.first,
                first.last,
                preparations=preparations,
                budget=coordinate_budget,
            )
        ),
        np.asarray(
            second.curve.bounding_box(
                second.first,
                second.last,
                preparations=preparations,
                budget=coordinate_budget,
            )
        ),
    ]
    scales = _scales(policy_, _scale(boxes))
    budget = _Budget(policy_.maximum_boxes)
    points: list[ParametricIntersectionPoint] = []
    contacts: list[ParametricIntersectionPoint] = []
    cells: list[tuple[np.ndarray, np.ndarray, bool]] = []
    unresolved: list[UnresolvedParameterRegion] = []
    bound = 0.0
    for piece_a in first_pieces:
        for piece_b in second_pieces:
            pair = prepare(
                _CurvePairSystem(piece_a.evaluator, piece_b.evaluator),
                2,
                policy_.batch,
                preparations=preparations,
            )
            point_map = _curve_point_map(
                piece_a.evaluator, budget=coordinate_budget, preparations=preparations
            )
            if ambient == 2:
                problem = _Problem(
                    pair,
                    point_map,
                    pair,
                    _Layout((0,), (1,), 2),
                    prepare(
                        _PlanarCurveTangencySystem(piece_a.evaluator, piece_b.evaluator),
                        2,
                        policy_.batch,
                        preparations=preparations,
                    ),
                    False,
                )
            else:
                problem = _Problem(
                    prepare(
                        _CurveProximitySystem(piece_a.evaluator, piece_b.evaluator),
                        2,
                        policy_.batch,
                        preparations=preparations,
                    ),
                    point_map,
                    pair,
                    None,
                    None,
                    True,
                )
            isolated = _isolate(
                problem,
                np.asarray((piece_a.lower, piece_b.lower)),
                np.asarray((piece_a.upper, piece_b.upper)),
                scales,
                budget,
                policy_.batch,
                coordinate_budget=coordinate_budget,
                preparations=preparations,
            )
            points.extend(isolated.roots)
            contacts.extend(isolated.contacts)
            cells.extend(isolated.coincident)
            unresolved.extend(isolated.unresolved)
            bound = max(bound, isolated.coincidence_bound)
    periods = (
        first.last - first.first if first.periodic else None,
        second.last - second.first if second.periodic else None,
    )
    return CurveIntersectionResult(
        _dedupe_points(points + contacts, periods),
        _components(cells, bound),
        unresolved,
        budget.work(),
    )


@checked
def intersect_curve_region(
    curve: intersection_curve_owner.CurveRange,
    surface: intersection_curve_owner.SurfaceRegion,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
    preparations: TrimRequestPreparations | None = None,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> CurveIntersectionResult:
    """Intersect a space curve with a surface region."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if coordinate_budget is None:
        with coordinate_source_request(
            (
                curve,
                surface,
            )
        ) as coordinate_budget:
            return intersect_curve_region(
                curve,
                surface,
                policy=policy,
                preparations=preparations,
                coordinate_budget=coordinate_budget,
            )
    policy_ = ParametricIntersectionPolicy() if policy is None else policy
    if curve.ambient_dimension != 3:
        raise ValueError(
            "intersect_curve_region.curve must have dimension three; got another dimension"
        )
    boxes = [
        np.asarray(
            curve.curve.bounding_box(
                curve.first,
                curve.last,
                preparations=preparations,
                budget=coordinate_budget,
            )
        ),
        np.asarray(
            surface.patch.bounding_box(
                surface.parameter_box, preparations=preparations, budget=coordinate_budget
            )
        ),
    ]
    scales = _scales(policy_, _scale(boxes))
    budget = _Budget(policy_.maximum_boxes)
    points: list[ParametricIntersectionPoint] = []
    cells: list[tuple[np.ndarray, np.ndarray, bool]] = []
    unresolved: list[UnresolvedParameterRegion] = []
    bound = 0.0
    for piece_a in intersection_curve_owner.curve_pieces(curve, budget=coordinate_budget):
        for piece_b in intersection_curve_owner.surface_pieces(
            surface, budget=coordinate_budget
        ):
            pair = prepare(
                CurveSurfaceSystem(piece_a.evaluator, piece_b.evaluator),
                3,
                policy_.batch,
                preparations=preparations,
            )
            point_map = _curve_point_map(
                piece_a.evaluator, budget=coordinate_budget, preparations=preparations
            )
            problem = _Problem(
                pair,
                point_map,
                pair,
                _Layout((0,), (1, 2), 3),
                prepare(
                    _CurveSurfaceTangencySystem(piece_a.evaluator, piece_b.evaluator),
                    3,
                    policy_.batch,
                    preparations=preparations,
                ),
                False,
            )
            isolated = _isolate(
                problem,
                np.asarray((piece_a.lower, *piece_b.lower)),
                np.asarray((piece_a.upper, *piece_b.upper)),
                scales,
                budget,
                policy_.batch,
                coordinate_budget=coordinate_budget,
                preparations=preparations,
            )
            points.extend(isolated.roots + isolated.contacts)
            cells.extend(isolated.coincident)
            unresolved.extend(isolated.unresolved)
            bound = max(bound, isolated.coincidence_bound)
    periods = (
        curve.last - curve.first if curve.periodic else None,
        *(
            surface.upper[axis] - surface.lower[axis] if surface.periodic[axis] else None
            for axis in range(2)
        ),
    )
    return CurveIntersectionResult(
        _dedupe_points(points, periods),
        _components(cells, bound),
        unresolved,
        budget.work(),
    )


# ---------------------------------------------------------- surface route


@dataclass
class _Start:
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    face: tuple[int, int] | None
    junction: int | None = None
    center: np.ndarray | None = None
    consumed: bool = False


class _Chart(StrictModule):
    axis: int = eqx.field(static=True)
    start: np.ndarray
    end: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    preconditioner: np.ndarray
    contraction: float = eqx.field(static=True)
    certified: bool = eqx.field(static=True)


type _Endpoint = tuple[
    intersection_curve_owner.IntersectionEndpointKind, tuple[int, int] | None, np.ndarray
]


@dataclass
class _Segment:
    pair_index: int
    charts: list[_Chart]
    start: _Endpoint
    end: _Endpoint
    closed: bool = False


@dataclass(frozen=True)
class _PairContext:
    index: int
    pair: Prepared
    lower: np.ndarray
    upper: np.ndarray
    pieces: tuple[int, int]

    @property
    def system(self) -> intersection_curve_owner.SurfacePairSystem:
        system = self.pair.system
        if not isinstance(system, intersection_curve_owner.SurfacePairSystem):
            raise TypeError(
                "surface_continuation.system must be a surface pair; got another system"
            )
        return system


def _certify_chart(
    context: _PairContext,
    start: np.ndarray,
    end: np.ndarray,
    scale: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> _Chart:
    """Parametric Krawczyk chart for the branch arc between two nodes."""
    jacobian = context.pair.point_jacobians(
        start[None], budget=budget, preparations=preparations
    )[0]
    speed = np.linalg.norm(jacobian, axis=0)
    axis = int(np.argmax(np.abs(end - start) * speed))
    free = [index for index in range(4) if index != axis]
    delta = np.abs(end - start)
    for margin in (0.5, 2.0, 8.0, 16.0, 32.0):
        pad = margin * np.max(delta[free]) + 1.0e-3 * np.abs(end[axis] - start[axis])
        pad = pad + 256.0 * _UNIT * (1.0 + np.max(np.abs(start)))
        lower = np.minimum(start, end)
        upper = np.maximum(start, end)
        lower[free] -= pad
        upper[free] += pad
        axis_pad = 1.0e-3 * abs(end[axis] - start[axis]) + 256.0 * _UNIT * (
            1 + max(abs(start[axis]), abs(end[axis]))
        )
        lower[axis] -= axis_pad
        upper[axis] += axis_pad
        if isinstance(context.system.first, BernsteinSurfacePiece):
            lower[:2] = np.maximum(lower[:2], context.lower[:2])
            upper[:2] = np.minimum(upper[:2], context.upper[:2])
        if isinstance(context.system.second, BernsteinSurfacePiece):
            lower[2:] = np.maximum(lower[2:], context.lower[2:])
            upper[2:] = np.minimum(upper[2:], context.upper[2:])
        result = krawczyk(
            context.pair,
            lower[None],
            upper[None],
            parameter_axis=axis,
            budget=budget,
            preparations=preparations,
        )
        if bool(result.certified[0]):
            return _Chart(
                axis,
                start,
                end,
                lower,
                upper,
                result.preconditioner[0],
                float(result.contraction[0]),
                True,
            )
    del scale
    return _Chart(axis, start, end, lower, upper, np.eye(3, dtype=np.float64), 1.0, False)


def _surface_discovery(
    context: _PairContext,
    scales: _Scales,
    budget: _Budget,
    batch: int,
    point_map: Callable[[np.ndarray], np.ndarray],
    junction_radius: np.ndarray,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[
    list[_Start], list[_Start], list[_Start], list[ParametricIntersectionPoint], _Isolated
]:
    """Certified boundary, turning and junction starts and contacts of one pair."""
    boundary: list[_Start] = []
    collected = _Isolated()
    pair_system = context.system
    layout = _Layout((0, 1), (2, 3), 3)
    pair_problem = _Problem(context.pair, point_map, context.pair, layout, None, False)
    cells, bound = _coincident_cells(
        pair_problem,
        context,
        scales,
        budget,
        batch,
        coordinate_budget=coordinate_budget,
        preparations=preparations,
    )
    collected.coincident.extend(cells)
    collected.coincidence_bound = bound
    cell_lower = np.asarray([cell[0] for cell in cells]).reshape((-1, 4))
    cell_upper = np.asarray([cell[1] for cell in cells]).reshape((-1, 4))
    for axis in range(4):
        for side, value in enumerate((context.lower[axis], context.upper[axis])):
            face = _FaceSystem(pair_system, axis, jnp.asarray(value))
            free = [index for index in range(4) if index != axis]
            prepared = prepare(face, 3, batch, preparations=preparations)

            def full(x: np.ndarray, axis: int = axis, value: float = value) -> np.ndarray:
                return np.insert(x, axis, value)

            touching = (cell_lower[:, axis] <= value) & (value <= cell_upper[:, axis])
            problem = _Problem(
                prepared,
                lambda x, full=full: point_map(full(x)),
                prepared,
                _Layout(
                    tuple(i for i, index in enumerate(free) if index < 2),
                    tuple(i for i, index in enumerate(free) if index >= 2),
                    3,
                ),
                None,
                False,
                excluded=(cell_lower[touching][:, free], cell_upper[touching][:, free]),
            )
            isolated = _isolate(
                problem,
                context.lower[free],
                context.upper[free],
                scales,
                budget,
                batch,
                coordinate_budget=coordinate_budget,
                preparations=preparations,
            )
            boundary.extend(
                _Start(
                    full(root.parameters),
                    full(root.parameter_lower),
                    full(root.parameter_upper),
                    (axis, side),
                )
                for root in isolated.roots
            )
            # A region boundary curve lying in the other surface is a
            # coincident component only where the surfaces are tangent. Where
            # rank-3 transversality is certified, it is a segment of the 1-D
            # intersection branch, traced from its boundary-system starts.
            for low, high, partial in isolated.coincident:
                cell = full(low), full(high)
                if _certified_transversal(
                    context.pair,
                    cell[0],
                    cell[1],
                    budget=coordinate_budget,
                    preparations=preparations,
                ):
                    continue
                collected.coincident.append((cell[0], cell[1], partial))
                collected.coincidence_bound = max(
                    collected.coincidence_bound, isolated.coincidence_bound
                )
            collected.unresolved.extend(
                UnresolvedParameterRegion(
                    full(region.lower), full(region.upper), region.reason
                )
                for region in isolated.unresolved
            )
    widths = context.upper - context.lower
    problem = _Problem(
        prepare(
            _TurningSystem(pair_system, jnp.asarray(_TURNING_DIRECTION / widths)),
            4,
            batch,
            preparations=preparations,
        ),
        point_map,
        context.pair,
        layout,
        prepare(_SurfaceTangencySystem(pair_system), 4, batch, preparations=preparations),
        False,
        excluded=(cell_lower, cell_upper),
        junction=junction_radius,
    )
    isolated = _isolate(
        problem,
        context.lower,
        context.upper,
        scales,
        budget,
        batch,
        coordinate_budget=coordinate_budget,
        preparations=preparations,
    )
    interior = [
        _Start(root.parameters, root.parameter_lower, root.parameter_upper, None)
        for root in isolated.roots
    ]
    collected.coincident.extend(isolated.coincident)
    collected.coincidence_bound = max(
        collected.coincidence_bound, isolated.coincidence_bound
    )
    collected.unresolved.extend(isolated.unresolved)
    junctions: list[_Start] = []
    for index, contact in enumerate(isolated.contacts):
        found, unresolved = _junction_starts(
            context,
            contact.parameters,
            junction_radius,
            scales,
            budget,
            batch,
            point_map,
            coordinate_budget=coordinate_budget,
            preparations=preparations,
        )
        junctions.extend(
            _Start(
                root.parameters,
                root.parameter_lower,
                root.parameter_upper,
                None,
                junction=index,
                center=contact.parameters,
            )
            for root in found
        )
        collected.unresolved.extend(unresolved)
        certificate = contact_indicatrix(
            context.system.first,
            context.system.second,
            contact.parameter_lower,
            contact.parameter_upper,
            budget=coordinate_budget,
        )
        if (
            certificate.kind != "unresolved"
            and len(found) == certificate.branches
            and not unresolved
        ):
            isolated.contacts[index] = ParametricIntersectionPoint(
                kind=certificate.kind,
                certificate=contact.certificate,
                parameters=contact.parameters,
                parameter_lower=contact.parameter_lower,
                parameter_upper=contact.parameter_upper,
                point=contact.point,
                gap_bound=contact.gap_bound,
                condition_estimate=contact.condition_estimate,
                branch_count=certificate.branches,
            )
            continue
        # Auxiliary critical-point uniqueness does not prove absence of a tiny
        # closed component wholly inside the removed junction neighborhood.
        collected.unresolved.append(
            UnresolvedParameterRegion(
                np.maximum(contact.parameters - junction_radius, context.lower),
                np.minimum(contact.parameters + junction_radius, context.upper),
                "singular",
            )
        )
        budget.unresolved += 1
    return boundary, interior, junctions, isolated.contacts, collected


class _JunctionSystem(StrictModule):
    """Branch points on the scaled sphere bounding a junction neighborhood."""

    pair: intersection_curve_owner.SurfacePairSystem
    center: Array
    inverse_radius: Array

    def residual(self, parameters: Array, /) -> Array:
        offset = (parameters - self.center) * self.inverse_radius
        return jnp.concatenate(
            (self.pair.residual(parameters), (jnp.dot(offset, offset) - 1.0)[None])
        )


def _junction_starts(
    context: _PairContext,
    center: np.ndarray,
    radius: np.ndarray,
    scales: _Scales,
    budget: _Budget,
    batch: int,
    point_map: Callable[[np.ndarray], np.ndarray],
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[list[ParametricIntersectionPoint], list[UnresolvedParameterRegion]]:
    """Isolate every branch leaving a certified contact's neighborhood."""
    system = _JunctionSystem(
        context.system, jnp.asarray(center), jnp.asarray(1.0 / radius)
    )
    prepared = prepare(system, 4, batch, preparations=preparations)
    problem = _Problem(prepared, point_map, context.pair, None, None, False)
    lower = np.maximum(center - 1.25 * radius, context.lower)
    upper = np.minimum(center + 1.25 * radius, context.upper)
    isolated = _isolate(
        problem,
        lower,
        upper,
        scales,
        budget,
        batch,
        coordinate_budget=coordinate_budget,
        preparations=preparations,
    )
    return isolated.roots, isolated.unresolved


@eqx.filter_jit
def _foot_points(
    pair: intersection_curve_owner.SurfacePairSystem,
    anchors: Array,
    plan: VectorLocalRootPlan,
) -> tuple[Array, Array]:
    """Closest-point candidates on the second surface of first-surface points."""

    def one(anchor: Array) -> tuple[Array, Array]:
        def residual(parameters: Array) -> Array:
            gap = pair.residual(jnp.concatenate((anchor[:2], parameters)))
            return jax.jacfwd(pair.second.evaluate)(parameters).T @ gap

        root, diagnostics = plan.solve_with_diagnostics(residual, anchor[2:])
        return root, diagnostics.converged

    return jax.vmap(one)(anchors)


def _coincident_cells(
    problem: _Problem,
    context: _PairContext,
    scales: _Scales,
    budget: _Budget,
    batch: int,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, bool]], float]:
    """Cells of the first surface lying on the second within the tolerance.

    Cells are ``(P, full second box)``; subdivision continues only where the
    surfaces are tangent and within tolerance at the cell's foot point, down to
    the probe scale.
    """
    plan = VectorLocalRootPlan(
        2, maximum_steps=40, tolerance=1.0e-14, plan_id="intersection-foot-point"
    )
    second_lower = context.lower[2:]
    second_upper = context.upper[2:]
    queue = [(context.lower[:2][None], context.upper[:2][None])]
    cells: list[tuple[np.ndarray, np.ndarray, bool]] = []
    bound = 0.0
    while queue and not budget.exhausted:
        first_lower = np.concatenate([item[0] for item in queue])
        first_upper = np.concatenate([item[1] for item in queue])
        queue = []
        if budget.processed + first_lower.shape[0] > budget.maximum:
            budget.exhausted = True
            break
        budget.processed += first_lower.shape[0]
        count = first_lower.shape[0]
        center = 0.5 * (first_lower + first_upper)
        seeds = np.concatenate(
            (center, np.broadcast_to(0.5 * (second_lower + second_upper), (count, 2))),
            axis=1,
        )
        feet, converged = _padded_call(
            lambda chunk: _foot_points(context.system, jnp.asarray(chunk), plan),
            seeds,
            batch,
        )
        feet = np.clip(
            np.where(converged[:, None], feet, seeds[:, 2:]), second_lower, second_upper
        )
        lower = np.concatenate(
            (first_lower, np.broadcast_to(second_lower, (count, 2))), 1
        )
        upper = np.concatenate(
            (first_upper, np.broadcast_to(second_upper, (count, 2))), 1
        )
        anchor = np.concatenate((center, feet), axis=1)
        full, partial, distance = _coincidence(
            problem,
            lower,
            upper,
            scales.coincidence,
            anchor=anchor,
            budget=coordinate_budget,
            preparations=preparations,
        )
        jacobian_lower, jacobian_upper = evaluate_interval(
            problem.system.jacobian,
            lower,
            upper,
            budget=coordinate_budget,
            preparations=preparations,
        )
        size = _image_size(
            jacobian_lower[:, :, :2],
            jacobian_upper[:, :, :2],
            upper[:, :2] - lower[:, :2],
        )
        gap = np.linalg.norm(problem.system.point_values(anchor), axis=1)
        candidate = (gap <= scales.coincidence) & ~(full | partial)
        partial &= size <= scales.probe
        for row in np.flatnonzero(full | partial):
            cells.append((lower[row], upper[row], bool(partial[row])))
            bound = max(bound, float(distance[row]))
            budget.coincident += 1
        split = candidate & (size > 0.25 * scales.probe)
        if not np.any(split):
            continue
        rows = np.flatnonzero(split)
        widths = first_upper[rows] - first_lower[rows]
        axis = np.argmax(widths, axis=1)
        middle = 0.5 * (first_lower[rows, axis] + first_upper[rows, axis])
        left_upper = first_upper[rows].copy()
        left_upper[np.arange(rows.shape[0]), axis] = middle
        right_lower = first_lower[rows].copy()
        right_lower[np.arange(rows.shape[0]), axis] = middle
        queue.append((first_lower[rows], left_upper))
        queue.append((right_lower, first_upper[rows]))
    return cells, bound


def _pair_coordinate_bounds(context: _PairContext, /) -> np.ndarray:
    """Intersect chart walls with rational support equations."""
    pair = context.system
    regions = (
        intersection_curve_owner.SurfaceRegion(
            pair.first, np.stack((context.lower[:2], context.upper[:2]))
        ),
        intersection_curve_owner.SurfaceRegion(
            pair.second, np.stack((context.lower[2:], context.upper[2:]))
        ),
    )
    return np.concatenate(
        [
            _joint_coordinate_image(region, regions, region.parameter_box)
            for region in regions
        ],
        axis=1,
    )


def _outside(context: _PairContext, point: np.ndarray, /) -> np.ndarray:
    return (point < context.lower) | (point > context.upper)


def _nearest_start(
    starts: list[_Start], target: np.ndarray, widths: np.ndarray, limit: float, /
) -> int | None:
    best: int | None = None
    best_distance = limit
    for index, start in enumerate(starts):
        if start.consumed:
            continue
        distance = float(np.max(np.abs(start.point - target) / widths))
        if distance <= best_distance:
            best, best_distance = index, distance
    return best


def _consume(starts: list[_Start], chart: _Chart, /) -> None:
    if not chart.certified:
        return
    for start in starts:
        if not start.consumed and np.all(
            (start.point >= chart.lower) & (start.point <= chart.upper)
        ):
            start.consumed = True


@dataclass
class _Starts:
    boundary: list[_Start]
    interior: list[_Start]
    junctions: list[_Start]
    radius: np.ndarray


def _entered_junction(starts: _Starts, point: np.ndarray, /) -> bool:
    return any(
        float(np.sum(((point - start.center) / starts.radius) ** 2)) <= 1.0
        for start in starts.junctions
        if start.center is not None
    )


def _land(
    context: _PairContext,
    point: np.ndarray,
    candidates: list[_Start],
    target: np.ndarray,
    span: float,
    scale: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[_Chart, _Start] | None:
    """Certify a final chart onto the start point nearest ``target``."""
    widths = context.upper - context.lower
    index = _nearest_start(candidates, target, widths, 1.5 * span + 1.0e-9)
    if index is None:
        return None
    chart = _certify_chart(
        context,
        point,
        candidates[index].point,
        scale,
        budget=budget,
        preparations=preparations,
    )
    if not chart.certified:
        return None
    candidates[index].consumed = True
    return chart, candidates[index]


def _march(
    context: _PairContext,
    origin: np.ndarray,
    direction: np.ndarray,
    starts: _Starts,
    closable: bool,
    policy: ParametricIntersectionPolicy,
    scale: float,
    budget: _Budget,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[list[_Chart], _Endpoint]:
    """Certified predictor/corrector continuation from one start point.

    Every accepted step is a certified chart. The march ends on a certified
    boundary start, on a junction start of a singular contact, on its own
    origin (closed loop), or explicitly uncertified when the step control
    underflows.
    """
    plan = VectorLocalRootPlan(
        4, maximum_steps=30, tolerance=1.0e-14, plan_id="intersection-march"
    )
    pair = context.system
    widths = context.upper - context.lower
    coordinate_bounds = _pair_coordinate_bounds(context)
    fixed = coordinate_bounds[0] == coordinate_bounds[1]
    maximum_step = policy.relative_step * scale
    minimum_step = policy.relative_minimum_step * scale
    step = maximum_step
    point = origin
    tangent = direction
    charts: list[_Chart] = []
    for _ in range(policy.maximum_march_steps):
        budget.march_steps += 1
        root, following, _, converged = (
            np.asarray(value)
            for value in _march_step(
                pair, jnp.asarray(point), jnp.asarray(tangent), jnp.asarray(step), plan
            )
        )
        # A represented support equation, not numerical wall slack, owns
        # constant coordinates (for example a cylinder's cap height).
        root = np.where(fixed, coordinate_bounds[0], root)
        accepted = (
            bool(converged)
            and bool(np.all(np.isfinite(root)))
            and float(np.dot(following, tangent)) >= math.cos(policy.maximum_turn)
        )
        span = float(np.max(np.abs(root - point) / widths)) if accepted else 0.0
        if accepted and closable and len(charts) >= 2:
            near_origin = float(np.max(np.abs(origin - point) / widths)) <= 1.5 * span
            if near_origin and float(np.dot(origin - point, tangent)) > 0.0:
                chart = _certify_chart(
                    context,
                    point,
                    origin,
                    scale,
                    budget=coordinate_budget,
                    preparations=preparations,
                )
                if chart.certified:
                    charts.append(chart)
                    return charts, ("closed", None, origin)
        if accepted and _entered_junction(starts, root):
            landed = _land(
                context,
                point,
                starts.junctions,
                root,
                span,
                scale,
                budget=coordinate_budget,
                preparations=preparations,
            )
            if landed is not None:
                charts.append(landed[0])
                _consume(starts.interior, landed[0])
                return charts, ("singular", None, landed[1].point)
            accepted = False
        if accepted and np.any(_outside(context, root)):
            landed = _land(
                context,
                point,
                starts.boundary,
                root,
                span,
                scale,
                budget=coordinate_budget,
                preparations=preparations,
            )
            if landed is not None:
                charts.append(landed[0])
                _consume(starts.interior, landed[0])
                # Duplicate starts of the same branch end (several boundary
                # systems meet at a corner) lie in the same certified chart.
                _consume(starts.boundary, landed[0])
                return charts, ("boundary", landed[1].face, landed[1].point)
            accepted = False
        if accepted:
            chart = _certify_chart(
                context,
                point,
                root,
                scale,
                budget=coordinate_budget,
                preparations=preparations,
            )
            accepted = chart.certified
        if not accepted:
            step *= 0.5
            if step < minimum_step:
                return charts, ("uncertified", None, point)
            continue
        charts.append(chart)
        _consume(starts.interior, chart)
        point, tangent = root, following
        step = min(1.5 * step, maximum_step)
    return charts, ("uncertified", None, point)


def _trace(
    context: _PairContext,
    starts: _Starts,
    policy: ParametricIntersectionPolicy,
    scale: float,
    budget: _Budget,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> list[_Segment]:
    """March every unconsumed start: boundary, then junction, then turning points."""
    if not (starts.boundary or starts.interior or starts.junctions):
        return []
    segments: list[_Segment] = []
    reference = np.asarray((0.0, 0.0, 0.0, 1.0)) + _TURNING_DIRECTION * 1.0e-3
    pair = context.system
    coordinate_bounds = _pair_coordinate_bounds(context)
    fixed = coordinate_bounds[0] == coordinate_bounds[1]
    for start in (*starts.boundary, *starts.interior, *starts.junctions):
        start.point = np.where(fixed, coordinate_bounds[0], start.point)

    def tangent_at(point: np.ndarray) -> np.ndarray:
        return np.asarray(
            _branch_tangent(pair, jnp.asarray(point), jnp.asarray(reference))
        )

    for start in starts.boundary:
        if start.consumed or start.face is None:
            continue
        start.consumed = True
        tangent = tangent_at(start.point)
        axis, side = start.face
        if abs(tangent[axis]) <= 1.0e-12:
            continue
        tangent = tangent * np.sign(tangent[axis] * (1.0 if side == 0 else -1.0))
        charts, end = _march(
            context,
            start.point,
            tangent,
            starts,
            False,
            policy,
            scale,
            budget,
            coordinate_budget=coordinate_budget,
            preparations=preparations,
        )
        if charts:
            # Padding may contain an endpoint before the march reaches its
            # boundary. Consume only the origin's duplicates here; the landing
            # chart owns consumption of the opposite endpoint.
            _consume(starts.boundary, charts[0])
        segments.append(
            _Segment(context.index, charts, ("boundary", start.face, start.point), end)
        )
    for start in starts.junctions:
        if start.consumed or start.center is None:
            continue
        start.consumed = True
        tangent = tangent_at(start.point)
        outward = float(np.dot(tangent, (start.point - start.center) / starts.radius))
        charts, end = _march(
            context,
            start.point,
            tangent * np.sign(outward),
            starts,
            False,
            policy,
            scale,
            budget,
            coordinate_budget=coordinate_budget,
            preparations=preparations,
        )
        segments.append(
            _Segment(context.index, charts, ("singular", None, start.point), end)
        )
    for start in starts.interior:
        if start.consumed:
            continue
        start.consumed = True
        tangent = tangent_at(start.point)
        forward, forward_end = _march(
            context,
            start.point,
            tangent,
            starts,
            True,
            policy,
            scale,
            budget,
            coordinate_budget=coordinate_budget,
            preparations=preparations,
        )
        if forward_end[0] == "closed":
            segments.append(
                _Segment(context.index, forward, forward_end, forward_end, closed=True)
            )
            continue
        backward, backward_end = _march(
            context,
            start.point,
            -tangent,
            starts,
            False,
            policy,
            scale,
            budget,
            coordinate_budget=coordinate_budget,
            preparations=preparations,
        )
        segments.append(
            _Segment(
                context.index,
                _reverse_charts(backward) + forward,
                backward_end,
                forward_end,
            )
        )
    for segment in segments:
        for chart in segment.charts:
            if chart.certified:
                budget.charts_certified += 1
            else:
                budget.charts_uncertified += 1
    return segments


type _EndpointKey = tuple[int, int]


class _EndpointBox(StrictModule):
    key: _EndpointKey = eqx.field(static=True)
    lower: np.ndarray
    upper: np.ndarray
    partner_lower: np.ndarray
    partner_upper: np.ndarray


class _BranchLinks(StrictModule):
    pairs: tuple[tuple[_EndpointKey, _EndpointKey], ...] = eqx.field(static=True)
    unresolved: tuple[UnresolvedParameterRegion, ...]


class _BranchTraversal(StrictModule):
    steps: tuple[tuple[int, bool], ...] = eqx.field(static=True)
    closed: bool = eqx.field(static=True)


class _BranchCharts(StrictModule):
    charts: tuple[_Chart, ...]
    pieces: np.ndarray
    transitions: np.ndarray
    node_axes: np.ndarray
    node_values: np.ndarray
    closed: bool = eqx.field(static=True)
    start_kind: intersection_curve_owner.IntersectionEndpointKind = eqx.field(static=True)
    end_kind: intersection_curve_owner.IntersectionEndpointKind = eqx.field(static=True)


class _BranchAssembly(StrictModule):
    curves: tuple[intersection_curve_owner.IntersectionCurve, ...]
    unresolved: tuple[UnresolvedParameterRegion, ...]


def _segment_endpoint(segment: _Segment, side: int, /) -> _Endpoint:
    return segment.start if side == 0 else segment.end


def _endpoint_boxes(
    segments: list[_Segment],
    contexts: list[_PairContext],
    first: intersection_curve_owner.SurfaceRegion,
    second: intersection_curve_owner.SurfaceRegion,
    /,
) -> tuple[list[_EndpointBox], list[UnresolvedParameterRegion]]:
    region_lower = np.asarray((*first.lower, *second.lower))
    region_upper = np.asarray((*first.upper, *second.upper))
    periodic = (*first.periodic, *second.periodic)
    boxes: list[_EndpointBox] = []
    unresolved: list[UnresolvedParameterRegion] = []
    for index, segment in enumerate(segments):
        if segment.closed:
            continue
        context = contexts[segment.pair_index]
        for side in (0, 1):
            kind, face, _ = _segment_endpoint(segment, side)
            if kind != "boundary" or face is None:
                continue
            axis, face_side = face
            value = (context.lower if face_side == 0 else context.upper)[axis]
            exterior = value == region_lower[axis] or value == region_upper[axis]
            if exterior and not periodic[axis]:
                continue
            if not segment.charts:
                unresolved.append(
                    UnresolvedParameterRegion(
                        context.lower, context.upper, "continuation"
                    )
                )
                continue
            chart = segment.charts[0 if side == 0 else -1]
            if not chart.certified or not chart.lower[axis] <= value <= chart.upper[axis]:
                unresolved.append(
                    UnresolvedParameterRegion(chart.lower, chart.upper, "continuation")
                )
                continue
            # Slice the certified chart at its represented boundary atom.
            # No coordinate-distance threshold can identify another branch.
            lower, upper = chart.lower.copy(), chart.upper.copy()
            lower[axis] = upper[axis] = value
            partner_lower, partner_upper = lower.copy(), upper.copy()
            if exterior:
                shift = np.zeros(4, dtype=np.float64)
                shift[axis] = (
                    region_upper[axis] - region_lower[axis]
                    if value == region_lower[axis]
                    else region_lower[axis] - region_upper[axis]
                )
                shifted_lower, shifted_upper = interval_add(
                    (lower, upper), (shift, shift)
                )
                partner_lower = np.asarray(shifted_lower, dtype=np.float64)
                partner_upper = np.asarray(shifted_upper, dtype=np.float64)
                # The seam atom is represented by the opposite domain boundary.
                partner_lower[axis] = partner_upper[axis] = (
                    region_upper[axis] if face_side == 0 else region_lower[axis]
                )
            boxes.append(
                _EndpointBox(
                    key=(index, side),
                    lower=lower,
                    upper=upper,
                    partner_lower=partner_lower,
                    partner_upper=partner_upper,
                )
            )
    return boxes, unresolved


def _link_segments(
    segments: list[_Segment],
    contexts: list[_PairContext],
    first: intersection_curve_owner.SurfaceRegion,
    second: intersection_curve_owner.SurfaceRegion,
    /,
) -> _BranchLinks:
    boxes, unresolved = _endpoint_boxes(segments, contexts, first, second)
    candidates = {
        box.key: tuple(
            other.key
            for other in boxes
            if other.key != box.key
            and np.all(box.partner_lower <= other.upper)
            and np.all(other.lower <= box.partner_upper)
        )
        for box in boxes
    }
    pairs: list[tuple[_EndpointKey, _EndpointKey]] = []
    for box in boxes:
        partners = candidates[box.key]
        if len(partners) == 1 and candidates[partners[0]] == (box.key,):
            if box.key < partners[0]:
                pairs.append((box.key, partners[0]))
        else:
            unresolved.append(
                UnresolvedParameterRegion(box.lower, box.upper, "continuation")
            )
    return _BranchLinks(pairs=tuple(pairs), unresolved=tuple(unresolved))


def _traverse_segments(
    segments: list[_Segment], phase: _BranchLinks, /
) -> tuple[_BranchTraversal, ...]:
    links = {key: partner for pair in phase.pairs for key, partner in (pair, pair[::-1])}
    visited: set[int] = set()
    heads = [
        index
        for index, segment in enumerate(segments)
        if segment.closed or (index, 0) not in links or (index, 1) not in links
    ]
    heads += list(range(len(segments)))
    traversals: list[_BranchTraversal] = []
    for head in heads:
        if head in visited:
            continue
        closed = segments[head].closed
        forward = (head, 0) not in links or (head, 1) in links
        traversal: list[tuple[int, bool]] = []
        current: tuple[int, bool] | None = (head, forward)
        while current is not None:
            index, direction = current
            visited.add(index)
            traversal.append(current)
            if segments[index].closed:
                break
            link = links.get((index, 1 if direction else 0))
            if link is None:
                current = None
            elif link[0] == head:
                closed = True
                current = None
            elif link[0] in visited:
                current = None
            else:
                current = (link[0], link[1] == 0)
        traversals.append(_BranchTraversal(steps=tuple(traversal), closed=closed))
    return tuple(traversals)


def _accumulate_branch(
    traversal: _BranchTraversal,
    segments: list[_Segment],
    contexts: list[_PairContext],
    /,
) -> _BranchCharts | None:
    charts: list[_Chart] = []
    chart_pieces: list[tuple[int, int]] = []
    transitions: list[np.ndarray] = []
    node_axes: list[int] = []
    node_values: list[float] = []
    for index, direction in traversal.steps:
        segment = segments[index]
        ordered = segment.charts if direction else _reverse_charts(segment.charts)
        pieces = contexts[segment.pair_index].pieces
        entry = segment.start if direction else segment.end
        exit_ = segment.end if direction else segment.start
        if not charts and ordered:
            axis = (
                entry[1][0]
                if entry[0] == "boundary" and entry[1] is not None
                else ordered[0].axis
            )
            node_axes.append(axis)
            node_values.append(float(ordered[0].start[axis]))
        if charts and ordered:
            transitions[-1] = ordered[0].start - charts[-1].end
        for position, chart in enumerate(ordered):
            charts.append(chart)
            chart_pieces.append(pieces)
            transitions.append(np.zeros((4,), dtype=np.float64))
            axis = chart.axis
            if (
                position == len(ordered) - 1
                and exit_[0] == "boundary"
                and exit_[1] is not None
            ):
                axis = exit_[1][0]
            node_axes.append(axis)
            node_values.append(float(chart.end[axis]))
    if not charts:
        return None
    if traversal.closed and not segments[traversal.steps[0][0]].closed:
        transitions[-1] = charts[0].start - charts[-1].end
    first_index, first_direction = traversal.steps[0]
    last_index, last_direction = traversal.steps[-1]
    return _BranchCharts(
        charts=tuple(charts),
        pieces=np.asarray(chart_pieces, dtype=np.int32),
        transitions=np.stack(transitions),
        node_axes=np.asarray(node_axes, dtype=np.int32),
        node_values=np.asarray(node_values, dtype=np.float64),
        closed=traversal.closed,
        start_kind=(
            "closed"
            if traversal.closed
            else _segment_endpoint(segments[first_index], 0 if first_direction else 1)[0]
        ),
        end_kind=(
            "closed"
            if traversal.closed
            else _segment_endpoint(segments[last_index], 1 if last_direction else 0)[0]
        ),
    )


def _build_branch(
    phase: _BranchCharts,
    first: intersection_curve_owner.SurfaceRegion,
    second: intersection_curve_owner.SurfaceRegion,
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> intersection_curve_owner.IntersectionCurve:
    charts = phase.charts
    return intersection_curve_owner.IntersectionCurve(
        first,
        second,
        chart_axes=np.asarray([chart.axis for chart in charts], dtype=np.int32),
        chart_start=np.stack([chart.start for chart in charts]),
        chart_end=np.stack([chart.end for chart in charts]),
        chart_pieces=phase.pieces,
        box_lower=np.stack([chart.lower for chart in charts]),
        box_upper=np.stack([chart.upper for chart in charts]),
        preconditioners=np.stack([chart.preconditioner for chart in charts]),
        contraction=np.asarray([chart.contraction for chart in charts]),
        certified=np.asarray([chart.certified for chart in charts]),
        transition_shifts=phase.transitions,
        closed=phase.closed,
        start_kind=phase.start_kind,
        end_kind=phase.end_kind,
        node_axes=phase.node_axes,
        node_values=phase.node_values,
        budget=budget,
        preparations=preparations,
    )


def _glue(
    segments: list[_Segment],
    contexts: list[_PairContext],
    first: intersection_curve_owner.SurfaceRegion,
    second: intersection_curve_owner.SurfaceRegion,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> _BranchAssembly:
    """Assemble branches only across uniquely overlapping certified chart slices."""
    links = _link_segments(segments, contexts, first, second)
    curves: list[intersection_curve_owner.IntersectionCurve] = []
    for traversal in _traverse_segments(segments, links):
        phase = _accumulate_branch(traversal, segments, contexts)
        if phase is not None:
            curves.append(
                _build_branch(
                    phase, first, second, preparations=preparations, budget=budget
                )
            )
    return _BranchAssembly(curves=tuple(curves), unresolved=links.unresolved)


def _reverse_charts(charts: list[_Chart], /) -> list[_Chart]:
    return [
        _Chart(
            chart.axis,
            chart.end,
            chart.start,
            chart.lower,
            chart.upper,
            chart.preconditioner,
            chart.contraction,
            chart.certified,
        )
        for chart in reversed(charts)
    ]


@eqx.filter_jit
def _evaluate_surface(evaluator: AbstractSurfacePatch, parameters: Array) -> Array:
    return jax.vmap(evaluator.evaluate)(parameters)


class _SurfacePairPhase(StrictModule):
    segments: tuple[_Segment, ...]
    contacts: tuple[ParametricIntersectionPoint, ...]
    cells: tuple[tuple[np.ndarray, np.ndarray, bool], ...]
    unresolved: tuple[UnresolvedParameterRegion, ...]
    coincidence_bound: float = eqx.field(static=True)


def _execute_surface_pair(
    context: _PairContext,
    system: intersection_curve_owner.SurfacePairSystem,
    first_piece: SurfacePiece[AbstractSurfacePatch],
    policy: ParametricIntersectionPolicy,
    scales: _Scales,
    scale: float,
    budget: _Budget,
    /,
    *,
    preparations: TrimRequestPreparations,
    coordinate_budget: CoordinateEnclosureBudget,
) -> _SurfacePairPhase | None:
    cancelled = _pose_cancelled_pair(system)
    pose = None if cancelled is None else cancelled[1]
    value_lower, value_upper = evaluate_interval(
        context.pair.value,
        context.lower[None],
        context.upper[None],
        budget=coordinate_budget,
        preparations=preparations,
    )
    budget.processed += 1
    if not bool(_contains_zero(value_lower, value_upper)[0]):
        budget.excluded += 1
        return None
    point_map = _surface_point_map(first_piece)
    radius = policy.relative_junction * (context.upper - context.lower)
    boundary, interior, junctions, pair_contacts, collected = _surface_discovery(
        context,
        scales if pose is None else _source_frame_scales(scales, pose),
        budget,
        policy.batch,
        point_map,
        radius,
        coordinate_budget=coordinate_budget,
        preparations=preparations,
    )
    if pose is not None:
        pair_contacts = [_world_gap_contact(contact, pose) for contact in pair_contacts]
        collected.coincidence_bound = _world_gap(pose, collected.coincidence_bound)
    segments = _trace(
        context,
        _Starts(boundary, interior, junctions, radius),
        policy,
        scale,
        budget,
        coordinate_budget=coordinate_budget,
        preparations=preparations,
    )
    for segment in segments:
        if segment.start[0] == "uncertified" or segment.end[0] == "uncertified":
            collected.unresolved.append(
                UnresolvedParameterRegion(context.lower, context.upper, "continuation")
            )
            budget.unresolved += 1
    branched = {start.junction for start in junctions}
    return _SurfacePairPhase(
        segments=tuple(segments),
        contacts=tuple(
            _junction_contact(contact, radius) if index in branched else contact
            for index, contact in enumerate(pair_contacts)
        ),
        cells=tuple(collected.coincident),
        unresolved=tuple(collected.unresolved),
        coincidence_bound=collected.coincidence_bound,
    )


@checked
def intersect_surface_regions(
    first: intersection_curve_owner.SurfaceRegion,
    second: intersection_curve_owner.SurfaceRegion,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
    preparations: TrimRequestPreparations | None = None,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> SurfaceIntersectionResult:
    """Intersect surface regions into certified branches and contacts."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if coordinate_budget is None:
        with coordinate_source_request(
            (
                first,
                second,
            )
        ) as coordinate_budget:
            return intersect_surface_regions(
                first,
                second,
                policy=policy,
                preparations=preparations,
                coordinate_budget=coordinate_budget,
            )
    policy_ = ParametricIntersectionPolicy() if policy is None else policy
    analytic = _analytic_sphere_contact(first, second)
    if analytic is not None:
        return analytic
    analytic = _analytic_plane_sphere_contact(first, second)
    if analytic is not None:
        return analytic
    return _execute_surface_intersection(
        first,
        second,
        policy_,
        preparations=preparations,
        coordinate_budget=coordinate_budget,
    )


def _execute_surface_intersection(
    first: intersection_curve_owner.SurfaceRegion,
    second: intersection_curve_owner.SurfaceRegion,
    policy_: ParametricIntersectionPolicy,
    /,
    *,
    preparations: TrimRequestPreparations,
    coordinate_budget: CoordinateEnclosureBudget,
) -> SurfaceIntersectionResult:
    """Execute certified subdivision and continuation for a surface pair."""
    boxes = [
        np.asarray(
            first.patch.bounding_box(
                first.parameter_box, preparations=preparations, budget=coordinate_budget
            )
        ),
        np.asarray(
            second.patch.bounding_box(
                second.parameter_box, preparations=preparations, budget=coordinate_budget
            )
        ),
    ]
    scale = _scale(boxes)
    scales = _scales(policy_, scale)
    budget = _Budget(policy_.maximum_boxes)
    contexts: list[_PairContext] = []
    segments: list[_Segment] = []
    contacts: list[ParametricIntersectionPoint] = []
    cells: list[tuple[np.ndarray, np.ndarray, bool]] = []
    unresolved: list[UnresolvedParameterRegion] = []
    bound = 0.0
    for piece_a in intersection_curve_owner.surface_pieces(
        first, budget=coordinate_budget
    ):
        for piece_b in intersection_curve_owner.surface_pieces(
            second, budget=coordinate_budget
        ):
            pair_system = intersection_curve_owner.SurfacePairSystem(
                piece_a.evaluator, piece_b.evaluator
            )
            # Preparation preserves the source pose identity used by execution.
            context = _PairContext(
                len(contexts),
                prepare(pair_system, 4, policy_.batch, preparations=preparations),
                np.concatenate((piece_a.lower, piece_b.lower)),
                np.concatenate((piece_a.upper, piece_b.upper)),
                (piece_a.index, piece_b.index),
            )
            contexts.append(context)
            phase = _execute_surface_pair(
                context,
                pair_system,
                piece_a,
                policy_,
                scales,
                scale,
                budget,
                preparations=preparations,
                coordinate_budget=coordinate_budget,
            )
            if phase is None:
                continue
            cells.extend(phase.cells)
            bound = max(bound, phase.coincidence_bound)
            unresolved.extend(phase.unresolved)
            segments.extend(phase.segments)
            contacts.extend(phase.contacts)
    assembly = _glue(
        segments,
        contexts,
        first,
        second,
        preparations=preparations,
        budget=coordinate_budget,
    )
    return _finish_surface_intersection(
        assembly,
        contacts,
        cells,
        unresolved,
        bound,
        budget,
        first,
        second,
        coordinate_budget=coordinate_budget,
    )


def _finish_surface_intersection(
    assembly: _BranchAssembly,
    contacts: list[ParametricIntersectionPoint],
    cells: list[tuple[np.ndarray, np.ndarray, bool]],
    unresolved: list[UnresolvedParameterRegion],
    bound: float,
    budget: _Budget,
    first: intersection_curve_owner.SurfaceRegion,
    second: intersection_curve_owner.SurfaceRegion,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> SurfaceIntersectionResult:
    # A unique certified graph covers undecided leaves within its chart.
    charts = [
        (curve.box_lower[chart], curve.box_upper[chart])
        for curve in assembly.curves
        for chart in range(curve.num_charts)
        if curve.certified[chart]
    ]
    unresolved = [
        region
        for region in unresolved
        if not any(
            np.all(region.lower >= lower) and np.all(region.upper <= upper)
            for lower, upper in charts
        )
    ]
    # Assembly ambiguity is not settled by coverage of either candidate chart.
    unresolved.extend(assembly.unresolved)
    budget.unresolved += len(assembly.unresolved)
    periods = tuple(
        region.upper[axis] - region.lower[axis] if region.periodic[axis] else None
        for region in (first, second)
        for axis in range(2)
    )
    contacts = _dedupe_points(contacts, periods)
    contacts, unresolved = _certify_published_junctions(
        assembly.curves,
        contacts,
        unresolved,
        first,
        second,
        periods,
        coordinate_budget=coordinate_budget,
    )
    return SurfaceIntersectionResult(
        assembly.curves,
        contacts,
        _components(cells, bound),
        unresolved,
        budget.work(),
    )


def _junction_box_contains(
    contact: ParametricIntersectionPoint,
    lower: np.ndarray,
    upper: np.ndarray,
    periods: tuple[float | None, ...],
) -> bool:
    center = (lower + upper) / 2
    shift = np.zeros(4, dtype=np.float64)
    for axis, period in enumerate(periods):
        if period is not None:
            shift[axis] = (
                np.rint((contact.parameters[axis] - center[axis]) / period) * period
            )
    return bool(
        np.all(lower + shift >= contact.parameter_lower)
        and np.all(upper + shift <= contact.parameter_upper)
    )


def _published_junction_counts(
    curves: tuple[intersection_curve_owner.IntersectionCurve, ...],
    contacts: list[ParametricIntersectionPoint],
    periods: tuple[float | None, ...],
) -> list[int]:
    counts = [0] * len(contacts)
    for curve in curves:
        for node, kind in ((0, curve.start_kind), (-1, curve.end_kind)):
            if kind != "singular" or not curve.fully_certified:
                continue
            center = (curve.node_lower[node] + curve.node_upper[node]) / 2
            candidates = [
                index
                for index, contact in enumerate(contacts)
                if _junction_box_contains(contact, center, center, periods)
            ]
            if len(candidates) == 1:
                counts[candidates[0]] += 1
    return counts


def _certify_published_junctions(
    curves: tuple[intersection_curve_owner.IntersectionCurve, ...],
    contacts: list[ParametricIntersectionPoint],
    unresolved: list[UnresolvedParameterRegion],
    first: intersection_curve_owner.SurfaceRegion,
    second: intersection_curve_owner.SurfaceRegion,
    periods: tuple[float | None, ...],
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[list[ParametricIntersectionPoint], list[UnresolvedParameterRegion]]:
    """Reconcile half-branches across source-piece and periodic seam cuts."""
    counts = _published_junction_counts(curves, contacts, periods)
    certified_boxes = []
    for index, contact in enumerate(contacts):
        if contact.kind != "singular_point" or contact.branch_count >= 0:
            continue
        certificate = contact_indicatrix(
            first.patch,
            second.patch,
            contact.parameter_lower,
            contact.parameter_upper,
            budget=coordinate_budget,
        )
        if certificate.kind == "unresolved" or counts[index] != certificate.branches:
            continue
        contacts[index] = ParametricIntersectionPoint(
            kind=certificate.kind,
            certificate=contact.certificate,
            parameters=contact.parameters,
            parameter_lower=contact.parameter_lower,
            parameter_upper=contact.parameter_upper,
            point=contact.point,
            gap_bound=contact.gap_bound,
            condition_estimate=contact.condition_estimate,
            branch_count=certificate.branches,
        )
        certified_boxes.append(contact)
    retained = [
        region
        for region in unresolved
        if region.reason != "singular"
        or not any(
            _junction_box_contains(contact, region.lower, region.upper, periods)
            for contact in certified_boxes
        )
    ]
    return contacts, retained


def _certified_transversal(
    pair: Prepared,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> bool:
    """Outward interval proof that the 3x4 pair Jacobian has rank 3 on a cell.

    Some 3x3 minor whose interval determinant excludes zero certifies that the
    two surface normals are independent throughout the cell.
    """
    jacobian_lower, jacobian_upper = evaluate_interval(
        pair.jacobian, lower[None], upper[None], budget=budget, preparations=preparations
    )
    low, high = np.asarray(jacobian_lower)[0], np.asarray(jacobian_upper)[0]
    if low.shape != (3, 4):
        return False

    def entry(row: int, column: int) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(low[row, column]), np.asarray(high[row, column])

    def minor(first: int, second: int) -> tuple[np.ndarray, np.ndarray]:
        # Rows 1 and 2 of columns ``first`` and ``second``.
        bounds = interval_subtract(
            interval_multiply(entry(1, first), entry(2, second)),
            interval_multiply(entry(1, second), entry(2, first)),
        )
        return (
            np.asarray(bounds[0], dtype=np.float64),
            np.asarray(bounds[1], dtype=np.float64),
        )

    for a, b, c in combinations(range(4), 3):
        determinant = interval_add(
            interval_subtract(
                interval_multiply(entry(0, a), minor(b, c)),
                interval_multiply(entry(0, b), minor(a, c)),
            ),
            interval_multiply(entry(0, c), minor(a, b)),
        )
        if float(determinant[0]) > 0.0 or float(determinant[1]) < 0.0:
            return True
    return False


def _world_gap(rotation: np.ndarray, gap: float, /) -> float:
    """Bound the world residual of a pose-cancelled source-frame residual."""

    return source_correspondence_owner.placed_deviation_bound(rotation, gap)


def _world_gap_contact(
    contact: ParametricIntersectionPoint, rotation: np.ndarray, /
) -> ParametricIntersectionPoint:
    return ParametricIntersectionPoint(
        kind=contact.kind,
        certificate=contact.certificate,
        parameters=contact.parameters,
        parameter_lower=contact.parameter_lower,
        parameter_upper=contact.parameter_upper,
        point=contact.point,
        gap_bound=_world_gap(rotation, contact.gap_bound),
        condition_estimate=contact.condition_estimate,
        branch_count=contact.branch_count,
    )


def _junction_contact(
    contact: ParametricIntersectionPoint, radius: np.ndarray, /
) -> ParametricIntersectionPoint:
    """A contact where branches meet; its enclosure is the junction neighborhood."""
    return ParametricIntersectionPoint(
        kind="singular_point" if contact.branch_count == -1 else contact.kind,
        certificate=contact.certificate,
        parameters=contact.parameters,
        parameter_lower=np.minimum(contact.parameter_lower, contact.parameters - radius),
        parameter_upper=np.maximum(contact.parameter_upper, contact.parameters + radius),
        point=contact.point,
        gap_bound=contact.gap_bound,
        condition_estimate=contact.condition_estimate,
        branch_count=contact.branch_count,
    )


def _surface_point_map(
    piece: SurfacePiece[AbstractSurfacePatch], /
) -> Callable[[np.ndarray], np.ndarray]:
    evaluator = piece.evaluator

    def evaluate(parameters: np.ndarray) -> np.ndarray:
        return np.asarray(
            _evaluate_surface(evaluator, jnp.asarray(parameters[None, :2]))
        )[0]

    return evaluate


__all__ = [
    "CoincidentParameterRegion",
    "CurveIntersectionResult",
    "ParametricIntersectionCertificate",
    "ParametricIntersectionKind",
    "ParametricIntersectionPoint",
    "ParametricIntersectionPolicy",
    "ParametricIntersectionWork",
    "SurfaceIntersectionResult",
    "UnresolvedIntersectionReason",
    "UnresolvedParameterRegion",
    "intersect_curve_ranges",
    "intersect_curve_region",
    "intersect_surface_regions",
    "intersect_trim_curves",
    "TrimIntersectionRoot",
    "TrimRootEndpoint",
    "BranchRootEndpoint",
    "NativePeriodEndpoint",
    "RootEndpoint",
    "TripleSurfaceIntersectionRoot",
    "IntersectionCurvePointRoot",
    "CurveSurfaceIntersectionRoot",
]
