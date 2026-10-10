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
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import combinations
from typing import Literal, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    inverse_small_linear,
    LinearSolvePolicy,
    LinearSystem,
    orthonormal_frame,
    SmallLinearSolvePlan,
    solve,
    solve_small_linear,
)
from ...nonlinear import VectorLocalRootPlan
from ...typing import ConvertibleToArray, Dim, HostBool, HostFloat64, parse, Scope
from .._atlas import AbstractTrimCurve
from .._interval_enclosure import (
    interval_add,
    interval_divide,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from ._intersection_curve import (
    _PeriodOffset,
    _trim_source_jet,
    BernsteinSurfacePiece,
    coefficient_enclosures,
    curve_pieces,
    CurveEvaluator,
    CurveRange,
    CurveTrimSegment,
    encode_geometry,
    IntersectionCurve,
    IntersectionEndpointKind,
    IntersectionPCurve,
    surface_pieces,
    surface_pieces_for_box,
    SurfaceEvaluator,
    SurfacePairSystem,
    SurfacePiece,
    SurfaceRegion,
)
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    CircleCurve,
    CylinderPatch,
    ExtrusionSurface,
    LineCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
)
from ._placed import PlacedCurve, PlacedSurface, same_source_pose


ParametricIntersectionKind: TypeAlias = Literal["transversal", "tangent", "singular"]
ParametricIntersectionCertificate: TypeAlias = Literal[
    "krawczyk", "tangency-krawczyk", "proximity-krawczyk", "analytic-contact"
]
UnresolvedIntersectionReason: TypeAlias = Literal[
    "budget", "resolution", "singular", "continuation"
]


class _SourceCurveSurfaceLift(Protocol):
    @property
    def curve(self) -> AbstractCurve | IntersectionCurve: ...

    @property
    def pcurve(self) -> CurveEvaluator: ...

    @property
    def surface(self) -> SurfaceRegion: ...

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
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
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
                raise ValueError(f"{name} must be finite and positive.")
        if not relative_resolution < relative_probe:
            raise ValueError("relative_resolution must be smaller than relative_probe.")
        if not relative_minimum_step < relative_step:
            raise ValueError("relative_minimum_step must be smaller than relative_step.")
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
    """Authoritative UV junction: source curves plus a unique interval root box.

    Numerical parameters/points are bounded realizations, not exact floating
    constructions. Foot-point/tangency residuals are not exact junctions.
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

    def __init__(
        self,
        first: AbstractTrimCurve,
        second: AbstractTrimCurve,
        /,
        *,
        parameter_lower: np.ndarray,
        parameter_upper: np.ndarray,
    ) -> None:
        if not isinstance(first, AbstractTrimCurve) or not isinstance(
            second, AbstractTrimCurve
        ):
            raise TypeError("Trim roots require exact native trim-curve definitions.")
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
            raise ValueError("A trim root requires a finite nonempty parameter box.")
        prepared = _prepare(_CurvePairSystem(first, second), 2, 1)
        inclusion = _krawczyk(prepared, lower[None], upper[None])
        for factor in (32.0, 256.0, 2048.0):
            if bool(inclusion.certified[0]):
                break
            radius = factor * _UNIT * (1 + np.maximum(np.abs(lower), np.abs(upper)))
            lower = np.nextafter(lower - radius, -np.inf)
            upper = np.nextafter(upper + radius, np.inf)
            inclusion = _krawczyk(prepared, lower[None], upper[None])
        if not bool(inclusion.certified[0]):
            raise ValueError(
                "The trim junction has no regular unique source root certificate."
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

    def parameter_enclosure(self, /, *, maximum_steps: int = 16) -> np.ndarray:
        if type(maximum_steps) is not int or maximum_steps < 0:
            raise ValueError("Refinement steps must be nonnegative integers.")
        return _trim_root_parameter_enclosure(self, maximum_steps)

    def point_enclosure(self, /, *, maximum_steps: int = 16) -> np.ndarray:
        parameters = self.parameter_enclosure(maximum_steps=maximum_steps)
        first = np.stack(
            _trim_source_jet(
                self.first, float(parameters[0, 0]), float(parameters[1, 0]), 0
            )
        )
        second = np.stack(
            _trim_source_jet(
                self.second, float(parameters[0, 1]), float(parameters[1, 1]), 0
            )
        )
        lower, upper = np.maximum(first[0], second[0]), np.minimum(first[1], second[1])
        if np.any(lower > upper):
            raise ValueError("The certified junction lost its source point enclosure.")
        return np.stack((lower, upper))

    def evaluate(self) -> tuple[np.ndarray, float, bool]:
        point = np.asarray(self.first.evaluate(jnp.asarray(self.parameters[0])))
        box = self.point_enclosure()
        bound = float(np.max(np.nextafter(np.abs(box - point), np.inf)))
        return point, bound, True


_TRIM_ROOT_PREPARATIONS: ContextVar[
    dict[
        int,
        tuple[TrimIntersectionRoot, _Prepared | None, dict[int, np.ndarray]],
    ]
    | None
] = ContextVar("original_trim_root_preparations", default=None)


@contextmanager
def _original_trim_root_preparation_scope(
    curves: tuple[AbstractTrimCurve, ...], /
) -> Iterator[None]:
    """Retain root programs and contractions for one immutable trim workset."""
    sources = jax.tree_util.tree_leaves(
        curves, is_leaf=lambda value: isinstance(value, TrimIntersectionRoot)
    )
    owners = {
        id(source): (source, None, {})
        for source in sources
        if isinstance(source, TrimIntersectionRoot)
    }
    active = _TRIM_ROOT_PREPARATIONS.get()
    if active is not None:
        for source in sources:
            if isinstance(source, TrimIntersectionRoot):
                active.setdefault(id(source), (source, None, {}))
        yield
        return
    token = _TRIM_ROOT_PREPARATIONS.set(owners)
    try:
        yield
    finally:
        _TRIM_ROOT_PREPARATIONS.reset(token)


def _trim_root_parameter_enclosure(
    root: TrimIntersectionRoot, maximum_steps: int, /
) -> np.ndarray:
    owners = _TRIM_ROOT_PREPARATIONS.get()
    entry = None if owners is None else owners.get(id(root))
    if entry is not None and entry[0] is root:
        cached = entry[2].get(maximum_steps)
        if cached is not None:
            return cached.copy()
        prepared = entry[1]
    else:
        prepared = None
    if prepared is None:
        prepared = _prepare(_CurvePairSystem(root.first, root.second), 2, 1)
        if owners is not None and entry is not None and entry[0] is root:
            entry = (root, prepared, entry[2])
            owners[id(root)] = entry
    lower, upper = _contract(
        prepared,
        root.parameter_lower[None],
        root.parameter_upper[None],
        maximum_steps=maximum_steps,
    )
    result = np.stack((lower[0], upper[0]))
    if entry is not None and entry[0] is root:
        entry[2][maximum_steps] = result
    return result.copy()


class CurveSurfaceIntersectionRoot(StrictModule):
    """A source-backed spatial junction on a curve and another surface."""

    __strict_contract__ = True

    curve: AbstractCurve | IntersectionCurve
    surface: SurfaceRegion
    parameter_lower: HostFloat64[Literal[3]]
    parameter_upper: HostFloat64[Literal[3]]
    parameters: HostFloat64[Literal[3]]
    preconditioner: HostFloat64[Literal[3], Literal[3]]
    contraction: float = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    def __init__(
        self,
        curve: AbstractCurve | IntersectionCurve,
        surface: SurfaceRegion,
        /,
        *,
        parameter_lower: np.ndarray,
        parameter_upper: np.ndarray,
    ) -> None:
        if (
            not isinstance(curve, (AbstractCurve, IntersectionCurve))
            or curve.ambient_dimension != 3
            or not isinstance(surface, SurfaceRegion)
        ):
            raise TypeError("Spatial roots require a native 3D curve and SurfaceRegion.")
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
            raise ValueError("A spatial root requires a finite nonempty parameter box.")
        if isinstance(curve, IntersectionCurve):
            evaluator = curve
        else:
            pieces = curve_pieces(CurveRange(curve, float(lower[0]), float(upper[0])))
            if len(pieces) != 1:
                raise ValueError(
                    "A spatial root witness must lie in one canonical source curve span."
                )
            evaluator = pieces[0].evaluator
        # The canonical span is that of the root's own source box: a root on a
        # region wall (shared with a neighboring face) belongs to the patch law.
        spans = surface_pieces_for_box(surface.patch, np.stack((lower[1:], upper[1:])))
        if len(spans) != 1:
            raise ValueError(
                "A spatial root witness must lie in one canonical source surface span."
            )
        prepared = _prepare(_CurveSurfaceSystem(evaluator, spans[0].evaluator), 3, 1)
        inclusion = _krawczyk(prepared, lower[None], upper[None])
        # Contracted event boxes can meet their outward rounding floor and no
        # longer pass a STRICT re-inclusion on restoration. Re-isolate in a
        # slightly larger source box; only a fresh inclusion admits the root.
        for factor in (32.0, 256.0, 2048.0):
            if bool(inclusion.certified[0]):
                break
            radius = factor * _UNIT * (1 + np.maximum(np.abs(lower), np.abs(upper)))
            lower = np.nextafter(lower - radius, -np.inf)
            upper = np.nextafter(upper + radius, np.inf)
            inclusion = _krawczyk(prepared, lower[None], upper[None])
        if not bool(inclusion.certified[0]):
            raise ValueError(
                "The spatial junction has no regular unique source root certificate."
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

    def parameter_enclosure(self, /, *, maximum_steps: int = 16) -> np.ndarray:
        prepared = _curve_surface_root_preparation(self)
        lower, upper = _contract(
            prepared,
            self.parameter_lower[None],
            self.parameter_upper[None],
            maximum_steps=maximum_steps,
        )
        return np.stack((lower[0], upper[0]))

    def point_enclosure(self, /, *, maximum_steps: int = 16) -> np.ndarray:
        parameters = self.parameter_enclosure(maximum_steps=maximum_steps)
        curve_box = np.asarray(
            self.curve.bounding_box(float(parameters[0, 0]), float(parameters[1, 0]))
        )
        surface_box = np.asarray(self.surface.patch.bounding_box(parameters[:, 1:]))
        lower, upper = (
            np.maximum(curve_box[0], surface_box[0]),
            np.minimum(curve_box[1], surface_box[1]),
        )
        if np.any(lower > upper):
            raise ValueError(
                "The certified spatial junction lost its source point enclosure."
            )
        return np.stack((lower, upper))

    def evaluate(self) -> tuple[np.ndarray, float, bool]:
        if isinstance(self.curve, IntersectionCurve):
            point = np.asarray(self.curve.evaluate(jnp.asarray(self.parameters[0])).point)
        else:
            point = np.asarray(self.curve.evaluate(jnp.asarray(self.parameters[0])))
        return (
            point,
            float(np.max(np.nextafter(np.abs(self.point_enclosure() - point), np.inf))),
            True,
        )


_CURVE_SURFACE_ROOT_PREPARATIONS: ContextVar[
    dict[int, tuple[CurveSurfaceIntersectionRoot, _Prepared | None]] | None
] = ContextVar("original_curve_surface_root_preparations", default=None)


@contextmanager
def original_curve_surface_root_preparation(
    endpoints: tuple[RootEndpoint | None, ...],
    /,
) -> Iterator[None]:
    """Borrow root programs only during one immutable original trim workset."""
    sources = jax.tree_util.tree_leaves(
        endpoints,
        is_leaf=lambda value: isinstance(value, CurveSurfaceIntersectionRoot),
    )
    owners = {
        id(source): (source, None)
        for source in sources
        if isinstance(source, CurveSurfaceIntersectionRoot)
    }
    token = _CURVE_SURFACE_ROOT_PREPARATIONS.set(owners)
    try:
        yield
    finally:
        _CURVE_SURFACE_ROOT_PREPARATIONS.reset(token)


def _curve_surface_root_preparation(root: CurveSurfaceIntersectionRoot, /) -> _Prepared:
    owners = _CURVE_SURFACE_ROOT_PREPARATIONS.get()
    entry = None if owners is None else owners.get(id(root))
    if entry is not None and entry[0] is root and entry[1] is not None:
        return entry[1]
    curve = root.curve
    if not isinstance(curve, IntersectionCurve):
        curve = curve_pieces(
            CurveRange(
                curve,
                float(root.parameter_lower[0]),
                float(root.parameter_upper[0]),
            )
        )[0].evaluator
    surface = surface_pieces_for_box(
        root.surface.patch,
        np.stack((root.parameter_lower[1:], root.parameter_upper[1:])),
    )[0].evaluator
    prepared = _prepare(_CurveSurfaceSystem(curve, surface), 3, 1)
    if owners is not None and entry is not None and entry[0] is root:
        owners[id(root)] = (root, prepared)
    return prepared


class IntersectionCurvePointRoot(StrictModule):
    """An exact source graph point at one fixed curve parameter atom."""

    __strict_contract__ = True

    curve: IntersectionCurve
    parameter: float = eqx.field(static=True)
    parameters: HostFloat64[Literal[1]]
    parameter_lower: HostFloat64[Literal[1]]
    parameter_upper: HostFloat64[Literal[1]]
    root_id: str = eqx.field(static=True)
    point_id: str = eqx.field(static=True)

    def __init__(self, curve: IntersectionCurve, parameter: float, /) -> None:
        if not isinstance(curve, IntersectionCurve) or not curve.fully_certified:
            raise ValueError("A fixed point root requires a certified shared-node atlas.")
        if not math.isfinite(parameter) or not 0 <= parameter <= curve.num_charts:
            raise ValueError("A fixed source parameter must lie inside its curve domain.")
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

    def parameter_enclosure(self, /, *, maximum_steps: int = 16) -> np.ndarray:
        if maximum_steps < 0:
            raise ValueError("Refinement steps must be nonnegative.")
        return np.stack((self.parameter_lower, self.parameter_upper))

    def point_enclosure(self) -> np.ndarray:
        return self.curve.bounding_box(self.parameter, self.parameter)

    def evaluate(self) -> tuple[np.ndarray, float, bool]:
        point = np.asarray(self.curve.evaluate(jnp.asarray(self.parameter)).point)
        return (
            point,
            float(np.max(np.nextafter(np.abs(self.point_enclosure() - point), np.inf))),
            True,
        )


class TripleSurfaceIntersectionRoot(StrictModule):
    """One regular joint source root of three surfaces, with six UV parameters."""

    __strict_contract__ = True

    first: SurfaceRegion
    second: SurfaceRegion
    third: SurfaceRegion
    parameter_lower: HostFloat64[Literal[6]]
    parameter_upper: HostFloat64[Literal[6]]
    parameters: HostFloat64[Literal[6]]
    preconditioner: HostFloat64[Literal[6], Literal[6]]
    contraction: float = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    def __init__(
        self,
        first: SurfaceRegion,
        second: SurfaceRegion,
        third: SurfaceRegion,
        /,
        *,
        parameter_lower: np.ndarray,
        parameter_upper: np.ndarray,
    ) -> None:
        if not all(
            isinstance(region, SurfaceRegion) for region in (first, second, third)
        ):
            raise TypeError(
                "Joint surface roots require three native SurfaceRegion values."
            )
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
            raise ValueError("A joint root requires a finite nonempty six-parameter box.")
        evaluators = []
        for index, region in enumerate((first, second, third)):
            lo, hi = lower[2 * index : 2 * index + 2], upper[2 * index : 2 * index + 2]
            pieces = [
                piece
                for piece in surface_pieces(region)
                if np.all(lo >= piece.lower) and np.all(hi <= piece.upper)
            ]
            if len(pieces) != 1:
                raise ValueError(
                    "Each joint root witness must lie in one canonical source span."
                )
            evaluators.append(pieces[0].evaluator)
        prepared = _prepare(_TripleSurfaceSystem(*evaluators), 6, 1)
        inclusion = _krawczyk(prepared, lower[None], upper[None])
        if not bool(inclusion.certified[0]):
            raise ValueError(
                "The six-variable source system has no unique regular root certificate."
            )
        roots, _, _, _ = prepared.newton(
            jnp.asarray((0.5 * (lower + upper))[None]),
            VectorLocalRootPlan(
                6, maximum_steps=40, tolerance=1e-13, plan_id="triple-surface-junction"
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

    def parameter_enclosure(self, /, *, maximum_steps: int = 16) -> np.ndarray:
        evaluators = []
        for index, region in enumerate((self.first, self.second, self.third)):
            lower = self.parameter_lower[2 * index : 2 * index + 2]
            upper = self.parameter_upper[2 * index : 2 * index + 2]
            evaluators.append(
                next(
                    piece.evaluator
                    for piece in surface_pieces(region)
                    if np.all(lower >= piece.lower) and np.all(upper <= piece.upper)
                )
            )
        prepared = _prepare(_TripleSurfaceSystem(*evaluators), 6, 1)
        lower, upper = _contract(
            prepared,
            self.parameter_lower[None],
            self.parameter_upper[None],
            maximum_steps=maximum_steps,
        )
        return np.stack((lower[0], upper[0]))

    def point_enclosure(self) -> np.ndarray:
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
                        )
                    )
                )
            )
        boxes_ = np.asarray(boxes)
        lower, upper = np.max(boxes_[:, 0], axis=0), np.min(boxes_[:, 1], axis=0)
        if np.any(lower > upper):
            raise ValueError("The joint root lost its common source point enclosure.")
        return np.stack((lower, upper))

    def evaluate(self) -> tuple[np.ndarray, float, bool]:
        point = np.asarray(self.first.patch.evaluate(jnp.asarray(self.parameters[:2])))
        return (
            point,
            float(np.max(np.nextafter(np.abs(self.point_enclosure() - point), np.inf))),
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
    ) -> bool:
        """Prove equation identity and full source-parameter image inclusion.

        Original edges use exact construction identities or supplied whole-edge
        lifts. Neither point-box overlap nor a small numerical gap is evidence.
        """
        box = _joint_root_parameter_image(
            root,
            (self.first, self.second, self.third),
            source_edge_lifts=source_edge_lifts,
            joint_box=np.stack((self.parameter_lower, self.parameter_upper)),
        )
        return box is not None and bool(
            np.all(box[0] >= self.parameter_lower)
            and np.all(box[1] <= self.parameter_upper)
        )


def _surface_definition_key(region: SurfaceRegion, /) -> str:
    return canonical_fingerprint(encode_geometry(region.patch))


def _same_geometry(first: object, second: object, /) -> bool:
    return canonical_fingerprint(encode_geometry(first)) == canonical_fingerprint(
        encode_geometry(second)
    )


def _analytic_isoline_curve(curve: SurfaceIsoparametricCurve, /) -> AbstractCurve | None:
    """Exact analytic specialization of a native isoline, with the SAME t atom."""
    from fractions import Fraction as F

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
    region: SurfaceRegion,
    /,
) -> tuple[AbstractCurve, SurfaceRegion] | None:
    """Source definitions of a placed edge and placed support under one pose.

    ``R c(t) + t0 == R s(p(t)) + t0`` iff ``c(t) == s(p(t))`` for the proved
    identical invertible authored pose, so lifts are decided on the sources.
    """
    patch = region.patch
    if (
        isinstance(curve, PlacedCurve)
        and isinstance(curve.definition, AbstractCurve)
        and isinstance(patch, PlacedSurface)
        and same_source_pose(curve, patch)
    ):
        return curve.definition, SurfaceRegion(patch.definition, region.parameter_box)
    return None


def _exact_source_lift(
    curve: AbstractCurve,
    pcurve: CurveEvaluator,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    """An exact whole-interval construction identity, never sampled residuals."""
    from fractions import Fraction

    from ._correspondence import _circle_correspondence, _plane_correspondence
    from ._intersection_curve import _affine_curve_coefficients, pcurve_periodic_source
    from ._patches import _closure_tangent_continuous, BSplineCurve, OffsetSurface

    if not isinstance(curve, AbstractCurve) or not isinstance(pcurve, AbstractCurve):
        return False
    common = _common_source_pose(curve, region)
    if common is not None:
        return _exact_source_lift(common[0], pcurve, common[1], first, last)
    curve.validate_query_range(first, last)
    pcurve.validate_query_range(first, last)
    source = pcurve_periodic_source(pcurve, region.patch)
    if source is None or not isinstance(source[0], AbstractCurve):
        return False
    pcurve = source[0]
    surface = region.patch
    coefficients = _affine_curve_coefficients(pcurve, np.zeros(2))
    if isinstance(curve, SurfaceIsoparametricCurve):
        if _same_geometry(curve.surface, surface) and _same_geometry(
            curve.p_curve(), pcurve
        ):
            return True
        # The isoline at the opposite end of an exactly closed swept profile
        # is the same physical curve. An offset's normals also agree there
        # only across an exact tangent-continuous (G1) closure.
        base = surface.base if isinstance(surface, OffsetSurface) else surface
        expected = _affine_curve_coefficients(curve.p_curve(), np.zeros(2))
        fixed, free = curve.fixed_axis, 1 - curve.fixed_axis
        if (
            _same_geometry(curve.surface, surface)
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
            } and _closure_tangent_continuous(
                base,
                fixed,
                lower,
                upper,
                tangent=isinstance(surface, OffsetSurface),
            ):
                return True
        analytic = _analytic_isoline_curve(curve)
        return analytic is not None and _exact_source_lift(
            analytic, pcurve, region, first, last
        )
    if coefficients is not None:
        constant, derivative = coefficients
        if isinstance(surface, ExtrusionSurface):
            if (
                constant == (Fraction(0), Fraction(0))
                and derivative == (Fraction(1), Fraction(0))
                and _same_geometry(curve, surface.curve)
            ):
                return True
        if isinstance(surface, RevolutionSurface):
            if (
                constant == (Fraction(0), Fraction(0))
                and derivative == (Fraction(0), Fraction(1))
                and _same_geometry(curve, surface.curve)
            ):
                return True
        if isinstance(surface, CylinderPatch) and constant[0] == 0 and derivative[0] == 0:
            actual = _affine_curve_coefficients(curve, np.zeros(3))
            if actual is not None:
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
                    derivative[1] * Fraction(float(value))
                    for value in np.asarray(surface.axis)
                )
                if actual == (origin, tangent):
                    return True
    if (
        isinstance(surface, SpherePatch)
        and isinstance(curve, CircleCurve)
        and coefficients is not None
    ):
        if (
            coefficients[0] == (Fraction(0), Fraction(0))
            and coefficients[1][0] == 0
            and abs(coefficients[1][1]) == 1
            and np.array_equal(np.asarray(curve.center), np.asarray(surface.center))
            and np.array_equal(np.asarray(curve.radius), np.asarray(surface.radius))
            and np.array_equal(
                np.asarray(curve.first_axis), np.asarray(surface.first_axis)
            )
            and np.array_equal(
                np.asarray(curve.second_axis),
                int(coefficients[1][1]) * np.asarray(surface.axis),
            )
        ):
            return True
    if _circle_correspondence(curve, pcurve, surface) == 0.0:
        return True
    return (
        isinstance(surface, PlanePatch)
        and _plane_correspondence(
            curve,
            pcurve,
            surface,
            first,
            last,
            np.zeros(3),
        )
        == 0.0
    )


def _native_edge_pcurve(
    curve: AbstractCurve, region: SurfaceRegion, /
) -> AbstractCurve | None:
    """Recover only exact native support constructions, retaining source phase."""
    from fractions import Fraction as F

    common = _common_source_pose(curve, region)
    if common is not None:
        return _native_edge_pcurve(common[0], common[1])
    patch = region.patch
    if isinstance(curve, SurfaceIsoparametricCurve) and _same_geometry(
        curve.surface, patch
    ):
        return curve.p_curve()
    if isinstance(curve, SurfaceIsoparametricCurve):
        analytic = _analytic_isoline_curve(curve)
        return None if analytic is None else _native_edge_pcurve(analytic, region)
    if isinstance(patch, ExtrusionSurface) and _same_geometry(curve, patch.curve):
        return LineCurve(np.zeros(2), np.asarray((1.0, 0.0)))
    if isinstance(patch, RevolutionSurface) and _same_geometry(curve, patch.curve):
        return LineCurve(np.zeros(2), np.asarray((0.0, 1.0)))
    if isinstance(curve, CircleCurve) and isinstance(patch, SpherePatch):
        for sign in (1.0, -1.0):
            candidate = LineCurve(np.zeros(2), np.asarray((0.0, sign)))
            if _exact_source_lift(curve, candidate, region, 0.0, 1.0):
                return candidate
    if isinstance(curve, LineCurve) and isinstance(patch, CylinderPatch):
        axis = tuple(F(float(value)) for value in np.asarray(patch.axis))
        nonzero = next((index for index, value in enumerate(axis) if value), None)
        if nonzero is not None:
            delta = tuple(
                F(float(a)) - F(float(b)) - F(float(patch.radius)) * F(float(c))
                for a, b, c in zip(
                    np.asarray(curve.origin),
                    np.asarray(patch.origin),
                    np.asarray(patch.first_axis),
                    strict=True,
                )
            )
            tangent = tuple(F(float(value)) for value in np.asarray(curve.direction))
            offset, scale = (
                delta[nonzero] / axis[nonzero],
                tangent[nonzero] / axis[nonzero],
            )
            if scale and F(float(offset)) == offset and F(float(scale)) == scale:
                candidate = LineCurve(
                    np.asarray((0.0, float(offset))), np.asarray((0.0, float(scale)))
                )
                if _exact_source_lift(curve, candidate, region, 0.0, 1.0):
                    return candidate
    if isinstance(curve, CircleCurve) and isinstance(patch, CylinderPatch):
        delta = [
            F(float(a)) - F(float(b))
            for a, b in zip(
                np.asarray(curve.center), np.asarray(patch.origin), strict=True
            )
        ]
        direction = [F(float(a)) for a in np.asarray(patch.axis)]
        axis = next((i for i, value in enumerate(direction) if value), None)
        if axis is not None:
            height = delta[axis] / direction[axis]
            if (
                all(a == height * b for a, b in zip(delta, direction, strict=True))
                and F(float(height)) == height
            ):
                candidate = LineCurve(
                    np.asarray((0.0, float(height))), np.asarray((1.0, 0.0))
                )
                if _exact_source_lift(curve, candidate, region, 0.0, 1.0):
                    return candidate
    if isinstance(patch, PlanePatch) and isinstance(curve, (LineCurve, CircleCurve)):
        axes = [
            [F(float(value)) for value in np.asarray(axis)]
            for axis in (patch.first_axis, patch.second_axis)
        ]
        gram = [
            [sum((a * b for a, b in zip(x, y, strict=True)), F(0)) for y in axes]
            for x in axes
        ]
        determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
        if not determinant:
            return None

        def coordinates(vector: np.ndarray, translated: bool) -> np.ndarray | None:
            values = [
                F(float(value)) - (F(float(origin)) if translated else F(0))
                for value, origin in zip(vector, np.asarray(patch.origin), strict=True)
            ]
            dots = [
                sum((a * b for a, b in zip(axis, values, strict=True)), F(0))
                for axis in axes
            ]
            uv = (
                (dots[0] * gram[1][1] - dots[1] * gram[0][1]) / determinant,
                (dots[1] * gram[0][0] - dots[0] * gram[1][0]) / determinant,
            )
            if any(F(float(value)) != value for value in uv):
                return None
            return np.asarray(tuple(float(value) for value in uv))

        origin = coordinates(
            np.asarray(curve.origin if isinstance(curve, LineCurve) else curve.center),
            True,
        )
        first = coordinates(
            np.asarray(
                curve.direction if isinstance(curve, LineCurve) else curve.first_axis
            ),
            False,
        )
        if origin is None or first is None or not np.any(first):
            # A direction normal to the plane has no planar lift.
            return None
        if isinstance(curve, LineCurve):
            return LineCurve(origin, first)
        second = coordinates(np.asarray(curve.second_axis), False)
        if second is not None and np.any(second):
            return CircleCurve(origin, first, second, curve.radius)
    return None


def _source_pcurve_image(
    curve: CurveEvaluator, first: float, last: float, /
) -> np.ndarray:
    """Preserve exact constant chart walls in affine source-coordinate images."""
    from fractions import Fraction as F

    from ._intersection_curve import (
        _affine_curve_coefficients,
        pcurve_periodic_source,
        PeriodicPCurve,
    )

    if isinstance(curve, PeriodicPCurve):
        source = pcurve_periodic_source(curve, curve.patch)
        if source is None:
            raise ValueError(
                "The p-curve has no exact native periodic source factorization."
            )
        base, shifts = source
        image = _source_pcurve_image(base, first, last)
        if any(shifts):
            offset = PeriodicPCurve(base, curve.patch, shifts).period_offset_bounds()
            image = np.stack(interval_add((image[0], image[1]), (offset[0], offset[1])))
        return image
    coefficients = (
        _affine_curve_coefficients(curve, np.zeros(2))
        if isinstance(curve, AbstractCurve)
        else None
    )
    if coefficients is None:
        return np.asarray(
            curve.enclosure(first, last)
            if isinstance(curve, AbstractTrimCurve)
            else curve.bounding_box(first, last)
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
    region: SurfaceRegion,
    regions: tuple[SurfaceRegion, ...],
    box: np.ndarray,
    /,
) -> np.ndarray:
    """Intersect joint components with exact native plane-support equations."""
    from fractions import Fraction as F

    result = box.copy()
    patch = region.patch
    if not isinstance(patch, (CylinderPatch, PlanePatch)):
        return result

    def rational(vector: object) -> tuple[F, ...]:
        return tuple(F(float(value)) for value in np.asarray(vector))

    def dot(first: tuple[F, ...], second: tuple[F, ...]) -> F:
        return sum((a * b for a, b in zip(first, second, strict=True)), F(0))

    for supporting in regions:
        plane = supporting.patch
        if not isinstance(plane, PlanePatch):
            continue
        a, b = rational(plane.first_axis), rational(plane.second_axis)
        normal = (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
        delta = tuple(
            a - b
            for a, b in zip(rational(patch.origin), rational(plane.origin), strict=True)
        )
        constant = dot(normal, delta)
        if isinstance(patch, CylinderPatch):
            if dot(normal, rational(patch.first_axis)) or dot(
                normal, rational(patch.second_axis)
            ):
                continue
            axis, coefficient = 1, dot(normal, rational(patch.axis))
        else:
            coefficients = (
                dot(normal, rational(patch.first_axis)),
                dot(normal, rational(patch.second_axis)),
            )
            active = [index for index, value in enumerate(coefficients) if value]
            if len(active) != 1:
                continue
            axis = active[0]
            coefficient = coefficients[axis]
        if not coefficient:
            continue
        exact = -constant / coefficient
        lower = upper = float(exact)
        if F(lower) > exact:
            lower = np.nextafter(lower, -np.inf)
        if F(upper) < exact:
            upper = np.nextafter(upper, np.inf)
        result[0, axis] = max(result[0, axis], lower)
        result[1, axis] = min(result[1, axis], upper)
        if result[0, axis] > result[1, axis]:
            raise ValueError(
                "A joint root component contradicts its exact support equation."
            )
    return result


def _curve_plane_coordinate_image(
    curve: AbstractCurve, region: SurfaceRegion, box: np.ndarray, /
) -> np.ndarray:
    """Exact constant planar UV components implied by the spatial source edge."""
    from fractions import Fraction as F

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
    axes = np.asarray(
        [
            [F(float(value)) for value in np.asarray(axis)]
            for axis in (plane.first_axis, plane.second_axis)
        ],
        dtype=object,
    )
    gram = axes @ axes.T
    determinant = gram[0, 0] * gram[1, 1] - gram[0, 1] * gram[1, 0]
    if not determinant:
        return box
    inverse = (
        np.asarray(((gram[1, 1], -gram[0, 1]), (-gram[1, 0], gram[0, 0])), dtype=object)
        / determinant
    )
    dual = inverse @ axes
    origin = curve.origin if isinstance(curve, LineCurve) else curve.center
    relative = np.asarray(
        [
            F(float(a)) - F(float(b))
            for a, b in zip(np.asarray(origin), np.asarray(plane.origin), strict=True)
        ],
        dtype=object,
    )
    varying = (
        (curve.direction,)
        if isinstance(curve, LineCurve)
        else (curve.first_axis, curve.second_axis)
    )
    result = box.copy()
    for coordinate in range(2):
        if any(
            sum(
                (
                    a * F(float(b))
                    for a, b in zip(dual[coordinate], np.asarray(vector), strict=True)
                ),
                F(0),
            )
            for vector in varying
        ):
            continue
        exact = sum(
            (a * b for a, b in zip(dual[coordinate], relative, strict=True)), F(0)
        )
        lower = upper = float(exact)
        if F(lower) > exact:
            lower = np.nextafter(lower, -np.inf)
        if F(upper) < exact:
            upper = np.nextafter(upper, np.inf)
        result[0, coordinate] = max(result[0, coordinate], lower)
        result[1, coordinate] = min(result[1, coordinate], upper)
        if result[0, coordinate] > result[1, coordinate]:
            raise ValueError(
                "A spatial root contradicts its exact planar source coordinate."
            )
    return result


def _pcurve_source_chart(
    curve: CurveEvaluator,
    /,
) -> tuple[CurveEvaluator, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Original chart, exact affine operation and independent symbolic periods."""
    from fractions import Fraction

    from ._intersection_curve import AffinePCurve, PeriodicPCurve

    matrix = np.asarray(
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))), dtype=object
    )
    offset = np.asarray((Fraction(0), Fraction(0)), dtype=object)
    periods = np.asarray((Fraction(0), Fraction(0)), dtype=object)
    patches = []
    while isinstance(curve, (CurveTrimSegment, AffinePCurve, PeriodicPCurve)):
        if isinstance(curve, AffinePCurve):
            offset = offset + matrix @ np.asarray(curve.offset, dtype=object)
            matrix = matrix @ np.asarray(curve.matrix, dtype=object)
            curve = curve.curve
        elif isinstance(curve, PeriodicPCurve):
            periods = periods + matrix @ np.asarray(curve.period_shifts, dtype=object)
            patches.append(canonical_fingerprint(encode_geometry(curve.patch)))
            curve = curve.source_curve
        else:
            curve = curve.curve
    return curve, matrix, offset, periods, tuple(patches)


def _pcurve_parameter_image(
    curve: CurveEvaluator, lower: float, upper: float, /
) -> tuple[SurfaceRegion, SurfaceRegion, np.ndarray] | None:
    """Actual source coordinates behind a bounded p-curve root parameter."""
    from ._intersection_curve import AffinePCurve, PeriodicPCurve

    source_first: float | None = None
    source_last: float | None = None
    while isinstance(curve, (CurveTrimSegment, AffinePCurve, PeriodicPCurve)):
        if isinstance(curve, CurveTrimSegment):
            source_first, source_last = curve.first, curve.last
            interval = curve.carrier_parameter_enclosure(
                lower, upper, source_extension=True
            )
            lower, upper = float(interval[0]), float(interval[1])
        curve = curve.source_curve if isinstance(curve, PeriodicPCurve) else curve.curve
    if not isinstance(curve, IntersectionPCurve):
        return None
    if curve.reversed:
        total = interval_add(
            (np.asarray(curve.first), np.asarray(curve.first)),
            (np.asarray(curve.last), np.asarray(curve.last)),
        )
        image = interval_subtract(total, (np.asarray(lower), np.asarray(upper)))
        lower, upper = float(image[0]), float(image[1])
    from ._intersection_curve import _pcurve_chart_support

    head, tail = _pcurve_chart_support(
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
    )
    box = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
    return curve.curve.first, curve.curve.second, box


def _joint_root_parameter_image(
    root: TrimIntersectionRoot
    | CurveSurfaceIntersectionRoot
    | TripleSurfaceIntersectionRoot,
    regions: tuple[SurfaceRegion, SurfaceRegion, SurfaceRegion],
    /,
    *,
    source_edge_lifts: Sequence[_SourceCurveSurfaceLift] = (),
    joint_box: np.ndarray | None = None,
) -> np.ndarray | None:
    """Equation identity plus parameter inclusion, not point-box coincidence."""
    contributions: dict[str, list[np.ndarray]] = {}
    parameters = root.parameter_enclosure()

    def add(region: SurfaceRegion, box: np.ndarray) -> None:
        contributions.setdefault(_surface_definition_key(region), []).append(box)

    if isinstance(root, TripleSurfaceIntersectionRoot):
        for index, region in enumerate((root.first, root.second, root.third)):
            add(region, parameters[:, 2 * index : 2 * index + 2])
    elif isinstance(root, CurveSurfaceIntersectionRoot):
        if isinstance(root.curve, IntersectionCurve):
            boxes = root.curve.parameter_enclosures(
                float(parameters[0, 0]), float(parameters[1, 0]), source_extension=True
            )
            coupled = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
            add(root.curve.first, coupled[:, :2])
            add(root.curve.second, coupled[:, 2:])
        elif isinstance(root.curve, AbstractCurve):
            first, last = float(parameters[0, 0]), float(parameters[1, 0])
            for index, region in enumerate(regions):
                images: list[np.ndarray] = []
                if _surface_definition_key(region) == _surface_definition_key(
                    root.surface
                ):
                    continue
                pcurve = _native_edge_pcurve(root.curve, region)
                if pcurve is not None and _exact_source_lift(
                    root.curve, pcurve, region, first, last
                ):
                    images.append(_source_pcurve_image(pcurve, first, last))
                for lift in source_edge_lifts:
                    if (
                        isinstance(getattr(lift, "surface", None), SurfaceRegion)
                        and _surface_definition_key(lift.surface)
                        == _surface_definition_key(region)
                        and _same_geometry(lift.curve, root.curve)
                        and lift.first <= first <= last <= lift.last
                        and _exact_source_lift(
                            root.curve, lift.pcurve, region, lift.first, lift.last
                        )
                    ):
                        images.append(_source_pcurve_image(lift.pcurve, first, last))
                for image in images:
                    # Distinct declared UV sheets are alternatives, not boxes
                    # whose overlap may be mistaken for parameter equality.
                    if joint_box is None or (
                        np.all(image[0] >= joint_box[0, 2 * index : 2 * index + 2])
                        and np.all(image[1] <= joint_box[1, 2 * index : 2 * index + 2])
                    ):
                        add(region, image)
        else:
            return None
        add(root.surface, parameters[:, 1:])
    elif (
        isinstance(root, TrimIntersectionRoot)
        and (
            native := _native_trim_root_contributions(
                root, regions, source_edge_lifts, joint_box
            )
        )
        is not None
    ):
        contributions = native
    elif isinstance(root, TrimIntersectionRoot):
        images = []
        anchors = []
        maps = []
        for index, source in enumerate((root.first, root.second)):
            image = _pcurve_parameter_image(
                source, float(parameters[0, index]), float(parameters[1, index])
            )
            if image is None:
                return None
            first, second, box = image
            underlying, matrix, offset, periods, patches = _pcurve_source_chart(source)
            if not isinstance(underlying, IntersectionPCurve):
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
        for first, second, box in images:
            add(first, box[:, :2])
            add(second, box[:, 2:])
        if (
            np.array_equal(maps[0][0], np.eye(2, dtype=int))
            and not np.any(maps[0][1])
            and not np.any(maps[0][2])
        ):
            contributions[anchors[0]] = [root.point_enclosure()]
    else:
        return None
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


def _native_trim_root_contributions(
    root: TrimIntersectionRoot,
    regions: tuple[SurfaceRegion, SurfaceRegion, SurfaceRegion],
    lifts: Sequence[_SourceCurveSurfaceLift],
    joint_box: np.ndarray | None,
    /,
) -> dict[str, list[np.ndarray]] | None:
    """A UV junction of two native source p-curves on one face, through exact lifts.

    Supplied lifts only name the source edge behind each operand's p-curve;
    every edge/face identity is re-proved exactly over the root's own carrier
    parameter enclosure. The face contributes the root's UV enclosure and each
    edge its exact native p-curve image on every other face it lies on.
    """
    operands: list[tuple[AbstractCurve, float, float]] = []
    for operand in ("first", "second"):
        endpoint = TrimRootEndpoint(root, operand)
        carrier = endpoint.carrier
        if not isinstance(carrier, AbstractCurve) or carrier.ambient_dimension != 2:
            return None
        if isinstance(_pcurve_source_chart(carrier)[0], IntersectionPCurve):
            return None
        lower, upper = endpoint.parameter_enclosure()
        operands.append((carrier, lower, upper))
    keys = [_surface_definition_key(region) for region in regions]
    if len(set(keys)) != 3:
        return None

    def inside(image: np.ndarray, index: int) -> bool:
        return joint_box is None or bool(
            np.all(image[0] >= joint_box[0, 2 * index : 2 * index + 2])
            and np.all(image[1] <= joint_box[1, 2 * index : 2 * index + 2])
        )

    uv = root.point_enclosure()
    for face, region in enumerate(regions):
        if not inside(uv, face):
            continue
        edges: list[AbstractCurve] = []
        for carrier, lower, upper in operands:
            edge = next(
                (
                    lift.curve
                    for lift in lifts
                    if isinstance(lift.curve, AbstractCurve)
                    and _surface_definition_key(lift.surface) == keys[face]
                    and _same_geometry(lift.pcurve, carrier)
                    and _exact_source_lift(lift.curve, carrier, region, lower, upper)
                ),
                None,
            )
            if edge is None:
                break
            edges.append(edge)
        if len(edges) != 2:
            continue
        contributions: dict[str, list[np.ndarray]] = {keys[face]: [uv]}
        for edge, (_, lower, upper) in zip(edges, operands, strict=True):
            for index, other in enumerate(regions):
                if index == face:
                    continue
                candidates = [
                    lift.pcurve
                    for lift in lifts
                    if isinstance(lift.pcurve, AbstractCurve)
                    and _surface_definition_key(lift.surface) == keys[index]
                    and _same_geometry(lift.curve, edge)
                ]
                native = _native_edge_pcurve(edge, other)
                if native is not None:
                    candidates.append(native)
                for pcurve in candidates:
                    if not _exact_source_lift(edge, pcurve, other, lower, upper):
                        continue
                    image = _source_pcurve_image(pcurve, lower, upper)
                    if inside(image, index):
                        contributions.setdefault(keys[index], []).append(image)
        box = _intersect_contributions(contributions, keys)
        if box is not None and (
            joint_box is None
            or bool(np.all(box[0] >= joint_box[0]) and np.all(box[1] <= joint_box[1]))
        ):
            return contributions
    return None


class TrimRootEndpoint(StrictModule):
    """A source root's scalar atom, optionally lifted to its exact native p-curve."""

    root: TrimIntersectionRoot | CurveSurfaceIntersectionRoot | IntersectionCurvePointRoot
    operand: Literal["first", "second"] = eqx.field(static=True)
    affine_parameter_offset: float = eqx.field(static=True)
    affine_parameter_scale: float = eqx.field(static=True)
    affine_transforms: tuple[tuple[float, float], ...] = eqx.field(static=True)
    source_pcurve: AbstractCurve | IntersectionPCurve | None
    source_surface: SurfaceRegion | None

    def __init__(
        self,
        root: TrimIntersectionRoot
        | CurveSurfaceIntersectionRoot
        | IntersectionCurvePointRoot,
        operand: Literal["first", "second"],
        /,
        *,
        affine_parameter_offset: float = 0.0,
        affine_parameter_scale: float = 1.0,
        affine_transforms: tuple[tuple[float, float], ...] = (),
        source_pcurve: AbstractCurve | IntersectionPCurve | None = None,
        source_surface: SurfaceRegion | None = None,
    ) -> None:
        if not isinstance(
            root,
            (
                TrimIntersectionRoot,
                CurveSurfaceIntersectionRoot,
                IntersectionCurvePointRoot,
            ),
        ) or operand not in ("first", "second"):
            raise TypeError("A root endpoint requires a canonical root and one operand.")
        if (
            isinstance(root, (CurveSurfaceIntersectionRoot, IntersectionCurvePointRoot))
            and operand != "first"
        ):
            raise ValueError(
                "Spatial endpoint expressions select the source edge parameter only."
            )
        if (
            not math.isfinite(affine_parameter_offset)
            or not math.isfinite(affine_parameter_scale)
            or affine_parameter_scale == 0
        ):
            raise ValueError(
                "Root endpoint affine expressions must be finite with nonzero scale."
            )
        if (source_pcurve is None) != (source_surface is None):
            raise ValueError(
                "A lifted scalar endpoint requires both its p-curve and source surface."
            )
        if source_pcurve is not None and source_surface is not None:
            if not isinstance(root, CurveSurfaceIntersectionRoot):
                raise ValueError(
                    "Only a spatial source root needs a native p-curve lift."
                )
            if not isinstance(
                source_pcurve, (AbstractCurve, IntersectionPCurve)
            ) or not isinstance(source_surface, SurfaceRegion):
                raise TypeError(
                    "A scalar endpoint lift requires native p-curve and surface definitions."
                )
            if source_pcurve.ambient_dimension != 2:
                raise ValueError(
                    "A lifted scalar endpoint requires a two-dimensional p-curve."
                )
            if isinstance(root.curve, IntersectionCurve):
                generating = (
                    (
                        source_pcurve.curve.first
                        if source_pcurve.side == "first"
                        else source_pcurve.curve.second
                    )
                    if isinstance(source_pcurve, IntersectionPCurve)
                    else None
                )
                valid = (
                    isinstance(source_pcurve, IntersectionPCurve)
                    and source_pcurve.curve.branch_id == root.curve.branch_id
                    and generating is not None
                    and _surface_definition_key(generating)
                    == _surface_definition_key(source_surface)
                )
            else:
                valid = _exact_source_lift(
                    root.curve,
                    source_pcurve,
                    source_surface,
                    float(root.parameter_lower[0]),
                    float(root.parameter_upper[0]),
                )
            if not valid:
                raise ValueError(
                    "The scalar endpoint has no exact source edge/p-curve/support identity."
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
                "Composed root affine transforms must be finite and nonsingular."
            )
        self.affine_transforms = tuple(
            (float(scale), float(offset)) for scale, offset in affine_transforms
        )

    @property
    def root_id(self) -> str:
        return self.root.root_id

    def _expression(
        self, *, maximum_steps: int = 16
    ) -> tuple[CurveEvaluator | IntersectionCurve, tuple[np.ndarray, np.ndarray]]:
        if maximum_steps < 0:
            raise ValueError("Refinement steps must be nonnegative.")
        if isinstance(self.root, IntersectionCurvePointRoot):
            from fractions import Fraction

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
        parameters = self.root.parameter_enclosure(maximum_steps=maximum_steps)
        interval = (np.asarray(parameters[0, index]), np.asarray(parameters[1, index]))
        from ._intersection_curve import AffinePCurve, PeriodicPCurve

        operations: list[AffinePCurve | PeriodicPCurve] = []
        while isinstance(curve, (CurveTrimSegment, AffinePCurve, PeriodicPCurve)):
            if isinstance(curve, CurveTrimSegment):
                interval = curve.carrier_parameter_enclosure(
                    float(interval[0]), float(interval[1]), source_extension=True
                )
                curve = curve.curve
            elif isinstance(curve, PeriodicPCurve):
                operations.append(curve)
                curve = curve.source_curve
            else:
                operations.append(curve)
                curve = curve.curve
        if isinstance(curve, IntersectionPCurve) and curve.reversed:
            constant = interval_add(
                (np.asarray(curve.first), np.asarray(curve.first)),
                (np.asarray(curve.last), np.asarray(curve.last)),
            )
            interval = interval_subtract(constant, interval)
            curve = IntersectionPCurve(
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
                    "A UV source operation requires its original p-curve carrier."
                )
            if isinstance(operation, AffinePCurve):
                curve = AffinePCurve(curve, operation.matrix, operation.offset)
            else:
                curve = PeriodicPCurve(curve, operation.patch, operation.period_shifts)
        return curve, interval

    @property
    def carrier(self) -> CurveEvaluator | IntersectionCurve:
        return self._expression(maximum_steps=0)[0]

    @property
    def parameter(self) -> float:
        lower, upper = self.parameter_enclosure()
        return 0.5 * (lower + upper)

    def parameter_enclosure(self, /, *, maximum_steps: int = 16) -> tuple[float, float]:
        _, bounds = self._expression(maximum_steps=maximum_steps)
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
    """Exact chart scalar transport of a spatial or joint source root.

    The expression is ``chart + (q_axis - node_start)/(node_end-node_start)``.
    Both node coordinates remain source-root expressions, not floating chart
    representatives. Period shifts are explicit source gauge operations.
    """

    root: CurveSurfaceIntersectionRoot | TripleSurfaceIntersectionRoot
    curve: IntersectionCurve
    chart: int = eqx.field(static=True)
    source_pcurve: AbstractCurve | None
    source_side: Literal["first", "second"] | None = eqx.field(static=True)
    source_first: float | None = eqx.field(static=True)
    source_last: float | None = eqx.field(static=True)
    periodic_shifts: tuple[int, int, int, int] = eqx.field(static=True)
    affine_transforms: tuple[tuple[float, float], ...] = eqx.field(static=True)

    def __init__(
        self,
        root: CurveSurfaceIntersectionRoot | TripleSurfaceIntersectionRoot,
        curve: IntersectionCurve,
        chart: int,
        /,
        *,
        source_pcurve: AbstractCurve | None = None,
        source_side: Literal["first", "second"] | None = None,
        source_first: float | None = None,
        source_last: float | None = None,
        periodic_shifts: tuple[int, int, int, int] = (0, 0, 0, 0),
        affine_transforms: tuple[tuple[float, float], ...] = (),
    ) -> None:
        if not isinstance(
            root, (CurveSurfaceIntersectionRoot, TripleSurfaceIntersectionRoot)
        ):
            raise TypeError(
                "Branch endpoints require an authoritative spatial or joint root."
            )
        if not isinstance(curve, IntersectionCurve) or not curve.fully_certified:
            raise ValueError("Branch endpoints require a fully certified source atlas.")
        if type(chart) is not int or not 0 <= chart < curve.num_charts:
            raise ValueError("A branch endpoint must select an actual source chart.")
        if len(periodic_shifts) != 4 or any(
            type(value) is not int for value in periodic_shifts
        ):
            raise TypeError("Branch endpoint gauge shifts must be four exact integers.")
        if any(
            not math.isfinite(scale) or not math.isfinite(offset) or scale == 0
            for scale, offset in affine_transforms
        ):
            raise ValueError("Endpoint affine transforms must be finite and nonsingular.")
        self.root, self.curve, self.chart = root, curve, chart
        self.source_pcurve, self.source_side = source_pcurve, source_side
        self.source_first, self.source_last = source_first, source_last
        self.periodic_shifts = tuple(periodic_shifts)
        self.affine_transforms = tuple(
            (float(scale), float(offset)) for scale, offset in affine_transforms
        )
        if isinstance(root, TripleSurfaceIntersectionRoot):
            if any(
                value is not None
                for value in (source_pcurve, source_side, source_first, source_last)
            ):
                raise ValueError(
                    "Joint roots transport their generating UV components directly."
                )
        elif isinstance(root.curve, IntersectionCurve):
            if (
                source_pcurve is not None
                or source_side is not None
                or source_first is not None
                or source_last is not None
            ):
                raise ValueError(
                    "A coupled source branch already owns its exact UV images."
                )
            if {
                _surface_definition_key(root.curve.first),
                _surface_definition_key(root.curve.second),
            } != {
                _surface_definition_key(curve.first),
                _surface_definition_key(curve.second),
            }:
                raise ValueError(
                    "The spatial source branch does not generate the target surfaces."
                )
        else:
            if source_side not in ("first", "second") or not isinstance(
                source_pcurve, AbstractCurve
            ):
                raise TypeError(
                    "An original edge requires its exact pcurve and generating support side."
                )
            source = curve.first if source_side == "first" else curve.second
            opposite = curve.second if source_side == "first" else curve.first
            if _surface_definition_key(root.surface) != _surface_definition_key(opposite):
                raise ValueError(
                    "The spatial root does not intersect the opposite generating surface."
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
                or not first <= root.parameter_lower[0] <= root.parameter_upper[0] <= last
            ):
                raise ValueError(
                    "The whole-edge lift must cover the source root parameter box."
                )
            if not _exact_source_lift(root.curve, source_pcurve, source, first, last):
                raise ValueError(
                    "The source edge/pcurve/support have no exact construction identity."
                )
            self.source_first, self.source_last = first, last
        image = self._source_image(maximum_steps=16)
        if not np.all(image[0] >= curve.box_lower[chart]) or not np.all(
            image[1] <= curve.box_upper[chart]
        ):
            raise ValueError(
                "The full source parameter image is not inside the unique target chart box."
            )
        self.parameter_enclosure(maximum_steps=0)

    @property
    def root_id(self) -> str:
        return self.root.root_id

    @property
    def carrier(self) -> IntersectionCurve:
        return self.curve

    def _source_image(
        self, /, *, maximum_steps: int, include_periodic_shifts: bool = True
    ) -> np.ndarray:
        parameters = self.root.parameter_enclosure(maximum_steps=maximum_steps)
        if isinstance(self.root, TripleSurfaceIntersectionRoot):
            regions = (self.root.first, self.root.second, self.root.third)
            columns = []
            for target in (self.curve.first, self.curve.second):
                matches = [
                    index
                    for index, region in enumerate(regions)
                    if _surface_definition_key(region) == _surface_definition_key(target)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "The joint root has no unique generating patch definition."
                    )
                index = matches[0]
                columns.append(
                    _joint_coordinate_image(
                        target, regions, parameters[:, 2 * index : 2 * index + 2]
                    )
                )
            image = np.concatenate(columns, axis=1)
        elif isinstance(self.root.curve, IntersectionCurve):
            boxes = self.root.curve.parameter_enclosures(
                float(parameters[0, 0]), float(parameters[1, 0])
            )
            coupled = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
            columns = []
            for target in (self.curve.first, self.curve.second):
                offset = (
                    0
                    if _surface_definition_key(target)
                    == _surface_definition_key(self.root.curve.first)
                    else 2
                )
                columns.append(coupled[:, offset : offset + 2])
            image = np.concatenate(columns, axis=1)
        else:
            from ._intersection_curve import pcurve_periodic_source

            pcurve = self.source_pcurve
            if pcurve is None or self.source_side not in ("first", "second"):
                raise ValueError(
                    "An original spatial edge requires its exact source p-curve."
                )
            patch = (
                self.curve.first.patch
                if self.source_side == "first"
                else self.curve.second.patch
            )
            factorization = pcurve_periodic_source(pcurve, patch)
            if factorization is None:
                raise ValueError("The source p-curve has no exact integer native gauge.")
            source = _source_pcurve_image(
                factorization[0], float(parameters[0, 0]), float(parameters[1, 0])
            )
            opposite = _curve_plane_coordinate_image(
                self.root.curve, self.root.surface, parameters[:, 1:]
            )
            image = np.concatenate(
                (source, opposite) if self.source_side == "first" else (opposite, source),
                axis=1,
            )
        gauges = self._source_gauges()
        lower, upper = self.curve._shift_bounds(gauges)
        if include_periodic_shifts and np.any(gauges):
            image = np.stack(interval_add((image[0], image[1]), (lower, upper)))
        return image

    def _source_gauges(self) -> np.ndarray:
        """Compose explicit source-sheet and branch gauges before realization."""
        from ._intersection_curve import pcurve_periodic_source

        shifts = np.asarray(self.periodic_shifts, dtype=np.int64).copy()
        if isinstance(self.root, CurveSurfaceIntersectionRoot) and not isinstance(
            self.root.curve, IntersectionCurve
        ):
            pcurve = self.source_pcurve
            if pcurve is None or self.source_side not in ("first", "second"):
                raise ValueError(
                    "An original spatial edge requires its exact source p-curve."
                )
            patch = (
                self.curve.first.patch
                if self.source_side == "first"
                else self.curve.second.patch
            )
            source = pcurve_periodic_source(pcurve, patch)
            if source is None:
                raise ValueError("The source p-curve has no exact integer native gauge.")
            offset = 0 if self.source_side == "first" else 2
            shifts[offset : offset + 2] += np.asarray(source[1], dtype=np.int64)
        return shifts

    def certifies_point_root(
        self, point_root: IntersectionCurvePointRoot, /, *, maximum_steps: int = 16
    ) -> bool:
        """Prove a source spatial root is this exact shared-node point atom.

        The source must satisfy the node's actual coordinate constraint and
        lie wholly inside its unique source-system witness. Scalar interval
        overlap, representative equality and spatial boxes are not used.
        """
        if (
            not isinstance(point_root, IntersectionCurvePointRoot)
            or point_root.curve.branch_id != self.curve.branch_id
        ):
            return False
        if not point_root.parameter.is_integer():
            return False
        target = int(point_root.parameter)
        reference = int(self.curve.node_references[target])
        image = self._source_image(
            maximum_steps=maximum_steps, include_periodic_shifts=False
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
            difference = self._source_gauges() - shifts
            transported = image
            if np.any(difference):
                lower, upper = self.curve._shift_bounds(difference)
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

    def parameter_enclosure(self, /, *, maximum_steps: int = 16) -> tuple[float, float]:
        if type(maximum_steps) is not int or maximum_steps < 0:
            raise ValueError("Refinement steps must be nonnegative integers.")
        image = self._source_image(maximum_steps=maximum_steps)
        axis = int(self.curve.chart_axes[self.chart])
        start = (
            self.curve.chart_start_lower[self.chart, axis],
            self.curve.chart_start_upper[self.chart, axis],
        )
        end = (
            self.curve.chart_end_lower[self.chart, axis],
            self.curve.chart_end_upper[self.chart, axis],
        )
        advance = interval_subtract(end, start)
        if advance[0] <= 0 <= advance[1]:
            raise ValueError(
                "The exact source chart has no separated endpoint coordinates."
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
        lower, upper = self.parameter_enclosure()
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
    curve: CurveEvaluator | IntersectionCurve,
    value: Fraction | _PeriodOffset,
    /,
) -> bool | None:
    """Exact inclusion in a carrier domain that is one native period by construction.

    Only a derived isoline (no authored ``parameter_range``) along a proved
    native 2*pi surface axis has the exact domain ``[0, 2*pi]``; its binary
    domain endpoint is merely that period's representative. ``None`` means the
    carrier has an authored domain, whose binary bounds are never read as tau.
    """
    from ._intersection_curve import (
        _native_curve_period_symbol,
        _period_offset,
        _period_scalar_bounds,
    )

    source = curve.definition if isinstance(curve, PlacedCurve) else curve
    if (
        not isinstance(source, SurfaceIsoparametricCurve)
        or source.parameter_range is not None
        or _native_curve_period_symbol(source) != "two_pi"
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
            float(_period_scalar_bounds(_period_offset(scalar_rational, scalar_turns))[0])
            >= 0.0
        )

    return nonnegative(rational, turns) and nonnegative(-rational, 1 - turns)


class NativePeriodEndpoint(StrictModule):
    """Authored source parameter in Q + Q*(2*pi), not a floating root.

    ``patch`` and ``axis`` prove a surface period when supplied; otherwise the
    original curve must own a proved mathematical period. Numerical
    representatives never change a separately authored binary64 endpoint.
    """

    curve: CurveEvaluator | IntersectionCurve
    patch: AbstractSurfacePatch | None
    axis: Literal[0, 1] | None = eqx.field(static=True)
    rational: Fraction = eqx.field(static=True)
    turns: Fraction = eqx.field(static=True)

    def __init__(
        self,
        curve: CurveEvaluator | IntersectionCurve,
        patch: AbstractSurfacePatch | None = None,
        axis: Literal[0, 1] | None = None,
        /,
        *,
        rational: Fraction | int = 0,
        turns: Fraction | int = 1,
    ) -> None:
        from ._intersection_curve import (
            _native_curve_period_symbol,
            _native_period_symbols,
        )

        if not isinstance(curve, (AbstractCurve, AbstractTrimCurve, IntersectionCurve)):
            raise TypeError(
                "A native-period endpoint requires its original source curve."
            )
        if patch is None:
            if axis is not None:
                raise ValueError("Curve-owned period authority has no surface axis.")
            if (
                not isinstance(curve, (AbstractCurve, IntersectionCurve))
                or _native_curve_period_symbol(curve) != "two_pi"
            ):
                raise ValueError(
                    "The original source curve has no proved mathematical native period."
                )
        else:
            if (
                not isinstance(patch, AbstractSurfacePatch)
                or type(axis) is not int
                or axis not in (0, 1)
            ):
                raise TypeError(
                    "A native-period endpoint requires a native surface and axis zero or one."
                )
            if _native_period_symbols(patch)[axis] != "two_pi":
                raise ValueError(
                    "The selected source axis has no proved mathematical native period."
                )
        if any(
            not isinstance(value, (Fraction, int)) or isinstance(value, bool)
            for value in (rational, turns)
        ):
            raise TypeError(
                "Native-period scalar coefficients must be exact rationals or integers."
            )
        self.curve, self.patch, self.axis = curve, patch, axis
        self.rational, self.turns = Fraction(rational), Fraction(turns)
        lower, upper = self.parameter_enclosure()
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError(
                "Native-period endpoints require finite numerical enclosures."
            )
        domain = (
            (0.0, float(curve.num_charts))
            if isinstance(curve, IntersectionCurve)
            else curve.parameter_domain
        )
        derived = _within_derived_native_period(curve, self.exact_parameter)
        if derived is False or (
            derived is None
            and domain is not None
            and not domain[0] <= lower <= upper <= domain[1]
        ):
            raise ValueError(
                "The exact native-period endpoint lies outside its source curve domain."
            )

    @property
    def carrier(self) -> CurveEvaluator | IntersectionCurve:
        return self.curve

    @property
    def exact_parameter(self) -> Fraction | _PeriodOffset:
        from ._intersection_curve import _period_offset

        return _period_offset(self.rational, self.turns)

    @property
    def parameter(self) -> float:
        return float(self.rational + self.turns * Fraction(math.tau))

    @property
    def root_id(self) -> str:
        return canonical_fingerprint(encode_geometry(self))

    @property
    def affine_transforms(self) -> tuple[tuple[float, float], ...]:
        return ()

    def parameter_enclosure(self, /, *, maximum_steps: int = 16) -> tuple[float, float]:
        from ._intersection_curve import _period_scalar_bounds

        if type(maximum_steps) is not int or maximum_steps < 0:
            raise ValueError("Refinement steps must be nonnegative integers.")
        lower, upper = _period_scalar_bounds(self.exact_parameter)
        return float(lower), float(upper)

    def affine(self, scale: float, offset: float, /) -> NativePeriodEndpoint:
        if not math.isfinite(scale) or not math.isfinite(offset) or scale == 0:
            raise ValueError(
                "Endpoint affine expressions must be finite with nonzero scale."
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
    ``transversal`` events are unique regular roots certified inside
    ``[parameter_lower, parameter_upper]``. ``tangent`` events are certified
    unique foot-point/parallel-tangent critical points whose operand gap is at
    most ``gap_bound``; ``singular`` events are such contacts where an operand's
    parametrization is degenerate or where branches of a surface intersection
    meet.
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
    ) -> None:
        scope = Scope()
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

    curves: tuple[IntersectionCurve, ...]
    points: tuple[ParametricIntersectionPoint, ...]
    coincident: tuple[CoincidentParameterRegion, ...]
    unresolved: tuple[UnresolvedParameterRegion, ...]
    work: ParametricIntersectionWork

    def __init__(
        self,
        curves: Sequence[IntersectionCurve],
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


def _curve_value(curve: CurveEvaluator | IntersectionCurve, parameter: Array, /) -> Array:
    if isinstance(curve, IntersectionCurve):
        return curve.evaluate(parameter).point
    return curve.evaluate(parameter)


def _curve_derivative(
    curve: CurveEvaluator | IntersectionCurve, parameter: Array, /
) -> Array:
    return jax.jacfwd(lambda value: _curve_value(curve, value))(parameter)


def _surface_derivative(surface: SurfaceEvaluator, parameters: Array, /) -> Array:
    return jax.jacfwd(surface.evaluate)(parameters)


class _CurvePairSystem(StrictModule):
    first: CurveEvaluator | IntersectionCurve
    second: CurveEvaluator | IntersectionCurve

    def residual(self, parameters: Array, /) -> Array:
        return _curve_value(self.first, parameters[0]) - _curve_value(
            self.second, parameters[1]
        )


class _CurveProximitySystem(StrictModule):
    """Stationarity of ``|c1(s) - c2(t)|^2`` for space curves."""

    first: CurveEvaluator | IntersectionCurve
    second: CurveEvaluator | IntersectionCurve

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

    first: CurveEvaluator | IntersectionCurve
    second: CurveEvaluator | IntersectionCurve

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


class _TripleSurfaceSystem(StrictModule):
    first: SurfaceEvaluator
    second: SurfaceEvaluator
    third: SurfaceEvaluator

    def residual(self, parameters: Array, /) -> Array:
        point = self.first.evaluate(parameters[:2])
        return jnp.concatenate(
            (
                point - self.second.evaluate(parameters[2:4]),
                point - self.third.evaluate(parameters[4:]),
            )
        )


class _CurveSurfaceSystem(StrictModule):
    curve: CurveEvaluator | IntersectionCurve
    surface: SurfaceEvaluator

    def residual(self, parameters: Array, /) -> Array:
        if isinstance(self.curve, IntersectionCurve):
            point = self.curve.evaluate(parameters[0]).point
        else:
            point = self.curve.evaluate(parameters[0])
        return point - self.surface.evaluate(parameters[1:])


class _CurveSurfaceTangencySystem(StrictModule):
    """Foot point of ``c(t)`` on the surface with the curve tangent in its plane."""

    curve: CurveEvaluator | IntersectionCurve
    surface: SurfaceEvaluator

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

    pair: SurfacePairSystem

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

    pair: SurfacePairSystem
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

    pair: SurfacePairSystem
    axis: int = eqx.field(static=True)
    value: Array

    def residual(self, parameters: Array, /) -> Array:
        fixed = jnp.reshape(self.value, (1,)).astype(parameters.dtype)
        return self.pair.residual(
            jnp.concatenate((parameters[: self.axis], fixed, parameters[self.axis :]))
        )


class _CorrectorSystem(StrictModule):
    """Branch equations plus the hyperplane through a predictor."""

    pair: SurfacePairSystem
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
    | _CurveSurfaceSystem
    | _CurveSurfaceTangencySystem
    | _SurfaceTangencySystem
    | _TurningSystem
    | _FaceSystem
    | _CorrectorSystem
    | SurfacePairSystem
    | _TripleSurfaceSystem
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
def _inverses(matrices: Array, plan: SmallLinearSolvePlan) -> Array:
    return inverse_small_linear(plan, matrices).value


@eqx.filter_jit
def _dense_inverses(matrices: Array) -> Array:
    def one(matrix: Array) -> Array:
        result = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            jnp.eye(matrix.shape[0], dtype=matrix.dtype),
            policy=LinearSolvePolicy(DenseLU()),
        )
        return jnp.where(
            jnp.all(result.successful), result.value, jnp.zeros_like(result.value)
        )

    return jax.vmap(one)(matrices)


@eqx.filter_jit
def _branch_tangent(pair: SurfacePairSystem, point: Array, reference: Array) -> Array:
    jacobian = jax.jacfwd(pair.residual)(point)
    matrix = jnp.concatenate((jacobian, reference[None]), axis=0)
    right = jnp.zeros((4,), dtype=point.dtype).at[3].set(1.0)
    tangent = solve_small_linear(SmallLinearSolvePlan(4), matrix, right).value
    return tangent / jnp.linalg.norm(tangent)


@eqx.filter_jit
def _march_step(
    pair: SurfacePairSystem,
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


def _mid_rad(lower: np.ndarray, upper: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    middle = 0.5 * (lower + upper)
    radius = np.nextafter(np.maximum(middle - lower, upper - middle), np.inf)
    return middle, radius


def _round(lower: np.ndarray, upper: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    # Indeterminate midpoint products arise for valid unbounded enclosures
    # (for example, a coarse rational denominator interval straddling zero).
    # They mean no information, NEVER an exclusion certificate.
    lower = np.where(np.isnan(lower), -np.inf, lower)
    upper = np.where(np.isnan(upper), np.inf, upper)
    return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)


def _point_times_interval(
    matrix: np.ndarray, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """``matrix @ [lower, upper]`` for batched point matrices ``(B, m, k)``."""
    middle, radius = _mid_rad(lower, upper)
    absolute = np.abs(matrix)
    center = matrix @ middle
    width = absolute @ radius
    length = matrix.shape[-1] + 1
    gamma = length * _UNIT / (1.0 - length * _UNIT)
    width = np.nextafter(width + gamma * (absolute @ np.abs(middle) + width), np.inf)
    return _round(center - width, center + width)


def _interval_times_interval(
    first_lower: np.ndarray,
    first_upper: np.ndarray,
    second_lower: np.ndarray,
    second_upper: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    first_mid, first_rad = _mid_rad(first_lower, first_upper)
    second_mid, second_rad = _mid_rad(second_lower, second_upper)
    center = first_mid @ second_mid
    width = (
        np.abs(first_mid) @ second_rad
        + first_rad @ np.abs(second_mid)
        + first_rad @ second_rad
    )
    length = first_mid.shape[-1] + 1
    gamma = length * _UNIT / (1.0 - length * _UNIT)
    width = np.nextafter(
        width + gamma * (np.abs(first_mid) @ np.abs(second_mid) + width), np.inf
    )
    return _round(center - width, center + width)


def _contains_zero(lower: np.ndarray, upper: np.ndarray, /) -> np.ndarray:
    return np.all((lower <= 0.0) & (upper >= 0.0), axis=-1)


def _magnitude(lower: np.ndarray, upper: np.ndarray, /) -> np.ndarray:
    return np.maximum(np.abs(lower), np.abs(upper))


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

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        def jet(
            curve: CurveEvaluator | IntersectionCurve,
            first: float,
            last: float,
            order: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            if isinstance(curve, IntersectionCurve):
                domain = curve.parameter_domain
                if first < domain[0] or last > domain[1]:
                    return np.full(3, -np.inf), np.full(3, np.inf)
                if order == 0:
                    box = curve.bounding_box(first, last)
                    return box[0], box[1]
                return curve.derivative_bounds(first, last, order=order)
            if isinstance(curve, AbstractTrimCurve):
                return _trim_source_jet(curve, first, last, order)
            derivative = curve.evaluate
            for _ in range(order):
                derivative = jax.jacfwd(derivative)
            prepared = prepare_interval_function(
                lambda parameter: derivative(parameter[0]),
                1,
                batch_capacity=1,
                constant_bounds=coefficient_enclosures(curve),
            )
            lo, hi = prepared.evaluate(np.asarray([[first]]), np.asarray([[last]]))
            return lo[0], hi[0]

        def dot(
            a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray]:
            product = interval_multiply(a, b)
            lo = np.nextafter(
                np.sum(product[0]) - 8 * _UNIT * np.sum(np.abs(product[0])), -np.inf
            )
            hi = np.nextafter(
                np.sum(product[1]) + 8 * _UNIT * np.sum(np.abs(product[1])), np.inf
            )
            return lo, hi

        def determinant(
            a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray]:
            return interval_subtract(
                interval_multiply((a[0][0], a[1][0]), (b[0][1], b[1][1])),
                interval_multiply((a[0][1], a[1][1]), (b[0][0], b[1][0])),
            )

        lower_outputs, upper_outputs = [], []
        for lo, hi in zip(lower, upper, strict=True):
            a = jet(self.system.first, float(lo[0]), float(hi[0]), 0)
            b = jet(self.system.second, float(lo[1]), float(hi[1]), 0)
            gap = interval_subtract(a, b)
            if isinstance(self.system, _CurvePairSystem) and not self.jacobian:
                result = gap
            else:
                da = jet(self.system.first, float(lo[0]), float(hi[0]), 1)
                db = jet(self.system.second, float(lo[1]), float(hi[1]), 1)
                if isinstance(self.system, _CurvePairSystem):
                    result = (
                        np.stack((da[0], -db[1]), axis=1),
                        np.stack((da[1], -db[0]), axis=1),
                    )
                elif isinstance(self.system, _CurveProximitySystem):
                    if not self.jacobian:
                        first_value, second_value = dot(gap, da), dot(gap, db)
                        result = (
                            np.asarray((first_value[0], second_value[0])),
                            np.asarray((first_value[1], second_value[1])),
                        )
                    else:
                        dda = jet(self.system.first, float(lo[0]), float(hi[0]), 2)
                        ddb = jet(self.system.second, float(lo[1]), float(hi[1]), 2)
                        aa = interval_add(dot(da, da), dot(gap, dda))
                        ab = dot(da, db)
                        bb = dot(db, db)
                        bb = interval_add((-bb[1], -bb[0]), dot(gap, ddb))
                        result = (
                            np.asarray(((aa[0], -ab[1]), (ab[0], bb[0]))),
                            np.asarray(((aa[1], -ab[0]), (ab[1], bb[1]))),
                        )
                elif not self.jacobian:
                    value_a, value_b = dot(gap, db), determinant(da, db)
                    result = (
                        np.asarray((value_a[0], value_b[0])),
                        np.asarray((value_a[1], value_b[1])),
                    )
                else:
                    dda = jet(self.system.first, float(lo[0]), float(hi[0]), 2)
                    ddb = jet(self.system.second, float(lo[1]), float(hi[1]), 2)
                    aa = dot(da, db)
                    bb = dot(db, db)
                    ab = interval_add((-bb[1], -bb[0]), dot(gap, ddb))
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
    product = interval_multiply(first, second)
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
    """Spatial source branch bounds composed with a canonical surface evaluator."""

    system: _CurveSurfaceSystem | _CurveSurfaceTangencySystem
    jacobian: bool

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (3, 3) if self.jacobian else (3,)

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        curve = self.system.curve
        if not isinstance(curve, IntersectionCurve):
            raise TypeError(
                "Source branch interval preparation requires an IntersectionCurve."
            )
        if isinstance(self.system, _CurveSurfaceTangencySystem):
            return self._tangency_enclosures(lower, upper)
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
            surface_lo, surface_hi = prepared.evaluate(lo[None, 1:], hi[None, 1:])
            if self.jacobian:
                first, last = float(lo[0]), float(hi[0])
                curve_lo, curve_hi = curve.derivative_bounds(first, last)
                lows.append(np.concatenate((curve_lo[:, None], -surface_hi[0]), axis=1))
                highs.append(np.concatenate((curve_hi[:, None], -surface_lo[0]), axis=1))
            else:
                box = curve.bounding_box(float(lo[0]), float(hi[0]))
                value_lo, value_hi = interval_subtract(
                    (box[0], box[1]), (surface_lo[0], surface_hi[0])
                )
                lows.append(value_lo)
                highs.append(value_hi)
        return np.asarray(lows), np.asarray(highs)

    def _tangency_enclosures(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Differentiate the actual gap/frame/tangent equations by source jets."""
        curve = self.system.curve
        if not isinstance(curve, IntersectionCurve):
            raise TypeError("Source tangency intervals require an IntersectionCurve.")
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
            box = curve.bounding_box(first, last)
            surface_lo, surface_hi = values.evaluate(lo[None, 1:], hi[None, 1:])
            gap = interval_subtract((box[0], box[1]), (surface_lo[0], surface_hi[0]))
            frame_lo, frame_hi = frames.evaluate(lo[None, 1:], hi[None, 1:])
            u = (frame_lo[0, :, 0], frame_hi[0, :, 0])
            v = (frame_lo[0, :, 1], frame_hi[0, :, 1])
            normal = _vector_interval_cross(u, v)
            tangent = curve.derivative_bounds(first, last)
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
                    "Source tangency Jacobians require native surface Hessians."
                )
            hessian_lo, hessian_hi = hessians.evaluate(lo[None, 1:], hi[None, 1:])
            curvature = curve.derivative_bounds(first, last, order=2)
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
                normal_derivative = interval_add(
                    _vector_interval_cross(u_derivative, v),
                    _vector_interval_cross(u, v_derivative),
                )
                bounds = _vector_interval_dot(tangent, normal_derivative)
                matrix_lo[2, column + 1], matrix_hi[2, column + 1] = bounds
            lows.append(matrix_lo)
            highs.append(matrix_hi)
        return np.asarray(lows), np.asarray(highs)


class IntervalEvaluator(Protocol):
    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]: ...


class KrawczykEvaluator(Protocol):
    @property
    def value(self) -> IntervalEvaluator: ...

    @property
    def jacobian(self) -> IntervalEvaluator: ...

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray: ...


@dataclass(frozen=True)
class _Prepared:
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

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
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


_PAIR_PREPARATION_CACHE: dict[tuple[str, int], _Prepared] = {}
_SOURCE_PREPARATION_CACHE: dict[tuple[str, str, int, int], _Prepared] = {}


class _AxialRevolutionEvaluator(AbstractSurfacePatch):
    """Exact original-angle harmonic law of an on-axis native circle sweep."""

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
            raise TypeError("An axial revolution evaluator requires its original circle.")
        axis = np.asarray(source.axis_direction)
        active = np.flatnonzero(axis)
        if active.size != 1 or abs(axis[active[0]]) != 1.0:
            raise ValueError("Exact harmonic lowering requires a signed coordinate axis.")
        index = int(active[0])
        if any(
            np.asarray(source.curve.center)[i] != np.asarray(source.axis_origin)[i]
            for i in range(3)
            if i != index
        ):
            raise ValueError(
                "The source circle center must lie exactly on its rotation axis."
            )
        self.source, self.profile = source, source.curve
        first, second = (
            np.asarray(source.curve.first_axis),
            np.asarray(source.curve.second_axis),
        )
        radial_first, radial_second = first.copy(), second.copy()
        radial_first[index] = radial_second[index] = 0.0
        axial_first, axial_second = np.zeros(3), np.zeros(3)
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

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        return self.source.bounding_box(parameter_box)

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        return self.source.validate_parameter_box(parameter_box)

    def degenerate_isolines(
        self, parameter_box: ConvertibleToArray, /
    ) -> tuple[tuple[int, float], ...]:
        return self.source.degenerate_isolines(parameter_box)


def _canonical_surface_evaluator(surface: SurfaceEvaluator, /) -> SurfaceEvaluator:
    """Lower a proved Rodrigues law at its ORIGINAL (u, source-t).

    This is evaluator simplification only. Regions, topology, source records,
    parameter domains and native period declarations remain authoritative.
    """
    from fractions import Fraction as F

    if not isinstance(surface, RevolutionSurface) or not isinstance(
        surface.curve, CircleCurve
    ):
        return surface
    profile = surface.curve
    axis = tuple(F(float(value)) for value in np.asarray(surface.axis_direction))
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
    first = tuple(F(float(value)) for value in np.asarray(profile.first_axis))
    second = tuple(F(float(value)) for value in np.asarray(profile.second_axis))
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
    # No unit-frame/angle inference: keep every ORIGINAL float coefficient.
    # Axial terms are collected before evaluating u, removing the artificial
    # dependency in cos(u)*axial + (1-cos(u))*axial exactly.
    return _AxialRevolutionEvaluator(surface)


def _pose_cancelled_pair(
    system: SurfacePairSystem, /
) -> tuple[SurfacePairSystem, np.ndarray] | None:
    """Exact common-pose cancellation of a placed plane/plane equation.

    ``(R f(p) + t) - (R g(q) + t) = R (f(p) - g(q))`` holds in the authored
    operation tree, and ``same_source_pose`` proves one identical ``R, t``
    with positive determinant, so both equations have one zero set. Only the
    equations are prepared in the shared source frame: regions, curves, roots
    and point maps keep the original placed carriers. The returned rotation
    maps source-frame residual bounds back to world bounds.
    """
    first, second = system.first, system.second
    if not (isinstance(first, PlacedSurface) and isinstance(second, PlacedSurface)):
        return None
    if not (
        isinstance(first.definition, PlanePatch)
        and isinstance(second.definition, PlanePatch)
    ):
        return None
    if not same_source_pose(first, second):
        return None
    return SurfacePairSystem(first.definition, second.definition), np.asarray(
        first.rotation
    )


def _source_frame_scales(scales: _Scales, rotation: np.ndarray, /) -> _Scales:
    """World residual tolerances, rounded inward through the pose operator bound."""
    from ._correspondence import placed_deviation_bound

    norm = placed_deviation_bound(rotation, 1.0)
    return replace(
        scales,
        coincidence=float(np.nextafter(scales.coincidence / norm, 0.0)),
        tangency=float(np.nextafter(scales.tangency / norm, 0.0)),
    )


def _prepare(system: _System, dimension: int, batch: int, /) -> _Prepared:
    if isinstance(system, SurfacePairSystem):
        cancelled = _pose_cancelled_pair(system)
        if cancelled is not None:
            system = cancelled[0]
        first, second = (
            _canonical_surface_evaluator(system.first),
            _canonical_surface_evaluator(system.second),
        )
        if first is not system.first or second is not system.second:
            system = SurfacePairSystem(first, second)
    elif isinstance(system, _CurveSurfaceSystem):
        surface = _canonical_surface_evaluator(system.surface)
        if surface is not system.surface:
            system = _CurveSurfaceSystem(system.curve, surface)
    elif isinstance(system, _CurveSurfaceTangencySystem):
        surface = _canonical_surface_evaluator(system.surface)
        if surface is not system.surface:
            system = _CurveSurfaceTangencySystem(system.curve, surface)
    elif isinstance(system, _TripleSurfaceSystem):
        first = _canonical_surface_evaluator(system.first)
        second = _canonical_surface_evaluator(system.second)
        third = _canonical_surface_evaluator(system.third)
        if (
            first is not system.first
            or second is not system.second
            or third is not system.third
        ):
            system = _TripleSurfaceSystem(first, second, third)
    pair_key = None
    if isinstance(system, SurfacePairSystem):
        pair_key = (canonical_fingerprint(array_tree_fingerprint(system)), batch)
        cached = _PAIR_PREPARATION_CACHE.get(pair_key)
        if cached is not None:
            return cached
    capability_system = (
        isinstance(system, (_CurveSurfaceSystem, _CurveSurfaceTangencySystem))
        and isinstance(system.curve, IntersectionCurve)
    ) or (
        isinstance(
            system, (_CurvePairSystem, _CurveProximitySystem, _PlanarCurveTangencySystem)
        )
        and (
            isinstance(system.first, (AbstractTrimCurve, IntersectionCurve))
            or isinstance(system.second, (AbstractTrimCurve, IntersectionCurve))
        )
    )
    if capability_system:
        # Close over source definitions before JAX tracing. Passing them as a
        # dynamic module tree would trace host chart/span identities as arrays.
        key = (
            str(jax.tree_util.tree_structure(system)),
            canonical_fingerprint(array_tree_fingerprint(system)),
            dimension,
            batch,
        )
        cached = _SOURCE_PREPARATION_CACHE.get(key)
        if cached is not None:
            return cached
        value: _IntersectionSurfaceIntervals | _CurveCapabilityIntervals
        jacobian: _IntersectionSurfaceIntervals | _CurveCapabilityIntervals
        if isinstance(system, (_CurveSurfaceSystem, _CurveSurfaceTangencySystem)):
            value = _IntersectionSurfaceIntervals(system, False)
            jacobian = _IntersectionSurfaceIntervals(system, True)
        elif isinstance(
            system, (_CurvePairSystem, _CurveProximitySystem, _PlanarCurveTangencySystem)
        ):
            value = _CurveCapabilityIntervals(system, False)
            jacobian = _CurveCapabilityIntervals(system, True)
        else:
            raise TypeError("The source capability system has no native interval owner.")
        prepared = _Prepared(
            system,
            value,
            jacobian,
            dimension,
            batch,
            jax.jit(jax.vmap(system.residual)),
            jax.jit(jax.vmap(jax.jacfwd(system.residual))),
            eqx.filter_jit(lambda points, plan: _newton_values(system, points, plan)),
        )
        if len(_SOURCE_PREPARATION_CACHE) >= 128:
            del _SOURCE_PREPARATION_CACHE[next(iter(_SOURCE_PREPARATION_CACHE))]
        _SOURCE_PREPARATION_CACHE[key] = prepared
        return prepared
    coefficients = coefficient_enclosures(system)
    prepared = _Prepared(
        system,
        prepare_interval_function(
            system.residual, dimension, batch_capacity=batch, constant_bounds=coefficients
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
    if pair_key is not None:
        if len(_PAIR_PREPARATION_CACHE) >= 128:
            del _PAIR_PREPARATION_CACHE[next(iter(_PAIR_PREPARATION_CACHE))]
        _PAIR_PREPARATION_CACHE[pair_key] = prepared
    return prepared


def _approximate_inverses(matrices: np.ndarray, /) -> np.ndarray:
    """Floating preconditioners for Krawczyk; any matrix keeps the test sound."""
    dimension = matrices.shape[-1]
    count = matrices.shape[0]
    batch = 64
    parts = []
    for start in range(0, count, batch):
        chunk = matrices[start : start + batch]
        padding = batch - chunk.shape[0]
        padded = np.concatenate(
            (chunk, np.repeat(np.eye(dimension)[None], padding, axis=0))
        )
        if dimension <= 4:
            inverse = _inverses(jnp.asarray(padded), SmallLinearSolvePlan(dimension))
        else:
            inverse = _dense_inverses(jnp.asarray(padded))
        parts.append(np.asarray(inverse)[: chunk.shape[0]])
    inverses = np.concatenate(parts) if parts else np.zeros_like(matrices)
    return np.where(np.isfinite(inverses), inverses, 0.0)


@dataclass(frozen=True)
class _Krawczyk:
    certified: np.ndarray
    excluded: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    preconditioner: np.ndarray
    contraction: np.ndarray


def _krawczyk(
    prepared: KrawczykEvaluator,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
    *,
    parameter_axis: int | None = None,
) -> _Krawczyk:
    """(Parametric) Krawczyk inclusion over batched boxes.

    Without a parameter axis the system is square; with one, the chart axis is
    an interval parameter and inclusion certifies a unique graph point for every
    parameter value.
    """
    dimension = lower.shape[1]
    free = np.asarray(
        [axis for axis in range(dimension) if axis != parameter_axis], dtype=np.int64
    )
    center = 0.5 * (lower + upper)
    center_lower = center.copy()
    center_upper = center.copy()
    if parameter_axis is not None:
        center_lower[:, parameter_axis] = lower[:, parameter_axis]
        center_upper[:, parameter_axis] = upper[:, parameter_axis]
    value_lower, value_upper = prepared.value.evaluate(center_lower, center_upper)
    jacobian_lower, jacobian_upper = prepared.jacobian.evaluate(lower, upper)
    jacobian_lower = jacobian_lower[:, :, free]
    jacobian_upper = jacobian_upper[:, :, free]
    point_jacobian = prepared.point_jacobians(center)[:, :, free]
    preconditioner = _approximate_inverses(point_jacobian)
    shift_lower, shift_upper = _point_times_interval(
        preconditioner, value_lower[..., None], value_upper[..., None]
    )
    product_lower, product_upper = _point_times_interval(
        preconditioner, jacobian_lower, jacobian_upper
    )
    identity = np.eye(free.shape[0])[None]
    residual_lower, residual_upper = _round(
        identity - product_upper, identity - product_lower
    )
    offset_lower = (lower - center)[:, free, None]
    offset_upper = (upper - center)[:, free, None]
    spread_lower, spread_upper = _interval_times_interval(
        residual_lower, residual_upper, offset_lower, offset_upper
    )
    base = center[:, free]
    result_lower, result_upper = _round(
        base - shift_upper[..., 0] + spread_lower[..., 0],
        base - shift_lower[..., 0] + spread_upper[..., 0],
    )
    finite = np.all(np.isfinite(result_lower) & np.isfinite(result_upper), axis=1)
    box_lower = lower[:, free]
    box_upper = upper[:, free]
    certified = (
        finite
        & np.all(result_lower > box_lower, axis=1)
        & np.all(result_upper < box_upper, axis=1)
    )
    excluded = finite & np.any(
        (result_upper < box_lower) | (result_lower > box_upper), axis=1
    )
    contraction = np.max(
        np.nextafter(
            np.sum(_magnitude(residual_lower, residual_upper), axis=2)
            * (1.0 + 8.0 * _UNIT),
            np.inf,
        ),
        axis=1,
    )
    certified &= contraction < 1.0
    return _Krawczyk(
        certified,
        excluded,
        np.where(finite[:, None], result_lower, box_lower),
        np.where(finite[:, None], result_upper, box_upper),
        preconditioner,
        contraction,
    )


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

    system: _Prepared
    point: Callable[[np.ndarray], np.ndarray]
    pair: _Prepared | None
    layout: _Layout | None
    tangency: _Prepared | None
    proximity: bool
    excluded: tuple[np.ndarray, np.ndarray] | None = None
    junction: np.ndarray | None = None


def _image_size(
    jacobian_lower: np.ndarray, jacobian_upper: np.ndarray, widths: np.ndarray, /
) -> np.ndarray:
    return np.max(
        np.sum(_magnitude(jacobian_lower, jacobian_upper) * widths[:, None, :], axis=2),
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
    if isinstance(system, _CurveSurfaceSystem):
        target_evaluator = system.surface
    elif isinstance(system, (_CurvePairSystem, _CurveProximitySystem, SurfacePairSystem)):
        target_evaluator = system.second if sign > 0 else system.first
    elif isinstance(system, _FaceSystem):
        target_evaluator = system.pair.second if sign > 0 else system.pair.first
    else:
        return none, none, np.full((count,), np.inf)
    if not isinstance(target_evaluator, (LineCurve, PlanePatch)):
        return none, none, np.full((count,), np.inf)
    center = 0.5 * (lower + upper) if anchor is None else anchor
    probe_lower = lower.copy()
    probe_upper = upper.copy()
    probe_lower[:, target_] = center[:, target_]
    probe_upper[:, target_] = center[:, target_]
    gap_lower, gap_upper = pair.value.evaluate(probe_lower, probe_upper)
    if sign < 0.0:
        gap_lower, gap_upper = -gap_upper, -gap_lower
    jacobian = pair.point_jacobians(center)
    frame = -sign * jacobian[:, :, target_]
    tangents = jacobian[:, :, np.asarray(source)]
    if ambient == 2:
        normal = np.stack((-frame[:, 1, 0], frame[:, 0, 0]), axis=1)
    else:
        normal = np.cross(frame[:, :, 0], frame[:, :, 1])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    regular = length[:, 0] > 0.0
    normal = normal / np.where(length > 0.0, length, 1.0)
    distance_lower, distance_upper = _point_times_interval(
        normal[:, None, :], gap_lower[..., None], gap_upper[..., None]
    )
    distance = _magnitude(distance_lower, distance_upper)[:, 0, 0]
    gram = np.swapaxes(frame, 1, 2) @ frame
    projector = _approximate_inverses(gram) @ np.swapaxes(frame, 1, 2)
    step_lower, step_upper = _point_times_interval(
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
) -> None:
    """Record a certified root unless its uniqueness box already holds one."""
    for existing_lower, existing_upper in isolated.root_boxes:
        if np.all((point >= existing_lower) & (point <= existing_upper)):
            return
    isolated.root_boxes.append(unique)
    pair = problem.pair if problem.pair is not None else problem.system
    value_lower, value_upper = pair.value.evaluate(tight[0][None], tight[1][None])
    gap = float(np.linalg.norm(_magnitude(value_lower, value_upper)[0]))
    if problem.proximity and gap > tolerance:
        # A certified critical point of the operand distance with a positive gap
        # is not an intersection; its box is still accounted as decided.
        return
    isolated.roots.append(
        ParametricIntersectionPoint(
            kind="transversal",
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
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate Krawczyk on certified boxes; every iterate still holds the root."""
    if (
        isinstance(maximum_steps, bool)
        or not isinstance(maximum_steps, int)
        or maximum_steps < 0
    ):
        raise ValueError("Refinement steps must be a nonnegative integer.")
    for _ in range(maximum_steps):
        result = _krawczyk(prepared, lower, upper)
        shrink = result.certified[:, None]
        new_lower = np.where(shrink, np.maximum(lower, result.lower), lower)
        new_upper = np.where(shrink, np.minimum(upper, result.upper), upper)
        if np.array_equal(new_lower, lower) and np.array_equal(new_upper, upper):
            break
        lower, upper = new_lower, new_upper
    return lower, upper


def _certify_near(
    prepared: _Prepared,
    points: np.ndarray,
    widths: np.ndarray,
    /,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """epsilon-inflation Krawczyk around converged Newton points.

    Returns the certified mask, the uniqueness boxes and contracted enclosures.
    """
    count = points.shape[0]
    certified = np.zeros((count,), dtype=np.bool_)
    lower = points.copy()
    upper = points.copy()
    seed = _krawczyk(prepared, points, points)
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
        result = _krawczyk(prepared, trial_lower, trial_upper)
        rows = np.flatnonzero(pending)[result.certified]
        certified[rows] = True
        lower[rows] = trial_lower[result.certified]
        upper[rows] = trial_upper[result.certified]
    if not np.any(certified):
        empty = (lower[certified], upper[certified])
        return certified, empty, empty
    tight = _contract(prepared, lower[certified], upper[certified])
    return certified, (lower[certified], upper[certified]), tight


def _tangency_probe(
    problem: _Problem,
    isolated: _Isolated,
    centers: np.ndarray,
    widths: np.ndarray,
    tolerance: float,
    /,
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
    certified, unique, tight = _certify_near(tangency, roots[usable], widths[usable])
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
        gap_lower, gap_upper = pair.value.evaluate(tight_lower[None], tight_upper[None])
        gap = float(np.linalg.norm(_magnitude(gap_lower, gap_upper)[0]))
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
                kind="singular" if _operand_degenerate(problem, point) else "tangent",
                certificate="tangency-krawczyk",
                parameters=representative,
                parameter_lower=tight_lower,
                parameter_upper=tight_upper,
                point=problem.point(representative),
                gap_bound=gap,
                condition_estimate=float(estimate),
            )
        )


def _operand_degenerate(problem: _Problem, point: np.ndarray, /) -> bool:
    """Whether an operand's parametrization loses rank at ``point``."""
    pair = problem.pair if problem.pair is not None else problem.system
    layout = problem.layout
    if layout is None:
        return False
    jacobian = pair.point_jacobians(point[None])[0]
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
) -> tuple[np.ndarray, np.ndarray]:
    prepared = problem.system
    value_lower, value_upper = prepared.value.evaluate(lower, upper)
    alive = _contains_zero(value_lower, value_upper)
    if problem.pair is not None and problem.proximity:
        gap_lower, gap_upper = problem.pair.value.evaluate(lower, upper)
        alive &= np.all(
            (gap_lower <= scales.tangency) & (gap_upper >= -scales.tangency), axis=1
        )
    jacobian_lower, jacobian_upper = prepared.jacobian.evaluate(lower, upper)
    center = 0.5 * (lower + upper)
    center_lower, center_upper = prepared.value.evaluate(center, center)
    spread_lower, spread_upper = _interval_times_interval(
        jacobian_lower,
        jacobian_upper,
        (lower - center)[..., None],
        (upper - center)[..., None],
    )
    centered_lower, centered_upper = _round(
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
    krawczyk = _krawczyk(prepared, lower, upper)
    budget.excluded += int(np.sum(krawczyk.excluded))
    certified_rows = np.flatnonzero(krawczyk.certified)
    budget.certified += certified_rows.shape[0]
    if certified_rows.shape[0]:
        unique = (lower[certified_rows], upper[certified_rows])
        tight = _contract(
            prepared, krawczyk.lower[certified_rows], krawczyk.upper[certified_rows]
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
            )
    undecided = ~krawczyk.certified & ~krawczyk.excluded
    # Krawczyk contraction: intersect with the image when it shrinks the box.
    contracted_lower = np.maximum(lower, krawczyk.lower)
    contracted_upper = np.minimum(upper, krawczyk.upper)
    lower = np.where(undecided[:, None], contracted_lower, lower)[undecided]
    upper = np.where(undecided[:, None], contracted_upper, upper)[undecided]
    size = size[undecided]
    if lower.shape[0] == 0:
        return lower, upper
    full, partial, distance = _coincidence(problem, lower, upper, scales.coincidence)
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
        _probe(problem, isolated, lower[small], upper[small], newton_plan, scales)
        keep = ~_inside_any(lower, upper, isolated.root_boxes)
        lower, upper, size = lower[keep], upper[keep], size[keep]
    leaf = size <= scales.resolution
    for low, high in zip(lower[leaf], upper[leaf], strict=True):
        isolated.unresolved.append(UnresolvedParameterRegion(low, high, "resolution"))
    budget.unresolved += int(np.sum(leaf))
    lower, upper = lower[~leaf], upper[~leaf]
    if lower.shape[0] == 0:
        return lower, upper
    jacobian_lower, jacobian_upper = prepared.jacobian.evaluate(lower, upper)
    gap_rows = _gap_rows(problem)
    contribution = np.max(
        _magnitude(jacobian_lower[:, :gap_rows], jacobian_upper[:, :gap_rows]), axis=1
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
            problem.system, roots[usable], widths[usable]
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
                )
    # Newton failed or its limit is not a certifiable regular root: probe for a
    # certified tangential contact instead.
    singular = ~regular
    if np.any(singular):
        _tangency_probe(
            problem, isolated, center[singular], widths[singular], scales.tangency
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
def _evaluate_curves(evaluator: CurveEvaluator, parameters: Array) -> Array:
    return jax.vmap(evaluator.evaluate)(parameters)


def _curve_point_map(
    curve: CurveEvaluator | IntersectionCurve, /
) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(curve, IntersectionCurve):
        return lambda parameters: np.asarray(
            curve.evaluate(jnp.asarray(parameters[0])).point
        )
    if isinstance(curve, AbstractTrimCurve):
        # Chart identities are host preparation, not dynamic JIT arguments.
        return lambda parameters: np.asarray(curve.evaluate(jnp.asarray(parameters[0])))
    return lambda parameters: np.asarray(
        _evaluate_curves(curve, jnp.asarray(parameters[:1]))
    )[0]


def intersect_trim_curves(
    first: AbstractTrimCurve,
    second: AbstractTrimCurve,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
) -> CurveIntersectionResult:
    """Bounded trim/trim events in the operands' actual parameter intervals."""
    if not isinstance(first, AbstractTrimCurve) or not isinstance(
        second, AbstractTrimCurve
    ):
        raise TypeError("Trim intersections require two native AbstractTrimCurve values.")
    return intersect_curve_ranges(CurveRange(first), CurveRange(second), policy=policy)


def _exact_unit_axes(axes: Sequence[np.ndarray]) -> bool:
    from fractions import Fraction

    vectors = [[Fraction(float(value)) for value in axis] for axis in axes]
    return all(
        sum(a * b for a, b in zip(vectors[i], vectors[j], strict=True))
        == (1 if i == j else 0)
        for i in range(len(vectors))
        for j in range(len(vectors))
    )


def _analytic_point(
    parameters: np.ndarray,
    point: np.ndarray,
    gap: float,
    kind: ParametricIntersectionKind = "tangent",
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
    )


def _analytic_line_circle(
    first: CurveRange, second: CurveRange
) -> CurveIntersectionResult | None:
    from fractions import Fraction as F

    if isinstance(first.curve, LineCurve) and isinstance(second.curve, CircleCurve):
        line_range, circle_range, reversed_ = first, second, False
    elif isinstance(second.curve, LineCurve) and isinstance(first.curve, CircleCurve):
        line_range, circle_range, reversed_ = second, first, True
    else:
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
    first: CurveRange, second: CurveRange, /
) -> CurveIntersectionResult | None:
    """Prove a tangent contact of two complete coplanar exact circles."""
    from fractions import Fraction as F

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
    exact_axes = tuple(tuple(F(float(value)) for value in axis) for axis in axes)
    normal = (
        exact_axes[0][1] * exact_axes[1][2] - exact_axes[0][2] * exact_axes[1][1],
        exact_axes[0][2] * exact_axes[1][0] - exact_axes[0][0] * exact_axes[1][2],
        exact_axes[0][0] * exact_axes[1][1] - exact_axes[0][1] * exact_axes[1][0],
    )
    centers = tuple(
        tuple(F(float(value)) for value in np.asarray(curve.center)) for curve in (a, b)
    )
    delta = tuple(y - x for x, y in zip(*centers, strict=True))
    if sum(n * d for n, d in zip(normal, delta, strict=True)) != 0 or any(
        sum(n * value for n, value in zip(normal, axis, strict=True)) != 0
        for axis in exact_axes[2:]
    ):
        return None
    length_squared = sum(value * value for value in delta)
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
    first: SurfaceRegion, second: SurfaceRegion
) -> SurfaceIntersectionResult | None:
    from fractions import Fraction as F

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
    singular = False
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
        singular |= coordinates[0] == 0 and coordinates[1] == 0
        parameters.extend((longitude, latitude))
    point = np.asarray([float(value) for value in exact_point])
    gap = 256 * _UNIT * (1 + np.linalg.norm(point) + float(a.radius) + float(b.radius))
    budget = _Budget(1)
    budget.processed = budget.certified = 1
    event = _analytic_point(
        np.asarray(parameters), point, gap, "singular" if singular else "tangent"
    )
    return SurfaceIntersectionResult((), (event,), (), (), budget.work())


def _analytic_plane_sphere_contact(
    first: SurfaceRegion, second: SurfaceRegion
) -> SurfaceIntersectionResult | None:
    """Exact represented quadric support equality, not a gap-only tangency."""
    from fractions import Fraction as F

    if isinstance(first.patch, PlanePatch) and isinstance(second.patch, SpherePatch):
        plane_region, sphere_region, reverse = first, second, False
    elif isinstance(second.patch, PlanePatch) and isinstance(first.patch, SpherePatch):
        plane_region, sphere_region, reverse = second, first, True
    else:
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
    event = _analytic_point(
        parameters, world, gap, "singular" if unit[0] == 0 and unit[1] == 0 else "tangent"
    )
    work.certified = 1
    return SurfaceIntersectionResult((), (event,), (), (), work.work())


def intersect_curve_ranges(
    first: CurveRange,
    second: CurveRange,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
) -> CurveIntersectionResult:
    """Intersect two exact curves of equal dimension (planar p-curves or space curves).

    Planar events are certified roots of ``c1(s) = c2(t)``. Space curves meet
    only non-generically, so their events are certified distance-critical points
    whose gap is at most the policy's tangency tolerance.
    """
    policy_ = ParametricIntersectionPolicy() if policy is None else policy
    if not isinstance(first, CurveRange) or not isinstance(second, CurveRange):
        raise TypeError("intersect_curves requires two CurveRange operands.")
    analytic = _analytic_line_circle(first, second)
    if analytic is not None:
        return analytic
    analytic = _analytic_circle_circle(first, second)
    if analytic is not None:
        return analytic
    ambient = first.ambient_dimension
    if second.ambient_dimension != ambient or ambient not in (2, 3):
        raise ValueError("Curve operands must share dimension two or three.")
    first_pieces = curve_pieces(first)
    second_pieces = curve_pieces(second)
    boxes = [
        np.asarray(first.curve.bounding_box(first.first, first.last)),
        np.asarray(second.curve.bounding_box(second.first, second.last)),
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
            pair = _prepare(
                _CurvePairSystem(piece_a.evaluator, piece_b.evaluator), 2, policy_.batch
            )
            point_map = _curve_point_map(piece_a.evaluator)
            if ambient == 2:
                problem = _Problem(
                    pair,
                    point_map,
                    pair,
                    _Layout((0,), (1,), 2),
                    _prepare(
                        _PlanarCurveTangencySystem(piece_a.evaluator, piece_b.evaluator),
                        2,
                        policy_.batch,
                    ),
                    False,
                )
            else:
                problem = _Problem(
                    _prepare(
                        _CurveProximitySystem(piece_a.evaluator, piece_b.evaluator),
                        2,
                        policy_.batch,
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


def intersect_curve_region(
    curve: CurveRange,
    surface: SurfaceRegion,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
) -> CurveIntersectionResult:
    """Intersect an exact space curve with an exact surface region."""
    policy_ = ParametricIntersectionPolicy() if policy is None else policy
    if not isinstance(curve, CurveRange) or not isinstance(surface, SurfaceRegion):
        raise TypeError("intersect_curve_surface requires CurveRange and SurfaceRegion.")
    if curve.ambient_dimension != 3:
        raise ValueError("Curve/surface intersection requires a space curve.")
    boxes = [
        np.asarray(curve.curve.bounding_box(curve.first, curve.last)),
        np.asarray(surface.patch.bounding_box(surface.parameter_box)),
    ]
    scales = _scales(policy_, _scale(boxes))
    budget = _Budget(policy_.maximum_boxes)
    points: list[ParametricIntersectionPoint] = []
    cells: list[tuple[np.ndarray, np.ndarray, bool]] = []
    unresolved: list[UnresolvedParameterRegion] = []
    bound = 0.0
    for piece_a in curve_pieces(curve):
        for piece_b in surface_pieces(surface):
            pair = _prepare(
                _CurveSurfaceSystem(piece_a.evaluator, piece_b.evaluator),
                3,
                policy_.batch,
            )
            point_map = _curve_point_map(piece_a.evaluator)
            problem = _Problem(
                pair,
                point_map,
                pair,
                _Layout((0,), (1, 2), 3),
                _prepare(
                    _CurveSurfaceTangencySystem(piece_a.evaluator, piece_b.evaluator),
                    3,
                    policy_.batch,
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


@dataclass
class _Chart:
    axis: int
    start: np.ndarray
    end: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    preconditioner: np.ndarray
    contraction: float
    certified: bool


type _Endpoint = tuple[IntersectionEndpointKind, tuple[int, int] | None, np.ndarray]


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
    pair: _Prepared
    lower: np.ndarray
    upper: np.ndarray
    pieces: tuple[int, int]

    @property
    def system(self) -> SurfacePairSystem:
        system = self.pair.system
        if not isinstance(system, SurfacePairSystem):
            raise TypeError(
                "A surface continuation context requires its canonical surface pair."
            )
        return system


def _certify_chart(
    context: _PairContext, start: np.ndarray, end: np.ndarray, scale: float, /
) -> _Chart:
    """Parametric Krawczyk chart for the branch arc between two nodes."""
    jacobian = context.pair.point_jacobians(start[None])[0]
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
        result = _krawczyk(context.pair, lower[None], upper[None], parameter_axis=axis)
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
    return _Chart(axis, start, end, lower, upper, np.eye(3), 1.0, False)


def _surface_discovery(
    context: _PairContext,
    scales: _Scales,
    budget: _Budget,
    batch: int,
    point_map: Callable[[np.ndarray], np.ndarray],
    junction_radius: np.ndarray,
    /,
) -> tuple[
    list[_Start], list[_Start], list[_Start], list[ParametricIntersectionPoint], _Isolated
]:
    """Certified boundary, turning and junction starts and contacts of one pair."""
    boundary: list[_Start] = []
    collected = _Isolated()
    pair_system = context.system
    layout = _Layout((0, 1), (2, 3), 3)
    pair_problem = _Problem(context.pair, point_map, context.pair, layout, None, False)
    cells, bound = _coincident_cells(pair_problem, context, scales, budget, batch)
    collected.coincident.extend(cells)
    collected.coincidence_bound = bound
    cell_lower = np.asarray([cell[0] for cell in cells]).reshape((-1, 4))
    cell_upper = np.asarray([cell[1] for cell in cells]).reshape((-1, 4))
    for axis in range(4):
        for side, value in enumerate((context.lower[axis], context.upper[axis])):
            face = _FaceSystem(pair_system, axis, jnp.asarray(value))
            free = [index for index in range(4) if index != axis]
            prepared = _prepare(face, 3, batch)

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
                problem, context.lower[free], context.upper[free], scales, budget, batch
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
                if _certified_transversal(context.pair, cell[0], cell[1]):
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
        _prepare(
            _TurningSystem(pair_system, jnp.asarray(_TURNING_DIRECTION / widths)),
            4,
            batch,
        ),
        point_map,
        context.pair,
        layout,
        _prepare(_SurfaceTangencySystem(pair_system), 4, batch),
        False,
        excluded=(cell_lower, cell_upper),
        junction=junction_radius,
    )
    isolated = _isolate(problem, context.lower, context.upper, scales, budget, batch)
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

    pair: SurfacePairSystem
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
) -> tuple[list[ParametricIntersectionPoint], list[UnresolvedParameterRegion]]:
    """Isolate every branch leaving a certified contact's neighborhood."""
    system = _JunctionSystem(
        context.system, jnp.asarray(center), jnp.asarray(1.0 / radius)
    )
    prepared = _prepare(system, 4, batch)
    problem = _Problem(prepared, point_map, context.pair, None, None, False)
    lower = np.maximum(center - 1.25 * radius, context.lower)
    upper = np.minimum(center + 1.25 * radius, context.upper)
    isolated = _isolate(problem, lower, upper, scales, budget, batch)
    return isolated.roots, isolated.unresolved


@eqx.filter_jit
def _foot_points(
    pair: SurfacePairSystem, anchors: Array, plan: VectorLocalRootPlan
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
            problem, lower, upper, scales.coincidence, anchor=anchor
        )
        jacobian_lower, jacobian_upper = problem.system.jacobian.evaluate(lower, upper)
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
    """Intersect chart walls with exact native support equations."""
    pair = context.system
    regions = (
        SurfaceRegion(pair.first, np.stack((context.lower[:2], context.upper[:2]))),
        SurfaceRegion(pair.second, np.stack((context.lower[2:], context.upper[2:]))),
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
) -> tuple[_Chart, _Start] | None:
    """Certify a final chart onto the start point nearest ``target``."""
    widths = context.upper - context.lower
    index = _nearest_start(candidates, target, widths, 1.5 * span + 1.0e-9)
    if index is None:
        return None
    chart = _certify_chart(context, point, candidates[index].point, scale)
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
                chart = _certify_chart(context, point, origin, scale)
                if chart.certified:
                    charts.append(chart)
                    return charts, ("closed", None, origin)
        if accepted and _entered_junction(starts, root):
            landed = _land(context, point, starts.junctions, root, span, scale)
            if landed is not None:
                charts.append(landed[0])
                _consume(starts.interior, landed[0])
                return charts, ("singular", None, landed[1].point)
            accepted = False
        if accepted and np.any(_outside(context, root)):
            landed = _land(context, point, starts.boundary, root, span, scale)
            if landed is not None:
                charts.append(landed[0])
                _consume(starts.interior, landed[0])
                # Duplicate starts of the same branch end (several boundary
                # systems meet at a corner) lie in the same certified chart.
                _consume(starts.boundary, landed[0])
                return charts, ("boundary", landed[1].face, landed[1].point)
            accepted = False
        if accepted:
            chart = _certify_chart(context, point, root, scale)
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
            context, start.point, tangent, starts, False, policy, scale, budget
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
            context, start.point, tangent, starts, True, policy, scale, budget
        )
        if forward_end[0] == "closed":
            segments.append(
                _Segment(context.index, forward, forward_end, forward_end, closed=True)
            )
            continue
        backward, backward_end = _march(
            context, start.point, -tangent, starts, False, policy, scale, budget
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


def _glue(
    segments: list[_Segment],
    contexts: list[_PairContext],
    first: SurfaceRegion,
    second: SurfaceRegion,
    /,
) -> list[IntersectionCurve]:
    """Join segments across periodic seams and spline-span faces into branches.

    Segment endpoints on identified faces are linked regardless of marching
    direction; chains traverse segments forward or reversed accordingly.
    """
    region_lower = np.asarray((*first.lower, *second.lower))
    region_upper = np.asarray((*first.upper, *second.upper))
    periodic = (*first.periodic, *second.periodic)
    widths = region_upper - region_lower
    tolerance = 1.0e-8 * np.maximum(widths, 1.0)

    def endpoint(index: int, side: int) -> _Endpoint:
        segment = segments[index]
        return segment.start if side == 0 else segment.end

    def partners(index: int, side: int) -> list[np.ndarray]:
        kind, face, point = endpoint(index, side)
        if kind != "boundary" or face is None:
            return []
        axis, _ = face
        at_lower = abs(point[axis] - region_lower[axis]) <= tolerance[axis]
        at_upper = abs(point[axis] - region_upper[axis]) <= tolerance[axis]
        if not (at_lower or at_upper):
            return [point]
        if not periodic[axis]:
            return []
        shift = np.zeros((4,))
        shift[axis] = widths[axis] if at_lower else -widths[axis]
        return [point + shift]

    links: dict[tuple[int, int], tuple[int, int]] = {}
    for index, segment in enumerate(segments):
        if segment.closed:
            continue
        for side in (0, 1):
            if (index, side) in links:
                continue
            for candidate in partners(index, side):
                for other, following in enumerate(segments):
                    if following.closed:
                        continue
                    for other_side in (0, 1):
                        key = (other, other_side)
                        if key == (index, side) or key in links:
                            continue
                        kind, _, point = endpoint(other, other_side)
                        if kind == "boundary" and np.all(
                            np.abs(point - candidate) <= tolerance
                        ):
                            links[(index, side)] = key
                            links[key] = (index, side)
                            break
                    if (index, side) in links:
                        break
    curves: list[IntersectionCurve] = []
    visited: set[int] = set()
    heads = [
        index
        for index, segment in enumerate(segments)
        if segment.closed or (index, 0) not in links or (index, 1) not in links
    ]
    heads += list(range(len(segments)))
    for head in heads:
        if head in visited:
            continue
        closed = segments[head].closed
        forward = True
        if (head, 0) not in links:
            forward = True
        elif (head, 1) not in links:
            forward = False
        traversal: list[tuple[int, bool]] = []
        current: tuple[int, bool] | None = (head, forward)
        while current is not None:
            index, direction = current
            visited.add(index)
            traversal.append(current)
            if segments[index].closed:
                break
            exit_side = 1 if direction else 0
            link = links.get((index, exit_side))
            if link is None:
                current = None
            elif link[0] == head:
                closed = True
                current = None
            elif link[0] in visited:
                current = None
            else:
                current = (link[0], link[1] == 0)
        charts: list[_Chart] = []
        chart_pieces: list[tuple[int, int]] = []
        transitions: list[np.ndarray] = []
        node_axes: list[int] = []
        node_values: list[float] = []
        for index, direction in traversal:
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
                transitions.append(np.zeros((4,)))
                boundary = (
                    position == len(ordered) - 1
                    and exit_[0] == "boundary"
                    and exit_[1] is not None
                )
                axis = exit_[1][0] if boundary else chart.axis
                node_axes.append(axis)
                node_values.append(float(chart.end[axis]))
        if not charts:
            continue
        if closed and not segments[head].closed:
            transitions[-1] = charts[0].start - charts[-1].end
        first_index, first_direction = traversal[0]
        last_index, last_direction = traversal[-1]
        start_kind = (
            "closed" if closed else endpoint(first_index, 0 if first_direction else 1)[0]
        )
        end_kind = (
            "closed" if closed else endpoint(last_index, 1 if last_direction else 0)[0]
        )
        curves.append(
            IntersectionCurve(
                first,
                second,
                chart_axes=np.asarray([chart.axis for chart in charts], dtype=np.int32),
                chart_start=np.stack([chart.start for chart in charts]),
                chart_end=np.stack([chart.end for chart in charts]),
                chart_pieces=np.asarray(chart_pieces, dtype=np.int32),
                box_lower=np.stack([chart.lower for chart in charts]),
                box_upper=np.stack([chart.upper for chart in charts]),
                preconditioners=np.stack([chart.preconditioner for chart in charts]),
                contraction=np.asarray([chart.contraction for chart in charts]),
                certified=np.asarray([chart.certified for chart in charts]),
                transition_shifts=np.stack(transitions),
                closed=closed,
                start_kind=start_kind,
                end_kind=end_kind,
                node_axes=np.asarray(node_axes, dtype=np.int32),
                node_values=np.asarray(node_values, dtype=np.float64),
            )
        )
    return curves


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
def _evaluate_surface(evaluator: SurfaceEvaluator, parameters: Array) -> Array:
    return jax.vmap(evaluator.evaluate)(parameters)


def intersect_surface_regions(
    first: SurfaceRegion,
    second: SurfaceRegion,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
) -> SurfaceIntersectionResult:
    """Intersect two exact surface regions into certified branches and contacts."""
    policy_ = ParametricIntersectionPolicy() if policy is None else policy
    if not isinstance(first, SurfaceRegion) or not isinstance(second, SurfaceRegion):
        raise TypeError("intersect_surfaces requires two SurfaceRegion operands.")
    analytic = _analytic_sphere_contact(first, second)
    if analytic is not None:
        return analytic
    analytic = _analytic_plane_sphere_contact(first, second)
    if analytic is not None:
        return analytic
    boxes = [
        np.asarray(first.patch.bounding_box(first.parameter_box)),
        np.asarray(second.patch.bounding_box(second.parameter_box)),
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
    for piece_a in surface_pieces(first):
        for piece_b in surface_pieces(second):
            pair_system = SurfacePairSystem(piece_a.evaluator, piece_b.evaluator)
            cancelled = _pose_cancelled_pair(pair_system)
            pose = None if cancelled is None else cancelled[1]
            context = _PairContext(
                len(contexts),
                _prepare(pair_system, 4, policy_.batch),
                np.concatenate((piece_a.lower, piece_b.lower)),
                np.concatenate((piece_a.upper, piece_b.upper)),
                (piece_a.index, piece_b.index),
            )
            contexts.append(context)
            value_lower, value_upper = context.pair.value.evaluate(
                context.lower[None], context.upper[None]
            )
            budget.processed += 1
            if not bool(_contains_zero(value_lower, value_upper)[0]):
                budget.excluded += 1
                continue
            point_map = _surface_point_map(piece_a)
            radius = policy_.relative_junction * (context.upper - context.lower)
            boundary, interior, junctions, pair_contacts, collected = _surface_discovery(
                context,
                scales if pose is None else _source_frame_scales(scales, pose),
                budget,
                policy_.batch,
                point_map,
                radius,
            )
            if pose is not None:
                pair_contacts = [
                    _world_gap_contact(contact, pose) for contact in pair_contacts
                ]
                collected.coincidence_bound = _world_gap(
                    pose, collected.coincidence_bound
                )
            cells.extend(collected.coincident)
            bound = max(bound, collected.coincidence_bound)
            unresolved.extend(collected.unresolved)
            segments.extend(
                _trace(
                    context,
                    _Starts(boundary, interior, junctions, radius),
                    policy_,
                    scale,
                    budget,
                )
            )
            for segment in segments:
                if segment.pair_index != context.index:
                    continue
                if segment.start[0] == "uncertified" or segment.end[0] == "uncertified":
                    unresolved.append(
                        UnresolvedParameterRegion(
                            context.lower, context.upper, "continuation"
                        )
                    )
                    budget.unresolved += 1
            branched = {start.junction for start in junctions}
            contacts.extend(
                _junction_contact(contact, radius) if index in branched else contact
                for index, contact in enumerate(pair_contacts)
            )
    curves = _glue(segments, contexts, first, second)
    # A certified chart box holds exactly one solution piece, its branch; an
    # undecided leaf inside one (e.g. a wall contact of that very branch) has
    # no other solution and is decided by the traced branch.
    charts = [
        (curve.box_lower[chart], curve.box_upper[chart])
        for curve in curves
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
    points = contacts
    periods = tuple(
        region.upper[axis] - region.lower[axis] if region.periodic[axis] else None
        for region in (first, second)
        for axis in range(2)
    )
    return SurfaceIntersectionResult(
        curves,
        _dedupe_points(points, periods),
        _components(cells, bound),
        unresolved,
        budget.work(),
    )


def _certified_transversal(
    pair: _Prepared, lower: np.ndarray, upper: np.ndarray, /
) -> bool:
    """Outward interval proof that the 3x4 pair Jacobian has rank 3 on a cell.

    Some 3x3 minor whose interval determinant excludes zero certifies that the
    two surface normals are independent throughout the cell.
    """
    jacobian_lower, jacobian_upper = pair.jacobian.evaluate(lower[None], upper[None])
    low, high = np.asarray(jacobian_lower)[0], np.asarray(jacobian_upper)[0]
    if low.shape != (3, 4):
        return False

    def entry(row: int, column: int) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(low[row, column]), np.asarray(high[row, column])

    def minor(first: int, second: int) -> tuple[np.ndarray, np.ndarray]:
        # Rows 1 and 2 of columns ``first`` and ``second``.
        return interval_subtract(
            interval_multiply(entry(1, first), entry(2, second)),
            interval_multiply(entry(1, second), entry(2, first)),
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
    from ._correspondence import placed_deviation_bound

    return placed_deviation_bound(rotation, gap)


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
    )


def _junction_contact(
    contact: ParametricIntersectionPoint, radius: np.ndarray, /
) -> ParametricIntersectionPoint:
    """A contact where branches meet; its enclosure is the junction neighborhood."""
    return ParametricIntersectionPoint(
        kind="singular",
        certificate=contact.certificate,
        parameters=contact.parameters,
        parameter_lower=np.minimum(contact.parameter_lower, contact.parameters - radius),
        parameter_upper=np.maximum(contact.parameter_upper, contact.parameters + radius),
        point=contact.point,
        gap_bound=contact.gap_bound,
        condition_estimate=contact.condition_estimate,
    )


def _surface_point_map(piece: SurfacePiece, /) -> Callable[[np.ndarray], np.ndarray]:
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
