#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared, revision-bound meshing-domain view of parametric surface sources.

A `MeshingDomain` is a stratified complex of parametric surface patches
(`AbstractSurfacePatch`: analytic planes, cylinders, cones, spheres, tori,
revolutions, extrusions, ruled and rational B-spline surfaces) bounded by
declared curves and corners. It owns one invariant: every corner, curve,
surface and region identity, every incidence and every geometry query refers
to one source revision. It is a view, not a second geometry representation:
curves are the images of parameter-plane curves (``pcurve``) under their owning
patch, so a curve shared by several patches has exactly one physical definition
and every other use is validated against it at preparation.

Patch boundaries are loops of uses in the patch parameter plane: the outer loop
counterclockwise, holes clockwise. A `PatchCurveUse` traverses a declared curve;
a periodic seam is one curve used twice by the same patch, and a
`PatchPoleUse` is a parameter side collapsing to one corner (a pole or apex).
Each patch declares whether its oriented normal is ``d_u x d_v`` or its
opposite, and the uses of a curve shared by two patches must traverse it in
opposite oriented directions. Regions are declared closed unions of oriented
patches; a patch bounding two regions is an interface with an oriented region
pair.

Queries are batched: evaluation, oriented normals with regularity status and
closest-point projection (bounded Newton through `phydrax.nonlinear`) with
per-point convergence status. Parametric sources are exactly represented: the
patches are the authoritative geometry, evaluated in binary64.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import charge_native_geometry_queries, exact_orient2d
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import HermitianSpectrum
from ..nonlinear import VectorLocalRootPlan
from ..typing import parse
from ._atlas import (
    AbstractBoundaryMap,
    AbstractTrimCurve,
    BoundaryAtlas,
    CurveTrimLoop,
    overlapping_box_pairs,
    PolygonTrimLoop,
    TrimDomain,
    TrimLoop,
)
from ._chart_restriction import validate_chart_restrictions
from ._interval_enclosure import (
    _interval_tan,
    interval_add,
    interval_multiply,
    interval_subtract,
)
from .brep._intersection import (
    BranchRootEndpoint,
    NativePeriodEndpoint,
    RootEndpoint,
    TrimRootEndpoint,
)
from .brep._intersection_curve import (
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
    original_trim_intersection_preparation,
    pcurve_periodic_source,
)
from .brep._model import BRepGeometry, BRepModel
from .brep._patches import (
    _coordinate_budget,
    _meridian_frame,
    _meridian_normal_turns,
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    LineCurve,
    OffsetSurface,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    source_bernstein_restriction_scope,
    sphere_source_equivalence,
    sphere_source_pole_normal_limits,
    SpherePatch,
    surface_differential,
    SurfaceIsoparametricCurve,
    TorusPatch,
)
from .brep._placed import PlacedCurve, PlacedSurface, source_transform_bounds
from .brep._root_bindings import BRepPlacedVertex, BRepVertexRoot


if TYPE_CHECKING:
    from ..discretization._coordinate_enclosure import CoordinateEnclosureBudget
    from ._mesh_certificates import (
        MeshCertificateFinding,
        SourceBoundaryChartCover,
        SourceBoundaryDistance,
        SourceBoundarySamples,
    )
    from .brep._model import BRepEntityId


MeshingSourceAccuracy: TypeAlias = Literal[
    "exact_represented",
    "bounded_approximation",
    "certified_enclosure",
    "sampled_interpretation",
    "repaired_envelope",
]

type PatchBoundaryUse = PatchCurveUse | PatchPoleUse

_KINDS = {0: "corner", 1: "curve", 2: "surface", 3: "region"}
# Samples per curve use and pole side at which shared geometry is compared.
_CONSISTENCY_SAMPLES = 9
_PROJECTION_STEPS = 32
# Chart samples per boundary use when a loop is polygonized for membership.
_LOOP_SAMPLES = 65
# Query points compared with chart samples per nearest-seed block.
_SEED_BLOCK = 256


def _index(value: int, name: str, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return int(value)


def _finite(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _identifier(value: str, name: str, /) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@final
class PatchCurveUse(StrictModule, NonTrainableState):
    """One traversal of a declared domain curve along a patch boundary loop.

    ``pcurve`` is a two-dimensional `AbstractCurve` in the patch parameter
    plane, traversed from curve parameter ``first`` to ``last``. All uses of one
    curve share its parameterization: the curve's owner (its first use in
    patch/loop order) defines the physical curve and every other use must map
    each curve parameter to the same physical point.
    Canonical scalar trim roots and transported branch roots both retain their
    source definitions; nominal endpoint parameters must lie in their enclosed
    realizations and never replace the implicit junction identity.
    """

    curve: int = eqx.field(static=True)
    pcurve: AbstractCurve | AbstractTrimCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    first_root: RootEndpoint | None
    last_root: RootEndpoint | None
    start_vertex_root: BRepVertexRoot | None
    end_vertex_root: BRepVertexRoot | None
    trim_curve: AbstractTrimCurve | None

    def __init__(
        self,
        curve: int,
        pcurve: AbstractCurve | AbstractTrimCurve,
        first: float,
        last: float,
        /,
        *,
        first_root: RootEndpoint | None = None,
        last_root: RootEndpoint | None = None,
        start_vertex_root: BRepVertexRoot | None = None,
        end_vertex_root: BRepVertexRoot | None = None,
        trim_curve: AbstractTrimCurve | None = None,
    ) -> None:
        if not isinstance(pcurve, (AbstractCurve, AbstractTrimCurve)) or (
            isinstance(pcurve, AbstractCurve) and pcurve.ambient_dimension != 2
        ):
            raise TypeError("pcurve must be a two-dimensional native curve query.")
        first_ = _finite(first, "first")
        last_ = _finite(last, "last")
        if first_ == last_:
            raise ValueError("A curve use must traverse a nonempty parameter range.")
        for parameter, endpoint in ((first_, first_root), (last_, last_root)):
            if endpoint is not None:
                if not isinstance(
                    endpoint, (TrimRootEndpoint, BranchRootEndpoint, NativePeriodEndpoint)
                ):
                    raise TypeError(
                        "Root-valued coedges require canonical source RootEndpoint definitions."
                    )
                lower, upper = endpoint.parameter_enclosure()
                if not lower <= parameter <= upper:
                    raise ValueError(
                        "A realized coedge endpoint lies outside its implicit source root."
                    )
        for vertex in (start_vertex_root, end_vertex_root):
            if vertex is not None and not isinstance(vertex, BRepVertexRoot):
                raise TypeError(
                    "Root-valued coedge joins require BRepVertexRoot definitions."
                )
        if trim_curve is not None and not isinstance(trim_curve, AbstractTrimCurve):
            raise TypeError(
                "Canonical coedge trims must provide the exact native trim capability."
            )
        self.curve = _index(curve, "curve")
        self.pcurve = pcurve
        self.first = first_
        self.last = last_
        self.first_root, self.last_root = first_root, last_root
        self.start_vertex_root, self.end_vertex_root = start_vertex_root, end_vertex_root
        self.trim_curve = trim_curve

    def charts(self, parameters: np.ndarray, /) -> np.ndarray:
        """Parameter-plane points ``(k, 2)`` at curve parameters ``(k,)``."""

        values = np.asarray(parameters, dtype=np.float64).reshape((-1,))
        if not isinstance(self.pcurve, AbstractCurve):
            return np.asarray(
                self.pcurve.evaluate(jnp.asarray(values)), dtype=np.float64
            ).reshape((-1, 2))
        if not values.size:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(
            _curve_evaluation(self.pcurve, _bucketed(values)), dtype=np.float64
        ).reshape((-1, 2))[: values.size]


@final
class PatchPoleUse(StrictModule, NonTrainableState):
    """A straight parameter side from ``start`` to ``end`` collapsing to a corner.

    ``trim_curve`` retains the authoritative degenerate coedge trim, including
    native-period endpoint roots, so trim topology proves its joins exactly
    rather than through the literal binary side endpoints.
    """

    corner: int = eqx.field(static=True)
    start: tuple[float, float] = eqx.field(static=True)
    end: tuple[float, float] = eqx.field(static=True)
    trim_curve: AbstractTrimCurve | None

    def __init__(
        self,
        corner: int,
        start: tuple[float, float],
        end: tuple[float, float],
        /,
        *,
        trim_curve: AbstractTrimCurve | None = None,
    ) -> None:
        start_ = tuple(_finite(value, "start") for value in start)
        end_ = tuple(_finite(value, "end") for value in end)
        if len(start_) != 2 or len(end_) != 2 or start_ == end_:
            raise ValueError("A pole side joins two distinct parameter points.")
        if trim_curve is not None and not isinstance(trim_curve, AbstractTrimCurve):
            raise TypeError(
                "Canonical pole trims must provide the exact native trim capability."
            )
        self.corner = _index(corner, "corner")
        self.start = (start_[0], start_[1])
        self.end = (end_[0], end_[1])
        self.trim_curve = trim_curve

    def charts(self, fractions: np.ndarray, /) -> np.ndarray:
        """Parameter-plane points ``(k, 2)`` at side fractions ``(k,)`` in [0, 1]."""

        start = np.asarray(self.start, dtype=np.float64)
        end = np.asarray(self.end, dtype=np.float64)
        return start + np.asarray(fractions, dtype=np.float64)[:, None] * (end - start)


def _use_endpoints(use: PatchBoundaryUse, /) -> np.ndarray:
    match use:
        case PatchCurveUse():
            return use.charts(np.asarray((use.first, use.last)))
        case PatchPoleUse():
            return np.asarray((use.start, use.end), dtype=np.float64)
        case _:
            raise TypeError("Patch boundary uses are PatchCurveUse or PatchPoleUse.")


def _corner_use_endpoints(use: PatchBoundaryUse, /) -> np.ndarray:
    """Physical endpoint representatives honoring exact integer source periods."""
    parameters = _use_endpoints(use)
    if not isinstance(use, PatchCurveUse):
        return parameters
    values = np.asarray((use.first, use.last), dtype=np.float64)
    for index, root in enumerate((use.first_root, use.last_root)):
        if isinstance(root, NativePeriodEndpoint) and root.turns.denominator == 1:
            values[index] = float(root.rational)
    return use.charts(values)


@final
class MeshingSurfacePatch(StrictModule, NonTrainableState):
    """One parametric patch with its boundary loops and oriented normal.

    ``loops`` holds the outer loop (counterclockwise in the parameter plane)
    followed by holes (clockwise). ``reversed`` selects the oriented normal
    ``-(d_u x d_v)`` instead of ``d_u x d_v``.
    """

    surface: AbstractSurfacePatch
    loops: tuple[tuple[PatchBoundaryUse, ...], ...]
    reversed: bool = eqx.field(static=True)

    def __init__(
        self,
        surface: AbstractSurfacePatch,
        loops: tuple[tuple[PatchBoundaryUse, ...], ...],
        /,
        *,
        reversed: bool = False,
    ) -> None:
        if not isinstance(surface, AbstractSurfacePatch):
            raise TypeError("surface must be an AbstractSurfacePatch.")
        loops_ = tuple(tuple(loop) for loop in loops)
        if not loops_ or any(not loop for loop in loops_):
            raise ValueError("A patch needs a nonempty outer loop of boundary uses.")
        for loop in loops_:
            for use in loop:
                if not isinstance(use, (PatchCurveUse, PatchPoleUse)):
                    raise TypeError(
                        "Patch boundary uses are PatchCurveUse or PatchPoleUse."
                    )
        if not isinstance(reversed, (bool, np.bool_)):
            raise TypeError("reversed must be a bool.")
        self.surface = surface
        self.loops = loops_
        self.reversed = bool(reversed)

    @property
    def orientation(self) -> float:
        """Sign relating the oriented normal to ``d_u x d_v``."""

        return -1.0 if self.reversed else 1.0


@final
class MeshingDomainCurve(StrictModule, NonTrainableState):
    """A declared curve stratum joining two corners (equal for a closed curve)."""

    start: int = eqx.field(static=True)
    end: int = eqx.field(static=True)

    def __init__(self, start: int, end: int, /) -> None:
        self.start = _index(start, "start")
        self.end = _index(end, "end")


@final
class MeshingDomainRegion(StrictModule, NonTrainableState):
    """A region bounded by oriented patches.

    ``boundary`` pairs a patch index with ``+1`` when the patch's oriented
    normal points out of the region and ``-1`` when it points in.
    """

    name: str = eqx.field(static=True)
    boundary: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __init__(self, name: str, boundary: tuple[tuple[int, int], ...], /) -> None:
        entries = []
        for patch, side in boundary:
            if side not in (-1, 1):
                raise ValueError("Region boundary sides are +1 or -1.")
            entries.append((_index(patch, "patch"), int(side)))
        if not entries or len({patch for patch, _ in entries}) != len(entries):
            raise ValueError("A region is bounded by distinct patches.")
        self.name = _identifier(name, "name")
        self.boundary = tuple(sorted(entries))


@final
class MeshingDomainProjection(StrictModule, NonTrainableState):
    """Closest points on requested patches with per-point status.

    ``converged`` holds where the bounded Newton iteration reached a
    stationary point of the squared distance with a finite, solvable
    linearization; ``points``/``parameters``/``distances`` are meaningful only
    there.
    """

    points: np.ndarray
    parameters: np.ndarray
    distances: np.ndarray
    converged: np.ndarray


class _DomainCurveMap(AbstractBoundaryMap):
    """Source curve realizations retaining exact implicit endpoint definitions."""

    surfaces: tuple[AbstractSurfacePatch, ...]
    pcurves: tuple[AbstractCurve | AbstractTrimCurve, ...]
    firsts: Array
    lasts: Array
    endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    endpoint_errors: np.ndarray

    def __init__(
        self,
        surfaces: tuple[AbstractSurfacePatch, ...],
        pcurves: tuple[AbstractCurve | AbstractTrimCurve, ...],
        firsts: np.ndarray,
        lasts: np.ndarray,
        endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
        | None = None,
    ) -> None:
        roots = (
            ((None, None),) * len(pcurves) if endpoint_roots is None else endpoint_roots
        )
        if len(roots) != len(pcurves) or len(surfaces) != len(pcurves):
            raise ValueError(
                "Curve root definitions must align with source chart carriers."
            )
        errors = np.zeros((len(pcurves), 2), dtype=np.float64)
        for row, (surface, pcurve, pair) in enumerate(
            zip(surfaces, pcurves, roots, strict=True)
        ):
            if not any(endpoint is not None for endpoint in pair):
                continue
            increasing = float(firsts[row]) <= float(lasts[row])
            ordered_roots = pair if increasing else pair[::-1]
            support = sorted((float(firsts[row]), float(lasts[row])))
            if ordered_roots[0] is not None:
                lower, _ = ordered_roots[0].parameter_enclosure()
                support[0] = min(support[0], lower)
            if ordered_roots[1] is not None:
                _, upper = ordered_roots[1].parameter_enclosure()
                support[1] = max(support[1], upper)
            if isinstance(pcurve, IntersectionPCurve):
                box = pcurve.enclosure(*support, endpoint_roots=ordered_roots)
                lower, upper = pcurve.derivative_bounds(
                    *support, order=1, endpoint_roots=ordered_roots
                )
            elif isinstance(pcurve, AbstractCurve):
                box = pcurve.bounding_box(*support)
                lower, upper = pcurve.derivative_bounds(*support, order=1)
            else:
                box = pcurve.enclosure(*support)
                lower, upper = pcurve.derivative_bounds(*support, order=1)
            velocity = np.maximum(np.abs(lower), np.abs(upper))
            jl, ju = surface.derivative_bounds(box, order=1)
            speed = float(
                np.linalg.norm(np.maximum(np.abs(jl), np.abs(ju)), axis=0) @ velocity
            )
            for end, endpoint in enumerate(pair):
                if endpoint is not None:
                    lower, upper = endpoint.parameter_enclosure()
                    nominal = float((firsts, lasts)[end][row])
                    errors[row, end] = np.nextafter(
                        speed * max(abs(lower - nominal), abs(upper - nominal)), np.inf
                    )
        self.surfaces = surfaces
        self.pcurves = pcurves
        self.firsts = jnp.asarray(firsts, dtype=jnp.float64)
        self.lasts = jnp.asarray(lasts, dtype=jnp.float64)
        self.endpoint_roots = roots
        self.endpoint_errors = errors

    @property
    def num_charts(self) -> int:
        return len(self.surfaces)

    @property
    def reference_dimension(self) -> int:
        return 1

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        # Curves are heterogeneous; every curve is evaluated and the requested
        # one selected, which keeps the map traceable in the chart index.
        parameter = reference[..., 0]
        images = []
        for curve, (surface, pcurve) in enumerate(
            zip(self.surfaces, self.pcurves, strict=True)
        ):
            value = self.firsts[curve] + parameter * (
                self.lasts[curve] - self.firsts[curve]
            )
            images.append(surface.evaluate(pcurve.evaluate(value)))
        stacked = jnp.stack(images, axis=0)
        selected = jnp.asarray(chart_indices, dtype=jnp.int32)[None, ..., None]
        return jnp.take_along_axis(stacked, selected, axis=0)[0]

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        _, velocity = jax.jvp(
            lambda value: self.map(chart_indices, value),
            (reference,),
            (jnp.ones_like(reference),),
        )
        return jnp.linalg.norm(velocity, axis=-1)

    def select_chart(self, row: int, /) -> AbstractBoundaryMap:
        """A numerical evaluation view preserving this chart's root evidence."""
        return eqx.tree_at(
            lambda value: (
                value.surfaces,
                value.pcurves,
                value.firsts,
                value.lasts,
                value.endpoint_roots,
                value.endpoint_errors,
            ),
            self,
            (
                (self.surfaces[row],),
                (self.pcurves[row],),
                self.firsts[row : row + 1],
                self.lasts[row : row + 1],
                (self.endpoint_roots[row],),
                self.endpoint_errors[row : row + 1],
            ),
            is_leaf=lambda value: value is None,
        )


class _PhysicalCurveMap(AbstractBoundaryMap):
    """Exact native edge carriers over the normalized reference interval."""

    curves: tuple[AbstractCurve | IntersectionCurve, ...]
    ranges: Array
    endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    endpoint_errors: np.ndarray

    def __init__(
        self,
        curves: tuple[AbstractCurve | IntersectionCurve, ...],
        ranges: np.ndarray,
        endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
        | None = None,
        vertex_realization_errors: np.ndarray | None = None,
    ) -> None:
        roots = (
            ((None, None),) * len(curves) if endpoint_roots is None else endpoint_roots
        )
        if len(roots) != len(curves) or np.asarray(ranges).shape != (len(curves), 2):
            raise ValueError(
                "Physical edge roots and ranges must align with their source carriers."
            )
        errors = (
            np.zeros((len(curves), 2), dtype=np.float64)
            if vertex_realization_errors is None
            else np.array(vertex_realization_errors, dtype=np.float64, copy=True)
        )
        if (
            errors.shape != (len(curves), 2)
            or np.any(~np.isfinite(errors))
            or np.any(errors < 0)
        ):
            raise ValueError(
                "Edge vertex realizations require finite nonnegative endpoint error bounds."
            )
        for row, (curve, pair) in enumerate(zip(curves, roots, strict=True)):
            increasing = float(ranges[row, 0]) <= float(ranges[row, 1])
            ordered_roots = pair if increasing else pair[::-1]
            support = sorted(float(value) for value in ranges[row])
            if ordered_roots[0] is not None:
                lower, _ = ordered_roots[0].parameter_enclosure()
                support[0] = min(support[0], lower)
            if ordered_roots[1] is not None:
                _, upper = ordered_roots[1].parameter_enclosure()
                support[1] = max(support[1], upper)
            if any(endpoint is not None for endpoint in pair):
                lower, upper = (
                    curve.derivative_bounds(
                        support[0],
                        support[1],
                        order=1,
                        endpoint_roots=ordered_roots,
                    )
                    if isinstance(curve, IntersectionCurve)
                    else curve.derivative_bounds(support[0], support[1], order=1)
                )
                speed = float(np.linalg.norm(np.maximum(np.abs(lower), np.abs(upper))))
                for end, endpoint in enumerate(pair):
                    if endpoint is not None:
                        lower, upper = endpoint.parameter_enclosure()
                        radius = max(
                            abs(lower - float(ranges[row, end])),
                            abs(upper - float(ranges[row, end])),
                        )
                        errors[row, end] = np.nextafter(
                            errors[row, end] + speed * radius, np.inf
                        )
        errors.setflags(write=False)
        self.curves = curves
        self.ranges = jnp.asarray(ranges, dtype=jnp.float64)
        self.endpoint_roots = roots
        self.endpoint_errors = errors

    @property
    def num_charts(self) -> int:
        return len(self.curves)

    @property
    def reference_dimension(self) -> int:
        return 1

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        images = []
        for index, curve in enumerate(self.curves):
            parameters = self.ranges[index, 0] + reference[..., 0] * (
                self.ranges[index, 1] - self.ranges[index, 0]
            )
            if isinstance(curve, IntersectionCurve):
                if not curve.fully_certified:
                    raise ValueError(
                        "Meshing requires certified intersection continuation charts."
                    )
                result = curve.evaluate(parameters)
                if not isinstance(result.parameter_bound, jax_core.Tracer) and not np.all(
                    np.isfinite(np.asarray(result.parameter_bound))
                ):
                    raise ValueError(
                        "Intersection curve evaluation has unresolved source bounds."
                    )
                images.append(result.point)
            else:
                images.append(curve.evaluate(parameters))
        selected = jnp.asarray(chart_indices, dtype=jnp.int32)[None, ..., None]
        return jnp.take_along_axis(jnp.stack(images), selected, axis=0)[0]

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        _, velocity = jax.jvp(
            lambda value: self.map(chart_indices, value),
            (reference,),
            (jnp.ones_like(reference),),
        )
        return jnp.linalg.norm(velocity, axis=-1)

    def select_chart(self, row: int, /) -> AbstractBoundaryMap:
        """A numerical evaluation view, not a sampled replacement source."""
        return eqx.tree_at(
            lambda value: (
                value.curves,
                value.ranges,
                value.endpoint_roots,
                value.endpoint_errors,
            ),
            self,
            (
                (self.curves[row],),
                self.ranges[row : row + 1],
                (self.endpoint_roots[row],),
                self.endpoint_errors[row : row + 1],
            ),
            is_leaf=lambda value: value is None,
        )


def _domain_trim_curve(
    source: MeshingSurfacePatch, use: PatchCurveUse, /
) -> AbstractTrimCurve:
    """Reuse the CAD owner's exact rooted coedge and UV-join representation."""
    from .brep._constructors import _NormalizedTrimCurve

    if use.trim_curve is not None:
        return use.trim_curve
    first, last = use.first, use.last
    segment = CurveTrimSegment(
        use.pcurve,
        min(first, last),
        max(first, last),
        reversed=last < first,
        first_root=use.first_root if first < last else use.last_root,
        last_root=use.last_root if first < last else use.first_root,
    )
    definition = (
        source.surface.definition
        if isinstance(source.surface, PlacedSurface)
        else source.surface
    )
    return _NormalizedTrimCurve(
        segment,
        jnp.zeros((2,), dtype=jnp.float64),
        jnp.ones((2,), dtype=jnp.float64),
        definition,
        use.start_vertex_root,
        use.end_vertex_root,
        use.first_root,
        use.last_root,
    )


def _root_endpoint_error(
    source: MeshingSurfacePatch, use: PatchCurveUse, end: int, point: np.ndarray, /
) -> float:
    """Radius of the exact root image about this numerical endpoint realization."""
    endpoint = (use.first_root, use.last_root)[end]
    vertex = (use.start_vertex_root, use.end_vertex_root)[end]
    if endpoint is None:
        return 0.0
    # An authored native-period scalar is not an implicit spatial junction.
    # Transport its exact scalar enclosure by the actual coedge and surface
    # expression below, without inventing a common spatial vertex root.
    if not isinstance(endpoint, NativePeriodEndpoint) and (
        vertex is None or not vertex.supports_endpoint(endpoint)
    ):
        raise ValueError(
            "A rooted coedge endpoint requires its common authoritative spatial vertex."
        )
    curve = _domain_trim_curve(source, use)
    box = source.surface.bounding_box(curve.enclosure(float(end), float(end)))
    return float(
        np.nextafter(np.linalg.norm(np.max(np.abs(box - point), axis=0)), np.inf)
    )


def _uses(patches: tuple[MeshingSurfacePatch, ...], /) -> list[tuple[int, int, int]]:
    """``(patch, loop, position)`` of every curve and pole use in canonical order."""

    return [
        (patch, loop, position)
        for patch, value in enumerate(patches)
        for loop, members in enumerate(value.loops)
        for position in range(len(members))
    ]


# Host source batches pad to power-of-two buckets from this floor, so a growing
# refinement compiles one program per bucket rather than one per batch size.
_SOURCE_BATCH_FLOOR = 64


def _source_batch_bucket(count: int, /) -> int:
    """Bounded padded capacity of a host batch of ``count`` source queries."""
    return max(_SOURCE_BATCH_FLOOR, 1 << max(count - 1, 0).bit_length())


def _bucketed(values: np.ndarray, /) -> Array:
    """Nonempty query rows padded with copies of their first row to a bucket.

    Padding lanes repeat an admitted query and are sliced off before any
    host value is formed; they never reach a published result.
    """
    padded = np.empty(
        (_source_batch_bucket(values.shape[0]), *values.shape[1:]), dtype=np.float64
    )
    padded[: values.shape[0]] = values
    padded[values.shape[0] :] = values[0]
    return jnp.asarray(padded)


@eqx.filter_jit
def _source_evaluation(surface: AbstractSurfacePatch, charts: Array, /) -> Array:
    """One owned source program, with scientific coefficients kept dynamic."""
    return surface.evaluate(charts)


@eqx.filter_jit
def _source_differential(surface: AbstractSurfacePatch, charts: Array, /) -> Array:
    """One owned source-differential program, coefficients kept dynamic."""
    return surface_differential(surface, charts)


@eqx.filter_jit
def _curve_evaluation(curve: AbstractCurve, parameters: Array, /) -> Array:
    """One owned source-curve program, coefficients kept dynamic."""
    return curve.evaluate(parameters)


def _evaluate(surface: AbstractSurfacePatch, charts: np.ndarray, /) -> np.ndarray:
    values = np.asarray(charts, dtype=np.float64).reshape((-1, 2))
    if not values.shape[0]:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(_source_evaluation(surface, _bucketed(values)), dtype=np.float64)[
        : values.shape[0]
    ]


def _differential(surface: AbstractSurfacePatch, charts: np.ndarray, /) -> np.ndarray:
    values = np.asarray(charts, dtype=np.float64).reshape((-1, 2))
    if not values.shape[0]:
        return np.empty((0, 3, 2), dtype=np.float64)
    return np.asarray(_source_differential(surface, _bucketed(values)), dtype=np.float64)[
        : values.shape[0]
    ]


def _gram_spectrum_bounds(matrix: np.ndarray, /) -> tuple[float, float]:
    """Outward Gram eigenvalue enclosure with native spectral residual evidence."""
    gram = matrix.T @ matrix
    spectrum = HermitianSpectrum(jnp.asarray(gram, dtype=jnp.float64))
    if not bool(np.asarray(spectrum.valid)):
        raise ValueError("Source derivative Gram spectrum is not finite Hermitian.")
    values = np.asarray(spectrum.eigenvalues)
    vectors = np.asarray(spectrum.eigenvectors)
    eps = np.finfo(np.float64).eps
    orthogonal_error = np.max(
        np.sum(
            np.abs(vectors.T @ vectors - np.eye(values.size))
            + 32 * eps * (np.abs(vectors).T @ np.abs(vectors)),
            axis=1,
        )
    )
    reconstruction = (vectors * values[None]) @ vectors.T
    roundoff = (
        64
        * eps
        * (
            np.abs(matrix).T @ np.abs(matrix)
            + (np.abs(vectors) * np.abs(values)[None]) @ np.abs(vectors).T
        )
    )
    residual = np.max(np.sum(np.abs(gram - reconstruction) + roundoff, axis=1))
    lower = float(values[0] * (1 - orthogonal_error) - residual)
    upper = float(values[-1] * (1 + orthogonal_error) + residual)
    return float(np.nextafter(lower, -np.inf)), float(np.nextafter(upper, np.inf))


def _matrix_norm_upper(matrix: np.ndarray, /) -> float:
    """Sharp outward operator norm from the native Gram spectral owner."""
    _, upper = _gram_spectrum_bounds(matrix)
    return float(np.nextafter(np.sqrt(max(upper, 0.0)), np.inf))


def _hessian_bounds(
    surface: AbstractSurfacePatch, /
) -> tuple[float, float, float] | None:
    """Global norms of the three second parameter derivatives."""
    if isinstance(surface, PlacedSurface):
        bounds = _hessian_bounds(surface.definition)
        factor = _matrix_norm_upper(np.asarray(surface.rotation))
        return (
            None
            if bounds is None
            else (bounds[0] * factor, bounds[1] * factor, bounds[2] * factor)
        )
    if isinstance(surface, PlanePatch):
        return 0.0, 0.0, 0.0
    sphere = sphere_source_equivalence(surface)
    if sphere is not None:
        original, exact_radius = sphere
        radius = float(np.nextafter(float(abs(exact_radius)), np.inf))
        radial = _matrix_norm_upper(
            np.column_stack(
                (np.asarray(original.first_axis), np.asarray(original.second_axis))
            )
        )
        whole = _matrix_norm_upper(
            np.column_stack(
                (
                    np.asarray(original.first_axis),
                    np.asarray(original.second_axis),
                    np.asarray(original.axis),
                )
            )
        )
        first = float(np.nextafter(radius * radial, np.inf))
        third = float(np.nextafter(radius * whole, np.inf))
        return first, first, third
    if isinstance(surface, OffsetSurface):
        # The exact lowered primitive is the same mathematical source map.
        equivalent = surface.analytic_equivalent()
        return None if equivalent is None else _hessian_bounds(equivalent)
    if isinstance(surface, (CylinderPatch, TorusPatch)):
        radial = _matrix_norm_upper(
            np.column_stack(
                (np.asarray(surface.first_axis), np.asarray(surface.second_axis))
            )
        )
        whole = _matrix_norm_upper(
            np.column_stack(
                (
                    np.asarray(surface.first_axis),
                    np.asarray(surface.second_axis),
                    np.asarray(surface.axis),
                )
            )
        )
        if isinstance(surface, CylinderPatch):
            return abs(float(surface.radius)) * radial, 0.0, 0.0
        major, minor = abs(float(surface.major_radius)), abs(float(surface.minor_radius))
        return (major + minor) * radial, minor * radial, minor * whole
    return None


def _curve_second_bound(curve: AbstractCurve | IntersectionCurve, /) -> float:
    if isinstance(curve, PlacedCurve):
        return _curve_second_bound(curve.definition) * _matrix_norm_upper(
            np.asarray(curve.rotation)
        )
    if isinstance(curve, LineCurve):
        return 0.0
    if isinstance(curve, CircleCurve):
        return float(curve.radius) * _matrix_norm_upper(
            np.column_stack((np.asarray(curve.first_axis), np.asarray(curve.second_axis)))
        )
    if isinstance(curve, SurfaceIsoparametricCurve):
        bounds = _hessian_bounds(curve.surface)
        if bounds is not None:
            return bounds[2] if curve.fixed_axis == 0 else bounds[0]
    return np.inf


def _affine_bspline_interval(
    curve: AbstractCurve | AbstractTrimCurve | IntersectionCurve,
    first: float,
    last: float,
    /,
) -> bool:
    """Prove one closed degree-one source span is exactly affine."""
    if isinstance(curve, PlacedCurve):
        return _affine_bspline_interval(curve.definition, first, last)
    if not isinstance(curve, BSplineCurve) or curve.degree != 1:
        return False
    lower, upper = sorted((first, last))
    for piece in curve.bezier_pieces():
        ((start, end),) = piece.parameter_bounds
        if not start <= lower <= upper <= end:
            continue
        controls = np.asarray(piece.homogeneous_controls)
        weights = controls[:, -1]
        return controls.shape[0] == 2 and weights[0] > 0.0 and weights[0] == weights[1]
    return False


def _physical_curve_chord_bound(
    curve: AbstractCurve | IntersectionCurve,
    first: float,
    last: float,
    /,
    *,
    known_second: float | None = None,
) -> float:
    width = last - first
    if width == 0:
        return 0.0
    if _affine_bspline_interval(curve, first, last):
        return 0.0
    if _curve_c1_interval(curve, first, last):
        second = _curve_second_bound(curve) if known_second is None else known_second
        if not np.isfinite(second):
            low, high = curve.derivative_bounds(first, last, order=2)
            second = float(np.linalg.norm(np.maximum(np.abs(low), np.abs(high))))
        return second * width * width / 8
    low, high = curve.derivative_bounds(first, last, order=1)
    return float(np.linalg.norm(np.nextafter(high - low, np.inf))) * width / 4


def _chart_curve_chord_bound(
    surface: AbstractSurfacePatch,
    curve: AbstractCurve | AbstractTrimCurve,
    first: float,
    last: float,
    /,
    *,
    known_second: float | None = None,
) -> float:
    width = last - first
    if width == 0:
        return 0.0
    if _affine_bspline_interval(curve, first, last):
        hessian = _hessian_bounds(surface)
        if hessian == (0.0, 0.0, 0.0):
            return 0.0
    if known_second is not None:
        return known_second * width * width / 8
    if isinstance(curve, LineCurve):
        hessian = _hessian_bounds(surface)
        if hessian is not None:
            direction = np.abs(np.asarray(curve.direction))
            second = (
                hessian[0] * direction[0] ** 2
                + 2 * hessian[1] * direction[0] * direction[1]
                + hessian[2] * direction[1] ** 2
            )
            return second * width * width / 8
    if isinstance(surface, PlanePatch) and isinstance(curve, CircleCurve):
        axes = np.stack((np.asarray(surface.first_axis), np.asarray(surface.second_axis)))
        mapped = np.column_stack(
            (
                np.asarray(curve.first_axis) @ axes,
                np.asarray(curve.second_axis) @ axes,
            )
        )
        return float(curve.radius) * _matrix_norm_upper(mapped) * width * width / 8
    box = curve.bounding_box(first, last)
    first_low, first_high = curve.derivative_bounds(first, last, order=1)
    surface_low, surface_high = surface.derivative_bounds(box, order=1)
    if not _curve_c1_interval(curve, first, last):
        products = np.stack(
            (
                surface_low * first_low,
                surface_low * first_high,
                surface_high * first_low,
                surface_high * first_high,
            )
        )
        low = np.sum(np.min(products, axis=0), axis=1)
        high = np.sum(np.max(products, axis=0), axis=1)
        slack = (
            64
            * np.finfo(np.float64).eps
            * np.sum(np.max(np.abs(products), axis=0), axis=1)
        )
        diameter = np.nextafter(high - low + 2 * slack, np.inf)
        return float(np.linalg.norm(diameter)) * width / 4
    second_low, second_high = curve.derivative_bounds(first, last, order=2)
    hessian_low, hessian_high = surface.derivative_bounds(box, order=2)
    if not all(
        np.all(np.isfinite(bound))
        for bound in (
            first_low,
            first_high,
            second_low,
            second_high,
            surface_low,
            surface_high,
            hessian_low,
            hessian_high,
        )
    ):
        return np.inf
    direction = np.maximum(np.abs(first_low), np.abs(first_high))
    acceleration = np.maximum(np.abs(second_low), np.abs(second_high))
    speed = np.linalg.norm(np.maximum(np.abs(surface_low), np.abs(surface_high)), axis=0)
    hessian = np.linalg.norm(
        np.maximum(np.abs(hessian_low), np.abs(hessian_high)), axis=0
    )
    second = float(direction @ hessian @ direction + speed @ acceleration)
    return second * width * width / 8


def _cross_interval(
    a_low: np.ndarray,
    a_high: np.ndarray,
    b_low: np.ndarray,
    b_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = [], []
    for first, second in ((1, 2), (2, 0), (0, 1)):
        positive = interval_multiply(
            (a_low[first], a_high[first]),
            (b_low[second], b_high[second]),
        )
        negative = interval_multiply(
            (a_low[second], a_high[second]),
            (b_low[first], b_high[first]),
        )
        bounds = interval_subtract(positive, negative)
        lower.append(float(bounds[0]))
        upper.append(float(bounds[1]))
    return np.asarray(lower), np.asarray(upper)


def _normal_interval_diameter(lower: np.ndarray, upper: np.ndarray, /) -> float:
    """Gauss-set diameter of a complete unnormalized-normal interval box.

    A convex box separated from zero has normalization derivative norm at most
    ``1 / rho``. Its diameter ``D`` therefore bounds normalized chord length by
    ``D / rho``; the angular diameter is at most ``2 asin(D / (2 rho))``.
    Exact squared distances preserve separation and ratio rounding even when
    jets have different scales. No second derivative or C1 premise is needed.
    """
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        return np.inf
    separated = np.maximum(np.maximum(lower, -upper), 0.0)
    distance_squared = sum(
        (Fraction(float(value)) ** 2 for value in separated),
        Fraction(),
    )
    if distance_squared == 0:
        return np.inf
    diameter_squared = sum(
        (
            (Fraction(float(high)) - Fraction(float(low))) ** 2
            for low, high in zip(lower, upper, strict=True)
        ),
        Fraction(),
    )
    ratio_squared = diameter_squared / (4 * distance_squared)
    if ratio_squared >= 1:
        return float(np.nextafter(np.pi, np.inf))
    from .brep._sphere_membership import _square_root_interval

    radius_upper = _square_root_interval(ratio_squared)[1]
    angle = 2 * np.arcsin(radius_upper)
    return float(np.nextafter(angle * (1 + 32 * np.finfo(np.float64).eps), np.inf))


def _jet_normal_turns(
    surface: AbstractSurfacePatch, boxes: np.ndarray, widths: np.ndarray, /
) -> np.ndarray:
    """Normal-cone or Taylor turn from the surface's complete interval jets."""
    result = []
    first_lower, first_upper = surface.derivative_bounds_batch(boxes, order=1)
    second_lower, second_upper = surface.derivative_bounds_batch(boxes, order=2)
    for low, high, second_low, second_high, width in zip(
        first_lower,
        first_upper,
        second_lower,
        second_upper,
        widths,
        strict=True,
    ):
        cross_low, cross_high = _cross_interval(
            low[:, 0], high[:, 0], low[:, 1], high[:, 1]
        )
        cone = _normal_interval_diameter(cross_low, cross_high)
        if not np.isfinite(cone):
            result.append(np.inf)
            continue
        separated = np.maximum(np.maximum(cross_low, -cross_high), 0.0)
        regularity = float(np.linalg.norm(separated))
        first = np.linalg.norm(np.maximum(np.abs(low), np.abs(high)), axis=0)
        second = np.linalg.norm(
            np.maximum(np.abs(second_low), np.abs(second_high)), axis=0
        )
        bound = cone
        if (
            np.isfinite(regularity)
            and regularity > 0
            and np.all(np.isfinite(first))
            and np.all(np.isfinite(second))
        ):
            speed = np.asarray(
                (
                    second[0, 0] * first[1] + first[0] * second[1, 0],
                    second[0, 1] * first[1] + first[0] * second[1, 1],
                )
            )
            taylor = float(speed @ width / regularity)
            if np.isfinite(taylor):
                bound = min(bound, taylor)
        result.append(bound)
    return np.asarray(result, dtype=np.float64)


def _basis_distortion(
    surface: ConePatch | CylinderPatch | SpherePatch | TorusPatch, /
) -> float:
    return _matrix_distortion(
        np.column_stack(
            (
                np.asarray(surface.first_axis),
                np.asarray(surface.second_axis),
                np.asarray(surface.axis),
            )
        )
    )


def _matrix_distortion(matrix: np.ndarray, /) -> float:
    lower, upper = _gram_spectrum_bounds(matrix)
    if lower <= 0:
        return np.inf
    return float(np.nextafter(np.sqrt(upper / lower), np.inf))


def _full_sphere_frame(
    domain: MeshingDomain, patch: int, /
) -> tuple[np.ndarray, float, float] | None:
    """Radial frame only for an authored complete sphere, never a trimmed cap."""
    value = domain.patches[patch]
    surface = value.surface
    operation = surface.definition if isinstance(surface, PlacedSurface) else surface
    equivalence = sphere_source_equivalence(operation)
    if equivalence is None or len(value.loops) != 1 or len(value.loops[0]) != 4:
        return None
    definition, exact_radius = equivalence
    poles = [use for use in value.loops[0] if isinstance(use, PatchPoleUse)]
    seams = [use for use in value.loops[0] if isinstance(use, PatchCurveUse)]
    if len(poles) != 2 or len(seams) != 2 or seams[0].curve != seams[1].curve:
        return None
    if not all(isinstance(use.pcurve, LineCurve) for use in seams):
        factors = tuple(pcurve_periodic_source(use.pcurve, operation) for use in seams)
        if any(
            factor is None or not isinstance(factor[0], LineCurve) for factor in factors
        ):
            return None
        first, second = factors
        if (
            first is None
            or second is None
            or not isinstance(first[0], LineCurve)
            or not isinstance(second[0], LineCurve)
        ):
            return None
        a, b = first[0], second[0]
        if (
            not np.array_equal(np.asarray(a.origin), np.asarray(b.origin))
            or not np.array_equal(np.asarray(a.direction), np.asarray(b.direction))
            or float(a.direction[0]) != 0.0
            or first[1][1]
            or second[1][1]
            or abs(first[1][0] - second[1][0]) != 1
            or {seams[0].first, seams[0].last} != {seams[1].first, seams[1].last}
        ):
            return None
    endpoints = np.concatenate([_use_endpoints(use) for use in value.loops[0]])
    slack = 128 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(endpoints))))
    if (
        abs(float(np.ptp(endpoints[:, 0])) - 2 * np.pi) > slack
        or abs(float(np.min(endpoints[:, 1])) + np.pi / 2) > slack
        or abs(float(np.max(endpoints[:, 1])) - np.pi / 2) > slack
        or any(abs(use.start[1] - use.end[1]) > slack for use in poles)
    ):
        return None
    matrix = np.column_stack(
        (
            np.asarray(definition.first_axis),
            np.asarray(definition.second_axis),
            np.asarray(definition.axis),
        )
    )
    low, high = _gram_spectrum_bounds(matrix)
    if low <= 0:
        return None
    represented = float(abs(exact_radius))
    radius_low = (
        float(np.nextafter(represented, -np.inf))
        if Fraction(represented) > abs(exact_radius)
        else represented
    )
    radius_high = (
        float(np.nextafter(represented, np.inf))
        if Fraction(represented) < abs(exact_radius)
        else represented
    )
    radius_low = float(
        np.nextafter(radius_low * np.nextafter(np.sqrt(low), -np.inf), -np.inf)
    )
    radius_high = float(
        np.nextafter(radius_high * np.nextafter(np.sqrt(high), np.inf), np.inf)
    )
    center = np.asarray(definition.center, dtype=np.float64)
    if isinstance(surface, PlacedSurface):
        rotation, translation = (
            np.asarray(surface.rotation),
            np.asarray(surface.translation),
        )
        pose_low, pose_high = _gram_spectrum_bounds(rotation)
        if pose_low <= 0:
            return None
        center_low, center_high = source_transform_bounds(rotation, center, center)
        center = rotation @ center + translation
        center_low, center_high = interval_add(
            (center_low, center_high), (translation, translation)
        )
        center_error = float(
            np.linalg.norm(
                np.maximum(np.abs(center_low - center), np.abs(center_high - center))
            )
        )
        center_error = float(
            np.nextafter(center_error * (1 + 64 * np.finfo(np.float64).eps), np.inf)
        )
        radius_low = float(
            np.nextafter(
                radius_low * np.nextafter(np.sqrt(pose_low), -np.inf) - center_error,
                -np.inf,
            )
        )
        radius_high = float(
            np.nextafter(
                radius_high * np.nextafter(np.sqrt(pose_high), np.inf) + center_error,
                np.inf,
            )
        )
    if radius_low <= 0:
        return None
    return center, radius_low, radius_high


def _interval_vector_norm(
    value: tuple[np.ndarray, np.ndarray], /
) -> tuple[np.ndarray, np.ndarray]:
    low, high = value
    absolute_low = np.maximum(np.maximum(low, -high), 0.0)
    absolute_high = np.maximum(np.abs(low), np.abs(high))
    lower = np.zeros(low.shape[:-1], dtype=np.float64)
    upper = lower.copy()
    for axis in range(3):
        lower = np.maximum(
            0.0,
            np.nextafter(
                lower
                + np.maximum(0.0, np.nextafter(absolute_low[..., axis] ** 2, -np.inf)),
                -np.inf,
            ),
        )
        upper = np.nextafter(
            upper + np.nextafter(absolute_high[..., axis] ** 2, np.inf), np.inf
        )
    return np.maximum(0.0, np.nextafter(np.sqrt(lower), -np.inf)), np.nextafter(
        np.sqrt(upper), np.inf
    )


def _interval_vector_cross(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    j, k = [1, 2, 0], [2, 0, 1]
    return interval_subtract(
        interval_multiply(
            (first[0][..., j], first[1][..., j]), (second[0][..., k], second[1][..., k])
        ),
        interval_multiply(
            (first[0][..., k], first[1][..., k]), (second[0][..., j], second[1][..., j])
        ),
    )


def _sphere_triangle_distance_bounds(
    frame: tuple[np.ndarray, float, float], corners: np.ndarray, /
) -> np.ndarray:
    """True Euclidean source distance bounds, not a same-UV interpolant norm.

    The full convex source is contained in the radial shell. Triangle interiors
    are contained in the source plus their vertex-distance enclosure by
    convexity. Source-to-mesh coverage additionally needs the independently
    verified closed radial degree-one chain.
    """
    center, radius_low, radius_high = frame
    relative = interval_subtract((corners, corners), (center, center))
    first = interval_subtract(
        (corners[:, 1], corners[:, 1]), (corners[:, 0], corners[:, 0])
    )
    second = interval_subtract(
        (corners[:, 2], corners[:, 2]), (corners[:, 0], corners[:, 0])
    )
    normal = _interval_vector_cross(first, second)
    product = interval_multiply(normal, (relative[0][:, 0], relative[1][:, 0]))
    dot_low, dot_high = np.zeros(corners.shape[0]), np.zeros(corners.shape[0])
    for axis in range(3):
        dot_low, dot_high = interval_add(
            (dot_low, dot_high), (product[0][:, axis], product[1][:, axis])
        )
    normal_high = _interval_vector_norm(normal)[1]
    numerator = np.maximum(np.maximum(dot_low, -dot_high), 0.0)
    distance_low = np.nextafter(
        numerator / np.where(normal_high > 0.0, normal_high, 1.0), -np.inf
    )
    repeated01 = np.all(corners[:, 0] == corners[:, 1], axis=1)
    repeated02 = np.all(corners[:, 0] == corners[:, 2], axis=1)
    repeated12 = np.all(corners[:, 1] == corners[:, 2], axis=1)
    collapsed = repeated01 | repeated02 | repeated12
    if np.any(collapsed):
        rows = np.flatnonzero(collapsed)
        endpoint = np.where(repeated01[rows], 2, 1)
        a = (relative[0][rows, 0], relative[1][rows, 0])
        b = (relative[0][rows, endpoint], relative[1][rows, endpoint])
        cross_low = _interval_vector_norm(_interval_vector_cross(a, b))[0]
        edge_high = _interval_vector_norm(interval_subtract(a, b))[1]
        segment_low = np.nextafter(
            cross_low / np.where(edge_high > 0.0, edge_high, 1.0), -np.inf
        )
        same = np.all(corners[rows, 0] == corners[rows, endpoint], axis=1)
        segment_low[same] = _interval_vector_norm(a)[0][same]
        distance_low[rows] = segment_low
    # The closest point of an obtuse or thin triangle can lie on an edge,
    # outside its supporting plane's orthogonal foot. Additional separating
    # planes give rigorous triangle-specific lower radii without a solve.
    for first, second in ((0, 1), (1, 2), (2, 0)):
        direction = 0.5 * corners[:, first] + 0.5 * corners[:, second] - center
        norm_high = _interval_vector_norm((direction, direction))[1]
        products = interval_multiply(
            relative,
            (direction[:, None], direction[:, None]),
        )
        dot_low = np.zeros(corners.shape[:2], dtype=np.float64)
        dot_high = dot_low.copy()
        for axis in range(3):
            dot_low, dot_high = interval_add(
                (dot_low, dot_high),
                (products[0][..., axis], products[1][..., axis]),
            )
        support = np.nextafter(
            np.min(dot_low, axis=1) / np.where(norm_high > 0.0, norm_high, 1.0),
            -np.inf,
        )
        distance_low = np.maximum(distance_low, np.maximum(support, 0.0))
    node_low, node_high = _interval_vector_norm(relative)
    node_error = np.max(
        np.maximum(node_high - radius_low, radius_high - node_low), axis=1
    )
    return np.nextafter(
        np.maximum(np.maximum(radius_high - distance_low, node_error), 0.0), np.inf
    )


def _sphere_triangle_normal_bounds(
    frame: tuple[np.ndarray, float, float], corners: np.ndarray, /
) -> np.ndarray:
    """Normal diameter over the radial source pieces of a complete sphere.

    A normalized convex combination of acute unit directions has diameter at
    most their largest pairwise angle. The ellipsoid Gauss map contributes its
    squared axis-condition bound; the shell width encloses center displacement.
    """
    center, radius_low, radius_high = frame
    relative = interval_subtract((corners, corners), (center, center))
    norms = _interval_vector_norm(relative)[1]
    cosine = np.ones((corners.shape[0],), dtype=np.float64)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        products = interval_multiply(
            (relative[0][:, first], relative[1][:, first]),
            (relative[0][:, second], relative[1][:, second]),
        )
        low, high = np.zeros(corners.shape[0]), np.zeros(corners.shape[0])
        for axis in range(3):
            low, high = interval_add(
                (low, high), (products[0][:, axis], products[1][:, axis])
            )
        denominator = np.nextafter(norms[:, first] * norms[:, second], np.inf)
        candidate = np.nextafter(
            low / np.where(denominator > 0, denominator, 1.0), -np.inf
        )
        cosine = np.minimum(cosine, candidate)
    diameter = np.nextafter(np.arccos(np.clip(cosine, 0.0, 1.0)), np.inf)
    diameter = np.where(cosine > 0.0, diameter, np.pi)
    displacement = 2 * np.nextafter(
        np.arcsin(min(1.0, (radius_high - radius_low) / radius_low)), np.inf
    )
    condition = np.nextafter((radius_high / radius_low) ** 2, np.inf)
    return np.minimum(np.pi, np.nextafter(condition * (diameter + displacement), np.inf))


def _sphere_radial_degree_one(
    frame: tuple[np.ndarray, float, float], points: np.ndarray, cells: np.ndarray, /
) -> bool:
    """Exact signed ray degree plus closed, consistently radial triangle chain."""
    from ._mesh_certificates import (
        _degenerate_simplices,
        _dyadic_integers,
        _ray_crossings_3d,
    )

    active = cells[~_degenerate_simplices(points, cells)]
    if not active.size:
        return False
    integers, _ = _dyadic_integers(np.concatenate((frame[0][None], points)))
    corners = integers[1:][active]
    x = corners - integers[0]
    cross = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    determinant = np.sum(cross * x[:, 0], axis=1)
    positive, negative = determinant > 0, determinant < 0
    if not (np.all(positive) or np.all(negative)):
        return False
    # Geometry equality is checked here, not reconstructed as scientific IDs.
    # Seam/pole aliases already follow the verified source-use provenance.
    directed = {}
    for triangle in points[active]:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            a, b = tuple(triangle[first]), tuple(triangle[second])
            if directed.get((b, a), 0):
                directed[(b, a)] -= 1
            else:
                directed[(a, b)] = directed.get((a, b), 0) + 1
    if any(directed.values()):
        return False
    degree, certain = _ray_crossings_3d(frame[0], points[active])
    return certain and abs(degree) == 1


@final
class MeshingDomain(StrictModule, NonTrainableState):
    """Revision-bound stratified parametric surface domain.

    Strata are corners (dimension 0), curves (1) and surfaces (2), identified by
    their declaration index as ``<revision>:<corner|curve|surface>:<index>``.
    ``tolerance`` is the relative coincidence tolerance used to validate shared
    corners, shared curves, pole sides and closed parameter loops against the
    source; the absolute bound is ``tolerance * scale`` with ``scale`` the
    largest coordinate magnitude of the domain (at least one).
    """

    patches: tuple[MeshingSurfacePatch, ...]
    curves: tuple[MeshingDomainCurve, ...]
    regions: tuple[MeshingDomainRegion, ...]
    corner_points: np.ndarray
    curve_owners: np.ndarray
    patch_regions: np.ndarray
    curve_atlas: BoundaryAtlas
    # B-Rep views retain the owner that authored physical edges and vertex banks.
    # Patch/pcurve composition alone is not that source representation.
    brep_authority: BRepGeometry | BRepModel | None
    source_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    source_occurrences: tuple[tuple[tuple[str, ...], ...], ...] = eqx.field(static=True)
    region_source_indices: tuple[int, ...] = eqx.field(static=True)
    region_source_occurrences: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_kinds: tuple[str, ...] = eqx.field(static=True)
    authority_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    accuracy: MeshingSourceAccuracy = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        patches: tuple[MeshingSurfacePatch, ...],
        curves: tuple[MeshingDomainCurve, ...],
        corner_count: int,
        /,
        *,
        source_id: str,
        source_revision: str,
        regions: tuple[MeshingDomainRegion, ...] = (),
        tolerance: float = 1.0e-9,
        accuracy: MeshingSourceAccuracy = "exact_represented",
        source_indices: tuple[tuple[int, ...], ...] | None = None,
        source_occurrences: tuple[tuple[tuple[str, ...], ...], ...] | None = None,
        region_source_indices: tuple[int, ...] | None = None,
        region_source_occurrences: tuple[tuple[str, ...], ...] | None = None,
        source_kinds: tuple[str, ...] = ("corner", "curve", "surface"),
        authority_id: str = "",
    ) -> None:
        patches_ = tuple(patches)
        curves_ = tuple(curves)
        regions_ = tuple(regions)
        if not patches_ or not all(
            isinstance(patch, MeshingSurfacePatch) for patch in patches_
        ):
            raise TypeError("patches must be a nonempty tuple of MeshingSurfacePatch.")
        if not all(isinstance(curve, MeshingDomainCurve) for curve in curves_):
            raise TypeError("curves must contain MeshingDomainCurve values.")
        if not all(isinstance(region, MeshingDomainRegion) for region in regions_):
            raise TypeError("regions must contain MeshingDomainRegion values.")
        corners = _index(corner_count, "corner_count")
        tolerance_ = _finite(tolerance, "tolerance")
        if tolerance_ <= 0.0:
            raise ValueError("tolerance must be positive.")
        accuracy_ = parse(accuracy, MeshingSourceAccuracy, "accuracy")
        owners = _curve_owners(patches_, curves_, corners)
        points, scale = _corner_points(patches_, curves_, owners, corners)
        _validate_uses(patches_, curves_, owners, points, tolerance_ * scale)
        _validate_loops(patches_, tolerance_)
        _validate_orientation(patches_, owners)
        patch_regions = _patch_regions(patches_, regions_)
        counts = (corners, len(curves_), len(patches_))
        indices = (
            tuple(tuple(range(count)) for count in counts)
            if source_indices is None
            else tuple(
                tuple(_index(index, "source index") for index in row)
                for row in source_indices
            )
        )
        paths = (
            tuple(((),) * count for count in counts)
            if source_occurrences is None
            else source_occurrences
        )
        if (
            not isinstance(paths, tuple)
            or len(indices) != 3
            or len(paths) != 3
            or any(
                len(row) != count
                or len(path_row) != count
                or len(set(zip(path_row, row, strict=True))) != count
                or any(
                    not isinstance(path, tuple)
                    or any(not isinstance(name, str) or not name for name in path)
                    for path in path_row
                )
                for row, path_row, count in zip(indices, paths, counts, strict=True)
            )
        ):
            raise ValueError(
                "Source strata must have unique authoritative occurrence/definition pairs."
            )
        region_indices = (
            tuple(range(len(regions_)))
            if region_source_indices is None
            else tuple(
                _index(index, "region source index") for index in region_source_indices
            )
        )
        region_paths = (
            ((),) * len(regions_)
            if region_source_occurrences is None
            else region_source_occurrences
        )
        if (
            not isinstance(region_paths, tuple)
            or len(region_indices) != len(regions_)
            or len(region_paths) != len(regions_)
            or len(set(zip(region_paths, region_indices, strict=True))) != len(regions_)
            or any(
                not isinstance(path, tuple)
                or any(not isinstance(name, str) or not name for name in path)
                for path in region_paths
            )
        ):
            raise ValueError(
                "Source regions must have unique authoritative occurrence/definition pairs."
            )
        if source_kinds not in (
            ("corner", "curve", "surface"),
            ("vertex", "edge", "face"),
        ):
            raise ValueError(
                "source_kinds must name domain strata or authoritative B-Rep strata."
            )
        owner_uses = [patches_[p].loops[loop][position] for p, loop, position in owners]
        self.patches = patches_
        self.curves = curves_
        self.regions = regions_
        self.corner_points = points
        self.curve_owners = np.asarray(owners, dtype=np.int64).reshape((-1, 3))
        self.patch_regions = patch_regions
        self.brep_authority = None
        self.source_id = _identifier(source_id, "source_id")
        self.source_revision = _identifier(source_revision, "source_revision")
        self.accuracy = accuracy_
        self.tolerance = tolerance_
        self.scale = scale
        self.source_indices = indices
        self.source_occurrences = paths
        self.region_source_indices = region_indices
        self.region_source_occurrences = region_paths
        self.source_kinds = tuple(source_kinds)
        self.authority_id = str(authority_id)
        self.curve_atlas = BoundaryAtlas(
            _DomainCurveMap(
                tuple(patches_[p].surface for p, _, _ in owners),
                tuple(_curve_use(use).pcurve for use in owner_uses),
                np.asarray([_curve_use(use).first for use in owner_uses]),
                np.asarray([_curve_use(use).last for use in owner_uses]),
                tuple(
                    (_curve_use(use).first_root, _curve_use(use).last_root)
                    for use in owner_uses
                ),
            ),
            source_entity_ids=jnp.arange(len(curves_), dtype=jnp.int32),
            source_id=self.source_id,
            physical_tags=tuple(f"curve:{index}" for index in range(len(curves_))),
        )
        self.domain_id = canonical_fingerprint(
            {
                "kind": "meshing-domain",
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "accuracy": accuracy_,
                "tolerance": tolerance_,
                "source_indices": indices,
                "source_kinds": self.source_kinds,
                "source_occurrences": paths,
                "region_source_indices": region_indices,
                "region_source_occurrences": region_paths,
                "authority_id": self.authority_id,
                "patches": [_patch_identity(patch) for patch in patches_],
                "curves": [[curve.start, curve.end] for curve in curves_],
                "corners": corners,
                "regions": [[region.name, region.boundary] for region in regions_],
            }
        )

    @property
    def corner_count(self) -> int:
        return self.corner_points.shape[0]

    def entity_id(self, dimension: int, index: int, /) -> str:
        """Canonical identity of one stratum of this revision."""

        if dimension not in _KINDS:
            raise ValueError("Meshing-domain entities have dimensions 0, 1, 2 or 3.")
        indices = (
            self.region_source_indices
            if dimension == 3
            else self.source_indices[dimension]
        )
        paths = (
            self.region_source_occurrences
            if dimension == 3
            else self.source_occurrences[dimension]
        )
        if self.source_kinds == ("vertex", "edge", "face"):
            from .brep._projection_contracts import brep_entity_id

            return brep_entity_id(
                self.source_revision,
                dimension,
                indices[index],
                occurrence_path=paths[index],
            )
        kind = "region" if dimension == 3 else self.source_kinds[dimension]
        return f"{self.source_revision}:{kind}:{indices[index]}"

    def curve_uses(self, curve: int, /) -> tuple[tuple[int, PatchCurveUse], ...]:
        """``(patch, use)`` pairs of one curve in canonical order (owner first)."""

        return tuple(
            (patch, use)
            for patch, value in enumerate(self.patches)
            for loop in value.loops
            for use in loop
            if isinstance(use, PatchCurveUse) and use.curve == curve
        )

    def patch_curves(self, patch: int, /) -> np.ndarray:
        """Sorted distinct curves on the boundary of one patch."""

        return np.unique(
            np.asarray(
                [
                    use.curve
                    for loop in self.patches[patch].loops
                    for use in loop
                    if isinstance(use, PatchCurveUse)
                ],
                dtype=np.int64,
            )
        )

    def entity_set_id(self, dimension: int, /) -> str:
        """Identity of the geometry entity set of one stratum dimension."""

        if dimension not in _KINDS:
            raise ValueError("Meshing-domain entities have dimensions 0, 1, 2 or 3.")
        kind = (
            ("solid" if self.source_kinds == ("vertex", "edge", "face") else "region")
            if dimension == 3
            else self.source_kinds[dimension]
        )
        paths = (
            self.region_source_occurrences
            if dimension == 3
            else self.source_occurrences[dimension]
        )
        plural = "vertices" if kind == "vertex" else f"{kind}s"
        namespace = "occurrence:" if any(paths) else ""
        return f"{self.source_id}:{namespace}{plural}"

    def scope_indices(self, dimension: int, /) -> tuple[int, ...]:
        """Scope slots; placed pairs remain explicit separate source metadata."""
        if dimension not in _KINDS:
            raise ValueError("Meshing-domain entities have dimensions 0, 1, 2 or 3.")
        indices = (
            self.region_source_indices
            if dimension == 3
            else self.source_indices[dimension]
        )
        paths = (
            self.region_source_occurrences
            if dimension == 3
            else self.source_occurrences[dimension]
        )
        return tuple(range(len(indices))) if any(paths) else indices

    def resolve_indices(self, dimension: int, identifiers: np.ndarray, /) -> np.ndarray:
        """Resolve source-local stratum indices without coordinate inference."""
        lookup = {
            source: local for local, source in enumerate(self.scope_indices(dimension))
        }
        try:
            return np.asarray(
                [lookup[int(index)] for index in identifiers], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError(
                "Scope entities must name strata of the meshing domain."
            ) from error

    def curve_breakpoints(self, curve: int, /) -> np.ndarray:
        """Authoritative chart/span junctions in normalized owner coordinates."""
        mapping = self.curve_atlas.mapping
        if isinstance(mapping, _PhysicalCurveMap):
            carrier = mapping.curves[curve]
            first, last = np.asarray(mapping.ranges[curve])
        elif isinstance(mapping, _DomainCurveMap):
            carrier = mapping.pcurves[curve]
            first, last = float(mapping.firsts[curve]), float(mapping.lasts[curve])
        else:
            raise TypeError(
                "A meshing-domain curve atlas must retain its authoritative carrier."
            )
        values = _carrier_breakpoints(
            carrier, float(min(first, last)), float(max(first, last))
        )
        return np.sort((values - first) / (last - first))

    def interpolation_bounds(
        self,
        patch: int,
        charts: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        reserve_queries: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Continuous chart-to-affine interpolation bounds, not nodal residuals.

        Taylor's integral remainder and the variance bound for a bounded
        scalar variable give ``(Huu du² + 2 Huv du dv + Hvv dv²) / 8``.
        This bounds both images under the common barycentric parameterization.
        It does not prove trim coverage or global injectivity. Unsupported
        carrier families return infinity, never a sampled derivative estimate.
        """
        coordinates = np.asarray(charts, dtype=np.float64)
        if coordinates.ndim != 3 or coordinates.shape[2] != 2:
            raise ValueError(
                "Surface interpolation charts must have shape (cell, node, 2)."
            )
        widths_raw = np.ptp(coordinates, axis=1)
        expected_widths = (coordinates.shape[0], 2)
        if np.shape(widths_raw) != expected_widths:
            raise RuntimeError("Surface interpolation widths lost their chart axes.")
        widths = np.empty(expected_widths, dtype=np.float64)
        widths[...] = widths_raw
        surface = self.patches[patch].surface
        definition = surface.definition if isinstance(surface, PlacedSurface) else surface
        from .brep._patches import ExtrusionSurface

        if isinstance(definition, ExtrusionSurface) and isinstance(
            definition.curve, BSplineCurve
        ):
            from ..discretization._coordinate_enclosure import (
                _COORDINATE_BUDGET,
                coordinate_enclosure_budget,
            )
            from ._source_interpolation import extrusion_interpolation_bounds

            owner = budget if budget is not None else _COORDINATE_BUDGET.get()
            if owner is None:
                from .brep._query import BRepQueryPolicy

                policy = BRepQueryPolicy()
                owner = coordinate_enclosure_budget(
                    policy.maximum_operations, policy.maximum_scratch_bytes
                )
            with owner.activate():
                enclosed = extrusion_interpolation_bounds(
                    surface, coordinates, owner, reserve_queries
                )
                owner.charge_native_work(
                    owner.work_units - owner.native_charged_work_units
                )
            if enclosed is None:
                raise RuntimeError(
                    "An admitted extrusion lost its original source operation tree."
                )
            return enclosed
        if isinstance(definition, ConePatch):
            # A cone has no global axial Hessian bound, but its radial factor
            # is affine on this triangle's box. Bound that factor before taking
            # the radial frame norm, rather than losing the cone cancellation
            # in componentwise interval automatic differentiation.
            axial = (
                np.min(coordinates[..., 1], axis=1),
                np.max(coordinates[..., 1], axis=1),
            )
            angle = np.asarray(definition.semi_angle)
            slope = _interval_tan((angle, angle))
            reference = np.asarray(definition.reference_radius)
            radius = interval_add((reference, reference), interval_multiply(axial, slope))
            radial = _matrix_norm_upper(
                np.column_stack(
                    (
                        np.asarray(definition.first_axis),
                        np.asarray(definition.second_axis),
                    )
                )
            )
            if isinstance(surface, PlacedSurface):
                radial = float(
                    np.nextafter(
                        radial * _matrix_norm_upper(np.asarray(surface.rotation)), np.inf
                    )
                )
            huu = np.nextafter(
                np.maximum(np.abs(radius[0]), np.abs(radius[1])) * radial, np.inf
            )
            huv = np.nextafter(
                np.maximum(np.abs(slope[0]), np.abs(slope[1])) * radial, np.inf
            )
            du, dv = widths.T
            bound = (huu * du * du + 2 * huv * du * dv) / 8
            return np.nextafter(bound * (1 + 64 * np.finfo(np.float64).eps), np.inf)
        hessian = _hessian_bounds(surface)
        if hessian is None:
            bounds = []
            boxes = np.stack(
                (np.min(coordinates, axis=1), np.max(coordinates, axis=1)), axis=1
            )
            from ._source_interpolation import revolution_interpolation_bounds

            swept = revolution_interpolation_bounds(
                definition,
                coordinates,
                _matrix_norm_upper(np.asarray(surface.rotation))
                if isinstance(surface, PlacedSurface)
                else 1.0,
            )
            if swept is not None:
                rough = np.flatnonzero(np.isnan(swept))
                if rough.size:
                    if reserve_queries is not None:
                        reserve_queries(int(rough.size))
                    charge_native_geometry_queries(int(rough.size))
                    first_lower, first_upper = surface.derivative_bounds_batch(
                        boxes[rough], order=1
                    )
                    swept[rough] = np.nextafter(
                        np.asarray(
                            [
                                _first_jet_chord_remainder(low, high, widths[row])
                                for low, high, row in zip(
                                    first_lower, first_upper, rough, strict=True
                                )
                            ]
                        )
                        * (1 + 64 * np.finfo(np.float64).eps),
                        np.inf,
                    )
                return swept
            lower, upper = surface.derivative_bounds_batch(boxes, order=2)
            continuous_spline = _continuous_spline_carrier(surface)
            c0_rows = np.asarray(
                [
                    row
                    for row, box in enumerate(boxes)
                    if continuous_spline and not surface.is_c1_on(box)
                ],
                dtype=np.int64,
            )
            first_jets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if c0_rows.size:
                if reserve_queries is not None:
                    reserve_queries(int(c0_rows.size))
                charge_native_geometry_queries(int(c0_rows.size))
                first_lower_raw, first_upper_raw = surface.derivative_bounds_batch(
                    boxes[c0_rows],
                    order=1,
                )
                expected = (c0_rows.size, 3, 2)
                if (
                    np.shape(first_lower_raw) != expected
                    or np.shape(first_upper_raw) != expected
                ):
                    raise ValueError(
                        "First-jet source bounds have an invalid batch shape."
                    )
                first_lower = np.empty(expected, dtype=np.float64)
                first_upper = np.empty(expected, dtype=np.float64)
                first_lower[...] = first_lower_raw
                first_upper[...] = first_upper_raw
                first_jets = {
                    int(row): (first_lower[index], first_upper[index])
                    for index, row in enumerate(c0_rows)
                }
            for row, (low, high, width) in enumerate(
                zip(lower, upper, widths, strict=True)
            ):
                if row in first_jets:
                    first_lower_row, first_upper_row = first_jets[row]
                    bounds.append(
                        _first_jet_chord_remainder(
                            first_lower_row, first_upper_row, width
                        )
                    )
                    continue
                absolute = np.maximum(np.abs(low), np.abs(high))
                h = np.linalg.norm(absolute, axis=0)
                if np.any(~np.isfinite(h)) and np.any(width > 0):
                    bounds.append(np.inf)
                else:
                    bounds.append(float(width @ h @ width / 8))
            return np.nextafter(
                np.asarray(bounds) * (1 + 64 * np.finfo(np.float64).eps), np.inf
            )
        du, dv = widths.T
        huu = np.full((coordinates.shape[0],), hessian[0])
        huv = np.full((coordinates.shape[0],), hessian[1])
        definition = surface.definition if isinstance(surface, PlacedSurface) else surface
        sphere = sphere_source_equivalence(definition)
        if sphere is not None:
            definition = sphere[0]
        elif isinstance(definition, OffsetSurface):
            equivalent = definition.analytic_equivalent()
            if equivalent is not None:
                definition = equivalent
        lower = np.min(coordinates[..., 1], axis=1)
        upper = np.max(coordinates[..., 1], axis=1)
        if isinstance(definition, SpherePatch):
            closest = np.where(
                (lower <= 0) & (upper >= 0), 0.0, np.minimum(np.abs(lower), np.abs(upper))
            )
            huu *= np.maximum(np.cos(closest), 0.0)
        elif isinstance(definition, TorusPatch) and float(
            definition.major_radius
        ) > float(definition.minor_radius):
            # |x_uu| = (R + r cos v)|radial| and |x_uv| = r |sin v| |radial|;
            # the global norms take cos v = |sin v| = 1 on every triangle.
            major, minor = float(definition.major_radius), float(definition.minor_radius)
            cosine = np.where(
                np.ceil(lower / (2 * np.pi)) <= np.floor(upper / (2 * np.pi)),
                1.0,
                np.maximum(np.cos(lower), np.cos(upper)),
            )
            sine = np.where(
                np.ceil((lower - 0.5 * np.pi) / np.pi)
                <= np.floor((upper - 0.5 * np.pi) / np.pi),
                1.0,
                np.maximum(np.abs(np.sin(lower)), np.abs(np.sin(upper))),
            )
            huu *= np.clip((major + minor * cosine) / (major + minor), 0.0, 1.0)
            huv *= np.clip(sine, 0.0, 1.0)
        bound = (huu * du * du + 2 * huv * du * dv + hessian[2] * dv * dv) / 8
        return np.nextafter(bound * (1 + 64 * np.finfo(np.float64).eps), np.inf)

    def normal_turn_bounds(self, patch: int, charts: np.ndarray, /) -> np.ndarray:
        """Continuous normal diameter; singular interval charts stay unresolved."""
        surface = self.patches[patch].surface
        source_surface = surface
        coordinates = np.asarray(charts, dtype=np.float64)
        widths = np.ptp(coordinates, axis=1)
        placement_distortion = 1.0
        if isinstance(surface, PlacedSurface):
            placement_distortion = _matrix_distortion(np.asarray(surface.rotation))
            surface = surface.definition
        sphere = sphere_source_equivalence(surface)
        if sphere is not None and sphere[1] != 0:
            # The proved offset tree retains the original sphere Gauss map.
            surface = sphere[0]
        elif isinstance(surface, OffsetSurface):
            equivalent = surface.analytic_equivalent()
            if equivalent is not None:
                surface = equivalent
        if isinstance(surface, PlanePatch):
            regular = (
                np.linalg.norm(
                    np.cross(
                        np.asarray(surface.first_axis), np.asarray(surface.second_axis)
                    )
                )
                > 0
            )
            return np.full((coordinates.shape[0],), 0.0 if regular else np.inf)
        if isinstance(surface, (ConePatch, CylinderPatch, SpherePatch, TorusPatch)):
            distortion = _basis_distortion(surface) * placement_distortion
            du, dv = widths.T
            if isinstance(surface, (ConePatch, CylinderPatch)):
                result = distortion * du
            else:
                lower = np.min(coordinates[..., 1], axis=1)
                upper = np.max(coordinates[..., 1], axis=1)
                crossing = np.ceil(lower / np.pi) <= np.floor(upper / np.pi)
                cosine = np.where(
                    crossing,
                    1.0,
                    np.maximum(np.abs(np.cos(lower)), np.abs(np.cos(upper))),
                )
                result = distortion * (cosine * du + dv)
            return np.nextafter(result * (1 + 128 * np.finfo(np.float64).eps), np.inf)
        boxes = np.stack(
            (np.min(coordinates, axis=1), np.max(coordinates, axis=1)), axis=1
        )
        revolution, distance = None, 0.0
        if isinstance(surface, RevolutionSurface):
            revolution = surface
        elif isinstance(surface, OffsetSurface) and isinstance(
            surface.base, RevolutionSurface
        ):
            revolution, distance = surface.base, float(surface.distance)
        turns = (
            None
            if revolution is None
            else _meridian_normal_turns(revolution, distance, coordinates)
        )
        if turns is None:
            result = _jet_normal_turns(source_surface, boxes, widths)
        else:
            result = placement_distortion * turns
            unresolved = np.isnan(turns)
            if np.any(unresolved):
                result[unresolved] = _jet_normal_turns(
                    source_surface, boxes[unresolved], widths[unresolved]
                )
        return np.nextafter(result * (1 + 128 * np.finfo(np.float64).eps), np.inf)

    def curve_interpolation_bounds(self, curve: int, values: np.ndarray, /) -> np.ndarray:
        """Continuous chord bounds, including non-C1 source chart transitions."""
        boundary_map = self.curve_atlas.mapping
        if isinstance(boundary_map, _PhysicalCurveMap):
            first, last = np.asarray(boundary_map.ranges[curve])
        else:
            patch, loop, position = self.curve_owners[curve]
            use = _curve_use(self.patches[patch].loops[loop][position])
            first, last = use.first, use.last
        span = last - first
        known_second = None
        if isinstance(boundary_map, _PhysicalCurveMap):
            candidate = _curve_second_bound(boundary_map.curves[curve])
            if np.isfinite(candidate):
                known_second = candidate
        else:
            surface, carrier = self.patches[patch].surface, use.pcurve
            if isinstance(carrier, LineCurve):
                hessian = _hessian_bounds(surface)
                if hessian is not None:
                    direction = np.abs(np.asarray(carrier.direction))
                    known_second = (
                        hessian[0] * direction[0] ** 2
                        + 2 * hessian[1] * direction[0] * direction[1]
                        + hessian[2] * direction[1] ** 2
                    )
            elif isinstance(surface, PlanePatch) and isinstance(carrier, CircleCurve):
                axes = np.stack(
                    (np.asarray(surface.first_axis), np.asarray(surface.second_axis))
                )
                known_second = float(carrier.radius) * _matrix_norm_upper(
                    np.column_stack(
                        (
                            np.asarray(carrier.first_axis) @ axes,
                            np.asarray(carrier.second_axis) @ axes,
                        )
                    )
                )
        bounds = []
        for start, end in zip(values[:-1], values[1:], strict=True):
            lower, upper = sorted(
                (float(first + span * start), float(first + span * end))
            )
            if isinstance(boundary_map, _PhysicalCurveMap):
                bound = _physical_curve_chord_bound(
                    boundary_map.curves[curve], lower, upper, known_second=known_second
                )
            else:
                patch, loop, position = self.curve_owners[curve]
                use = _curve_use(self.patches[patch].loops[loop][position])
                bound = _chart_curve_chord_bound(
                    self.patches[patch].surface,
                    use.pcurve,
                    lower,
                    upper,
                    known_second=known_second,
                )
            bounds.append(bound)
        result = np.asarray(bounds, dtype=np.float64)
        if isinstance(boundary_map, (_PhysicalCurveMap, _DomainCurveMap)):
            result += float(np.max(boundary_map.endpoint_errors[curve]))
        return np.nextafter(result * (1 + 64 * np.finfo(np.float64).eps), np.inf)

    def trim_ribbon_bounds(
        self,
        patch: int,
        use: PatchCurveUse,
        parameters: np.ndarray,
        points: np.ndarray,
        /,
        *,
        intervals: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-source-arc physical ribbon bounds for one boundary discretization.

        ``intervals`` selects arcs ``[parameters[i], parameters[i + 1]]`` of the
        complete discretization; each selected bound equals its full-batch value.
        """
        patch_ = _index(patch, "patch")
        if patch_ >= len(self.patches) or not isinstance(use, PatchCurveUse):
            raise ValueError("A trim ribbon requires one present patch curve use.")
        values = np.asarray(parameters, dtype=np.float64)
        physical = np.asarray(points, dtype=np.float64)
        if (
            values.ndim != 1
            or values.size < 2
            or physical.shape != (values.size, 3)
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(physical))
            or values[0] != use.first
            or values[-1] != use.last
            or np.any(np.diff(values) * (use.last - use.first) <= 0.0)
        ):
            raise ValueError("Trim-ribbon parameters must cover one ordered source use.")
        arcs = np.arange(values.size - 1, dtype=np.int64)
        if intervals is not None:
            selected = np.asarray(intervals)
            if (
                selected.ndim != 1
                or not np.issubdtype(selected.dtype, np.integer)
                or np.any(selected < 0)
                or np.any(selected >= values.size - 1)
            ):
                raise ValueError("Trim-ribbon intervals must index source arcs.")
            arcs = selected.astype(np.int64)
            if not arcs.size:
                return np.empty((0,), dtype=np.float64)
        normalized = (values - use.first) / (use.last - use.first)
        normalized[0], normalized[-1] = 0.0, 1.0
        curve = _domain_trim_curve(self.patches[patch_], use)
        edges = np.stack((arcs, arcs + 1), axis=1)
        nodes, local = np.unique(edges, return_inverse=True)
        charts = use.charts(values[nodes])
        return _trim_curve_interval_bounds(
            self,
            patch_,
            tuple(curve for _ in range(edges.shape[0])),
            normalized[arcs],
            normalized[arcs + 1],
            charts[local.reshape(edges.shape)],
            physical[edges],
        )

    @classmethod
    def from_brep(cls, model: object, /) -> MeshingDomain:
        """Prepare exact native B-Rep carriers, never its query tessellation.

        Degenerate source edges become pole uses; nondegenerate edges retain
        their original source indices and their authoritative 3D curves.
        Authored occurrence incidence determines placed strata and material
        sharing; equal positions or common path prefixes never infer identity.
        """
        if not isinstance(model, BRepModel):
            raise TypeError("model must be BRepModel.")
        geometry = model.geometry
        if geometry is None:
            raise ValueError("Native meshing requires exact B-Rep curves and trim loops.")
        if geometry.occurrences:
            return _placed_brep_domain(model)
        domain = cls.from_brep_geometry(
            geometry,
            model.patches,
            np.asarray(model.orientation),
            source_id=model.source_id,
            source_revision=model.source_revision,
            authority_id=model.model_id,
        )
        return eqx.tree_at(
            lambda value: value.brep_authority,
            domain,
            model,
            is_leaf=lambda value: value is None,
        )

    @classmethod
    def from_brep_geometry(
        cls,
        geometry: object,
        surfaces: tuple[AbstractSurfacePatch, ...],
        orientation: np.ndarray,
        /,
        *,
        source_id: str,
        source_revision: str,
        authority_id: str = "",
    ) -> MeshingDomain:
        """Prepare native CAD construction staging without tessellation recursion."""
        from .brep._constructors import brep_trim_curve

        if not isinstance(geometry, BRepGeometry):
            raise TypeError("geometry must be BRepGeometry.")
        signs = np.asarray(orientation, dtype=np.float64)
        if len(surfaces) != len(geometry.face_loops) or signs.shape != (len(surfaces),):
            raise ValueError("Surfaces and orientation must align with exact face loops.")
        if not np.all(np.isin(signs, (-1, 1))):
            raise ValueError("Face orientation must be -1 or 1.")
        ranges = np.asarray(geometry.edge_ranges)
        first_senses: dict[int, int] = {}
        for loops in geometry.face_loops:
            for loop in loops:
                for coedge in loop:
                    first_senses.setdefault(
                        geometry.coedge_edges[coedge], geometry.coedge_senses[coedge]
                    )
        source_edges = tuple(
            edge for edge in sorted(first_senses) if geometry.edge_curves[edge] >= 0
        )
        local_edges = {edge: local for local, edge in enumerate(source_edges)}
        source_vertices = tuple(
            sorted(
                {
                    vertex
                    for edge in first_senses
                    for vertex in geometry.edge_vertices[edge]
                }
            )
        )
        local_vertices = {vertex: local for local, vertex in enumerate(source_vertices)}
        original_endpoints = tuple(
            geometry.edge_vertices[edge]
            if first_senses[edge] > 0
            else geometry.edge_vertices[edge][::-1]
            for edge in source_edges
        )
        endpoints = tuple(
            (local_vertices[first], local_vertices[last])
            for first, last in original_endpoints
        )
        owner_ranges = np.asarray(
            [
                ranges[edge] if first_senses[edge] > 0 else ranges[edge][::-1]
                for edge in source_edges
            ]
        )
        patches = []
        for face, loops in enumerate(geometry.face_loops):
            uses = []
            for loop in loops:
                members = []
                for coedge in loop:
                    edge = geometry.coedge_edges[coedge]
                    first, last = ranges[edge]
                    if geometry.coedge_senses[coedge] < 0:
                        first, last = last, first
                    pcurve = geometry.pcurves[coedge]
                    if edge in local_edges:
                        roots = geometry.coedge_endpoint_roots[coedge]
                        vertices = geometry.edge_vertices[edge]
                        if geometry.coedge_senses[coedge] < 0:
                            roots, vertices = roots[::-1], vertices[::-1]
                        members.append(
                            PatchCurveUse(
                                local_edges[edge],
                                pcurve,
                                first,
                                last,
                                first_root=roots[0],
                                last_root=roots[1],
                                start_vertex_root=geometry.vertex_roots[vertices[0]],
                                end_vertex_root=geometry.vertex_roots[vertices[1]],
                                trim_curve=brep_trim_curve(
                                    geometry, coedge, surfaces[face]
                                ),
                            )
                        )
                    else:
                        ends = np.asarray(pcurve.evaluate(jnp.asarray((first, last))))
                        vertex = geometry.edge_vertices[edge][0]
                        members.append(
                            PatchPoleUse(
                                local_vertices[vertex],
                                tuple(ends[0]),
                                tuple(ends[1]),
                                trim_curve=brep_trim_curve(
                                    geometry, coedge, surfaces[face]
                                ),
                            )
                        )
                uses.append(tuple(members))
            patches.append(
                MeshingSurfacePatch(
                    surfaces[face],
                    tuple(uses),
                    reversed=signs[face] < 0,
                )
            )
        regions = []
        for solid, shells in enumerate(geometry.solid_shells):
            boundary = []
            for shell in shells:
                for face, sign in zip(
                    geometry.shell_faces[shell],
                    geometry.shell_orientations[shell],
                    strict=True,
                ):
                    boundary.append((face, sign))
            regions.append(MeshingDomainRegion(f"solid:{solid}", tuple(boundary)))
        domain = cls(
            tuple(patches),
            tuple(MeshingDomainCurve(*ends) for ends in endpoints),
            len(source_vertices),
            source_id=source_id,
            source_revision=source_revision,
            regions=tuple(regions),
            source_indices=(
                source_vertices,
                source_edges,
                tuple(range(len(patches))),
            ),
            source_kinds=("vertex", "edge", "face"),
            authority_id=authority_id or geometry.geometry_id,
        )
        atlas = BoundaryAtlas(
            _PhysicalCurveMap(
                tuple(
                    geometry.curves[geometry.edge_curves[edge]] for edge in source_edges
                ),
                owner_ranges,
                tuple(
                    geometry.edge_endpoint_roots[edge]
                    if first_senses[edge] > 0
                    else geometry.edge_endpoint_roots[edge][::-1]
                    for edge in source_edges
                ),
                np.asarray(geometry.vertex_evaluation_bounds)[
                    np.asarray(original_endpoints, dtype=np.int64).reshape((-1, 2))
                ],
            ),
            source_entity_ids=jnp.arange(len(source_edges), dtype=jnp.int32),
            source_id=source_id,
            physical_tags=tuple(f"edge:{edge}" for edge in source_edges),
        )
        points = np.array(
            np.asarray(geometry.vertex_points)[np.asarray(source_vertices)],
            dtype=np.float64,
            copy=True,
        )
        points.setflags(write=False)
        return eqx.tree_at(
            lambda value: (value.curve_atlas, value.corner_points, value.brep_authority),
            domain,
            (atlas, points, geometry),
            is_leaf=lambda value: value is None,
        )

    def require_current(self, source_id: str, source_revision: str, /) -> None:
        """Refuse a request bound to another source or a stale revision."""

        if (source_id, source_revision) != (self.source_id, self.source_revision):
            raise ValueError(
                f"The request binds {source_id!r} at revision {source_revision!r}, "
                f"but the meshing domain is {self.source_id!r} at revision "
                f"{self.source_revision!r}."
            )

    def evaluate(self, patches: np.ndarray, charts: np.ndarray, /) -> np.ndarray:
        """Physical points ``(k, 3)`` of patch parameters ``(k, 2)``."""

        rows, charts_ = self._batch(patches, charts)
        charge_native_geometry_queries(rows.size)
        result = np.zeros((rows.size, 3), dtype=np.float64)
        for patch in np.unique(rows).tolist():
            selected = rows == patch
            result[selected] = _evaluate(self.patches[patch].surface, charts_[selected])
        return result

    def oriented_normals(
        self, patches: np.ndarray, charts: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unit oriented normals ``(k, 3)`` and whether each chart point is regular.

        A point is regular where ``|d_u x d_v|`` exceeds the rounding bound of
        its evaluation; singular normals (poles, apexes) are returned as zero.
        """

        rows, charts_ = self._batch(patches, charts)
        charge_native_geometry_queries(rows.size)
        result = np.zeros((rows.size, 3), dtype=np.float64)
        regular = np.zeros((rows.size,), dtype=np.bool_)
        for patch in np.unique(rows).tolist():
            selected = rows == patch
            value = self.patches[patch]
            differential = _differential(value.surface, charts_[selected])
            normal = value.orientation * np.cross(
                differential[..., 0], differential[..., 1]
            )
            length = np.linalg.norm(normal, axis=1)
            speeds = np.linalg.norm(differential, axis=1)
            bound = 64.0 * np.finfo(np.float64).eps * np.prod(speeds, axis=1)
            ok = (length > bound) & np.isfinite(length)
            parameter_box = np.stack(
                (np.min(charts_[selected], axis=0), np.max(charts_[selected], axis=0))
            )
            for axis, coordinate in value.surface.degenerate_isolines(parameter_box):
                ok &= charts_[selected, axis] != coordinate
            normal[ok] /= length[ok, None]
            normal[~ok] = 0.0
            result[selected] = normal
            regular[selected] = ok
        return result, regular

    def oriented_pole_limits(
        self, patch: int, charts: np.ndarray, center: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact analytic normal limits on explicitly declared collapsed sides."""
        source = self.patches[patch]
        definition = (
            source.surface.definition
            if isinstance(source.surface, PlacedSurface)
            else source.surface
        )
        transform = (
            np.asarray(source.surface.rotation)
            if isinstance(source.surface, PlacedSurface)
            else np.eye(3)
        )
        result = np.zeros((charts.shape[0], 3), dtype=np.float64)
        declared = np.zeros((charts.shape[0],), dtype=np.bool_)
        for loop in source.loops:
            for use in loop:
                if isinstance(use, PatchPoleUse):
                    first, last = np.asarray(use.start), np.asarray(use.end)
                    declared |= (
                        (
                            exact_orient2d(
                                np.broadcast_to(first, charts.shape),
                                np.broadcast_to(last, charts.shape),
                                charts,
                            )
                            == 0
                        )
                        & np.all(charts >= np.minimum(first, last), axis=1)
                        & np.all(charts <= np.maximum(first, last), axis=1)
                    )
        sphere = sphere_source_equivalence(definition)
        if sphere is not None and sphere[1] != 0:
            original, _ = sphere
            _, supported_poles = sphere_source_pole_normal_limits(definition, charts)
            declared &= supported_poles
            first = transform @ np.asarray(original.first_axis)
            second = transform @ np.asarray(original.second_axis)
            result = np.sign(charts[:, 1])[:, None] * np.cross(first, second)[None]
        elif isinstance(definition, ConePatch):
            angles = charts[:, 0]
            radial = np.cos(angles)[:, None] * np.asarray(definition.first_axis) + np.sin(
                angles
            )[:, None] * np.asarray(definition.second_axis)
            tangent = -np.sin(angles)[:, None] * np.asarray(
                definition.first_axis
            ) + np.cos(angles)[:, None] * np.asarray(definition.second_axis)
            slope = float(np.tan(float(definition.semi_angle)))
            radial_sign = np.sign(slope * (center[1] - charts[:, 1]))
            result = radial_sign[:, None] * np.cross(
                tangent @ transform.T,
                (np.asarray(definition.axis)[None] + slope * radial) @ transform.T,
            )
        else:
            declared[:] = False
        lengths = np.linalg.norm(result, axis=1)
        supported = declared & np.isfinite(lengths) & (lengths > 0)
        result[supported] *= source.orientation / lengths[supported, None]
        result[~supported] = 0.0
        return result, supported

    def project(
        self, points: np.ndarray, patches: np.ndarray, seeds: np.ndarray, /
    ) -> MeshingDomainProjection:
        """Closest points of ``points`` on the requested patches from chart seeds.

        Solves the stationarity ``d X(uv)^T (X(uv) - p) = 0`` with the bounded
        native vector root; the status is the root's convergence evidence.
        """

        points_ = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        rows, seeds_ = self._batch(patches, seeds)
        if points_.shape[0] != rows.size:
            raise ValueError("points, patches and seeds must align.")
        charge_native_geometry_queries(rows.size)
        parameters = np.zeros_like(seeds_)
        converged = np.zeros((rows.size,), dtype=np.bool_)
        plan = VectorLocalRootPlan(
            2,
            maximum_steps=_PROJECTION_STEPS,
            tolerance=1.0e-12,
            plan_id="meshing-domain-closest-point",
        )
        scale = self.scale
        for patch in np.unique(rows).tolist():
            selected = rows == patch
            surface = self.patches[patch].surface

            def solve(point: Array, seed: Array) -> tuple[Array, Array]:
                def residual(uv: Array) -> Array:
                    jacobian = jax.jacfwd(surface.evaluate)(uv)
                    return jacobian.T @ (surface.evaluate(uv) - point) / (scale * scale)

                root, diagnostics = plan.solve_with_diagnostics(residual, seed)
                return root, diagnostics.converged

            roots, status = jax.vmap(solve)(
                jnp.asarray(points_[selected]), jnp.asarray(seeds_[selected])
            )
            parameters[selected] = np.asarray(roots, dtype=np.float64)
            converged[selected] = np.asarray(status, dtype=np.bool_)
        closest = self.evaluate(rows, parameters)
        return MeshingDomainProjection(
            closest,
            parameters,
            np.linalg.norm(closest - points_, axis=1),
            converged,
        )

    def _batch(
        self, patches: np.ndarray, charts: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(patches, dtype=np.int64).reshape((-1,))
        charts_ = np.asarray(charts, dtype=np.float64).reshape((-1, 2))
        if rows.size != charts_.shape[0]:
            raise ValueError("patches and chart points must align.")
        if np.any((rows < 0) | (rows >= len(self.patches))):
            raise ValueError("patch indices must name patches of the domain.")
        return rows, charts_


def _placed_brep_domain(model: BRepModel) -> MeshingDomain:
    """Lower the geometry-owned qualified incidence graph without spatial welding."""
    from .brep._constructors import brep_trim_curve
    from .brep._model import BRepEntityId
    from .brep._projection_contracts import brep_entity_id

    geometry = model.geometry
    if geometry is None:
        raise ValueError("Placed source lowering requires authoritative B-Rep geometry.")
    records = geometry.qualified_entity_incidence(model.source_revision)
    outgoing = {}
    for record in records:
        if record.container != record.member:
            outgoing.setdefault(record.container, []).append(record)
    occurrences = {occurrence.path: occurrence for occurrence in geometry.occurrences}
    placements = {}
    solid_ids = []
    for occurrence in geometry.occurrences:
        solid = BRepEntityId(
            model.source_revision, "solid", occurrence.solid, occurrence.path
        )
        solid_ids.append(solid)
        pending, seen = [solid], set()
        while pending:
            entity = pending.pop()
            if entity in seen:
                continue
            seen.add(entity)
            old = placements.setdefault(entity, occurrence)
            if (
                old.rotation != occurrence.rotation
                or old.translation != occurrence.translation
            ):
                raise ValueError(
                    "Authored shared source strata have inconsistent occurrence placements."
                )
            pending.extend(record.member for record in outgoing.get(entity, ()))
    face_ids = sorted(
        {record.member for record in records if record.member.kind == "face"}
    )
    edge_ids = sorted(
        {record.member for record in records if record.member.kind == "edge"}
    )
    vertex_ids = sorted(
        {record.member for record in records if record.member.kind == "vertex"}
    )
    nondegenerate = [edge for edge in edge_ids if geometry.edge_curves[edge.index] >= 0]
    faces = {entity: index for index, entity in enumerate(face_ids)}
    edges = {entity: index for index, entity in enumerate(nondegenerate)}
    vertices = {entity: index for index, entity in enumerate(vertex_ids)}
    transforms = {
        path: (
            jnp.asarray(occurrence.rotation, dtype=jnp.float64),
            jnp.asarray(occurrence.translation, dtype=jnp.float64),
        )
        for path, occurrence in occurrences.items()
    }

    def transform(entity: BRepEntityId) -> tuple[Array, Array]:
        occurrence = placements.get(entity)
        return (
            (jnp.eye(3, dtype=jnp.float64), jnp.zeros((3,), dtype=jnp.float64))
            if occurrence is None
            else transforms[occurrence.path]
        )

    def child(
        container: BRepEntityId,
        kind: Literal["vertex", "edge", "face", "solid"],
        *,
        definition_index: int | None = None,
        use_index: int | None = None,
    ) -> BRepEntityId:
        matches = [
            record.member
            for record in outgoing.get(container, ())
            if record.member.kind == kind
            and (definition_index is None or record.member.index == definition_index)
            and (use_index is None or record.use_index == use_index)
        ]
        if len(set(matches)) != 1:
            raise ValueError(
                "Qualified native incidence must identify one exact source use."
            )
        return matches[0]

    patch_values, first_senses = [], {}
    ranges = np.asarray(geometry.edge_ranges)
    for face in face_ids:
        loops = []
        for loop in geometry.face_loops[face.index]:
            uses = []
            for coedge in loop:
                edge = child(face, "edge", use_index=coedge)
                first, last = ranges[edge.index]
                sense = geometry.coedge_senses[coedge]
                if sense < 0:
                    first, last = last, first
                first_senses.setdefault(edge, sense)
                pcurve = geometry.pcurves[coedge]
                if edge in edges:
                    roots = geometry.coedge_endpoint_roots[coedge]
                    endpoints = geometry.edge_vertices[edge.index]
                    if sense < 0:
                        roots, endpoints = roots[::-1], endpoints[::-1]
                    uses.append(
                        PatchCurveUse(
                            edges[edge],
                            pcurve,
                            first,
                            last,
                            first_root=roots[0],
                            last_root=roots[1],
                            start_vertex_root=geometry.vertex_roots[endpoints[0]],
                            end_vertex_root=geometry.vertex_roots[endpoints[1]],
                            trim_curve=brep_trim_curve(
                                geometry, coedge, model.patches[face.index]
                            ),
                        )
                    )
                else:
                    vertex = child(
                        edge,
                        "vertex",
                        definition_index=geometry.edge_vertices[edge.index][0],
                    )
                    points = np.asarray(pcurve.evaluate(jnp.asarray((first, last))))
                    uses.append(
                        PatchPoleUse(
                            vertices[vertex],
                            tuple(points[0]),
                            tuple(points[1]),
                            trim_curve=brep_trim_curve(
                                geometry, coedge, model.patches[face.index]
                            ),
                        )
                    )
            loops.append(tuple(uses))
        rotation, translation = transform(face)
        patch_values.append(
            MeshingSurfacePatch(
                PlacedSurface(model.patches[face.index], rotation, translation),
                tuple(loops),
                reversed=float(model.orientation[face.index]) < 0,
            )
        )
    curve_values, curve_carriers, owner_ranges, vertex_errors = [], [], [], []
    for edge in nondegenerate:
        endpoints = geometry.edge_vertices[edge.index]
        if first_senses[edge] < 0:
            endpoints = endpoints[::-1]
        start, end = [
            child(edge, "vertex", definition_index=endpoint) for endpoint in endpoints
        ]
        curve_values.append(MeshingDomainCurve(vertices[start], vertices[end]))
        rotation, translation = transform(edge)
        curve_carriers.append(
            PlacedCurve(
                geometry.curves[geometry.edge_curves[edge.index]], rotation, translation
            )
        )
        owner_ranges.append(
            ranges[edge.index] if first_senses[edge] > 0 else ranges[edge.index][::-1]
        )
        original_points = np.asarray(geometry.vertex_points)[
            np.asarray(endpoints, dtype=np.int64)
        ]
        placement_roundoff = (
            64
            * np.finfo(np.float64).eps
            * np.linalg.norm(
                np.abs(np.asarray(rotation)) @ np.abs(original_points).T
                + np.abs(np.asarray(translation))[:, None],
                axis=0,
            )
        )
        vertex_errors.append(
            np.nextafter(
                _matrix_norm_upper(np.asarray(rotation))
                * np.asarray(geometry.vertex_evaluation_bounds)[
                    np.asarray(endpoints, dtype=np.int64)
                ]
                + placement_roundoff,
                np.inf,
            )
        )
    regions = tuple(
        MeshingDomainRegion(
            brep_entity_id(
                model.source_revision,
                3,
                solid.index,
                occurrence_path=solid.occurrence_path,
            ),
            tuple(
                (faces[record.member], record.orientation)
                for record in outgoing.get(solid, ())
                if record.member.kind == "face"
            ),
        )
        for solid in solid_ids
    )
    strata = (vertex_ids, nondegenerate, face_ids)
    domain = MeshingDomain(
        tuple(patch_values),
        tuple(curve_values),
        len(vertex_ids),
        source_id=model.source_id,
        source_revision=model.source_revision,
        regions=regions,
        source_indices=tuple(tuple(entity.index for entity in row) for row in strata),
        source_occurrences=tuple(
            tuple(entity.occurrence_path for entity in row) for row in strata
        ),
        region_source_indices=tuple(solid.index for solid in solid_ids),
        region_source_occurrences=tuple(solid.occurrence_path for solid in solid_ids),
        source_kinds=("vertex", "edge", "face"),
        authority_id=model.model_id,
    )
    point_rows = []
    for vertex in vertex_ids:
        rotation, translation = transform(vertex)
        point_rows.append(
            np.asarray(rotation) @ np.asarray(geometry.vertex_points)[vertex.index]
            + np.asarray(translation)
        )
    points = np.asarray(point_rows, dtype=np.float64).reshape((-1, 3))
    points.setflags(write=False)
    atlas = BoundaryAtlas(
        _PhysicalCurveMap(
            tuple(curve_carriers),
            np.asarray(owner_ranges),
            tuple(
                geometry.edge_endpoint_roots[edge.index]
                if first_senses[edge] > 0
                else geometry.edge_endpoint_roots[edge.index][::-1]
                for edge in nondegenerate
            ),
            np.asarray(vertex_errors, dtype=np.float64).reshape((-1, 2)),
        ),
        source_entity_ids=jnp.arange(len(curve_values), dtype=jnp.int32),
        source_id=model.source_id,
        physical_tags=tuple(
            domain.entity_id(1, index) for index in range(len(curve_values))
        ),
    )
    return eqx.tree_at(
        lambda value: (value.corner_points, value.curve_atlas, value.brep_authority),
        domain,
        (points, atlas, model),
        is_leaf=lambda value: value is None,
    )


def _curve_use(use: PatchBoundaryUse, /) -> PatchCurveUse:
    if not isinstance(use, PatchCurveUse):
        raise TypeError("A curve owner must be a PatchCurveUse.")
    return use


def _patch_identity(patch: MeshingSurfacePatch, /) -> dict[str, object]:
    from .brep._model import _endpoint_root_payload, _vertex_root_payload

    uses = []
    for loop in patch.loops:
        members = []
        for use in loop:
            match use:
                case PatchCurveUse():
                    members.append(
                        {
                            "curve": use.curve,
                            "pcurve": type(use.pcurve).__name__,
                            "pcurve_data": array_tree_fingerprint(
                                eqx.filter(use.pcurve, eqx.is_array)
                            ),
                            "range": [use.first, use.last],
                            "endpoint_roots": tuple(
                                None if root is None else _endpoint_root_payload(root)
                                for root in (use.first_root, use.last_root)
                            ),
                            "vertex_roots": tuple(
                                None if root is None else _vertex_root_payload(root)
                                for root in (use.start_vertex_root, use.end_vertex_root)
                            ),
                            "trim_curve": None
                            if use.trim_curve is None
                            else array_tree_fingerprint(use.trim_curve),
                        }
                    )
                case PatchPoleUse():
                    members.append(
                        {
                            "pole": use.corner,
                            "side": [use.start, use.end],
                            "trim_curve": None
                            if use.trim_curve is None
                            else array_tree_fingerprint(use.trim_curve),
                        }
                    )
                case _:
                    raise TypeError("Patch boundary uses are curve or pole uses.")
        uses.append(members)
    return {
        "surface": type(patch.surface).__name__,
        "surface_data": array_tree_fingerprint(eqx.filter(patch.surface, eqx.is_array)),
        "loops": uses,
        "reversed": patch.reversed,
    }


def _curve_owners(
    patches: tuple[MeshingSurfacePatch, ...],
    curves: tuple[MeshingDomainCurve, ...],
    corners: int,
    /,
) -> list[tuple[int, int, int]]:
    """First use of every curve; every curve and corner must be used."""

    owners: dict[int, tuple[int, int, int]] = {}
    used_corners: set[int] = set()
    for patch, loop, position in _uses(patches):
        use = patches[patch].loops[loop][position]
        match use:
            case PatchCurveUse():
                if use.curve >= len(curves):
                    raise ValueError(f"Curve use names undeclared curve {use.curve}.")
                owners.setdefault(use.curve, (patch, loop, position))
            case PatchPoleUse():
                if use.corner >= corners:
                    raise ValueError(f"Pole use names undeclared corner {use.corner}.")
                used_corners.add(use.corner)
            case _:
                raise TypeError("Patch boundary uses are curve or pole uses.")
    missing = sorted(set(range(len(curves))) - set(owners))
    if missing:
        raise ValueError(f"Declared curves {missing} bound no patch.")
    for curve in curves:
        if curve.start >= corners or curve.end >= corners:
            raise ValueError("Curve endpoints must name declared corners.")
        used_corners.update((curve.start, curve.end))
    if used_corners != set(range(corners)):
        raise ValueError("Every declared corner must bound a curve or a pole side.")
    return [owners[curve] for curve in range(len(curves))]


def _corner_points(
    patches: tuple[MeshingSurfacePatch, ...],
    curves: tuple[MeshingDomainCurve, ...],
    owners: list[tuple[int, int, int]],
    corners: int,
    /,
) -> tuple[np.ndarray, float]:
    """Corner positions from owner curve endpoints (or pole sides) and the scale."""

    points = np.full((corners, 3), np.nan, dtype=np.float64)
    root_ids: dict[int, str] = {}
    for curve, (patch, loop, position) in zip(curves, owners, strict=True):
        use = _curve_use(patches[patch].loops[loop][position])
        surface = patches[patch].surface
        ends = _evaluate(surface, _corner_use_endpoints(use))
        if ends.shape != (2, 3):
            raise ValueError(
                "Owner curve endpoints must be two three-dimensional source realizations."
            )
        if use.start_vertex_root is not None or use.end_vertex_root is not None:
            ends = np.array(ends, dtype=np.float64, copy=True)
        for end, vertex in enumerate((use.start_vertex_root, use.end_vertex_root)):
            if vertex is not None:
                corner = (curve.start, curve.end)[end]
                old = root_ids.setdefault(corner, vertex.root_id)
                if old != vertex.root_id:
                    raise ValueError(
                        "An authored corner cannot name two independent implicit roots."
                    )
                point, _, certified = vertex.evaluate()
                if not certified:
                    raise ValueError("A source corner root realization is unresolved.")
                if isinstance(surface, PlacedSurface) and not (
                    isinstance(vertex.primary, BRepPlacedVertex)
                    and vertex.primary.matches_pose(surface)
                ):
                    point = np.asarray(surface.rotation) @ point + np.asarray(
                        surface.translation
                    )
                ends[end] = point
        for corner, point in ((curve.start, ends[0]), (curve.end, ends[1])):
            if np.isnan(points[corner, 0]):
                points[corner] = point
    # A closed native-period curve owns one physical corner even when its
    # first declared patch use carries a rounded one-turn representative.
    for patch, loop, position in _uses(patches):
        use = patches[patch].loops[loop][position]
        if not isinstance(use, PatchCurveUse):
            continue
        curve = curves[use.curve]
        if curve.start != curve.end:
            continue
        charts = _corner_use_endpoints(use)
        images = _evaluate(patches[patch].surface, charts)
        for end, root in enumerate((use.first_root, use.last_root)):
            if isinstance(root, NativePeriodEndpoint) and root.turns.denominator == 1:
                points[curve.start] = images[end]

    for patch, loop, position in _uses(patches):
        use = patches[patch].loops[loop][position]
        if isinstance(use, PatchPoleUse) and np.isnan(points[use.corner, 0]):
            points[use.corner] = _evaluate(
                patches[patch].surface, np.asarray((use.start,), dtype=np.float64)
            )[0]
    if not np.all(np.isfinite(points)):
        raise ValueError("Corner positions must be finite.")
    points.setflags(write=False)
    return points, max(1.0, float(np.max(np.abs(points), initial=0.0)))


def _validate_uses(
    patches: tuple[MeshingSurfacePatch, ...],
    curves: tuple[MeshingDomainCurve, ...],
    owners: list[tuple[int, int, int]],
    corner_points: np.ndarray,
    bound: float,
    /,
) -> None:
    """Every use reproduces its owner curve; pole sides collapse to their corner."""

    fractions = np.linspace(0.0, 1.0, _CONSISTENCY_SAMPLES)
    for patch, loop, position in _uses(patches):
        use = patches[patch].loops[loop][position]
        surface = patches[patch].surface
        match use:
            case PatchCurveUse():
                owner_patch, owner_loop, owner_position = owners[use.curve]
                owner = _curve_use(patches[owner_patch].loops[owner_loop][owner_position])
                if {use.first, use.last} != {owner.first, owner.last}:
                    raise ValueError(
                        f"Uses of curve {use.curve} must span its owner parameter range."
                    )
                parameters = owner.first + fractions * (owner.last - owner.first)
                reference = _evaluate(
                    patches[owner_patch].surface, owner.charts(parameters)
                )
                image = _evaluate(surface, use.charts(parameters))
                gaps = np.linalg.norm(image - reference, axis=1)
                allowed = np.full(gaps.shape, bound)
                for end, row in ((0, 0), (1, fractions.size - 1)):
                    use_end = end if use.first == owner.first else 1 - end
                    owner_vertex = (owner.start_vertex_root, owner.end_vertex_root)[end]
                    use_vertex = (use.start_vertex_root, use.end_vertex_root)[use_end]
                    if owner_vertex is not None or use_vertex is not None:
                        if (
                            owner_vertex is None
                            or use_vertex is None
                            or owner_vertex.root_id != use_vertex.root_id
                        ):
                            raise ValueError(
                                "Shared rooted coedges must name one authoritative vertex."
                            )
                    allowed[row] += _root_endpoint_error(
                        patches[owner_patch], owner, end, reference[row]
                    )
                    allowed[row] += _root_endpoint_error(
                        patches[patch], use, use_end, image[row]
                    )
                gap = float(np.max(gaps))
                if np.any(gaps > allowed):
                    raise ValueError(
                        f"Patch {patch} does not reproduce curve {use.curve}: "
                        f"gap {gap:.3e} exceeds {bound:.3e}."
                    )
                curve = curves[use.curve]
                corners = corner_points[[curve.start, curve.end]]
                ends = reference[[0, -1]]
                endpoint_allowance = np.asarray(
                    [
                        bound
                        + _root_endpoint_error(
                            patches[owner_patch], owner, end, ends[end]
                        )
                        + _root_endpoint_error(
                            patches[owner_patch], owner, end, corners[end]
                        )
                        for end in range(2)
                    ]
                )
                if np.any(np.linalg.norm(ends - corners, axis=1) > endpoint_allowance):
                    raise ValueError(
                        f"Curve {use.curve} does not end at its declared corners."
                    )
            case PatchPoleUse():
                image = _evaluate(surface, use.charts(fractions))
                gap = float(
                    np.max(np.linalg.norm(image - corner_points[use.corner], axis=1))
                )
                if gap > bound:
                    raise ValueError(
                        f"Pole side of patch {patch} does not collapse to corner "
                        f"{use.corner}: gap {gap:.3e} exceeds {bound:.3e}."
                    )
            case _:
                raise TypeError("Patch boundary uses are curve or pole uses.")


def _validate_loops(
    patches: tuple[MeshingSurfacePatch, ...], tolerance: float, /
) -> None:
    """Consecutive uses of every loop join in the parameter plane."""

    for index, patch in enumerate(patches):
        for loop in patch.loops:
            ends = [_use_endpoints(use) for use in loop]
            scale = max(1.0, max(float(np.max(np.abs(value))) for value in ends))
            for position, value in enumerate(ends):
                following = ends[(position + 1) % len(ends)]
                if float(np.max(np.abs(value[1] - following[0]))) > tolerance * scale:
                    previous, following_use = (
                        loop[position],
                        loop[(position + 1) % len(loop)],
                    )
                    if (
                        isinstance(previous, PatchCurveUse)
                        and isinstance(following_use, PatchCurveUse)
                        and previous.last_root is not None
                        and following_use.first_root is not None
                        and _domain_trim_curve(patch, previous).shares_endpoint(
                            _domain_trim_curve(patch, following_use)
                        )
                    ):
                        continue
                    raise ValueError(
                        f"A boundary loop of patch {index} is not closed in its "
                        "parameter plane."
                    )


def _validate_orientation(
    patches: tuple[MeshingSurfacePatch, ...], owners: list[tuple[int, int, int]], /
) -> None:
    """Two patches sharing a curve traverse it in opposite oriented directions."""

    directions: dict[int, list[float]] = {curve: [] for curve in range(len(owners))}
    for patch, loop, position in _uses(patches):
        use = patches[patch].loops[loop][position]
        if isinstance(use, PatchCurveUse):
            directions[use.curve].append(
                patches[patch].orientation * np.sign(use.last - use.first)
            )
    for curve, signs in directions.items():
        if len(signs) == 2 and signs[0] == signs[1]:
            raise ValueError(
                f"The patches sharing curve {curve} have inconsistent orientations."
            )


def _patch_regions(
    patches: tuple[MeshingSurfacePatch, ...],
    regions: tuple[MeshingDomainRegion, ...],
    /,
) -> np.ndarray:
    """Region on the negative and positive oriented side of each patch (-1: none)."""

    sides = np.full((len(patches), 2), -1, dtype=np.int64)
    for region_index, region in enumerate(regions):
        counts: dict[int, int] = {}
        for patch, side in region.boundary:
            if patch >= len(patches):
                raise ValueError(f"Region {region.name!r} names an undeclared patch.")
            # An outward oriented normal (+1) puts the region on the negative side.
            column = 0 if side == 1 else 1
            if sides[patch, column] >= 0:
                raise ValueError(
                    f"Patch {patch} bounds two regions on one oriented side."
                )
            sides[patch, column] = region_index
            for loop in patches[patch].loops:
                for use in loop:
                    if isinstance(use, PatchCurveUse):
                        counts[use.curve] = counts.get(use.curve, 0) + 1
        if any(count != 2 for count in counts.values()):
            raise ValueError(
                f"Region {region.name!r} is not bounded by a closed surface."
            )
    return sides


def _loop_polygon(loop: tuple[PatchBoundaryUse, ...], /) -> np.ndarray:
    pieces = []
    for use in loop:
        match use:
            case PatchCurveUse():
                values = np.linspace(use.first, use.last, _LOOP_SAMPLES)
                pieces.append(use.charts(values)[:-1])
            case PatchPoleUse():
                pieces.append(use.charts(np.asarray((0.0,))))
            case _:
                raise TypeError("Patch boundary uses are curve or pole uses.")
    return np.concatenate(pieces, axis=0)


def _polygon_sign(points: np.ndarray, /) -> int:
    area = Fraction(0)
    for first, second in zip(points, np.roll(points, -1, axis=0), strict=True):
        area += Fraction(float(first[0])) * Fraction(float(second[1]))
        area -= Fraction(float(first[1])) * Fraction(float(second[0]))
    return (area > 0) - (area < 0)


def _winding(point: np.ndarray, polygon: np.ndarray, /) -> int:
    following = np.roll(polygon, -1, axis=0)
    signs = exact_orient2d(polygon, following, np.broadcast_to(point, polygon.shape))
    up = (polygon[:, 1] <= point[1]) & (following[:, 1] > point[1]) & (signs > 0)
    down = (polygon[:, 1] > point[1]) & (following[:, 1] <= point[1]) & (signs < 0)
    return int(np.sum(up) - np.sum(down))


def _carrier_breakpoints(
    curve: AbstractCurve | AbstractTrimCurve | IntersectionCurve,
    first: float,
    last: float,
    /,
) -> np.ndarray:
    """Preserve source span/chart boundaries under exact authored affine maps."""
    from .brep._constructors import _NormalizedTrimCurve

    if isinstance(curve, PlacedCurve):
        return _carrier_breakpoints(curve.definition, first, last)
    if isinstance(curve, _NormalizedTrimCurve):
        return _carrier_breakpoints(curve.curve, first, last)
    if isinstance(curve, CurveTrimSegment):
        ends = np.asarray((curve._carrier(first), curve._carrier(last)), dtype=np.float64)
        points = _carrier_breakpoints(
            curve.curve, float(np.min(ends)), float(np.max(ends))
        )
        local = (points - curve.first) / (curve.last - curve.first)
        return np.sort(1.0 - local if curve.reversed else local)
    if isinstance(curve, IntersectionPCurve):
        ends = np.asarray((curve._carrier(first), curve._carrier(last)), dtype=np.float64)
        points = _carrier_breakpoints(
            curve.curve, float(np.min(ends)), float(np.max(ends))
        )
        return np.sort(curve.first + curve.last - points if curve.reversed else points)
    if isinstance(curve, IntersectionCurve):
        points = np.arange(np.ceil(first), np.floor(last) + 1, dtype=np.float64)
        return points[(points > first) & (points < last)]
    if isinstance(curve, BSplineCurve):
        points = np.unique(np.asarray(curve.knots))
        return points[(points > first) & (points < last)]
    return np.empty((0,), dtype=np.float64)


def _curve_c1_interval(
    curve: AbstractCurve | AbstractTrimCurve | IntersectionCurve,
    first: float,
    last: float,
    /,
) -> bool:
    """Whether a second-jet Taylor bound applies over this entire parameter arc."""
    from .brep._constructors import _NormalizedTrimCurve

    if isinstance(curve, PlacedCurve):
        return _curve_c1_interval(curve.definition, first, last)
    if isinstance(curve, _NormalizedTrimCurve):
        return _curve_c1_interval(curve.curve, first, last)
    if isinstance(curve, CurveTrimSegment):
        # The segment owns endpoint roots and its original chart range. Calling
        # the owner preserves certified source extensions instead of reducing
        # them to rounded carrier endpoints.
        return curve.is_c1_on(first, last)
    if isinstance(curve, IntersectionPCurve):
        ends = sorted((float(curve._carrier(first)), float(curve._carrier(last))))
        return _curve_c1_interval(curve.curve, ends[0], ends[1])
    if isinstance(curve, IntersectionCurve):
        # Closed source intervals touching an atlas node include both
        # one-sided graph velocities. Shared nodes prove C0, not C1; use the
        # canonical jet admission rather than a half-open chart-index guess.
        return curve.fully_certified and curve.is_c1_on(first, last)
    if isinstance(curve, SurfaceIsoparametricCurve):
        return curve.is_c1_on(first, last)
    if isinstance(curve, BSplineCurve):
        # Closed intervals touching a C0 knot (multiplicity >= degree) are
        # refused by the canonical order-two jet; admit exactly as it does so
        # the first-jet remainder, not an infinite second jet, bounds the arc.
        return curve.is_c1_on(first, last)
    return isinstance(curve, (LineCurve, CircleCurve, EllipseCurve))


def _first_jet_chord_remainder(
    lower: np.ndarray,
    upper: np.ndarray,
    width: np.ndarray,
    /,
) -> float:
    """Continuous piecewise-source interpolation from its complete first jet.

    Subtract the midpoint Jacobian, whose affine contribution cancels. Each
    derivative column then has Lipschitz constant half its interval diameter;
    barycentric mean absolute deviation is at most half the chart width.
    This is the same integral remainder used by the original extrusion owner,
    and does not assert a Hessian or C1 continuity across a source knot.
    """
    from ..discretization._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        _reserve_polynomial,
    )
    from ._planar_coverage import _reserve_fraction_work
    from ._source_interpolation import _norm_upper

    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(4 * lower.size, 256 + 32 * lower.size + 64 * width.size)
        _reserve_polynomial(0, 2 * width.size, 0, 2200)
    diameter = interval_subtract((upper, upper), (lower, lower))[1]
    terms = []
    for axis, extent in enumerate(width):
        if extent == 0:
            continue
        norm = _norm_upper(diameter[:, axis])
        if not np.isfinite(norm) or not np.isfinite(extent) or extent < 0:
            return np.inf
        terms.append((Fraction(norm), Fraction(float(np.nextafter(extent, np.inf)))))
    _reserve_fraction_work(tuple(terms), 3 * len(terms) + 1, 1, 2 * len(terms) + 1)
    bound = sum((norm * extent / 4 for norm, extent in terms), Fraction())
    if bound > Fraction(float(np.finfo(np.float64).max)):
        return np.inf
    value = float(bound)
    return value if Fraction(value) >= bound else float(np.nextafter(value, np.inf))


def _continuous_spline_carrier(surface: AbstractSurfacePatch, /) -> bool:
    """Whether a validated C0 spline carrier can require first-jet bounds.

    An offset is continuous across a base knot only where the base unit normal
    is; it is admitted only over a planar-meridian revolution, whose offset
    jets prove G1 at every straddled junction and stay unbounded otherwise.
    """
    while isinstance(surface, PlacedSurface):
        surface = surface.definition
    if isinstance(surface, OffsetSurface):
        return (
            isinstance(surface.base, RevolutionSurface)
            and _meridian_frame(surface.base, _coordinate_budget()) is not None
        )
    return isinstance(
        surface,
        (BSplineSurfacePatch, ExtrusionSurface, RevolutionSurface, RuledSurface),
    )


def _curve_chord_error(
    curve: AbstractTrimCurve, first: float, last: float, /
) -> np.ndarray:
    """Continuous chord error, including legal first-jet jumps at source spans."""
    if _curve_c1_interval(curve, first, last):
        lower, upper = curve.derivative_bounds(first, last, order=2)
        return np.maximum(np.abs(lower), np.abs(upper)) * (last - first) ** 2 / 8
    lower, upper = curve.derivative_bounds(first, last, order=1)
    # Integral remainders with f' in [lower, upper] give t(1-t) Δf' Δs.
    # Unlike an order-two union, this remains valid across proved C0 joins.
    return np.nextafter(upper - lower, np.inf) * (last - first) / 4


def _simple_boundary(
    charts: np.ndarray, boundary: np.ndarray, maximum_pairs: int, /
) -> tuple[bool, int, bool]:
    """Exact simplicity of a chart boundary; returns validity, pair work, exhaustion."""
    boxes = charts[boundary]
    candidates, work = overlapping_box_pairs(
        np.min(boxes, axis=1), np.max(boxes, axis=1), maximum_pairs
    )
    if candidates is None:
        return False, work, True
    first, second = boundary[candidates[:, 0]].T
    others = boundary[candidates[:, 1]]
    selected = np.all((others != first[:, None]) & (others != second[:, None]), axis=1)
    if not np.any(selected):
        return True, work, False
    aa, bb = charts[first[selected]], charts[second[selected]]
    c, d = charts[others[selected, 0]], charts[others[selected, 1]]
    s0, s1 = exact_orient2d(aa, bb, c), exact_orient2d(aa, bb, d)
    s2, s3 = exact_orient2d(c, d, aa), exact_orient2d(c, d, bb)
    contact = (s0 * s1 < 0) & (s2 * s3 < 0)
    for sign, point, start, end in (
        (s0, c, aa, bb),
        (s1, d, aa, bb),
        (s2, aa, c, d),
        (s3, bb, c, d),
    ):
        contact |= (sign == 0) & np.all(
            (point >= np.minimum(start, end)) & (point <= np.maximum(start, end)),
            axis=1,
        )
    return not bool(np.any(contact)), work, False


def _trim_curve_interval_bounds(
    domain: MeshingDomain,
    patch: int,
    curves: tuple[AbstractTrimCurve, ...],
    firsts: np.ndarray,
    lasts: np.ndarray,
    chart_edges: np.ndarray,
    point_edges: np.ndarray,
    /,
) -> np.ndarray:
    """Bound each source-curve arc against only its own chord segment."""
    count = len(curves)
    if (
        firsts.shape != (count,)
        or lasts.shape != (count,)
        or chart_edges.shape != (count, 2, 2)
        or point_edges.shape != (count, 2, 3)
    ):
        raise ValueError("Trim-ribbon source intervals and boundary edges must align.")
    source_boxes = np.asarray(
        [
            curve.enclosure(float(first), float(last))
            for curve, first, last in zip(curves, firsts, lasts, strict=True)
        ],
        dtype=np.float64,
    )
    boxes = np.stack(
        (
            np.minimum(source_boxes[:, 0], np.min(chart_edges, axis=1)),
            np.maximum(source_boxes[:, 1], np.max(chart_edges, axis=1)),
        ),
        axis=1,
    )
    surface = domain.patches[patch].surface
    first_jet = np.zeros((count,), dtype=np.bool_)
    if _continuous_spline_carrier(surface):
        first_jet = np.asarray(
            [not surface.is_c1_on(box) for box in boxes], dtype=np.bool_
        )
    second_lower = np.zeros((count, 3, 2, 2), dtype=np.float64)
    second_upper = np.zeros((count, 3, 2, 2), dtype=np.float64)
    second_rows = np.flatnonzero(~first_jet)
    if second_rows.size:
        lower, upper = surface.derivative_bounds_batch(boxes[second_rows], order=2)
        second_lower[second_rows], second_upper[second_rows] = lower, upper
    first_lower, first_upper = surface.derivative_bounds_batch(boxes, order=1)
    mapped_edges = domain.evaluate(
        np.full((2 * count,), patch),
        chart_edges.reshape((-1, 2)),
    ).reshape((-1, 2, 3))
    result = []
    for arc, (curve, first_, last_) in enumerate(zip(curves, firsts, lasts, strict=True)):
        first, last = float(first_), float(last_)
        nodal = float(
            np.max(np.linalg.norm(mapped_edges[arc] - point_edges[arc], axis=1))
        )
        jl, ju = first_lower[arc], first_upper[arc]
        hessian = np.linalg.norm(
            np.maximum(np.abs(second_lower[arc]), np.abs(second_upper[arc])),
            axis=0,
        )
        width = boxes[arc, 1] - boxes[arc, 0]
        endpoint_error = np.maximum(
            np.max(
                np.abs(np.asarray(curve.enclosure(first, first)) - chart_edges[arc, 0]),
                axis=0,
            ),
            np.max(
                np.abs(np.asarray(curve.enclosure(last, last)) - chart_edges[arc, 1]),
                axis=0,
            ),
        )
        chart_error = _curve_chord_error(curve, first, last) + np.nextafter(
            endpoint_error, np.inf
        )
        jacobian = np.linalg.norm(np.maximum(np.abs(jl), np.abs(ju)), axis=0)
        if first_jet[arc]:
            carrier_error = _first_jet_chord_remainder(jl, ju, width)
        else:
            carrier_error = float(width @ hessian @ width / 8)
        bound = carrier_error + float(jacobian @ chart_error) + nodal
        result.append(
            np.nextafter(
                bound * (1 + 256 * np.finfo(np.float64).eps)
                + 256 * np.finfo(np.float64).eps * domain.scale,
                np.inf,
            )
        )
    return np.asarray(result, dtype=np.float64)


def _trim_ribbon_bounds(
    domain: MeshingDomain,
    patch: int,
    cover: CurveTrimLoop,
    charts: np.ndarray,
    points: np.ndarray,
    edges: np.ndarray,
    /,
) -> np.ndarray:
    """Bound the complete source/chord homotopy per owning source-curve arc."""
    count = edges.shape[0]
    return _trim_curve_interval_bounds(
        domain,
        patch,
        tuple(cover.curves[int(curve)] for curve in cover.arc_curves[:count]),
        np.asarray(cover.arc_first[:count]),
        np.asarray(cover.arc_last[:count]),
        charts[edges],
        points[edges],
    )


def _boundary_cell_ribbon_bounds(
    cells: np.ndarray,
    boundary: np.ndarray,
    ribbon_bounds: np.ndarray,
    /,
) -> np.ndarray:
    """Assign each local boundary ribbon only to its incident source cell."""
    if ribbon_bounds.shape != (boundary.shape[0],):
        raise ValueError("Boundary ribbon bounds must align with source edges.")
    by_edge = {
        tuple(sorted((int(edge[0]), int(edge[1])))): float(bound)
        for edge, bound in zip(boundary, ribbon_bounds, strict=True)
    }
    result = np.zeros((cells.shape[0],), dtype=np.float64)
    for row, cell in enumerate(cells):
        result[row] = max(
            (
                by_edge.get(
                    tuple(sorted((int(cell[first]), int(cell[second])))),
                    0.0,
                )
                for first, second in ((0, 1), (1, 2), (2, 0))
            ),
            default=0.0,
        )
    return result


def _disjoint_trim_covers(
    covers: list[tuple[np.ndarray, np.ndarray]], maximum_pairs: int, /
) -> tuple[bool, int, bool]:
    """Bounded cross-loop homotopy separation, preserving every source hole."""
    if len(covers) < 2:
        return True, 0, False
    lower = np.concatenate([cover[0] for cover in covers])
    upper = np.concatenate([cover[1] for cover in covers])
    owners = np.concatenate(
        [np.full((cover[0].shape[0],), index) for index, cover in enumerate(covers)]
    )
    pairs, work = overlapping_box_pairs(lower, upper, maximum_pairs)
    if pairs is None:
        return False, work, True
    return not bool(np.any(owners[pairs[:, 0]] != owners[pairs[:, 1]])), work, False


def _certified_trim_cover(
    trim_curves: list[AbstractTrimCurve],
    partitions: list[np.ndarray],
    polygon: np.ndarray,
    relative_closure_tolerance: float,
    maximum_pairs: int,
    reserve_queries: Callable[[int], None] | None,
    /,
) -> tuple[CurveTrimLoop, CurveTrimLoop | None, int, int, bool]:
    """Prove source/chord isotopy of the original chord chain.

    Returns the original cover, its topology proof cover, the pair work, the
    largest proof arc count, and pair exhaustion. Only arcs named by an
    unresolved topology finding are bisected. Inserted chord vertices lie on
    their original chord segment at the affine image of the source parameter,
    so the proof binds exactly the original chord polygon; subdivision only
    tightens arc and homotopy boxes. Every subdivided cover is charged before
    construction, whether or not it certifies: per arc two enclosures, one
    endpoint evaluation, and one derivative bound.
    """
    count = polygon.shape[0]
    # Original arc rows: (curve, parameter start, parameter end), loop order.
    rows = [
        (curve, float(first), float(last))
        for curve, values in enumerate(partitions)
        for first, last in zip(values[:-1], values[1:], strict=True)
    ]
    if len(rows) != count:
        raise ValueError("Trim partitions must align with the original chord chain.")

    def chord(arc: int, parameter: float, /) -> np.ndarray:
        _, start, end = rows[arc]
        if parameter == start:
            return polygon[arc]
        return polygon[arc] + (parameter - start) / (end - start) * (
            polygon[(arc + 1) % count] - polygon[arc]
        )

    # Proof arcs: (original arc, first, last, bisection depth), loop order.
    arcs = [(arc, start, end, 0) for arc, (_, start, end) in enumerate(rows)]
    original: CurveTrimLoop | None = None
    used = 0
    while True:
        if original is not None and reserve_queries is not None:
            reserve_queries(4 * len(arcs))
        parameters = [
            np.asarray(
                [first for arc, first, _, _ in arcs if rows[arc][0] == curve]
                + [float(values[-1])]
            )
            for curve, values in enumerate(partitions)
        ]
        boxes = [
            trim_curves[rows[arc][0]].enclosure(first, last)
            for arc, first, last, _ in arcs
        ]
        tolerance = float(
            np.nextafter(max(np.linalg.norm(box[1] - box[0]) for box in boxes), np.inf)
        )
        cover = CurveTrimLoop(
            trim_curves,
            tolerance=max(tolerance, np.finfo(np.float64).tiny),
            maximum_arcs=65_536,
            arc_parameters=parameters,
            chord_vertices=np.asarray([chord(arc, first) for arc, first, _, _ in arcs]),
            relative_closure_tolerance=relative_closure_tolerance,
        )
        if original is None:
            original = cover
        if maximum_pairs - used < 1:
            return original, None, used, len(arcs), True
        topology = cover.certify_topology(maximum_pairs=maximum_pairs - used)
        used += topology.pairs_checked
        if topology.certified:
            return original, cover, used, len(arcs), False
        if topology.budget_exhausted:
            return original, None, used, len(arcs), True
        split = _trim_proof_splits(cover.chords, topology.unresolved_pairs)
        if any(arcs[arc][3] >= 32 for arc in split) or len(arcs) + len(split) > 65_536:
            return original, None, used, len(arcs) + len(split), False
        refined: list[tuple[int, float, float, int]] = []
        for index, (arc, first, last, depth) in enumerate(arcs):
            if index in split:
                middle = 0.5 * (first + last)
                refined.extend(
                    ((arc, first, middle, depth + 1), (arc, middle, last, depth + 1))
                )
            else:
                refined.append((arc, first, last, depth))
        arcs = refined


def _trim_proof_splits(
    chords: np.ndarray, unresolved: tuple[tuple[int, int], ...], /
) -> set[int]:
    """Proof arcs to bisect for each unresolved topology finding.

    A self-monotonicity or nonadjacent-overlap finding needs every named arc
    tighter. An adjacent pair across a junction is decided by its two chord
    lengths together; bisecting both keeps their ratio and can never resolve
    a sharp corner, so only the longer chord is bisected.
    """
    count = chords.shape[0]
    lengths = np.linalg.norm(np.roll(chords, -1, axis=0) - chords, axis=1)
    split: set[int] = set()
    for first, second in unresolved:
        if first != second and (second - first) % count in (1, count - 1):
            split.add(first if lengths[first] >= lengths[second] else second)
        else:
            split.update((first, second))
    return split


def _verify_chart_chain(
    domain: MeshingDomain,
    patch: int,
    charts: np.ndarray,
    points: np.ndarray,
    cells: np.ndarray,
    boundary: np.ndarray,
    provenance: np.ndarray,
    restriction_required: np.ndarray,
    restriction_vertices: np.ndarray,
    restriction_edges: np.ndarray,
    restriction_parameters: np.ndarray,
    findings: list[MeshCertificateFinding],
    resource_counts: list[tuple[str, int, int]],
    /,
    *,
    maximum_pairs: int = 200_000,
    reserve_queries: Callable[[int], None] | None = None,
    topology_only: bool = False,
) -> tuple[bool, np.ndarray, np.ndarray, tuple[str, ...]]:
    from ._mesh_certificates import MeshCertificateFinding

    maximum_pairs = min(200_000, maximum_pairs)
    empty = (False, np.empty((0, 3, 3)), np.empty((0,)), ())
    if (
        charts.ndim != 2
        or charts.shape[1] != 2
        or points.shape != (charts.shape[0], 3)
        or cells.ndim != 2
        or cells.shape[1] != 3
        or boundary.ndim != 2
        or boundary.shape[1] != 2
        or provenance.shape != (boundary.shape[0], 4)
        or not np.all(np.isfinite(charts))
        or not np.all(np.isfinite(points))
        or not cells.size
        or np.any(cells < 0)
        or np.any(cells >= charts.shape[0])
        or np.any(boundary < 0)
        or np.any(boundary >= charts.shape[0])
    ):
        return empty
    resource_counts.append((f"source_trim_arcs:{patch}", boundary.shape[0], 65_536))
    if boundary.shape[0] > 65_536:
        findings.append(
            MeshCertificateFinding(
                "source_trim_arc_capacity", "unresolved", "source_facet", (patch,)
            )
        )
        return empty
    try:
        if (
            restriction_edges.shape != (restriction_vertices.size, 2)
            or np.any(restriction_edges < 0)
            or np.any(restriction_edges >= charts.shape[0])
        ):
            raise ValueError("Chart restriction edges leave the source topology.")
        exact, restrictions, _ = validate_chart_restrictions(
            charts,
            restriction_required,
            restriction_vertices,
            restriction_edges,
            charts[restriction_edges]
            if restriction_vertices.size
            else np.empty((0, 2, 2), dtype=np.float64),
            restriction_parameters,
            domain.source_revision,
            patch,
        )
        resource_counts.append(
            (
                f"source_chart_restrictions:{patch}",
                len(restrictions),
                charts.shape[0],
            )
        )
    except (TypeError, ValueError):
        findings.append(
            MeshCertificateFinding(
                "source_chart_restriction_authority",
                "unresolved",
                "source_facet",
                (patch,),
            )
        )
        return empty
    signs = exact_orient2d(charts[cells[:, 0]], charts[cells[:, 1]], charts[cells[:, 2]])
    for row in np.flatnonzero(np.any(restriction_required[cells], axis=1)):
        first, second, third = cells[row]
        ax, ay = exact[first]
        bx, by = exact[second]
        cx, cy = exact[third]
        determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        signs[row] = (determinant > 0) - (determinant < 0)
    if np.any(signs <= 0):
        return empty
    chain: dict[tuple[int, int], int] = {}
    for first, second in np.concatenate(
        (cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]])
    ):
        key = (int(first), int(second))
        reverse = (key[1], key[0])
        if chain.get(reverse, 0):
            chain[reverse] -= 1
        else:
            chain[key] = chain.get(key, 0) + 1
    remaining = {edge: count for edge, count in chain.items() if count}
    expected = {tuple(map(int, edge)): 1 for edge in boundary}
    simple, pair_work, exhausted = _simple_boundary(charts, boundary, maximum_pairs)
    resource_counts.append((f"source_trim_chain_pairs:{patch}", pair_work, maximum_pairs))
    if exhausted:
        findings.append(
            MeshCertificateFinding(
                "source_trim_pair_capacity", "unresolved", "source_facet", (patch,)
            )
        )
    if len(expected) != boundary.shape[0] or remaining != expected or not simple:
        return empty
    patch_source = domain.patches[patch]
    polygons, ribbons, ribbon_bounds = [], [], []
    covers: list[tuple[np.ndarray, np.ndarray]] = []
    remaining_pairs = maximum_pairs - pair_work
    for loop_index, loop in enumerate(patch_source.loops):
        selected = provenance[:, 0] == loop_index
        loop_edges = boundary[selected]
        if not loop_edges.size:
            return empty
        polygon = charts[loop_edges[:, 0]]
        if not np.array_equal(loop_edges[:, 1], np.roll(loop_edges[:, 0], -1)):
            return empty
        if _polygon_sign(polygon) != (1 if loop_index == 0 else -1):
            return empty
        polygons.append(polygon)
        trim_curves: list[AbstractTrimCurve] = []
        partitions: list[np.ndarray] = []
        bounded_loop = False
        for use_index, use in enumerate(loop):
            mask = selected & (provenance[:, 1] == use_index)
            intervals = provenance[mask, 2:]
            edges = boundary[mask]
            if not intervals.size:
                return empty
            first, last = (
                (use.first, use.last) if isinstance(use, PatchCurveUse) else (0.0, 1.0)
            )
            if intervals[0, 0] != first or intervals[-1, 1] != last:
                return empty
            if not np.array_equal(intervals[:-1, 1], intervals[1:, 0]) or np.any(
                (intervals[:, 1] - intervals[:, 0]) * (last - first) <= 0
            ):
                return empty
            bounded_ribbon = isinstance(use, PatchCurveUse) and (
                not isinstance(use.pcurve, LineCurve)
                or use.first_root is not None
                or use.last_root is not None
            )
            if not bounded_ribbon:
                source_ends = use.charts(intervals.reshape(-1)).reshape((-1, 2, 2))
                scale = max(1.0, float(np.max(np.abs(source_ends))))
                roundoff = 128 * np.finfo(np.float64).eps * scale
                gap = float(np.max(np.abs(source_ends - charts[edges])))
                if gap > domain.tolerance * scale:
                    return empty
                bounded_ribbon = gap > roundoff
            bounded_loop |= bounded_ribbon
            if isinstance(use, PatchCurveUse):
                curve = _domain_trim_curve(patch_source, use)
            elif use.trim_curve is not None:
                curve = use.trim_curve
            else:
                curve = CurveTrimSegment(
                    LineCurve(
                        np.asarray(use.start), np.asarray(use.end) - np.asarray(use.start)
                    ),
                    0.0,
                    1.0,
                )
            trim_curves.append(curve)
            partition = (np.concatenate((intervals[:, 0], intervals[-1:, 1])) - first) / (
                last - first
            )
            partition[0], partition[-1] = 0.0, 1.0
            partitions.append(partition)
            if np.any(np.diff(partition) <= 0):
                return empty
        if bounded_loop:
            if remaining_pairs < 1:
                findings.append(
                    MeshCertificateFinding(
                        "source_trim_pair_capacity",
                        "unresolved",
                        "source_facet",
                        (patch,),
                    )
                )
                return empty
            with original_trim_intersection_preparation(tuple(trim_curves)):
                cover, proof, pairs, arcs, exhausted = _certified_trim_cover(
                    trim_curves,
                    partitions,
                    polygon,
                    domain.tolerance,
                    remaining_pairs,
                    reserve_queries,
                )
                remaining_pairs -= pairs
                resource_counts.append(
                    (
                        f"source_trim_loop_pairs:{patch}:{loop_index}",
                        pairs,
                        maximum_pairs,
                    )
                )
                resource_counts.append(
                    (
                        f"source_trim_proof_arcs:{patch}:{loop_index}",
                        arcs,
                        65_536,
                    )
                )
                if exhausted:
                    findings.append(
                        MeshCertificateFinding(
                            "source_trim_pair_capacity",
                            "unresolved",
                            "source_facet",
                            (patch,),
                        )
                    )
                if arcs > 65_536:
                    findings.append(
                        MeshCertificateFinding(
                            "source_trim_arc_capacity",
                            "unresolved",
                            "source_facet",
                            (patch,),
                        )
                    )
                if proof is None:
                    findings.append(
                        MeshCertificateFinding(
                            "source_trim_topology",
                            "unresolved",
                            "source_facet",
                            (patch,),
                        )
                    )
                    return empty
                if topology_only:
                    bound = np.zeros((loop_edges.shape[0],), dtype=np.float64)
                else:
                    if reserve_queries is not None:
                        reserve_queries(2 * loop_edges.shape[0])
                    bound = _trim_ribbon_bounds(
                        domain,
                        patch,
                        cover,
                        charts,
                        points,
                        loop_edges,
                    )
            if not np.all(np.isfinite(bound)):
                return empty
            covers.append((proof.arc_lower, proof.arc_upper))
            ribbons.append(points[loop_edges[:, [0, 1, 1]]])
            ribbon_bounds.append(bound)
        else:
            line_images = charts[loop_edges]
            covers.append((np.min(line_images, axis=1), np.max(line_images, axis=1)))
            ribbons.append(points[loop_edges[:, [0, 1, 1]]])
            ribbon_bounds.append(np.zeros((loop_edges.shape[0],), dtype=np.float64))
    if np.any((provenance[:, 0] < 0) | (provenance[:, 0] >= len(patch_source.loops))):
        return empty
    for index, hole in enumerate(polygons[1:], start=1):
        if _winding(hole[0], polygons[0]) != 1:
            return empty
        if any(
            _winding(hole[0], other) != 0
            for j, other in enumerate(polygons[1:], start=1)
            if j != index
        ):
            return empty
    separated, separation_work, exhausted = _disjoint_trim_covers(covers, remaining_pairs)
    resource_counts.append(
        (f"source_trim_cross_loop_pairs:{patch}", separation_work, maximum_pairs)
    )
    if exhausted:
        findings.append(
            MeshCertificateFinding(
                "source_trim_pair_capacity", "unresolved", "source_facet", (patch,)
            )
        )
    if not separated:
        return empty
    return (
        True,
        np.concatenate(ribbons) if ribbons else np.empty((0, 3, 3)),
        np.concatenate(ribbon_bounds) if ribbon_bounds else np.empty((0,)),
        tuple(restriction.restriction_id for restriction in restrictions),
    )


def _sampling_affine_loop(
    source: MeshingSurfacePatch, loop: tuple[PatchBoundaryUse, ...], /
) -> np.ndarray | None:
    """Exact affine-image polygon, never a sampled fit of the original curves."""
    from .brep._constructors import _NormalizedTrimCurve
    from .brep._intersection_curve import _affine_curve_coefficients

    starts: list[tuple[Fraction, ...]] = []
    ends: list[tuple[Fraction, ...]] = []
    for use in loop:
        if isinstance(use, PatchPoleUse):
            if use.trim_curve is not None:
                return None
            starts.append(tuple(Fraction(value) for value in use.start))
            ends.append(tuple(Fraction(value) for value in use.end))
            continue
        if any(
            root is not None
            for root in (
                use.first_root,
                use.last_root,
                use.start_vertex_root,
                use.end_vertex_root,
            )
        ):
            return None
        carrier = _domain_trim_curve(source, use)
        lower = (Fraction(0), Fraction(0))
        extent = (Fraction(1), Fraction(1))
        if isinstance(carrier, _NormalizedTrimCurve):
            if any(
                root is not None
                for root in (
                    carrier.start_root,
                    carrier.end_root,
                    carrier.start_endpoint,
                    carrier.end_endpoint,
                )
            ):
                return None
            origin = np.asarray(carrier.lower, dtype=np.float64)
            scale = np.asarray(carrier.extent, dtype=np.float64)
            if (
                origin.ndim != 1
                or origin.shape != (2,)
                or scale.ndim != 1
                or scale.shape != (2,)
            ):
                return None
            lower = (Fraction(float(origin[0])), Fraction(float(origin[1])))
            extent = (Fraction(float(scale[0])), Fraction(float(scale[1])))
            carrier = carrier.curve
        if (
            not isinstance(carrier, CurveTrimSegment)
            or carrier.first_root is not None
            or carrier.last_root is not None
            or not isinstance(carrier.curve, AbstractCurve)
            or len(lower) != 2
            or len(extent) != 2
            or any(value == 0 for value in extent)
        ):
            return None
        coefficients = _affine_curve_coefficients(
            carrier.curve, np.zeros((2,), dtype=np.float64)
        )
        if coefficients is None:
            return None
        parameters = (
            (carrier.last, carrier.first)
            if carrier.reversed
            else (carrier.first, carrier.last)
        )
        endpoints = [
            tuple(
                (constant + Fraction(parameter) * slope - offset) / scale
                for constant, slope, offset, scale in zip(
                    coefficients[0],
                    coefficients[1],
                    lower,
                    extent,
                    strict=True,
                )
            )
            for parameter in parameters
        ]
        starts.append(endpoints[0])
        ends.append(endpoints[1])
    if any(last != starts[(index + 1) % len(starts)] for index, last in enumerate(ends)):
        return None
    # Polygon predicates must see the exact same endpoint coordinates, not a
    # rounded replacement of a non-binary-rational affine source endpoint.
    vertices = np.asarray(starts, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape != (len(starts), 2):
        return None
    if not np.all(np.isfinite(vertices)) or any(
        Fraction(float(vertices[row, axis])) != starts[row][axis]
        for row in range(vertices.shape[0])
        for axis in range(vertices.shape[1])
    ):
        return None
    return vertices


def _sampling_trim(
    source: MeshingSurfacePatch, intervals: int, relative_closure_tolerance: float, /
) -> TrimDomain:
    """Retain authored trims and their validated bounded joins for source queries."""
    loops: list[TrimLoop] = []
    for loop in source.loops:
        affine = _sampling_affine_loop(source, loop)
        if affine is not None:
            loops.append(PolygonTrimLoop(affine))
            continue
        curves: list[AbstractTrimCurve] = []
        for use in loop:
            if isinstance(use, PatchCurveUse):
                curves.append(_domain_trim_curve(source, use))
            elif use.trim_curve is not None:
                curves.append(use.trim_curve)
            else:
                curves.append(
                    CurveTrimSegment(
                        LineCurve(
                            np.asarray(use.start),
                            np.asarray(use.end) - np.asarray(use.start),
                        ),
                        0.0,
                        1.0,
                    )
                )
        partitions = [
            np.linspace(*curve.parameter_interval, intervals + 1, dtype=np.float64)
            for curve in curves
        ]
        diameter = max(
            np.linalg.norm(box[1] - box[0])
            for curve, parameters in zip(curves, partitions, strict=True)
            for first, last in zip(parameters[:-1], parameters[1:], strict=True)
            for box in (curve.enclosure(float(first), float(last)),)
        )
        loops.append(
            CurveTrimLoop(
                curves,
                tolerance=float(
                    np.nextafter(max(diameter, np.finfo(np.float64).tiny), np.inf)
                ),
                maximum_arcs=len(curves) * intervals,
                arc_parameters=partitions,
                relative_closure_tolerance=relative_closure_tolerance,
            )
        )
    return TrimDomain(loops[0], loops[1:])


@dataclass(frozen=True, slots=True)
class _PreparedBoundaryDistanceGrid:
    """One immutable sampled query grid shared by a complete certification."""

    source: MeshingDomainBoundarySource
    rows: np.ndarray
    charts: np.ndarray
    samples: np.ndarray
    radius: float
    trims: tuple[TrimDomain, ...]

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def source_revision(self) -> str:
        return self.source.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.source.ambient_dimension

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        from ._mesh_certificates import SourceBoundaryDistance

        queries = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        nearest = np.concatenate(
            [
                np.argmin(
                    np.linalg.norm(
                        queries[start : start + _SEED_BLOCK, None] - self.samples[None],
                        axis=-1,
                    ),
                    axis=1,
                )
                for start in range(0, queries.shape[0], _SEED_BLOCK)
            ]
            + [np.zeros((0,), dtype=np.int64)]
        )
        seed = np.linalg.norm(queries - self.samples[nearest], axis=1)
        projection = self.source.domain.project(
            queries, self.rows[nearest], self.charts[nearest]
        )
        inside = np.zeros((queries.shape[0],), dtype=np.bool_)
        for patch, trim in zip(self.source.patches, self.trims, strict=True):
            selected = (self.rows[nearest] == patch) & projection.converged
            if np.any(selected):
                classification = trim.classify(projection.parameters[selected])
                inside[selected] = classification.inside & classification.resolved
        distance = np.where(
            projection.converged & inside,
            np.minimum(projection.distances, seed),
            seed,
        )
        return SourceBoundaryDistance(distance, distance, "sampled")

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        from ._mesh_certificates import SourceBoundarySamples

        count = self.source._sampling_capacity()
        if count > maximum_samples:
            empty = np.empty((0,), dtype=np.float64)
            return SourceBoundarySamples(
                np.empty((0, 3), dtype=np.float64),
                empty,
                empty,
                "sampled",
                complete=False,
            )
        size = self.samples.shape[0]
        return SourceBoundarySamples(
            self.samples,
            np.full((size,), self.radius, dtype=np.float64),
            np.zeros((size,), dtype=np.float64),
            "sampled",
            complete=True,
        )


@final
class MeshingDomainBoundarySource(StrictModule):
    """Sampled boundary queries of the selected surfaces of one meshing domain.

    Each patch retains authored trim curves and samples its boundary chains as
    well as a ``resolution x resolution`` interior chart grid. Exact native trim
    classification excludes holes and points outside curved boundaries.
    Distances use a converged source projection only inside the authored trim;
    otherwise they retain the nearest sampled boundary/interior point. The
    covering radius is a sampled cell/chain half-diagonal estimate. Distances
    and radii remain ``sampled`` estimates, not certified bounds.

    With an explicit chart chain, ``boundary_chart_cover`` instead verifies
    oriented coverage and returns continuous source enclosures. A complete
    sphere may use convex radial-shell/plane bounds only after an independent
    exact degree-one radial-chain check; a same-UV interpolation residual is
    neither necessary nor substituted for the requested Euclidean deviation.
    """

    domain: MeshingDomain
    patches: tuple[int, ...] = eqx.field(static=True)
    resolution: int = eqx.field(static=True)
    chart_triangulations: tuple[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        ...,
    ]

    def __init__(
        self,
        domain: MeshingDomain,
        patches: tuple[int, ...],
        /,
        *,
        resolution: int = 48,
        chart_triangulations: tuple[
            tuple[
                int,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
            ...,
        ] = (),
    ) -> None:
        if not isinstance(domain, MeshingDomain):
            raise TypeError("domain must be MeshingDomain.")
        selected = tuple(sorted({_index(patch, "patch") for patch in patches}))
        if not selected or selected[-1] >= len(domain.patches):
            raise ValueError("patches must name surfaces of the domain.")
        count = _index(resolution, "resolution")
        if count < 2:
            raise ValueError("resolution must be at least two.")
        self.domain = domain
        self.patches = selected
        self.resolution = count
        self.chart_triangulations = chart_triangulations

    @property
    def source_id(self) -> str:
        return self.domain.source_id

    @property
    def source_revision(self) -> str:
        return self.domain.source_revision

    @property
    def source_scope_id(self) -> str:
        """Bind the actual original patch subset, including occurrence-qualified IDs."""
        indices = self.domain.scope_indices(2)
        return canonical_fingerprint(
            {
                "kind": "meshing-domain-boundary-source-scope",
                "domain": self.domain.domain_id,
                "entity_set": self.domain.entity_set_id(2),
                "entity_ids": tuple(indices[patch] for patch in self.patches),
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return 3

    def _sampling_capacity(self) -> int:
        """Bound grid and authored boundary candidates before source evaluation."""
        return len(self.patches) * self.resolution * self.resolution + max(
            3, self.resolution - 1
        ) * sum(
            len(loop)
            for patch in self.patches
            for loop in self.domain.patches[patch].loops
        )

    def _grid(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[TrimDomain, ...]]:
        """Build inside-chart and boundary samples with exact trim membership."""
        rows: list[np.ndarray] = []
        charts: list[np.ndarray] = []
        images: list[np.ndarray] = []
        trims: list[TrimDomain] = []
        radius = 0.0
        intervals = max(3, self.resolution - 1)
        for patch in self.patches:
            source = self.domain.patches[patch]
            trim = _sampling_trim(source, intervals, self.domain.tolerance)
            trims.append(trim)
            lower = np.min(trim.outer.chords, axis=0)
            upper = np.max(trim.outer.chords, axis=0)
            axes = [
                np.linspace(lower[k], upper[k], self.resolution, dtype=np.float64)
                for k in range(2)
            ]
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            flat = grid.reshape((-1, 2))
            grid_points = self.domain.evaluate(
                np.full((flat.shape[0],), patch, dtype=np.int64), flat
            ).reshape(grid.shape[:2] + (3,))
            diagonal = np.linalg.norm(
                grid_points[1:, 1:] - grid_points[:-1, :-1], axis=-1
            )
            radius = max(radius, 0.5 * float(np.max(diagonal)))
            classification = trim.classify(flat)
            inside = classification.inside & classification.resolved
            rows.append(np.full((np.count_nonzero(inside),), patch, dtype=np.int64))
            charts.append(flat[inside])
            images.append(grid_points.reshape((-1, 3))[inside])
            for loop in source.loops:
                boundary_charts = []
                for use in loop:
                    first, last = (
                        (use.first, use.last)
                        if isinstance(use, PatchCurveUse)
                        else (0.0, 1.0)
                    )
                    boundary_charts.append(
                        use.charts(
                            np.linspace(first, last, intervals + 1, dtype=np.float64)
                        )[:-1]
                    )
                boundary = np.concatenate(boundary_charts, axis=0)
                boundary_rows = np.full((boundary.shape[0],), patch, dtype=np.int64)
                points = self.domain.evaluate(boundary_rows, boundary)
                radius = max(
                    radius,
                    0.5
                    * float(
                        np.max(
                            np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
                        )
                    ),
                )
                rows.append(boundary_rows)
                charts.append(boundary)
                images.append(points)
        return (
            np.concatenate(rows),
            np.concatenate(charts, axis=0),
            np.concatenate(images, axis=0),
            radius,
            tuple(trims),
        )

    def prepare_boundary_queries(self) -> _PreparedBoundaryDistanceGrid:
        """Prepare one immutable distance/sample artifact for a certification."""
        rows, charts, samples, radius, trims = self._grid()
        return _PreparedBoundaryDistanceGrid(self, rows, charts, samples, radius, trims)

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return self.prepare_boundary_queries().boundary_distance(points)

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        return self.prepare_boundary_queries().boundary_samples(maximum_samples)

    @source_bernstein_restriction_scope()
    def boundary_chart_cover(
        self,
        maximum_patches: int,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        reserve_queries: Callable[[int], None] | None = None,
        _topology_only: bool = False,
    ) -> SourceBoundaryChartCover:
        """Independently verify a complete oriented chart chain and exact trims.

        This does not trust generation's deviation or coverage flags. Interior
        edges cancel exactly, chart triangles have positive exact orientation,
        and the remaining chain must equal the declared, simple trim boundary.
        General source trims include interval-bounded chord homotopy ribbons.
        Collapsed pole charts remain as degenerate physical simplices.
        """
        from ._mesh_certificates import MeshCertificateFinding, SourceBoundaryChartCover

        if not isinstance(_topology_only, bool):
            raise TypeError("_topology_only must be bool.")

        images, bounds = [], []
        findings: list[MeshCertificateFinding] = []
        resource_counts: list[tuple[str, int, int]] = []
        restriction_ids: list[str] = []
        complete = bool(self.chart_triangulations)
        seen = []
        work = 0
        for (
            patch,
            charts,
            points,
            cells,
            boundary,
            provenance,
            restriction_required,
            restriction_vertices,
            restriction_edges,
            restriction_parameters,
        ) in self.chart_triangulations:
            if patch not in self.patches:
                continue
            work += cells.shape[0] + boundary.shape[0]
            if work > maximum_patches or patch in seen:
                complete = False
                findings.append(
                    MeshCertificateFinding(
                        "source_chart_cover_capacity"
                        if work > maximum_patches
                        else "source_trim_coverage_premise",
                        "unresolved",
                        "source_facet",
                        (patch,),
                    )
                )
                break
            seen.append(patch)
            valid, ribbon_images, ribbon_bounds, patch_restriction_ids = (
                _verify_chart_chain(
                    self.domain,
                    patch,
                    charts,
                    points,
                    cells,
                    boundary,
                    provenance,
                    restriction_required,
                    restriction_vertices,
                    restriction_edges,
                    restriction_parameters,
                    findings,
                    resource_counts,
                    topology_only=_topology_only,
                )
            )
            restriction_ids.extend(patch_restriction_ids)
            complete &= valid
            if not valid:
                findings.append(
                    MeshCertificateFinding(
                        "source_trim_coverage_premise",
                        "unresolved",
                        "source_facet",
                        (patch,),
                    )
                )
                continue
            if _topology_only:
                images.append(points[cells])
                bounds.append(np.zeros((cells.shape[0],), dtype=np.float64))
                continue
            coordinates = charts[cells]
            source_nodes = self.domain.evaluate(
                np.full((coordinates.size // 2,), patch), coordinates.reshape((-1, 2))
            ).reshape((-1, 3, 3))
            _, _, coordinate_errors = validate_chart_restrictions(
                charts,
                restriction_required,
                restriction_vertices,
                restriction_edges,
                charts[restriction_edges]
                if restriction_vertices.size
                else np.empty((0, 2, 2), dtype=np.float64),
                restriction_parameters,
                self.source_revision,
                patch,
            )
            restriction_error = np.zeros((charts.shape[0],), dtype=np.float64)
            rows = np.flatnonzero(restriction_required)
            if rows.size:
                boxes = np.stack(
                    (
                        charts[rows] - coordinate_errors[rows],
                        charts[rows] + coordinate_errors[rows],
                    ),
                    axis=1,
                )
                lower, upper = self.domain.patches[patch].surface.derivative_bounds_batch(
                    boxes, order=1
                )
                jacobian = np.linalg.norm(
                    np.maximum(np.abs(lower), np.abs(upper)), axis=1
                )
                restriction_error[rows] = np.nextafter(
                    np.sum(jacobian * coordinate_errors[rows], axis=1),
                    np.inf,
                )
            restriction_cell_error = np.max(restriction_error[cells], axis=1)
            nodal = (
                np.max(np.linalg.norm(source_nodes - points[cells], axis=2), axis=1)
                + restriction_cell_error
            )
            deviation = (
                self.domain.interpolation_bounds(
                    patch,
                    coordinates,
                    budget=budget,
                    reserve_queries=reserve_queries,
                )
                + nodal
            )
            trim_cell_bound = _boundary_cell_ribbon_bounds(cells, boundary, ribbon_bounds)
            deviation += trim_cell_bound
            deviation += 256 * np.finfo(np.float64).eps * self.domain.scale
            sphere_frame = _full_sphere_frame(self.domain, patch)
            if sphere_frame is not None and _sphere_radial_degree_one(
                sphere_frame, points, cells
            ):
                analytic = (
                    _sphere_triangle_distance_bounds(sphere_frame, points[cells])
                    + restriction_cell_error
                    + trim_cell_bound
                )
                deviation = np.minimum(deviation, analytic)
            complete &= bool(np.all(np.isfinite(deviation)))
            # Curved trim uncertainty is already included in every physical
            # source simplex. Only authored collapsed pole wedges remain as
            # measure-zero source support; ordinary chord placeholders are not
            # independent facets.
            images.append(points[cells])
            bounds.append(deviation)
            if any(
                isinstance(use, PatchPoleUse)
                for loop in self.domain.patches[patch].loops
                for use in loop
            ):
                images.append(ribbon_images)
                bounds.append(ribbon_bounds)
        complete &= sorted(seen) == list(self.patches)
        return SourceBoundaryChartCover(
            np.concatenate(images) if images else np.empty((0, 3, 3)),
            np.concatenate(bounds) if bounds else np.empty((0,)),
            "certified" if complete else "sampled",
            complete=complete,
            source_id=self.source_id,
            source_revision=self.source_revision,
            findings=tuple(findings),
            resource_counts=tuple(resource_counts),
            restriction_ids=tuple(restriction_ids),
        )


__all__ = [
    "MeshingDomain",
    "MeshingDomainBoundarySource",
    "MeshingDomainCurve",
    "MeshingDomainProjection",
    "MeshingDomainRegion",
    "MeshingSourceAccuracy",
    "MeshingSurfacePatch",
    "PatchBoundaryUse",
    "PatchCurveUse",
    "PatchPoleUse",
]
