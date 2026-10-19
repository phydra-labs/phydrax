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
curves are the images of parameter-plane curves (``pcurve``) under their
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
patches are the source-defined geometry, evaluated in binary64.
"""

from __future__ import annotations

from typing import assert_never, Callable, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core

import phydrax.geometry._domain_chart_chain as domain_chart_chain_owner
import phydrax.geometry._domain_numerics as domain_numerics_owner
import phydrax.geometry._domain_preparation as domain_preparation_owner
import phydrax.geometry._source_interpolation as source_interpolation_owner
import phydrax.geometry.brep._constructors as constructors_owner
import phydrax.geometry.brep._model as model_owner
import phydrax.geometry.brep._projection_contracts as projection_contracts_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import exact_orient2d
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import (
    finite_real_scalar,
    nonnegative_integer,
    normalized_identifier,
    positive_real_scalar,
)
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..nonlinear import VectorLocalRootPlan
from ..typing import checked, parse
from ._atlas import AbstractBoundaryMap, AbstractTrimCurve, BoundaryAtlas
from .brep._coordinate_request import coordinate_source_request
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
)
from .brep._model import BRepGeometry, BRepModel
from .brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BernsteinRestrictionBank,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    LineCurve,
    meridian_normal_turns,
    MeridianOverlaps,
    OffsetSurface,
    PlanePatch,
    RevolutionSurface,
    shared_meridian_jets,
    source_bernstein_restriction_scope,
    sphere_source_equivalence,
    sphere_source_pole_normal_limits,
    SpherePatch,
    TorusPatch,
)
from .brep._placed import PlacedSurface
from .brep._request_preparations import TrimRequestPreparations
from .brep._root_bindings import BRepPlacedVertex, BRepVertexRoot


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


@final
class PatchCurveUse(StrictModule, NonTrainableState):
    """One traversal of a declared domain curve along a patch boundary loop.

    ``pcurve`` is a two-dimensional `AbstractCurve` in the patch parameter
    plane, traversed from curve parameter ``first`` to ``last``. All uses of one
    curve share its parameterization: the curve's owner (its first use in
    patch/loop order) defines the physical curve and every other use must map
    each curve parameter to the same physical point.
    scalar trim roots and transported branch roots both retain their
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
        traversal = domain_preparation_owner.prepare_curve_parameter_range(
            pcurve, first, last
        )
        first_, last_ = traversal.first, traversal.last
        for parameter, endpoint in ((first_, first_root), (last_, last_root)):
            if endpoint is not None:
                domain_preparation_owner.validate_root_parameter(parameter, endpoint)
        roots = domain_preparation_owner.prepare_curve_root_bindings(
            start_vertex_root, end_vertex_root, trim_curve
        )
        self.curve = nonnegative_integer(curve, "curve")
        self.pcurve = pcurve
        self.first = first_
        self.last = last_
        self.first_root, self.last_root = first_root, last_root
        self.start_vertex_root, self.end_vertex_root = roots.start, roots.end
        self.trim_curve = roots.trim

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
            domain_numerics_owner.curve_evaluation(
                self.pcurve, domain_numerics_owner.bucketed(values)
            ),
            dtype=np.float64,
        ).reshape((-1, 2))[: values.size]


@final
class PatchPoleUse(StrictModule, NonTrainableState):
    """A straight parameter side from ``start`` to ``end`` collapsing to a corner.

    ``trim_curve`` retains the source-defined degenerate coedge trim, including
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
        start_ = tuple(finite_real_scalar(float(value), "start") for value in start)
        end_ = tuple(finite_real_scalar(float(value), "end") for value in end)
        if len(start_) != 2 or len(end_) != 2 or start_ == end_:
            raise ValueError(
                f"PatchPoleUse.parameter_range must join distinct two-dimensional points; got {start_} and {end_}"
            )
        side = domain_preparation_owner.prepare_pole_side(
            corner, (start_[0], start_[1]), (end_[0], end_[1]), trim_curve
        )
        self.corner = side.corner
        self.start, self.end = side.start, side.end
        self.trim_curve = side.trim

    def charts(self, fractions: np.ndarray, /) -> np.ndarray:
        """Parameter-plane points ``(k, 2)`` at side fractions ``(k,)`` in [0, 1]."""

        start = np.asarray(self.start, dtype=np.float64)
        end = np.asarray(self.end, dtype=np.float64)
        return start + np.asarray(fractions, dtype=np.float64)[:, None] * (end - start)


def use_endpoints(use: PatchBoundaryUse, /) -> np.ndarray:
    match use:
        case PatchCurveUse():
            return use.charts(np.asarray((use.first, use.last)))
        case PatchPoleUse():
            return np.asarray((use.start, use.end), dtype=np.float64)
        case _:
            raise TypeError(
                f"MeshingSurfacePatch.loops must contain curve or pole uses; got {type(use).__name__}"
            )


def _corner_use_endpoints(use: PatchBoundaryUse, /) -> np.ndarray:
    """Physical endpoint representatives honoring integer source periods."""
    parameters = use_endpoints(use)
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

    @checked
    def __init__(
        self,
        surface: AbstractSurfacePatch,
        loops: tuple[tuple[PatchBoundaryUse, ...], ...],
        /,
        *,
        reversed: bool = False,
    ) -> None:
        loops_ = tuple(tuple(loop) for loop in loops)
        if not loops_ or any(not loop for loop in loops_):
            raise ValueError(
                "MeshingSurfacePatch.loops must contain a nonempty outer loop and nonempty holes; got an empty boundary loop"
            )
        for loop in loops_:
            for use in loop:
                if not isinstance(use, (PatchCurveUse, PatchPoleUse)):
                    raise TypeError(
                        f"MeshingSurfacePatch.loops must contain curve or pole uses; got {type(use).__name__}"
                    )
        if not isinstance(reversed, (bool, np.bool_)):
            raise TypeError(
                f"MeshingSurfacePatch.reversed must be boolean; got {type(reversed).__name__}"
            )
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
        self.start = nonnegative_integer(start, "start")
        self.end = nonnegative_integer(end, "end")


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
                raise ValueError(
                    f"MeshingDomainRegion.boundary_sides must be +1 or -1; got {side}"
                )
            entries.append((nonnegative_integer(patch, "patch"), int(side)))
        if not entries or len({patch for patch, _ in entries}) != len(entries):
            raise ValueError(
                "MeshingDomainRegion.boundary must contain distinct patches and be nonempty; got empty or repeated patches"
            )
        self.name = normalized_identifier(str(name), "name")
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


class DomainCurveMap(AbstractBoundaryMap):
    """Source curve realizations retaining implicit endpoint definitions."""

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
                "DomainCurveMap.endpoint_roots must align with the source carriers; got different carrier and root counts"
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
            match pcurve:
                case IntersectionPCurve():
                    box = pcurve.enclosure(*support, endpoint_roots=ordered_roots)
                    lower, upper = pcurve.derivative_bounds(
                        *support, order=1, endpoint_roots=ordered_roots
                    )
                case AbstractCurve():
                    box = pcurve.bounding_box(*support)
                    lower, upper = pcurve.derivative_bounds(*support, order=1)
                case _:
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


class PhysicalCurveMap(AbstractBoundaryMap):
    """native edge carriers over the normalized reference interval."""

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
                "PhysicalCurveMap.ranges must align with carriers and endpoint roots; got inconsistent edge arrays"
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
                "PhysicalCurveMap.endpoint_errors must be finite nonnegative edge-end bounds; got invalid error array"
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PhysicalCurveMap], stored: PhysicalCurveMap, /
    ) -> PhysicalCurveMap:
        """Validate the stored entire endpoint-error bank without altering it.

        Stored errors include vertex realization and scalar-root uncertainty.
        Their raw components are not recovered. Container restoration separately
        validates atlas fields and recomputed domain identity.
        """
        if not isinstance(stored.curves, tuple):
            raise TypeError(
                "PhysicalCurveMap.curves must be a tuple of native carriers; "
                f"got {type(stored.curves).__name__}"
            )
        for curve in stored.curves:
            match curve:
                case AbstractCurve() | IntersectionCurve():
                    pass
                case curve:
                    raise TypeError(
                        "PhysicalCurveMap.curves must contain native curve carriers; "
                        f"got {type(curve).__name__}"
                    )
        if not isinstance(stored.endpoint_roots, tuple) or len(
            stored.endpoint_roots
        ) != len(stored.curves):
            raise ValueError(
                "PhysicalCurveMap.endpoint_roots must align with native carriers; "
                "got invalid root arity"
            )
        for pair in stored.endpoint_roots:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    "PhysicalCurveMap.endpoint_roots must contain endpoint pairs; "
                    "got invalid pair arity"
                )
            for endpoint in pair:
                match endpoint:
                    case (
                        None
                        | TrimRootEndpoint()
                        | BranchRootEndpoint()
                        | NativePeriodEndpoint()
                    ):
                        pass
                    case endpoint:
                        raise TypeError(
                            "PhysicalCurveMap.endpoint_roots must contain native "
                            f"scalar-root definitions or None; got {type(endpoint).__name__}"
                        )
        ranges = np.asarray(jax.device_get(stored.ranges))
        expected = (len(stored.curves), 2)
        if ranges.shape != expected or not np.all(np.isfinite(ranges)):
            raise ValueError(
                f"PhysicalCurveMap.ranges must be finite with shape {expected}; "
                f"got shape {ranges.shape} or nonfinite parameters"
            )
        errors = stored.endpoint_errors
        if not isinstance(errors, np.ndarray):
            raise TypeError(
                "PhysicalCurveMap.endpoint_errors must be a NumPy array; "
                f"got {type(errors).__name__}"
            )
        if (
            errors.shape != expected
            or not np.all(np.isfinite(errors))
            or np.any(errors < 0)
        ):
            raise ValueError(
                "PhysicalCurveMap.endpoint_errors must be finite nonnegative "
                f"edge-end bounds with shape {expected}; got invalid error bank"
            )
        root_bounds = cls(stored.curves, ranges, stored.endpoint_roots)
        if ranges.dtype != root_bounds.ranges.dtype:
            raise TypeError(
                "PhysicalCurveMap.ranges must retain the constructor's float64 "
                f"representation; got {ranges.dtype}"
            )
        if errors.dtype != root_bounds.endpoint_errors.dtype:
            raise TypeError(
                "PhysicalCurveMap.endpoint_errors must retain the constructor's "
                f"float64 representation; got {errors.dtype}"
            )
        if np.any(errors < root_bounds.endpoint_errors):
            raise ValueError(
                "PhysicalCurveMap.endpoint_errors must enclose the declared "
                "scalar-root uncertainty; got an underbounded endpoint"
            )
        return stored

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
                        "PhysicalCurveMap.intersection_curve must have certified continuation charts; got unresolved charts"
                    )
                result = curve.evaluate(parameters)
                if not isinstance(result.parameter_bound, jax_core.Tracer) and not np.all(
                    np.isfinite(np.asarray(result.parameter_bound))
                ):
                    raise ValueError(
                        "PhysicalCurveMap.intersection_evaluation must have finite parameter bounds; got unresolved source bounds"
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


def domain_trim_curve(
    source: MeshingSurfacePatch, use: PatchCurveUse, /
) -> AbstractTrimCurve:
    """Reuse the CAD owner's rooted coedge and UV-join representation."""

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
    return constructors_owner.NormalizedTrimCurve(
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
    """Radius of the root image about this numerical endpoint realization."""
    endpoint = (use.first_root, use.last_root)[end]
    vertex = (use.start_vertex_root, use.end_vertex_root)[end]
    if endpoint is None:
        return 0.0
    # An authored native-period scalar is not an implicit spatial junction.
    # Transport its scalar enclosure by the coedge and surface
    # expression below, without inventing a common spatial vertex root.
    if not isinstance(endpoint, NativePeriodEndpoint) and (
        vertex is None or not vertex.supports_endpoint(endpoint)
    ):
        raise ValueError(
            "MeshingDomain.coedge_vertex_root must bind the endpoint to a shared spatial root; got an unbound endpoint"
        )
    curve = domain_trim_curve(source, use)
    box = source.surface.bounding_box(curve.enclosure(float(end), float(end)))
    return float(
        np.nextafter(np.linalg.norm(np.max(np.abs(box - point), axis=0)), np.inf)
    )


def _uses(patches: tuple[MeshingSurfacePatch, ...], /) -> list[tuple[int, int, int]]:
    """``(patch, loop, position)`` of every curve and pole use in order."""

    return [
        (patch, loop, position)
        for patch, value in enumerate(patches)
        for loop, members in enumerate(value.loops)
        for position in range(len(members))
    ]


@final
class MeshingDomain(StrictModule, NonTrainableState):
    """Revision-bound stratified parametric surface domain.

    Strata are corners (dimension 0), curves (1) and surfaces (2), identified by
    their declaration index as ``<revision>:<corner|curve|surface>:<index>``.
    ``tolerance`` is the authored relative modeling allowance. Each consistency
    check adds its source expression's outward-rounded evaluation radius to
    ``tolerance * local_source_magnitude``, without a physical unit floor.
    Native certified gaps add only their revalidated source-owned triangle
    allowances to curve-use and corner agreement, never to ``tolerance``.
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

    @checked
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
        _native_source: domain_preparation_owner.NativeDomainSource | None = None,
    ) -> None:
        patches_ = tuple(patches)
        curves_ = tuple(curves)
        regions_ = tuple(regions)
        if not patches_ or not all(
            isinstance(patch, MeshingSurfacePatch) for patch in patches_
        ):
            raise TypeError(
                "MeshingDomain.patches must be a nonempty tuple of MeshingSurfacePatch; got empty or invalid patch entries"
            )
        if not all(isinstance(curve, MeshingDomainCurve) for curve in curves_):
            raise TypeError(
                "MeshingDomain.curves must contain MeshingDomainCurve values; got invalid curve entries"
            )
        if not all(isinstance(region, MeshingDomainRegion) for region in regions_):
            raise TypeError(
                "MeshingDomain.regions must contain MeshingDomainRegion values; got invalid region entries"
            )
        corners = nonnegative_integer(corner_count, "corner_count")
        tolerance_ = positive_real_scalar(float(tolerance), "MeshingDomain.tolerance")
        accuracy_ = parse(accuracy, MeshingSourceAccuracy, "accuracy")
        owners = _curve_owners(patches_, curves_, corners)
        points, corner_errors, scale = _corner_points(patches_, curves_, owners, corners)
        gap_source = None
        match _native_source:
            case domain_preparation_owner.PlacedDomainSource():
                points, corner_errors = _native_source.validation_corners()
            case domain_preparation_owner.DefinitionDomainSource():
                gap_source = _native_source
            case None:
                pass
            case _:
                assert_never(_native_source)
        if gap_source is None:
            _validate_uses(
                patches_,
                curves_,
                owners,
                points,
                corner_errors,
                tolerance_,
            )
        else:
            # Retain the root checks above, but compare against the source vertex
            # bank rather than substituting the owner face's displaced lift.
            points = np.asarray(gap_source.geometry.vertex_points, dtype=np.float64)[
                np.asarray(gap_source.source_vertices, dtype=np.int64)
            ]
            if points.shape != (corners, 3) or not np.all(np.isfinite(points)):
                raise ValueError(
                    "MeshingDomain.corner_points must align with the finite source vertex bank; got invalid source positions"
                )
            points.setflags(write=False)
            if domain_preparation_owner.has_brep_gaps(gap_source.geometry):
                scale = max(1.0, float(np.max(np.abs(points), initial=0.0)))
            corner_errors = np.asarray(gap_source.geometry.vertex_evaluation_bounds)[
                np.asarray(gap_source.source_vertices, dtype=np.int64)
            ]
            use_gaps, endpoint_gaps = gap_source.validation_gap_bounds()
            _validate_uses(
                patches_,
                curves_,
                owners,
                points,
                corner_errors,
                tolerance_,
                use_gap_bounds=use_gaps,
                endpoint_gap_bounds=endpoint_gaps,
            )
        _validate_loops(patches_, tolerance_)
        _validate_orientation(patches_, owners)
        patch_regions = _patch_regions(patches_, regions_)
        counts = (corners, len(curves_), len(patches_))
        strata = domain_preparation_owner.prepare_source_strata(
            counts, source_indices, source_occurrences
        )
        indices, paths = strata.indices, strata.paths
        region_strata = domain_preparation_owner.prepare_source_regions(
            len(regions_), region_source_indices, region_source_occurrences
        )
        region_indices, region_paths = region_strata.indices, region_strata.paths
        if source_kinds not in (
            ("corner", "curve", "surface"),
            ("vertex", "edge", "face"),
        ):
            raise ValueError(
                f"MeshingDomain.source_kinds must name corner/curve/surface or vertex/edge/face strata; got {source_kinds}"
            )
        owner_uses = [patches_[p].loops[loop][position] for p, loop, position in owners]
        self.patches = patches_
        self.curves = curves_
        self.regions = regions_
        self.corner_points = points
        self.curve_owners = np.asarray(owners, dtype=np.int64).reshape((-1, 3))
        self.patch_regions = patch_regions
        self.brep_authority = None if gap_source is None else gap_source.authority
        self.source_id = normalized_identifier(str(source_id), "source_id")
        self.source_revision = normalized_identifier(
            str(source_revision), "source_revision"
        )
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
            DomainCurveMap(
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
        self.domain_id = self.content_domain_id()
        if _native_source is not None:
            physical = _native_source.physical_state(self)
            self.curve_atlas = physical.atlas
            self.corner_points = physical.points
            self.brep_authority = _native_source.authority

    def content_domain_id(self) -> str:
        """Fingerprint this domain's current content.

        Equals ``domain_id`` for a constructed domain. Authentication boundaries
        compare it, because tree surgery replaces content without rerunning the
        constructor and leaves the stored ``domain_id`` stale.
        """
        geometry = (
            self.brep_authority.geometry
            if isinstance(self.brep_authority, BRepModel)
            else self.brep_authority
        )
        return canonical_fingerprint(
            {
                "kind": "meshing-domain",
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "accuracy": self.accuracy,
                "tolerance": self.tolerance,
                "source_indices": self.source_indices,
                "source_kinds": self.source_kinds,
                "source_occurrences": self.source_occurrences,
                "region_source_indices": self.region_source_indices,
                "region_source_occurrences": self.region_source_occurrences,
                "authority_id": self.authority_id,
                **(
                    {
                        "brep_gap_authority": {
                            "geometry_id": geometry.geometry_id,
                            "edge_vertex_gap_bounds": geometry.edge_vertex_gap_bounds,
                            "edge_face_gap_bounds": geometry.edge_face_gap_bounds,
                        }
                    }
                    if geometry is not None
                    and domain_preparation_owner.has_brep_gaps(geometry)
                    else {}
                ),
                "patches": [patch_identity(patch) for patch in self.patches],
                "curves": [[curve.start, curve.end] for curve in self.curves],
                "corners": self.corner_count,
                "regions": [[region.name, region.boundary] for region in self.regions],
            }
        )

    @property
    def corner_count(self) -> int:
        return self.corner_points.shape[0]

    @property
    def physical_extent(self) -> float:
        """Maximum absolute source-corner coordinate, a physical length without a unit floor."""
        return float(np.max(np.abs(self.corner_points), initial=0.0))

    def entity_id(self, dimension: int, index: int, /) -> str:
        """identity of one stratum of this revision."""

        if dimension not in _KINDS:
            raise ValueError(
                f"MeshingDomain.entity_dimension must be 0, 1, 2, or 3; got {dimension}"
            )
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
            return projection_contracts_owner.brep_entity_id(
                self.source_revision,
                dimension,
                indices[index],
                occurrence_path=paths[index],
            )
        kind = "region" if dimension == 3 else self.source_kinds[dimension]
        return f"{self.source_revision}:{kind}:{indices[index]}"

    def curve_uses(self, curve: int, /) -> tuple[tuple[int, PatchCurveUse], ...]:
        """``(patch, use)`` pairs of one curve in order (owner first)."""

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
            raise ValueError(
                f"MeshingDomain.entity_dimension must be 0, 1, 2, or 3; got {dimension}"
            )
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
            raise ValueError(
                f"MeshingDomain.entity_dimension must be 0, 1, 2, or 3; got {dimension}"
            )
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
        resolved: list[int] = []
        for raw_index in identifiers:
            index = int(raw_index)
            if index not in lookup:
                raise ValueError(
                    "MeshingDomain.scope_entities must name declared source strata; "
                    f"got unknown source index {index}"
                )
            resolved.append(lookup[index])
        return np.asarray(resolved, dtype=np.int64)

    def curve_breakpoints(self, curve: int, /) -> np.ndarray:
        """source-defined chart/span junctions in normalized owner coordinates."""
        mapping = self.curve_atlas.mapping
        match mapping:
            case PhysicalCurveMap():
                carrier = mapping.curves[curve]
                first, last = np.asarray(mapping.ranges[curve])
            case DomainCurveMap():
                carrier = mapping.pcurves[curve]
                first, last = float(mapping.firsts[curve]), float(mapping.lasts[curve])
            case _:
                raise TypeError(
                    f"MeshingDomain.curve_atlas must use native physical or parameter-composed curve carriers; got {type(mapping).__name__}"
                )
        values = domain_numerics_owner.carrier_breakpoints(
            carrier, float(min(first, last)), float(max(first, last))
        )
        return np.sort((values - first) / (last - first))

    def interpolation_bounds(
        self,
        patch: int,
        charts: np.ndarray,
        /,
        *,
        reserve_queries: Callable[[int], None] | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> np.ndarray:
        """Continuous chart-to-affine interpolation bounds, not nodal residuals.

        Taylor's integral remainder and the variance bound for a bounded
        scalar variable give ``(Huu du² + 2 Huv du dv + Hvv dv²) / 8``.
        This bounds both images under the common barycentric parameterization.
        It does not prove trim coverage or global injectivity. Unsupported
        carrier families return infinity, never a sampled derivative estimate.
        """
        standalone = budget is None
        with (
            coordinate_source_request(
                (self.patches[patch].surface, charts),
                rows=charts.shape[0],
                queries_upper=2 * charts.shape[0],
                budget=budget,
            ) as budget,
            source_bernstein_restriction_scope(
                budget=budget, restriction_bank=restriction_bank
            ) as restriction_bank,
            shared_meridian_jets(meridian_jets=meridian_jets) as meridian_jets,
        ):
            return source_interpolation_owner.surface_interpolation_bounds(
                self.patches[patch].surface,
                charts,
                reserve_queries,
                standalone=standalone,
                budget=budget,
                preparations=preparations,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            )

    def normal_turn_bounds(
        self,
        patch: int,
        charts: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> np.ndarray:
        """Continuous normal diameter; singular interval charts stay unresolved."""
        with (
            coordinate_source_request(
                (self.patches[patch].surface, charts),
                rows=charts.shape[0],
                queries_upper=2 * charts.shape[0],
                budget=budget,
            ) as budget,
            source_bernstein_restriction_scope(
                budget=budget, restriction_bank=restriction_bank
            ) as restriction_bank,
            shared_meridian_jets(meridian_jets=meridian_jets) as meridian_jets,
        ):
            surface = self.patches[patch].surface
            source_surface = surface
            coordinates = np.asarray(charts, dtype=np.float64)
            widths = np.ptp(coordinates, axis=1)
            placement_distortion = 1.0
            if isinstance(surface, PlacedSurface):
                placement_distortion = domain_numerics_owner.matrix_distortion(
                    np.asarray(surface.rotation)
                )
                surface = surface.definition
            sphere = sphere_source_equivalence(surface, budget=budget)
            if sphere is not None and sphere[1] != 0:
                surface = sphere[0]
            else:
                match surface:
                    case OffsetSurface():
                        equivalent = surface.analytic_equivalent(budget=budget)
                        if equivalent is not None:
                            surface = equivalent
                    case _:
                        pass
            if isinstance(surface, PlanePatch):
                regular = (
                    np.linalg.norm(
                        np.cross(
                            np.asarray(surface.first_axis),
                            np.asarray(surface.second_axis),
                        )
                    )
                    > 0
                )
                return np.full((coordinates.shape[0],), 0.0 if regular else np.inf)
            if isinstance(surface, (ConePatch, CylinderPatch, SpherePatch, TorusPatch)):
                distortion = (
                    domain_numerics_owner.basis_distortion(surface) * placement_distortion
                )
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
            match surface:
                case RevolutionSurface() as revolution:
                    pass
                case OffsetSurface(base=RevolutionSurface() as base):
                    revolution, distance = base, float(surface.distance)
                case _:
                    pass
            turns = (
                None
                if revolution is None
                else meridian_normal_turns(
                    revolution,
                    distance,
                    coordinates,
                    budget=budget,
                    meridian_jets=meridian_jets,
                )
            )
            if turns is None:
                result = domain_numerics_owner.jet_normal_turns(
                    source_surface,
                    boxes,
                    widths,
                    budget=budget,
                    preparations=preparations,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
            else:
                result = placement_distortion * turns
                unresolved = np.isnan(turns)
                if np.any(unresolved):
                    result[unresolved] = domain_numerics_owner.jet_normal_turns(
                        source_surface,
                        boxes[unresolved],
                        widths[unresolved],
                        budget=budget,
                        preparations=preparations,
                        restriction_bank=restriction_bank,
                        meridian_jets=meridian_jets,
                    )
            return np.nextafter(result * (1 + 128 * np.finfo(np.float64).eps), np.inf)

    def curve_interpolation_bounds(
        self,
        curve: int,
        values: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> np.ndarray:
        """Continuous chord bounds, including non-C1 source chart transitions."""
        with (
            coordinate_source_request(
                (self.curve_atlas.mapping, values),
                rows=values.size,
                queries_upper=2 * values.size,
                budget=budget,
            ) as budget,
            source_bernstein_restriction_scope(
                budget=budget, restriction_bank=restriction_bank
            ) as restriction_bank,
            shared_meridian_jets(meridian_jets=meridian_jets) as meridian_jets,
        ):
            boundary_map = self.curve_atlas.mapping
            if isinstance(boundary_map, PhysicalCurveMap):
                first, last = np.asarray(boundary_map.ranges[curve])
            else:
                patch, loop, position = self.curve_owners[curve]
                use = _curve_use(self.patches[patch].loops[loop][position])
                first, last = use.first, use.last
            span = last - first
            known_second = None
            if isinstance(boundary_map, PhysicalCurveMap):
                candidate = domain_numerics_owner.curve_second_bound(
                    boundary_map.curves[curve], budget=budget
                )
                if np.isfinite(candidate):
                    known_second = candidate
            else:
                surface, carrier = self.patches[patch].surface, use.pcurve
                match (surface, carrier):
                    case (_, LineCurve()):
                        hessian = domain_numerics_owner.hessian_bounds(
                            surface, budget=budget
                        )
                        if hessian is not None:
                            direction = np.abs(np.asarray(carrier.direction))
                            known_second = (
                                hessian[0] * direction[0] ** 2
                                + 2 * hessian[1] * direction[0] * direction[1]
                                + hessian[2] * direction[1] ** 2
                            )
                    case (PlanePatch(), CircleCurve()):
                        axes = np.stack(
                            (
                                np.asarray(surface.first_axis),
                                np.asarray(surface.second_axis),
                            )
                        )
                        known_second = float(
                            carrier.radius
                        ) * domain_numerics_owner.matrix_norm_upper(
                            np.column_stack(
                                (
                                    np.asarray(carrier.first_axis) @ axes,
                                    np.asarray(carrier.second_axis) @ axes,
                                )
                            )
                        )
                    case _:
                        pass
            bounds = []
            for start, end in zip(values[:-1], values[1:], strict=True):
                lower, upper = sorted(
                    (float(first + span * start), float(first + span * end))
                )
                if isinstance(boundary_map, PhysicalCurveMap):
                    bound = domain_numerics_owner.physical_curve_chord_bound(
                        boundary_map.curves[curve],
                        lower,
                        upper,
                        known_second=known_second,
                        budget=budget,
                    )
                else:
                    patch, loop, position = self.curve_owners[curve]
                    use = _curve_use(self.patches[patch].loops[loop][position])
                    bound = domain_numerics_owner.chart_curve_chord_bound(
                        self.patches[patch].surface,
                        use.pcurve,
                        lower,
                        upper,
                        known_second=known_second,
                        budget=budget,
                    )
                bounds.append(bound)
            result = np.asarray(bounds, dtype=np.float64)
            if isinstance(boundary_map, (PhysicalCurveMap, DomainCurveMap)):
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
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> np.ndarray:
        """Per-source-arc physical ribbon bounds for one boundary discretization.

        ``intervals`` selects arcs ``[parameters[i], parameters[i + 1]]`` of the
        whole discretization; each selected bound equals its full-batch value.
        """
        with (
            coordinate_source_request(
                (self.patches[patch].surface, parameters, points),
                rows=parameters.size,
                queries_upper=2 * parameters.size,
                budget=budget,
            ) as budget,
            source_bernstein_restriction_scope(
                budget=budget, restriction_bank=restriction_bank
            ) as restriction_bank,
            shared_meridian_jets(meridian_jets=meridian_jets) as meridian_jets,
        ):
            patch_ = nonnegative_integer(patch, "patch")
            if patch_ >= len(self.patches) or not isinstance(use, PatchCurveUse):
                raise ValueError(
                    f"MeshingDomain.trim_ribbon_patch must index a present patch with a curve use; got patch {patch_}"
                )
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
                raise ValueError(
                    "MeshingDomain.trim_ribbon_parameters must cover one ordered source use with aligned points; got invalid endpoints, ordering, or array shapes"
                )
            arcs = np.arange(values.size - 1, dtype=np.int64)
            if intervals is not None:
                selected = np.asarray(intervals)
                if (
                    selected.ndim != 1
                    or not np.issubdtype(selected.dtype, np.integer)
                    or np.any(selected < 0)
                    or np.any(selected >= values.size - 1)
                ):
                    raise ValueError(
                        "MeshingDomain.trim_ribbon_intervals must index source arcs; got invalid interval indices"
                    )
                arcs = selected.astype(np.int64)
                if not arcs.size:
                    return np.empty((0,), dtype=np.float64)
            normalized = (values - use.first) / (use.last - use.first)
            normalized[0], normalized[-1] = 0.0, 1.0
            curve = domain_trim_curve(self.patches[patch_], use)
            edges = np.stack((arcs, arcs + 1), axis=1)
            nodes, local = np.unique(edges, return_inverse=True)
            charts = use.charts(values[nodes])
            return domain_chart_chain_owner.trim_curve_interval_bounds(
                self,
                patch_,
                tuple(curve for _ in range(edges.shape[0])),
                normalized[arcs],
                normalized[arcs + 1],
                charts[local.reshape(edges.shape)],
                physical[edges],
                budget=budget,
                preparations=preparations,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            )

    @classmethod
    @checked
    def from_brep(cls: type[MeshingDomain], model: BRepModel, /) -> MeshingDomain:
        """Prepare native B-Rep carriers, never its query tessellation.

        Degenerate source edges become pole uses; nondegenerate edges retain
        their source indices and their source-defined 3D curves.
        Authored occurrence incidence determines placed strata and material
        sharing; equal positions or common path prefixes never infer identity.
        """
        geometry = model.geometry
        if geometry is None:
            raise ValueError(
                "MeshingDomain.model.geometry must contain native B-Rep curves and trim loops; got None"
            )
        if geometry.occurrences:
            return domain_preparation_owner.prepare_placed_brep_domain(model)
        return domain_preparation_owner.prepare_brep_domain(
            cls,
            geometry,
            model.patches,
            np.asarray(model.orientation),
            source_id=model.source_id,
            source_revision=model.source_revision,
            authority_id=model.model_id,
            budget=None,
            authority=model,
        )

    @classmethod
    @checked
    def from_brep_geometry(
        cls: type[MeshingDomain],
        geometry: BRepGeometry,
        surfaces: tuple[AbstractSurfacePatch, ...],
        orientation: np.ndarray,
        /,
        *,
        source_id: str,
        source_revision: str,
        authority_id: str = "",
        budget: CoordinateEnclosureBudget | None = None,
    ) -> MeshingDomain:
        """Prepare native CAD construction staging without tessellation recursion."""

        return domain_preparation_owner.prepare_brep_domain(
            cls,
            geometry,
            surfaces,
            orientation,
            source_id=source_id,
            source_revision=source_revision,
            authority_id=authority_id,
            budget=budget,
            authority=geometry,
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[MeshingDomain], stored: MeshingDomain, /
    ) -> MeshingDomain:
        """Rebuild a stored domain through its source constructor validation.

        Native BRepModel or BRepGeometry rebuilds physical carriers and incidence;
        independent parametric domains retain their declared preparation inputs. The
        restoration consumer compares the rebuilt content with stored fields.
        """
        match stored.brep_authority:
            case BRepModel() as authority:
                return cls.from_brep(authority)
            case BRepGeometry() as authority:
                return cls.from_brep_geometry(
                    authority,
                    tuple(patch.surface for patch in stored.patches),
                    np.asarray(
                        [patch.orientation for patch in stored.patches],
                        dtype=np.float64,
                    ),
                    source_id=stored.source_id,
                    source_revision=stored.source_revision,
                    authority_id=stored.authority_id,
                )
            case None:
                return cls(
                    stored.patches,
                    stored.curves,
                    stored.corner_points.shape[0],
                    source_id=stored.source_id,
                    source_revision=stored.source_revision,
                    regions=stored.regions,
                    tolerance=stored.tolerance,
                    accuracy=stored.accuracy,
                    source_indices=stored.source_indices,
                    source_occurrences=stored.source_occurrences,
                    region_source_indices=stored.region_source_indices,
                    region_source_occurrences=stored.region_source_occurrences,
                    source_kinds=stored.source_kinds,
                    authority_id=stored.authority_id,
                )
            case authority:
                raise TypeError(
                    "MeshingDomain.brep_authority must be BRepModel, BRepGeometry, "
                    f"or None; got {type(authority).__name__}"
                )

    def require_current(self, source_id: str, source_revision: str, /) -> None:
        """Refuse a request bound to another source or a stale revision."""

        if (source_id, source_revision) != (self.source_id, self.source_revision):
            raise ValueError(
                f"MeshingDomain.source_binding must match {self.source_id!r} at revision {self.source_revision!r}; got {source_id!r} at revision {source_revision!r}"
            )

    def evaluate(
        self,
        patches: np.ndarray,
        charts: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Physical points ``(k, 3)`` of patch parameters ``(k, 2)``."""

        rows, charts_ = self._batch(patches, charts)
        with domain_numerics_owner.source_query_request(
            rows.size, budget=budget
        ) as budget:
            domain_numerics_owner.consume_source_queries(rows.size, budget=budget)
            result = np.zeros((rows.size, 3), dtype=np.float64)
            for patch in np.unique(rows).tolist():
                selected = rows == patch
                result[selected] = domain_numerics_owner.host_surface_evaluation(
                    self.patches[patch].surface, charts_[selected]
                )
            return result

    def oriented_normals(
        self,
        patches: np.ndarray,
        charts: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unit oriented normals ``(k, 3)`` and whether each chart point is regular.

        A point is regular where ``|d_u x d_v|`` exceeds the rounding bound of
        its evaluation; singular normals (poles, apexes) are returned as zero.
        """

        rows, charts_ = self._batch(patches, charts)
        with domain_numerics_owner.source_query_request(
            rows.size, budget=budget
        ) as budget:
            domain_numerics_owner.consume_source_queries(rows.size, budget=budget)
            result = np.zeros((rows.size, 3), dtype=np.float64)
            regular = np.zeros((rows.size,), dtype=np.bool_)
            for patch in np.unique(rows).tolist():
                selected = rows == patch
                value = self.patches[patch]
                differential = domain_numerics_owner.host_surface_differential(
                    value.surface, charts_[selected]
                )
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
        self,
        patch: int,
        charts: np.ndarray,
        center: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """analytic normal limits on explicitly declared collapsed sides."""
        with (
            coordinate_source_request(
                (self.patches[patch].surface, charts),
                rows=charts.shape[0],
                queries_upper=2 * charts.shape[0],
                budget=budget,
            ) as budget,
            source_bernstein_restriction_scope(
                budget=budget, restriction_bank=restriction_bank
            ) as restriction_bank,
            shared_meridian_jets(meridian_jets=meridian_jets) as meridian_jets,
        ):
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
            sphere = sphere_source_equivalence(definition, budget=budget)
            if sphere is not None and sphere[1] != 0:
                original, _ = sphere
                _, supported_poles = sphere_source_pole_normal_limits(
                    definition, charts, budget=budget
                )
                declared &= supported_poles
                first = transform @ np.asarray(original.first_axis)
                second = transform @ np.asarray(original.second_axis)
                result = np.sign(charts[:, 1])[:, None] * np.cross(first, second)[None]
            else:
                match definition:
                    case ConePatch():
                        angles = charts[:, 0]
                        radial = np.cos(angles)[:, None] * np.asarray(
                            definition.first_axis
                        ) + np.sin(angles)[:, None] * np.asarray(definition.second_axis)
                        tangent = -np.sin(angles)[:, None] * np.asarray(
                            definition.first_axis
                        ) + np.cos(angles)[:, None] * np.asarray(definition.second_axis)
                        slope = float(np.tan(float(definition.semi_angle)))
                        radial_sign = np.sign(slope * (center[1] - charts[:, 1]))
                        result = radial_sign[:, None] * np.cross(
                            tangent @ transform.T,
                            (np.asarray(definition.axis)[None] + slope * radial)
                            @ transform.T,
                        )
                    case _:
                        declared[:] = False
            lengths = np.linalg.norm(result, axis=1)
            supported = declared & np.isfinite(lengths) & (lengths > 0)
            result[supported] *= source.orientation / lengths[supported, None]
            result[~supported] = 0.0
            return result, supported

    def project(
        self,
        points: np.ndarray,
        patches: np.ndarray,
        seeds: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> MeshingDomainProjection:
        """Closest points of ``points`` on the requested patches from chart seeds.

        Solves the stationarity ``d X(uv)^T (X(uv) - p) = 0`` with the bounded
        native vector root; the status is the root's convergence evidence.
        """

        points_ = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        rows, seeds_ = self._batch(patches, seeds)
        if points_.shape[0] != rows.size:
            raise ValueError(
                f"MeshingDomain.projection_points must align with patch and seed rows; got {points_.shape[0]} points and {rows.size} rows"
            )
        with domain_numerics_owner.source_query_request(
            2 * rows.size, budget=budget
        ) as budget:
            domain_numerics_owner.consume_source_queries(rows.size, budget=budget)
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
                        return (
                            jacobian.T @ (surface.evaluate(uv) - point) / (scale * scale)
                        )

                    root, diagnostics = plan.solve_with_diagnostics(residual, seed)
                    return root, diagnostics.converged

                roots, status = jax.vmap(solve)(
                    jnp.asarray(points_[selected]), jnp.asarray(seeds_[selected])
                )
                parameters[selected] = np.asarray(roots, dtype=np.float64)
                converged[selected] = np.asarray(status, dtype=np.bool_)
            closest = self.evaluate(rows, parameters, budget=budget)
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
            raise ValueError(
                f"MeshingDomain.query_charts must align with patch rows; got {charts_.shape[0]} charts and {rows.size} rows"
            )
        if np.any((rows < 0) | (rows >= len(self.patches))):
            raise ValueError(
                "MeshingDomain.query_patches must index the domain patches; got out-of-range indices"
            )
        return rows, charts_


def _curve_use(use: PatchBoundaryUse, /) -> PatchCurveUse:
    match use:
        case PatchCurveUse():
            return use
        case _:
            raise TypeError(
                f"MeshingDomain.curve_owner must be a PatchCurveUse; got {type(use).__name__}"
            )


def patch_identity(patch: MeshingSurfacePatch, /) -> dict[str, object]:

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
                                None
                                if root is None
                                else model_owner.endpoint_root_payload(root)
                                for root in (use.first_root, use.last_root)
                            ),
                            "vertex_roots": tuple(
                                None
                                if root is None
                                else model_owner.vertex_root_payload(root)
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
                    raise TypeError(
                        f"MeshingSurfacePatch.loops must contain curve or pole uses; got {type(use).__name__}"
                    )
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
                    raise ValueError(
                        f"MeshingDomain.curve_uses must index declared curves; got curve {use.curve}"
                    )
                owners.setdefault(use.curve, (patch, loop, position))
            case PatchPoleUse():
                if use.corner >= corners:
                    raise ValueError(
                        f"MeshingDomain.pole_uses must index declared corners; got corner {use.corner}"
                    )
                used_corners.add(use.corner)
            case _:
                raise TypeError(
                    f"MeshingSurfacePatch.loops must contain curve or pole uses; got {type(use).__name__}"
                )
    missing = sorted(set(range(len(curves))) - set(owners))
    if missing:
        raise ValueError(
            f"MeshingDomain.curves must bound at least one patch each; got unused curves {missing}"
        )
    for curve in curves:
        if curve.start >= corners or curve.end >= corners:
            raise ValueError(
                "MeshingDomain.curve_endpoints must index declared corners; got out-of-range endpoints"
            )
        used_corners.update((curve.start, curve.end))
    if used_corners != set(range(corners)):
        raise ValueError(
            f"MeshingDomain.corners must bound a curve or pole side; got unused corners {sorted(set(range(corners)) - used_corners)}"
        )
    return [owners[curve] for curve in range(len(curves))]


def _corner_points(
    patches: tuple[MeshingSurfacePatch, ...],
    curves: tuple[MeshingDomainCurve, ...],
    owners: list[tuple[int, int, int]],
    corners: int,
    /,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Corner positions from owner curve endpoints (or pole sides) and the scale."""

    points = np.full((corners, 3), np.nan, dtype=np.float64)
    errors = np.zeros(corners, dtype=np.float64)
    root_ids: dict[int, str] = {}
    for curve, (patch, loop, position) in zip(curves, owners, strict=True):
        use = _curve_use(patches[patch].loops[loop][position])
        surface = patches[patch].surface
        ends = domain_numerics_owner.host_surface_evaluation(
            surface, _corner_use_endpoints(use)
        )
        endpoint_errors = _use_evaluation_errors(
            patches[patch], use, np.asarray((use.first, use.last)), ends
        )
        if ends.shape != (2, 3):
            raise ValueError(
                f"MeshingDomain.curve_endpoints must realize two three-dimensional points; got shape {ends.shape}"
            )
        if use.start_vertex_root is not None or use.end_vertex_root is not None:
            ends = np.array(ends, dtype=np.float64, copy=True)
        for end, vertex in enumerate((use.start_vertex_root, use.end_vertex_root)):
            if vertex is not None:
                corner = (curve.start, curve.end)[end]
                old = root_ids.setdefault(corner, vertex.root_id)
                if old != vertex.root_id:
                    raise ValueError(
                        f"MeshingDomain.corner_roots must share one spatial-root identity; got conflicting roots at corner {corner}"
                    )
                point, _, certified = vertex.evaluate()
                if not certified:
                    raise ValueError(
                        "MeshingDomain.corner_roots must have certified point realizations; got an unresolved root"
                    )
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
                errors[corner] = endpoint_errors[0 if corner == curve.start else 1]
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
        images = domain_numerics_owner.host_surface_evaluation(
            patches[patch].surface, charts
        )
        for end, root in enumerate((use.first_root, use.last_root)):
            if isinstance(root, NativePeriodEndpoint) and root.turns.denominator == 1:
                points[curve.start] = images[end]
                errors[curve.start] = _use_evaluation_errors(
                    patches[patch], use, np.asarray((use.first, use.last)), images
                )[end]

    for patch, loop, position in _uses(patches):
        use = patches[patch].loops[loop][position]
        if isinstance(use, PatchPoleUse) and np.isnan(points[use.corner, 0]):
            points[use.corner] = domain_numerics_owner.host_surface_evaluation(
                patches[patch].surface, np.asarray((use.start,), dtype=np.float64)
            )[0]
            errors[use.corner] = _use_evaluation_errors(
                patches[patch], use, np.asarray((0.0,)), points[[use.corner]]
            )[0]
    if not np.all(np.isfinite(points)):
        raise ValueError(
            "MeshingDomain.corner_points must be finite; got nonfinite source positions"
        )
    points.setflags(write=False)
    return points, errors, max(1.0, float(np.max(np.abs(points), initial=0.0)))


def _use_point_box(use: PatchBoundaryUse, parameter: float, /) -> np.ndarray:
    """Enclose a source pcurve at one binary parameter."""
    match use:
        case PatchCurveUse():
            match use.pcurve:
                case AbstractCurve():
                    box = use.pcurve.bounding_box(parameter, parameter)
                case _:
                    box = use.pcurve.enclosure(parameter, parameter)
        case PatchPoleUse():
            chart = use.charts(np.asarray((parameter,)))[0]
            box = np.stack((chart, chart))
        case _:
            raise TypeError(
                "MeshingSurfacePatch.loops must contain curve or pole uses; "
                f"got {type(use).__name__}"
            )
    if not np.all(np.isfinite(box)):
        raise ValueError(
            "MeshingDomain.pcurve_enclosure must be finite; "
            f"got an unbounded point enclosure at parameter {parameter}"
        )
    return box


def _use_endpoint_box(
    source: MeshingSurfacePatch, use: PatchBoundaryUse, end: int, /
) -> np.ndarray:
    """Transport the endpoint's scalar source enclosure into the parameter plane."""
    match use:
        case PatchCurveUse():
            if (use.first_root, use.last_root)[end] is None:
                return _use_point_box(use, (use.first, use.last)[end])
            # A Q + Q*(2*pi) endpoint is not its binary representative. Reuse
            # the rooted trim owner's scalar transport, including argument
            # realization error at partial turns. Coordinate-relative trig
            # padding cannot cover that error near a sine/cosine zero.
            box = domain_trim_curve(source, use).enclosure(float(end), float(end))
        case PatchPoleUse():
            return _use_point_box(use, float(end))
        case _:
            raise TypeError(
                "MeshingSurfacePatch.loops must contain curve or pole uses; "
                f"got {type(use).__name__}"
            )
    if not np.all(np.isfinite(box)):
        raise ValueError(
            "MeshingDomain.pcurve_enclosure must be finite; "
            f"got an unbounded source enclosure at endpoint {end}"
        )
    return box


def _use_evaluation_errors(
    patch: MeshingSurfacePatch,
    use: PatchBoundaryUse,
    parameters: np.ndarray,
    images: np.ndarray,
    /,
) -> np.ndarray:
    """Bound composed source evaluation roundoff at fixed binary parameters."""
    errors = []
    for parameter, image in zip(parameters, images, strict=True):
        enclosure = patch.surface.bounding_box(_use_point_box(use, float(parameter)))
        if not np.all(np.isfinite(enclosure)):
            raise ValueError(
                "MeshingDomain.surface_enclosure must be finite; "
                f"got an unbounded point enclosure at parameter {parameter}"
            )
        # Reuse the source's operand-magnitude/operation-count bound: analytic
        # _interval_box uses 64*eps*sum(abs(operands)) for up to three product/add
        # terms (including radial/trig products); spline restrictions propagate
        # directed coefficient enclosures. Thus cancellation and spline degree
        # stay with the evaluator rather than a global coordinate-scale guess.
        # Three squares, two additions and sqrt need at most 6 roundings; 8 eps
        # also covers the subtraction and final outward rounding of the radius.
        radius = np.linalg.norm(np.max(np.abs(enclosure - image), axis=0))
        errors.append(np.nextafter(radius * (1 + 8 * np.finfo(np.float64).eps), np.inf))
    return np.asarray(errors, dtype=np.float64)


def _consistency_allowance(
    tolerance: float, magnitude: float, evaluation_radius: np.ndarray, /
) -> np.ndarray:
    """Add the declared relative modeling allowance to source evaluation uncertainty."""
    return np.nextafter(tolerance * magnitude + evaluation_radius, np.inf)


def _gap_triangle_allowance(bound: float, first: float, second: float, /) -> float:
    """Outward triangle sum, retaining an unchanged zero-gap allowance."""
    if first > 0.0:
        bound = float(np.nextafter(bound + first, np.inf))
    if second > 0.0:
        bound = float(np.nextafter(bound + second, np.inf))
    return bound


def _validate_curve_corners(
    source: MeshingSurfacePatch,
    use: PatchCurveUse,
    ends: np.ndarray,
    corners: np.ndarray,
    roundoff: np.ndarray,
    owner_gap: float,
    endpoint_gaps: np.ndarray | None,
    /,
) -> None:
    """Require endpoint agreement within modeling, evaluation, root, and gap bounds."""
    allowance = np.asarray(
        [
            roundoff[end]
            + _root_endpoint_error(source, use, end, ends[end])
            + _root_endpoint_error(source, use, end, corners[end])
            for end in range(2)
        ]
    )
    if endpoint_gaps is not None:
        allowance = np.asarray(
            [
                _gap_triangle_allowance(
                    float(allowance[end]), owner_gap, float(endpoint_gaps[end])
                )
                for end in range(2)
            ],
            dtype=np.float64,
        )
    if np.any(np.linalg.norm(ends - corners, axis=1) > allowance):
        raise ValueError(
            f"MeshingDomain.curve_corners must agree with their declared source corners; got mismatched endpoints for curve {use.curve}"
        )


def _validate_uses(
    patches: tuple[MeshingSurfacePatch, ...],
    curves: tuple[MeshingDomainCurve, ...],
    owners: list[tuple[int, int, int]],
    corner_points: np.ndarray,
    corner_errors: np.ndarray,
    tolerance: float,
    /,
    *,
    use_gap_bounds: dict[tuple[int, int, int], float] | None = None,
    endpoint_gap_bounds: np.ndarray | None = None,
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
                        f"MeshingDomain.curve_ranges must span their owner parameter range; got inconsistent use for curve {use.curve}"
                    )
                parameters = owner.first + fractions * (owner.last - owner.first)
                reference = domain_numerics_owner.host_surface_evaluation(
                    patches[owner_patch].surface, owner.charts(parameters)
                )
                image = domain_numerics_owner.host_surface_evaluation(
                    surface, use.charts(parameters)
                )
                gaps = np.linalg.norm(image - reference, axis=1)
                reference_errors = _use_evaluation_errors(
                    patches[owner_patch], owner, parameters, reference
                )
                image_errors = _use_evaluation_errors(
                    patches[patch], use, parameters, image
                )
                # Modeling tolerance is relative to the local source-coordinate
                # operands being compared, not a unit-floored domain-wide bank.
                reference_magnitude = float(np.max(np.abs(reference)))
                magnitude = max(reference_magnitude, float(np.max(np.abs(image))))
                allowed = _consistency_allowance(
                    tolerance,
                    magnitude,
                    np.nextafter(reference_errors + image_errors, np.inf),
                )
                owner_gap = (
                    0.0
                    if use_gap_bounds is None
                    else use_gap_bounds.get(
                        (owner_patch, owner_loop, owner_position), 0.0
                    )
                )
                if use_gap_bounds is not None:
                    allowed = np.asarray(
                        [
                            _gap_triangle_allowance(
                                float(value),
                                owner_gap,
                                use_gap_bounds.get((patch, loop, position), 0.0),
                            )
                            for value in allowed
                        ]
                    )
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
                                "MeshingDomain.coedge_vertex_roots must share one spatial-root identity; got inconsistent rooted joins"
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
                        f"MeshingDomain.curve_geometry must reproduce the shared source carrier within {float(np.max(allowed)):.3e}; got patch {patch}, curve {use.curve}, gap {gap:.3e}"
                    )
                curve = curves[use.curve]
                _validate_curve_corners(
                    patches[owner_patch],
                    owner,
                    reference[[0, -1]],
                    corner_points[[curve.start, curve.end]],
                    _consistency_allowance(
                        tolerance,
                        max(
                            reference_magnitude,
                            float(
                                np.max(np.abs(corner_points[[curve.start, curve.end]]))
                            ),
                        ),
                        np.nextafter(
                            reference_errors[[0, -1]]
                            + corner_errors[[curve.start, curve.end]],
                            np.inf,
                        ),
                    ),
                    owner_gap,
                    None
                    if endpoint_gap_bounds is None
                    else endpoint_gap_bounds[use.curve],
                )
            case PatchPoleUse():
                image = domain_numerics_owner.host_surface_evaluation(
                    surface, use.charts(fractions)
                )
                allowed = _consistency_allowance(
                    tolerance,
                    max(
                        float(np.max(np.abs(image))),
                        float(np.max(np.abs(corner_points[use.corner]))),
                    ),
                    np.nextafter(
                        _use_evaluation_errors(patches[patch], use, fractions, image)
                        + corner_errors[use.corner],
                        np.inf,
                    ),
                )
                gaps = np.linalg.norm(image - corner_points[use.corner], axis=1)
                gap = float(np.max(gaps))
                if np.any(gaps > allowed):
                    raise ValueError(
                        f"MeshingDomain.pole_geometry must collapse to its corner within {float(np.max(allowed)):.3e}; got patch {patch}, corner {use.corner}, gap {gap:.3e}"
                    )
            case _:
                raise TypeError(
                    f"MeshingSurfacePatch.loops must contain curve or pole uses; got {type(use).__name__}"
                )


def _validate_loops(
    patches: tuple[MeshingSurfacePatch, ...], tolerance: float, /
) -> None:
    """Consecutive uses of every loop join in the parameter plane."""

    for index, patch in enumerate(patches):
        for loop in patch.loops:
            ends = [use_endpoints(use) for use in loop]
            magnitude = max(float(np.max(np.abs(value))) for value in ends)
            for position, value in enumerate(ends):
                following = ends[(position + 1) % len(ends)]
                previous, following_use = (
                    loop[position],
                    loop[(position + 1) % len(loop)],
                )
                # Preserve the declared L-infinity join metric. The maximum
                # coordinate radius bounds its evaluation error; comparing each
                # coordinate to a separate radius would impose a stricter metric.
                radii = np.max(
                    np.abs(_use_endpoint_box(patch, previous, 1) - value[1]),
                    axis=0,
                ) + np.max(
                    np.abs(_use_endpoint_box(patch, following_use, 0) - following[0]),
                    axis=0,
                )
                allowance = _consistency_allowance(
                    tolerance,
                    magnitude,
                    np.nextafter(radii, np.inf),
                )
                if float(np.max(np.abs(value[1] - following[0]))) > float(
                    np.max(allowance)
                ):
                    if (
                        isinstance(previous, PatchCurveUse)
                        and isinstance(following_use, PatchCurveUse)
                        and previous.last_root is not None
                        and following_use.first_root is not None
                        and domain_trim_curve(patch, previous).shares_endpoint(
                            domain_trim_curve(patch, following_use)
                        )
                    ):
                        continue
                    raise ValueError(
                        f"MeshingDomain.patch_loops must close in the parameter plane; got an open loop on patch {index}"
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
                f"MeshingDomain.curve_orientation must traverse shared curves oppositely; got equal oriented senses for curve {curve}"
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
                raise ValueError(
                    f"MeshingDomain.region_patches must index declared patches; got patch {patch} in region {region.name!r}"
                )
            # An outward oriented normal (+1) puts the region on the negative side.
            column = 0 if side == 1 else 1
            if sides[patch, column] >= 0:
                raise ValueError(
                    f"MeshingDomain.patch_regions must contain at most one region on each oriented side; got repeated side on patch {patch}"
                )
            sides[patch, column] = region_index
            for loop in patches[patch].loops:
                for use in loop:
                    if isinstance(use, PatchCurveUse):
                        counts[use.curve] = counts.get(use.curve, 0) + 1
        if any(count != 2 for count in counts.values()):
            raise ValueError(
                f"MeshingDomain.region_boundary must form a closed surface; got unpaired curves in region {region.name!r}"
            )
    return sides


__all__ = [
    "MeshingDomain",
    "MeshingDomainCurve",
    "MeshingDomainProjection",
    "MeshingDomainRegion",
    "MeshingSourceAccuracy",
    "MeshingSurfacePatch",
    "PatchBoundaryUse",
    "PatchCurveUse",
    "PatchPoleUse",
]
