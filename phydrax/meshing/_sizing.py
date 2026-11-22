#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._bvh import bvh_nearest_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry.brep._model import BRepModel
from ..typing import checked
from ._metric import (
    _grade_scalar_sizes,
    _scalar_violation,
    MeshMetricField,
    MetricGradationKind,
)
from ._scope import MeshingScope


class SizeControlStrength(StrEnum):
    HARD = "hard"
    SOFT = "soft"


class SizeFieldDomain(StrEnum):
    EUCLIDEAN_VOLUME = "euclidean_volume"
    SURFACE_GEODESIC = "surface_geodesic"
    MESH_GEODESIC = "mesh_geodesic"
    BACKGROUND_GRID = "background_grid"
    SAMPLE_CLOUD = "sample_cloud"


class SizeCombinationPolicy(StrEnum):
    REJECT_HARD_CONFLICTS = "reject_hard_conflicts"
    EXPLICIT_PRIORITY = "explicit_priority"


class SizeCompliancePolicy(StrictModule, NonTrainableState):
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    target_statistics: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 0.0,
        target_statistics: tuple[str, ...] = ("p50", "p95"),
    ) -> None:
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        statistics = tuple(str(value).strip() for value in target_statistics)
        supported = {"p50", "p95"}
        if (
            not np.isfinite(absolute)
            or not np.isfinite(relative)
            or absolute < 0.0
            or relative < 0.0
        ):
            raise ValueError(
                "Size compliance tolerances must be finite and non-negative."
            )
        if (
            not statistics
            or len(set(statistics)) != len(statistics)
            or not set(statistics) <= supported
        ):
            raise ValueError(
                "target_statistics must contain distinct supported statistics p50/p95."
            )
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.target_statistics = statistics
        self.policy_id = canonical_fingerprint(
            {
                "kind": "size-compliance-policy",
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "target_statistics": statistics,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[SizeCompliancePolicy], value: SizeCompliancePolicy, /
    ) -> SizeCompliancePolicy:
        if type(value) is not cls:
            raise TypeError("value must be SizeCompliancePolicy.")
        if (
            type(value.absolute_tolerance) is not float
            or type(value.relative_tolerance) is not float
            or type(value.target_statistics) is not tuple
        ):
            raise TypeError("Stored size-compliance fields are not canonical.")
        rebuilt = cls(
            absolute_tolerance=value.absolute_tolerance,
            relative_tolerance=value.relative_tolerance,
            target_statistics=value.target_statistics,
        )
        if (
            rebuilt.policy_id != value.policy_id
            or rebuilt.absolute_tolerance != value.absolute_tolerance
            or rebuilt.relative_tolerance != value.relative_tolerance
            or rebuilt.target_statistics != value.target_statistics
        ):
            raise ValueError("Stored size-compliance policy differs from its fields.")
        return rebuilt


class SizeComplianceComparison(StrEnum):
    """How a route compares its declared size statistic q with the bound b.

    ``AT_MOST``: q <= b (1 + relative_tolerance) + absolute_tolerance.
    ``TWO_SIDED``: |q - b| <= absolute_tolerance + relative_tolerance * b.
    ``REPORT_ONLY``: q is reported and never an issue.
    ``CONSTRUCTION_OBJECTIVE``: the construction searches for q = b exactly.
    """

    AT_MOST = "at_most"
    TWO_SIDED = "two_sided"
    REPORT_ONLY = "report_only"
    CONSTRUCTION_OBJECTIVE = "construction_objective"


# Unit roundoff of float64.
_UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0


def _sign(value: Fraction, /) -> int:
    return int(value > 0) - int(value < 0)


def _rational_plus_root(
    rational: Fraction, coefficient: Fraction, radicand: Fraction
) -> int:
    """The exact sign of ``rational + coefficient * sqrt(radicand)``."""
    if coefficient == 0 or radicand == 0:
        return _sign(rational)
    if rational == 0 or _sign(rational) == _sign(coefficient):
        return _sign(coefficient)
    return _sign(rational) * _sign(rational * rational - coefficient**2 * radicand)


@dataclass(frozen=True, slots=True)
class SizeMeasures:
    """Per-entity squared size Q with a float value, its error bound and Q exactly.

    ``squared[i]`` is the float evaluation of Q_i and ``relative_error[i]`` a
    proved bound on ``|squared[i] - Q_i| / Q_i`` (inf when none is proved);
    ``exact(i)`` is Q_i as a rational in the stored coordinates, or ``None``
    for an infinite Q_i (a degenerate triangle).
    """

    squared: np.ndarray
    relative_error: np.ndarray
    exact: Callable[[int], Fraction | None]


def edge_length_measures(points: np.ndarray, edges: np.ndarray, /) -> SizeMeasures:
    """Squared edge lengths |p - q|^2.

    Each component difference and square is one rounding (relative error u
    each), and the three-term sum adds 2u: the float value is within about 5u
    of Q relatively; 16u is the stated bound.
    """
    stored = np.asarray(points, dtype=np.float64)
    rows = np.asarray(edges, dtype=np.int64)
    vectors = stored[rows[:, 1]] - stored[rows[:, 0]]
    squared = np.sum(vectors * vectors, axis=1)

    def exact(row: int, /) -> Fraction:
        first, second = (stored[int(index)] for index in rows[row])
        return sum(
            (
                (Fraction(float(b)) - Fraction(float(a))) ** 2
                for a, b in zip(first, second)
            ),
            Fraction(0),
        )

    return SizeMeasures(squared, np.full(squared.shape, 16.0 * _UNIT_ROUNDOFF), exact)


def triangle_circumradius_measures(
    points: np.ndarray, triangles: np.ndarray, /
) -> SizeMeasures:
    """Squared circumradii R^2 = a^2 b^2 c^2 / (4 |e0 x e1|^2).

    Error bound (u the unit roundoff, first order): each squared edge length
    is within 5u, so the numerator within 17u; each cross component
    fl(fl(ab) - fl(cd)) of the rounded edges is within 4u (|ab| + |cd|), whose
    norm is at most 4 sqrt(2) u |e0||e1|, so 4 |cross|^2 is within
    3u D + 12u |e0|^2 |e1|^2 of D = 4 |cross|^2; the division adds u. The
    relative bound is thus below 21u + 12u |e0|^2 |e1|^2 / |cross|^2; the
    stated bound is 64u (1 + |e0|^2 |e1|^2 / |cross|^2) from the float values,
    inf when that exceeds 1/4 (where first order no longer holds) or the
    float cross product vanishes.
    """
    stored = np.asarray(points, dtype=np.float64)
    rows = np.asarray(triangles, dtype=np.int64)
    corners = stored[rows]
    edges = corners[:, (1, 2, 0)] - corners[:, (2, 0, 1)]
    lengths = np.sum(edges * edges, axis=2)
    cross = np.cross(edges[:, 0], edges[:, 1])
    denominator = 4.0 * np.sum(cross * cross, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        squared = np.prod(lengths, axis=1) / denominator
        relative = (
            64.0
            * _UNIT_ROUNDOFF
            * (1.0 + 4.0 * lengths[:, 0] * lengths[:, 1] / denominator)
        )
    proved = (denominator > 0.0) & np.isfinite(relative) & (relative <= 0.25)
    squared = np.where(denominator > 0.0, squared, np.inf)
    relative = np.where(proved, relative, np.inf)

    def exact(row: int, /) -> Fraction | None:
        p, q, r = (
            tuple(Fraction(float(value)) for value in stored[int(index)])
            for index in rows[row]
        )
        sides = [
            tuple(b - a for a, b in zip(first, second, strict=True))
            for first, second in ((q, r), (r, p), (p, q))
        ]
        u, v = sides[0], sides[1]
        normal = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        area = 4 * sum((value * value for value in normal), Fraction(0))
        if area == 0:
            return None
        numerator = Fraction(1)
        for side in sides:
            numerator *= sum((value * value for value in side), Fraction(0))
        return numerator / area

    return SizeMeasures(squared, relative, exact)


@dataclass(frozen=True, slots=True)
class SizeComplianceDeclaration:
    """The size quantity one route judges and the kernel guarantee behind it.

    ``sqrt(bound_factor_squared)`` times a hard uniform control's
    ``target_size`` h is the bound b of :class:`SizeComplianceComparison`
    (held squared so that b is exact: h / sqrt(2) is ``Fraction(1, 2)``);
    ``statistic`` names the per-entity quantity whose p50/p95
    (``SizeCompliancePolicy``) is judged.
    """

    route: str
    entity: str
    statistic: str
    comparison: SizeComplianceComparison
    bound_factor_squared: Fraction
    guarantee: str

    @property
    def bound_factor(self) -> float:
        """The bound factor as a float (reported requested values)."""
        return float(np.sqrt(float(self.bound_factor_squared)))

    def exceeds(
        self,
        measures: SizeMeasures,
        statistic: str,
        target_size: float,
        policy: SizeCompliancePolicy,
        /,
    ) -> bool:
        """Whether an AT_MOST statistic lies above its tolerated bound, exactly.

        The statistic is :func:`size_statistic` of sqrt(Q) (the order
        statistics at numpy's linear virtual index (n - 1) q and its float
        weight g), compared with t = sqrt(F) c + d, F the declared squared
        factor, c = h (1 + relative_tolerance), d = absolute_tolerance, all
        rationals of the stored floats. Each entity's side of t is decided in
        floating point when its value is farther from t^2 than the proved
        error bounds of both, else exactly; an interpolated statistic between
        an entity at or below t and one above is decided exactly.
        """
        if self.comparison is not SizeComplianceComparison.AT_MOST:
            raise ValueError(f"{self.route} does not declare an AT_MOST size bound.")
        factor = self.bound_factor_squared
        scale = Fraction(target_size) * (1 + Fraction(policy.relative_tolerance))
        offset = Fraction(policy.absolute_tolerance)
        squared = np.asarray(measures.squared, dtype=np.float64)
        relative = np.asarray(measures.relative_error, dtype=np.float64)
        count = squared.shape[0]
        if count == 0:
            raise ValueError("A size statistic needs at least one entity.")

        def side(row: int, /) -> int:
            # sign(sqrt(Q) - t) = sign(Q - F c^2 - d^2 - 2 c d sqrt(F)).
            value = measures.exact(row)
            if value is None:
                return 1
            return _rational_plus_root(
                value - factor * scale**2 - offset**2, -2 * scale * offset, factor
            )

        threshold = float(np.sqrt(float(factor))) * float(scale) + float(offset)
        limit = threshold * threshold
        # |squared - Q| <= rho Q <= 2 rho squared (rho <= 1/4); t^2 in floats
        # is within ~6u of the exact t^2, 16u is stated.
        margin = 2.0 * relative * squared + 16.0 * _UNIT_ROUNDOFF * limit
        with np.errstate(invalid="ignore"):
            gap = squared - limit
        decided = np.isfinite(margin) & (np.abs(gap) > margin)
        above = np.where(decided, gap > 0, False)
        for row in np.flatnonzero(~decided).tolist():
            above[row] = side(row) > 0
        at_most = int(count - np.count_nonzero(above))
        virtual = (count - 1) * {"p50": 0.5, "p95": 0.95}[statistic]
        lower = int(np.floor(virtual))
        weight = Fraction(float(virtual - lower))
        if lower >= count - 1 or weight == 0:
            return at_most < lower + 1
        if at_most >= lower + 2:
            return False
        if at_most <= lower:
            return True
        # Order statistic lower is the largest Q at or below t; lower + 1 the
        # smallest above it.
        below_value = _extreme(measures, np.flatnonzero(~above), largest=True)
        above_value = _extreme(measures, np.flatnonzero(above), largest=False)
        if above_value is None:
            return True
        assert below_value is not None
        # sign((1-g) sqrt(x) + g sqrt(y) - sqrt(F) c - d), both sides >= 0:
        # sign(r + p sqrt(xy) - q sqrt(F)) with p, q >= 0.
        alpha, beta = 1 - weight, weight
        rational = (
            alpha**2 * below_value + beta**2 * above_value - factor * scale**2 - offset**2
        )
        positive = 2 * alpha * beta
        negative = 2 * scale * offset
        product = below_value * above_value
        left = _rational_plus_root(rational, positive, product)
        if negative == 0 or factor == 0:
            return left > 0
        if left <= 0:
            return False
        # r + p sqrt(X) > 0: compare its square with q^2 F.
        return (
            _rational_plus_root(
                rational**2 + positive**2 * product - negative**2 * factor,
                2 * rational * positive,
                product,
            )
            > 0
        )


def _extreme(
    measures: SizeMeasures, rows: np.ndarray, /, *, largest: bool
) -> Fraction | None:
    """The exact largest (or smallest) Q of ``rows``; None for an infinite one."""
    squared = np.asarray(measures.squared, dtype=np.float64)[rows]
    relative = np.asarray(measures.relative_error, dtype=np.float64)[rows]
    finite = np.isfinite(relative)
    with np.errstate(invalid="ignore"):
        # Q lies in [squared / (1 + rho), squared / (1 - rho)], inside these.
        low = np.where(finite, squared * (1.0 - 2.0 * relative), 0.0)
        high = np.where(finite, squared * (1.0 + 4.0 * relative), np.inf)
    if largest:
        candidates = rows[high >= np.max(low)]
    else:
        candidates = rows[low <= np.min(high)]
    values = [measures.exact(int(row)) for row in candidates.tolist()]
    finite_values = [value for value in values if value is not None]
    if largest:
        # Rows at or below t are finite.
        return max(finite_values)
    return min(finite_values) if finite_values else None


_AT_MOST = SizeComplianceComparison.AT_MOST
_TWO_SIDED = SizeComplianceComparison.TWO_SIDED

# One row per route that evaluates a size control's p50/p95 statistic: the
# quantity the route judges, how, and the producing kernel's guarantee it
# comes from. Only the CAD chart surface and the S3 boundary rows are consumed
# in code; the others record what their routes already do.
CAD_CHART_SURFACE_SIZE = SizeComplianceDeclaration(
    "providers/_native_cad_mesh.py:_size_compliance",
    "scoped CAD-solid boundary triangles",
    "triangle circumradius R",
    _AT_MOST,
    Fraction(1, 2),
    "chart kernel: metric circumradius <= sqrt(1/2) under I/h^2 "
    "(native/meshcore/src/cad_refine2d.hpp:83, cad_geometry.hpp:199-209), "
    "so R <= h/sqrt(2)",
)
S3_BOUNDARY_EDGE_SIZE = SizeComplianceDeclaration(
    "providers/_native_volume.py:_volume_compliance",
    "boundary faces of facet-scoped controls",
    "edge length",
    _AT_MOST,
    Fraction(1),
    "S3 splits every splittable boundary edge longer than 4/sqrt(6) of its "
    "stored size (native/meshcore/src/refine3d.cpp:1697-1706, 2655-2690); the "
    "PLC edge-length convention stores sqrt(6)/4 h "
    "(phydrax/meshing/_volume_generation.py:79-84), so edge <= h",
)
SIZE_COMPLIANCE_DECLARATIONS: tuple[SizeComplianceDeclaration, ...] = (
    CAD_CHART_SURFACE_SIZE,
    S3_BOUNDARY_EDGE_SIZE,
    SizeComplianceDeclaration(
        "providers/_native_implicit.py:_implicit_volume_compliance",
        "implicit carrier boundary faces",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "construction stop rule: octree cell diagonal <= h(1+rel)+abs "
        "(providers/_native_implicit.py:713-729, "
        "geometry/implicit/_adaptive_discovery.py:2740-2754)",
    ),
    SizeComplianceDeclaration(
        "providers/_native_implicit.py:_compliance",
        "implicit surface triangles",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "construction stop rule: octree cell diagonal <= h(1+rel)+abs "
        "(providers/_native_implicit.py:713-729)",
    ),
    SizeComplianceDeclaration(
        "_implicit_volume.py:_compliance",
        "restricted implicit boundary facets",
        "edge length",
        _AT_MOST,
        Fraction(1),
        "restricted Delaunay facet longest edge <= min(h, reach diameter) "
        "(_implicit_volume.py:769, 1060-1072)",
    ),
    SizeComplianceDeclaration(
        "providers/_native_reconstruction.py:_compliance",
        "reconstructed surface triangles",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "construction stop rule: red refinement until the statistic is met "
        "(providers/_native_reconstruction.py:340-397, 505)",
    ),
    SizeComplianceDeclaration(
        "providers/_native_planar.py:_compliance",
        "planar triangle edges",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "construction stop rule: free edges <= local size, h or h(1+rel)+abs "
        "(providers/_native_planar.py:198-268, "
        "_native_planar_construction.py:39-50)",
    ),
    SizeComplianceDeclaration(
        "providers/_native_curve.py:_curve_compliance",
        "curve intervals",
        "arc length",
        _TWO_SIDED,
        Fraction(1),
        "equidistribution with ceil(length / h) intervals "
        "(providers/_native_curve.py:1824)",
    ),
    SizeComplianceDeclaration(
        "providers/_gmsh_sizing.py:_edge_size_evidence",
        "Gmsh mesh edges",
        "edge length",
        SizeComplianceComparison.REPORT_ONLY,
        Fraction(1),
        "Gmsh size fields are target edge lengths without a bound "
        "(providers/_gmsh_sizing.py:83-193)",
    ),
    SizeComplianceDeclaration(
        "providers/_native_structured.py:_size_compliance",
        "structured quad/hex physical edges",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "none: the user's divisions are checked against h",
    ),
    SizeComplianceDeclaration(
        "providers/_native_surface_quad.py:_mapped_size_compliance",
        "mapped quad edge enclosures",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "none: the user's divisions are checked against h",
    ),
    SizeComplianceDeclaration(
        "providers/_native_dual.py:_quad_compliance",
        "dual quad edges",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "pairs or subdivides planar triangles whose free edges are <= h "
        "(providers/_native_dual.py:2692)",
    ),
    SizeComplianceDeclaration(
        "providers/_native_dual.py:_dual_hex_compliance",
        "balanced/frame grid hex edges",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "grid knots at spacing h (_hex_cut_preparation.py:211-224)",
    ),
    SizeComplianceDeclaration(
        "providers/_native_dual.py:_mapped_grid_compliance",
        "mapped grid hex edges",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "ceil(Jacobian length bound / h) divisions (_hex_generation.py:1466-1469)",
    ),
    SizeComplianceDeclaration(
        "providers/_native_polyhedral.py:_polyhedral_compliance",
        "polyhedral cell edge enclosures",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "construction stop rule: cell diameter <= maximum_size or h(1+rel)+abs "
        "(_polyhedral_generation.py:136-158)",
    ),
    SizeComplianceDeclaration(
        "providers/_native_voronoi.py:_compliance",
        "conforming Voronoi cells",
        "cube root of cell volume",
        _TWO_SIDED,
        Fraction(1),
        "seed placement: body-centered cubic interior lattice of cell volume h^3, "
        "surface samples at spacing 0.75 h (native/meshcore/src/conforming_voronoi.cpp, "
        "Seeder)",
    ),
    SizeComplianceDeclaration(
        "providers/_native_periodic_size.py:periodic_size_compliance",
        "periodic quotient edges",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "periodic metric I/h^2 from uniform controls "
        "(providers/_native_periodic_size.py:180-238); exact at zero tolerance",
    ),
    SizeComplianceDeclaration(
        "_device_adaptation.py:_owner_local_periodic_size_epoch",
        "owner-local periodic quotient edges",
        "edge length",
        _TWO_SIDED,
        Fraction(1),
        "periodic metric I/h^2; exact rational quantile",
    ),
    SizeComplianceDeclaration(
        "providers/_native_periodic_algebraic.py:construct_exact_algebraic_carrier",
        "algebraic periodic carrier edges",
        "edge length",
        SizeComplianceComparison.CONSTRUCTION_OBJECTIVE,
        Fraction(1),
        "the edge-graph search targets the p50/p95 edge quantile exactly "
        "(providers/_native_periodic_algebraic.py:955-1060, 1721-1725)",
    ),
)


def size_statistic(values: np.ndarray, statistic: str, /) -> float:
    """The p50 or p95 (linear interpolation) of per-entity size values."""
    return float(np.quantile(values, {"p50": 0.5, "p95": 0.95}[statistic]))


def triangle_circumradii(points: np.ndarray, triangles: np.ndarray, /) -> np.ndarray:
    """Circumradius abc / (4 area) of every triangle; inf for a degenerate one."""
    corners = np.asarray(points, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    edges = corners[:, (1, 2, 0)] - corners[:, (2, 0, 1)]
    product = np.prod(np.linalg.norm(edges, axis=2), axis=1)
    twice_area = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        radii = product / (2.0 * twice_area)
    return np.where(twice_area > 0.0, radii, np.inf)


def _size(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


class UniformSizeControl(StrictModule, NonTrainableState):
    scope: MeshingScope
    target_size: float = eqx.field(static=True)
    minimum_size: float | None = eqx.field(static=True)
    maximum_size: float | None = eqx.field(static=True)
    maximum_growth_rate: float | None = eqx.field(static=True)
    strength: SizeControlStrength = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scope: MeshingScope,
        target_size: float,
        /,
        *,
        minimum_size: float | None = None,
        maximum_size: float | None = None,
        maximum_growth_rate: float | None = None,
        strength: SizeControlStrength = SizeControlStrength.HARD,
        priority: int = 0,
    ) -> None:
        target = _size(target_size, "target_size")
        minimum = None if minimum_size is None else _size(minimum_size, "minimum_size")
        maximum = None if maximum_size is None else _size(maximum_size, "maximum_size")
        growth = None if maximum_growth_rate is None else float(maximum_growth_rate)
        if minimum is not None and minimum > target:
            raise ValueError("Sizes must satisfy minimum <= target when supplied.")
        if maximum is not None and target > maximum:
            raise ValueError("Sizes must satisfy target <= maximum when supplied.")
        if growth is not None and (not np.isfinite(growth) or growth < 1.0):
            raise ValueError("maximum_growth_rate must be finite and at least one.")
        if not isinstance(strength, SizeControlStrength):
            raise TypeError("strength must be SizeControlStrength.")
        self.scope = scope
        self.target_size = target
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.maximum_growth_rate = growth
        self.strength = strength
        self.priority = int(priority)
        self.control_id = canonical_fingerprint(
            {
                "kind": "uniform-size-control",
                "scope": scope.scope_id,
                "target_size": target,
                "minimum_size": minimum,
                "maximum_size": maximum,
                "maximum_growth_rate": growth,
                "strength": strength.value,
                "priority": int(priority),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[UniformSizeControl],
        value: UniformSizeControl,
        /,
        *,
        source_model: BRepModel | None = None,
    ) -> UniformSizeControl:
        if type(value) is not cls:
            raise TypeError("value must be UniformSizeControl.")
        if (
            type(value.target_size) is not float
            or type(value.priority) is not int
            or any(
                item is not None and type(item) is not float
                for item in (
                    value.minimum_size,
                    value.maximum_size,
                    value.maximum_growth_rate,
                )
            )
        ):
            raise TypeError("Stored uniform size-control fields are not canonical.")
        rebuilt = cls(
            MeshingScope.from_stored_fields(value.scope, source_model=source_model),
            value.target_size,
            minimum_size=value.minimum_size,
            maximum_size=value.maximum_size,
            maximum_growth_rate=value.maximum_growth_rate,
            strength=value.strength,
            priority=value.priority,
        )
        if (
            rebuilt.control_id != value.control_id
            or rebuilt.target_size != value.target_size
            or rebuilt.minimum_size != value.minimum_size
            or rebuilt.maximum_size != value.maximum_size
            or rebuilt.maximum_growth_rate != value.maximum_growth_rate
            or rebuilt.strength is not value.strength
            or rebuilt.priority != value.priority
        ):
            raise ValueError("Stored uniform size control differs from its fields.")
        return rebuilt


class CurvatureSizeControl(StrictModule, NonTrainableState):
    scope: MeshingScope
    normal_angle: float = eqx.field(static=True)
    minimum_size: float | None = eqx.field(static=True)
    maximum_size: float | None = eqx.field(static=True)
    use_faceted_curvature: bool = eqx.field(static=True)
    strength: SizeControlStrength = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scope: MeshingScope,
        normal_angle: float,
        /,
        *,
        minimum_size: float | None = None,
        maximum_size: float | None = None,
        use_faceted_curvature: bool = False,
        strength: SizeControlStrength = SizeControlStrength.SOFT,
        priority: int = 0,
    ) -> None:
        angle = float(normal_angle)
        minimum = None if minimum_size is None else _size(minimum_size, "minimum_size")
        maximum = None if maximum_size is None else _size(maximum_size, "maximum_size")
        if not np.isfinite(angle) or angle <= 0.0 or angle >= np.pi:
            raise ValueError("normal_angle must lie strictly between zero and pi.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum_size cannot exceed maximum_size.")
        if not isinstance(strength, SizeControlStrength):
            raise TypeError("strength must be SizeControlStrength.")
        self.scope = scope
        self.normal_angle = angle
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.use_faceted_curvature = bool(use_faceted_curvature)
        self.strength = strength
        self.priority = int(priority)
        self.control_id = canonical_fingerprint(
            {
                "kind": "curvature-size-control",
                "scope": scope.scope_id,
                "normal_angle": angle,
                "minimum_size": minimum,
                "maximum_size": maximum,
                "use_faceted_curvature": bool(use_faceted_curvature),
                "strength": strength.value,
                "priority": int(priority),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[CurvatureSizeControl],
        value: CurvatureSizeControl,
        /,
        *,
        source_model: BRepModel | None = None,
    ) -> CurvatureSizeControl:
        if type(value) is not cls:
            raise TypeError("value must be CurvatureSizeControl.")
        if (
            type(value.normal_angle) is not float
            or type(value.use_faceted_curvature) is not bool
            or type(value.priority) is not int
            or any(
                item is not None and type(item) is not float
                for item in (value.minimum_size, value.maximum_size)
            )
        ):
            raise TypeError("Stored curvature size-control fields are not canonical.")
        rebuilt = cls(
            MeshingScope.from_stored_fields(value.scope, source_model=source_model),
            value.normal_angle,
            minimum_size=value.minimum_size,
            maximum_size=value.maximum_size,
            use_faceted_curvature=value.use_faceted_curvature,
            strength=value.strength,
            priority=value.priority,
        )
        if (
            rebuilt.control_id != value.control_id
            or rebuilt.normal_angle != value.normal_angle
            or rebuilt.minimum_size != value.minimum_size
            or rebuilt.maximum_size != value.maximum_size
            or rebuilt.use_faceted_curvature != value.use_faceted_curvature
            or rebuilt.strength is not value.strength
            or rebuilt.priority != value.priority
        ):
            raise ValueError("Stored curvature size control differs from its fields.")
        return rebuilt


class ProximitySizeControl(StrictModule, NonTrainableState):
    source_scope: MeshingScope
    target_scope: MeshingScope
    elements_per_gap: int = eqx.field(static=True)
    minimum_size: float | None = eqx.field(static=True)
    maximum_size: float | None = eqx.field(static=True)
    opposite_normals_only: bool = eqx.field(static=True)
    strength: SizeControlStrength = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        elements_per_gap: int,
        /,
        *,
        minimum_size: float | None = None,
        maximum_size: float | None = None,
        opposite_normals_only: bool = True,
        strength: SizeControlStrength = SizeControlStrength.HARD,
        priority: int = 0,
    ) -> None:
        if not isinstance(source_scope, MeshingScope) or not isinstance(
            target_scope, MeshingScope
        ):
            raise TypeError("source_scope and target_scope must be MeshingScope.")
        binding = (
            source_scope.source_id,
            source_scope.source_revision,
            source_scope.entity_kind,
            source_scope.entity_dimension,
            source_scope.entity_set_id,
        )
        target_binding = (
            target_scope.source_id,
            target_scope.source_revision,
            target_scope.entity_kind,
            target_scope.entity_dimension,
            target_scope.entity_set_id,
        )
        if binding != target_binding:
            raise ValueError("Proximity scopes must share one exact entity binding.")
        count = int(elements_per_gap)
        minimum = None if minimum_size is None else _size(minimum_size, "minimum_size")
        maximum = None if maximum_size is None else _size(maximum_size, "maximum_size")
        if count <= 0:
            raise ValueError("elements_per_gap must be positive.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum_size cannot exceed maximum_size.")
        if not isinstance(strength, SizeControlStrength):
            raise TypeError("strength must be SizeControlStrength.")
        self.source_scope = source_scope
        self.target_scope = target_scope
        self.elements_per_gap = count
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.opposite_normals_only = bool(opposite_normals_only)
        self.strength = strength
        self.priority = int(priority)
        self.control_id = canonical_fingerprint(
            {
                "kind": "proximity-size-control",
                "source_scope": source_scope.scope_id,
                "target_scope": target_scope.scope_id,
                "elements_per_gap": count,
                "minimum_size": minimum,
                "maximum_size": maximum,
                "opposite_normals_only": bool(opposite_normals_only),
                "strength": strength.value,
                "priority": int(priority),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ProximitySizeControl],
        value: ProximitySizeControl,
        /,
        *,
        source_model: BRepModel | None = None,
    ) -> ProximitySizeControl:
        if type(value) is not cls:
            raise TypeError("value must be ProximitySizeControl.")
        if (
            type(value.elements_per_gap) is not int
            or type(value.opposite_normals_only) is not bool
            or type(value.priority) is not int
            or any(
                item is not None and type(item) is not float
                for item in (value.minimum_size, value.maximum_size)
            )
        ):
            raise TypeError("Stored proximity size-control fields are not canonical.")
        rebuilt = cls(
            MeshingScope.from_stored_fields(
                value.source_scope, source_model=source_model
            ),
            MeshingScope.from_stored_fields(
                value.target_scope, source_model=source_model
            ),
            value.elements_per_gap,
            minimum_size=value.minimum_size,
            maximum_size=value.maximum_size,
            opposite_normals_only=value.opposite_normals_only,
            strength=value.strength,
            priority=value.priority,
        )
        if (
            rebuilt.control_id != value.control_id
            or rebuilt.elements_per_gap != value.elements_per_gap
            or rebuilt.minimum_size != value.minimum_size
            or rebuilt.maximum_size != value.maximum_size
            or rebuilt.opposite_normals_only != value.opposite_normals_only
            or rebuilt.strength is not value.strength
            or rebuilt.priority != value.priority
        ):
            raise ValueError("Stored proximity size control differs from its fields.")
        return rebuilt


SizeControl = UniformSizeControl | CurvatureSizeControl | ProximitySizeControl


class ResolvedSizeField(StrictModule, NonTrainableState):
    """Isotropic target sizes at explicitly identified sample entities."""

    domain: SizeFieldDomain = eqx.field(static=True)
    sample_points: Array
    sample_entity_ids: Array
    values: Array
    source_control_ids: tuple[str, ...] = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: SizeFieldDomain,
        sample_points: ArrayLike,
        values: ArrayLike,
        /,
        *,
        sample_entity_ids: ArrayLike,
        source_control_ids: tuple[str, ...],
    ) -> None:
        if not isinstance(domain, SizeFieldDomain):
            raise TypeError("domain must be SizeFieldDomain.")
        points = np.asarray(sample_points, dtype=np.float64)
        sizes = np.asarray(values, dtype=np.float64)
        identifiers = np.asarray(sample_entity_ids)
        if points.ndim != 2 or points.shape[0] == 0 or not np.all(np.isfinite(points)):
            raise ValueError("sample_points must be one non-empty finite matrix.")
        if (
            sizes.shape != (points.shape[0],)
            or np.any(~np.isfinite(sizes))
            or np.any(sizes <= 0)
        ):
            raise ValueError("Resolved size values must be positive and match samples.")
        if identifiers.shape != (points.shape[0],) or not np.issubdtype(
            identifiers.dtype, np.integer
        ):
            raise ValueError(
                "sample_entity_ids must be integer IDs aligned with samples."
            )
        controls = tuple(str(value) for value in source_control_ids)
        if not controls or any(not value for value in controls):
            raise ValueError("Resolved size fields require source control identities.")
        identifiers = identifiers.astype(np.int64, copy=False)
        self.domain = domain
        self.sample_points = jnp.asarray(points)
        self.sample_entity_ids = jnp.asarray(identifiers)
        self.values = jnp.asarray(sizes)
        self.source_control_ids = controls
        self.field_id = canonical_fingerprint(
            {
                "kind": "resolved-size-field",
                "domain": domain.value,
                "sample_points": array_tree_fingerprint(points),
                "sample_entity_ids": array_tree_fingerprint(identifiers),
                "values": array_tree_fingerprint(sizes),
                "source_controls": controls,
            }
        )


class SizeResolutionReport(StrictModule, NonTrainableState):
    """Resolution decisions, including exact edge-length-aware growth evidence.

    ``graded_count`` counts samples lowered by hard growth limits and
    ``maximum_gradation_violation`` is the a-posteriori relative excess of any
    graded size over its growth bound (zero when every limited edge is satisfied).
    """

    control_ids: tuple[str, ...] = eqx.field(static=True)
    winning_control_ids: tuple[str, ...] = eqx.field(static=True)
    overlapping_scope_pairs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    clamped: bool = eqx.field(static=True)
    provider_resolved: bool = eqx.field(static=True)
    graded_count: int = eqx.field(static=True)
    maximum_gradation_violation: float = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        controls: tuple[SizeControl, ...],
        field: ResolvedSizeField,
        /,
        *,
        winning_control_ids: tuple[str, ...],
        overlapping_scope_pairs: tuple[tuple[str, str], ...] = (),
        clamped: bool = False,
        provider_resolved: bool = False,
        graded_count: int = 0,
        maximum_gradation_violation: float = 0.0,
    ) -> None:
        if not controls or not all(
            isinstance(
                control,
                (UniformSizeControl, CurvatureSizeControl, ProximitySizeControl),
            )
            for control in controls
        ):
            raise TypeError("controls must contain supported size controls.")
        identifiers = tuple(control.control_id for control in controls)
        winners = tuple(str(identifier) for identifier in winning_control_ids)
        if len(winners) != field.values.shape[0] or any(
            identifier not in identifiers for identifier in winners
        ):
            raise ValueError("winning_control_ids must identify one control per sample.")
        overlaps = tuple(
            (str(first), str(second)) for first, second in overlapping_scope_pairs
        )
        graded = int(graded_count)
        violation = float(maximum_gradation_violation)
        if graded < 0 or not np.isfinite(violation) or violation < 0.0:
            raise ValueError("Gradation evidence must be finite and non-negative.")
        self.control_ids = identifiers
        self.winning_control_ids = winners
        self.overlapping_scope_pairs = overlaps
        self.clamped = bool(clamped)
        self.provider_resolved = bool(provider_resolved)
        self.graded_count = graded
        self.maximum_gradation_violation = violation
        self.field_id = field.field_id
        self.report_id = canonical_fingerprint(
            {
                "kind": "size-resolution-report",
                "controls": identifiers,
                "winners": winners,
                "overlaps": overlaps,
                "clamped": bool(clamped),
                "provider_resolved": bool(provider_resolved),
                "graded_count": graded,
                "maximum_gradation_violation": violation,
                "field": field.field_id,
            }
        )


def _control_scopes(control: SizeControl, /) -> tuple[MeshingScope, ...]:
    if isinstance(control, ProximitySizeControl):
        return control.source_scope, control.target_scope
    return (control.scope,)


def _nearest_rows(
    points: np.ndarray, queries: np.ndarray, candidates: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest candidate row and distance for each query row (BVH)."""
    candidate_points = points[candidates]
    hierarchy = prepare_bvh(candidate_points, candidate_points, dtype=jnp.float64)
    nearest = bvh_nearest_items(hierarchy, points[queries])
    items = np.asarray(nearest.items[:, 0])
    return candidates[items], np.sqrt(np.asarray(nearest.distance_squared[:, 0]))


def _proximity_gaps(
    control: ProximitySizeControl,
    points: np.ndarray,
    identifiers: np.ndarray,
    normals: np.ndarray | None,
    /,
) -> np.ndarray:
    """Gap to the nearest sample on the opposite scope's distinct entities.

    Source samples measure to target-only entities and vice versa, so a surface
    never measures a gap to itself. With ``opposite_normals_only`` a gap counts
    only when the nearest opposite sample faces the query (negative normal dot
    product); other samples receive an infinite gap.
    """
    source = np.isin(identifiers, np.asarray(control.source_scope.entity_ids))
    target = np.isin(identifiers, np.asarray(control.target_scope.entity_ids))
    source_only = np.flatnonzero(source & ~target)
    target_only = np.flatnonzero(target & ~source)
    if not source_only.size or not target_only.size:
        raise ValueError(
            "Proximity controls require samples on disjoint source and target entities."
        )
    if control.opposite_normals_only and normals is None:
        raise ValueError("Opposite-normal proximity controls require sample normals.")
    gaps = np.full((points.shape[0],), np.inf)
    for queries, candidates in (
        (np.flatnonzero(source), target_only),
        (np.flatnonzero(target), source_only),
    ):
        nearest, distance = _nearest_rows(points, queries, candidates)
        if control.opposite_normals_only:
            assert normals is not None
            facing = np.sum(normals[queries] * normals[nearest], axis=1) < 0.0
            distance = np.where(facing, distance, np.inf)
        gaps[queries] = np.minimum(gaps[queries], distance)
    return gaps


def _control_candidate(
    control: SizeControl,
    mask: np.ndarray,
    points: np.ndarray,
    identifiers: np.ndarray,
    curvature: np.ndarray | None,
    normals: np.ndarray | None,
    /,
) -> np.ndarray:
    if isinstance(control, UniformSizeControl):
        value = np.full((points.shape[0],), control.target_size)
    elif isinstance(control, CurvatureSizeControl):
        if curvature is None or curvature.shape != identifiers.shape:
            raise ValueError("Curvature controls require aligned curvature samples.")
        if np.any(~np.isfinite(curvature[mask])) or np.any(curvature[mask] < 0.0):
            raise ValueError("Curvature samples must be finite and non-negative.")
        value = np.divide(
            2.0 * np.sin(0.5 * control.normal_angle),
            curvature,
            out=np.full(curvature.shape, np.inf),
            where=curvature > 0.0,
        )
    elif isinstance(control, ProximitySizeControl):
        value = (
            _proximity_gaps(control, points, identifiers, normals)
            / control.elements_per_gap
        )
    else:
        raise TypeError("Unsupported size control.")
    if control.minimum_size is not None:
        value = np.maximum(value, control.minimum_size)
    if control.maximum_size is not None:
        value = np.minimum(value, control.maximum_size)
    return value


def _hard_growth(
    controls: tuple[SizeControl, ...],
    masks: np.ndarray,
    edges: np.ndarray,
    /,
) -> np.ndarray:
    growth = np.full((edges.shape[0],), np.inf)
    for index, control in enumerate(controls):
        if (
            isinstance(control, UniformSizeControl)
            and control.strength is SizeControlStrength.HARD
            and control.maximum_growth_rate is not None
        ):
            within = masks[index, edges[:, 0]] & masks[index, edges[:, 1]]
            growth[within] = np.minimum(growth[within], control.maximum_growth_rate)
    return growth


def _select_controls(
    candidates: np.ndarray,
    masks: np.ndarray,
    hard: np.ndarray,
    priorities: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    combination: SizeCombinationPolicy,
    /,
) -> np.ndarray:
    """Winning control of every sample under the explicit combination policy.

    Per sample, the pool is the active hard controls, else every active control;
    targets are clipped to the sample's hard interval. The first failing sample
    decides the refusal, as a sample-ordered evaluation would.
    """
    hard_active = masks & hard[:, None]
    has_hard = np.any(hard_active, axis=0)
    pool = np.where(has_hard[None, :], hard_active, masks)
    choices = np.clip(candidates, lower[None, :], upper[None, :])
    match combination:
        case SizeCombinationPolicy.REJECT_HARD_CONFLICTS:
            finalists = pool
            checked = has_hard
            message = "Overlapping hard size-control targets conflict."
        case SizeCombinationPolicy.EXPLICIT_PRIORITY:
            ranks = np.where(pool, priorities[:, None], np.iinfo(np.int64).min)
            finalists = pool & (ranks == np.max(ranks, axis=0)[None, :])
            checked = np.ones(has_hard.shape, dtype=np.bool_)
            message = "Equal-priority size-control targets conflict on one sample."
        case _:
            raise ValueError(f"Unsupported size combination policy {combination!r}.")
    first = np.argmax(finalists, axis=0)
    leading = np.take_along_axis(choices, first[None, :], axis=0)
    uncovered = ~np.any(masks, axis=0)
    failed = uncovered | (checked & np.any(finalists & (choices != leading), axis=0))
    if np.any(failed):
        if uncovered[int(np.argmax(failed))]:
            raise ValueError("Size controls do not cover every sample.")
        raise ValueError(message)
    if combination is SizeCombinationPolicy.EXPLICIT_PRIORITY:
        return first
    # The first least clipped target wins; a NaN target wins as argmin's would.
    undefined = finalists & np.isnan(choices)
    least = np.min(np.where(finalists, choices, np.inf), axis=0)
    return np.where(
        np.any(undefined, axis=0),
        np.argmax(undefined, axis=0),
        np.argmax(finalists & (choices == least[None, :]), axis=0),
    )


def _apply_hard_growth(
    controls: tuple[SizeControl, ...],
    masks: np.ndarray,
    adjacency: ArrayLike | None,
    points: np.ndarray,
    raw: np.ndarray,
    admissible_minimum: np.ndarray,
    /,
) -> tuple[np.ndarray, int, float]:
    """Exact edge-length-aware growth limits through the metric owner."""
    if adjacency is None:
        return raw, 0, 0.0
    edges = np.asarray(adjacency)
    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError("Size-field adjacency must contain integer sample rows.")
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or np.any(edges < 0)
        or np.any(edges >= points.shape[0])
    ):
        raise ValueError("Size-field adjacency must have shape (edges, 2).")
    growth = _hard_growth(controls, masks, edges)
    limited = np.isfinite(growth)
    limited_edges = edges[limited].astype(np.int64, copy=False)
    lengths = np.linalg.norm(
        points[limited_edges[:, 1]] - points[limited_edges[:, 0]], axis=1
    )
    resolved, _, _ = _grade_scalar_sizes(
        raw, limited_edges, lengths, growth[limited], MetricGradationKind.PHYSICAL
    )
    if np.any(resolved < admissible_minimum):
        raise ValueError("Hard growth limits are incompatible with hard size intervals.")
    violation = _scalar_violation(
        resolved, limited_edges, lengths, growth[limited], MetricGradationKind.PHYSICAL
    )
    return resolved, int(np.count_nonzero(resolved < raw)), violation


def resolve_size_controls(
    controls: tuple[SizeControl, ...],
    sample_points: ArrayLike,
    sample_entity_ids: ArrayLike,
    domain: SizeFieldDomain,
    /,
    *,
    curvature: ArrayLike | None = None,
    normals: ArrayLike | None = None,
    adjacency: ArrayLike | None = None,
    combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
) -> tuple[ResolvedSizeField, SizeResolutionReport]:
    """Resolve targets only after intersecting every active hard size interval.

    Proximity gaps are measured by exact BVH nearest queries between samples of
    the source and target scopes. Hard growth limits on ``adjacency`` edges
    enforce ``h_i <= h_j + (rate - 1) |x_i - x_j|`` exactly through the metric
    owner's minimum-first relaxation.
    """

    controls_ = tuple(controls)
    if not controls_:
        raise ValueError("At least one size control is required.")
    if not all(
        isinstance(
            control,
            (UniformSizeControl, CurvatureSizeControl, ProximitySizeControl),
        )
        for control in controls_
    ):
        raise TypeError("controls must contain supported size controls.")
    if not isinstance(domain, SizeFieldDomain):
        raise TypeError("domain must be SizeFieldDomain.")
    if not isinstance(combination, SizeCombinationPolicy):
        raise TypeError("combination must be SizeCombinationPolicy.")
    points = np.asarray(sample_points, dtype=np.float64)
    identifiers = np.asarray(sample_entity_ids, dtype=np.int64)
    if (
        points.ndim != 2
        or not np.all(np.isfinite(points))
        or identifiers.shape != (points.shape[0],)
    ):
        raise ValueError("Size samples and entity IDs must be finite and aligned.")
    curvature_values = (
        None if curvature is None else np.asarray(curvature, dtype=np.float64)
    )
    normal_values = None if normals is None else np.asarray(normals, dtype=np.float64)
    if normal_values is not None and (
        normal_values.shape != points.shape or not np.all(np.isfinite(normal_values))
    ):
        raise ValueError("Sample normals must be finite and aligned with samples.")
    candidates = []
    masks = []
    lower_bounds = []
    upper_bounds = []
    overlaps = []
    for left, control in enumerate(controls_):
        scopes = _control_scopes(control)
        entity_ids = np.unique(
            np.concatenate(
                tuple(np.asarray(scope.entity_ids, dtype=np.int64) for scope in scopes)
            )
        )
        mask = np.isin(identifiers, entity_ids)
        if not np.any(mask):
            raise ValueError("A size control resolves to no supplied sample entities.")
        for second in controls_[left + 1 :]:
            if any(
                first_scope.entity_set_id == second_scope.entity_set_id
                and np.intersect1d(first_scope.entity_ids, second_scope.entity_ids).size
                for first_scope in scopes
                for second_scope in _control_scopes(second)
            ):
                overlaps.append((control.control_id, second.control_id))
        candidates.append(
            _control_candidate(
                control, mask, points, identifiers, curvature_values, normal_values
            )
        )
        masks.append(mask)
        lower_bounds.append(
            0.0
            if control.strength is SizeControlStrength.SOFT
            or control.minimum_size is None
            else control.minimum_size
        )
        upper_bounds.append(
            np.inf
            if control.strength is SizeControlStrength.SOFT
            or control.maximum_size is None
            else control.maximum_size
        )
    candidate_array = np.stack(candidates)
    mask_array = np.stack(masks)
    hard_array = np.asarray(
        tuple(control.strength is SizeControlStrength.HARD for control in controls_)
    )
    priority_array = np.asarray(tuple(control.priority for control in controls_))
    lower_array = np.asarray(lower_bounds, dtype=np.float64)[:, None]
    upper_array = np.asarray(upper_bounds, dtype=np.float64)[:, None]
    hard_mask = mask_array & hard_array[:, None]
    admissible_minimum = np.max(np.where(hard_mask, lower_array, 0.0), axis=0)
    admissible_maximum = np.min(np.where(hard_mask, upper_array, np.inf), axis=0)
    if np.any(admissible_minimum > admissible_maximum):
        raise ValueError("Active hard size intervals have an empty intersection.")

    selected = _select_controls(
        candidate_array,
        mask_array,
        hard_array,
        priority_array,
        admissible_minimum,
        admissible_maximum,
        combination,
    )
    preferred = np.asarray(
        candidate_array[selected, np.arange(points.shape[0])], dtype=np.float64
    )
    raw = np.clip(preferred, admissible_minimum, admissible_maximum)
    control_ids = tuple(control.control_id for control in controls_)
    winners = [control_ids[index] for index in selected.tolist()]
    if np.any(~np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("Resolved size targets must be positive and finite.")

    resolved, graded_count, violation = _apply_hard_growth(
        controls_, mask_array, adjacency, points, raw, admissible_minimum
    )
    field = ResolvedSizeField(
        domain,
        points,
        resolved,
        sample_entity_ids=identifiers,
        source_control_ids=tuple(control.control_id for control in controls_),
    )
    return field, SizeResolutionReport(
        controls_,
        field,
        winning_control_ids=tuple(winners),
        overlapping_scope_pairs=tuple(overlaps),
        clamped=not np.array_equal(resolved, preferred),
        graded_count=graded_count,
        maximum_gradation_violation=violation,
    )


def size_field_metric(
    field: ResolvedSizeField, scope: MeshingScope, /
) -> MeshMetricField:
    """Compile a resolved size field into isotropic metric constraints.

    Row ``i`` becomes ``h_i**-2 I`` for the entity ``scope.entity_ids[i]``, which
    must equal the field's sample entity IDs in order. Declared size bounds are
    the resolved extremes, so the metric can be combined with solution-adaptive
    metrics through :func:`combine_mesh_metrics`. The field's hard growth limits
    are already resolved into the sizes; requested execution gradation belongs to
    a `MetricGradationPolicy` or the provider options.
    """
    if not isinstance(field, ResolvedSizeField):
        raise TypeError("field must be ResolvedSizeField.")
    if not isinstance(scope, MeshingScope):
        raise TypeError("scope must be MeshingScope.")
    if not np.array_equal(
        np.asarray(scope.entity_ids, dtype=np.int64),
        np.asarray(field.sample_entity_ids),
    ):
        raise ValueError("Scope entity IDs must equal the size-field sample entity IDs.")
    sizes = np.asarray(field.values, dtype=np.float64)
    dimension = field.sample_points.shape[1]
    values = np.eye(dimension)[None, :, :] / sizes[:, None, None] ** 2
    return MeshMetricField(
        scope,
        values,
        minimum_size=float(np.min(sizes)),
        maximum_size=float(np.max(sizes)),
        maximum_anisotropy=1.0,
    )


__all__ = [
    "CurvatureSizeControl",
    "ProximitySizeControl",
    "ResolvedSizeField",
    "SizeCombinationPolicy",
    "SizeCompliancePolicy",
    "SizeControl",
    "SizeControlStrength",
    "SizeFieldDomain",
    "SizeResolutionReport",
    "UniformSizeControl",
    "resolve_size_controls",
    "size_field_metric",
]
