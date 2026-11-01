#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cache, partial
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._bvh import bvh_overlap_pair_blocks, PackedBVH, prepare_bvh
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...discretization._cell_geometry import (
    CellGeometrySpec,
    ScalarCoordinateElement,
)
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._cell_mesh import CellMesh
from ...discretization._coordinate_enclosure import (
    compose_expression,
    coordinate_map_expressions,
    CoordinateSourceBank,
    derivative,
    evaluate_expression,
    Expression,
    Polynomial,
    polynomial_bounds,
)
from ...discretization._exact_algebraic_geometry import (
    coefficient_endpoint,
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from ...geometry._certificate import FieldRegularity
from ...geometry._contracts import CompiledGeometry
from ...geometry._embedding_state import MeshCertificateFinding
from ...geometry._interval_enclosure import (
    IntervalExtensionError,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from ...geometry._mesh_certificates import MeshCertificateBinding
from ...geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryProjectionQuery,
    SourceProjectionStatus,
)
from ...linalg._exact import (
    ExactRationalAlgebraicFactorization,
    factor_exact_rational_algebraic,
)
from ...typing import ConvertibleToArray
from .._source_identity import source_entity_id
from ._adaptive_discovery import (
    _evaluate_interval_boxes,
    _FieldEnclosures,
    _root_on_edges,
    AdaptiveImplicitBoundarySource,
    AdaptiveImplicitBoxIssue,
    AdaptiveImplicitSurface,
)
from ._analytic_profile import AnalyticImplicitProfile


if TYPE_CHECKING:
    from ...geometry._source_fidelity import ValidatedSourceFidelityWitness
    from ...geometry._source_fidelity_certificate import SourceFidelityCertificate
type Interval = tuple[float, float]
type GridCorner = tuple[int, int, int]
type GridEdge = tuple[GridCorner, GridCorner]
type SourceCellKey = tuple[int, int, int, int]
type RootEdgeTriangle = tuple[GridEdge, GridEdge, GridEdge]
type ReferencePoint = tuple[Fraction, ...]
type ExactReferencePoint = tuple[ExactScalar, ...]


def _exact_reference_scalar(value: object, /) -> ExactScalar:
    if isinstance(value, (Fraction, RadicalElement)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if math.isfinite(scalar):
            return Fraction.from_float(scalar)
        raise ValueError("Target reference coordinates must be finite.")
    raise TypeError(
        "Target reference coordinates require exact scalars or authored floats."
    )


type ReferenceTriangle = tuple[ReferencePoint, ReferencePoint, ReferencePoint]
type TrianglePoint = tuple[Fraction, Fraction]
type TriangleReference = tuple[TrianglePoint, TrianglePoint, TrianglePoint]
type _BarycentricSolver = tuple[
    tuple[int, int], ExactRationalAlgebraicFactorization[RadicalElement]
]
type SplitTriangles = tuple[
    TriangleReference,
    TriangleReference,
    TriangleReference,
    TriangleReference,
]
type Polynomial3 = tuple[Polynomial, Polynomial, Polynomial]
type IntervalVector2 = tuple[Interval, Interval]
type IntervalMatrix2 = tuple[IntervalVector2, IntervalVector2]
type FractionMatrix2 = tuple[
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
]
type IntervalVector3 = tuple[Interval, Interval, Interval]
type IntervalMatrix3 = tuple[IntervalVector3, IntervalVector3, IntervalVector3]
type IntervalDerivative3x2 = tuple[
    IntervalVector2,
    IntervalVector2,
    IntervalVector2,
]
type RootCorner = tuple[GridEdge, np.ndarray, float] | None


@runtime_checkable
class _AdaptiveImplicitBoundaryCarrier(SourceBoundaryProjectionQuery, Protocol):
    @property
    def adaptive_source(self) -> AdaptiveImplicitBoundarySource: ...

    @property
    def authoring_profile(self) -> AnalyticImplicitProfile | None: ...

    def require_current(self, /) -> None: ...


type AdaptiveImplicitSourceQuery = (
    AdaptiveImplicitBoundarySource | _AdaptiveImplicitBoundaryCarrier
)


@dataclass(frozen=True, slots=True)
class _PieceProof:
    triangle: TriangleReference
    solver: _BarycentricSolver
    maximum_root_parameter: float
    root_parameter_bound: float
    field_residual: Interval
    target_to_source: Interval
    normal_deviation: Interval
    chart_jacobian_lower: float
    source_gradient_lower: float
    source_gradient_upper: float
    directional_derivative_lower: float
    directional_derivative_upper: float
    target_jacobian: IntervalMatrix2
    # Enclosure of the target-map derivative DP over the piece.
    target_derivatives: IntervalDerivative3x2
    # DP K over the piece for a conditioned face (see `_tangent_conditioner`).
    conditioned_derivatives: IntervalDerivative3x2 | None
    # Enclosure of the piece's target points P.
    target_box: IntervalVector3
    # Enclosure of grad f over the piece's target box.
    source_gradient: IntervalVector3
    # Certified normal-line root parameters of every target point of the piece.
    root_parameters: Interval
    # Enclosure of the lifted points P + t grad f(P) over the piece.
    root_box: IntervalVector3


type _PieceOutcome = tuple[_PieceProof | None, str, bool]
type _PieceProver = Callable[[Sequence[TriangleReference]], tuple[_PieceSteps, ...]]


@dataclass(frozen=True, slots=True)
class ImplicitRootChartEvidence:
    """Recomputable whole-face evidence for one adaptive implicit root chart."""

    row: int
    target_cell_global_id: int
    target_entity_global_id: int
    target_local_entity_index: int
    source_cell_key: SourceCellKey
    root_edge_keys: RootEdgeTriangle
    status: str
    field_residual_bound: float
    target_to_root_lower_bound: float
    target_to_root_bound: float
    source_normal_deviation_bounds: Interval
    chart_jacobian_lower: float
    chart_injectivity_margin: float
    source_gradient_lower: float
    detail: str


@dataclass(frozen=True, slots=True)
class _PreparedFace:
    row: int
    block_index: int
    cell_row: int
    cell_global_id: int
    entity_global_id: int
    local_entity_index: int
    source_cell_key: SourceCellKey
    root_edge_keys: RootEdgeTriangle
    reference_corners: ReferenceTriangle
    reference_solver: _BarycentricSolver
    source_parameters: TriangleReference
    cell_expressions: Polynomial3
    face_expressions: Polynomial3
    face_enclosure: _FaceEnclosure
    pieces: tuple[_PieceProof, ...]
    root_corner_points: tuple[RootCorner, ...]
    projected_axes: tuple[int, int, int]
    orientation: int
    maximum_root_parameter: float
    evidence: ImplicitRootChartEvidence


@dataclass(frozen=True, slots=True)
class ImplicitRootChartPoint:
    """One certified local source-root lift; not a global closest-point result."""

    point: np.ndarray
    displacement_upper: float
    root_error: float
    source_cell_key: SourceCellKey
    root_edge_keys: RootEdgeTriangle
    status: str


class AdaptiveImplicitRootChartMap:
    """Whole-face target-to-root map over a retained adaptive scaffold triangle.

    The lift follows the scalar-field gradient line through the target point.
    Its local root is isolated by a monotone interval tube, but is deliberately
    not described as the globally closest point. The direction is a deterministic
    function of the physical point, so adjacent target facets share the same map
    wherever their full coordinate maps agree.
    """

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: AdaptiveImplicitSourceQuery,
        witness: SourceBoundaryChartWitness,
        frame_id: str,
        faces: tuple[_PreparedFace, ...],
        bounds: _FieldEnclosures,
        tolerance: float | None,
        maximum_normal_deviation: float | None,
        surface: AdaptiveImplicitSurface,
        hessian_program: PreparedIntervalFunction,
        /,
    ) -> None:
        self.mesh = mesh
        self.geometry = geometry
        self.source = source
        self.witness = witness
        self.frame_id = canonical_identifier(frame_id, "frame_id")
        self.source_id = str(source.source_id)
        self.source_revision = str(source.source_revision)
        self.mesh_id = mesh.mesh_id
        self.topology_id = mesh.topology_id
        self.geometry_id = cell_geometry_id(geometry)
        self.chart_evidence = tuple(face.evidence for face in faces)
        self._faces = faces
        self._bounds = bounds
        self._surface = surface
        self._hessian_program = hessian_program
        self.tolerance = tolerance
        self.maximum_normal_deviation = maximum_normal_deviation
        self.witness_id = canonical_fingerprint(
            {
                "kind": "adaptive-implicit-root-chart-map",
                "source": (self.source_id, self.source_revision, self.frame_id),
                "mesh": (self.mesh_id, self.topology_id, self.geometry_id),
                "controls": (self.tolerance, self.maximum_normal_deviation),
                "chart_witness": witness.witness_id,
                "evidence": tuple(
                    (
                        row.status,
                        row.source_cell_key,
                        row.root_edge_keys,
                        row.field_residual_bound,
                        row.target_to_root_lower_bound,
                        row.target_to_root_bound,
                        row.source_normal_deviation_bounds,
                        row.chart_jacobian_lower,
                        row.chart_injectivity_margin,
                        row.source_gradient_lower,
                    )
                    for row in self.chart_evidence
                ),
            }
        )

    @property
    def status(self) -> str:
        return (
            "certified"
            if all(row.status == "certified" for row in self.chart_evidence)
            else "unresolved"
        )

    def project_target(
        self,
        row: int,
        target_cell_reference: ConvertibleToArray | Sequence[ExactScalar],
        /,
    ) -> ImplicitRootChartPoint:
        return self.project_targets((row,), (target_cell_reference,))[0]

    def project_targets(
        self,
        rows: Sequence[int],
        target_cell_references: Sequence[ConvertibleToArray | Sequence[ExactScalar]],
        /,
    ) -> tuple[ImplicitRootChartPoint, ...]:
        """Lift many full target-cell reference points; each lift is the single-point lift.

        The interval queries of all lifts run batched, which leaves every lift
        unchanged; a point shared by several queries is lifted once.
        """
        if len(rows) != len(target_cell_references):
            raise ValueError("rows and target_cell_references must align.")
        results: list[ImplicitRootChartPoint | None] = []
        lifts: list[tuple[np.ndarray, _PreparedFace, float]] = []
        lift_rows: list[int] = []
        for row, target_cell_reference in zip(rows, target_cell_references, strict=True):
            query = self._target_query(row, target_cell_reference)
            if isinstance(query, ImplicitRootChartPoint):
                results.append(query)
                continue
            lift_rows.append(len(results))
            results.append(None)
            lifts.append(query)
        for index, lifted in zip(
            lift_rows,
            _lift_points(self._surface, self.frame_id, self._bounds, lifts),
            strict=True,
        ):
            results[index] = lifted
        return tuple(result for result in results if result is not None)

    def excludes_point(
        self,
        rows: Sequence[int],
        point: np.ndarray,
        error: float,
        /,
    ) -> tuple[bool, ...]:
        """Decide per certified face that no chart point lifts within ``error`` of ``point``.

        A piece is excluded when the distance lower bound from ``point`` to its
        root box (the enclosure of its lifted points) exceeds ``error``.
        Starting from the prepared pieces, only pieces that are not
        excluded are split and re-proved, until every piece is excluded (True)
        or a remaining piece cannot be proved or split further in binary64
        (False, the face may cover the point). The prepared chart is unchanged.
        """
        target = np.asarray(point, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("point must be one finite physical point.")
        error_ = float(error)
        if not math.isfinite(error_) or error_ < 0.0:
            raise ValueError("error must be finite and nonnegative.")
        faces = [self._faces[int(row)] for row in rows]
        if any(face.evidence.status != "certified" for face in faces):
            raise ValueError("Only certified implicit charts can exclude a point.")
        return tuple(
            _run_faces(
                self._bounds,
                self._hessian_program,
                [
                    _point_exclusion_steps(self._surface, face, target, error_)
                    for face in faces
                ],
            )
        )

    def _target_query(
        self,
        row: int,
        target_cell_reference: ConvertibleToArray | Sequence[ExactScalar],
        /,
    ) -> ImplicitRootChartPoint | tuple[np.ndarray, _PreparedFace, float]:
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
            raise TypeError("row must be an integer chart-witness row.")
        index = int(row)
        if index < 0 or index >= len(self._faces):
            raise IndexError("implicit root-chart row is outside its witness.")
        face = self._faces[index]
        if face.evidence.status != "certified":
            raise ValueError(
                face.evidence.detail or "The implicit root chart is unresolved."
            )
        reference = tuple(
            _exact_reference_scalar(value)
            for value in np.asarray(target_cell_reference, dtype=object).reshape(-1)
        )
        if len(reference) != len(face.reference_corners[0]):
            raise ValueError("target_cell_reference has the wrong reference dimension.")
        barycentric = _barycentric(
            reference, face.reference_corners, solver=face.reference_solver
        )
        if any(value < 0 for value in barycentric) or sum(barycentric, Fraction(0)) != 1:
            raise ValueError(
                "target_cell_reference lies outside its witnessed source face."
            )
        target_chart_point = (barycentric[1], barycentric[2])
        physical_exact = tuple(
            evaluate_expression(expression, reference)
            for expression in face.cell_expressions
        )
        point = np.asarray(
            tuple(float(value) for value in physical_exact), dtype=np.float64
        )
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(
                "target coordinate map produced a non-finite implicit query point."
            )
        for corner, root_corner in enumerate(face.root_corner_points):
            unit = tuple(Fraction(int(index == corner)) for index in range(3))
            if barycentric == unit and root_corner is not None:
                _, root_point, root_error = root_corner
                return ImplicitRootChartPoint(
                    np.array(root_point, copy=True),
                    math.nextafter(root_error, math.inf),
                    math.nextafter(root_error, math.inf),
                    face.source_cell_key,
                    face.root_edge_keys,
                    "certified_root_chart",
                )
        for piece in face.pieces:
            local = _barycentric(target_chart_point, piece.triangle, solver=piece.solver)
            if all(value >= 0 for value in local):
                return point, face, piece.maximum_root_parameter
        raise ValueError(
            "Target reference point is not covered by a certified implicit chart subtriangle."
        )

    def evaluate_target(
        self,
        row: int,
        target_cell_reference: ConvertibleToArray | Sequence[ExactScalar],
        /,
    ) -> np.ndarray:
        """Evaluate the source-root lift at one full target-cell reference point."""
        return np.array(self.project_target(row, target_cell_reference).point, copy=True)

    def evaluate_targets(
        self,
        rows: Sequence[int],
        target_cell_references: Sequence[ConvertibleToArray | Sequence[ExactScalar]],
        /,
    ) -> np.ndarray:
        """Evaluate many source-root lifts with batched interval queries, shape ``(N, 3)``."""
        lifted = self.project_targets(rows, target_cell_references)
        return np.asarray([item.point for item in lifted], dtype=np.float64).reshape(
            (len(lifted), 3)
        )

    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: AdaptiveImplicitSourceQuery,
        witness: SourceBoundaryChartWitness,
        /,
    ) -> None:
        fresh = prepare_adaptive_implicit_root_charts(
            mesh,
            geometry,
            source,
            witness,
            frame_id=self.frame_id,
            tolerance=self.tolerance,
            maximum_normal_deviation=self.maximum_normal_deviation,
        )
        if fresh.witness_id != self.witness_id:
            raise ValueError(
                "Adaptive implicit root-chart map no longer matches its source and target map."
            )

    def require_request(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: AdaptiveImplicitSourceQuery,
        witness: SourceBoundaryChartWitness,
        /,
        *,
        tolerance: float | None,
        maximum_normal_deviation: float | None,
    ) -> None:
        """Require this map to be the one prepared for exactly these operands."""
        _, inherited_frame = _source_authority(source)
        tolerance_ = None if tolerance is None else float(tolerance)
        normal_ = (
            None if maximum_normal_deviation is None else float(maximum_normal_deviation)
        )
        if (
            self.mesh_id != mesh.mesh_id
            or self.topology_id != mesh.topology_id
            or self.geometry_id != cell_geometry_id(geometry)
            or self.source_id != str(source.source_id)
            or self.source_revision != str(source.source_revision)
            or (inherited_frame and self.frame_id != inherited_frame)
            or self.witness.witness_id != witness.witness_id
            or self.tolerance != tolerance_
            or self.maximum_normal_deviation != normal_
        ):
            raise ValueError(
                "Prepared implicit root charts were prepared for a different mesh, "
                "geometry, source, witness or control."
            )


def _interval_add(first: Interval, second: Interval, /) -> Interval:
    return (
        math.nextafter(first[0] + second[0], -math.inf),
        math.nextafter(first[1] + second[1], math.inf),
    )


def _interval_scale(value: Interval, scalar: float, /) -> Interval:
    products = (value[0] * scalar, value[1] * scalar)
    return (
        math.nextafter(min(products), -math.inf),
        math.nextafter(max(products), math.inf),
    )


def _interval_multiply(first: Interval, second: Interval, /) -> Interval:
    a, b = first
    c, d = second
    ac, ad, bc, bd = a * c, a * d, b * c, b * d
    if ac != ac or ad != ad or bc != bc or bd != bd:
        return (-math.inf, math.inf)
    return (
        math.nextafter(min(ac, ad, bc, bd), -math.inf),
        math.nextafter(max(ac, ad, bc, bd), math.inf),
    )


def _interval_subtract(first: Interval, second: Interval, /) -> Interval:
    return _interval_add(first, (-second[1], -second[0]))


def _interval_divide(numerator: Interval, denominator: Interval, /) -> Interval:
    if denominator[0] <= 0.0 <= denominator[1]:
        return (-math.inf, math.inf)
    reciprocal = (
        math.nextafter(1.0 / denominator[1], -math.inf),
        math.nextafter(1.0 / denominator[0], math.inf),
    )
    return _interval_multiply(numerator, reciprocal)


def _interval_vector_norm_bounds(
    intervals: tuple[Interval, ...],
) -> Interval:
    lower_squares = []
    upper_squares = []
    for lower, upper in intervals:
        minimum = 0.0 if lower <= 0.0 <= upper else min(abs(lower), abs(upper))
        maximum = max(abs(lower), abs(upper))
        lower_squares.append(minimum * minimum)
        upper_squares.append(maximum * maximum)
    lower = math.nextafter(math.sqrt(max(0.0, math.fsum(lower_squares))), 0.0)
    upper = math.nextafter(math.sqrt(math.fsum(upper_squares)), math.inf)
    return lower, upper


def _exact_interval(value: ExactScalar) -> tuple[float, float]:
    if isinstance(value, RadicalElement):
        return (
            coefficient_endpoint(value, upper=False, saturate=True),
            coefficient_endpoint(value, upper=True, saturate=True),
        )
    represented = float(value)
    if not math.isfinite(represented):
        return (-math.inf, math.inf)
    numerator, denominator = represented.as_integer_ratio()
    left = numerator * value.denominator
    right = value.numerator * denominator
    sign = (left > right) - (left < right)
    return (
        represented if sign <= 0 else math.nextafter(represented, -math.inf),
        represented if sign >= 0 else math.nextafter(represented, math.inf),
    )


def _poly_interval(expression: Polynomial, triangle: TriangleReference) -> Interval:
    controls = _simplex_controls(expression)
    if controls is not None:
        return _restricted_bounds(controls, triangle)
    arguments = _triangle_arguments(triangle)
    restricted = compose_expression(expression, arguments)
    if not isinstance(restricted, dict):
        raise TypeError(
            "Implicit source chart target maps must be polynomial coordinate maps."
        )
    return polynomial_bounds(restricted, "simplex", 2)


# A standard-triangle Bernstein control index: the exponents of the barycentric
# weights of the corners (0, 0), (1, 0) and (0, 1).
type _ControlIndex = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _SimplexControls:
    """Exact standard-triangle Bernstein controls as integers over one denominator.

    A polynomial restricted to a subtriangle ``(A, B, C)`` has the blossom
    values ``b(A^a, B^b, C^c)`` as its Bernstein controls, which integer de
    Casteljau steps with the corners' common-denominator barycentric weights
    give exactly. An invertible affine restriction keeps the least degree, so
    these are the controls whose extremes `polynomial_bounds` rounds outward
    for the composed polynomial.
    """

    degree: int
    numerators: dict[_ControlIndex, int]
    denominator: int


@cache
def _control_indices(degree: int, /) -> tuple[_ControlIndex, ...]:
    return tuple(
        (degree - first - second, first, second)
        for first in range(degree + 1)
        for second in range(degree + 1 - first)
    )


def _simplex_controls(expression: Polynomial, /) -> _SimplexControls | None:
    """Standard-triangle Bernstein controls of a rational bivariate polynomial.

    ``u^a v^b`` contributes ``j! k! (n-a-b)! / ((j-a)! (k-b)! n!)`` to control
    ``(i, j, k)``. Non-rational coefficients return ``None``.
    """
    terms: list[tuple[int, int, Fraction]] = []
    for exponent, coefficient in expression.items():
        if not isinstance(coefficient, Fraction) or len(exponent) != 2:
            return None
        if coefficient:
            terms.append((exponent[0], exponent[1], coefficient))
    degree = max((first + second for first, second, _ in terms), default=0)
    factorial = math.factorial
    controls: dict[_ControlIndex, Fraction] = {}
    for index in _control_indices(degree):
        _, along_u, along_v = index
        total = Fraction(0)
        for first, second, coefficient in terms:
            if first <= along_u and second <= along_v:
                total += coefficient * Fraction(
                    factorial(along_u)
                    * factorial(along_v)
                    * factorial(degree - first - second),
                    factorial(along_u - first)
                    * factorial(along_v - second)
                    * factorial(degree),
                )
        controls[index] = total
    denominator = math.lcm(*(value.denominator for value in controls.values()))
    return _SimplexControls(
        degree,
        {
            index: value.numerator * (denominator // value.denominator)
            for index, value in controls.items()
        },
        denominator,
    )


def _restricted_bounds(
    controls: _SimplexControls, triangle: TriangleReference, /
) -> Interval:
    """`polynomial_bounds` of the polynomial restricted to a rational subtriangle."""
    degree = controls.degree
    if degree == 0 or triangle == _STANDARD_TRIANGLE:
        values = tuple(controls.numerators.values())
        scale = controls.denominator
    else:
        common = math.lcm(*(value.denominator for point in triangle for value in point))
        corners = []
        for first, second in triangle:
            scaled_first = first.numerator * (common // first.denominator)
            scaled_second = second.numerator * (common // second.denominator)
            corners.append(
                (common - scaled_first - scaled_second, scaled_first, scaled_second)
            )
        # Nets after applying a nondecreasing corner sequence ending at `last`.
        nets: list[tuple[int, dict[_ControlIndex, int]]] = [(0, controls.numerators)]
        for remaining in range(degree - 1, -1, -1):
            indices = _control_indices(remaining)
            following: list[tuple[int, dict[_ControlIndex, int]]] = []
            for last, net in nets:
                for corner in range(last, 3):
                    weight_0, weight_1, weight_2 = corners[corner]
                    following.append(
                        (
                            corner,
                            {
                                (i, j, k): weight_0 * net[(i + 1, j, k)]
                                + weight_1 * net[(i, j + 1, k)]
                                + weight_2 * net[(i, j, k + 1)]
                                for i, j, k in indices
                            },
                        )
                    )
            nets = following
        values = tuple(net[(0, 0, 0)] for _, net in nets)
        scale = controls.denominator * common**degree
    return (
        _ratio_endpoint(min(values), scale, upper=False),
        _ratio_endpoint(max(values), scale, upper=True),
    )


_MAXIMUM_FLOAT = float(np.finfo(np.float64).max)


def _ratio_endpoint(numerator: int, scale: int, /, *, upper: bool) -> float:
    """``coefficient_endpoint(Fraction(numerator, scale), saturate=True)`` for ``scale > 0``.

    Integer true division rounds to nearest, so one exact comparison of the
    rounded quotient with the ratio fixes the directed endpoint.
    """
    try:
        rounded = numerator / scale
    except OverflowError:
        if numerator > 0:
            return math.inf if upper else _MAXIMUM_FLOAT
        return -_MAXIMUM_FLOAT if upper else -math.inf
    rounded_numerator, rounded_denominator = rounded.as_integer_ratio()
    excess = rounded_numerator * scale - numerator * rounded_denominator
    if upper:
        return rounded if excess >= 0 else math.nextafter(rounded, math.inf)
    return rounded if excess <= 0 else math.nextafter(rounded, -math.inf)


def _triangle_arguments(
    triangle: TriangleReference,
) -> tuple[Polynomial, Polynomial]:
    first, second, third = triangle
    return (
        {
            (0, 0): first[0],
            (1, 0): second[0] - first[0],
            (0, 1): third[0] - first[0],
        },
        {
            (0, 0): first[1],
            (1, 0): second[1] - first[1],
            (0, 1): third[1] - first[1],
        },
    )


@dataclass(frozen=True, slots=True)
class _EnclosedPolynomial:
    """A face polynomial on the standard triangle as rational centre plus error.

    Rational coefficients stay exact. Each quadratic-radical coefficient is
    enclosed once by an outward-rounded binary64 (dyadic) interval and replaced
    by its dyadic midpoint, with the radius kept per monomial. Every monomial
    ``u^a v^b`` lies in ``[0, 1]`` on the standard triangle and its
    subtriangles, so the polynomial is within ``error`` of its centre there and
    interval proofs compose only rational coefficients.
    """

    center: Polynomial
    radii: dict[tuple[int, ...], float]
    error: float
    # Exact Bernstein controls of `center`, built on the first enclosure.
    controls: list[_SimplexControls | None] = field(
        default_factory=list, compare=False, repr=False
    )


def _coefficient_error(radii: dict[tuple[int, ...], float], /) -> float:
    if not radii:
        return 0.0
    return math.nextafter(math.fsum(radii.values()), math.inf)


def _enclose_polynomial(expression: Polynomial, /) -> _EnclosedPolynomial:
    center: Polynomial = {}
    radii: dict[tuple[int, ...], float] = {}
    for exponent, coefficient in expression.items():
        if isinstance(coefficient, Fraction):
            center[exponent] = coefficient
            continue
        lower, upper = _exact_interval(coefficient)
        middle = 0.5 * lower + 0.5 * upper
        if not (
            math.isfinite(lower) and math.isfinite(upper) and lower <= middle <= upper
        ):
            raise ValueError(
                "Implicit target map coefficients require finite binary64 enclosures."
            )
        center[exponent] = Fraction.from_float(middle)
        radius = max(middle - lower, upper - middle)
        if radius > 0.0:
            radii[exponent] = math.nextafter(radius, math.inf)
    return _EnclosedPolynomial(center, radii, _coefficient_error(radii))


def _enclosed_derivative(value: _EnclosedPolynomial, axis: int, /) -> _EnclosedPolynomial:
    radii: dict[tuple[int, ...], float] = {}
    for exponent, radius in value.radii.items():
        if exponent[axis]:
            lowered = (*exponent[:axis], exponent[axis] - 1, *exponent[axis + 1 :])
            radii[lowered] = math.nextafter(radius * exponent[axis], math.inf)
    return _EnclosedPolynomial(
        derivative(value.center, axis), radii, _coefficient_error(radii)
    )


def _widen(value: Interval, error: float, /) -> Interval:
    if error == 0.0:
        return value
    return (
        math.nextafter(value[0] - error, -math.inf),
        math.nextafter(value[1] + error, math.inf),
    )


def _enclosed_interval(
    value: _EnclosedPolynomial, triangle: TriangleReference, /
) -> Interval:
    """Enclose an enclosed face polynomial over a subtriangle of the standard triangle."""
    if not value.controls:
        value.controls.append(_simplex_controls(value.center))
    controls = value.controls[0]
    if controls is None:
        return _widen(_poly_interval(value.center, triangle), value.error)
    return _widen(_restricted_bounds(controls, triangle), value.error)


@dataclass(frozen=True, slots=True)
class _FaceEnclosure:
    """Enclosed face coordinates and their first and second derivatives, built once per face."""

    values: tuple[_EnclosedPolynomial, _EnclosedPolynomial, _EnclosedPolynomial]
    tangents: tuple[
        tuple[_EnclosedPolynomial, _EnclosedPolynomial],
        tuple[_EnclosedPolynomial, _EnclosedPolynomial],
        tuple[_EnclosedPolynomial, _EnclosedPolynomial],
    ]
    # Per coordinate: (P_uu, P_uv, P_vv).
    curvatures: tuple[
        tuple[_EnclosedPolynomial, _EnclosedPolynomial, _EnclosedPolynomial],
        tuple[_EnclosedPolynomial, _EnclosedPolynomial, _EnclosedPolynomial],
        tuple[_EnclosedPolynomial, _EnclosedPolynomial, _EnclosedPolynomial],
    ]
    # Reference directions (columns) whose physical tangents are orthonormal at
    # the face centroid, kept only for ill-conditioned (needle) faces.
    conditioner: FractionMatrix2 | None
    # Per coordinate, the conditioned tangent columns ``[P_u, P_v] K e_j`` with
    # their outward-rounded radical-coefficient error; empty without ``K``.
    conditioned_tangents: tuple[tuple[_EnclosedPolynomial, _EnclosedPolynomial], ...]


# Faces whose centroid tangent condition number exceeds this get conditioned
# tangents for their chart Jacobian.
_CONDITIONING_THRESHOLD = 8.0


def _tangent_conditioner(
    tangents: tuple[
        tuple[_EnclosedPolynomial, _EnclosedPolynomial],
        tuple[_EnclosedPolynomial, _EnclosedPolynomial],
        tuple[_EnclosedPolynomial, _EnclosedPolynomial],
    ],
    /,
) -> FractionMatrix2 | None:
    """``R^-1`` of the QR factorization of the centroid tangents, if ill-conditioned.

    Along its columns ``d`` the physical tangents ``DP d`` are orthonormal at
    the centroid, so a needle face's nearly parallel tangent pair becomes a
    well-conditioned pair whose cross product does not cancel.
    """
    centroid = (Fraction(1, 3), Fraction(1, 3))
    matrix = np.asarray(
        [
            [
                float(evaluate_expression(value.center, centroid))
                if value.center
                else 0.0
                for value in row
            ]
            for row in tangents
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(matrix)):
        return None
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[1] <= 0.0 or singular[0] <= _CONDITIONING_THRESHOLD * singular[1]:
        return None
    _, upper = np.linalg.qr(matrix)
    if upper[0, 0] == 0.0 or upper[1, 1] == 0.0:
        return None
    inverse = np.linalg.inv(upper)
    if not np.all(np.isfinite(inverse)):
        return None
    return (
        (Fraction(float(inverse[0, 0])), Fraction(float(inverse[0, 1]))),
        (Fraction(float(inverse[1, 0])), Fraction(float(inverse[1, 1]))),
    )


def _enclose_face(expressions: Polynomial3, /) -> _FaceEnclosure:
    first, second, third = (_enclose_polynomial(value) for value in expressions)

    def tangent(
        value: _EnclosedPolynomial, /
    ) -> tuple[_EnclosedPolynomial, _EnclosedPolynomial]:
        return _enclosed_derivative(value, 0), _enclosed_derivative(value, 1)

    def curvature(
        value: _EnclosedPolynomial, /
    ) -> tuple[_EnclosedPolynomial, _EnclosedPolynomial, _EnclosedPolynomial]:
        along_u, along_v = tangent(value)
        return (
            _enclosed_derivative(along_u, 0),
            _enclosed_derivative(along_u, 1),
            _enclosed_derivative(along_v, 1),
        )

    tangents = (tangent(first), tangent(second), tangent(third))
    conditioner = _tangent_conditioner(tangents)
    conditioned: list[tuple[_EnclosedPolynomial, _EnclosedPolynomial]] = []
    if conditioner is not None:
        zero = Fraction(0)
        for tangent_u, tangent_v in tangents:
            columns: list[_EnclosedPolynomial] = []
            for column in range(2):
                first_weight = conditioner[0][column]
                second_weight = conditioner[1][column]
                # The weights are binary64 values, so these outward-rounded
                # products bound the radical-coefficient error exactly.
                error = _interval_add(
                    _interval_scale(
                        (tangent_u.error, tangent_u.error), abs(float(first_weight))
                    ),
                    _interval_scale(
                        (tangent_v.error, tangent_v.error), abs(float(second_weight))
                    ),
                )[1]
                columns.append(
                    _EnclosedPolynomial(
                        {
                            exponent: tangent_u.center.get(exponent, zero) * first_weight
                            + tangent_v.center.get(exponent, zero) * second_weight
                            for exponent in set(tangent_u.center) | set(tangent_v.center)
                        },
                        {},
                        error,
                    )
                )
            conditioned.append((columns[0], columns[1]))
    return _FaceEnclosure(
        (first, second, third),
        tangents,
        (curvature(first), curvature(second), curvature(third)),
        conditioner,
        tuple(conditioned),
    )


def _enclosed_box(
    face: _FaceEnclosure, triangle: TriangleReference, /
) -> tuple[np.ndarray, np.ndarray]:
    intervals = tuple(_enclosed_interval(value, triangle) for value in face.values)
    return (
        np.asarray([value[0] for value in intervals], dtype=np.float64),
        np.asarray([value[1] for value in intervals], dtype=np.float64),
    )


def _restricted_face(
    face: _FaceEnclosure, triangle: TriangleReference, /
) -> tuple[
    tuple[np.ndarray, np.ndarray], IntervalDerivative3x2, IntervalDerivative3x2 | None
]:
    """Enclose each face coordinate and its gradient over a piece.

    The restriction ``R = C(a + E s)`` to the piece corners ``a, b, c`` has
    ``[C_u, C_v](a + E s) = [R_s, R_t] E^-1`` exactly, so the face's own
    derivative polynomials restricted to the piece are the piece derivatives.
    A face conditioner ``K`` also yields ``[C_u, C_v] K``, each column
    enclosed from its own exact polynomial.
    """
    a, b, c = triangle
    if (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]) == 0:
        raise ValueError("Implicit chart pieces must be nondegenerate triangles.")
    lower: list[float] = []
    upper: list[float] = []
    derivatives: list[IntervalVector2] = []
    for value, (tangent_u, tangent_v) in zip(face.values, face.tangents, strict=True):
        low, high = _enclosed_interval(value, triangle)
        lower.append(low)
        upper.append(high)
        derivatives.append(
            (
                _enclosed_interval(tangent_u, triangle),
                _enclosed_interval(tangent_v, triangle),
            )
        )
    conditioned = tuple(
        (_enclosed_interval(first, triangle), _enclosed_interval(second, triangle))
        for first, second in face.conditioned_tangents
    )
    return (
        (np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
        (derivatives[0], derivatives[1], derivatives[2]),
        (conditioned[0], conditioned[1], conditioned[2]) if conditioned else None,
    )


def _evaluate_exact_polynomial(
    expression: Polynomial, point: tuple[Fraction, ...], /
) -> ExactScalar:
    value = evaluate_expression(expression, point)
    if isinstance(value, RadicalElement) and (
        not value.terms or (len(value.terms) == 1 and value.terms[0][0] == 1)
    ):
        return rational_coefficient(value, role="Implicit target map value")
    if not isinstance(value, (Fraction, RadicalElement)):
        raise TypeError(
            "Certified implicit target maps require exact algebraic coordinate values."
        )
    return value


def _prepare_barycentric_solver(
    corners: Sequence[Sequence[Fraction]], /
) -> _BarycentricSolver:
    if len(corners) != 3:
        raise ValueError("Implicit source charts require exactly three corners.")
    dimension = len(corners[0])
    if dimension not in (2, 3) or any(len(point) != dimension for point in corners):
        raise ValueError("Implicit source chart corners have an unsupported dimension.")
    first = tuple(corners[1][axis] - corners[0][axis] for axis in range(dimension))
    second = tuple(corners[2][axis] - corners[0][axis] for axis in range(dimension))
    for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
        if axis_b >= dimension:
            continue
        matrix = (
            (first[axis_a], second[axis_a]),
            (first[axis_b], second[axis_b]),
        )
        try:
            return (
                (axis_a, axis_b),
                factor_exact_rational_algebraic(matrix, RadicalElement),
            )
        except ValueError:
            continue
    raise ValueError("Implicit source chart target reference face is degenerate.")


def _barycentric(
    point: ExactReferencePoint,
    corners: ReferenceTriangle,
    /,
    *,
    solver: _BarycentricSolver | None = None,
) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
    if len(point) != len(corners[0]):
        raise ValueError(
            "Implicit source chart point has a different reference dimension."
        )
    if solver is None:
        solver = _prepare_barycentric_solver(corners)
    axes, factorization = solver
    right_hand_side = (
        (point[axes[0]] - corners[0][axes[0]],),
        (point[axes[1]] - corners[0][axes[1]],),
    )
    solved = factorization.solve(right_hand_side)
    alpha, beta = solved[0][0], solved[1][0]
    if any(
        corners[0][axis]
        + alpha * (corners[1][axis] - corners[0][axis])
        + beta * (corners[2][axis] - corners[0][axis])
        != point[axis]
        for axis in range(len(point))
    ):
        raise ValueError(
            "target reference point is not on its exact witnessed face plane."
        )
    return Fraction(1) - alpha - beta, alpha, beta


def _split_triangle(triangle: TriangleReference, /) -> SplitTriangles | None:
    a, b, c = triangle
    ab: TrianglePoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    bc: TrianglePoint = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
    ca: TrianglePoint = ((c[0] + a[0]) / 2, (c[1] + a[1]) / 2)
    midpoints = ((a, b, ab), (b, c, bc), (c, a, ca))
    for first, second, middle in midpoints:
        represented_first = (float(first[0]), float(first[1]))
        represented_second = (float(second[0]), float(second[1]))
        represented_middle = (float(middle[0]), float(middle[1]))
        if represented_middle in (represented_first, represented_second):
            return None
    return ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))


def _field_gradient_function(
    geometry: CompiledGeometry,
) -> Callable[[jax.Array], jax.Array]:
    kernel, state = geometry.kernel, geometry.state

    def field(point: jax.Array) -> jax.Array:
        return kernel.boundary_field(state, point[None, :])[0]

    return jax.grad(field)


def _prepare_hessian(geometry: CompiledGeometry) -> PreparedIntervalFunction:
    kernel, state = geometry.kernel, geometry.state

    def hessian(point: jax.Array) -> jax.Array:
        def field(point: jax.Array) -> jax.Array:
            return kernel.boundary_field(state, point[None, :])[0]

        return jax.hessian(field)(point)

    return prepare_interval_function(hessian, 3)


def _normal_angle_bounds(
    tangent_u: IntervalVector3,
    tangent_v: IntervalVector3,
    source_gradient: IntervalVector3,
    orientation: int,
) -> Interval:
    cross = (
        _interval_subtract(
            _interval_multiply(tangent_u[1], tangent_v[2]),
            _interval_multiply(tangent_u[2], tangent_v[1]),
        ),
        _interval_subtract(
            _interval_multiply(tangent_u[2], tangent_v[0]),
            _interval_multiply(tangent_u[0], tangent_v[2]),
        ),
        _interval_subtract(
            _interval_multiply(tangent_u[0], tangent_v[1]),
            _interval_multiply(tangent_u[1], tangent_v[0]),
        ),
    )
    oriented_gradient = tuple(
        _interval_scale(value, float(orientation)) for value in source_gradient
    )
    dot = (0.0, 0.0)
    for first, second in zip(cross, oriented_gradient, strict=True):
        dot = _interval_add(dot, _interval_multiply(first, second))
    target_norm = _interval_vector_norm_bounds(cross)
    source_norm = _interval_vector_norm_bounds(oriented_gradient)
    if target_norm[0] <= 0.0 or source_norm[0] <= 0.0:
        return 0.0, math.pi
    denominator = _interval_multiply(target_norm, source_norm)
    if denominator[0] <= 0.0:
        return 0.0, math.pi
    cosines = tuple(numerator / divisor for numerator in dot for divisor in denominator)
    cosine_lower = math.nextafter(min(cosines), -math.inf)
    cosine_upper = math.nextafter(max(cosines), math.inf)
    cosine_lower = min(1.0, max(-1.0, cosine_lower))
    cosine_upper = min(1.0, max(-1.0, cosine_upper))
    return (
        math.nextafter(math.acos(cosine_upper), 0.0),
        math.nextafter(math.acos(cosine_lower), math.inf),
    )


def _chart_derivative_intervals(
    target_derivatives: IntervalDerivative3x2,
    gradient_p: IntervalVector3,
    gradient_q: IntervalVector3,
    hessian_p: IntervalMatrix3,
    root_parameters: Interval,
) -> IntervalDerivative3x2 | None:
    identity: IntervalMatrix3 = (
        ((1.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), (1.0, 1.0), (0.0, 0.0)),
        ((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)),
    )
    matrix_b: list[list[Interval]] = []
    t_interval: Interval = root_parameters
    for row in range(3):
        matrix_row: list[Interval] = []
        for column in range(3):
            entry = _interval_multiply(hessian_p[row][column], t_interval)
            matrix_row.append(_interval_add(identity[row][column], entry))
        matrix_b.append(matrix_row)
    b: IntervalMatrix3 = (
        (matrix_b[0][0], matrix_b[0][1], matrix_b[0][2]),
        (matrix_b[1][0], matrix_b[1][1], matrix_b[1][2]),
        (matrix_b[2][0], matrix_b[2][1], matrix_b[2][2]),
    )
    b_transpose_gradient: IntervalVector3 = (
        sum_interval(
            (
                _interval_multiply(b[0][0], gradient_q[0]),
                _interval_multiply(b[1][0], gradient_q[1]),
                _interval_multiply(b[2][0], gradient_q[2]),
            )
        ),
        sum_interval(
            (
                _interval_multiply(b[0][1], gradient_q[0]),
                _interval_multiply(b[1][1], gradient_q[1]),
                _interval_multiply(b[2][1], gradient_q[2]),
            )
        ),
        sum_interval(
            (
                _interval_multiply(b[0][2], gradient_q[0]),
                _interval_multiply(b[1][2], gradient_q[1]),
                _interval_multiply(b[2][2], gradient_q[2]),
            )
        ),
    )
    denominator = sum_interval(
        (
            _interval_multiply(gradient_q[0], gradient_p[0]),
            _interval_multiply(gradient_q[1], gradient_p[1]),
            _interval_multiply(gradient_q[2], gradient_p[2]),
        )
    )
    if denominator[0] <= 0.0:
        return None
    dt_values: list[Interval] = []
    for column in range(2):
        numerator = sum_interval(
            (
                _interval_multiply(
                    b_transpose_gradient[0], target_derivatives[0][column]
                ),
                _interval_multiply(
                    b_transpose_gradient[1], target_derivatives[1][column]
                ),
                _interval_multiply(
                    b_transpose_gradient[2], target_derivatives[2][column]
                ),
            )
        )
        quotient = _interval_divide(numerator, denominator)
        dt_values.append((-quotient[1], -quotient[0]))
    dt: IntervalVector2 = (dt_values[0], dt_values[1])

    def derivative_row(row: int, /) -> IntervalVector2:
        result: list[Interval] = []
        for column in range(2):
            base = sum_interval(
                (
                    _interval_multiply(b[row][0], target_derivatives[0][column]),
                    _interval_multiply(b[row][1], target_derivatives[1][column]),
                    _interval_multiply(b[row][2], target_derivatives[2][column]),
                )
            )
            correction = _interval_multiply(gradient_p[row], dt[column])
            result.append(_interval_add(base, correction))
        return result[0], result[1]

    return (derivative_row(0), derivative_row(1), derivative_row(2))


def sum_interval(values: tuple[Interval, ...], /) -> Interval:
    total = (0.0, 0.0)
    for value in values:
        total = _interval_add(total, value)
    return total


def _preconditioned_symmetric_part(
    jacobian: IntervalMatrix2, preconditioner: FractionMatrix2, /
) -> tuple[Interval, Interval]:
    """Enclose ``S_11`` and ``det S`` for ``S = sym(C J)`` over a Jacobian box."""
    rows: list[list[Interval]] = []
    for row in range(2):
        values: list[Interval] = []
        for column in range(2):
            total: Interval = (0.0, 0.0)
            for axis in range(2):
                total = _interval_add(
                    total,
                    _interval_multiply(
                        _exact_interval(preconditioner[row][axis]),
                        jacobian[axis][column],
                    ),
                )
            values.append(total)
        rows.append(values)
    symmetric_11 = rows[0][0]
    symmetric_22 = rows[1][1]
    symmetric_12 = _interval_scale(_interval_add(rows[0][1], rows[1][0]), 0.5)
    determinant = _interval_subtract(
        _interval_multiply(symmetric_11, symmetric_22),
        _interval_multiply(symmetric_12, symmetric_12),
    )
    return symmetric_11, determinant


def _projected_injectivity_margin(
    jacobian: IntervalMatrix2, preconditioner: FractionMatrix2, /
) -> tuple[float, bool]:
    """Return the certified margin of ``sym(C J)`` and whether it is refuted.

    The margin is positive when every matrix of the box has a positive-definite
    preconditioned symmetric part; the box is refuted when none has one.
    """
    symmetric_11, determinant = _preconditioned_symmetric_part(jacobian, preconditioner)
    if symmetric_11[0] > 0.0 and determinant[0] > 0.0:
        return min(symmetric_11[0], determinant[0]), False
    return 0.0, symmetric_11[1] <= 0.0 or determinant[1] <= 0.0


def _fraction_barycentric_area(points: TriangleReference, /) -> Fraction:
    return (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) - (
        points[1][1] - points[0][1]
    ) * (points[2][0] - points[0][0])


def _point_interval(value: Interval, /) -> float:
    return 0.0 if value[0] <= 0.0 <= value[1] else min(abs(value[0]), abs(value[1]))


def _gradient_norm_interval(lower: np.ndarray, upper: np.ndarray, /) -> Interval:
    return _interval_vector_norm_bounds(
        tuple((float(lower[axis]), float(upper[axis])) for axis in range(3))
    )


def _triangle_coordinate_expressions(
    cell_expressions: Polynomial3,
    corners: ReferenceTriangle,
) -> Polynomial3:
    arguments: tuple[Polynomial, ...] = tuple(
        {
            (0, 0): corners[0][axis],
            (1, 0): corners[1][axis] - corners[0][axis],
            (0, 1): corners[2][axis] - corners[0][axis],
        }
        for axis in range(len(corners[0]))
    )
    first = compose_expression(cell_expressions[0], arguments)
    second = compose_expression(cell_expressions[1], arguments)
    third = compose_expression(cell_expressions[2], arguments)
    if (
        not isinstance(first, dict)
        or not isinstance(second, dict)
        or not isinstance(third, dict)
    ):
        raise TypeError(
            "Adaptive implicit target maps require three polynomial coordinates."
        )
    return first, second, third


def _source_authority(
    source: AdaptiveImplicitSourceQuery,
) -> tuple[AdaptiveImplicitBoundarySource, str]:
    if isinstance(source, AdaptiveImplicitBoundarySource):
        adaptive = source
        frame_id = source.frame_id
    else:
        source.require_current()
        adaptive = source.adaptive_source
        frame_id = source.frame_id
    adaptive.require_current()
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError(
            "Adaptive implicit root charts require a bound source coordinate frame."
        )
    return adaptive, canonical_identifier(frame_id, "frame_id")


def _source_authoring_profile(
    source: AdaptiveImplicitSourceQuery, /
) -> AnalyticImplicitProfile | None:
    if isinstance(source, AdaptiveImplicitBoundarySource):
        return None
    profile = source.authoring_profile
    if profile is not None and not isinstance(profile, AnalyticImplicitProfile):
        raise TypeError(
            "Adaptive implicit boundary authority must retain an AnalyticImplicitProfile or None."
        )
    return profile


def _profile_covers_singularities_away_from_zero_set(
    source: AdaptiveImplicitSourceQuery,
    surface: AdaptiveImplicitSurface,
    frame_id: str,
    /,
) -> bool:
    profile = _source_authoring_profile(source)
    if profile is None:
        return False
    if (
        profile.source_id != source.source_id
        or profile.source_revision != source.source_revision
        or profile.coordinate_contract.spatial_id != frame_id
    ):
        raise ValueError(
            "Implicit authoring profile must bind the exact boundary source and frame."
        )
    profile.require_bound(surface.geometry, profile.coordinate_contract)
    certificate = surface.geometry.field_certificate
    cover = surface.cover
    if (
        certificate.validity_region != "all_space"
        or cover is None
        or not cover.complete
        or not cover.certified
        or cover.enclosure != "interval"
        or not surface.volume.bounds.gradient_rigorous
    ):
        return False
    possible_zero_boxes = cover.boxes[cover.possible_zero_mask]
    return bool(possible_zero_boxes.shape[0]) and not any(
        profile._field_singular_set_intersects_box(box) for box in possible_zero_boxes
    )


@dataclass(frozen=True, slots=True)
class _BoxRequests:
    """Field and Hessian boxes one piece proof needs before its next step."""

    field: tuple[tuple[np.ndarray, np.ndarray], ...]
    hessian: tuple[tuple[np.ndarray, np.ndarray], ...] = ()


type _FieldBox = tuple[Interval, IntervalVector3]
type _BoxResults = tuple[tuple[_FieldBox, ...], tuple[IntervalMatrix3, ...]]
type _PieceSteps = Generator[_BoxRequests, _BoxResults, _PieceOutcome]


@dataclass(frozen=True, slots=True)
class _FaceRound:
    """Piece proofs and standalone box queries one face needs before its next step."""

    pieces: tuple[_PieceSteps, ...] = ()
    queries: tuple[_BoxRequests, ...] = ()


type _FaceReply = tuple[list[_PieceOutcome], list[_BoxResults]]
type _FaceSteps[T] = Generator[_FaceRound, _FaceReply, T]


def _interval_vector3(lower: np.ndarray, upper: np.ndarray, /) -> IntervalVector3:
    return (
        (float(lower[0]), float(upper[0])),
        (float(lower[1]), float(upper[1])),
        (float(lower[2]), float(upper[2])),
    )


def _evaluate_box_requests(
    bounds: _FieldEnclosures,
    hessian_program: PreparedIntervalFunction,
    requests: Sequence[_BoxRequests],
    /,
) -> list[_BoxResults]:
    """Enclose every requested box of all pieces in one batched evaluation."""
    field_boxes = [box for request in requests for box in request.field]
    hessian_boxes = [box for request in requests for box in request.hessian]
    field_values: list[_FieldBox] = []
    if field_boxes:
        result = bounds.boxes(
            np.stack([low for low, _ in field_boxes]),
            np.stack([high for _, high in field_boxes]),
        )
        value_lower = np.asarray(result.value_lower, dtype=np.float64).reshape((-1,))
        value_upper = np.asarray(result.value_upper, dtype=np.float64).reshape((-1,))
        gradient_lower = np.asarray(result.gradient_lower, dtype=np.float64).reshape(
            (-1, 3)
        )
        gradient_upper = np.asarray(result.gradient_upper, dtype=np.float64).reshape(
            (-1, 3)
        )
        field_values = [
            (
                (float(value_lower[row]), float(value_upper[row])),
                _interval_vector3(gradient_lower[row], gradient_upper[row]),
            )
            for row in range(len(field_boxes))
        ]
    hessian_values: list[IntervalMatrix3] = []
    if hessian_boxes:
        lower, upper = hessian_program.evaluate(
            np.stack([low for low, _ in hessian_boxes]),
            np.stack([high for _, high in hessian_boxes]),
        )
        hessian_lower = np.asarray(lower, dtype=np.float64).reshape((-1, 3, 3))
        hessian_upper = np.asarray(upper, dtype=np.float64).reshape((-1, 3, 3))
        hessian_values = [
            (
                _interval_vector3(hessian_lower[row, 0], hessian_upper[row, 0]),
                _interval_vector3(hessian_lower[row, 1], hessian_upper[row, 1]),
                _interval_vector3(hessian_lower[row, 2], hessian_upper[row, 2]),
            )
            for row in range(len(hessian_boxes))
        ]
    results: list[_BoxResults] = []
    field_start = hessian_start = 0
    for request in requests:
        field_stop = field_start + len(request.field)
        hessian_stop = hessian_start + len(request.hessian)
        results.append(
            (
                tuple(field_values[field_start:field_stop]),
                tuple(hessian_values[hessian_start:hessian_stop]),
            )
        )
        field_start, hessian_start = field_stop, hessian_stop
    return results


def _run_piece_steps(
    bounds: _FieldEnclosures,
    hessian_program: PreparedIntervalFunction,
    steps: Sequence[_PieceSteps],
    queries: Sequence[_BoxRequests] = (),
    /,
) -> tuple[list[_PieceOutcome], list[_BoxResults]]:
    """Advance all piece proofs together, batching each round of box queries.

    Standalone ``queries`` join the first round's evaluation.
    """
    outcomes: list[_PieceOutcome | None] = [None] * len(steps)
    pending: dict[int, _BoxRequests] = {}
    for index, step in enumerate(steps):
        try:
            pending[index] = next(step)
        except StopIteration as stop:
            outcomes[index] = stop.value
    query_results: list[_BoxResults] = []
    while pending or len(query_results) < len(queries):
        indices = tuple(pending)
        extra = list(queries) if len(query_results) < len(queries) else []
        results = _evaluate_box_requests(
            bounds, hessian_program, [pending[index] for index in indices] + extra
        )
        query_results.extend(results[len(indices) :])
        pending = {}
        for index, result in zip(indices, results[: len(indices)], strict=True):
            try:
                pending[index] = steps[index].send(result)
            except StopIteration as stop:
                outcomes[index] = stop.value
    finished: list[_PieceOutcome] = []
    for outcome in outcomes:
        if outcome is None:
            raise RuntimeError("An implicit root-chart piece proof did not finish.")
        finished.append(outcome)
    return finished, query_results


def _run_faces[T](
    bounds: _FieldEnclosures,
    hessian_program: PreparedIntervalFunction,
    faces: Sequence[_FaceSteps[T]],
    /,
) -> list[T]:
    """Advance every face proof together, one batched round of piece proofs at a time.

    Each face yields its next round independently of the others; every round
    evaluates the field and Hessian boxes of all faces' pending pieces at once,
    which leaves each face's proof unchanged.
    """
    results: list[T | None] = [None] * len(faces)
    active: dict[int, _FaceRound] = {}
    for index, face in enumerate(faces):
        try:
            active[index] = next(face)
        except StopIteration as stop:
            results[index] = stop.value
    while active:
        order = tuple(active)
        outcomes, query_results = _run_piece_steps(
            bounds,
            hessian_program,
            [step for index in order for step in active[index].pieces],
            [query for index in order for query in active[index].queries],
        )
        following: dict[int, _FaceRound] = {}
        piece_start = query_start = 0
        for index in order:
            round_ = active[index]
            piece_stop = piece_start + len(round_.pieces)
            query_stop = query_start + len(round_.queries)
            reply: _FaceReply = (
                outcomes[piece_start:piece_stop],
                query_results[query_start:query_stop],
            )
            piece_start, query_start = piece_stop, query_stop
            try:
                following[index] = faces[index].send(reply)
            except StopIteration as stop:
                results[index] = stop.value
        active = following
    finished: list[T] = []
    for result in results:
        if result is None:
            raise RuntimeError("An implicit root-chart face proof did not finish.")
        finished.append(result)
    return finished


def _intersect_intervals(first: Interval, second: Interval, /) -> Interval:
    """Intersect two enclosures of one range; disjoint input keeps the first."""
    lower, upper = max(first[0], second[0]), min(first[1], second[1])
    return (lower, upper) if lower <= upper else first


def _mean_value_interval(
    center: Interval, slopes: IntervalVector2, offsets: IntervalVector2, /
) -> Interval:
    return sum_interval(
        (
            center,
            _interval_multiply(slopes[0], offsets[0]),
            _interval_multiply(slopes[1], offsets[1]),
        )
    )


def _chain_slopes(
    row: IntervalVector3, tangents: IntervalDerivative3x2, /
) -> IntervalVector2:
    """Enclose ``row · ∂P/∂u_j`` for both target parameters."""
    return (
        sum_interval(
            tuple(_interval_multiply(row[axis], tangents[axis][0]) for axis in range(3))
        ),
        sum_interval(
            tuple(_interval_multiply(row[axis], tangents[axis][1]) for axis in range(3))
        ),
    )


def _shifted_tangents(
    tangents: IntervalDerivative3x2, hessian: IntervalMatrix3, scale: Interval, /
) -> IntervalDerivative3x2:
    """Enclose ``∂/∂u (P + t ∇f(P)) = (I + t H) ∂P/∂u`` for ``t`` in ``scale``."""

    def row(axis: int, /) -> IntervalVector2:
        curvature = _chain_slopes(hessian[axis], tangents)
        return (
            _interval_add(tangents[axis][0], _interval_multiply(scale, curvature[0])),
            _interval_add(tangents[axis][1], _interval_multiply(scale, curvature[1])),
        )

    return row(0), row(1), row(2)


def _piece_center(
    face: _FaceEnclosure, triangle: TriangleReference, /
) -> tuple[np.ndarray, np.ndarray, IntervalVector2, IntervalDerivative3x2]:
    """Enclose the target point and tangents at the piece centroid, and the offsets."""
    center: TrianglePoint = (
        (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3,
        (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3,
    )

    def at_center(value: _EnclosedPolynomial, /) -> Interval:
        if not value.center:
            return _widen((0.0, 0.0), value.error)
        return _widen(
            _exact_interval(_evaluate_exact_polynomial(value.center, center)),
            value.error,
        )

    coordinates = tuple(at_center(value) for value in face.values)
    tangents: IntervalDerivative3x2 = (
        (at_center(face.tangents[0][0]), at_center(face.tangents[0][1])),
        (at_center(face.tangents[1][0]), at_center(face.tangents[1][1])),
        (at_center(face.tangents[2][0]), at_center(face.tangents[2][1])),
    )
    offsets: IntervalVector2 = (
        (
            _exact_interval(min(point[0] for point in triangle) - center[0])[0],
            _exact_interval(max(point[0] for point in triangle) - center[0])[1],
        ),
        (
            _exact_interval(min(point[1] for point in triangle) - center[1])[0],
            _exact_interval(max(point[1] for point in triangle) - center[1])[1],
        ),
    )
    return (
        np.asarray([value[0] for value in coordinates], dtype=np.float64),
        np.asarray([value[1] for value in coordinates], dtype=np.float64),
        offsets,
        tangents,
    )


def _piece_curvatures(
    face: _FaceEnclosure, triangle: TriangleReference, /
) -> tuple[IntervalVector3, IntervalVector3, IntervalVector3]:
    """Enclose ``(P_uu, P_uv, P_vv)`` per coordinate over a piece; zero for P1 faces."""

    def enclose(value: _EnclosedPolynomial, /) -> Interval:
        if not value.center and value.error == 0.0:
            return (0.0, 0.0)
        return _enclosed_interval(value, triangle)

    return (
        (
            enclose(face.curvatures[0][0]),
            enclose(face.curvatures[0][1]),
            enclose(face.curvatures[0][2]),
        ),
        (
            enclose(face.curvatures[1][0]),
            enclose(face.curvatures[1][1]),
            enclose(face.curvatures[1][2]),
        ),
        (
            enclose(face.curvatures[2][0]),
            enclose(face.curvatures[2][1]),
            enclose(face.curvatures[2][2]),
        ),
    )


def _taylor_value_interval(
    center_value: Interval,
    center_gradient: IntervalVector3,
    center_tangents: IntervalDerivative3x2,
    gradient: IntervalVector3,
    hessian: IntervalMatrix3,
    tangents: IntervalDerivative3x2,
    curvatures: tuple[IntervalVector3, IntervalVector3, IntervalVector3],
    offsets: IntervalVector2,
    /,
) -> Interval:
    """Second-order Taylor form of ``g(u) = f(P(u))`` about the piece centre ``c``.

    ``g(c + d) = g(c) + Dg(c) d + d^T D^2 g(xi) d / 2`` with ``xi`` in the piece and
    ``D^2 g_jk = P_j^T H P_k + grad f . P_jk`` enclosed over the piece. The
    first-order term uses the narrow centre gradient, so a face whose gradient
    turns across the piece is not charged that turn times the full offset.
    """
    linear = _mean_value_interval(
        center_value, _chain_slopes(center_gradient, center_tangents), offsets
    )

    # H P_j per Hessian row: hessian_tangents[row][j] encloses (H P_j)_row.
    hessian_tangents = tuple(_chain_slopes(hessian[row], tangents) for row in range(3))

    def second(first: int, second_axis: int, curvature: int, /) -> Interval:
        return sum_interval(
            (
                *(
                    _interval_multiply(
                        tangents[row][first], hessian_tangents[row][second_axis]
                    )
                    for row in range(3)
                ),
                *(
                    _interval_multiply(gradient[row], curvatures[row][curvature])
                    for row in range(3)
                ),
            )
        )

    def square(value: Interval, /) -> Interval:
        magnitude = max(abs(value[0]), abs(value[1]))
        lower = 0.0 if value[0] <= 0.0 <= value[1] else min(value[0] ** 2, value[1] ** 2)
        return (
            math.nextafter(lower, -math.inf) if lower > 0.0 else 0.0,
            math.nextafter(magnitude * magnitude, math.inf),
        )

    quadratic = sum_interval(
        (
            _interval_multiply(second(0, 0, 0), square(offsets[0])),
            _interval_scale(
                _interval_multiply(second(0, 1, 1), _interval_multiply(*offsets)), 2.0
            ),
            _interval_multiply(second(1, 1, 2), square(offsets[1])),
        )
    )
    return _interval_add(linear, _interval_scale(quadratic, 0.5))


def _shifted_center_box(
    center_low: np.ndarray,
    center_high: np.ndarray,
    center_gradient: IntervalVector3,
    shift: float,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    scale: Interval = (shift, shift)
    shifted = tuple(
        _interval_add(
            (float(center_low[axis]), float(center_high[axis])),
            _interval_multiply(scale, center_gradient[axis]),
        )
        for axis in range(3)
    )
    lower = np.asarray([value[0] for value in shifted], dtype=np.float64)
    upper = np.asarray([value[1] for value in shifted], dtype=np.float64)
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return None
    return lower, upper


def _face_piece_proofs(
    surface: AdaptiveImplicitSurface,
    face: _FaceEnclosure,
    triangles: Sequence[TriangleReference],
    *,
    source_orientation: int,
    projected_axes: tuple[int, int, int],
    tolerance: float | None,
    maximum_normal_deviation: float | None,
    root_newton_steps: int = 0,
) -> tuple[_PieceSteps, ...]:
    """Piece proofs of sibling subtriangles, run together by `_run_piece_steps`."""
    return tuple(
        _face_piece_steps(
            surface,
            face,
            triangle,
            source_orientation,
            projected_axes,
            tolerance,
            maximum_normal_deviation,
            root_newton_steps,
        )
        for triangle in triangles
    )


def _face_piece_steps(
    surface: AdaptiveImplicitSurface,
    face: _FaceEnclosure,
    triangle: TriangleReference,
    source_orientation: int,
    projected_axes: tuple[int, int, int],
    tolerance: float | None,
    maximum_normal_deviation: float | None,
    root_newton_steps: int = 0,
) -> _PieceSteps:
    """Prove one subtriangle's scalar root chart and interval differential.

    The proof yields its field and Hessian box queries, so sibling pieces share
    one batched interval evaluation per step. ``root_newton_steps`` further
    narrows the root parameter interval, and so the piece's root box, by
    midpoint interval Newton steps after the proof.
    """
    (p_low, p_high), target_derivatives, conditioned_derivatives = _restricted_face(
        face, triangle
    )
    if np.array_equal(p_low, p_high):
        return (
            None,
            "Target coordinate map has no physical two-dimensional chart extent.",
            False,
        )
    center_low, center_high, offsets, center_tangents = _piece_center(face, triangle)
    fields, hessians = yield _BoxRequests(
        ((p_low, p_high), (center_low, center_high)), ((p_low, p_high),)
    )
    (natural_value, natural_gradient), (center_value, center_gradient) = fields
    hessian_intervals = hessians[0]
    # Mean-value forms about the piece centre c: any smooth g of the target
    # point satisfies g(P(u)) in g(P(c)) + Dg(P box) P_u(piece) (u - c). A
    # natural box extension straddles the zero set across its whole box; these
    # forms shrink with the piece and keep the root tube close to the face.
    gradient_p: IntervalVector3 = (
        _intersect_intervals(
            natural_gradient[0],
            _mean_value_interval(
                center_gradient[0],
                _chain_slopes(hessian_intervals[0], target_derivatives),
                offsets,
            ),
        ),
        _intersect_intervals(
            natural_gradient[1],
            _mean_value_interval(
                center_gradient[1],
                _chain_slopes(hessian_intervals[1], target_derivatives),
                offsets,
            ),
        ),
        _intersect_intervals(
            natural_gradient[2],
            _mean_value_interval(
                center_gradient[2],
                _chain_slopes(hessian_intervals[2], target_derivatives),
                offsets,
            ),
        ),
    )
    gradient_lower = np.asarray([value[0] for value in gradient_p], dtype=np.float64)
    gradient_upper = np.asarray([value[1] for value in gradient_p], dtype=np.float64)
    gradient_norm = _gradient_norm_interval(gradient_lower, gradient_upper)
    if gradient_norm[0] <= 0.0:
        return None, "target face has no strict source-gradient norm enclosure", True
    value_low, value_high = _intersect_intervals(
        _intersect_intervals(
            natural_value,
            _mean_value_interval(
                center_value, _chain_slopes(gradient_p, target_derivatives), offsets
            ),
        ),
        _taylor_value_interval(
            center_value,
            center_gradient,
            center_tangents,
            gradient_p,
            hessian_intervals,
            target_derivatives,
            _piece_curvatures(face, triangle),
            offsets,
        ),
    )
    residual = (value_low, value_high)
    exact_on_source = value_low == 0.0 and value_high == 0.0
    # The normal-line root parameter interval T contains 0 and the root estimated
    # at the piece centre with a margin for the residual spread across the piece;
    # strict field signs at both ends of T certify it below.
    if exact_on_source:
        root_parameters: Interval = (0.0, 0.0)
    else:
        center_slope = float(
            sum(
                (0.5 * (component[0] + component[1])) ** 2
                for component in center_gradient
            )
        )
        center_parameter = (
            -0.5 * (center_value[0] + center_value[1]) / center_slope
            if center_slope > 0.0
            else math.nan
        )
        radius = 2.0 * (value_high - value_low) / (
            gradient_norm[0] * gradient_norm[0]
        ) + 0.5 * abs(center_parameter)
        lower = min(0.0, center_parameter - radius)
        upper = max(0.0, center_parameter + radius)
        # The root side spans twice its estimate: the face-independent doubling
        # bracket of a point lift then stays inside the certified interval.
        if center_parameter >= 0.0:
            upper *= 2.0
        else:
            lower *= 2.0
        root_parameters = (
            math.nextafter(lower, -math.inf),
            math.nextafter(upper, math.inf),
        )
    if not (math.isfinite(root_parameters[0]) and math.isfinite(root_parameters[1])):
        return None, "implicit root-chart normal tube is not finitely enclosed", True
    base_box = _interval_vector3(p_low, p_high)
    if exact_on_source:
        gradient_q = gradient_p
        directional_derivative = (
            math.nextafter(gradient_norm[0] ** 2, 0.0),
            math.nextafter(gradient_norm[1] ** 2, math.inf),
        )
    else:
        # A bracket that misses a strict endpoint sign widens on the failing
        # side until it is signed, its tube leaves the certified domain, or
        # phi' loses strict positivity on it; sign success then never hinges on
        # how the bracket estimate and the endpoint enclosures scale with the
        # piece. Each widening at least doubles the side, so the tube reaches
        # the domain boundary (or binary64 overflow) after finitely many rounds.
        while True:
            tube = _lifted_box(base_box, root_parameters, gradient_p)
            tube_lower, tube_upper = _box_arrays(tube)
            if (
                not (np.all(np.isfinite(tube_lower)) and np.all(np.isfinite(tube_upper)))
                or np.any(tube_lower < surface.volume.domain[0])
                or np.any(tube_upper > surface.volume.domain[1])
            ):
                return (
                    None,
                    "implicit root-chart tube leaves the certified source domain",
                    True,
                )
            endpoint_boxes = [
                _box_arrays(_lifted_box(base_box, (shift, shift), gradient_p))
                for shift in root_parameters
            ]
            shifted_centers = [
                _shifted_center_box(center_low, center_high, center_gradient, shift)
                for shift in root_parameters
            ]
            fields, _ = yield _BoxRequests(
                (
                    (tube_lower, tube_upper),
                    *endpoint_boxes,
                    *(box for box in shifted_centers if box is not None),
                )
            )
            gradient_q = fields[0][1]
            (_, minus_value_upper), minus_gradient = fields[1]
            (plus_value_lower, _), plus_gradient = fields[2]
            center_fields = iter(fields[3:])
            if shifted_centers[0] is not None:
                minus_value_upper = min(
                    minus_value_upper,
                    _mean_value_interval(
                        next(center_fields)[0],
                        _chain_slopes(
                            minus_gradient,
                            _shifted_tangents(
                                target_derivatives,
                                hessian_intervals,
                                (root_parameters[0], root_parameters[0]),
                            ),
                        ),
                        offsets,
                    )[1],
                )
            if shifted_centers[1] is not None:
                plus_value_lower = max(
                    plus_value_lower,
                    _mean_value_interval(
                        next(center_fields)[0],
                        _chain_slopes(
                            plus_gradient,
                            _shifted_tangents(
                                target_derivatives,
                                hessian_intervals,
                                (root_parameters[1], root_parameters[1]),
                            ),
                        ),
                        offsets,
                    )[0],
                )
            directional_derivative = sum_interval(
                (
                    _interval_multiply(gradient_q[0], gradient_p[0]),
                    _interval_multiply(gradient_q[1], gradient_p[1]),
                    _interval_multiply(gradient_q[2], gradient_p[2]),
                )
            )
            if directional_derivative[0] <= 0.0:
                return (
                    None,
                    "normal-line field derivative is not strictly positive",
                    True,
                )
            # phi(t) = f(P + t grad f(P)) has phi' >= D_lo on the tube of T, which
            # contains 0, so phi(T_hi) >= f(P) + T_hi D_lo and phi(T_lo) <=
            # f(P) + T_lo D_lo: the piece residual itself bounds both endpoint signs.
            if root_parameters[1] > 0.0:
                plus_value_lower = max(
                    plus_value_lower,
                    _interval_add(
                        (value_low, value_low),
                        _interval_multiply(
                            (root_parameters[1], root_parameters[1]),
                            (directional_derivative[0], directional_derivative[0]),
                        ),
                    )[0],
                )
            if root_parameters[0] < 0.0:
                minus_value_upper = min(
                    minus_value_upper,
                    _interval_add(
                        (value_high, value_high),
                        _interval_multiply(
                            (root_parameters[0], root_parameters[0]),
                            (directional_derivative[0], directional_derivative[0]),
                        ),
                    )[1],
                )
            minus_signed = minus_value_upper < 0.0
            plus_signed = plus_value_lower > 0.0
            if minus_signed and plus_signed:
                break
            span = max(-root_parameters[0], root_parameters[1])
            lower, upper = root_parameters
            # The monotone bounds above sign an endpoint once |T| D_lo exceeds
            # the residual on its side; jump past that point when it beats
            # doubling (the margin absorbs D_lo shrinking on the wider tube).
            if not minus_signed:
                lower = min(
                    2.0 * lower if lower < 0.0 else -span,
                    -1.25 * value_high / directional_derivative[0],
                )
            if not plus_signed:
                upper = max(
                    2.0 * upper if upper > 0.0 else span,
                    -1.25 * value_low / directional_derivative[0],
                )
            root_parameters = (
                math.nextafter(lower, -math.inf),
                math.nextafter(upper, math.inf),
            )
    maximum_root_parameter = max(-root_parameters[0], root_parameters[1])
    if not exact_on_source:
        # Mean-value (interval Newton) step: 0 = f(P) + t* grad f(xi) . grad f(P)
        # with xi on the certified segment, so t* lies in -f(P box) / D.
        root_parameters = _intersect_intervals(
            root_parameters,
            _interval_divide((-value_high, -value_low), directional_derivative),
        )
    map_derivatives = (
        target_derivatives
        if exact_on_source
        else _chart_derivative_intervals(
            target_derivatives,
            gradient_p,
            gradient_q,
            hessian_intervals,
            root_parameters,
        )
    )
    if map_derivatives is None:
        return (
            None,
            "implicit root-chart differential denominator contains zero",
            True,
        )
    # With R_u = Pi B P_u, where Pi projects along grad f(P) onto grad f(R)^perp,
    # det[R_u, R_v, grad f(R)] = det[B P_u, B P_v, grad f(P)] |grad f(R)|^2 /
    # (grad f(R) . grad f(P)); this avoids the dependency of the projected rows.
    # A needle face uses det[B DP K, grad f(P)] / det K with its conditioner K,
    # whose conditioned tangents do not cancel in the cross product.
    source_norm = _interval_vector_norm_bounds(gradient_q)
    if conditioned_derivatives is None or face.conditioner is None:
        normal_area = _normal_area_interval(
            _shifted_tangents(target_derivatives, hessian_intervals, root_parameters),
            gradient_p,
            source_orientation,
        )
    else:
        (k00, k01), (k10, k11) = face.conditioner
        conditioner_determinant = k00 * k11 - k01 * k10
        normal_area = _interval_multiply(
            _normal_area_interval(
                _shifted_tangents(
                    conditioned_derivatives, hessian_intervals, root_parameters
                ),
                gradient_p,
                source_orientation,
            ),
            _exact_interval(1 / conditioner_determinant),
        )
    cross_dot = _interval_multiply(
        normal_area,
        _interval_divide(
            _interval_multiply(source_norm, source_norm), directional_derivative
        ),
    )
    chart_jacobian = cross_dot[0]
    if cross_dot[0] <= 0.0:
        if cross_dot[1] <= 0.0:
            # Every Jacobian in the enclosure is non-positive, so the actual chart
            # folds against its source orientation on this piece.
            return (
                None,
                "implicit source chart Jacobian is certifiably non-positive in its "
                "source orientation",
                False,
            )
        return (
            None,
            "implicit source chart Jacobian is not strictly positive in its source orientation",
            True,
        )
    # The normal of DP K is det K times the normal of DP, so a conditioned
    # needle face measures its normal from the non-cancelling tangent pair.
    normal_derivatives = target_derivatives
    normal_orientation = source_orientation
    if conditioned_derivatives is not None and face.conditioner is not None:
        (k00, k01), (k10, k11) = face.conditioner
        normal_derivatives = conditioned_derivatives
        if k00 * k11 - k01 * k10 < 0:
            normal_orientation = -source_orientation
    tangent_u: IntervalVector3 = (
        normal_derivatives[0][0],
        normal_derivatives[1][0],
        normal_derivatives[2][0],
    )
    tangent_v: IntervalVector3 = (
        normal_derivatives[0][1],
        normal_derivatives[1][1],
        normal_derivatives[2][1],
    )
    target_normal = _normal_angle_bounds(
        tangent_u, tangent_v, gradient_q, normal_orientation
    )
    lower_distance = 0.0
    f_abs_lower = _point_interval(residual)
    directional_upper = sum(
        max(abs(gradient_q[axis][0]), abs(gradient_q[axis][1]))
        * max(abs(gradient_p[axis][0]), abs(gradient_p[axis][1]))
        for axis in range(3)
    )
    if directional_upper > 0.0:
        lower_distance = math.nextafter(
            f_abs_lower * gradient_norm[0] / directional_upper, 0.0
        )
    upper_distance = math.nextafter(
        max(-root_parameters[0], root_parameters[1]) * gradient_norm[1], math.inf
    )
    # A limit of pi admits every normal; the angle enclosure itself may round
    # past the binary64 pi.
    normal_limit = (
        maximum_normal_deviation
        if maximum_normal_deviation is not None and maximum_normal_deviation < math.pi
        else None
    )
    unresolved_requirement = (tolerance is not None and upper_distance > tolerance) or (
        normal_limit is not None and target_normal[1] > normal_limit
    )
    violated_requirement = (tolerance is not None and lower_distance > tolerance) or (
        normal_limit is not None and target_normal[0] > normal_limit
    )
    if violated_requirement:
        detail = (
            f"certified lower target-to-root deviation {lower_distance!r}; "
            f"normal-angle lower bound {target_normal[0]!r}"
        )
        return None, detail, False
    if unresolved_requirement:
        return (
            None,
            "target-to-root or normal interval bound needs face subdivision",
            True,
        )
    lifted_parameters = root_parameters
    for _ in range(0 if exact_on_source else root_newton_steps):
        # Midpoint interval Newton for phi(t) = f(P + t grad f(P)): every root
        # in T lies in m - phi(m) / phi'(T), with phi'(T) enclosed over the
        # tube of T; unlike the step from t = 0 its width shrinks with the piece.
        middle = 0.5 * lifted_parameters[0] + 0.5 * lifted_parameters[1]
        fields, _ = yield _BoxRequests(
            (
                _box_arrays(_lifted_box(base_box, (middle, middle), gradient_p)),
                _box_arrays(_lifted_box(base_box, lifted_parameters, gradient_p)),
            )
        )
        (middle_value, _), (_, tube_gradient) = fields
        slope = sum_interval(
            (
                _interval_multiply(tube_gradient[0], gradient_p[0]),
                _interval_multiply(tube_gradient[1], gradient_p[1]),
                _interval_multiply(tube_gradient[2], gradient_p[2]),
            )
        )
        if not slope[0] > 0.0:
            break
        lifted_parameters = _intersect_intervals(
            lifted_parameters,
            _interval_subtract((middle, middle), _interval_divide(middle_value, slope)),
        )
    return (
        _PieceProof(
            triangle,
            _prepare_barycentric_solver(triangle),
            maximum_root_parameter,
            max(-root_parameters[0], root_parameters[1]),
            (
                max(0.0, math.nextafter(value_low, -math.inf)),
                math.nextafter(value_high, math.inf),
            ),
            (lower_distance, upper_distance),
            target_normal,
            math.nextafter(chart_jacobian, 0.0),
            gradient_norm[0],
            gradient_norm[1],
            directional_derivative[0],
            directional_derivative[1],
            (
                map_derivatives[projected_axes[1]],
                map_derivatives[projected_axes[2]],
            ),
            target_derivatives,
            conditioned_derivatives,
            base_box,
            gradient_p,
            root_parameters,
            _lifted_box(base_box, lifted_parameters, gradient_p),
        ),
        "",
        False,
    )


def _lifted_box(
    base: Sequence[Interval], scale: Interval, gradient: IntervalVector3, /
) -> IntervalVector3:
    """Enclose ``P + t grad f(P)`` for ``P`` in ``base`` and ``t`` in ``scale``."""
    return (
        _interval_add(base[0], _interval_multiply(scale, gradient[0])),
        _interval_add(base[1], _interval_multiply(scale, gradient[1])),
        _interval_add(base[2], _interval_multiply(scale, gradient[2])),
    )


def _box_arrays(box: IntervalVector3, /) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([value[0] for value in box], dtype=np.float64),
        np.asarray([value[1] for value in box], dtype=np.float64),
    )


def _normal_area_interval(
    derivatives: IntervalDerivative3x2,
    gradient: IntervalVector3,
    orientation: int,
) -> Interval:
    tangent_u = tuple(row[0] for row in derivatives)
    tangent_v = tuple(row[1] for row in derivatives)
    cross = (
        _interval_subtract(
            _interval_multiply(tangent_u[1], tangent_v[2]),
            _interval_multiply(tangent_u[2], tangent_v[1]),
        ),
        _interval_subtract(
            _interval_multiply(tangent_u[2], tangent_v[0]),
            _interval_multiply(tangent_u[0], tangent_v[2]),
        ),
        _interval_subtract(
            _interval_multiply(tangent_u[0], tangent_v[1]),
            _interval_multiply(tangent_u[1], tangent_v[0]),
        ),
    )
    result = sum_interval(
        tuple(_interval_multiply(cross[axis], gradient[axis]) for axis in range(3))
    )
    return _interval_scale(result, float(orientation))


_STANDARD_TRIANGLE = (
    (Fraction(0), Fraction(0)),
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
)


def _prove_face_triangles(
    prove: _PieceProver,
    triangles: Sequence[TriangleReference],
) -> _FaceSteps[tuple[tuple[_PieceProof, ...], str]]:
    """Prove target subtriangles, splitting only the pieces that need refinement.

    Each refinement generation is one face round, batched across faces.
    """
    pending = list(triangles)
    accepted: list[_PieceProof] = []
    while pending:
        children: list[TriangleReference] = []
        outcomes, _ = yield _FaceRound(prove(pending))
        for triangle, (proof, detail, refine) in zip(pending, outcomes, strict=True):
            if proof is not None:
                accepted.append(proof)
                continue
            if not refine:
                return tuple(accepted), detail
            split = _split_triangle(triangle)
            if split is None:
                return tuple(
                    accepted
                ), detail or "implicit chart reached target-parameter representability"
            children.extend(split)
        pending = children
    return tuple(accepted), ""


_IDENTITY_BASIS: FractionMatrix2 = (
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
)
# Vertices of the chord set of the standard triangle: the differences u1 - u2
# fill the hexagon spanned by these directions and their negatives.
_CHORD_DIRECTIONS = ((1, 0), (0, 1), (1, -1))


def _chord_errors(
    face: _FaceEnclosure, basis: FractionMatrix2, /
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Bound the chord-mean error ``|D^3 P[d, d, B e_j]| / 24`` per coordinate.

    For a map of degree at most three the chord mean of ``DP B`` differs from
    its midpoint value by ``D^2(DP)[d, d] B / 24``, whose column ``j`` is
    ``D^3 P[d, d, B e_j] / 24``. With ``d = B s`` this is
    ``sum_ab D^3 P[B e_a, B e_b, B e_j] s_a s_b / 24``: the directional third
    derivatives are formed exactly before their magnitudes are taken, and
    ``|s_a|`` is bounded by ``B^-1`` over the chord hexagon. A needle face's
    reference derivatives then cancel exactly along its conditioned basis
    instead of being bounded term by term and amplified by ``B``.
    """
    (k00, k01), (k10, k11) = basis
    determinant = k00 * k11 - k01 * k10
    inverse = (
        (k11 / determinant, -k01 / determinant),
        (-k10 / determinant, k00 / determinant),
    )
    extents = tuple(
        max(
            abs(inverse[axis][0] * x + inverse[axis][1] * y) for x, y in _CHORD_DIRECTIONS
        )
        for axis in range(2)
    )
    columns = ((k00, k10), (k01, k11))
    rows: list[tuple[float, float]] = []
    for along_uu, along_uv, along_vv in face.curvatures:
        # Third derivatives indexed by their number of v differentiations.
        thirds = (
            _enclosed_derivative(along_uu, 0),
            _enclosed_derivative(along_uu, 1),
            _enclosed_derivative(along_uv, 1),
            _enclosed_derivative(along_vv, 1),
        )
        if all(not third.center and not third.error for third in thirds):
            # A coordinate of degree at most two has an exact chord mean.
            rows.append((0.0, 0.0))
            continue
        row: list[float] = []
        for column in range(2):
            total = Fraction(0)
            for first in range(2):
                for second in range(2):
                    weights = [Fraction(0)] * 4
                    for p in range(2):
                        for q in range(2):
                            for r in range(2):
                                weights[p + q + r] += (
                                    columns[first][p]
                                    * columns[second][q]
                                    * columns[column][r]
                                )
                    center: Polynomial = {}
                    error = 0.0
                    for weight, third in zip(weights, thirds, strict=True):
                        if weight == 0:
                            continue
                        for exponent, coefficient in third.center.items():
                            center[exponent] = (
                                center.get(exponent, Fraction(0)) + weight * coefficient
                            )
                        if third.error:
                            error = math.nextafter(
                                error
                                + math.nextafter(
                                    _exact_interval(abs(weight))[1] * third.error,
                                    math.inf,
                                ),
                                math.inf,
                            )
                    magnitude = (
                        max(
                            abs(value)
                            for value in _widen(
                                _poly_interval(center, _STANDARD_TRIANGLE), error
                            )
                        )
                        if center or error
                        else 0.0
                    )
                    if not math.isfinite(magnitude):
                        unbounded = (math.inf, math.inf)
                        return unbounded, unbounded, unbounded
                    total += Fraction(magnitude) * extents[first] * extents[second]
            row.append(_exact_interval(total / 24)[1])
        rows.append((row[0], row[1]))
    return rows[0], rows[1], rows[2]


type _TransversalCertificate = Callable[
    [Sequence[_PieceProof]], _FaceSteps[tuple[float, tuple[int, ...]]]
]


def _interval_hull(values: Sequence[Interval], /) -> Interval:
    return min(value[0] for value in values), max(value[1] for value in values)


def _interval_square(value: Interval, /) -> Interval:
    lower, upper = value
    largest = math.nextafter(max(lower * lower, upper * upper), math.inf)
    if lower <= 0.0 <= upper:
        return 0.0, largest
    smallest = min(abs(lower), abs(upper))
    return max(0.0, math.nextafter(smallest * smallest, 0.0)), largest


def _transversal_injectivity(
    face: _FaceEnclosure,
    orientation: int,
    /,
) -> _FaceSteps[_TransversalCertificate]:
    """Certify whole-face injectivity of the normal-line root chart directly.

    If ``R(u1) = R(u2)`` with ``R = P + t grad f(P)``, the mean-value forms
    ``P1 - P2 = A (u1 - u2)`` and ``grad f(P2) - grad f(P1) = H (P2 - P1)`` give
    ``(I + t1 H) A (u1 - u2) = (t2 - t1) grad f(P2)``, so
    ``det[(I + t1 H) A_u, (I + t1 H) A_v, grad f(P2)] = 0``: the polynomial
    ``M + t1 L + t1^2 Q`` in the root parameter ``t1`` of ``P1``. ``H`` is the
    mean Hessian on the segment ``P1 P2``. ``A`` is the chord mean of ``DP``;
    with ``m`` the chord midpoint and ``d = u1 - u2``, a map of degree at most
    three has ``A = DP(m) + D^2(DP)[d, d] / 24`` exactly.

    With ``u1`` in piece ``i`` and ``u2`` in piece ``j``, a collision needs the
    pieces' root boxes to meet. The certificate is a strictly signed enclosure
    of the determinant for every such pair, in one of three forms: every
    piece's ``A`` with the whole-face gradient and Hessian; the face hull of
    ``A`` with every piece's gradient and the whole-face Hessian; or, per piece
    ``j``, its gradient, the Hessian over the target boxes of the pieces whose
    root boxes meet its own, their root parameters, and the chord means of the
    pieces meeting the reference bounding box of those pieces (which contains
    every chord midpoint). The last form localizes with refinement, so pieces
    blocking it are returned for splitting.

    Returns a step function of the face pieces giving the normalized lower
    bound of the determinant (0.0 when no form is strictly signed) and the
    indices of the pieces blocking the local form.
    """
    face_low, face_high = _enclosed_box(face, _STANDARD_TRIANGLE)
    face_fields: tuple[IntervalVector3, IntervalMatrix3] | None = None
    if np.all(np.isfinite(face_low)) and np.all(np.isfinite(face_high)):
        _, results = yield _FaceRound(
            queries=(_BoxRequests(((face_low, face_high),), ((face_low, face_high),)),)
        )
        ((_, face_gradient),), (face_hessian,) = results[0]
        face_fields = (face_gradient, face_hessian)
    conditioner = face.conditioner
    # For a conditioned face A K replaces A: det[A K, w] = det K det[A, w].
    errors = _chord_errors(face, _IDENTITY_BASIS if conditioner is None else conditioner)
    inverse_determinant = (
        None
        if conditioner is None
        else _exact_interval(
            1
            / (
                conditioner[0][0] * conditioner[1][1]
                - conditioner[0][1] * conditioner[1][0]
            )
        )
    )

    def chord_mean(piece: _PieceProof, /) -> IntervalDerivative3x2 | None:
        derivatives = (
            piece.target_derivatives
            if conditioner is None
            else piece.conditioned_derivatives
        )
        if derivatives is None:
            return None

        def widened(axis: int, /) -> IntervalVector2:
            row, error = derivatives[axis], errors[axis]
            return (
                _interval_add(row[0], (-error[0], error[0])),
                _interval_add(row[1], (-error[1], error[1])),
            )

        return widened(0), widened(1), widened(2)

    def chord_hull(chords: Sequence[IntervalDerivative3x2], /) -> IntervalDerivative3x2:
        def row(axis: int, /) -> IntervalVector2:
            return (
                _interval_hull([chord[axis][0] for chord in chords]),
                _interval_hull([chord[axis][1] for chord in chords]),
            )

        return row(0), row(1), row(2)

    def determinant(
        chord: IntervalDerivative3x2,
        direction: IntervalVector3,
        hessian: IntervalMatrix3,
        parameters: Interval,
        /,
    ) -> Interval:
        def hessian_times(column: IntervalVector3, /) -> IntervalVector3:
            return (
                sum_interval(
                    tuple(_interval_multiply(hessian[0][k], column[k]) for k in range(3))
                ),
                sum_interval(
                    tuple(_interval_multiply(hessian[1][k], column[k]) for k in range(3))
                ),
                sum_interval(
                    tuple(_interval_multiply(hessian[2][k], column[k]) for k in range(3))
                ),
            )

        def triple(first: IntervalVector3, second: IntervalVector3, /) -> Interval:
            return _normal_area_interval(
                (
                    (first[0], second[0]),
                    (first[1], second[1]),
                    (first[2], second[2]),
                ),
                direction,
                orientation,
            )

        column_u: IntervalVector3 = (chord[0][0], chord[1][0], chord[2][0])
        column_v: IntervalVector3 = (chord[0][1], chord[1][1], chord[2][1])
        bent_u = hessian_times(column_u)
        bent_v = hessian_times(column_v)
        linear = _interval_add(triple(bent_u, column_v), triple(column_u, bent_v))
        value = _interval_add(
            _interval_add(
                triple(column_u, column_v), _interval_multiply(parameters, linear)
            ),
            _interval_multiply(_interval_square(parameters), triple(bent_u, bent_v)),
        )
        if inverse_determinant is None:
            return value
        return _interval_multiply(value, inverse_determinant)

    face_tangents = tuple(
        tuple(_enclosed_interval(value, _STANDARD_TRIANGLE) for value in row)
        for row in face.tangents
    )
    area_norm = _interval_vector_norm_bounds(
        (
            _interval_subtract(
                _interval_multiply(face_tangents[1][0], face_tangents[2][1]),
                _interval_multiply(face_tangents[2][0], face_tangents[1][1]),
            ),
            _interval_subtract(
                _interval_multiply(face_tangents[2][0], face_tangents[0][1]),
                _interval_multiply(face_tangents[0][0], face_tangents[2][1]),
            ),
            _interval_subtract(
                _interval_multiply(face_tangents[0][0], face_tangents[1][1]),
                _interval_multiply(face_tangents[1][0], face_tangents[0][1]),
            ),
        )
    )[1]

    def margin(
        pieces: Sequence[_PieceProof], /
    ) -> _FaceSteps[tuple[float, tuple[int, ...]]]:
        scale = area_norm * max(
            _interval_vector_norm_bounds(piece.source_gradient)[1] for piece in pieces
        )
        chords = [chord_mean(piece) for piece in pieces]
        known = [chord for chord in chords if chord is not None]
        if not math.isfinite(scale) or scale <= 0.0 or len(known) != len(chords):
            return 0.0, ()
        best = 0.0
        if face_fields is not None:
            face_gradient, face_hessian = face_fields
            parameters = _interval_hull([piece.root_parameters for piece in pieces])
            # |(I + tH)^-1 g - g|_inf <= rho / (1 - rho) |g|_inf with
            # rho = |t| ||H||_inf bounds the direction of the inverted form
            # det[A_u, A_v, (I + t1 H)^-1 grad f(P2)] without its t1 expansion.
            contraction = math.nextafter(
                max(-parameters[0], parameters[1])
                * max(
                    sum(max(abs(entry[0]), abs(entry[1])) for entry in row)
                    for row in face_hessian
                ),
                math.inf,
            )
            if math.isfinite(contraction) and contraction < 1.0:
                perturbation = math.nextafter(
                    contraction
                    / math.nextafter(1.0 - contraction, 0.0)
                    * max(max(abs(entry[0]), abs(entry[1])) for entry in face_gradient),
                    math.inf,
                )
                direction: IntervalVector3 = (
                    _interval_add(face_gradient[0], (-perturbation, perturbation)),
                    _interval_add(face_gradient[1], (-perturbation, perturbation)),
                    _interval_add(face_gradient[2], (-perturbation, perturbation)),
                )
                best = max(
                    best,
                    min(
                        determinant(chord, direction, face_hessian, (0.0, 0.0))[0]
                        for chord in known
                    ),
                )
            # Every midpoint piece's chord mean against the whole-face gradient.
            best = max(
                best,
                min(
                    determinant(chord, face_gradient, face_hessian, parameters)[0]
                    for chord in known
                ),
            )
            # The face hull of the chord means against every piece's gradient.
            hull = chord_hull(known)
            best = max(
                best,
                min(
                    determinant(hull, piece.source_gradient, face_hessian, parameters)[0]
                    for piece in pieces
                ),
            )
            if best > 0.0:
                return math.nextafter(best / scale, 0.0), ()
        root_boxes = np.asarray(
            [[list(bounds) for bounds in piece.root_box] for piece in pieces],
            dtype=np.float64,
        )
        target_boxes = np.asarray(
            [[list(bounds) for bounds in piece.target_box] for piece in pieces],
            dtype=np.float64,
        )
        corners = np.asarray(
            [
                [[float(value) for value in point] for point in piece.triangle]
                for piece in pieces
            ],
            dtype=np.float64,
        )
        reference_low = corners.min(axis=1)
        reference_high = corners.max(axis=1)
        neighborhoods: list[np.ndarray] = []
        hessian_boxes: list[tuple[np.ndarray, np.ndarray]] = []
        for row in range(len(pieces)):
            meets = np.all(
                (root_boxes[:, :, 0] <= root_boxes[row, :, 1])
                & (root_boxes[row, :, 0] <= root_boxes[:, :, 1]),
                axis=1,
            )
            neighborhood = np.flatnonzero(meets)
            neighborhoods.append(neighborhood)
            hessian_boxes.append(
                (
                    target_boxes[neighborhood, :, 0].min(axis=0),
                    target_boxes[neighborhood, :, 1].max(axis=0),
                )
            )
        _, results = yield _FaceRound(queries=(_BoxRequests((), tuple(hessian_boxes)),))
        hessians = results[0][1]
        # Reference corners are rounded to binary64; widening the region by a
        # few ulps of the unit triangle only admits more chord means.
        slack = 4.0 * np.finfo(np.float64).eps
        lower = math.inf
        blocking: set[int] = set()
        for row, (piece, neighborhood, hessian) in enumerate(
            zip(pieces, neighborhoods, hessians, strict=True)
        ):
            region_low = reference_low[neighborhood].min(axis=0) - slack
            region_high = reference_high[neighborhood].max(axis=0) + slack
            midpoint_pieces = np.flatnonzero(
                np.all(
                    (reference_low <= region_high) & (region_low <= reference_high),
                    axis=1,
                )
            )
            value = determinant(
                chord_hull([known[index] for index in midpoint_pieces]),
                piece.source_gradient,
                hessian,
                _interval_hull([pieces[index].root_parameters for index in neighborhood]),
            )[0]
            if value > 0.0:
                lower = min(lower, value)
            else:
                blocking.update(int(index) for index in neighborhood)
                blocking.add(row)
        if blocking:
            return 0.0, tuple(sorted(blocking))
        return math.nextafter(lower / scale, 0.0), ()

    return margin


def _injectivity_preconditioners(
    pieces: tuple[_PieceProof, ...], /
) -> tuple[FractionMatrix2, ...]:
    """Candidate preconditioners from the area-weighted mean projected Jacobian.

    With the mean ``M``, ``M^-1`` makes ``C J`` the identity at the mean, while
    ``M^T / |M|_F^2`` pairs each chord image with its mean image,
    ``(v-u)·C J (v-u) = (M d)·(J d)``, which a rotating Jacobian keeps positive
    on thin faces where ``M^-1`` amplifies the rotation by the condition number.
    """
    total = np.zeros((2, 2), dtype=np.float64)
    weight = 0.0
    for piece in pieces:
        midpoint = np.asarray(
            [
                [0.5 * (value[0] + value[1]) for value in row]
                for row in piece.target_jacobian
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(midpoint)):
            continue
        area = float(abs(_fraction_barycentric_area(piece.triangle)))
        total += area * midpoint
        weight += area
    if weight <= 0.0:
        return ()
    mean = total / weight
    if not np.all(np.isfinite(mean)) or np.linalg.det(mean) == 0.0:
        return ()
    candidates: list[np.ndarray] = []
    try:
        candidates.append(np.linalg.inv(mean))
    except np.linalg.LinAlgError:
        pass
    candidates.append(mean.T / float(np.sum(mean * mean)))
    return tuple(
        (
            (Fraction(float(matrix[0, 0])), Fraction(float(matrix[0, 1]))),
            (Fraction(float(matrix[1, 0])), Fraction(float(matrix[1, 1]))),
        )
        for matrix in candidates
        if np.all(np.isfinite(matrix))
    )


def _preconditioners_agree(first: FractionMatrix2, second: FractionMatrix2, /) -> bool:
    """Whether two mean preconditioners agree to 1% of their largest entry."""
    scale = max(abs(value) for row in first for value in row)
    return all(
        abs(left - right) <= scale / 100
        for row_first, row_second in zip(first, second, strict=True)
        for left, right in zip(row_first, row_second, strict=True)
    )


def _certify_face_injectivity(
    prove: _PieceProver,
    transversal: _TransversalCertificate,
    pieces: tuple[_PieceProof, ...],
) -> _FaceSteps[tuple[tuple[_PieceProof, ...], float, str]]:
    """Prove the projected root chart injective on the whole convex target face.

    Each generation first tries the transversal certificate on the current
    pieces. Otherwise, for a preconditioner ``C`` of the current pieces and
    distinct face points ``u, v``, the mean-value form gives
    ``(v-u)·C(F(v)-F(u)) = ∫ (v-u)·C J(u+s(v-u)) (v-u) ds``, so ``F`` is
    injective when ``sym(C J)`` is positive definite at every face point. Each
    piece checks its own Jacobian enclosure against every candidate ``C``. The
    pieces blocking the local transversal form split together with the
    undecided pieces of the best unrefuted candidate, so both certificates
    sharpen with the pieces. A candidate set is final once every candidate is
    refuted by a piece and refinement no longer moves them or leaves nothing
    undecided; the proof refuses when it is final and no piece blocks the local
    transversal form, and refinement ends at target-parameter representability.
    """
    current = list(pieces)
    previous: tuple[FractionMatrix2, ...] = ()
    while True:
        transversal_margin, blocking = yield from transversal(current)
        if transversal_margin > 0.0:
            return tuple(current), transversal_margin, ""
        candidates = _injectivity_preconditioners(tuple(current))
        monotone_split: list[int] = []
        refusal = "Implicit root chart has no invertible mean projected Jacobian."
        if candidates:
            outcomes: list[tuple[bool, list[int]]] = []
            for preconditioner in candidates:
                undecided: list[int] = []
                margin = math.inf
                refuted_any = False
                for index, piece in enumerate(current):
                    piece_margin, refuted = _projected_injectivity_margin(
                        piece.target_jacobian, preconditioner
                    )
                    if piece_margin > 0.0 or refuted:
                        margin = min(margin, piece_margin)
                        refuted_any = refuted_any or refuted
                    else:
                        undecided.append(index)
                if not refuted_any and not undecided:
                    return tuple(current), margin, ""
                outcomes.append((refuted_any, undecided))
            # A refutation is final once refinement no longer moves the mean
            # preconditioners; refusing is always sound, it only withholds a proof.
            settled = len(previous) == len(candidates) and all(
                _preconditioners_agree(first, second)
                for first, second in zip(previous, candidates, strict=True)
            )
            open_outcomes = [outcome for outcome in outcomes if outcome[1]]
            refusal = (
                "Implicit root chart is not strongly monotone under its mean "
                "projected-Jacobian preconditioners."
            )
            if not (
                all(outcome[0] for outcome in outcomes) and (settled or not open_outcomes)
            ):
                unrefuted = [outcome for outcome in open_outcomes if not outcome[0]]
                _, monotone_split = min(
                    unrefuted or open_outcomes, key=lambda outcome: len(outcome[1])
                )
            previous = candidates
        split_rows = sorted(set(blocking) | set(monotone_split))
        if not split_rows:
            return pieces, 0.0, refusal
        children: list[TriangleReference] = []
        for row in split_rows:
            split = _split_triangle(current[row].triangle)
            if split is None:
                return (
                    pieces,
                    0.0,
                    "Implicit root-chart injectivity reached target-parameter representability.",
                )
            children.extend(split)
        refined, detail = yield from _prove_face_triangles(prove, children)
        if detail:
            return pieces, 0.0, detail
        kept = set(split_rows)
        current = [piece for row, piece in enumerate(current) if row not in kept] + list(
            refined
        )


def _grid_corner(values: np.ndarray, /) -> GridCorner:
    return (int(values[0]), int(values[1]), int(values[2]))


def _canonical_grid_edge(edge: GridEdge, /) -> GridEdge:
    return edge if edge[0] <= edge[1] else (edge[1], edge[0])


def _grid_edge(values: np.ndarray, /) -> GridEdge:
    if values.shape != (2, 3):
        raise ValueError(
            "An adaptive root edge requires two three-dimensional grid keys."
        )
    return _canonical_grid_edge((_grid_corner(values[0]), _grid_corner(values[1])))


def _source_cell_key(values: np.ndarray, /) -> SourceCellKey:
    if values.shape != (4,):
        raise ValueError(
            "An adaptive source cell key requires level and three grid indices."
        )
    return (int(values[0]), int(values[1]), int(values[2]), int(values[3]))


def _root_edge_triangle(values: np.ndarray, /) -> RootEdgeTriangle:
    if values.shape != (3, 2, 3):
        raise ValueError(
            "An implicit chart requires three exact scaffold root-edge keys."
        )
    return (
        (_grid_corner(values[0, 0]), _grid_corner(values[0, 1])),
        (_grid_corner(values[1, 0]), _grid_corner(values[1, 1])),
        (_grid_corner(values[2, 0]), _grid_corner(values[2, 1])),
    )


def _token_chart(
    tokens: tuple[tuple[tuple[int, int], ...], ...], /
) -> tuple[tuple[Fraction, ...], ...]:
    # Per-row witness reads use token fields: binding a witness method builds an
    # equinox bound-method module that flattens the whole witness on every row.
    return tuple(
        tuple(Fraction(numerator, denominator) for numerator, denominator in point)
        for point in tokens
    )


def _reference_triangle(points: tuple[tuple[Fraction, ...], ...], /) -> ReferenceTriangle:
    if (
        len(points) != 3
        or not points[0]
        or len(points[1]) != len(points[0])
        or len(points[2]) != len(points[0])
    ):
        raise ValueError(
            "An implicit target chart requires three equal-dimensional exact corners."
        )
    return points[0], points[1], points[2]


def _source_parameter_triangle(
    points: tuple[tuple[Fraction, ...], ...], /
) -> TriangleReference:
    if len(points) != 3 or any(len(point) != 2 for point in points):
        raise ValueError(
            "An implicit source parameter chart requires three exact 2D corners."
        )
    return (
        (points[0][0], points[0][1]),
        (points[1][0], points[1][1]),
        (points[2][0], points[2][1]),
    )


@dataclass(frozen=True, slots=True)
class _TargetMapSource:
    """Resolved coordinate elements, routes and rational source bank of a target map."""

    elements: tuple[object, ...]
    routes: tuple[np.ndarray, ...]
    source_bank: CoordinateSourceBank


def _target_map_source(mesh: CellMesh, geometry: CellGeometrySpec, /) -> _TargetMapSource:
    elements, routes, _ = geometry.resolve(mesh)
    return _TargetMapSource(
        tuple(elements),
        tuple(np.asarray(route, dtype=np.int64) for route in routes),
        geometry.rational_source_coordinates(),
    )


def _target_cell_coordinate_expressions(
    target: _TargetMapSource,
    block_index: int,
    cell_row: int,
    /,
) -> Polynomial3:
    element = target.elements[block_index]
    if not isinstance(element, ScalarCoordinateElement):
        raise TypeError(
            "Implicit source charts require standard scalar coordinate elements."
        )
    local_route = target.routes[block_index][cell_row]
    local: CoordinateSourceBank = tuple(
        target.source_bank[int(index)] for index in local_route
    )
    raw: tuple[Expression, ...] = coordinate_map_expressions(
        element, local, cell_row=cell_row
    )
    if len(raw) != 3:
        raise ValueError(
            "Adaptive implicit P2/P3 charts require three exact target coordinates."
        )
    first, second, third = raw
    if (
        not isinstance(first, dict)
        or not isinstance(second, dict)
        or not isinstance(third, dict)
        or any(
            not isinstance(coefficient, (Fraction, RadicalElement))
            for expression in (first, second, third)
            if isinstance(expression, dict)
            for coefficient in expression.values()
        )
    ):
        raise ValueError(
            "Adaptive implicit P2/P3 charts require exact algebraic polynomial target maps."
        )
    return first, second, third


def _source_root_rows(
    surface: AdaptiveImplicitSurface,
) -> dict[GridEdge, tuple[np.ndarray, float]]:
    scaffold = surface.scaffold
    result: dict[GridEdge, tuple[np.ndarray, float]] = {}
    for key, point, error in zip(
        np.asarray(scaffold.root_edge_ids, dtype=np.int64),
        np.asarray(scaffold.root_points, dtype=np.float64),
        np.asarray(scaffold.root_errors, dtype=np.float64),
        strict=True,
    ):
        result[_grid_edge(np.asarray(key, dtype=np.int64))] = (
            np.array(point, copy=True),
            float(error),
        )
    return result


def _root_rows_by_point(
    root_rows: dict[GridEdge, tuple[np.ndarray, float]], /
) -> dict[bytes, GridEdge]:
    """Index retained scaffold roots by their exact point bits; the least edge wins."""
    result: dict[bytes, GridEdge] = {}
    for edge in sorted(root_rows):
        result.setdefault(root_rows[edge][0].tobytes(), edge)
    return result


def _source_cell_rows(surface: AdaptiveImplicitSurface, /) -> dict[SourceCellKey, int]:
    return {
        _source_cell_key(np.asarray(key, dtype=np.int64)): index
        for index, key in enumerate(
            np.asarray(surface.scaffold.cell_indices, dtype=np.int64)
        )
    }


def _box_distance_lower(point: np.ndarray, box: IntervalVector3, /) -> float:
    """A rounded-down lower bound on the distance from a point to a closed box."""
    gaps = []
    for axis in range(3):
        gap = max(box[axis][0] - float(point[axis]), float(point[axis]) - box[axis][1])
        gaps.append(math.nextafter(gap, 0.0) if gap > 0.0 else 0.0)
    squared = math.nextafter(
        math.fsum(math.nextafter(gap * gap, 0.0) for gap in gaps), 0.0
    )
    return max(0.0, math.nextafter(math.sqrt(max(squared, 0.0)), 0.0))


def _point_exclusion_steps(
    surface: AdaptiveImplicitSurface,
    face: _PreparedFace,
    point: np.ndarray,
    error: float,
) -> _FaceSteps[bool]:
    """Exclude ``point`` from one face's lift, splitting only the pieces that reach it.

    A piece's lifted points lie in its root box (target box plus the certified
    root parameter times the gradient box), whose width shrinks with the piece;
    the piece is excluded once that box is farther than ``error`` from the point.
    A piece whose root box is within ``error`` of the point and no wider than
    ``error`` on every axis already lifts into the point's uncertainty, so the
    face is not excluded; so is a piece that cannot be proved or split in binary64.
    """
    prove: _PieceProver = partial(
        _face_piece_proofs,
        surface,
        _enclose_face(face.face_expressions),
        source_orientation=face.orientation,
        projected_axes=face.projected_axes,
        tolerance=None,
        maximum_normal_deviation=None,
        root_newton_steps=3,
    )

    def decide(proof: _PieceProof, /) -> bool | None:
        """True: excluded; False: reaches the point at its resolution; None: split."""
        if _box_distance_lower(point, proof.root_box) > error:
            return True
        if all(upper - lower <= error for lower, upper in proof.root_box):
            return False
        return None

    pending: list[TriangleReference] = []
    proofs: list[_PieceProof | None] = list(face.pieces)
    triangles: list[TriangleReference] = [piece.triangle for piece in face.pieces]
    refinable = [True] * len(proofs)
    while True:
        for triangle, proof, refine in zip(triangles, proofs, refinable, strict=True):
            if proof is not None:
                decision = decide(proof)
                if decision is True:
                    continue
                if decision is False:
                    return False
            elif not refine:
                return False
            split = _split_triangle(triangle)
            if split is None:
                return False
            pending.extend(split)
        if not pending:
            return True
        outcomes, _ = yield _FaceRound(prove(pending))
        triangles, pending = pending, []
        proofs = [proof for proof, _detail, _refine in outcomes]
        refinable = [refine for _proof, _detail, refine in outcomes]


def _build_face(
    row: int,
    target_row: tuple[int, int],
    target: _TargetMapSource,
    witness: SourceBoundaryChartWitness,
    surface: AdaptiveImplicitSurface,
    root_rows: dict[GridEdge, tuple[np.ndarray, float]],
    root_by_point: dict[bytes, GridEdge],
    cell_rows: dict[SourceCellKey, int],
    *,
    tolerance: float | None,
    maximum_normal_deviation: float | None,
) -> _FaceSteps[_PreparedFace]:
    """Prove one face's root chart; its piece rounds are batched across faces."""

    block_index, cell_row = target_row
    cell_expressions = _target_cell_coordinate_expressions(target, block_index, cell_row)
    reference_corners = _reference_triangle(
        _token_chart(witness.target_reference_rational_tokens[row])
    )
    target_expressions = _triangle_coordinate_expressions(
        cell_expressions, reference_corners
    )
    if any(
        max((sum(power) for power in expression), default=0) > 3
        for expression in target_expressions
    ):
        raise ValueError(
            "Adaptive implicit root charts support target maps of degree at most P3."
        )
    scaffold = surface.scaffold
    cell_key = _source_cell_key(
        np.asarray(witness.implicit_cell_keys[row], dtype=np.int64)
    )
    cell_row_source = cell_rows.get(cell_key)
    if cell_row_source is None:
        raise ValueError("Implicit chart witness names an absent scaffold root cell.")
    cell_gradient_lower = np.asarray(
        scaffold.cell_gradient_lower[cell_row_source], dtype=np.float64
    )
    cell_gradient_upper = np.asarray(
        scaffold.cell_gradient_upper[cell_row_source], dtype=np.float64
    )
    strict_axes = [
        axis
        for axis in range(3)
        if cell_gradient_lower[axis] > 0.0 or cell_gradient_upper[axis] < 0.0
    ]
    if not strict_axes:
        raise ValueError("Implicit root cell has no strictly regular gradient component.")
    selected_axis = max(
        strict_axes,
        key=lambda axis: min(
            abs(cell_gradient_lower[axis]), abs(cell_gradient_upper[axis])
        ),
    )
    if selected_axis == 0:
        projected_axes = (0, 1, 2)
    elif selected_axis == 1:
        projected_axes = (1, 0, 2)
    else:
        projected_axes = (2, 0, 1)
    source_parameters = _source_parameter_triangle(
        _token_chart(witness.source_parameter_rational_tokens[row])
    )
    source_area = _fraction_barycentric_area(source_parameters)
    if source_area == 0:
        raise ValueError("Implicit source chart parameters are degenerate.")
    orientation = 1 if source_area > 0 else -1
    root_edges = _root_edge_triangle(
        np.asarray(witness.implicit_root_edge_keys[row], dtype=np.int64)
    )
    for edge in root_edges:
        if _canonical_grid_edge(edge) not in root_rows:
            raise ValueError(
                "Implicit chart root edge has no retained adaptive root witness."
            )
    face = _enclose_face(target_expressions)
    prove: _PieceProver = partial(
        _face_piece_proofs,
        surface,
        face,
        source_orientation=orientation,
        projected_axes=projected_axes,
        tolerance=tolerance,
        maximum_normal_deviation=maximum_normal_deviation,
    )
    pieces, detail = yield from _prove_face_triangles(prove, (_STANDARD_TRIANGLE,))
    if not pieces:
        raise ValueError(
            detail or "Implicit root chart has no certified target subtriangle."
        )
    injectivity_margin = 0.0
    if not detail:
        transversal = yield from _transversal_injectivity(face, orientation)
        pieces, injectivity_margin, detail = yield from _certify_face_injectivity(
            prove, transversal, pieces
        )
    jacobian_lower = min(piece.chart_jacobian_lower for piece in pieces)
    residual_upper = max(piece.field_residual[1] for piece in pieces)
    target_to_root_lower = max(piece.target_to_source[0] for piece in pieces)
    target_to_root_upper = max(piece.target_to_source[1] for piece in pieces)
    normal_lower = max(piece.normal_deviation[0] for piece in pieces)
    normal_upper = max(piece.normal_deviation[1] for piece in pieces)
    target_cell_id = int(witness.cell_global_ids[row])
    entity_id = int(witness.entity_global_ids[row])
    local_entity = int(witness.local_entity_indices[row])
    status = "certified" if not detail else "unresolved"
    evidence = ImplicitRootChartEvidence(
        row,
        target_cell_id,
        entity_id,
        local_entity,
        cell_key,
        root_edges,
        status,
        math.nextafter(residual_upper, math.inf),
        math.nextafter(target_to_root_lower, 0.0),
        math.nextafter(target_to_root_upper, math.inf),
        (
            math.nextafter(normal_lower, 0.0),
            math.nextafter(normal_upper, math.inf),
        ),
        math.nextafter(jacobian_lower, 0.0),
        math.nextafter(injectivity_margin, 0.0),
        min(piece.source_gradient_lower for piece in pieces),
        detail,
    )
    root_corners: list[RootCorner] = []
    for corner in reference_corners:
        corner_point = np.asarray(
            tuple(
                float(evaluate_expression(expression, corner))
                for expression in cell_expressions
            ),
            dtype=np.float64,
        )
        corner_edge = root_by_point.get(corner_point.tobytes())
        if corner_edge is None:
            root_corners.append(None)
        else:
            point, error = root_rows[corner_edge]
            root_corners.append((corner_edge, point, error))
    return _PreparedFace(
        row,
        block_index,
        cell_row,
        target_cell_id,
        entity_id,
        local_entity,
        cell_key,
        root_edges,
        reference_corners,
        _prepare_barycentric_solver(reference_corners),
        source_parameters,
        cell_expressions,
        target_expressions,
        face,
        pieces,
        tuple(root_corners),
        projected_axes,
        orientation,
        max(piece.maximum_root_parameter for piece in pieces),
        evidence,
    )


def _edge_polynomial(
    expression: Polynomial, start: TrianglePoint, stop: TrianglePoint, /
) -> Expression:
    """``expression(start + s (stop - start))`` as an exact univariate map.

    Rational coefficients expand binomially into the canonical form
    `compose_expression` returns; other coefficients compose there.
    """
    first_start, second_start = start
    first_step, second_step = stop[0] - first_start, stop[1] - second_start
    result: dict[int, Fraction] = {}
    for (first_power, second_power), coefficient in expression.items():
        if not isinstance(coefficient, Fraction):
            return compose_expression(
                expression,
                (
                    {(0,): first_start, (1,): first_step},
                    {(0,): second_start, (1,): second_step},
                ),
            )
        for first_order in range(first_power + 1):
            first_term = (
                math.comb(first_power, first_order)
                * first_start ** (first_power - first_order)
                * first_step**first_order
            )
            if not first_term:
                continue
            for second_order in range(second_power + 1):
                term = (
                    first_term
                    * math.comb(second_power, second_order)
                    * second_start ** (second_power - second_order)
                    * second_step**second_order
                )
                if term:
                    power = first_order + second_order
                    result[power] = result.get(power, Fraction(0)) + coefficient * term
    return {(power,): value for power, value in result.items() if value}


def _shared_edge_map_issues(
    mesh: CellMesh,
    witness: SourceBoundaryChartWitness,
    target_rows: tuple[tuple[int, int], ...],
    faces: tuple[_PreparedFace, ...],
) -> dict[int, str]:
    """Require exact target polynomial agreement and root ancestry on shared edges.

    A chart corner whose target point is exactly a retained scaffold root point
    takes that root (the scaffold path, where every corner is its named root);
    any other corner lifts its own physical point. Both depend only on the
    point, so a shared target vertex must be taken from the same scaffold root
    edge by every adjacent chart, or be lifted by all of them.
    """
    from ...discretization._reference_cell import reference_cell_topology

    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    edge_maps: dict[tuple[int, int], tuple[int, Polynomial3]] = {}
    root_at_vertex: dict[int, GridEdge | None] = {}
    issues: dict[int, str] = {}
    standard: TriangleReference = _STANDARD_TRIANGLE
    for row, (face, (block_index, cell_row)) in enumerate(
        zip(faces, target_rows, strict=True)
    ):
        block = mesh.blocks[block_index]
        topology = reference_cell_topology(block.cell_kind)
        local_positions: tuple[int, ...]
        if mesh.topological_dimension == 2:
            local_positions = tuple(range(3))
        else:
            local_entity = int(witness.local_entity_indices[row])
            if local_entity < 0 or local_entity >= len(topology.entities[2]):
                issues[row] = "Target implicit chart has no canonical local face route."
                continue
            local_positions = tuple(
                int(value) for value in topology.entities[2][local_entity]
            )
        reference_vertices: tuple[ReferencePoint, ...] = tuple(
            tuple(Fraction(float(value)) for value in point)
            for point in np.asarray(topology.vertices, dtype=np.float64)
        )
        cell_vertices = np.asarray(block.vertices, dtype=np.int64)[cell_row]
        chart_vertex_global_ids: list[int] = []
        for chart_corner in face.reference_corners:
            matches = tuple(
                position
                for position in local_positions
                if reference_vertices[position] == chart_corner
            )
            if len(matches) != 1:
                issues[row] = (
                    "Target chart corner does not bind one unique target-face vertex."
                )
                break
            local_position = matches[0]
            target_vertex_id = int(vertex_ids[int(cell_vertices[local_position])])
            chart_vertex_global_ids.append(target_vertex_id)
        if row in issues:
            continue
        for corner, target_vertex_id in enumerate(chart_vertex_global_ids):
            root_corner = face.root_corner_points[corner]
            root_key: GridEdge | None = None if root_corner is None else root_corner[0]
            if (
                target_vertex_id in root_at_vertex
                and root_at_vertex[target_vertex_id] != root_key
            ):
                issues[row] = (
                    "Adjacent target charts assign different exact scaffold root edges "
                    "to one shared target vertex."
                )
                break
            root_at_vertex[target_vertex_id] = root_key
        if row in issues:
            continue
        for first_corner, second_corner in ((0, 1), (1, 2), (2, 0)):
            first_id = chart_vertex_global_ids[first_corner]
            second_id = chart_vertex_global_ids[second_corner]
            if first_id == second_id:
                issues[row] = "Implicit source target face has a collapsed shared edge."
                break
            if first_id < second_id:
                start_corner, stop_corner = first_corner, second_corner
            else:
                start_corner, stop_corner = second_corner, first_corner
            start_uv = standard[start_corner]
            stop_uv = standard[stop_corner]
            first_map, second_map, third_map = (
                _edge_polynomial(expression, start_uv, stop_uv)
                for expression in face.face_expressions
            )
            if (
                not isinstance(first_map, dict)
                or not isinstance(second_map, dict)
                or not isinstance(third_map, dict)
            ):
                issues[row] = (
                    "Implicit shared-edge target maps must remain exact polynomials."
                )
                break
            edge_key = (min(first_id, second_id), max(first_id, second_id))
            canonical_map: Polynomial3 = (first_map, second_map, third_map)
            previous_edge = edge_maps.get(edge_key)
            if previous_edge is not None and previous_edge[1] != canonical_map:
                issues[row] = (
                    "Adjacent high-order target charts disagree exactly on a shared "
                    "physical edge map."
                )
                break
            edge_maps[edge_key] = (row, canonical_map)
    return issues


def _mark_chart_unresolved(face: _PreparedFace, detail: str, /) -> _PreparedFace:
    evidence = replace(
        face.evidence,
        status="unresolved",
        detail=detail,
        chart_jacobian_lower=0.0,
        chart_injectivity_margin=0.0,
    )
    return replace(face, evidence=evidence)


def prepare_adaptive_implicit_root_charts(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: AdaptiveImplicitSourceQuery,
    witness: SourceBoundaryChartWitness,
    /,
    *,
    frame_id: str | None = None,
    tolerance: float | None = None,
    maximum_normal_deviation: float | None = None,
) -> AdaptiveImplicitRootChartMap:
    """Prepare interval-certified nonlinear target-to-root charts for P1–P3 maps.

    `frame_id` may be supplied only by the caller that owns the exact spatial
    coordinate contract. No frame is inferred from field values or source IDs.
    """
    return _prepare_root_charts(
        mesh,
        geometry,
        source,
        witness,
        frame_id=frame_id,
        tolerance=tolerance,
        maximum_normal_deviation=maximum_normal_deviation,
        target_rows=None,
    )


def _prepare_root_charts(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: AdaptiveImplicitSourceQuery,
    witness: SourceBoundaryChartWitness,
    /,
    *,
    frame_id: str | None,
    tolerance: float | None,
    maximum_normal_deviation: float | None,
    target_rows: tuple[tuple[int, int], ...] | None,
) -> AdaptiveImplicitRootChartMap:
    """`prepare_adaptive_implicit_root_charts` body.

    ``target_rows`` are the witness's target chart rows from a caller that has
    already validated this mesh, geometry, source and witness; ``None``
    validates them and derives the rows here.
    """
    from ...geometry._source_fidelity import (
        _surface_target_chart_rows,
        validate_source_fidelity_witness,
    )

    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError(
            "Implicit root charts require canonical CellMesh and CellGeometrySpec operands."
        )
    if not isinstance(witness, SourceBoundaryChartWitness):
        raise TypeError("witness must be SourceBoundaryChartWitness.")
    adaptive, inherited_frame = _source_authority(source)
    selected_frame = (
        inherited_frame
        if frame_id is None
        else canonical_identifier(frame_id, "frame_id")
    )
    if inherited_frame and selected_frame != inherited_frame:
        raise ValueError(
            "Explicit frame_id differs from the source-bound coordinate frame."
        )
    if not selected_frame:
        raise ValueError(
            "Adaptive implicit root charts require a bound source coordinate frame."
        )
    surface = adaptive.surface
    if surface.geometry.field_certificate.regularity is not FieldRegularity.SMOOTH:
        if (
            surface.geometry.field_certificate.regularity
            is not FieldRegularity.PIECEWISE_SMOOTH
            or not _profile_covers_singularities_away_from_zero_set(
                source, surface, selected_frame
            )
        ):
            raise ValueError(
                "Implicit root charts require a smooth field or an exact profile singular set disjoint from the certified zero-set cover."
            )
    if target_rows is None:
        validate_source_fidelity_witness(mesh, geometry, source, witness)
    if (
        surface.cover is None
        or not surface.cover.complete
        or not surface.cover.certified
        or surface.cover.enclosure != "interval"
        or surface.volume.bounds.selector != "interval"
        or not surface.volume.bounds.gradient_rigorous
        or np.any(surface.scaffold.cell_issues != 0)
        or np.any(surface.scaffold.root_issues != 0)
    ):
        raise ValueError(
            "Implicit root charts require a complete interval source cover and regular root scaffold."
        )
    tolerance_ = None if tolerance is None else float(tolerance)
    normal_ = (
        None if maximum_normal_deviation is None else float(maximum_normal_deviation)
    )
    if tolerance_ is not None and (not math.isfinite(tolerance_) or tolerance_ < 0.0):
        raise ValueError("tolerance must be finite and nonnegative when supplied.")
    if normal_ is not None and (
        not math.isfinite(normal_) or normal_ < 0.0 or normal_ > math.pi
    ):
        raise ValueError("maximum_normal_deviation must be radians in [0, pi].")
    try:
        hessian_program = _prepare_hessian(surface.geometry)
    except (IntervalExtensionError, ArithmeticError, TypeError, ValueError) as error:
        raise ValueError(
            "Implicit root charts require a continuous interval Hessian enclosure."
        ) from error
    bounds = surface.volume.bounds
    if target_rows is None:
        target_rows, _, _ = _surface_target_chart_rows(
            mesh, geometry, None, witness, full_source=False
        )
    if len(target_rows) != len(witness.source_chart_ids):
        raise ValueError(
            "Implicit target chart rows do not align with the retained source witness."
        )
    root_rows = _source_root_rows(surface)
    root_by_point = _root_rows_by_point(root_rows)
    cell_rows = _source_cell_rows(surface)
    target = _target_map_source(mesh, geometry)
    prepared = _run_faces(
        bounds,
        hessian_program,
        [
            _build_face(
                row,
                target_row,
                target,
                witness,
                surface,
                root_rows,
                root_by_point,
                cell_rows,
                tolerance=tolerance_,
                maximum_normal_deviation=normal_,
            )
            for row, target_row in enumerate(target_rows)
        ],
    )
    edge_issues = _shared_edge_map_issues(mesh, witness, target_rows, tuple(prepared))
    for row, detail in edge_issues.items():
        prepared[row] = _mark_chart_unresolved(prepared[row], detail)
    return AdaptiveImplicitRootChartMap(
        mesh,
        geometry,
        source,
        witness,
        selected_frame,
        tuple(prepared),
        bounds,
        tolerance_,
        normal_,
        surface,
        hessian_program,
    )


# Rows per eager point-field block; blocks pad to power-of-two row buckets.
_POINT_FIELD_BLOCK = 1024
_MINIMUM_POINT_BUCKET = 8


def _field_values_and_gradients(
    geometry: CompiledGeometry, points: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary-field values and gradients of concrete host points, per block.

    Each row is the single-point eager evaluation ``boundary_field(state,
    p[None])[0]`` and its gradient, vmapped over bounded blocks of rows that
    are padded with copies of their last row to a power-of-two bucket, so the
    eager primitives see a bounded set of shapes.
    """

    def scalar(value: jax.Array, /) -> jax.Array:
        return geometry.kernel.boundary_field(geometry.state, value[None, :])[0]

    values = jax.vmap(scalar)
    gradients = jax.vmap(jax.grad(scalar))
    count = points.shape[0]
    field = np.empty((count,), dtype=np.float64)
    gradient = np.empty((count, 3), dtype=np.float64)
    for start in range(0, count, _POINT_FIELD_BLOCK):
        block = points[start : start + _POINT_FIELD_BLOCK]
        rows = block.shape[0]
        bucket = max(_MINIMUM_POINT_BUCKET, 1 << (rows - 1).bit_length())
        padded = jnp.asarray(
            np.concatenate((block, np.repeat(block[-1:], bucket - rows, axis=0)))
        )
        field[start : start + rows] = np.asarray(values(padded), dtype=np.float64)[:rows]
        gradient[start : start + rows] = np.asarray(
            gradients(padded), dtype=np.float64
        ).reshape((bucket, -1))[:rows]
    return field, gradient


def _doubling_steps(start: float, maximum_step: float, /) -> list[float]:
    """The coordinate-spacing doubling sequence inside a normal tube; never empty."""
    steps = [start]
    while True:
        next_step = math.nextafter(steps[-1] * 2.0, math.inf)
        if not math.isfinite(next_step) or next_step <= steps[-1]:
            break
        if next_step > maximum_step:
            break
        steps.append(next_step)
    return steps


def _lift_points(
    surface: AdaptiveImplicitSurface,
    frame_id: str,
    bounds: _FieldEnclosures,
    requests: Sequence[tuple[np.ndarray, _PreparedFace, float]],
    /,
) -> list[ImplicitRootChartPoint]:
    """Lift target points along their gradient lines to certified local roots.

    Doubling from the coordinate spacing keeps each bracket independent of the
    face, so faces sharing a node lift it to the same root: every doubling
    candidate inside a face's normal tube is signed and the first strictly
    opposite one closes the bracket. Requests sharing a point are lifted once
    over the widest tube and each keeps exactly the single-request outcome; all
    interval queries are batched, which leaves every enclosure unchanged.
    ``surface`` and ``frame_id`` are the source authority the caller's map
    resolved once at preparation.
    """
    if not requests:
        return []
    if not frame_id:
        raise ValueError(
            "Implicit root-chart point evaluation requires a bound source frame."
        )
    geometry = surface.geometry
    unique_rows: dict[bytes, int] = {}
    points: list[np.ndarray] = []
    widest: list[float] = []
    request_rows: list[int] = []
    for point, _, maximum_step in requests:
        key = point.tobytes()
        unique = unique_rows.get(key)
        if unique is None:
            unique = unique_rows[key] = len(points)
            points.append(point)
            widest.append(maximum_step)
        else:
            widest[unique] = max(widest[unique], maximum_step)
        request_rows.append(unique)
    stacked = np.stack(points)
    field_values, gradient_rows = _field_values_and_gradients(geometry, stacked)
    if gradient_rows.shape != (stacked.shape[0], 3) or not (
        np.all(np.isfinite(gradient_rows)) and np.all(np.isfinite(field_values))
    ):
        raise ValueError("Implicit root-chart point has a nonfinite field or gradient.")
    gradients = list(gradient_rows)
    boxes = bounds.boxes(stacked, stacked)
    for gradient_lower, gradient_upper in zip(
        boxes.gradient_lower, boxes.gradient_upper, strict=True
    ):
        if _gradient_norm_interval(gradient_lower, gradient_upper)[0] <= 0.0:
            raise ValueError(
                "Implicit root-chart point is singular or lacks an interval gradient norm."
            )
    field_lower, field_upper = bounds.point_values(stacked)
    signs: list[int] = []
    step_rows: list[list[float]] = []
    candidate_rows: list[np.ndarray] = []
    for unique, point in enumerate(points):
        low_value = float(field_lower[unique])
        high_value = float(field_upper[unique])
        if low_value <= 0.0 <= high_value:
            if low_value == 0.0 and high_value == 0.0:
                signs.append(0)
                step_rows.append([])
                candidate_rows.append(np.empty((0, 3), dtype=np.float64))
                continue
            raise ValueError(
                "Pointwise implicit field sign is unresolved at target-map resolution."
            )
        point_sign = -1 if high_value < 0.0 else 1
        step = float(np.spacing(max(1.0, float(np.linalg.norm(point)))))
        if step <= 0.0 or not math.isfinite(step):
            raise ValueError(
                "Implicit normal root bracket reached coordinate representability."
            )
        steps = _doubling_steps(step, widest[unique])
        direction = -1.0 if point_sign > 0 else 1.0
        signs.append(point_sign)
        step_rows.append(steps)
        candidate_rows.append(
            point[None, :]
            + (
                direction
                * np.asarray(steps, dtype=np.float64)[:, None]
                * gradients[unique][None, :]
            )
        )
    candidate_lower, candidate_upper = bounds.point_values(np.concatenate(candidate_rows))
    # Per point: the first doubling index that either closes the bracket
    # (True) or reaches coordinate representability (False).
    events: list[tuple[int, bool] | None] = []
    brackets: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    bracket_rows: dict[int, int] = {}
    offset = 0
    for unique, (point, candidates) in enumerate(
        zip(points, candidate_rows, strict=True)
    ):
        event: tuple[int, bool] | None = None
        for index, second in enumerate(candidates):
            if not np.all(np.isfinite(second)) or np.array_equal(second, point):
                event = (index, False)
                break
            if float(candidate_upper[offset + index]) < 0.0:
                second_sign = -1
            elif float(candidate_lower[offset + index]) > 0.0:
                second_sign = 1
            else:
                continue
            if second_sign != signs[unique]:
                event = (index, True)
                bracket_rows[unique] = len(brackets)
                brackets.append(
                    (point, np.array(second, copy=True), signs[unique], second_sign)
                )
                break
        offset += candidates.shape[0]
        events.append(event)
    roots = (
        _root_on_edges(
            bounds,
            np.asarray(
                [(first, second) for first, second, _, _ in brackets], dtype=np.float64
            ),
            np.asarray(
                [(first, second) for _, _, first, second in brackets], dtype=np.int8
            ),
            surface.policy.root_tolerance,
        )
        if brackets
        else ()
    )
    lifted: list[ImplicitRootChartPoint] = []
    for (point, face, maximum_step), unique in zip(requests, request_rows, strict=True):
        if signs[unique] == 0:
            lifted.append(
                ImplicitRootChartPoint(
                    np.array(point, copy=True),
                    0.0,
                    0.0,
                    face.source_cell_key,
                    face.root_edge_keys,
                    "certified_root_chart",
                )
            )
            continue
        steps = step_rows[unique]
        tube = 1 + sum(1 for value in steps[1:] if value <= maximum_step)
        event = events[unique]
        if event is None or event[0] >= tube:
            raise ValueError(
                "Implicit root is not isolated inside the certified face-local normal tube."
            )
        index, closes = event
        if not closes:
            raise ValueError(
                "Implicit normal root bracket reached coordinate representability."
            )
        if steps[index] > maximum_step:
            raise ValueError(
                "Implicit root bracket exceeds its certified face-local normal tube."
            )
        root_point, error, issue = roots[bracket_rows[unique]][:3]
        if issue:
            label = AdaptiveImplicitBoxIssue(issue).name.lower()
            raise ValueError(f"Implicit normal root isolation is unresolved ({label}).")
        displacement = np.nextafter(
            float(np.linalg.norm(root_point - point)) + float(error), np.inf
        )
        root_copy = np.array(root_point, dtype=np.float64, copy=True)
        root_copy.setflags(write=False)
        lifted.append(
            ImplicitRootChartPoint(
                root_copy,
                float(displacement),
                float(error),
                face.source_cell_key,
                face.root_edge_keys,
                "certified_root_chart",
            )
        )
    return lifted


# Monomials of a bivariate target face map of degree at most three.
_MONOMIALS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (1, 1),
    (0, 2),
    (3, 0),
    (2, 1),
    (1, 2),
    (0, 3),
)
# A possible-zero box that halving has shrunk below this fraction of the
# tolerance without being proved near the mesh or zero-free is refused.
_REVERSE_COVER_FLOOR = 2.0**-24
# Possible-zero boxes decided per interval evaluation round.
_REVERSE_COVER_ROUND = 8192


@dataclass(frozen=True, slots=True)
class _FaceMaps:
    """Binary64 coefficient enclosures of every target face map and their boxes."""

    # Nearest coefficients, used only to choose a face parameter to enclose.
    middle: np.ndarray
    # Outward coefficient enclosures (faces, coordinates, monomials) and each
    # coordinate's radical-coefficient error over the standard triangle.
    lower: np.ndarray
    upper: np.ndarray
    error: np.ndarray
    box_lower: np.ndarray
    box_upper: np.ndarray
    bvh: PackedBVH


def _face_maps(faces: Sequence[_PreparedFace], /) -> _FaceMaps:
    count = len(faces)
    columns = {monomial: column for column, monomial in enumerate(_MONOMIALS)}
    middle = np.zeros((count, 3, len(_MONOMIALS)), dtype=np.float64)
    lower = np.zeros_like(middle)
    upper = np.zeros_like(middle)
    error = np.zeros((count, 3), dtype=np.float64)
    box_lower = np.empty((count, 3), dtype=np.float64)
    box_upper = np.empty((count, 3), dtype=np.float64)
    for row, face in enumerate(faces):
        for axis, value in enumerate(face.face_enclosure.values):
            for exponent, coefficient in value.center.items():
                column = columns[(exponent[0], exponent[1])]
                middle[row, axis, column] = float(coefficient)
                lower[row, axis, column], upper[row, axis, column] = _exact_interval(
                    coefficient
                )
            error[row, axis] = value.error
        box_lower[row], box_upper[row] = _enclosed_box(
            face.face_enclosure, _STANDARD_TRIANGLE
        )
    return _FaceMaps(
        middle,
        lower,
        upper,
        error,
        box_lower,
        box_upper,
        prepare_bvh(box_lower, box_upper, dtype=np.float64),
    )


def _monomial_values(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    one = np.ones_like(first)
    return np.stack(
        (
            one,
            first,
            second,
            first * first,
            first * second,
            second * second,
            first * first * first,
            first * first * second,
            first * second * second,
            second * second * second,
        ),
        axis=-1,
    )


def _inside_standard_triangle(
    first: np.ndarray, second: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Clamp binary64 parameters into the closed standard triangle exactly.

    ``second`` stays two ulps below the rounded ``1 - first``, so the exact
    sum of both parameters never exceeds one.
    """
    first = np.clip(np.nan_to_num(first, nan=0.0), 0.0, 1.0)
    second = np.clip(np.nan_to_num(second, nan=0.0), 0.0, 1.0)
    total = first + second
    over = total > 1.0
    first = np.where(over, first / np.where(over, total, 1.0), first)
    second = np.where(over, second / np.where(over, total, 1.0), second)
    first = np.clip(first, 0.0, 1.0)
    second = np.clip(
        np.minimum(second, np.nextafter(np.nextafter(1.0 - first, 0.0), 0.0)), 0.0, 1.0
    )
    return first, second


def _nearest_face_parameters(
    maps: _FaceMaps, faces: np.ndarray, points: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Parameters of the face points nearest the query points, by Gauss-Newton.

    Only the choice of the enclosed face point depends on these; their accuracy
    never enters a bound.
    """
    coefficients = maps.middle[faces]
    first = np.full((faces.shape[0],), 1.0 / 3.0)
    second = np.full((faces.shape[0],), 1.0 / 3.0)
    zero = np.zeros_like(first)
    one = np.ones_like(first)
    for _ in range(8):
        residual = np.einsum(
            "nam,nm->na", coefficients, _monomial_values(first, second)
        ) - (points)
        along_first = np.einsum(
            "nam,nm->na",
            coefficients,
            np.stack(
                (
                    zero,
                    one,
                    zero,
                    2.0 * first,
                    second,
                    zero,
                    3.0 * first * first,
                    2.0 * first * second,
                    second * second,
                    zero,
                ),
                axis=-1,
            ),
        )
        along_second = np.einsum(
            "nam,nm->na",
            coefficients,
            np.stack(
                (
                    zero,
                    zero,
                    one,
                    zero,
                    first,
                    2.0 * second,
                    zero,
                    first * first,
                    2.0 * first * second,
                    3.0 * second * second,
                ),
                axis=-1,
            ),
        )
        a11 = np.sum(along_first * along_first, axis=1)
        a12 = np.sum(along_first * along_second, axis=1)
        a22 = np.sum(along_second * along_second, axis=1)
        b1 = -np.sum(along_first * residual, axis=1)
        b2 = -np.sum(along_second * residual, axis=1)
        determinant = a11 * a22 - a12 * a12
        valid = np.isfinite(determinant) & (determinant > 0.0)
        safe = np.where(valid, determinant, 1.0)
        first, second = _inside_standard_triangle(
            first + np.where(valid, (b1 * a22 - b2 * a12) / safe, 0.0),
            second + np.where(valid, (a11 * b2 - a12 * b1) / safe, 0.0),
        )
    return first, second


def _face_point_enclosures(
    maps: _FaceMaps,
    faces: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Outward enclosures ``(n, 3)`` of exact face points at standard-triangle parameters."""

    def product(
        left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray], /
    ) -> tuple[np.ndarray, np.ndarray]:
        # Both factors are nonnegative.
        return (
            np.nextafter(left[0] * right[0], -np.inf),
            np.nextafter(left[1] * right[1], np.inf),
        )

    one = (np.ones_like(first), np.ones_like(first))
    along_first = (first, first)
    along_second = (second, second)
    first_squared = product(along_first, along_first)
    mixed = product(along_first, along_second)
    second_squared = product(along_second, along_second)
    monomials = (
        one,
        along_first,
        along_second,
        first_squared,
        mixed,
        second_squared,
        product(first_squared, along_first),
        product(first_squared, along_second),
        product(along_first, second_squared),
        product(second_squared, along_second),
    )
    coefficient_lower = maps.lower[faces]
    coefficient_upper = maps.upper[faces]
    lower = np.zeros((faces.shape[0], 3), dtype=np.float64)
    upper = np.zeros((faces.shape[0], 3), dtype=np.float64)
    for column, (monomial_lower, monomial_upper) in enumerate(monomials):
        products = np.stack(
            (
                coefficient_lower[:, :, column] * monomial_lower[:, None],
                coefficient_lower[:, :, column] * monomial_upper[:, None],
                coefficient_upper[:, :, column] * monomial_lower[:, None],
                coefficient_upper[:, :, column] * monomial_upper[:, None],
            )
        )
        lower = np.nextafter(
            lower + np.nextafter(np.min(products, axis=0), -np.inf), -np.inf
        )
        upper = np.nextafter(
            upper + np.nextafter(np.max(products, axis=0), np.inf), np.inf
        )
    errors = maps.error[faces]
    return (
        np.nextafter(lower - errors, -np.inf),
        np.nextafter(upper + errors, np.inf),
    )


def _distance_upper(
    lower: np.ndarray,
    upper: np.ndarray,
    point_lower: np.ndarray,
    point_upper: np.ndarray,
    /,
) -> np.ndarray:
    """Upper bounds of ``|x - q|`` over ``x`` in each box and ``q`` in each point box."""
    extent = np.maximum(
        np.nextafter(upper - point_lower, np.inf),
        np.nextafter(point_upper - lower, np.inf),
    )
    squared = np.nextafter(extent * extent, np.inf)
    total = np.zeros((lower.shape[0],), dtype=np.float64)
    for axis in range(3):
        total = np.nextafter(total + squared[:, axis], np.inf)
    return np.nextafter(np.sqrt(total), np.inf)


def _distance_lower(
    lower: np.ndarray,
    upper: np.ndarray,
    other_lower: np.ndarray,
    other_upper: np.ndarray,
    /,
) -> np.ndarray:
    """Lower bounds of the distance between paired boxes."""
    gap = np.maximum(
        np.maximum(
            np.nextafter(other_lower - upper, -np.inf),
            np.nextafter(lower - other_upper, -np.inf),
        ),
        0.0,
    )
    squared = np.nextafter(gap * gap, 0.0)
    total = np.zeros((lower.shape[0],), dtype=np.float64)
    for axis in range(3):
        total = np.maximum(np.nextafter(total + squared[:, axis], -np.inf), 0.0)
    return np.nextafter(np.sqrt(total), 0.0)


def _box_value_bounds(
    bounds: _FieldEnclosures, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Field value enclosures of boxes, without the gradient enclosures `boxes` adds."""
    program = bounds._values
    if bounds.selector == "interval" and program is not None:
        value_lower, value_upper = _evaluate_interval_boxes(program, lower, upper)
        return value_lower.reshape((-1,)), value_upper.reshape((-1,))
    values = bounds.boxes(lower, upper)
    return values.value_lower, values.value_upper


def _box_face_pair_blocks(
    maps: _FaceMaps, lower: np.ndarray, upper: np.ndarray, reach: float, /
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Blocks of ``(box, face)`` pairs whose boxes come within ``reach`` in every coordinate.

    A face missing a box's pairs is farther than ``reach`` from the whole box.
    Blocks stream, so no query materializes all pairs at once.
    """
    if lower.shape[0] == 0:
        return
    reached = prepare_bvh(
        np.nextafter(lower - reach, -np.inf),
        np.nextafter(upper + reach, np.inf),
        dtype=np.float64,
    )
    for faces, boxes in bvh_overlap_pair_blocks(maps.bvh, reached, include_touching=True):
        yield np.asarray(boxes, dtype=np.int64), np.asarray(faces, dtype=np.int64)


def _box_face_distance_lower(
    maps: _FaceMaps, lower: np.ndarray, upper: np.ndarray, reach: float, /
) -> np.ndarray:
    """Per box, the least lower distance to the face boxes within ``reach``; else ``inf``."""
    nearest = np.full((lower.shape[0],), np.inf, dtype=np.float64)
    for boxes, faces in _box_face_pair_blocks(maps, lower, upper, reach):
        np.minimum.at(
            nearest,
            boxes,
            _distance_lower(
                lower[boxes], upper[boxes], maps.box_lower[faces], maps.box_upper[faces]
            ),
        )
    return nearest


def _witness_distance_lower(
    maps: _FaceMaps, points: np.ndarray, errors: np.ndarray, tolerance: float, /
) -> float:
    """The largest certified distance from a retained source root to the target mesh.

    Faces outside a reach ``r`` are farther than ``r`` from a root, so its
    distance is the least lower distance to the faces within ``r`` once that is
    below ``r``, and at least ``r`` when nothing comes within ``r``. The reach
    doubles from a small fraction of the tolerance until one of these holds or
    it exceeds the tolerance, so each query only meets the faces near a root.
    """
    nearest = np.empty((points.shape[0],), dtype=np.float64)
    pending = np.arange(points.shape[0])
    reach = tolerance * 2.0**-6
    while pending.shape[0]:
        found = _box_face_distance_lower(maps, points[pending], points[pending], reach)
        settled = (found < reach) | (reach > tolerance)
        nearest[pending[settled]] = np.minimum(found[settled], reach)
        pending = pending[~settled]
        reach = math.nextafter(2.0 * reach, math.inf)
    return float(
        np.max(np.maximum(np.nextafter(nearest - errors, -np.inf), 0.0), initial=0.0)
    )


def _source_to_mesh_bounds(
    surface: AdaptiveImplicitSurface, maps: _FaceMaps, tolerance: float, /
) -> tuple[float, float]:
    """Certified lower and upper bounds of the zero set's distance to the target mesh.

    The retained cover proves every box outside its possible-zero boxes
    sign-definite. Each possible-zero box is accepted once one exact face point
    ``q`` has ``max_{x in box} |x - q| <= tolerance``, is dropped once its own
    enclosure is sign-definite, and is otherwise halved, so only boxes along
    the zero set refine and only down to the tolerance scale. A box farther
    than the tolerance from every face box whose corners certify a sign change
    holds a zero point beyond the tolerance (the violated lower bound). The
    upper bound is infinite once a box reaches the refinement floor undecided.
    Retained source roots bound the distance from below.
    """
    cover = surface.cover
    if cover is None:
        return 0.0, math.inf
    bounds = surface.volume.bounds
    lower_bound = _witness_distance_lower(
        maps,
        np.asarray(surface.boundary_witnesses, dtype=np.float64).reshape((-1, 3)),
        np.asarray(surface.boundary_errors, dtype=np.float64).reshape((-1,)),
        tolerance,
    )
    upper_bound = 0.0
    floor = tolerance * _REVERSE_COVER_FLOOR
    pending = np.asarray(cover.boxes[cover.possible_zero_mask], dtype=np.float64)
    # Boxes are decided in bounded rounds, so one interval evaluation and its
    # face pairs never hold more than `_REVERSE_COVER_ROUND` boxes.
    queue = [pending]
    while queue:
        pending = queue.pop()
        if pending.shape[0] > _REVERSE_COVER_ROUND:
            queue.append(pending[_REVERSE_COVER_ROUND:])
            pending = pending[:_REVERSE_COVER_ROUND]
        value_lower, value_upper = _box_value_bounds(bounds, pending[:, 0], pending[:, 1])
        definite = (value_lower > 0.0) | (value_upper < 0.0)
        pending = pending[~definite]
        if not pending.shape[0]:
            continue
        box_lower, box_upper = pending[:, 0], pending[:, 1]
        radius = _distance_upper(
            box_lower,
            box_upper,
            0.5 * box_lower + 0.5 * box_upper,
            0.5 * box_lower + 0.5 * box_upper,
        )
        # Acceptance tries the faces nearest each small box's centre: the
        # centre's reach doubles from a small fraction of the tolerance until
        # some face box comes within it, so each query meets only nearby faces.
        reach = np.full((pending.shape[0],), np.inf, dtype=np.float64)
        trial = np.flatnonzero(radius <= tolerance)
        centers = 0.5 * box_lower + 0.5 * box_upper
        search = tolerance * 2.0**-6
        while trial.shape[0] and search <= 2.0 * tolerance:
            found = np.zeros((trial.shape[0],), dtype=np.bool_)
            for boxes, faces in _box_face_pair_blocks(
                maps, centers[trial], centers[trial], search
            ):
                found[boxes] = True
                rows = trial[boxes]
                first, second = _nearest_face_parameters(maps, faces, centers[rows])
                point_lower, point_upper = _face_point_enclosures(
                    maps, faces, first, second
                )
                np.minimum.at(
                    reach,
                    rows,
                    _distance_upper(
                        box_lower[rows], box_upper[rows], point_lower, point_upper
                    ),
                )
            trial = trial[~found]
            search = math.nextafter(2.0 * search, math.inf)
        accepted = reach <= tolerance
        if np.any(accepted):
            upper_bound = max(upper_bound, float(np.max(reach[accepted])))
        near = accepted.copy()
        undecided = np.flatnonzero(~accepted)
        for boxes, _faces in _box_face_pair_blocks(
            maps, box_lower[undecided], box_upper[undecided], tolerance
        ):
            near[undecided[boxes]] = True
        far = ~near & ~accepted
        for row in np.flatnonzero(far):
            corners = np.where(
                np.asarray(
                    [[(index >> axis) & 1 for axis in range(3)] for index in range(8)]
                )
                == 0,
                box_lower[row],
                box_upper[row],
            )
            corner_lower, corner_upper = bounds.point_values(corners)
            if np.any(corner_lower > 0.0) and np.any(corner_upper < 0.0):
                # A certified zero lies in a box farther than the tolerance
                # from every face box: the reverse tolerance is violated.
                count = maps.box_lower.shape[0]
                beyond = float(
                    np.min(
                        _distance_lower(
                            np.broadcast_to(box_lower[row], (count, 3)),
                            np.broadcast_to(box_upper[row], (count, 3)),
                            maps.box_lower,
                            maps.box_upper,
                        )
                    )
                )
                return max(lower_bound, beyond), math.inf
        remaining = pending[~accepted]
        middle = 0.5 * remaining[:, 0] + 0.5 * remaining[:, 1]
        splittable = np.all(
            (remaining[:, 0] < middle) & (middle < remaining[:, 1]), axis=1
        ) & (radius[~accepted] > floor)
        if not np.all(splittable):
            upper_bound = math.inf
        remaining, middle = remaining[splittable], middle[splittable]
        children = []
        for index in range(8):
            choice = np.asarray(
                [(index >> axis) & 1 for axis in range(3)], dtype=np.bool_
            )
            children.append(
                np.stack(
                    (
                        np.where(choice, middle, remaining[:, 0]),
                        np.where(choice, remaining[:, 1], middle),
                    ),
                    axis=1,
                )
            )
        if remaining.shape[0]:
            queue.append(np.concatenate(children, axis=0))
    return lower_bound, upper_bound


def certify_adaptive_implicit_charts(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: AdaptiveImplicitSourceQuery,
    witness: SourceBoundaryChartWitness | ValidatedSourceFidelityWitness,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
    root_charts: AdaptiveImplicitRootChartMap | None = None,
) -> SourceFidelityCertificate:
    """Certify nonlinear P1/P2/P3 target maps against an adaptive implicit root cover.

    ``root_charts`` is a chart map the caller already prepared for exactly this
    mesh, geometry, source, witness and controls; it is checked against them
    and used instead of preparing the charts again.
    """
    from ...geometry._source_fidelity import (
        _surface_target_chart_rows,
        _unresolved_fidelity,
        validate_source_fidelity_witness,
    )
    from ...geometry._source_fidelity_certificate import SourceFidelityCertificate

    chart_witness = validate_source_fidelity_witness(
        mesh, geometry, source, witness
    ).witness
    if not isinstance(chart_witness, SourceBoundaryChartWitness):
        raise TypeError("witness must be SourceBoundaryChartWitness.")
    witness = chart_witness
    if root_charts is not None:
        root_charts.require_request(
            mesh,
            geometry,
            source,
            witness,
            tolerance=tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
        )
        root_map = root_charts
    else:
        try:
            target_rows, _, _ = _surface_target_chart_rows(
                mesh, geometry, None, witness, full_source=False
            )
            root_map = _prepare_root_charts(
                mesh,
                geometry,
                source,
                witness,
                frame_id=None,
                tolerance=tolerance,
                maximum_normal_deviation=maximum_normal_deviation,
                target_rows=target_rows,
            )
        except (ArithmeticError, IndexError, TypeError, ValueError) as error:
            return _unresolved_fidelity(
                mesh,
                geometry,
                source,
                tolerance=tolerance,
                maximum_normal_deviation=maximum_normal_deviation,
                witness=witness,
                coverage=None,
                findings=(
                    MeshCertificateFinding(
                        "implicit_root_chart",
                        "unresolved",
                        "mesh",
                        detail=f"Source-owned implicit target-to-root chart is unresolved: {error}",
                    ),
                ),
            )
    findings: list[MeshCertificateFinding] = []
    for evidence in root_map.chart_evidence:
        if evidence.status != "certified":
            findings.append(
                MeshCertificateFinding(
                    "implicit_root_chart",
                    "unresolved",
                    "cell",
                    (evidence.target_cell_global_id,),
                    detail=evidence.detail,
                )
            )
        if evidence.target_to_root_lower_bound > tolerance:
            findings.append(
                MeshCertificateFinding(
                    "source_fidelity_tolerance",
                    "violated",
                    "cell",
                    (evidence.target_cell_global_id,),
                    detail=(
                        "Continuous target-to-root lower bound "
                        f"{evidence.target_to_root_lower_bound!r} exceeds tolerance {tolerance!r}."
                    ),
                )
            )
        elif evidence.target_to_root_bound > tolerance:
            findings.append(
                MeshCertificateFinding(
                    "source_fidelity_tolerance",
                    "unresolved",
                    "cell",
                    (evidence.target_cell_global_id,),
                    detail=(
                        "Continuous target-to-root upper bound "
                        f"{evidence.target_to_root_bound!r} exceeds tolerance {tolerance!r}."
                    ),
                )
            )
        if maximum_normal_deviation is not None and maximum_normal_deviation < math.pi:
            normal_status = (
                "violated"
                if evidence.source_normal_deviation_bounds[0] > maximum_normal_deviation
                else "unresolved"
            )
            if evidence.source_normal_deviation_bounds[1] > maximum_normal_deviation:
                findings.append(
                    MeshCertificateFinding(
                        "normal_deviation_range",
                        normal_status,
                        "cell",
                        (evidence.target_cell_global_id,),
                        detail=(
                            "Continuous implicit source-normal angle bound "
                            f"{evidence.source_normal_deviation_bounds!r} exceeds "
                            f"{maximum_normal_deviation!r}."
                        ),
                    )
                )
    adaptive, _ = _source_authority(source)
    if tolerance <= 0.0:
        return _unresolved_fidelity(
            mesh,
            geometry,
            source,
            tolerance=tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
            witness=witness,
            coverage=None,
            findings=tuple(
                findings
                + [
                    MeshCertificateFinding(
                        "implicit_source_cover",
                        "unresolved",
                        "source_facet",
                        detail="A positive fidelity tolerance is required to construct a finite implicit source cover.",
                    )
                ]
            ),
        )
    cover = adaptive.surface.cover
    if (
        cover is None
        or not cover.complete
        or not cover.certified
        or not np.any(cover.possible_zero_mask)
    ):
        return _unresolved_fidelity(
            mesh,
            geometry,
            source,
            tolerance=tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
            witness=witness,
            coverage=None,
            findings=tuple(
                findings
                + [
                    MeshCertificateFinding(
                        "implicit_source_cover",
                        "unresolved",
                        "source_facet",
                        detail="The current adaptive source root cover is incomplete.",
                    )
                ]
            ),
        )
    source_to_mesh_lower, source_to_mesh_upper = _source_to_mesh_bounds(
        adaptive.surface, _face_maps(root_map._faces), float(tolerance)
    )
    per_face = np.asarray(
        [
            (
                item.target_to_root_lower_bound,
                max(item.target_to_root_bound, source_to_mesh_upper),
            )
            for item in root_map.chart_evidence
        ],
        dtype=np.float64,
    )
    normal_bounds = np.asarray(
        [item.source_normal_deviation_bounds[1] for item in root_map.chart_evidence],
        dtype=np.float64,
    )
    if source_to_mesh_lower > tolerance:
        findings.append(
            MeshCertificateFinding(
                "source_fidelity_tolerance",
                "violated",
                "source_sample",
                detail="A certified adaptive source root is farther than the requested tolerance from every target chart.",
            )
        )
    elif source_to_mesh_upper > tolerance:
        findings.append(
            MeshCertificateFinding(
                "source_fidelity_tolerance",
                "unresolved",
                "source_sample",
                detail="The local adaptive source cover does not yet prove the reverse source-to-target tolerance.",
            )
        )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    forward_lower = max(
        (
            Fraction.from_float(float(item.target_to_root_lower_bound))
            for item in root_map.chart_evidence
        ),
        default=Fraction(0),
    )
    forward_upper = max(
        (
            Fraction.from_float(float(item.target_to_root_bound))
            for item in root_map.chart_evidence
        ),
        default=Fraction(0),
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance,
        maximum_normal_deviation,
        per_face,
        # Distances are nonnegative: rounding a zero lower bound down stays at 0.
        max(0.0, math.nextafter(float(forward_lower), -math.inf)),
        math.nextafter(float(forward_upper), math.inf),
        max(0.0, math.nextafter(float(source_to_mesh_lower), -math.inf)),
        math.nextafter(float(source_to_mesh_upper), math.inf),
        normal_bounds,
        max(
            (item.source_normal_deviation_bounds[0] for item in root_map.chart_evidence),
            default=0.0,
        ),
        tuple(int(value) for value in witness.entity_global_ids),
        tuple(0 for _ in witness.entity_global_ids),
        witness.occurrence_paths,
        witness,
        None,
        tuple(findings),
    )


class _AdaptiveImplicitProjectionResult(StrictModule, NonTrainableState):
    """Conservative globally scoped result; candidates never imply uniqueness."""

    source_id: str
    source_revision: str
    surface_entity_id: str = eqx.field(static=True)
    frame_id: str
    queries: ArrayLike
    points: ArrayLike
    parameters: ArrayLike
    residuals: ArrayLike
    distance_lower_bounds: ArrayLike
    status: ArrayLike
    normals: ArrayLike
    tangents: ArrayLike
    occurrence_paths: tuple[tuple[str, ...], ...]
    candidate_root_rows: np.ndarray
    reasons: tuple[str, ...]

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        frame_id: str,
        queries: np.ndarray,
        points: np.ndarray,
        residuals: np.ndarray,
        lower: np.ndarray,
        candidate_rows: np.ndarray,
        reasons: tuple[str, ...],
        /,
    ) -> None:
        source_id_ = str(source_id)
        source_revision_ = str(source_revision)
        surface_entity_id_ = source_entity_id(
            source_id_,
            source_revision_,
            "implicit-surface",
            "zero-set",
        )
        frame_id_ = canonical_identifier(frame_id, "frame_id")
        queries_ = np.asarray(queries, dtype=np.float64)
        points_ = np.asarray(points, dtype=np.float64)
        parameters_ = np.full((queries_.shape[0], 0), np.nan, dtype=np.float64)
        residuals_ = np.asarray(residuals, dtype=np.float64).reshape((-1,))
        lower_ = np.asarray(lower, dtype=np.float64).reshape((-1,))
        status_ = np.full(
            (queries_.shape[0],),
            int(SourceProjectionStatus.UNRESOLVED),
            dtype=np.int8,
        )
        normals_ = np.full((queries_.shape[0], 3), np.nan, dtype=np.float64)
        tangents_ = np.full((queries_.shape[0], 2, 3), np.nan, dtype=np.float64)
        occurrence_paths_ = tuple(() for _ in range(queries_.shape[0]))
        candidate_rows_ = np.asarray(candidate_rows, dtype=np.int64).reshape((-1,))
        reasons_ = tuple(reasons)
        if (
            queries_.ndim != 2
            or queries_.shape[1] != 3
            or points_.shape != queries_.shape
            or residuals_.shape != (queries_.shape[0],)
            or lower_.shape != residuals_.shape
            or candidate_rows_.shape != residuals_.shape
            or len(reasons_) != residuals_.size
        ):
            raise ValueError("Adaptive implicit projection evidence rows must align.")
        for array in (
            queries_,
            points_,
            parameters_,
            residuals_,
            lower_,
            status_,
            normals_,
            tangents_,
            candidate_rows_,
        ):
            array.setflags(write=False)
        self.source_id = source_id_
        self.source_revision = source_revision_
        self.surface_entity_id = surface_entity_id_
        self.frame_id = frame_id_
        self.queries = queries_
        self.points = points_
        self.parameters = parameters_
        self.residuals = residuals_
        self.distance_lower_bounds = lower_
        self.status = status_
        self.normals = normals_
        self.tangents = tangents_
        self.occurrence_paths = occurrence_paths_
        self.candidate_root_rows = candidate_rows_
        self.reasons = reasons_

    def entity_ids(self, /) -> tuple[str, ...]:
        """Return the canonical source entity even when its closest point is unresolved."""
        return tuple(
            self.surface_entity_id for _ in range(np.asarray(self.queries).shape[0])
        )


def project_adaptive_implicit_boundary(
    source: AdaptiveImplicitBoundarySource,
    points: ConvertibleToArray,
    /,
    *,
    frame_id: str,
    entity_ids: tuple[str, ...] | None = None,
    occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
) -> _AdaptiveImplicitProjectionResult:
    """Return conservative global bounds without converting a root anchor to a closest-point proof."""
    source.require_current()
    frame = canonical_identifier(frame_id, "frame_id")
    queries = np.asarray(points, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 3 or not np.all(np.isfinite(queries)):
        raise ValueError("points must be finite with shape (N, 3).")
    surface_entity_id = source_entity_id(
        source.source_id,
        source.source_revision,
        "implicit-surface",
        "zero-set",
    )
    if entity_ids is not None and (
        len(entity_ids) != queries.shape[0]
        or any(value != surface_entity_id for value in entity_ids)
    ):
        raise ValueError(
            "Adaptive implicit projection scope must name its canonical zero-set entity."
        )
    if occurrence_paths is not None and (
        len(occurrence_paths) != queries.shape[0]
        or any(tuple(path) for path in occurrence_paths)
    ):
        raise ValueError(
            "Adaptive implicit sources have a definition-only empty occurrence path."
        )
    distance = source.boundary_distance(queries)
    roots = np.asarray(source.surface.boundary_witnesses, dtype=np.float64)
    root_errors = np.asarray(source.surface.boundary_errors, dtype=np.float64)
    lower = np.asarray(distance.lower, dtype=np.float64)
    if roots.shape[0] == 0:
        candidate_points = np.full_like(queries, np.nan)
        candidate_rows = np.full((queries.shape[0],), -1, dtype=np.int64)
        upper = np.full((queries.shape[0],), np.inf, dtype=np.float64)
    else:
        delta = np.nextafter(np.abs(queries[:, None, :] - roots[None, :, :]), np.inf)
        squared = np.nextafter(delta * delta, np.inf)
        sum_upper = np.nextafter(np.sum(squared, axis=2), np.inf)
        distance_upper = np.nextafter(np.sqrt(sum_upper), np.inf)
        candidate_upper = np.nextafter(distance_upper + root_errors[None, :], np.inf)
        candidate_rows = np.argmin(candidate_upper, axis=1).astype(np.int64)
        candidate_points = np.array(roots[candidate_rows], copy=True)
        upper = candidate_upper[np.arange(queries.shape[0]), candidate_rows]
    reason = (
        "A globally closest implicit projection is unresolved: complete possible-zero boxes and root anchors bound distance, but do not isolate the unique global minimizer."
        if distance.semantics == "certified"
        else "A globally closest implicit projection is unresolved because the source has no complete certified possible-zero cover."
    )
    reasons = tuple(reason for _ in range(queries.shape[0]))
    return _AdaptiveImplicitProjectionResult(
        source.source_id,
        source.source_revision,
        frame,
        queries,
        candidate_points,
        upper,
        lower,
        candidate_rows,
        reasons,
    )


__all__ = [
    "AdaptiveImplicitRootChartMap",
    "ImplicitRootChartEvidence",
    "ImplicitRootChartPoint",
    "prepare_adaptive_implicit_root_charts",
    "certify_adaptive_implicit_charts",
    "project_adaptive_implicit_boundary",
]
