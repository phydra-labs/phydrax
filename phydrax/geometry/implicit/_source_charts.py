#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

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
from ...discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
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
    _FieldEnclosures,
    _root_on_edge,
    AdaptiveImplicitBoundarySource,
    AdaptiveImplicitBoxIssue,
    AdaptiveImplicitSurface,
)
from ._analytic_profile import AnalyticImplicitProfile


if TYPE_CHECKING:
    from ...geometry._source_fidelity import SourceFidelityCertificate
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
type PhysicalPoint3 = tuple[Fraction, Fraction, Fraction]
type RootCorner = tuple[TrianglePoint, np.ndarray, float]


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
    field_residual: Interval
    target_to_source: Interval
    normal_deviation: Interval
    chart_jacobian_lower: float
    source_gradient_lower: float
    source_gradient_upper: float
    directional_derivative_lower: float
    directional_derivative_upper: float
    target_jacobian: IntervalMatrix2


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
        for corner, (source_uv, root_point, root_error) in enumerate(
            face.root_corner_points
        ):
            unit = tuple(Fraction(int(index == corner)) for index in range(3))
            if (
                barycentric == unit
                and face.source_parameters[corner] == source_uv
                and np.array_equal(point.view(np.uint64), root_point.view(np.uint64))
            ):
                return ImplicitRootChartPoint(
                    np.array(root_point, copy=True),
                    float(np.nextafter(root_error, np.inf)),
                    float(np.nextafter(root_error, np.inf)),
                    face.source_cell_key,
                    face.root_edge_keys,
                    "certified_root_chart",
                )
        piece_parameter: float | None = None
        for piece in face.pieces:
            local = _barycentric(target_chart_point, piece.triangle, solver=piece.solver)
            if all(value >= 0 for value in local):
                piece_parameter = piece.maximum_root_parameter
                break
        if piece_parameter is None:
            raise ValueError(
                "Target reference point is not covered by a certified implicit chart subtriangle."
            )
        return _lift_point(
            self.source,
            self._bounds,
            point,
            face,
            piece_parameter,
        )

    def evaluate_target(
        self,
        row: int,
        target_cell_reference: ConvertibleToArray | Sequence[ExactScalar],
        /,
    ) -> np.ndarray:
        """Evaluate the source-root lift at one full target-cell reference point."""
        return np.array(self.project_target(row, target_cell_reference).point, copy=True)

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


def _interval_add(first: Interval, second: Interval, /) -> Interval:
    return (
        float(np.nextafter(first[0] + second[0], -np.inf)),
        float(np.nextafter(first[1] + second[1], np.inf)),
    )


def _interval_scale(value: Interval, scalar: float, /) -> Interval:
    products = (value[0] * scalar, value[1] * scalar)
    return (
        float(np.nextafter(min(products), -np.inf)),
        float(np.nextafter(max(products), np.inf)),
    )


def _interval_multiply(first: Interval, second: Interval, /) -> Interval:
    products = tuple(a * b for a in first for b in second)
    if any(math.isnan(value) for value in products):
        return (-math.inf, math.inf)
    return (
        float(np.nextafter(min(products), -np.inf)),
        float(np.nextafter(max(products), np.inf)),
    )


def _interval_subtract(first: Interval, second: Interval, /) -> Interval:
    return _interval_add(first, (-second[1], -second[0]))


def _interval_divide(numerator: Interval, denominator: Interval, /) -> Interval:
    if denominator[0] <= 0.0 <= denominator[1]:
        return (-math.inf, math.inf)
    reciprocal = (1.0 / denominator[1], 1.0 / denominator[0])
    if reciprocal[0] > reciprocal[1]:
        reciprocal = (reciprocal[1], reciprocal[0])
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
    lower = float(np.nextafter(math.sqrt(max(0.0, math.fsum(lower_squares))), 0.0))
    upper = float(np.nextafter(math.sqrt(math.fsum(upper_squares)), np.inf))
    return lower, upper


def _fraction_interval(value: Fraction) -> tuple[float, float]:
    represented = float(value)
    if not math.isfinite(represented):
        return (-math.inf, math.inf)
    exact = Fraction.from_float(represented)
    return (
        represented if exact <= value else float(np.nextafter(represented, -np.inf)),
        represented if exact >= value else float(np.nextafter(represented, np.inf)),
    )


def _poly_interval(expression: Polynomial, triangle: TriangleReference) -> Interval:
    arguments = _triangle_arguments(triangle)
    restricted = compose_expression(expression, arguments)
    if not isinstance(restricted, dict):
        raise TypeError(
            "Implicit source chart target maps must be polynomial coordinate maps."
        )
    return polynomial_bounds(restricted, "simplex", 2)


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


def _restrict_polynomial(
    expression: Polynomial, triangle: TriangleReference, /
) -> Polynomial:
    restricted = compose_expression(expression, _triangle_arguments(triangle))
    if not isinstance(restricted, dict):
        raise TypeError("Implicit source chart target maps must remain polynomial.")
    return restricted


def _evaluate_fraction_polynomial(
    expression: Polynomial, point: tuple[Fraction, ...], /
) -> Fraction:
    value = evaluate_expression(expression, point)
    if not isinstance(value, Fraction):
        raise TypeError(
            "Certified implicit target maps require rational coordinate values."
        )
    return value


def _physical_point3(values: np.ndarray, /) -> PhysicalPoint3:
    if values.shape != (3,):
        raise ValueError("A physical implicit source sample must have three coordinates.")
    return (
        Fraction.from_float(float(values[0])),
        Fraction.from_float(float(values[1])),
        Fraction.from_float(float(values[2])),
    )


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


def _hessian_boxes(
    program: PreparedIntervalFunction, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    lower, upper = program.evaluate(low[None, :], high[None, :])
    return np.stack(
        (
            np.asarray(lower, dtype=np.float64).reshape((3, 3)),
            np.asarray(upper, dtype=np.float64).reshape((3, 3)),
        ),
        axis=-1,
    )


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
    cosine_lower = float(np.nextafter(min(cosines), -np.inf))
    cosine_upper = float(np.nextafter(max(cosines), np.inf))
    cosine_lower = min(1.0, max(-1.0, cosine_lower))
    cosine_upper = min(1.0, max(-1.0, cosine_upper))
    return (
        float(np.nextafter(math.acos(cosine_upper), 0.0)),
        float(np.nextafter(math.acos(cosine_lower), np.inf)),
    )


def _chart_derivative_intervals(
    target_derivatives: IntervalDerivative3x2,
    gradient_p: IntervalVector3,
    gradient_q: IntervalVector3,
    hessian_p: IntervalMatrix3,
    maximum_root_parameter: float,
) -> IntervalDerivative3x2 | None:
    identity: IntervalMatrix3 = (
        ((1.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), (1.0, 1.0), (0.0, 0.0)),
        ((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)),
    )
    matrix_b: list[list[Interval]] = []
    t_interval: Interval = (-maximum_root_parameter, maximum_root_parameter)
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


def _projected_injectivity_margin(
    jacobian: IntervalMatrix2, preconditioner: FractionMatrix2, /
) -> float:
    rows: list[list[Interval]] = []
    for row in range(2):
        values: list[Interval] = []
        for column in range(2):
            total: Interval = (0.0, 0.0)
            for axis in range(2):
                total = _interval_add(
                    total,
                    _interval_multiply(
                        _fraction_interval(preconditioner[row][axis]),
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
    if symmetric_11[0] <= 0.0 or determinant[0] <= 0.0:
        return 0.0
    return min(symmetric_11[0], determinant[0])


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


def _box_from_polynomials(
    expressions: Polynomial3,
    triangle: TriangleReference,
) -> tuple[np.ndarray, np.ndarray]:
    intervals = tuple(_poly_interval(expression, triangle) for expression in expressions)
    return (
        np.asarray([value[0] for value in intervals], dtype=np.float64),
        np.asarray([value[1] for value in intervals], dtype=np.float64),
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


def _field_box(
    bounds: _FieldEnclosures, lower: np.ndarray, upper: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    result = bounds.boxes(np.asarray(lower)[None, :], np.asarray(upper)[None, :])
    return (
        float(np.asarray(result.value_lower).reshape(-1)[0]),
        float(np.asarray(result.value_upper).reshape(-1)[0]),
        np.asarray(result.gradient_lower, dtype=np.float64).reshape((3,)),
        np.asarray(result.gradient_upper, dtype=np.float64).reshape((3,)),
    )


def _face_piece_proof(
    surface: AdaptiveImplicitSurface,
    bounds: _FieldEnclosures,
    hessian_program: PreparedIntervalFunction,
    target_expressions: Polynomial3,
    triangle: TriangleReference,
    source_orientation: int,
    projected_axes: tuple[int, int, int],
    tolerance: float | None,
    maximum_normal_deviation: float | None,
) -> tuple[_PieceProof | None, str, bool]:
    """Prove one subtriangle's scalar root chart and interval differential."""
    restricted: Polynomial3 = (
        _restrict_polynomial(target_expressions[0], triangle),
        _restrict_polynomial(target_expressions[1], triangle),
        _restrict_polynomial(target_expressions[2], triangle),
    )
    p_low, p_high = _box_from_polynomials(restricted, _STANDARD_TRIANGLE)
    if np.array_equal(p_low, p_high):
        return (
            None,
            "Target coordinate map has no physical two-dimensional chart extent.",
            False,
        )
    value_low, value_high, gradient_lower, gradient_upper = _field_box(
        bounds, p_low, p_high
    )
    gradient_p: IntervalVector3 = (
        (float(gradient_lower[0]), float(gradient_upper[0])),
        (float(gradient_lower[1]), float(gradient_upper[1])),
        (float(gradient_lower[2]), float(gradient_upper[2])),
    )
    gradient_norm = _gradient_norm_interval(gradient_lower, gradient_upper)
    if gradient_norm[0] <= 0.0:
        return None, "target face has no strict source-gradient norm enclosure", True
    residual = (value_low, value_high)
    residual_upper = max(abs(value_low), abs(value_high))
    exact_on_source = value_low == 0.0 and value_high == 0.0
    if exact_on_source:
        maximum_root_parameter = 0.0
    else:
        maximum_root_parameter = float(
            np.nextafter(
                4.0 * residual_upper / (gradient_norm[0] * gradient_norm[0]), np.inf
            )
        )
    if not math.isfinite(maximum_root_parameter):
        return None, "implicit root-chart normal tube is not finitely enclosed", True
    target_derivatives: IntervalDerivative3x2 = (
        (
            _poly_interval(derivative(target_expressions[0], 0), triangle),
            _poly_interval(derivative(target_expressions[0], 1), triangle),
        ),
        (
            _poly_interval(derivative(target_expressions[1], 0), triangle),
            _poly_interval(derivative(target_expressions[1], 1), triangle),
        ),
        (
            _poly_interval(derivative(target_expressions[2], 0), triangle),
            _poly_interval(derivative(target_expressions[2], 1), triangle),
        ),
    )
    if exact_on_source:
        gradient_q = gradient_p
        map_derivatives = target_derivatives
        directional_derivative = (
            float(np.nextafter(gradient_norm[0] ** 2, 0.0)),
            float(np.nextafter(gradient_norm[1] ** 2, np.inf)),
        )
    else:
        displacement = np.asarray(
            [
                maximum_root_parameter
                * max(abs(gradient_lower[axis]), abs(gradient_upper[axis]))
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        tube_lower = np.nextafter(p_low - displacement, -np.inf)
        tube_upper = np.nextafter(p_high + displacement, np.inf)
        if np.any(tube_lower < surface.volume.domain[0]) or np.any(
            tube_upper > surface.volume.domain[1]
        ):
            return (
                None,
                "implicit root-chart tube leaves the certified source domain",
                True,
            )
        _, _, gradient_q_lower, gradient_q_upper = _field_box(
            bounds, tube_lower, tube_upper
        )
        gradient_q: IntervalVector3 = (
            (float(gradient_q_lower[0]), float(gradient_q_upper[0])),
            (float(gradient_q_lower[1]), float(gradient_q_upper[1])),
            (float(gradient_q_lower[2]), float(gradient_q_upper[2])),
        )
        directional_derivative = sum_interval(
            (
                _interval_multiply(gradient_q[0], gradient_p[0]),
                _interval_multiply(gradient_q[1], gradient_p[1]),
                _interval_multiply(gradient_q[2], gradient_p[2]),
            )
        )
        if directional_derivative[0] <= 0.0:
            return None, "normal-line field derivative is not strictly positive", True
        normal_shifts = tuple(
            _interval_multiply(
                (maximum_root_parameter, maximum_root_parameter),
                component,
            )
            for component in gradient_p
        )
        base_box = tuple(
            (float(p_low[axis]), float(p_high[axis])) for axis in range(3)
        )
        minus_box = tuple(
            _interval_subtract(base_box[axis], normal_shifts[axis])
            for axis in range(3)
        )
        plus_box = tuple(
            _interval_add(base_box[axis], normal_shifts[axis])
            for axis in range(3)
        )
        minus_lower = np.asarray(
            [component[0] for component in minus_box], dtype=np.float64
        )
        minus_upper = np.asarray(
            [component[1] for component in minus_box], dtype=np.float64
        )
        plus_lower = np.asarray(
            [component[0] for component in plus_box], dtype=np.float64
        )
        plus_upper = np.asarray(
            [component[1] for component in plus_box], dtype=np.float64
        )
        _, minus_value_upper, _, _ = _field_box(bounds, minus_lower, minus_upper)
        plus_value_lower, _, _, _ = _field_box(bounds, plus_lower, plus_upper)
        if not (minus_value_upper < 0.0 and plus_value_lower > 0.0):
            return (
                None,
                "normal-line root chart lacks strict interval endpoint signs",
                True,
            )
        hessian = _hessian_boxes(hessian_program, p_low, p_high)
        hessian_intervals: IntervalMatrix3 = (
            (
                (float(hessian[0, 0, 0]), float(hessian[0, 0, 1])),
                (float(hessian[0, 1, 0]), float(hessian[0, 1, 1])),
                (float(hessian[0, 2, 0]), float(hessian[0, 2, 1])),
            ),
            (
                (float(hessian[1, 0, 0]), float(hessian[1, 0, 1])),
                (float(hessian[1, 1, 0]), float(hessian[1, 1, 1])),
                (float(hessian[1, 2, 0]), float(hessian[1, 2, 1])),
            ),
            (
                (float(hessian[2, 0, 0]), float(hessian[2, 0, 1])),
                (float(hessian[2, 1, 0]), float(hessian[2, 1, 1])),
                (float(hessian[2, 2, 0]), float(hessian[2, 2, 1])),
            ),
        )
        map_derivatives = _chart_derivative_intervals(
            target_derivatives,
            gradient_p,
            gradient_q,
            hessian_intervals,
            maximum_root_parameter,
        )
        if map_derivatives is None:
            return (
                None,
                "implicit root-chart differential denominator contains zero",
                True,
            )
    cross_dot = _normal_area_interval(map_derivatives, gradient_q, source_orientation)
    chart_jacobian = cross_dot[0]
    if cross_dot[0] <= 0.0:
        return (
            None,
            "implicit source chart Jacobian is not strictly positive in its source orientation",
            True,
        )
    tangent_u: IntervalVector3 = (
        target_derivatives[0][0],
        target_derivatives[1][0],
        target_derivatives[2][0],
    )
    tangent_v: IntervalVector3 = (
        target_derivatives[0][1],
        target_derivatives[1][1],
        target_derivatives[2][1],
    )
    target_normal = _normal_angle_bounds(
        tangent_u, tangent_v, gradient_q, source_orientation
    )
    lower_distance = 0.0
    f_abs_lower = _point_interval(residual)
    directional_upper = sum(
        max(abs(gradient_q[axis][0]), abs(gradient_q[axis][1]))
        * max(abs(gradient_p[axis][0]), abs(gradient_p[axis][1]))
        for axis in range(3)
    )
    if directional_upper > 0.0:
        lower_distance = float(
            np.nextafter(
                f_abs_lower * gradient_norm[0] / directional_upper,
                0.0,
            )
        )
    upper_distance = float(
        np.nextafter(maximum_root_parameter * gradient_norm[1], np.inf)
    )
    unresolved_requirement = (tolerance is not None and upper_distance > tolerance) or (
        maximum_normal_deviation is not None
        and target_normal[1] > maximum_normal_deviation
    )
    violated_requirement = (tolerance is not None and lower_distance > tolerance) or (
        maximum_normal_deviation is not None
        and target_normal[0] > maximum_normal_deviation
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
    return (
        _PieceProof(
            triangle,
            _prepare_barycentric_solver(triangle),
            maximum_root_parameter,
            (
                max(0.0, float(np.nextafter(value_low, -np.inf))),
                float(np.nextafter(value_high, np.inf)),
            ),
            (lower_distance, upper_distance),
            target_normal,
            float(np.nextafter(chart_jacobian, 0.0)),
            gradient_norm[0],
            gradient_norm[1],
            directional_derivative[0],
            directional_derivative[1],
            (
                map_derivatives[projected_axes[1]],
                map_derivatives[projected_axes[2]],
            ),
        ),
        "",
        False,
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


def _make_face_pieces(
    surface: AdaptiveImplicitSurface,
    bounds: _FieldEnclosures,
    hessian_program: PreparedIntervalFunction,
    target_expressions: Polynomial3,
    orientation: int,
    projected_axes: tuple[int, int, int],
    tolerance: float | None,
    maximum_normal_deviation: float | None,
) -> tuple[tuple[_PieceProof, ...], str]:
    pending = [_STANDARD_TRIANGLE]
    accepted: list[_PieceProof] = []
    while pending:
        triangle = pending.pop()
        proof, detail, refine = _face_piece_proof(
            surface,
            bounds,
            hessian_program,
            target_expressions,
            triangle,
            orientation,
            projected_axes,
            tolerance,
            maximum_normal_deviation,
        )
        if proof is not None:
            accepted.append(proof)
            continue
        if not refine:
            return tuple(accepted), detail
        children = _split_triangle(triangle)
        if children is None:
            return tuple(
                accepted
            ), detail or "implicit chart reached target-parameter representability"
        pending.extend(reversed(children))
    return tuple(accepted), ""


def _strong_monotonicity_margin(
    pieces: tuple[_PieceProof, ...], projected_axes: tuple[int, int, int]
) -> float:
    if not pieces:
        return 0.0
    intervals: list[list[Interval]] = []
    for projected_row in range(2):
        row_values: list[Interval] = []
        for input_axis in range(2):
            values: tuple[Interval, ...] = tuple(
                piece.target_jacobian[projected_row][input_axis] for piece in pieces
            )
            row_values.append(
                (min(value[0] for value in values), max(value[1] for value in values))
            )
        intervals.append(row_values)
    midpoint = np.asarray(
        [[0.5 * (value[0] + value[1]) for value in row] for row in intervals],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(midpoint)) or np.linalg.det(midpoint) == 0.0:
        return 0.0
    try:
        inverse = np.linalg.inv(midpoint)
    except np.linalg.LinAlgError:
        return 0.0
    preconditioner: FractionMatrix2 = (
        (
            Fraction(float(inverse[0, 0])),
            Fraction(float(inverse[0, 1])),
        ),
        (
            Fraction(float(inverse[1, 0])),
            Fraction(float(inverse[1, 1])),
        ),
    )
    jacobian: IntervalMatrix2 = (
        (intervals[0][0], intervals[0][1]),
        (intervals[1][0], intervals[1][1]),
    )
    return _projected_injectivity_margin(jacobian, preconditioner)


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


def _target_cell_coordinate_expressions(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    block_index: int,
    cell_row: int,
    /,
) -> Polynomial3:
    elements, routes, _ = geometry.resolve(mesh)
    element = elements[block_index]
    if not isinstance(element, ScalarCoordinateElement):
        raise TypeError(
            "Implicit source charts require standard scalar coordinate elements."
        )
    source_bank: CoordinateSourceBank = geometry.rational_source_coordinates()
    local_route = np.asarray(routes[block_index], dtype=np.int64)[cell_row]
    local: CoordinateSourceBank = tuple(source_bank[int(index)] for index in local_route)
    raw: tuple[Expression, ...] = coordinate_map_expressions(
        element, local, cell_row=cell_row
    )
    if len(raw) != 3:
        raise ValueError(
            "Adaptive implicit P2/P3 charts require three exact rational target coordinates."
        )
    first, second, third = raw
    if (
        not isinstance(first, dict)
        or not isinstance(second, dict)
        or not isinstance(third, dict)
        or any(
            not isinstance(coefficient, Fraction)
            for expression in (first, second, third)
            if isinstance(expression, dict)
            for coefficient in expression.values()
        )
    ):
        raise ValueError(
            "Adaptive implicit P2/P3 charts require exact rational polynomial target maps."
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


def _build_face(
    row: int,
    target_row: tuple[int, int],
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: SourceBoundaryChartWitness,
    surface: AdaptiveImplicitSurface,
    bounds: _FieldEnclosures,
    hessian_program: PreparedIntervalFunction,
    root_rows: dict[GridEdge, tuple[np.ndarray, float]],
    *,
    tolerance: float | None,
    maximum_normal_deviation: float | None,
) -> _PreparedFace:

    block_index, cell_row = target_row
    cell_expressions = _target_cell_coordinate_expressions(
        mesh, geometry, block_index, cell_row
    )
    reference_corners = _reference_triangle(witness.target_reference_chart(row))
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
    cell_rows: dict[SourceCellKey, int] = {
        _source_cell_key(np.asarray(key, dtype=np.int64)): index
        for index, key in enumerate(np.asarray(scaffold.cell_indices, dtype=np.int64))
    }
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
    source_parameters = _source_parameter_triangle(witness.source_parameter_chart(row))
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
    pieces, detail = _make_face_pieces(
        surface,
        bounds,
        hessian_program,
        target_expressions,
        orientation,
        projected_axes,
        tolerance,
        maximum_normal_deviation,
    )
    if not pieces:
        raise ValueError(
            detail or "Implicit root chart has no certified target subtriangle."
        )
    injectivity_margin = _strong_monotonicity_margin(pieces, projected_axes)
    while injectivity_margin <= 0.0 and not detail:
        refined: list[_PieceProof] = []
        for piece in pieces:
            children = _split_triangle(piece.triangle)
            if children is None:
                detail = "Implicit root-chart injectivity reached target-parameter representability."
                break
            for child in children:
                proof, piece_detail, _refine = _face_piece_proof(
                    surface,
                    bounds,
                    hessian_program,
                    target_expressions,
                    child,
                    orientation,
                    projected_axes,
                    tolerance,
                    maximum_normal_deviation,
                )
                if proof is None:
                    detail = piece_detail
                    break
                refined.append(proof)
            if detail:
                break
        if detail or not refined:
            break
        pieces = tuple(refined)
        injectivity_margin = _strong_monotonicity_margin(pieces, projected_axes)
    if injectivity_margin <= 0.0:
        detail = detail or "Implicit root chart does not prove global face injectivity."
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
        float(np.nextafter(residual_upper, np.inf)),
        float(np.nextafter(target_to_root_lower, 0.0)),
        float(np.nextafter(target_to_root_upper, np.inf)),
        (
            float(np.nextafter(normal_lower, 0.0)),
            float(np.nextafter(normal_upper, np.inf)),
        ),
        float(np.nextafter(jacobian_lower, 0.0)),
        float(np.nextafter(injectivity_margin, 0.0)),
        min(piece.source_gradient_lower for piece in pieces),
        detail,
    )
    root_corners: list[RootCorner] = []
    for parameter, edge in zip(source_parameters, root_edges, strict=True):
        point, error = root_rows[_canonical_grid_edge(edge)]
        root_corners.append((parameter, point, error))
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
        pieces,
        tuple(root_corners),
        projected_axes,
        orientation,
        max(piece.maximum_root_parameter for piece in pieces),
        evidence,
    )


def _shared_edge_map_issues(
    mesh: CellMesh,
    witness: SourceBoundaryChartWitness,
    target_rows: tuple[tuple[int, int], ...],
    faces: tuple[_PreparedFace, ...],
) -> dict[int, str]:
    """Require exact target polynomial agreement and root ancestry on shared edges."""
    from ...discretization._reference_cell import reference_cell_topology

    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    edge_maps: dict[tuple[int, int], tuple[int, Polynomial3]] = {}
    root_at_vertex: dict[int, GridEdge] = {}
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
            root_key = _canonical_grid_edge(face.root_edge_keys[corner])
            previous = root_at_vertex.get(target_vertex_id)
            if previous is not None and previous != root_key:
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
            argument_u: Polynomial = {
                (0,): start_uv[0],
                (1,): stop_uv[0] - start_uv[0],
            }
            argument_v: Polynomial = {
                (0,): start_uv[1],
                (1,): stop_uv[1] - start_uv[1],
            }
            first_map = compose_expression(
                face.face_expressions[0], (argument_u, argument_v)
            )
            second_map = compose_expression(
                face.face_expressions[1], (argument_u, argument_v)
            )
            third_map = compose_expression(
                face.face_expressions[2], (argument_u, argument_v)
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
    target_rows, _, _ = _surface_target_chart_rows(
        mesh, geometry, None, witness, full_source=False
    )
    if len(target_rows) != len(witness.source_chart_ids):
        raise ValueError(
            "Implicit target chart rows do not align with the retained source witness."
        )
    root_rows = _source_root_rows(surface)
    prepared: list[_PreparedFace] = []
    for row, target_row in enumerate(target_rows):
        prepared.append(
            _build_face(
                row,
                target_row,
                mesh,
                geometry,
                witness,
                surface,
                bounds,
                hessian_program,
                root_rows,
                tolerance=tolerance_,
                maximum_normal_deviation=normal_,
            )
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
    )


def _lift_point(
    source: AdaptiveImplicitSourceQuery,
    bounds: _FieldEnclosures,
    point: np.ndarray,
    face: _PreparedFace,
    maximum_root_parameter: float,
) -> ImplicitRootChartPoint:
    adaptive, frame_id = _source_authority(source)
    if not frame_id:
        raise ValueError(
            "Implicit root-chart point evaluation requires a bound source frame."
        )
    surface = adaptive.surface
    geometry = surface.geometry
    scalar = lambda value: geometry.kernel.boundary_field(geometry.state, value[None, :])[
        0
    ]
    gradient_function = jax.grad(scalar)
    field_value = float(np.asarray(scalar(jnp.asarray(point)), dtype=np.float64))
    gradient = np.asarray(gradient_function(jnp.asarray(point)), dtype=np.float64)
    if (
        gradient.shape != (3,)
        or not np.all(np.isfinite(gradient))
        or not math.isfinite(field_value)
    ):
        raise ValueError("Implicit root-chart point has a nonfinite field or gradient.")
    gradient_lower = bounds.boxes(point[None, :], point[None, :]).gradient_lower[0]
    gradient_upper = bounds.boxes(point[None, :], point[None, :]).gradient_upper[0]
    gradient_norm_lower, _ = _gradient_norm_interval(gradient_lower, gradient_upper)
    if gradient_norm_lower <= 0.0:
        raise ValueError(
            "Implicit root-chart point is singular or lacks an interval gradient norm."
        )
    field_lower, field_upper = bounds.point_values(point[None, :])
    low_value = float(field_lower[0])
    high_value = float(field_upper[0])
    if low_value <= 0.0 <= high_value:
        if low_value == 0.0 and high_value == 0.0:
            return ImplicitRootChartPoint(
                np.array(point, copy=True),
                0.0,
                0.0,
                face.source_cell_key,
                face.root_edge_keys,
                "certified_root_chart",
            )
        raise ValueError(
            "Pointwise implicit field sign is unresolved at target-map resolution."
        )
    point_sign = -1 if high_value < 0.0 else 1
    scale_value = max(1.0, float(np.linalg.norm(point)))
    step = float(np.spacing(scale_value))
    if step <= 0.0 or not math.isfinite(step):
        raise ValueError(
            "Implicit normal root bracket reached coordinate representability."
        )
    maximum_step = maximum_root_parameter
    while True:
        direction = -1.0 if point_sign > 0 else 1.0
        second = point + direction * step * gradient
        if not np.all(np.isfinite(second)) or np.array_equal(second, point):
            raise ValueError(
                "Implicit normal root bracket reached coordinate representability."
            )
        second_lower, second_upper = bounds.point_values(second[None, :])
        if float(second_upper[0]) < 0.0:
            second_sign = -1
        elif float(second_lower[0]) > 0.0:
            second_sign = 1
        else:
            second_sign = 0
        if second_sign != 0 and second_sign != point_sign:
            if step > maximum_step:
                raise ValueError(
                    "Implicit root bracket exceeds its certified face-local normal tube."
                )
            root = _root_on_edge(
                bounds,
                point,
                second,
                point_sign,
                second_sign,
                surface.policy.root_tolerance,
            )
            root_point, error, issue = root[:3]
            if issue:
                label = AdaptiveImplicitBoxIssue(issue).name.lower()
                raise ValueError(
                    f"Implicit normal root isolation is unresolved ({label})."
                )
            displacement = np.nextafter(
                float(np.linalg.norm(root_point - point)) + float(error), np.inf
            )
            root_point = np.array(root_point, dtype=np.float64, copy=True)
            root_point.setflags(write=False)
            return ImplicitRootChartPoint(
                root_point,
                float(displacement),
                float(error),
                face.source_cell_key,
                face.root_edge_keys,
                "certified_root_chart",
            )
        next_step = float(np.nextafter(step * 2.0, np.inf))
        if not math.isfinite(next_step) or next_step <= step or next_step > maximum_step:
            raise ValueError(
                "Implicit root is not isolated inside the certified face-local normal tube."
            )
        step = next_step


def _nearest_target_chart_bounds(
    sample: PhysicalPoint3,
    expressions: Polynomial3,
    threshold: Fraction,
    /,
) -> tuple[Fraction, Fraction, bool]:
    """Bound one source point's distance to a complete nonlinear target triangle."""
    from ...linalg._hermitian_spectral import fraction_sqrt_interval

    if threshold < 0:
        return Fraction(0), Fraction(1 << 1024), False
    pending: list[TriangleReference] = [_STANDARD_TRIANGLE]
    lower_squared_rows: list[Fraction] = []
    center: TrianglePoint = (Fraction(1, 3), Fraction(1, 3))
    center_point: PhysicalPoint3 = (
        _evaluate_fraction_polynomial(expressions[0], center),
        _evaluate_fraction_polynomial(expressions[1], center),
        _evaluate_fraction_polynomial(expressions[2], center),
    )
    best_upper_squared: Fraction = sum(
        ((sample[axis] - center_point[axis]) ** 2 for axis in range(3)),
        Fraction(0),
    )
    unresolved = False
    center_upper = fraction_sqrt_interval(best_upper_squared)[1]
    if center_upper <= threshold:
        return Fraction(0), center_upper, True
    candidate_parameters: tuple[TrianglePoint, ...] = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1, 3), Fraction(1, 3)),
    )
    while pending:
        triangle = pending.pop()
        intervals: IntervalVector3 = (
            _poly_interval(expressions[0], triangle),
            _poly_interval(expressions[1], triangle),
            _poly_interval(expressions[2], triangle),
        )
        if any(not math.isfinite(value) for interval in intervals for value in interval):
            unresolved = True
            continue
        lower_squared = Fraction(0)
        for axis, (lower, upper) in enumerate(intervals):
            low_exact = Fraction.from_float(lower)
            high_exact = Fraction.from_float(upper)
            gap = max(Fraction(0), low_exact - sample[axis], sample[axis] - high_exact)
            lower_squared += gap * gap
        lower_distance = fraction_sqrt_interval(lower_squared)[0]
        if lower_distance > threshold:
            lower_squared_rows.append(lower_squared)
            continue
        candidate_solver = _prepare_barycentric_solver(triangle)
        for parameter in candidate_parameters:
            if not all(
                value >= 0
                for value in _barycentric(parameter, triangle, solver=candidate_solver)
            ):
                continue
            point: PhysicalPoint3 = (
                _evaluate_fraction_polynomial(
                    _restrict_polynomial(expressions[0], triangle), parameter
                ),
                _evaluate_fraction_polynomial(
                    _restrict_polynomial(expressions[1], triangle), parameter
                ),
                _evaluate_fraction_polynomial(
                    _restrict_polynomial(expressions[2], triangle), parameter
                ),
            )
            squared: Fraction = sum(
                ((sample[axis] - point[axis]) ** 2 for axis in range(3)),
                Fraction(0),
            )
            if squared < best_upper_squared:
                best_upper_squared = squared
        upper_distance = fraction_sqrt_interval(best_upper_squared)[1]
        if upper_distance <= threshold:
            return Fraction(0), upper_distance, True
        if lower_squared >= best_upper_squared:
            lower_squared_rows.append(lower_squared)
            continue
        children = _split_triangle(triangle)
        if children is None:
            unresolved = True
            lower_squared_rows.append(lower_squared)
            continue
        pending.extend(reversed(children))
    nearest_lower_squared = min(lower_squared_rows, default=Fraction(0))
    nearest_lower = fraction_sqrt_interval(nearest_lower_squared)[0]
    nearest_upper = fraction_sqrt_interval(best_upper_squared)[1]
    if nearest_lower > threshold:
        return nearest_lower, nearest_upper, True
    return nearest_lower, nearest_upper, not unresolved


def certify_adaptive_implicit_charts(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: AdaptiveImplicitSourceQuery,
    witness: SourceBoundaryChartWitness,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SourceFidelityCertificate:
    """Certify nonlinear P1/P2/P3 target maps against an adaptive implicit root cover."""
    from ..._bvh import bvh_nearest_items, prepare_bvh
    from ...geometry._source_fidelity import (
        _surface_target_chart_rows,
        _unresolved_fidelity,
        SourceFidelityCertificate,
        validate_source_fidelity_witness,
    )
    from ...linalg._hermitian_spectral import fraction_sqrt_interval

    validate_source_fidelity_witness(mesh, geometry, source, witness)
    try:
        root_map = prepare_adaptive_implicit_root_charts(
            mesh,
            geometry,
            source,
            witness,
            tolerance=tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
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
        if maximum_normal_deviation is not None:
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
    source_radius = tolerance * 0.5
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
    samples = adaptive.boundary_cover(source_radius)
    if (
        not samples.complete
        or samples.semantics != "certified"
        or samples.points.ndim != 2
        or samples.points.shape[0] == 0
        or np.any(~np.isfinite(samples.points))
        or np.any(samples.covering_radius > source_radius)
        or np.any(samples.point_error < 0.0)
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
                        detail="The current adaptive source root cover is incomplete at the requested physical radius.",
                    )
                ]
            ),
        )
    sample_points = np.asarray(samples.points, dtype=np.float64)
    sample_exact: tuple[PhysicalPoint3, ...] = tuple(
        _physical_point3(sample_points[row]) for row in range(sample_points.shape[0])
    )
    target_rows, _, _ = _surface_target_chart_rows(
        mesh, geometry, None, witness, full_source=False
    )
    target_maps: list[Polynomial3] = []
    target_map_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    target_map_centers: list[PhysicalPoint3] = []
    center_parameter: TrianglePoint = (Fraction(1, 3), Fraction(1, 3))
    for row, (block_index, cell_row) in enumerate(target_rows):
        cell_expressions = _target_cell_coordinate_expressions(
            mesh, geometry, block_index, cell_row
        )
        reference_corners = _reference_triangle(witness.target_reference_chart(row))
        mapped = _triangle_coordinate_expressions(cell_expressions, reference_corners)
        target_maps.append(mapped)
        target_map_boxes.append(_box_from_polynomials(mapped, _STANDARD_TRIANGLE))
        target_map_centers.append(
            (
                _evaluate_fraction_polynomial(mapped[0], center_parameter),
                _evaluate_fraction_polynomial(mapped[1], center_parameter),
                _evaluate_fraction_polynomial(mapped[2], center_parameter),
            )
        )
    target_bvh = prepare_bvh(
        np.asarray([box[0] for box in target_map_boxes], dtype=np.float64),
        np.asarray([box[1] for box in target_map_boxes], dtype=np.float64),
        dtype=np.float64,
    )
    source_to_mesh_lower = Fraction(0)
    source_to_mesh_upper = Fraction(0)
    for sample_row, sample in enumerate(sample_exact):
        point_error = Fraction.from_float(float(samples.point_error[sample_row]))
        cover_radius = Fraction.from_float(float(samples.covering_radius[sample_row]))
        threshold = Fraction.from_float(float(tolerance)) - point_error - cover_radius
        center_upper_squared = min(
            sum(
                ((sample[axis] - center[axis]) ** 2 for axis in range(3)),
                Fraction(0),
            )
            for center in target_map_centers
        )
        nearest_upper = fraction_sqrt_interval(center_upper_squared)[1]
        sample_float = np.asarray(
            tuple(float(value) for value in sample), dtype=np.float64
        )
        lower_rows: list[tuple[float, Fraction]] = []
        for row, (lower, upper) in enumerate(target_map_boxes):
            first_gap = np.nextafter(lower - sample_float, -np.inf)
            second_gap = np.nextafter(sample_float - upper, -np.inf)
            gap = np.maximum(np.maximum(first_gap, second_gap), 0.0)
            squared = np.nextafter(gap * gap, 0.0)
            squared_sum = max(0.0, float(np.nextafter(float(np.sum(squared)), -np.inf)))
            lower_distance = float(np.nextafter(math.sqrt(squared_sum), 0.0))
            lower_rows.append((lower_distance, Fraction.from_float(lower_distance)))
        nearest_lower = min((value for _, value in lower_rows), default=Fraction(0))
        if nearest_upper > threshold:
            candidate_count = 1
            processed: set[int] = set()
            covered = False
            while True:
                nearest = bvh_nearest_items(
                    target_bvh,
                    sample_float,
                    k=candidate_count,
                )
                candidate_rows = np.asarray(nearest.items, dtype=np.int64).reshape(-1)
                for raw_row in candidate_rows:
                    row = int(raw_row)
                    if row < 0 or row >= len(target_maps) or row in processed:
                        continue
                    processed.add(row)
                    _, exact_lower = lower_rows[row]
                    if exact_lower > threshold:
                        continue
                    lower, upper, _resolved = _nearest_target_chart_bounds(
                        sample, target_maps[row], threshold
                    )
                    nearest_lower = min(nearest_lower, lower)
                    nearest_upper = min(nearest_upper, upper)
                    if upper <= threshold:
                        nearest_lower = Fraction(0)
                        nearest_upper = upper
                        covered = True
                        break
                if covered or candidate_count >= len(target_maps):
                    break
                candidate_count = min(len(target_maps), 2 * candidate_count)
        source_to_mesh_lower = max(
            source_to_mesh_lower,
            max(Fraction(0), nearest_lower - point_error),
        )
        source_to_mesh_upper = max(
            source_to_mesh_upper,
            nearest_upper + point_error + cover_radius,
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
    if source_to_mesh_lower > Fraction.from_float(float(tolerance)):
        findings.append(
            MeshCertificateFinding(
                "source_fidelity_tolerance",
                "violated",
                "source_sample",
                detail="A certified adaptive source sample is farther than the requested tolerance from every target chart.",
            )
        )
    elif source_to_mesh_upper > Fraction.from_float(float(tolerance)):
        findings.append(
            MeshCertificateFinding(
                "source_fidelity_tolerance",
                "unresolved",
                "source_sample",
                detail="The complete adaptive source cover does not yet prove the reverse source-to-target tolerance.",
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
        float(np.nextafter(float(forward_lower), -np.inf)),
        float(np.nextafter(float(forward_upper), np.inf)),
        float(np.nextafter(float(source_to_mesh_lower), -np.inf)),
        float(np.nextafter(float(source_to_mesh_upper), np.inf)),
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
