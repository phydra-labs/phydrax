#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Complete host inverse location on source-defined mapped finite-element cells."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from fractions import Fraction
from itertools import combinations
from typing import assert_never, cast, final, Literal, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from .._bvh import PackedBVH, point_select_leaf_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..linalg._exact import determinant_exact
from ..typing import checked, ConvertibleToArray, Dim, Float
from ._cell_geometry import (
    CellGeometrySpec,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ._cell_locator import AbstractCellLocator, CellLocationResult, CellLocationStatus
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import (
    add,
    axes,
    BernsteinDomain,
    compose_expression,
    constant,
    coordinate_map_bounds,
    coordinate_map_expressions,
    CoordinateCoefficients,
    CoordinateSourceBank,
    derivative,
    evaluate_polynomial,
    ExactScalar,
    Expression,
    polynomial_bounds,
    RationalPolynomial,
    scale,
    subtract,
)
from ._exact_algebraic_geometry import coefficient_payload
from ._reference_cell import reference_cell_topology
from .fem._cell_map import PreparedFiniteElementCellMap


if TYPE_CHECKING:
    from ._view_support import OwnerLocalCellGeometry


class _MappedCoordinateDim(Dim):
    """Coordinate nodes retained by one mapped locator."""


class _MappedCellDim(Dim):
    """Source cells retained by one mapped locator."""


class _MappedAmbientDim(Dim):
    """Physical coordinate axes shared by locator arrays."""


def reference_domain(cell_kind: str, /) -> BernsteinDomain:
    topology = reference_cell_topology(cell_kind)
    if cell_kind in ("triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        return "simplex"
    if cell_kind == "prism":
        return "prism"
    if (
        topology.dimension == 1
        or cell_kind in ("quadrilateral", "hexahedron")
        or cell_kind.startswith("tensor:")
        or cell_kind == "pyramid"
    ):
        return "box"
    raise ValueError(f"Mapped location does not support reference cell {cell_kind!r}.")


def reference_margin(cell_kind: str, reference: np.ndarray, /) -> float:
    if cell_kind in ("triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        return min(float(np.min(reference)), 1.0 - float(np.sum(reference)))
    if cell_kind == "prism":
        return min(
            float(np.min(reference[:2])),
            1.0 - float(np.sum(reference[:2])),
            float(reference[2]),
            1.0 - float(reference[2]),
        )
    if cell_kind == "pyramid":
        height = float(reference[2])
        lower = 0.5 * height
        upper = 1.0 - lower
        return min(
            height,
            1.0 - height,
            float(np.min(reference[:2] - lower)),
            float(np.min(upper - reference[:2])),
        )
    return float(np.min(np.minimum(reference, 1.0 - reference)))


@final
class MappedLocationPolicy(StrictModule, NonTrainableState):
    """Physical/reference tolerances and finite numerical resolution for inverse maps."""

    residual_tolerance: float = eqx.field(static=True)
    reference_tolerance: float = eqx.field(static=True)
    parameter_resolution: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        residual_tolerance: float = 1.0e-10,
        reference_tolerance: float = 1.0e-10,
        parameter_resolution: float = 1.0e-10,
    ) -> None:
        values = (
            residual_tolerance,
            reference_tolerance,
            parameter_resolution,
        )
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise TypeError(
                "Mapped location tolerances must be real scalars excluding bool."
            )
        residual, reference, resolution = (float(value) for value in values)
        if (
            not np.all(np.isfinite((residual, reference, resolution)))
            or residual <= 0.0
            or reference < 0.0
            or resolution <= 0.0
        ):
            raise ValueError(
                "Mapped location residual/resolution tolerances must be positive and reference tolerance non-negative."
            )
        self.residual_tolerance = residual
        self.reference_tolerance = reference
        self.parameter_resolution = resolution
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mapped-location-policy",
                "residual_tolerance": residual,
                "reference_tolerance": reference,
                "parameter_resolution": resolution,
            }
        )


def _reference_weights(cell_kind: str, reference: np.ndarray, /) -> np.ndarray:
    vertices = np.asarray(reference_cell_topology(cell_kind).vertices, dtype=np.float64)
    if cell_kind in ("triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        return np.concatenate(([1.0 - np.sum(reference)], reference))
    if cell_kind == "prism":
        base = np.asarray((1.0 - np.sum(reference[:2]), *reference[:2]))
        return np.concatenate((base * (1.0 - reference[2]), base * reference[2]))
    if cell_kind == "pyramid":
        height = reference[2]
        collapse = 1.0 - height
        chart = (reference[:2] - 0.5 * height) / collapse if collapse else np.zeros((2,))
        base = (
            np.prod(np.where(vertices[:4] == 1.0, chart[None], 1.0 - chart[None]), axis=1)
            * collapse
        )
        return np.concatenate((base, np.asarray((height,))))
    return np.prod(
        np.where(vertices == 1.0, reference[None], 1.0 - reference[None]), axis=1
    )


def _polynomial_value(
    expression: Mapping[tuple[int, ...], ExactScalar], point: np.ndarray, /
) -> float:
    total = 0.0
    for exponent, coefficient in expression.items():
        term = float(coefficient)
        for coordinate, power in zip(point, exponent, strict=True):
            if power:
                term *= float(coordinate) ** power
        total += term
    return total


def _expression_value(expression: Expression, point: np.ndarray, /) -> float:
    if not isinstance(expression, RationalPolynomial):
        return _polynomial_value(expression, point)
    numerator = _polynomial_value(expression.numerator, point)
    denominator = _polynomial_value(expression.denominator, point)
    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")
    return numerator / denominator


def _map_values(
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    point: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    physical = np.asarray(
        [_expression_value(value, point) for value, _ in expressions],
        dtype=np.float64,
    )
    jacobian = np.asarray(
        [
            [_expression_value(gradient, point) for gradient in gradients]
            for _, gradients in expressions
        ],
        dtype=np.float64,
    )
    return physical, jacobian


def _cell_expressions(
    element: ScalarCoordinateElement,
    local: CoordinateCoefficients,
    cell_row: int | None,
    dimension: int,
    /,
) -> tuple[tuple[Expression, tuple[Expression, ...]], ...]:
    values = coordinate_map_expressions(element, local, cell_row=cell_row)
    return tuple(
        (value, tuple(derivative(value, axis) for axis in range(dimension)))
        for value in values
    )


type _InverseBox = tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
type _Interval = tuple[float, float]


class _InverseRoot(NamedTuple):
    reference: np.ndarray
    residual: float
    steps: int
    condition: float
    fallback: bool
    enclosure_lower: tuple[Fraction, ...]
    enclosure_upper: tuple[Fraction, ...]
    selected_rows: tuple[int, ...]


def _interval_add(first: _Interval, second: _Interval, /) -> _Interval:
    return (
        float(np.nextafter(first[0] + second[0], -np.inf)),
        float(np.nextafter(first[1] + second[1], np.inf)),
    )


def _interval_subtract(first: _Interval, second: _Interval, /) -> _Interval:
    return _interval_add(first, (-second[1], -second[0]))


def _interval_multiply(first: _Interval, second: _Interval, /) -> _Interval:
    products = (
        first[0] * second[0],
        first[0] * second[1],
        first[1] * second[0],
        first[1] * second[1],
    )
    if any(np.isnan(value) for value in products):
        return -np.inf, np.inf
    return (
        float(np.nextafter(min(products), -np.inf)),
        float(np.nextafter(max(products), np.inf)),
    )


def _interval_residual_lower_bound(bounds: tuple[_Interval, ...], /) -> float:
    squared_lower = 0.0
    for lower, upper in bounds:
        if lower <= 0.0 <= upper:
            continue
        distance = min(abs(lower), abs(upper))
        square_lower = float(np.nextafter(distance * distance, -np.inf))
        squared_lower = float(np.nextafter(squared_lower + square_lower, -np.inf))
    return float(np.nextafter(np.sqrt(max(squared_lower, 0.0)), -np.inf))


def _box_arguments(
    lower: tuple[Fraction, ...], upper: tuple[Fraction, ...], /
) -> tuple[Expression, ...]:
    dimension = len(lower)
    variables = axes(dimension)
    return tuple(
        add(
            constant(lower[axis], dimension),
            scale(variables[axis], upper[axis] - lower[axis]),
        )
        for axis in range(dimension)
    )


def _box_expression_bounds(
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    target: np.ndarray,
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    /,
) -> (
    tuple[
        tuple[_Interval, ...],
        tuple[tuple[_Interval, ...], ...],
    ]
    | None
):
    dimension = len(lower)
    arguments = _box_arguments(lower, upper)
    target_exact = tuple(Fraction(float(value)) for value in target)
    function_bounds: list[_Interval] = []
    jacobian_bounds: list[tuple[_Interval, ...]] = []
    try:
        for axis, (value, gradients) in enumerate(expressions):
            local_value = compose_expression(value, arguments)
            local_equation = subtract(
                local_value, constant(target_exact[axis], dimension)
            )
            bounds = polynomial_bounds(local_equation, "box", dimension)
            if not np.all(np.isfinite(bounds)):
                return None
            function_bounds.append((float(bounds[0]), float(bounds[1])))
            row = []
            for gradient in gradients:
                local_gradient = compose_expression(gradient, arguments)
                derivative_bounds = polynomial_bounds(local_gradient, "box", dimension)
                if not np.all(np.isfinite(derivative_bounds)):
                    return None
                row.append((float(derivative_bounds[0]), float(derivative_bounds[1])))
            jacobian_bounds.append(tuple(row))
    except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
        return None
    return tuple(function_bounds), tuple(jacobian_bounds)


def _reference_margin_bounds(
    cell_kind: str,
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    /,
) -> tuple[Fraction, Fraction]:
    constraints: list[tuple[Fraction, Fraction]] = []
    match reference_domain(cell_kind):
        case "simplex":
            constraints.extend(zip(lower, upper, strict=True))
            constraints.append(
                (
                    Fraction(1) - sum(upper, Fraction(0)),
                    Fraction(1) - sum(lower, Fraction(0)),
                )
            )
        case "prism":
            constraints.extend(zip(lower[:2], upper[:2], strict=True))
            constraints.append(
                (
                    Fraction(1) - sum(upper[:2], Fraction(0)),
                    Fraction(1) - sum(lower[:2], Fraction(0)),
                )
            )
            constraints.extend(((lower[2], upper[2]), (1 - upper[2], 1 - lower[2])))
        case "box":
            for first, last in zip(lower, upper, strict=True):
                constraints.extend(((first, last), (1 - last, 1 - first)))
            if cell_kind == "pyramid":
                z_lower, z_upper = lower[2], upper[2]
                for coordinate in (0, 1):
                    x_lower, x_upper = lower[coordinate], upper[coordinate]
                    constraints.append((x_lower - z_upper / 2, x_upper - z_lower / 2))
                    constraints.append(
                        (
                            1 - z_upper / 2 - x_upper,
                            1 - z_lower / 2 - x_lower,
                        )
                    )
        case _:
            raise ValueError(f"Unsupported mapped reference domain for {cell_kind!r}.")
    return min(value[0] for value in constraints), min(value[1] for value in constraints)


def _exact_determinant_nonzero(matrix: np.ndarray, /) -> bool:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if np.any(~np.isfinite(matrix)):
        return False
    exact = tuple(
        tuple(Fraction(float(matrix[row, column])) for column in range(matrix.shape[0]))
        for row in range(matrix.shape[0])
    )
    return determinant_exact(exact) != 0


class _KrawczykOutcome(NamedTuple):
    """One Krawczyk decision with the interval Jacobian it was decided on.

    ``rootless`` certifies that the box holds no exact inverse root while its
    residual enclosure still reaches the physical tolerance.
    """

    status: Literal["excluded", "isolated", "rootless", "unresolved"]
    lower: tuple[Fraction, ...] | None
    upper: tuple[Fraction, ...] | None
    rows: tuple[int, ...] | None
    jacobian: tuple[tuple[_Interval, ...], ...] | None


def _krawczyk_box(
    target: np.ndarray,
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    policy: MappedLocationPolicy,
    /,
    *,
    selected_rows: tuple[int, ...] | None = None,
) -> _KrawczykOutcome:
    dimension = len(lower)
    ambient_dimension = len(expressions)
    bounds = _box_expression_bounds(expressions, target, lower, upper)
    if bounds is None:
        return _KrawczykOutcome("unresolved", None, None, None, None)
    function_bounds, jacobian_bounds = bounds
    residual = policy.residual_tolerance
    if any(value[0] > residual or value[1] < -residual for value in function_bounds):
        return _KrawczykOutcome("excluded", None, None, None, jacobian_bounds)
    center = np.asarray(
        [float((first + last) / 2) for first, last in zip(lower, upper, strict=True)],
        dtype=np.float64,
    )
    center_exact = tuple(Fraction.from_float(float(value)) for value in center)
    _, center_jacobian = _map_values(expressions, center)
    if not np.all(np.isfinite(center_jacobian)):
        return _KrawczykOutcome("unresolved", None, None, None, jacobian_bounds)
    row_choices = (
        (selected_rows,)
        if selected_rows is not None
        else tuple(combinations(range(ambient_dimension), dimension))
    )
    point_bounds = _box_expression_bounds(expressions, target, center_exact, center_exact)
    if point_bounds is None:
        return _KrawczykOutcome("unresolved", None, None, None, jacobian_bounds)
    point_functions, _ = point_bounds
    for rows in row_choices:
        if len(rows) != dimension:
            continue
        jacobian = center_jacobian[np.asarray(rows, dtype=np.int64)]
        try:
            preconditioner = np.linalg.solve(jacobian, np.eye(dimension))
        except np.linalg.LinAlgError:
            continue
        if not _exact_determinant_nonzero(preconditioner):
            continue
        krawczyk_matrix: list[list[_Interval]] = []
        for row in range(dimension):
            matrix_row = []
            for column in range(dimension):
                product_interval: _Interval = (0.0, 0.0)
                for selected_index, physical_row in enumerate(rows):
                    product_interval = _interval_add(
                        product_interval,
                        _interval_multiply(
                            (
                                float(preconditioner[row, selected_index]),
                                float(preconditioner[row, selected_index]),
                            ),
                            jacobian_bounds[physical_row][column],
                        ),
                    )
                identity: _Interval = (1.0, 1.0) if row == column else (0.0, 0.0)
                matrix_row.append(_interval_subtract(identity, product_interval))
            krawczyk_matrix.append(matrix_row)
        contraction = max(
            float(
                np.nextafter(
                    sum(max(abs(interval[0]), abs(interval[1])) for interval in row),
                    np.inf,
                )
            )
            for row in krawczyk_matrix
        )
        center_image = []
        for row in range(dimension):
            correction: _Interval = (0.0, 0.0)
            for selected_index, physical_row in enumerate(rows):
                correction = _interval_add(
                    correction,
                    _interval_multiply(
                        (
                            float(preconditioner[row, selected_index]),
                            float(preconditioner[row, selected_index]),
                        ),
                        point_functions[physical_row],
                    ),
                )
            center_value = float(center[row])
            center_image.append(
                _interval_subtract((center_value, center_value), correction)
            )
        image = []
        for row in range(dimension):
            total = center_image[row]
            for column in range(dimension):
                radius = _interval_subtract(
                    (float(lower[column]), float(upper[column])),
                    (float(center[column]), float(center[column])),
                )
                total = _interval_add(
                    total,
                    _interval_multiply(krawczyk_matrix[row][column], radius),
                )
            image.append(total)
        if any(
            interval[1] < float(lower[index]) or interval[0] > float(upper[index])
            for index, interval in enumerate(image)
        ):
            if _interval_residual_lower_bound(function_bounds) > residual:
                return _KrawczykOutcome("excluded", None, None, None, jacobian_bounds)
            return _KrawczykOutcome("rootless", None, None, None, jacobian_bounds)
        contained = all(
            interval[0] >= float(lower[index]) and interval[1] <= float(upper[index])
            for index, interval in enumerate(image)
        )
        if contained and contraction < 1.0:
            image_lower = tuple(
                max(lower[index], Fraction.from_float(interval[0]))
                for index, interval in enumerate(image)
            )
            image_upper = tuple(
                min(upper[index], Fraction.from_float(interval[1]))
                for index, interval in enumerate(image)
            )
            return _KrawczykOutcome(
                "isolated", image_lower, image_upper, tuple(rows), jacobian_bounds
            )
    return _KrawczykOutcome("unresolved", None, None, None, jacobian_bounds)


def _split_reference_box(
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    policy: MappedLocationPolicy,
    /,
) -> tuple[_InverseBox, _InverseBox] | None:
    widths = tuple(float(last - first) for first, last in zip(lower, upper, strict=True))
    axis = int(np.argmax(np.asarray(widths, dtype=np.float64)))
    if widths[axis] <= policy.parameter_resolution:
        return None
    first, last = float(lower[axis]), float(upper[axis])
    middle = first + 0.5 * (last - first)
    if not first < middle < last:
        middle = float(np.nextafter(first, last))
    if not first < middle < last:
        return None
    split = Fraction.from_float(middle)
    if not lower[axis] < split < upper[axis]:
        return None
    left_upper = list(upper)
    left_upper[axis] = split
    right_lower = list(lower)
    right_lower[axis] = split
    return (lower, tuple(left_upper)), (tuple(right_lower), upper)


class _InverseRefusal(Enum):
    """Why a mapped inverse search left part of the reference cell undecided."""

    SINGULAR_TOLERANCE_ROOT = "singular-tolerance-root"
    ROOTLESS_TOLERANCE_ROOT = "rootless-tolerance-root"
    PARAMETER_RESOLUTION = "parameter-resolution"
    UNBOUNDED_ENCLOSURE = "unbounded-enclosure"
    ROOT_REFINEMENT_STALLED = "root-refinement-stalled"
    DUPLICATE_ROOT_UNMERGED = "duplicate-root-unmerged"


class _UnresolvedInverseBox(NamedTuple):
    """Reference-box evidence for one undecided inverse-map region.

    ``witness`` is empty, one exact tolerance root, or the two ends of a segment
    of tolerance roots across which the Jacobian determinant changes sign.
    """

    refusal: _InverseRefusal
    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]
    witness: tuple[tuple[Fraction, ...], ...]


class _InverseIsolation(NamedTuple):
    """Certified inverse roots of one cell and every region left undecided."""

    roots: tuple[_InverseRoot, ...]
    unresolved: tuple[_UnresolvedInverseBox, ...]
    ambiguous: bool


def _exact_expression_value(
    expression: Expression, point: tuple[Fraction, ...], /
) -> ExactScalar | None:
    if not isinstance(expression, RationalPolynomial):
        return evaluate_polynomial(expression, point)
    denominator = evaluate_polynomial(expression.denominator, point)
    if not denominator:
        return None
    return evaluate_polynomial(expression.numerator, point) / denominator


def _is_tolerance_root(
    target_exact: tuple[Fraction, ...],
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    cell_kind: str,
    point: tuple[Fraction, ...],
    policy: MappedLocationPolicy,
    /,
) -> bool:
    """Exactly decide that ``point`` lies in the reference margin within residual tolerance."""
    margin, _ = _reference_margin_bounds(cell_kind, point, point)
    if margin < -Fraction.from_float(policy.reference_tolerance):
        return False
    squared: ExactScalar = Fraction(0)
    for (value, _), coordinate in zip(expressions, target_exact, strict=True):
        image = _exact_expression_value(value, point)
        if image is None:
            return False
        difference = image - coordinate
        squared = squared + difference * difference
    tolerance = Fraction.from_float(policy.residual_tolerance)
    return squared <= tolerance * tolerance


def _exact_determinant_sign(
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    point: tuple[Fraction, ...],
    /,
) -> int | None:
    rows: list[tuple[ExactScalar, ...]] = []
    for _, gradients in expressions:
        row: list[ExactScalar] = []
        for gradient in gradients:
            value = _exact_expression_value(gradient, point)
            if value is None:
                return None
            row.append(value)
        rows.append(tuple(row))
    determinant = determinant_exact(tuple(rows))
    zero = Fraction(0)
    if determinant > zero:
        return 1
    if determinant < zero:
        return -1
    return 0


def _interval_determinant(matrix: tuple[tuple[_Interval, ...], ...], /) -> _Interval:
    if len(matrix) == 1:
        return matrix[0][0]
    total: _Interval = (0.0, 0.0)
    for column, entry in enumerate(matrix[0]):
        minor = tuple(row[:column] + row[column + 1 :] for row in matrix[1:])
        term = _interval_multiply(entry, _interval_determinant(minor))
        total = (
            _interval_add(total, term)
            if column % 2 == 0
            else _interval_subtract(total, term)
        )
    return total


def _clamped_exact_point(
    point: np.ndarray, lower: tuple[Fraction, ...], upper: tuple[Fraction, ...], /
) -> tuple[Fraction, ...]:
    return tuple(
        min(max(Fraction.from_float(float(value)), first), last)
        for value, first, last in zip(point, lower, upper, strict=True)
    )


def _tolerance_root_witness(
    target: np.ndarray,
    target_exact: tuple[Fraction, ...],
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    cell_kind: str,
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    policy: MappedLocationPolicy,
    /,
) -> tuple[Fraction, ...] | None:
    """Return an exact tolerance root inside the box, or ``None`` when none is certified.

    Box-projected Gauss-Newton proposes the point in binary64 while its residual
    strictly decreases; only the exact check of the final iterate certifies it.
    """
    lower_float = np.asarray([float(value) for value in lower], dtype=np.float64)
    upper_float = np.asarray([float(value) for value in upper], dtype=np.float64)
    point = np.clip(0.5 * (lower_float + upper_float), lower_float, upper_float)
    physical, jacobian = _map_values(expressions, point)
    residual = float(np.linalg.norm(physical - target))
    while np.isfinite(residual) and np.all(np.isfinite(jacobian)):
        step = np.linalg.lstsq(jacobian, physical - target, rcond=None)[0]
        candidate = np.clip(point - step, lower_float, upper_float)
        candidate_physical, candidate_jacobian = _map_values(expressions, candidate)
        candidate_residual = float(np.linalg.norm(candidate_physical - target))
        if not candidate_residual < residual:
            break
        point, physical, jacobian = candidate, candidate_physical, candidate_jacobian
        residual = candidate_residual
    witness = _clamped_exact_point(point, lower, upper)
    if _is_tolerance_root(target_exact, expressions, cell_kind, witness, policy):
        return witness
    return None


def _segment_residual_upper_bound(
    target_exact: tuple[Fraction, ...],
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    first: tuple[Fraction, ...],
    last: tuple[Fraction, ...],
    /,
) -> float:
    """Bound the physical residual norm over a whole reference segment from above."""
    variable = axes(1)[0]
    arguments = tuple(
        add(constant(start, 1), scale(variable, end - start))
        for start, end in zip(first, last, strict=True)
    )
    squared = 0.0
    for (value, _), coordinate in zip(expressions, target_exact, strict=True):
        local = subtract(compose_expression(value, arguments), constant(coordinate, 1))
        bound_lower, bound_upper = polynomial_bounds(local, "box", 1)
        magnitude = max(abs(bound_lower), abs(bound_upper))
        squared = float(
            np.nextafter(
                squared + float(np.nextafter(magnitude * magnitude, np.inf)), np.inf
            )
        )
    return float(np.nextafter(np.sqrt(squared), np.inf))


def _singular_tolerance_segment(
    target_exact: tuple[Fraction, ...],
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    cell_kind: str,
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    witness: tuple[Fraction, ...],
    policy: MappedLocationPolicy,
    /,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    """Bracket a tolerance root with a singular Jacobian by an exact determinant sign change.

    Every point of the returned segment is a tolerance root (an enclosure bound of
    the whole segment) and the determinant vanishes or changes sign between its
    exact ends, so by continuity the segment holds a tolerance root with a singular
    Jacobian. The bracket walks the binary64 null direction of the Jacobian at the
    witness with doubling steps until the sign changes, an end leaves the
    tolerance, or both ends are pinned to the box.
    """
    sign = _exact_determinant_sign(expressions, witness)
    if sign is None:
        return None
    if sign == 0:
        return witness, witness
    base = np.asarray([float(value) for value in witness], dtype=np.float64)
    _, jacobian = _map_values(expressions, base)
    if not np.all(np.isfinite(jacobian)):
        return None
    direction = np.linalg.svd(jacobian)[2][-1]
    lower_float = np.asarray([float(value) for value in lower], dtype=np.float64)
    upper_float = np.asarray([float(value) for value in upper], dtype=np.float64)
    step = float(np.finfo(np.float64).eps)
    previous: tuple[tuple[Fraction, ...], ...] = ()
    while True:
        ends = tuple(
            _clamped_exact_point(
                np.clip(base + orientation * step * direction, lower_float, upper_float),
                lower,
                upper,
            )
            for orientation in (-1.0, 1.0)
        )
        if ends == previous:
            return None
        previous = ends
        for end in ends:
            if not _is_tolerance_root(target_exact, expressions, cell_kind, end, policy):
                return None
            end_sign = _exact_determinant_sign(expressions, end)
            if end_sign is None:
                return None
            if end_sign != sign:
                bound = _segment_residual_upper_bound(
                    target_exact, expressions, witness, end
                )
                return (witness, end) if bound <= policy.residual_tolerance else None
        step *= 2.0


def _tolerance_refusal(
    target: np.ndarray,
    target_exact: tuple[Fraction, ...],
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    cell_kind: str,
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    outcome: _KrawczykOutcome,
    policy: MappedLocationPolicy,
    /,
) -> _UnresolvedInverseBox | None:
    """Refuse an undecided box now when bisection provably cannot decide it.

    A tolerance root inside the reference margin keeps every box containing it
    from exclusion. Such a box is never isolated either when it is certified
    rootless, or when the tolerance root has a singular Jacobian: a Krawczyk
    contraction below one makes every enclosed Jacobian invertible. Bisection
    would refine the region to the parameter resolution only to refuse there.
    A non-square map has no determinant sign certificate for rank deficiency and
    keeps bisecting.
    """
    if outcome.status == "rootless":
        witness = _tolerance_root_witness(
            target, target_exact, expressions, cell_kind, lower, upper, policy
        )
        if witness is None:
            return None
        return _UnresolvedInverseBox(
            _InverseRefusal.ROOTLESS_TOLERANCE_ROOT, lower, upper, (witness,)
        )
    jacobian = outcome.jacobian
    if jacobian is None or len(jacobian) != len(lower):
        return None
    determinant = _interval_determinant(jacobian)
    if not determinant[0] <= 0.0 <= determinant[1]:
        return None
    witness = _tolerance_root_witness(
        target, target_exact, expressions, cell_kind, lower, upper, policy
    )
    if witness is None:
        return None
    segment = _singular_tolerance_segment(
        target_exact, expressions, cell_kind, lower, upper, witness, policy
    )
    if segment is None:
        return None
    return _UnresolvedInverseBox(
        _InverseRefusal.SINGULAR_TOLERANCE_ROOT, lower, upper, segment
    )


def _refine_isolated_root(
    target: np.ndarray,
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    cell_kind: str,
    policy: MappedLocationPolicy,
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    rows: tuple[int, ...],
    depth: int,
    /,
) -> _InverseRoot | _UnresolvedInverseBox | None:
    """Contract one isolated root to the parameter resolution.

    ``None`` means the unique root of the box lies outside the reference cell or
    off the non-selected physical rows.
    """
    reference_tolerance = Fraction.from_float(policy.reference_tolerance)
    nonselected = tuple(row for row in range(len(expressions)) if row not in rows)
    refinement_steps = 1
    while True:
        root_bounds = _box_expression_bounds(expressions, target, lower, upper)
        if root_bounds is None:
            return _UnresolvedInverseBox(
                _InverseRefusal.UNBOUNDED_ENCLOSURE, lower, upper, ()
            )
        residual_bounds, _ = root_bounds
        margin_lower, margin_upper = _reference_margin_bounds(cell_kind, lower, upper)
        margin_inside = margin_lower >= -reference_tolerance
        margin_outside = margin_upper < -reference_tolerance
        residual_outside = any(
            residual_bounds[row][0] > policy.residual_tolerance
            or residual_bounds[row][1] < -policy.residual_tolerance
            for row in nonselected
        )
        residual_inside = all(
            residual_bounds[row][0] >= -policy.residual_tolerance
            and residual_bounds[row][1] <= policy.residual_tolerance
            for row in nonselected
        )
        width = max(float(last - first) for first, last in zip(lower, upper, strict=True))
        reference = np.asarray(
            [float((first + last) / 2) for first, last in zip(lower, upper, strict=True)],
            dtype=np.float64,
        )
        physical, jacobian = _map_values(expressions, reference)
        residual = float(np.linalg.norm(physical - target))
        if (
            margin_inside
            and residual_inside
            and (
                width <= policy.parameter_resolution
                and np.isfinite(residual)
                and residual <= policy.residual_tolerance
            )
        ):
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            condition = (
                float(singular_values[0] / singular_values[-1])
                if singular_values.size
                and singular_values[-1] > np.finfo(np.float64).tiny
                else float("inf")
            )
            return _InverseRoot(
                reference,
                residual,
                depth + refinement_steps,
                condition,
                rows != tuple(range(len(lower))),
                lower,
                upper,
                rows,
            )
        if margin_outside or residual_outside:
            return None
        if width <= policy.parameter_resolution:
            return _UnresolvedInverseBox(
                _InverseRefusal.PARAMETER_RESOLUTION, lower, upper, ()
            )
        refined = _krawczyk_box(
            target, expressions, lower, upper, policy, selected_rows=rows
        )
        if (
            refined.status != "isolated"
            or refined.lower is None
            or refined.upper is None
            or refined.rows != rows
            or (refined.lower == lower and refined.upper == upper)
        ):
            return _UnresolvedInverseBox(
                _InverseRefusal.ROOT_REFINEMENT_STALLED, lower, upper, ()
            )
        lower, upper = refined.lower, refined.upper
        refinement_steps += 1


def _merge_duplicate_roots(
    target: np.ndarray,
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    policy: MappedLocationPolicy,
    roots: list[_InverseRoot],
    /,
) -> tuple[list[_InverseRoot], list[_UnresolvedInverseBox]]:
    """Merge roots found from adjacent boxes when one Krawczyk hull isolates them."""
    dimension = len(expressions[0][1])
    ordered = sorted(
        roots, key=lambda root: tuple(float(value) for value in root.reference)
    )
    unique_roots: list[_InverseRoot] = []
    unresolved: list[_UnresolvedInverseBox] = []
    duplicate_resolution = 4.0 * np.sqrt(float(dimension)) * policy.parameter_resolution
    for root in ordered:
        duplicate_index = next(
            (
                index
                for index, previous in enumerate(unique_roots)
                if float(np.linalg.norm(root.reference - previous.reference))
                <= duplicate_resolution
            ),
            None,
        )
        if duplicate_index is None:
            unique_roots.append(root)
            continue
        previous = unique_roots[duplicate_index]
        hull_lower = tuple(
            min(first, second)
            for first, second in zip(
                root.enclosure_lower, previous.enclosure_lower, strict=True
            )
        )
        hull_upper = tuple(
            max(first, second)
            for first, second in zip(
                root.enclosure_upper, previous.enclosure_upper, strict=True
            )
        )
        if root.selected_rows != previous.selected_rows:
            unresolved.append(
                _UnresolvedInverseBox(
                    _InverseRefusal.DUPLICATE_ROOT_UNMERGED, hull_lower, hull_upper, ()
                )
            )
            continue
        merged = _krawczyk_box(
            target,
            expressions,
            hull_lower,
            hull_upper,
            policy,
            selected_rows=root.selected_rows,
        )
        if (
            merged.status != "isolated"
            or merged.lower is None
            or merged.upper is None
            or merged.rows is None
            or merged.rows != root.selected_rows
        ):
            unresolved.append(
                _UnresolvedInverseBox(
                    _InverseRefusal.DUPLICATE_ROOT_UNMERGED, hull_lower, hull_upper, ()
                )
            )
            continue
        chosen = root if root.residual < previous.residual else previous
        unique_roots[duplicate_index] = _InverseRoot(
            chosen.reference,
            chosen.residual,
            chosen.steps,
            chosen.condition,
            chosen.fallback,
            merged.lower,
            merged.upper,
            merged.rows,
        )
    return unique_roots, unresolved


def _isolate_inverse_roots(
    target: np.ndarray,
    expressions: tuple[tuple[Expression, tuple[Expression, ...]], ...],
    cell_kind: str,
    policy: MappedLocationPolicy,
    /,
) -> _InverseIsolation:
    """Isolate every exact inverse root of one cell or refuse with box evidence."""
    dimension = len(expressions[0][1])
    target_exact = tuple(Fraction.from_float(float(value)) for value in target)
    padding = max(policy.reference_tolerance, policy.parameter_resolution)
    lower = tuple(Fraction.from_float(-padding) for _ in range(dimension))
    upper = tuple(Fraction.from_float(1.0 + padding) for _ in range(dimension))
    reference_tolerance = Fraction.from_float(policy.reference_tolerance)
    stack: list[tuple[_InverseBox, int]] = [((lower, upper), 0)]
    roots: list[_InverseRoot] = []
    unresolved: list[_UnresolvedInverseBox] = []
    while stack:
        (lower, upper), depth = stack.pop()
        _, margin_upper = _reference_margin_bounds(cell_kind, lower, upper)
        if margin_upper < -reference_tolerance:
            continue
        outcome = _krawczyk_box(target, expressions, lower, upper, policy)
        match outcome.status:
            case "excluded":
                continue
            case "isolated":
                if outcome.lower is None or outcome.upper is None or outcome.rows is None:
                    raise RuntimeError("Krawczyk isolation returned no root enclosure.")
                refined = _refine_isolated_root(
                    target,
                    expressions,
                    cell_kind,
                    policy,
                    outcome.lower,
                    outcome.upper,
                    outcome.rows,
                    depth,
                )
                if isinstance(refined, _InverseRoot):
                    roots.append(refined)
                elif refined is not None:
                    unresolved.append(refined)
                continue
            case "rootless" | "unresolved":
                refusal = _tolerance_refusal(
                    target,
                    target_exact,
                    expressions,
                    cell_kind,
                    lower,
                    upper,
                    outcome,
                    policy,
                )
                if refusal is not None:
                    unresolved.append(refusal)
                    continue
            case _:
                assert_never(outcome.status)
        split = _split_reference_box(lower, upper, policy)
        if split is None:
            unresolved.append(
                _UnresolvedInverseBox(
                    _InverseRefusal.PARAMETER_RESOLUTION, lower, upper, ()
                )
            )
            continue
        left, right = split
        stack.extend(((right, depth + 1), (left, depth + 1)))
    unique_roots, unmerged = _merge_duplicate_roots(target, expressions, policy, roots)
    unresolved.extend(unmerged)
    return _InverseIsolation(
        tuple(unique_roots),
        tuple(unresolved),
        len(unique_roots) > 1 and not unresolved,
    )


@final
class PreparedMappedCellMap(StrictModule, NonTrainableState):
    """One source-defined coordinate block with its complete reference map."""

    coordinate_element: ScalarCoordinateElement
    coordinate_dofs: Array
    cell_count: int = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    reference_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    block_name: str = eqx.field(static=True)
    block_index: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    cell_map_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        block_index: int,
        /,
    ) -> None:
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError("mesh and geometry must be CellMesh and CellGeometrySpec.")
        if isinstance(block_index, bool) or not isinstance(block_index, int):
            raise TypeError("block_index must be an integer excluding bool.")
        if block_index < 0 or block_index >= len(mesh.blocks):
            raise IndexError("block_index is outside the mapped cell mesh.")
        elements, routes, coordinates = geometry.resolve(mesh)
        block = mesh.blocks[block_index]
        element = elements[block_index]
        if not isinstance(element, ScalarCoordinateElement):
            raise TypeError(
                "Mapped cell maps require a supported scalar coordinate element."
            )
        route = np.asarray(routes[block_index])
        if (
            route.ndim != 2
            or route.shape[0] != block.cell_count
            or route.dtype.kind not in "iu"
            or element.cell_kind != block.cell_kind
            or coordinates.ndim != 2
            or coordinates.shape[1] != mesh.ambient_dimension
            or np.any(route < 0)
            or np.any(route >= coordinates.shape[0])
        ):
            raise ValueError(
                "Mapped coordinate block does not match its mesh realization."
            )
        self.coordinate_element = element
        self.coordinate_dofs = jnp.asarray(route)
        self.cell_count = block.cell_count
        self.coordinate_count = coordinates.shape[0]
        self.reference_dimension = block.topological_dimension
        self.ambient_dimension = mesh.ambient_dimension
        self.block_name = block.name
        self.block_index = block_index
        self.topology_id = mesh.topology_id
        self.geometry_layout_id = geometry.geometry_layout_id
        self.cell_map_id = canonical_fingerprint(
            {
                "kind": "prepared-source-mapped-cell-map",
                "topology": mesh.topology_id,
                "block": block.block_id,
                "geometry": geometry.geometry_layout_id,
                "element": element.element_id,
                "routes": array_tree_fingerprint(np.asarray(route)),
            }
        )


@final
class PreparedOwnerLocalMappedCellMap(StrictModule, NonTrainableState):
    """Canonical mapped-cell map over one compact source-bound owner view."""

    coordinate_element: ScalarCoordinateElement
    coordinate_dofs: Array
    coordinates: Array
    cell_global_ids: Array
    cell_count: int = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    reference_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    block_name: str = eqx.field(static=True)
    block_index: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    source_map_id: str | None = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_context_id: str | None = eqx.field(static=True)
    periodic_lift_id: str | None = eqx.field(static=True)
    coordinates_are_lifted: bool = eqx.field(static=True)
    cell_map_id: str = eqx.field(static=True)
    exact_coordinates: CoordinateSourceBank = eqx.field(static=True)
    local_view_id: str = eqx.field(static=True)

    def __init__(self, view: OwnerLocalCellGeometry, block_name: str, /) -> None:
        from ._view_support import OwnerLocalCellGeometry

        if not isinstance(view, OwnerLocalCellGeometry):
            raise TypeError("view must be a canonical OwnerLocalCellGeometry.")
        name = canonical_identifier(block_name, "block_name")
        matches = tuple(
            (index, block)
            for index, block in enumerate(view.blocks)
            if block.name == name
        )
        if len(matches) != 1:
            raise ValueError(
                "Owner-local mapped block name must identify one source block."
            )
        block_index, block = matches[0]
        element = block.element
        if not isinstance(element, ScalarCoordinateElement):
            raise TypeError(
                "Owner-local mapped blocks require a scalar coordinate element."
            )
        base_routes = np.asarray(block.coordinate_routes, dtype=np.int32)
        identifiers = np.asarray(block.cell_ids, dtype=np.int64)
        base_coordinates = np.asarray(view.coordinates, dtype=np.float64)
        base_exact = tuple(
            tuple(value for value in row) for row in view.exact_coordinates
        )
        if (
            base_routes.ndim != 2
            or base_routes.shape != (identifiers.size, element.local_dof_count)
            or identifiers.ndim != 1
            or np.any(base_routes < 0)
            or np.any(base_routes >= base_coordinates.shape[0])
            or element.cell_kind != block.cell_kind
        ):
            raise ValueError(
                "Owner-local mapped cell routes do not match their source block."
            )
        if isinstance(element, PerCellRestrictedCellGeometryElement) and (
            element.target_cell_ids != tuple(int(value) for value in identifiers)
        ):
            raise ValueError(
                "Owner-local restricted charts must retain their stable target IDs."
            )
        if not np.array_equal(np.asarray(base_exact, dtype=np.float64), base_coordinates):
            raise ValueError(
                "Owner-local exact source coefficients differ from their execution shadows."
            )
        routes = base_routes
        coordinates = base_coordinates
        exact = base_exact
        shifts = block.cell_vertex_shifts
        if (
            not view.coordinates_are_lifted
            and view.periodic_lift_matrix is not None
            and identifiers.size
            and shifts is None
        ):
            raise ValueError(
                "Unlifted periodic mapped cells require absolute vertex shifts."
            )
        if (
            not view.coordinates_are_lifted
            and shifts is not None
            and shifts.shape[-1] > 0
        ):
            if view.periodic_lift_matrix is None:
                raise ValueError(
                    "Unlifted periodic cell maps require the exact source lattice."
                )
            lattice = np.asarray(view.periodic_lift_matrix, dtype=np.float64)
            shifts_ = np.asarray(shifts, dtype=np.int64)
            vertex_count = len(reference_cell_topology(element.cell_kind).vertices)
            if (
                lattice.shape != (shifts_.shape[-1], base_coordinates.shape[1])
                or element.local_dof_count != vertex_count
                or base_routes.shape[1] != vertex_count
                or shifts_.shape[:2] != base_routes.shape
            ):
                raise ValueError(
                    "Periodic lifting is only supported for exact vertex-aligned coordinate maps."
                )
            if isinstance(element, PerCellRestrictedCellGeometryElement) and any(
                not np.all(row == row[:1]) for row in shifts_
            ):
                raise ValueError(
                    "Nonuniform periodic image shifts require an explicit restricted source chart."
                )
            lifted_routes = np.empty_like(base_routes)
            lifted_coordinates: list[np.ndarray] = []
            lifted_exact: list[tuple[ExactScalar, ...]] = []
            exact_lattice = tuple(
                tuple(Fraction(float(value)) for value in row) for row in lattice
            )
            for cell in range(identifiers.size):
                for local_dof in range(element.local_dof_count):
                    winding = shifts_[cell, local_dof]
                    floating_shift = winding @ lattice
                    source_route = int(base_routes[cell, local_dof])
                    lifted_coordinates.append(
                        base_coordinates[source_route] + floating_shift
                    )
                    exact_row = base_exact[source_route]
                    lifted_exact.append(
                        tuple(
                            exact_row[axis]
                            + sum(
                                (
                                    int(winding[rank]) * exact_lattice[rank][axis]
                                    for rank in range(winding.size)
                                ),
                                Fraction(0),
                            )
                            for axis in range(base_coordinates.shape[1])
                        )
                    )
                    lifted_routes[cell, local_dof] = len(lifted_coordinates) - 1
            routes = lifted_routes
            coordinates = (
                np.asarray(lifted_coordinates, dtype=np.float64).reshape(
                    (-1, base_coordinates.shape[1])
                )
                if lifted_coordinates
                else np.empty((0, base_coordinates.shape[1]), dtype=np.float64)
            )
            exact = tuple(lifted_exact)
            if not np.array_equal(np.asarray(exact, dtype=np.float64), coordinates):
                raise ValueError(
                    "Lifted exact source coefficients differ from their binary64 shadows."
                )
        reference_dimension = reference_cell_topology(element.cell_kind).dimension
        self.coordinate_element = element
        self.coordinate_dofs = jnp.asarray(routes)
        self.coordinates = jnp.asarray(coordinates)
        self.cell_global_ids = jnp.asarray(identifiers)
        self.cell_count = identifiers.size
        self.coordinate_count = coordinates.shape[0]
        self.reference_dimension = reference_dimension
        self.ambient_dimension = coordinates.shape[1]
        self.block_name = name
        self.block_index = block_index
        self.topology_id = view.source_topology_id
        self.geometry_layout_id = view.geometry_layout_id
        self.source_map_id = view.source_map_id
        self.source_mesh_id = view.source_mesh_id
        self.source_geometry_id = view.source_geometry_id
        self.source_context_id = view.source_context_id
        self.periodic_lift_id = view.periodic_lift_id
        self.coordinates_are_lifted = view.coordinates_are_lifted
        self.local_view_id = view.local_view_id
        self.exact_coordinates = cast(CoordinateSourceBank, exact)
        self.cell_map_id = canonical_fingerprint(
            {
                "kind": "owner-local-source-mapped-cell-map",
                "part": view.part_id,
                "local_view": view.local_view_id,
                "source_mesh": view.source_mesh_id,
                "source_topology": view.source_topology_id,
                "source_geometry": view.source_geometry_id,
                "source_context": view.source_context_id,
                "source_map": view.source_map_id,
                "periodic_lift": (
                    view.periodic_lift_id,
                    view.coordinates_are_lifted,
                    shifts,
                ),
                "geometry_layout": view.geometry_layout_id,
                "block": (name, element.element_id),
                "cells": identifiers,
                "routes": routes,
                "exact_coordinates": tuple(
                    tuple(coefficient_payload(value) for value in row) for row in exact
                ),
            }
        )


type _MappedCellMap = (
    PreparedFiniteElementCellMap | PreparedMappedCellMap | PreparedOwnerLocalMappedCellMap
)


def prepare_owner_local_mapped_cell_locator(
    view: OwnerLocalCellGeometry,
    block_name: str,
    /,
    policy: MappedLocationPolicy | None = None,
) -> PreparedMappedCellLocator:
    """Prepare the canonical complete locator over one owner-local source view."""
    cell_map = PreparedOwnerLocalMappedCellMap(view, block_name)
    return PreparedMappedCellLocator(cell_map, cell_map.coordinates, policy)


def _source_coefficients(
    cell_map: _MappedCellMap,
    coordinates: np.ndarray,
    source_geometry: CellGeometrySpec | None,
    /,
) -> tuple[CoordinateCoefficients, np.ndarray]:
    routes = np.asarray(cell_map.coordinate_dofs, dtype=np.int64)
    if isinstance(cell_map, PreparedOwnerLocalMappedCellMap):
        if source_geometry is not None or not np.array_equal(
            np.asarray(cell_map.exact_coordinates, dtype=np.float64), coordinates
        ):
            raise ValueError(
                "Owner-local source coefficients do not match their compact map."
            )
        return cell_map.exact_coordinates, routes
    if source_geometry is None:
        return coordinates, routes
    if (
        source_geometry.geometry_layout_id != cell_map.geometry_layout_id
        or not np.array_equal(np.asarray(source_geometry.coordinates), coordinates)
    ):
        raise ValueError(
            "Mapped locator source geometry does not match its prepared cell map."
        )
    if (
        isinstance(cell_map, PreparedFiniteElementCellMap)
        and cell_map.block_index is None
    ):
        if cell_map.block_name != "whole-mesh" or any(
            element.element_id != cell_map.coordinate_element.element_id
            for element in source_geometry.elements
        ):
            raise ValueError(
                "Whole-mesh mapped locators require one source coordinate element identity."
            )
        source_routes = np.concatenate(
            tuple(
                np.asarray(route, dtype=np.int64)
                for route in source_geometry.geometry_dofs
            ),
            axis=0,
        )
    else:
        if source_geometry.block_names.count(cell_map.block_name) != 1:
            raise ValueError(
                "Mapped locator source geometry does not name its exact cell block."
            )
        index = source_geometry.block_names.index(cell_map.block_name)
        if (
            source_geometry.elements[index].element_id
            != cell_map.coordinate_element.element_id
        ):
            raise ValueError(
                "Mapped locator source coordinate element differs from the prepared cell map."
            )
        source_routes = np.asarray(source_geometry.geometry_dofs[index], dtype=np.int64)
    if not np.array_equal(source_routes, routes):
        raise ValueError(
            "Mapped locator source DOF routes differ from the prepared cell map."
        )
    source_values = source_geometry.source_coordinates()
    return source_values, routes


@final
class PreparedMappedCellLocator(AbstractCellLocator, NonTrainableState):
    """Invert mapped cells using whole-source Bernstein bounds and complete candidates."""

    __strict_contract__ = True

    cell_map: _MappedCellMap
    coordinates: Float[_MappedCoordinateDim, _MappedAmbientDim]
    cell_lower: Float[_MappedCellDim, _MappedAmbientDim]
    cell_upper: Float[_MappedCellDim, _MappedAmbientDim]
    bvh: PackedBVH
    policy: MappedLocationPolicy
    map_expressions: tuple[tuple[tuple[Expression, tuple[Expression, ...]], ...], ...] = (
        eqx.field(static=True)
    )
    locator_id: str = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)

    @classmethod
    def from_cell_geometry(
        cls,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        block_index: int,
        policy: MappedLocationPolicy | None = None,
        /,
    ) -> PreparedMappedCellLocator:
        cell_map = PreparedMappedCellMap(mesh, geometry, block_index)
        return cls(cell_map, geometry.coordinates, policy, source_geometry=geometry)

    @checked
    def __init__(
        self,
        cell_map: _MappedCellMap,
        coordinates: ConvertibleToArray,
        policy: MappedLocationPolicy | None = None,
        /,
        *,
        source_geometry: CellGeometrySpec | None = None,
    ) -> None:
        if not isinstance(
            cell_map,
            (
                PreparedFiniteElementCellMap,
                PreparedMappedCellMap,
                PreparedOwnerLocalMappedCellMap,
            ),
        ):
            raise TypeError("cell_map must be a prepared mapped cell map.")
        policy_ = MappedLocationPolicy() if policy is None else policy
        if not isinstance(policy_, MappedLocationPolicy):
            raise TypeError("policy must be MappedLocationPolicy or None.")
        values = np.asarray(coordinates, dtype=np.float64)
        if (
            values.shape != (cell_map.coordinate_count, cell_map.ambient_dimension)
            or not np.all(np.isfinite(values))
            or cell_map.cell_count < 1
            or cell_map.cell_count > np.iinfo(np.int32).max
            or cell_map.reference_dimension < 1
            or cell_map.reference_dimension > cell_map.ambient_dimension
        ):
            raise ValueError("Mapped locator coordinates or dimensions are invalid.")
        element = cell_map.coordinate_element
        domain = reference_domain(element.cell_kind)
        source_values, routes = _source_coefficients(cell_map, values, source_geometry)
        lower = np.empty(
            (cell_map.cell_count, cell_map.ambient_dimension), dtype=np.float64
        )
        upper = np.empty_like(lower)
        expressions = []
        for cell, route in enumerate(routes):
            local = (
                tuple(source_values[int(index)] for index in route)
                if isinstance(source_values, tuple)
                else np.asarray(source_values, dtype=np.float64)[route]
            )
            row = (
                cell
                if isinstance(element, PerCellRestrictedCellGeometryElement)
                else None
            )
            cell_map_expression = _cell_expressions(
                element, local, row, cell_map.reference_dimension
            )
            bounds = np.asarray(
                coordinate_map_bounds(element, local, domain, cell_row=row),
                dtype=np.float64,
            )
            if bounds.shape != (cell_map.ambient_dimension, 2) or np.any(
                ~np.isfinite(bounds)
            ):
                raise ValueError(
                    "Mapped locator requires finite whole-cell source enclosures."
                )
            lower[cell], upper[cell] = bounds[:, 0], bounds[:, 1]
            expressions.append(cell_map_expression)
        padding = policy_.residual_tolerance * np.maximum(
            1.0, np.maximum(np.abs(lower), np.abs(upper))
        )
        lower = np.nextafter(lower - padding, -np.inf)
        upper = np.nextafter(upper + padding, np.inf)
        if np.any(lower > upper):
            raise ValueError("Mapped source enclosure produced an inverted cell bound.")
        bvh = prepare_bvh(lower, upper, dtype=jnp.float64)
        self.cell_map = cell_map
        self.coordinates = jnp.asarray(values)
        self.cell_lower, self.cell_upper = jnp.asarray(lower), jnp.asarray(upper)
        self.bvh, self.policy = bvh, policy_
        self.map_expressions = tuple(expressions)
        self.source_binding_id = canonical_fingerprint(
            {
                "kind": "whole-source-mapped-cell-binding",
                "cell_map": cell_map.cell_map_id,
                "coordinates": array_tree_fingerprint(values),
                "source_geometry": (
                    source_geometry.source_map_id
                    if source_geometry is not None
                    else getattr(cell_map, "source_map_id", None)
                ),
                "lower": array_tree_fingerprint(lower),
                "upper": array_tree_fingerprint(upper),
            }
        )
        self.locator_id = canonical_fingerprint(
            {
                "kind": "prepared-mapped-cell-locator",
                "source": self.source_binding_id,
                "policy": policy_.policy_id,
            }
        )

    def inverse_jacobian(
        self,
        cell_indices: ConvertibleToArray,
        reference_points: ConvertibleToArray,
        /,
    ) -> Array:
        """Evaluate the exact source map's physical inverse Jacobian at fixed routes."""
        if isinstance(cell_indices, Tracer) or isinstance(reference_points, Tracer):
            raise TypeError("Mapped inverse Jacobians are prepared on the host.")
        cells = np.asarray(cell_indices)
        references = np.asarray(reference_points, dtype=np.float64)
        if (
            cells.ndim != 1
            or cells.dtype.kind not in "iu"
            or references.shape != (cells.size, self.cell_map.reference_dimension)
            or np.any(cells < 0)
            or np.any(cells >= self.cell_map.cell_count)
            or np.any(~np.isfinite(references))
        ):
            raise ValueError(
                "Mapped inverse Jacobian routes do not match their cell map."
            )
        inverse = np.empty(
            (
                cells.size,
                self.cell_map.reference_dimension,
                self.cell_map.ambient_dimension,
            ),
            dtype=np.float64,
        )
        for row, cell in enumerate(cells):
            _, jacobian = _map_values(self.map_expressions[int(cell)], references[row])
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            if (
                not np.all(np.isfinite(jacobian))
                or singular_values.size != self.cell_map.reference_dimension
                or singular_values[-1] <= np.finfo(np.float64).tiny
            ):
                raise ValueError("Mapped query derivative has a singular physical map.")
            if self.cell_map.reference_dimension == self.cell_map.ambient_dimension:
                inverse[row] = np.linalg.solve(
                    jacobian, np.eye(self.cell_map.reference_dimension)
                )
            else:
                inverse[row] = np.linalg.lstsq(
                    jacobian, np.eye(self.cell_map.ambient_dimension), rcond=None
                )[0]
        return jnp.asarray(inverse, dtype=self.coordinates.dtype)

    def locate(
        self,
        points: ConvertibleToArray,
        /,
        *,
        cell_mask: ConvertibleToArray | None = None,
    ) -> CellLocationResult:
        if isinstance(points, Tracer):
            raise TypeError(
                "Mapped cell location is a host preparation query, not a traced operation."
            )
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.cell_map.ambient_dimension:
            raise ValueError("Locator points have incompatible physical dimensions.")
        if cell_mask is None:
            mask = np.ones((self.cell_map.cell_count,), dtype=np.bool_)
        else:
            mask = np.asarray(cell_mask)
            if mask.shape != (self.cell_map.cell_count,) or mask.dtype != np.dtype(
                np.bool_
            ):
                raise ValueError("cell_mask must contain one boolean per mapped cell.")
        count = values.shape[0]
        finite = np.all(np.isfinite(values), axis=1)
        located_rows: list[list[tuple[int, _InverseRoot]]] = []
        complete = np.ones((count,), dtype=np.bool_)
        search_complete_all = np.ones((count,), dtype=np.bool_)
        ambiguous = np.zeros((count,), dtype=np.bool_)
        for point_index, point in enumerate(values):
            target = np.asarray(point, dtype=np.float64)
            hits: list[tuple[int, _InverseRoot]] = []
            if not finite[point_index]:
                complete[point_index] = False
                located_rows.append(hits)
                continue
            # The structural capacity is every source cell, never a candidate quota.
            candidates, valid, search_complete = point_select_leaf_items(
                jnp.asarray(target[None]),
                bvh=self.bvh,
                maximum_candidates=self.cell_map.cell_count,
                tolerance=0.0,
            )
            candidate_rows = np.asarray(candidates[0], dtype=np.int64)
            candidate_valid = np.asarray(valid[0], dtype=np.bool_)
            search_complete_all[point_index] = bool(np.asarray(search_complete[0]))
            for cell in candidate_rows[candidate_valid]:
                if not mask[cell]:
                    continue
                isolation = _isolate_inverse_roots(
                    target,
                    self.map_expressions[int(cell)],
                    self.cell_map.coordinate_element.cell_kind,
                    self.policy,
                )
                # Undecided regions keep their box evidence in the isolation outcome; the
                # location contract reports them as an exhausted inverse map.
                if isolation.unresolved:
                    complete[point_index] = False
                ambiguous[point_index] |= isolation.ambiguous
                hits.extend((int(cell), root) for root in isolation.roots)
            hits.sort(
                key=lambda item: (
                    item[0],
                    tuple(float(value) for value in item[1].reference),
                )
            )
            if len(hits) > np.iinfo(np.int32).max:
                raise OverflowError(
                    "Mapped inverse candidates exceed the CellLocationResult int32 row index."
                )
            located_rows.append(hits)
        width = max((len(rows) for rows in located_rows), default=0)
        candidate_cells = np.full((count, width), -1, dtype=np.int32)
        candidate_reference = np.zeros(
            (count, width, self.cell_map.reference_dimension), dtype=np.float64
        )
        reference_vertex_count = len(
            reference_cell_topology(self.cell_map.coordinate_element.cell_kind).vertices
        )
        candidate_weights = np.zeros(
            (count, width, reference_vertex_count), dtype=np.float64
        )
        best_residual = np.full((count,), np.inf, dtype=np.float64)
        iterations = np.zeros((count,), dtype=np.int32)
        condition = np.full((count,), np.inf, dtype=np.float64)
        used_fallback = np.zeros((count,), dtype=np.bool_)
        for point_index, rows in enumerate(located_rows):
            for slot, (cell, root) in enumerate(rows):
                candidate_cells[point_index, slot] = cell
                candidate_reference[point_index, slot] = root.reference
                candidate_weights[point_index, slot] = _reference_weights(
                    self.cell_map.coordinate_element.cell_kind, root.reference
                )
                if slot == 0:
                    best_residual[point_index] = root.residual
                    iterations[point_index] = root.steps
                    condition[point_index] = root.condition
                    used_fallback[point_index] = root.fallback
        candidate_count = np.asarray([len(rows) for rows in located_rows], dtype=np.int32)
        inside = (
            (candidate_count > 0) & complete & search_complete_all & finite & ~ambiguous
        )
        resolved_ambiguous = ambiguous & complete & search_complete_all & finite
        cell_ids = np.where(inside, candidate_cells[:, 0] if width else -1, -1)
        references = np.zeros(
            (count, self.cell_map.reference_dimension), dtype=np.float64
        )
        barycentric = np.zeros((count, reference_vertex_count), dtype=np.float64)
        if width:
            references[inside] = candidate_reference[inside, 0]
            barycentric[inside] = candidate_weights[inside, 0]
        status = np.where(
            ~finite,
            int(CellLocationStatus.NONFINITE),
            np.where(
                inside,
                int(CellLocationStatus.LOCATED),
                np.where(
                    resolved_ambiguous,
                    int(CellLocationStatus.AMBIGUOUS),
                    np.where(
                        ~search_complete_all,
                        int(CellLocationStatus.CANDIDATE_SEARCH_INCOMPLETE),
                        np.where(
                            ~complete,
                            int(CellLocationStatus.INVERSE_MAP_EXHAUSTED),
                            int(CellLocationStatus.OUTSIDE),
                        ),
                    ),
                ),
            ),
        ).astype(np.int32)
        return CellLocationResult(
            jnp.asarray(cell_ids),
            jnp.asarray(references),
            jnp.asarray(barycentric),
            jnp.asarray(np.where(inside, best_residual, np.inf)),
            jnp.asarray(iterations),
            jnp.asarray(np.where(inside, condition, np.inf)),
            jnp.asarray(inside),
            jnp.asarray(used_fallback),
            jnp.asarray(candidate_count),
            jnp.asarray(status),
            jnp.asarray(inside),
            jnp.asarray(candidate_cells),
            jnp.asarray(candidate_reference),
            jnp.asarray(complete & search_complete_all),
            self.locator_id,
        )


__all__ = [
    "MappedLocationPolicy",
    "PreparedMappedCellMap",
    "PreparedOwnerLocalMappedCellMap",
    "prepare_owner_local_mapped_cell_locator",
    "PreparedMappedCellLocator",
    "reference_domain",
    "reference_margin",
]
