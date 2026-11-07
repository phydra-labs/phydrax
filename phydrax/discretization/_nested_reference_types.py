from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from heapq import heappop, heappush
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
from ..linalg._exact import solve_exact

from ._coordinate_enclosure import (
    add,
    axes,
    bernstein_coefficients,
    BernsteinDomain,
    constant,
    evaluate_expression,
    ExactScalar,
    expression_equal,
    normalize_pyramid_reference_expression,
    RationalPolynomial,
    scale,
)
from ._exact_algebraic_geometry import RadicalElement


type ExactReferenceChartExpression = (
    Mapping[tuple[int, ...], ExactScalar] | RationalPolynomial
)
type ExactReferenceChartArgumentBank = (
    tuple[tuple[ExactReferenceChartExpression, ...], ...] | None
)


class NestedReferenceWitnesses(NamedTuple):
    """Exact cell-to-parent reference corners for a nested simplex transition.

    ``fine_cell_ids[i]`` lies in ``coarse_cell_ids[i]``. Rows of
    ``fine_reference_vertices[i]`` are the fine cell's corners expressed in the
    coarse reference simplex (origin at corner zero, unit vectors at the others).
    An optional homogeneous bank preserves rational coordinates that are not
    themselves exactly represented by the binary64 execution view.
    """

    fine_cell_ids: np.ndarray
    coarse_cell_ids: np.ndarray
    fine_reference_vertices: np.ndarray
    fine_reference_homogeneous: np.ndarray | None = None
    fine_reference_exact_vertices: tuple[tuple[tuple[Fraction, ...], ...], ...] | None = (
        None
    )


def validate_nested_reference_witnesses(
    witnesses: NestedReferenceWitnesses,
    /,
    *,
    dimension: int | None = None,
) -> None:
    if not isinstance(witnesses, NestedReferenceWitnesses):
        raise TypeError("witnesses must be NestedReferenceWitnesses.")
    fine = np.asarray(witnesses.fine_cell_ids)
    coarse = np.asarray(witnesses.coarse_cell_ids)
    points = np.asarray(witnesses.fine_reference_vertices, dtype=np.float64)
    if (
        fine.ndim != 1
        or coarse.shape != fine.shape
        or fine.dtype.kind not in "iu"
        or coarse.dtype.kind not in "iu"
        or np.any(fine < 0)
        or np.any(coarse < 0)
        or np.unique(fine).size != fine.size
        or points.ndim != 3
        or points.shape[0] != fine.size
        or points.shape[1] != points.shape[2] + 1
        or not np.all(np.isfinite(points))
        or (dimension is not None and points.shape[2] != dimension)
    ):
        raise ValueError(
            "Nested reference witnesses must contain aligned finite simplex rows."
        )
    exact = witnesses.fine_reference_exact_vertices
    if exact is not None:
        if (
            not isinstance(exact, tuple)
            or len(exact) != fine.size
            or any(
                not isinstance(cell, tuple)
                or len(cell) != points.shape[1]
                or any(
                    not isinstance(vertex, tuple)
                    or len(vertex) != points.shape[2]
                    or any(not isinstance(value, Fraction) for value in vertex)
                    for vertex in cell
                )
                for cell in exact
            )
        ):
            raise ValueError(
                "Exact nested reference rows must be immutable Fraction tuples."
            )
        for cell, vertices in enumerate(exact):
            for vertex, coordinates in enumerate(vertices):
                barycentric = (Fraction(1) - sum(coordinates), *coordinates)
                if any(value < 0 for value in barycentric):
                    raise ValueError(
                        "Exact nested reference corners must lie in the parent simplex."
                    )
                rounded = np.asarray(
                    [float(value) for value in coordinates], dtype=np.float64
                )
                if not np.array_equal(
                    rounded.view(np.uint64), points[cell, vertex].view(np.uint64)
                ):
                    raise ValueError(
                        "Binary64 reference corners must round from their exact bank."
                    )
    homogeneous = witnesses.fine_reference_homogeneous
    if homogeneous is None:
        return
    bank = np.asarray(homogeneous, dtype=np.float64)
    if bank.shape != (fine.size, points.shape[1], points.shape[1]) or not np.all(
        np.isfinite(bank)
    ):
        raise ValueError("Homogeneous reference rows must align with simplex corners.")
    if np.any(bank < 0.0):
        raise ValueError(
            "Nested homogeneous barycentric coordinates must be nonnegative."
        )
    for cell in range(fine.size):
        for vertex in range(points.shape[1]):
            row = tuple(Fraction(float(value)) for value in bank[cell, vertex])
            total = sum(row, Fraction(0))
            if not total:
                continue
            coordinates = tuple(value / total for value in row[1:])
            rounded = np.asarray(
                [float(value) for value in coordinates], dtype=np.float64
            )
            if not np.array_equal(
                rounded.view(np.uint64), points[cell, vertex].view(np.uint64)
            ):
                raise ValueError(
                    "Stored reference corners must be the rounded homogeneous map."
                )
            if exact is not None and coordinates != exact[cell][vertex]:
                raise ValueError("Homogeneous and exact nested reference banks disagree.")


def exact_reference_vertices(
    witnesses: NestedReferenceWitnesses, /
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    """Return exact rational reference corners for every nested fine cell."""
    validate_nested_reference_witnesses(witnesses)
    if witnesses.fine_reference_exact_vertices is not None:
        return witnesses.fine_reference_exact_vertices
    points = np.asarray(witnesses.fine_reference_vertices, dtype=np.float64)
    homogeneous = witnesses.fine_reference_homogeneous
    result = []
    for cell in range(points.shape[0]):
        vertices = []
        for vertex in range(points.shape[1]):
            if homogeneous is not None:
                row = tuple(
                    Fraction(float(value))
                    for value in np.asarray(homogeneous)[cell, vertex]
                )
                total = sum(row, Fraction(0))
                if total:
                    vertices.append(tuple(value / total for value in row[1:]))
                    continue
            vertices.append(
                tuple(Fraction(float(value)) for value in points[cell, vertex])
            )
        result.append(tuple(vertices))
    return tuple(result)


class ReferenceCellMapWitnesses(NamedTuple):
    """Exact per-cell target-reference→source-reference chart witnesses.

    Exact corner banks are retained for execution and identity. When
    source_reference_chart_arguments is present, each row stores the complete
    target-local→source-reference expression map instead of inferring it from
    corners; validation rechecks corner agreement and whole-cell inclusion.
    """

    target_cell_ids: np.ndarray
    source_cell_ids: np.ndarray
    target_cell_kind: str
    source_cell_kind: str
    source_reference_vertices: np.ndarray
    source_reference_exact_vertices: (
        tuple[tuple[tuple[Fraction, ...], ...], ...] | None
    ) = None
    source_reference_chart_arguments: ExactReferenceChartArgumentBank = None


def _fractions_sum_at_most_one(point: tuple[Fraction, ...], /) -> bool:
    """Exact ``sum(point) <= 1`` on integer numerators and denominators."""
    numerator, denominator = 0, 1
    for value in point:
        numerator = numerator * value.denominator + value.numerator * denominator
        denominator *= value.denominator
    return numerator <= denominator


def _reference_cell_contains(kind: str, point: tuple[Fraction, ...], /) -> bool:
    from ._reference_cell import reference_cell_topology

    dimension = reference_cell_topology(kind).dimension
    if len(point) != dimension:
        return False
    if (kind in ("triangle", "tetrahedron") or kind.startswith("simplex:")) and all(
        type(value) is Fraction for value in point
    ):
        # Fraction denominators are positive: signs are the numerators' signs.
        return all(value.numerator >= 0 for value in point) and (
            _fractions_sum_at_most_one(point)
        )
    if kind.startswith("simplex:"):
        return all(value >= 0 for value in point) and sum(point) <= 1
    if kind.startswith("tensor:"):
        return all(0 <= value <= 1 for value in point)
    match kind:
        case "interval" | "quadrilateral" | "hexahedron":
            return all(0 <= value <= 1 for value in point)
        case "triangle" | "tetrahedron":
            return all(value >= 0 for value in point) and sum(point) <= 1
        case "prism":
            return (
                point[0] >= 0
                and point[1] >= 0
                and point[0] + point[1] <= 1
                and 0 <= point[2] <= 1
            )
        case "pyramid":
            x, y, z = point
            width = (Fraction(1) - z) / 2
            return (
                0 <= z <= 1
                and Fraction(1, 2) - width <= x <= Fraction(1, 2) + width
                and Fraction(1, 2) - width <= y <= Fraction(1, 2) + width
            )
        case _:
            return False


def _exact_reference_scalar(value: object, /) -> ExactScalar:
    if isinstance(value, (Fraction, RadicalElement)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return Fraction(float(value))
    raise TypeError("Reference-chart coefficients must be exact real scalars.")


def _freeze_reference_polynomial(
    value: Mapping[tuple[int, ...], object], /
) -> Mapping[tuple[int, ...], ExactScalar]:
    if not isinstance(value, Mapping):
        raise TypeError("Polynomial chart expressions must be mappings.")
    terms: dict[tuple[int, ...], ExactScalar] = {}
    for raw_exponent, raw_coefficient in value.items():
        if not isinstance(raw_exponent, tuple) or any(
            isinstance(power, (bool, np.bool_))
            or not isinstance(power, (int, np.integer))
            or int(power) < 0
            for power in raw_exponent
        ):
            raise ValueError(
                "Reference-chart polynomial exponents must be nonnegative integer tuples."
            )
        exponent = tuple(int(power) for power in raw_exponent)
        coefficient = _exact_reference_scalar(raw_coefficient)
        if coefficient:
            terms[exponent] = terms.get(exponent, Fraction(0)) + coefficient
    return MappingProxyType(
        {exponent: coefficient for exponent, coefficient in terms.items() if coefficient}
    )


def freeze_reference_chart_arguments(
    arguments: Sequence[Sequence[ExactReferenceChartExpression]], /
) -> tuple[tuple[ExactReferenceChartExpression, ...], ...]:
    """Freeze exact per-cell target-local→source-reference expression rows."""
    if not isinstance(arguments, Sequence):
        raise TypeError("Reference-chart argument rows must be a sequence.")
    result = []
    for row in arguments:
        if not isinstance(row, Sequence):
            raise TypeError("Each reference-chart row must be an expression sequence.")
        expressions = []
        for expression in row:
            if isinstance(expression, RationalPolynomial):
                expressions.append(
                    RationalPolynomial(
                        _freeze_reference_polynomial(expression.numerator),
                        _freeze_reference_polynomial(expression.denominator),
                    )
                )
            else:
                expressions.append(_freeze_reference_polynomial(expression))
        result.append(tuple(expressions))
    return tuple(result)


def _scalar_identity(value: ExactScalar, /) -> tuple[object, ...]:
    if isinstance(value, Fraction):
        return ("rational", value.numerator, value.denominator)
    return (
        "radical",
        tuple(
            (radicand, coefficient.numerator, coefficient.denominator)
            for radicand, coefficient in value.terms
        ),
    )


def reference_chart_arguments_identity(
    arguments: Sequence[Sequence[ExactReferenceChartExpression]] | None, /
) -> tuple[object, ...] | None:
    """Return an address-independent identity payload for an exact chart bank."""
    if arguments is None:
        return None
    frozen = freeze_reference_chart_arguments(arguments)
    return tuple(
        tuple(
            (
                "rational",
                tuple(
                    (exponent, _scalar_identity(coefficient))
                    for exponent, coefficient in sorted(expression.numerator.items())
                ),
                tuple(
                    (exponent, _scalar_identity(coefficient))
                    for exponent, coefficient in sorted(expression.denominator.items())
                ),
            )
            if isinstance(expression, RationalPolynomial)
            else (
                "polynomial",
                tuple(
                    (exponent, _scalar_identity(coefficient))
                    for exponent, coefficient in sorted(expression.items())
                ),
            )
            for expression in row
        )
        for row in frozen
    )


def _reference_expression_dimension(
    expression: ExactReferenceChartExpression, /
) -> int | None:
    banks = (
        (expression.numerator, expression.denominator)
        if isinstance(expression, RationalPolynomial)
        else (expression,)
    )
    dimensions = {len(exponent) for bank in banks for exponent in bank}
    if len(dimensions) > 1:
        raise ValueError("Reference-chart monomials must share one target dimension.")
    return None if not dimensions else next(iter(dimensions))


def _reference_bernstein_domain(cell_kind: str, /) -> tuple[BernsteinDomain, int]:
    from ._reference_cell import reference_cell_topology

    dimension = reference_cell_topology(cell_kind).dimension
    if cell_kind in ("triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        return "simplex", dimension
    if cell_kind == "prism":
        return "prism", dimension
    if cell_kind == "pyramid":
        return "box", dimension
    if cell_kind in ("interval", "quadrilateral", "hexahedron") or cell_kind.startswith(
        "tensor:"
    ):
        return "box", dimension
    raise ValueError(f"Reference-chart enclosure does not support {cell_kind!r} cells.")


def _vertex_chart_is_multilinear(target_kind: str, /) -> bool:
    """Whether whole-cell inclusion of a canonical vertex chart is its corner inclusion.

    On simplex, box and prism targets the vertex-family chart has degree at most
    one per Bernstein factor, and every source-domain inequality is affine in the
    source coordinates. The composed inequality is then multilinear, so its
    least-degree Bernstein controls are exactly its values at the target corners:
    the Bernstein inclusion proof holds if and only if every corner image lies in
    the source cell. Pyramid charts are rational and keep the full proof.
    """
    return (
        target_kind
        in ("interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron", "prism")
        or target_kind.startswith("simplex:")
        or target_kind.startswith("tensor:")
    )


def _reference_bernstein_expression(
    expression: ExactReferenceChartExpression, target_kind: str, /
) -> tuple[ExactReferenceChartExpression, BernsteinDomain, int]:
    domain, dimension = _reference_bernstein_domain(target_kind)
    if target_kind == "pyramid":
        expression = normalize_pyramid_reference_expression(expression).expression
    return expression, domain, dimension


def _reference_domain_inequalities(
    cell_kind: str,
    arguments: tuple[ExactReferenceChartExpression, ...],
    target_dimension: int,
    /,
) -> tuple[ExactReferenceChartExpression, ...]:
    one = constant(1, target_dimension)
    if cell_kind in ("triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        total = constant(0, target_dimension)
        for value in arguments:
            total = add(total, value)
        return (*arguments, add(one, scale(total, -1)))
    if cell_kind in ("interval", "quadrilateral", "hexahedron") or cell_kind.startswith(
        "tensor:"
    ):
        return tuple(
            expression
            for value in arguments
            for expression in (value, add(one, scale(value, -1)))
        )
    if cell_kind == "prism":
        x, y, z = arguments
        return (
            x,
            y,
            add(add(one, scale(x, -1)), scale(y, -1)),
            z,
            add(one, scale(z, -1)),
        )
    if cell_kind == "pyramid":
        x, y, z = arguments
        half_z = scale(z, Fraction(1, 2))
        return (
            add(x, scale(half_z, -1)),
            add(add(one, scale(half_z, -1)), scale(x, -1)),
            add(y, scale(half_z, -1)),
            add(add(one, scale(half_z, -1)), scale(y, -1)),
            z,
            add(one, scale(z, -1)),
        )
    raise ValueError(f"Reference-domain inclusion does not support {cell_kind!r} cells.")


def _rational_denominator_nonzero(
    expression: ExactReferenceChartExpression, target_kind: str, /
) -> bool:
    if not isinstance(expression, RationalPolynomial):
        return True
    if len(expression.denominator) == 1:
        exponent, coefficient = next(iter(expression.denominator.items()))
        if not any(exponent) and coefficient == Fraction(1):
            return True
    transformed, domain, dimension = _reference_bernstein_expression(
        expression, target_kind
    )
    if not isinstance(transformed, RationalPolynomial):
        return True
    controls = bernstein_coefficients(transformed.denominator, domain, dimension)
    return bool(controls) and (
        all(value > 0 for value in controls) or all(value < 0 for value in controls)
    )


def reference_chart_arguments_have_nonvanishing_denominators(
    target_kind: str,
    arguments: tuple[ExactReferenceChartExpression, ...],
    /,
) -> bool:
    """Prove every rational chart denominator is nonzero over its target cell."""
    from ._reference_cell import reference_cell_topology

    target_dimension = reference_cell_topology(target_kind).dimension
    if not isinstance(arguments, tuple):
        raise TypeError("Reference-chart arguments must be an immutable tuple.")
    for expression in arguments:
        expression_dimension = _reference_expression_dimension(expression)
        if expression_dimension is not None and expression_dimension != target_dimension:
            raise ValueError(
                "Reference-chart expressions must use target-local coordinates."
            )
        if isinstance(expression, RationalPolynomial):
            banks = (expression.numerator, expression.denominator)
        elif isinstance(expression, Mapping):
            banks = (expression,)
        else:
            raise TypeError(
                "Reference-chart expressions must be polynomials or RationalPolynomial values."
            )
        for bank in banks:
            if any(
                not isinstance(exponent, tuple)
                or len(exponent) != target_dimension
                or any(
                    isinstance(power, (bool, np.bool_))
                    or not isinstance(power, (int, np.integer))
                    or int(power) < 0
                    for power in exponent
                )
                for exponent in bank
            ):
                raise ValueError(
                    "Reference-chart polynomial terms must match the target reference dimension."
                )
            if any(
                not isinstance(value, (Fraction, RadicalElement))
                for value in bank.values()
            ):
                raise TypeError(
                    "Reference-chart coefficients must be rational or exact algebraic values."
                )
        if isinstance(
            expression, RationalPolynomial
        ) and not _rational_denominator_nonzero(expression, target_kind):
            return False
    return True


def _evaluate_reference_chart_arguments_at_point(
    target_kind: str,
    arguments: tuple[ExactReferenceChartExpression, ...],
    point: tuple[ExactScalar, ...],
    /,
) -> tuple[ExactScalar, ...]:
    if target_kind == "pyramid" and point == (
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1),
    ):
        values: list[ExactScalar] = []
        for expression in arguments:
            normalization = normalize_pyramid_reference_expression(expression)
            if normalization.apex_value is None:
                raise ValueError(
                    "A pyramid reference chart does not have one unique C0 apex image."
                )
            values.append(normalization.apex_value)
        return tuple(values)
    try:
        return tuple(evaluate_expression(expression, point) for expression in arguments)
    except ZeroDivisionError as error:
        raise ValueError(
            "Reference-chart evaluation encountered a rational pole."
        ) from error


def evaluate_reference_chart_arguments(
    target_kind: str,
    arguments: tuple[ExactReferenceChartExpression, ...],
    point: tuple[ExactScalar, ...],
    /,
) -> tuple[ExactScalar, ...]:
    """Evaluate an exact chart, extending only a uniquely continuous pyramid apex."""
    from ._reference_cell import reference_cell_topology

    dimension = reference_cell_topology(target_kind).dimension
    if not isinstance(arguments, tuple):
        raise TypeError("Reference-chart arguments must be an immutable tuple.")
    if len(point) != dimension:
        raise ValueError("Target reference coordinates must match their cell dimension.")
    if not reference_chart_arguments_have_nonvanishing_denominators(
        target_kind, arguments
    ):
        raise ValueError(
            "A rational reference-chart denominator is not proved nonzero over its target cell."
        )
    return _evaluate_reference_chart_arguments_at_point(target_kind, arguments, point)


def _reference_expression_nonnegative(
    expression: ExactReferenceChartExpression, target_kind: str, /
) -> bool:
    transformed, domain, dimension = _reference_bernstein_expression(
        expression, target_kind
    )
    if isinstance(transformed, RationalPolynomial):
        numerator = bernstein_coefficients(transformed.numerator, domain, dimension)
        denominator = bernstein_coefficients(transformed.denominator, domain, dimension)
        positive = bool(denominator) and all(value > 0 for value in denominator)
        negative = bool(denominator) and all(value < 0 for value in denominator)
        return (positive and all(value >= 0 for value in numerator)) or (
            negative and all(value <= 0 for value in numerator)
        )
    controls = bernstein_coefficients(transformed, domain, dimension)
    return bool(controls) and all(value >= 0 for value in controls)


def reference_chart_arguments_contain_source_cell(
    target_kind: str,
    source_kind: str,
    arguments: tuple[ExactReferenceChartExpression, ...],
    /,
) -> bool:
    """Prove the complete expression image lies in its canonical source cell."""
    from ._reference_cell import reference_cell_topology

    target_dimension = reference_cell_topology(target_kind).dimension
    source_dimension = reference_cell_topology(source_kind).dimension
    if not isinstance(arguments, tuple) or len(arguments) != source_dimension:
        raise ValueError(
            "Reference-chart arguments must match the source reference dimension."
        )
    if not reference_chart_arguments_have_nonvanishing_denominators(
        target_kind, arguments
    ):
        raise ValueError(
            "A rational reference-chart denominator is not proved nonzero over its target cell."
        )
    if target_kind == "pyramid" and any(
        normalize_pyramid_reference_expression(expression).apex_value is None
        for expression in arguments
    ):
        return False
    inequalities = _reference_domain_inequalities(
        source_kind, arguments, target_dimension
    )
    return all(
        _reference_expression_nonnegative(value, target_kind) for value in inequalities
    )


def reference_cell_strictly_contains(
    kind: str, point: tuple[ExactScalar, ...], /
) -> bool:
    """Test exact strict interior ownership in a canonical reference cell."""
    from ._reference_cell import reference_cell_topology

    if len(point) != reference_cell_topology(kind).dimension:
        return False
    if kind in ("interval", "quadrilateral", "hexahedron") or kind.startswith("tensor:"):
        return all(0 < value < 1 for value in point)
    if kind in ("triangle", "tetrahedron") or kind.startswith("simplex:"):
        return all(value > 0 for value in point) and sum(point) < 1
    if kind == "prism":
        x, y, z = point
        return x > 0 and y > 0 and x + y < 1 and 0 < z < 1
    if kind == "pyramid":
        x, y, z = point
        half_z = z * Fraction(1, 2)
        return 0 < z < 1 and half_z < x < 1 - half_z and half_z < y < 1 - half_z
    return False


def _validate_reference_chart_argument_rows(
    arguments: ExactReferenceChartArgumentBank,
    target_kind: str,
    source_kind: str,
    exact_vertices: tuple[tuple[tuple[Fraction, ...], ...], ...],
    /,
) -> None:
    if arguments is None:
        return
    from ._reference_cell import reference_cell_topology

    target_topology = reference_cell_topology(target_kind)
    source_dimension = reference_cell_topology(source_kind).dimension
    if not isinstance(arguments, tuple) or len(arguments) != len(exact_vertices):
        raise ValueError(
            "Exact chart argument rows must align with reference-cell witnesses."
        )
    target_points = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in target_topology.vertices
    )
    validated_rows: set[tuple[object, ...]] = set()
    for cell_index, (row, corners) in enumerate(
        zip(arguments, exact_vertices, strict=True)
    ):
        if not isinstance(row, tuple) or len(row) != source_dimension:
            raise ValueError(
                "Each exact chart row must map into its source reference dimension."
            )
        for expression in row:
            if not (
                isinstance(expression, RationalPolynomial)
                or isinstance(expression, MappingProxyType)
            ):
                raise TypeError(
                    "Exact chart rows must use immutable polynomial or rational expressions."
                )
        identity = reference_chart_arguments_identity((row,))
        key = (identity, corners)
        if key in validated_rows:
            continue
        if not reference_chart_arguments_contain_source_cell(
            target_kind, source_kind, row
        ):
            raise ValueError(
                f"Exact chart row {cell_index} does not prove whole-cell inclusion in its source reference cell."
            )
        evaluated = tuple(
            _evaluate_reference_chart_arguments_at_point(target_kind, row, point)
            for point in target_points
        )
        if (
            any(not isinstance(value, Fraction) for point in evaluated for value in point)
            or evaluated != corners
        ):
            raise ValueError(
                f"Exact chart row {cell_index} does not reproduce its retained rational reference corners."
            )
        validated_rows.add(key)


def exact_reference_cell_vertices(
    witnesses: ReferenceCellMapWitnesses, /
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    from ._reference_cell import reference_cell_topology

    if not isinstance(witnesses, ReferenceCellMapWitnesses):
        raise TypeError("witnesses must be ReferenceCellMapWitnesses.")
    target = np.asarray(witnesses.target_cell_ids)
    source = np.asarray(witnesses.source_cell_ids)
    points = np.asarray(witnesses.source_reference_vertices, dtype=np.float64)
    try:
        target_topology = reference_cell_topology(witnesses.target_cell_kind)
        source_topology = reference_cell_topology(witnesses.source_cell_kind)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Reference-cell witnesses need supported fixed cell kinds."
        ) from error
    if (
        target.ndim != 1
        or source.shape != target.shape
        or target.dtype.kind not in "iu"
        or source.dtype.kind not in "iu"
        or np.any(target < 0)
        or np.any(source < 0)
        or np.unique(target).size != target.size
        or points.shape
        != (
            target.size,
            len(target_topology.vertices),
            source_topology.dimension,
        )
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "Reference-cell map witnesses must align cell IDs and target corners."
        )
    exact = witnesses.source_reference_exact_vertices
    if exact is None:
        result = tuple(
            tuple(tuple(Fraction(float(value)) for value in point) for point in cell)
            for cell in points
        )
    else:
        if (
            not isinstance(exact, tuple)
            or len(exact) != target.size
            or any(
                not isinstance(cell, tuple)
                or len(cell) != len(target_topology.vertices)
                or any(
                    not isinstance(point, tuple)
                    or len(point) != source_topology.dimension
                    or any(not isinstance(value, Fraction) for value in point)
                    for point in cell
                )
                for cell in exact
            )
        ):
            raise ValueError(
                "Exact reference-cell maps must be immutable Fraction tuples."
            )
        result = exact
        rounded = np.fromiter(
            (float(value) for cell in result for point in cell for value in point),
            dtype=np.float64,
            count=points.size,
        ).reshape(points.shape)
        if not np.array_equal(
            rounded.view(np.uint64), np.ascontiguousarray(points).view(np.uint64)
        ):
            raise ValueError(
                "Reference-cell execution corners must round from their exact bank."
            )
    if any(
        not _reference_cell_contains(witnesses.source_cell_kind, point)
        for cell in result
        for point in cell
    ):
        raise ValueError(
            "Target reference corners must lie in their source reference cell."
        )
    _validate_reference_chart_argument_rows(
        witnesses.source_reference_chart_arguments,
        witnesses.target_cell_kind,
        witnesses.source_cell_kind,
        result,
    )
    if witnesses.source_reference_chart_arguments is None and not (
        _vertex_chart_is_multilinear(witnesses.target_cell_kind)
    ):
        from ._nested_reference import _chart_from_corners

        for cell_index, corners in enumerate(result):
            chart = _chart_from_corners(witnesses.target_cell_kind, corners)
            if not reference_chart_arguments_contain_source_cell(
                witnesses.target_cell_kind,
                witnesses.source_cell_kind,
                chart,
            ):
                raise ValueError(
                    f"Canonical vertex-family chart {cell_index} does not prove "
                    "whole-cell inclusion in its source reference cell."
                )
    return result


def validate_reference_cell_map_witnesses(
    witnesses: ReferenceCellMapWitnesses, /
) -> None:
    exact_reference_cell_vertices(witnesses)


class ReferenceCellPartition(NamedTuple):
    """Complete disjoint fine-reference partitions of coarse reference cells.

    Each map is oriented fine-to-coarse: its ``target_cell_ids`` name the fine
    cells, and its ``source_cell_ids`` name the coarse parents. This direction
    is also used when the partition witnesses a physical coarsening transition.
    """

    maps: tuple[ReferenceCellMapWitnesses, ...]


def _partition_subtract(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...], /
) -> tuple[Fraction, ...]:
    return tuple(a - b for a, b in zip(first, second, strict=True))


def _partition_cross(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...], /
) -> tuple[Fraction, ...]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _partition_dot(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...], /
) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _partition_affine_coefficients(
    kind: str, vertices: tuple[tuple[Fraction, ...], ...], /
) -> tuple[tuple[tuple[Fraction, ...], ...], Fraction]:
    from ._reference_cell import reference_cell_topology

    local = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology(kind).vertices
    )
    if kind in ("tetrahedron", "simplex:3", "prism"):
        basis = (0, 1, 2, 3)
    elif kind in ("hexahedron", "pyramid"):
        basis = (0, 1, 3, 4)
    elif kind in ("triangle", "simplex:2"):
        basis = (0, 1, 2)
    else:
        raise ValueError(
            "Reference partitions require standard 2D or 3D simplex/volume cells."
        )
    dimension = len(basis) - 1
    if len(vertices) != len(local) or any(len(point) != dimension for point in vertices):
        raise ValueError("Reference partition corners do not match their cell family.")
    coefficients = solve_exact(
        [[Fraction(1), *local[index]] for index in basis],
        [list(vertices[index]) for index in basis],
    )
    if any(
        tuple(
            coefficients[0][axis]
            + sum(
                (
                    point[column] * coefficients[column + 1][axis]
                    for column in range(dimension)
                ),
                Fraction(0),
            )
            for axis in range(dimension)
        )
        != image
        for point, image in zip(local, vertices, strict=True)
    ):
        raise ValueError("Reference partition maps must be exact affine restrictions.")
    columns = tuple(
        tuple(coefficients[column + 1][axis] for axis in range(dimension))
        for column in range(dimension)
    )
    if dimension == 2:
        determinant = columns[0][0] * columns[1][1] - columns[0][1] * columns[1][0]
    else:
        first, second, third = columns
        determinant = _partition_dot(first, _partition_cross(second, third))
    if not determinant:
        raise ValueError("Reference partition cells must be nondegenerate.")
    return tuple(tuple(row) for row in coefficients), abs(determinant)


def _partition_interiors_disjoint(
    first_kind: str,
    first: tuple[tuple[Fraction, ...], ...],
    second_kind: str,
    second: tuple[tuple[Fraction, ...], ...],
    /,
) -> bool:
    if first_kind == second_kind == "prism":
        def extruded(vertices: tuple[tuple[Fraction, ...], ...]) -> bool:
            return (
                len(vertices) == 6
                and all(vertices[index][:2] == vertices[index + 3][:2] for index in range(3))
                and len({vertices[index][2] for index in range(3)}) == 1
                and len({vertices[index][2] for index in range(3, 6)}) == 1
                and (
                    (vertices[1][0] - vertices[0][0]) * (vertices[2][1] - vertices[0][1])
                    - (vertices[1][1] - vertices[0][1]) * (vertices[2][0] - vertices[0][0])
                ) != 0
                and vertices[0][2] != vertices[3][2]
            )

        if extruded(first) and extruded(second):
            first_z = sorted((first[0][2], first[3][2]))
            second_z = sorted((second[0][2], second[3][2]))
            if first_z[1] <= second_z[0] or second_z[1] <= first_z[0]:
                return True
            # Each admitted image is exactly a triangle × interval; 2D
            # separating axes of the two triangle edges are complete.
            for triangle in (first, second):
                for start, end in ((0, 1), (1, 2), (2, 0)):
                    dx = triangle[end][0] - triangle[start][0]
                    dy = triangle[end][1] - triangle[start][1]
                    left = tuple(-dy * point[0] + dx * point[1] for point in first[:3])
                    right = tuple(-dy * point[0] + dx * point[1] for point in second[:3])
                    if max(left) <= min(right) or max(right) <= min(left):
                        return True
            return False
    from ._reference_cell import reference_cell_topology

    first_topology = reference_cell_topology(first_kind)
    second_topology = reference_cell_topology(second_kind)
    if first_topology.dimension == second_topology.dimension == 2:
        # Both images are convex polygons: their edge normals are a complete
        # set of separating axes.
        for topology, vertices in ((first_topology, first), (second_topology, second)):
            for start, end in topology.entities[1]:
                normal = (
                    vertices[start][1] - vertices[end][1],
                    vertices[end][0] - vertices[start][0],
                )
                left = tuple(_partition_dot(normal, point) for point in first)
                right = tuple(_partition_dot(normal, point) for point in second)
                if max(left) <= min(right) or max(right) <= min(left):
                    return True
        return False
    axes = []
    for topology, vertices in ((first_topology, first), (second_topology, second)):
        for face in topology.entities[2]:
            axes.append(
                _partition_cross(
                    _partition_subtract(vertices[face[1]], vertices[face[0]]),
                    _partition_subtract(vertices[face[2]], vertices[face[0]]),
                )
            )
    first_edges = tuple(
        _partition_subtract(first[edge[1]], first[edge[0]])
        for edge in first_topology.entities[1]
    )
    second_edges = tuple(
        _partition_subtract(second[edge[1]], second[edge[0]])
        for edge in second_topology.entities[1]
    )
    axes.extend(_partition_cross(a, b) for a in first_edges for b in second_edges)
    for axis in axes:
        if not any(axis):
            continue
        left = tuple(_partition_dot(axis, point) for point in first)
        right = tuple(_partition_dot(axis, point) for point in second)
        if max(left) <= min(right) or max(right) <= min(left):
            return True
    return False


def _partition_aabb_candidates(
    children: Sequence[
        tuple[str, tuple[tuple[Fraction, ...], ...], Fraction]
    ],
    /,
) -> Iterator[tuple[int, int]]:
    """Yield pairs whose exact axis-aligned bounding boxes overlap."""
    dimension = len(children[0][1][0]) if children else 3
    bounds = tuple(
        tuple(
            (min(point[axis] for point in vertices), max(point[axis] for point in vertices))
            for axis in range(dimension)
        )
        for _, vertices, _ in children
    )
    # Choose the axis with the fewest interval overlaps, not a caller-supplied
    # grid assumption. Touching intervals cannot contain intersecting interiors.
    orders = tuple(
        sorted(range(len(bounds)), key=lambda index: bounds[index][axis][0])
        for axis in range(dimension)
    )
    counts = []
    for axis, order in enumerate(orders):
        ends: list[Fraction] = []
        count = 0
        for index in order:
            lower, upper = bounds[index][axis]
            while ends and ends[0] <= lower:
                heappop(ends)
            count += len(ends)
            heappush(ends, upper)
        counts.append(count)
    axis = min(range(dimension), key=counts.__getitem__)
    remaining_axes = tuple(other for other in range(dimension) if other != axis)
    active: dict[int, None] = {}
    ends_with_ids: list[tuple[Fraction, int]] = []
    for index in orders[axis]:
        lower, upper = bounds[index][axis]
        while ends_with_ids and ends_with_ids[0][0] <= lower:
            _, expired = heappop(ends_with_ids)
            del active[expired]
        for other in active:
            if all(
                bounds[index][other_axis][0] < bounds[other][other_axis][1]
                and bounds[other][other_axis][0] < bounds[index][other_axis][1]
                for other_axis in remaining_axes
            ):
                yield other, index
        active[index] = None
        heappush(ends_with_ids, (upper, index))


def validate_reference_cell_partition(partition: ReferenceCellPartition, /) -> None:
    """Prove exact affine inclusion, disjoint interiors, and complete parent cover.

    Partitions are triangle-by-triangle in 2D or by standard volume cells in
    3D. Exact bounding-box separation excludes pairs with disjoint interiors;
    all remaining pairs receive the full separating-axis proof. Inclusion and
    disjointness make exact equality of the measure sum a complete-cover proof.
    """
    if (
        not isinstance(partition, ReferenceCellPartition)
        or not isinstance(partition.maps, tuple)
        or not partition.maps
    ):
        raise ValueError("A reference partition requires nonempty exact map groups.")
    volumes = {
        "triangle": Fraction(1, 2),
        "simplex:2": Fraction(1, 2),
        "tetrahedron": Fraction(1, 6),
        "simplex:3": Fraction(1, 6),
        "prism": Fraction(1, 2),
        "pyramid": Fraction(1, 3),
        "hexahedron": Fraction(1),
    }
    planar = frozenset(("triangle", "simplex:2"))
    partition_dimension: int | None = None
    parents: dict[
        int, tuple[str, list[tuple[str, tuple[tuple[Fraction, ...], ...], Fraction]]]
    ] = {}
    fine_ids: set[int] = set()
    for group in partition.maps:
        if not isinstance(group, ReferenceCellMapWitnesses):
            raise TypeError("Reference partition groups must be reference map witnesses.")
        if (
            group.target_cell_kind not in volumes
            or group.source_cell_kind not in volumes
        ):
            raise ValueError("Reference partitions require affine standard-volume maps.")
        group_dimension = 2 if group.target_cell_kind in planar else 3
        if (group.source_cell_kind in planar) != (group_dimension == 2) or (
            partition_dimension is not None and group_dimension != partition_dimension
        ):
            raise ValueError("Reference partition cells must share one dimension.")
        partition_dimension = group_dimension
        exact = exact_reference_cell_vertices(group)
        for row, (fine, coarse, vertices) in enumerate(zip(
            np.asarray(group.target_cell_ids, dtype=np.int64),
            np.asarray(group.source_cell_ids, dtype=np.int64),
            exact,
            strict=True,
        )):
            fine_id, coarse_id = int(fine), int(coarse)
            if fine_id in fine_ids:
                raise ValueError("A fine cell occurs more than once in a reference partition.")
            fine_ids.add(fine_id)
            parent_kind, children = parents.setdefault(
                coarse_id, (group.source_cell_kind, [])
            )
            if parent_kind != group.source_cell_kind:
                raise ValueError("A reference partition parent has conflicting cell families.")
            coefficients, determinant = _partition_affine_coefficients(
                group.target_cell_kind, vertices
            )
            arguments = group.source_reference_chart_arguments
            if arguments is None:
                if group.target_cell_kind == "pyramid":
                    raise ValueError("Pyramid partitions require explicit affine charts.")
            else:
                variables = axes(group_dimension)
                for axis, actual in enumerate(arguments[row]):
                    expected = constant(coefficients[0][axis], group_dimension)
                    for column in range(group_dimension):
                        expected = add(
                            expected,
                            scale(variables[column], coefficients[column + 1][axis]),
                        )
                    actual_expression = actual if isinstance(actual, RationalPolynomial) else dict(actual)
                    if not expression_equal(expected, actual_expression):
                        raise ValueError("Reference partition chart is not its affine corner map.")
            children.append((group.target_cell_kind, vertices, determinant))
    if not fine_ids:
        raise ValueError("A reference partition cannot be empty.")
    for parent_kind, children in parents.values():
        total = sum(
            (volumes[kind] * determinant for kind, _, determinant in children),
            Fraction(0),
        )
        if total != volumes[parent_kind]:
            raise ValueError("Reference partition does not cover its parent volume exactly.")
        for index, other_index in _partition_aabb_candidates(children):
            kind, vertices, _ = children[index]
            other_kind, other_vertices, _ = children[other_index]
            if not _partition_interiors_disjoint(
                kind, vertices, other_kind, other_vertices
            ):
                raise ValueError("Reference partition cells have overlapping interiors.")


__all__ = [
    "ExactReferenceChartArgumentBank",
    "ExactReferenceChartExpression",
    "NestedReferenceWitnesses",
    "ReferenceCellMapWitnesses",
    "ReferenceCellPartition",
    "exact_reference_cell_vertices",
    "exact_reference_vertices",
    "evaluate_reference_chart_arguments",
    "freeze_reference_chart_arguments",
    "reference_chart_arguments_contain_source_cell",
    "reference_chart_arguments_have_nonvanishing_denominators",
    "reference_chart_arguments_identity",
    "reference_cell_strictly_contains",
    "validate_nested_reference_witnesses",
    "validate_reference_cell_map_witnesses",
    "validate_reference_cell_partition",
]
