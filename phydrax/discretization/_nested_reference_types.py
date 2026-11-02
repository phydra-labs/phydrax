from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from types import MappingProxyType
from typing import NamedTuple

import numpy as np

from ._coordinate_enclosure import (
    add,
    bernstein_coefficients,
    BernsteinDomain,
    constant,
    evaluate_expression,
    ExactScalar,
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


__all__ = [
    "ExactReferenceChartArgumentBank",
    "ExactReferenceChartExpression",
    "NestedReferenceWitnesses",
    "ReferenceCellMapWitnesses",
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
]
