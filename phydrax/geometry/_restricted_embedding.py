#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact restrictions of mapped cell coordinates to declared common strata."""

from __future__ import annotations

from fractions import Fraction

from ..discretization._coordinate_enclosure import (
    add,
    axes,
    compose,
    compose_expression,
    constant,
    Expression,
    expression_equal,
    normalize_pyramid_reference_expression,
    Polynomial,
    RationalPolynomial,
    scale,
)
from ._embedding_cells import EmbeddingCell


def _affine_arguments(
    points: tuple[tuple[Fraction, ...], ...], /
) -> tuple[Polynomial, ...]:
    dimension = len(points[0])
    parameter_dimension = len(points) - 1
    result: list[Polynomial] = []
    for axis in range(dimension):
        expression: Polynomial = {
            (0,) * parameter_dimension: points[0][axis]
        }
        for parameter in range(parameter_dimension):
            coefficient = points[parameter + 1][axis] - points[0][axis]
            if coefficient:
                exponent = tuple(int(index == parameter) for index in range(parameter_dimension))
                expression[exponent] = coefficient
        result.append(expression)
    return tuple(result)


def _restrict(
    expression: Expression, arguments: tuple[Polynomial, ...], /
) -> Expression:
    if isinstance(expression, RationalPolynomial):
        numerator = compose(expression.numerator, arguments)
        denominator = compose(expression.denominator, arguments)
        if not isinstance(numerator, dict) or not isinstance(denominator, dict):
            raise RuntimeError(
                "Composing rational polynomial parts with affine arguments "
                "must remain polynomial."
            )
        return RationalPolynomial(numerator, denominator)
    return compose(expression, arguments)


def _triangle_vertex_order(
    first: EmbeddingCell, shared: tuple[int, ...], /
) -> tuple[int, ...]:
    for face in first.facet_routes:
        face_ids = tuple(first.vertex_ids[index] for index in face)
        if set(face_ids) == set(shared) and len(face_ids) == 3:
            return face_ids
    return ()


def _quadrilateral_vertex_order(
    first: EmbeddingCell, shared: tuple[int, ...], /
) -> tuple[int, ...]:
    for face in first.facet_routes:
        face_ids = tuple(first.vertex_ids[index] for index in face)
        if set(face_ids) != set(shared) or len(face_ids) != 4:
            continue
        start = min(range(4), key=lambda index: face_ids[index])
        forward = tuple(face_ids[(start + offset) % 4] for offset in range(4))
        backward = (forward[0], forward[3], forward[2], forward[1])
        return min(forward, backward)
    return ()


def _trace_arguments(
    cell: EmbeddingCell,
    ordered_ids: tuple[int, ...],
    /,
) -> tuple[Polynomial, ...] | None:
    reference_by_id = dict(zip(cell.vertex_ids, cell.reference_vertices, strict=True))
    try:
        references = tuple(reference_by_id[identifier] for identifier in ordered_ids)
    except KeyError:
        return None
    if len(references) not in (2, 3, 4):
        return None
    if len(references) == 4:
        # The shared-face route is cyclic; tensor coordinates use adjacent
        # corners in axis order, so swap the final two before affine lifting.
        tensor_references = (
            references[0],
            references[1],
            references[3],
            references[2],
        )
        if any(
            tensor_references[3][axis]
            != tensor_references[1][axis]
            + tensor_references[2][axis]
            - tensor_references[0][axis]
            for axis in range(len(tensor_references[0]))
        ):
            return None
        return _affine_arguments(tensor_references[:3])
    return _affine_arguments(references)


def _trace_equal_on_stratum(
    first: EmbeddingCell,
    second: EmbeddingCell,
    ordered_ids: tuple[int, ...],
    /,
) -> bool | None:
    first_arguments = _trace_arguments(first, ordered_ids)
    second_arguments = _trace_arguments(second, ordered_ids)
    if first_arguments is None or second_arguments is None:
        return None
    if len(first.coordinates) != len(second.coordinates):
        return False
    try:
        return all(
            expression_equal(
                _restrict(left, first_arguments),
                _restrict(right, second_arguments),
            )
            for left, right in zip(first.coordinates, second.coordinates, strict=True)
        )
    except (ValueError, ZeroDivisionError):
        return None


def _cell_coordinate_expressions(
    cell: EmbeddingCell, /
) -> tuple[Expression, ...] | None:
    if cell.coordinates:
        return cell.coordinates
    if cell.affine_offset is None or cell.affine_matrix is None:
        return None
    variables = axes(cell.dimension)
    result: list[Expression] = []
    for offset, row in zip(
        cell.affine_offset, cell.affine_matrix, strict=True
    ):
        if len(row) != cell.dimension:
            return None
        expression: Expression = constant(offset, cell.dimension)
        for coefficient, variable in zip(row, variables, strict=True):
            expression = add(expression, scale(variable, coefficient))
        result.append(expression)
    return tuple(result)


def reference_chart_composition_equal(
    parent: EmbeddingCell,
    target: EmbeddingCell,
    target_to_parent_arguments: tuple[Expression, ...],
    /,
) -> bool | None:
    """Prove the target physical map is the exact forward parent-map composition.

    Chart arguments are the authoritative target-reference→parent-reference
    expressions. This proof never reconstructs a chart from mapped vertices or
    consumes a geometry restriction-source alias.
    """
    if parent.dimension != target.dimension:
        return False
    parent_coordinates = _cell_coordinate_expressions(parent)
    target_coordinates = _cell_coordinate_expressions(target)
    if (
        parent_coordinates is None
        or target_coordinates is None
        or len(target_to_parent_arguments) != parent.dimension
    ):
        return None
    if len(parent_coordinates) != len(target_coordinates):
        return False
    try:
        composed = tuple(
            compose_expression(expression, target_to_parent_arguments)
            for expression in parent_coordinates
        )
        if target.kind == "pyramid":
            normalized_composed = tuple(
                normalize_pyramid_reference_expression(expression)
                for expression in composed
            )
            normalized_target = tuple(
                normalize_pyramid_reference_expression(expression)
                for expression in target_coordinates
            )
            if any(
                first.apex_trace is None
                or first.apex_value is None
                or second.apex_trace is None
                or second.apex_value is None
                for first, second in zip(
                    normalized_composed, normalized_target, strict=True
                )
            ):
                return None
            return all(
                first.apex_trace is not None
                and second.apex_trace is not None
                and expression_equal(first.expression, second.expression)
                and expression_equal(first.apex_trace, second.apex_trace)
                and first.apex_value == second.apex_value
                for first, second in zip(
                    normalized_composed, normalized_target, strict=True
                )
            )
        return all(
            expression_equal(expected, actual)
            for expected, actual in zip(
                composed, target_coordinates, strict=True
            )
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def restriction_trace_equal(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared_vertex_ids: tuple[int, ...],
    /,
) -> bool | None:
    """Prove equality of exact polynomial/rational traces on a shared edge or face.

    ``None`` is an explicit lack of a supported exact restriction theorem; it is
    never interpreted as matching geometry.
    """
    shared = tuple(sorted(set(shared_vertex_ids)))
    if not shared:
        return True
    first_ids = set(first.vertex_ids)
    second_ids = set(second.vertex_ids)
    if not set(shared) <= first_ids or not set(shared) <= second_ids:
        return False
    if len(shared) == 1:
        first_index = first.vertex_ids.index(shared[0])
        second_index = second.vertex_ids.index(shared[0])
        return first.corner_images[first_index] == second.corner_images[second_index]
    if (first.affine or not first.coordinates) and (second.affine or not second.coordinates):
        return all(
            first.corner_images[first.vertex_ids.index(identifier)]
            == second.corner_images[second.vertex_ids.index(identifier)]
            for identifier in shared
        )
    if len(shared) == 2:
        return _trace_equal_on_stratum(first, second, shared)
    if len(shared) == 3 and first.dimension == second.dimension == 3:
        ordered = _triangle_vertex_order(first, shared)
        return None if not ordered else _trace_equal_on_stratum(first, second, ordered)
    if len(shared) == 4 and first.dimension == second.dimension == 3:
        ordered = _quadrilateral_vertex_order(first, shared)
        return None if not ordered else _trace_equal_on_stratum(first, second, ordered)
    return None


__all__ = ["reference_chart_composition_equal", "restriction_trace_equal"]
