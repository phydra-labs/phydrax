#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact rational chart restrictions with bounded binary64 execution views."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


type ExactChartCoordinate = tuple[Fraction, Fraction]
type ChartRestrictionArrays = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def _fraction_payload(value: Fraction, /) -> tuple[int, int]:
    return value.numerator, value.denominator


def _coordinate_payload(value: ExactChartCoordinate, /) -> tuple[tuple[int, int], ...]:
    return tuple(_fraction_payload(coordinate) for coordinate in value)


def _coordinate(value: tuple[Fraction, Fraction], name: str, /) -> ExactChartCoordinate:
    if len(value) != 2 or any(
        not isinstance(coordinate, Fraction) for coordinate in value
    ):
        raise TypeError(f"{name} must contain two exact Fraction coordinates.")
    return value


def _upper_float(value: Fraction, /) -> float:
    """Smallest practical binary64 upper bound of one nonnegative rational."""
    if value < 0:
        raise ValueError("An upper-bound conversion requires a nonnegative rational.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("An exact chart-coordinate error exceeds binary64 range.")
    return float(np.nextafter(result, np.inf)) if Fraction(result) < value else result


@final
class ExactRationalChartRestriction(StrictModule, NonTrainableState):
    """One source-bound affine edge root and its rounded execution coordinate.

    The authoritative coordinate is ``(1-t) * start + t * end`` with rational
    ``t``. ``root_coefficients`` and ``isolation`` are mandatory exact evidence
    that this parameter is the unique root of a primitive affine source
    equation. The binary64 coordinate is only an execution view and must be the
    correctly rounded image of the exact coordinate.
    """

    source_revision: str = eqx.field(static=True)
    patch: int = eqx.field(static=True)
    vertex: int = eqx.field(static=True)
    edge_vertices: tuple[int, int] = eqx.field(static=True)
    edge_coordinates: tuple[ExactChartCoordinate, ExactChartCoordinate] = eqx.field(
        static=True
    )
    parameter: Fraction = eqx.field(static=True)
    root_coefficients: tuple[int, int] = eqx.field(static=True)
    isolation: tuple[Fraction, Fraction] = eqx.field(static=True)
    exact_coordinate: ExactChartCoordinate = eqx.field(static=True)
    execution_coordinate: np.ndarray
    restriction_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_revision: str,
        patch: int,
        vertex: int,
        edge_vertices: tuple[int, int],
        edge_coordinates: tuple[ExactChartCoordinate, ExactChartCoordinate],
        parameter: Fraction,
        execution_coordinate: np.ndarray,
        /,
        *,
        root_coefficients: tuple[int, int],
        isolation: tuple[Fraction, Fraction],
    ) -> None:
        revision = str(source_revision).strip()
        if not revision:
            raise ValueError("An exact chart restriction requires a source revision.")
        if isinstance(patch, bool) or not isinstance(patch, int) or patch < 0:
            raise ValueError(
                "An exact chart restriction requires a nonnegative patch index."
            )
        if isinstance(vertex, bool) or not isinstance(vertex, int) or vertex < 0:
            raise ValueError(
                "An exact chart restriction requires a nonnegative vertex index."
            )
        if (
            len(edge_vertices) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in edge_vertices
            )
            or edge_vertices[0] < 0
            or edge_vertices[1] < 0
            or edge_vertices[0] == edge_vertices[1]
            or max(edge_vertices) >= vertex
        ):
            raise ValueError(
                "An exact chart restriction must reference two distinct earlier vertices."
            )
        if len(edge_coordinates) != 2:
            raise TypeError("edge_coordinates must contain the two exact edge endpoints.")
        start = _coordinate(edge_coordinates[0], "edge start")
        end = _coordinate(edge_coordinates[1], "edge end")
        if start == end:
            raise ValueError(
                "An exact chart restriction requires a nonempty source edge."
            )
        if not isinstance(parameter, Fraction):
            raise TypeError(
                "An exact chart restriction parameter requires Fraction authority."
            )
        if not Fraction(0) < parameter < Fraction(1):
            raise ValueError(
                "An exact chart restriction root must lie inside its source edge."
            )
        if len(root_coefficients) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in root_coefficients
        ):
            raise TypeError("root_coefficients must be two exact integers.")
        constant, linear = root_coefficients
        if linear <= 0 or math.gcd(abs(constant), linear) != 1:
            raise ValueError(
                "The affine chart root equation must be primitive with positive slope."
            )
        if Fraction(-constant, linear) != parameter:
            raise ValueError(
                "The affine chart root equation does not prove its parameter."
            )
        if len(isolation) != 2 or any(
            not isinstance(value, Fraction) for value in isolation
        ):
            raise TypeError("isolation must contain two exact Fraction endpoints.")
        lower, upper = isolation
        if not lower < parameter < upper:
            raise ValueError(
                "The exact chart root is not strictly isolated by its interval."
            )
        if constant + linear * lower >= 0 or constant + linear * upper <= 0:
            raise ValueError(
                "The affine chart root interval does not prove a unique crossing."
            )
        exact = (
            (Fraction(1) - parameter) * start[0] + parameter * end[0],
            (Fraction(1) - parameter) * start[1] + parameter * end[1],
        )
        execution = np.asarray(execution_coordinate, dtype=np.float64)
        if execution.shape != (2,) or not np.all(np.isfinite(execution)):
            raise ValueError("An execution chart coordinate must be one finite 2-vector.")
        represented = np.asarray([float(value) for value in exact], dtype=np.float64)
        if not np.array_equal(execution, represented):
            raise ValueError(
                "The execution chart coordinate is not the correctly rounded exact restriction."
            )
        self.source_revision = revision
        self.patch = patch
        self.vertex = vertex
        self.edge_vertices = edge_vertices
        self.edge_coordinates = start, end
        self.parameter = parameter
        self.root_coefficients = root_coefficients
        self.isolation = isolation
        self.exact_coordinate = exact
        self.execution_coordinate = execution
        self.restriction_id = canonical_fingerprint(
            {
                "kind": "exact-rational-chart-restriction",
                "source_revision": revision,
                "patch": patch,
                "vertex": vertex,
                "edge_vertices": edge_vertices,
                "edge_coordinates": (
                    _coordinate_payload(start),
                    _coordinate_payload(end),
                ),
                "root_coefficients": root_coefficients,
                "isolation": tuple(_fraction_payload(value) for value in isolation),
                "execution": tuple(float(value).hex() for value in execution),
            }
        )

    @property
    def execution_error(self) -> np.ndarray:
        """Outward coordinatewise error of the rounded execution view."""
        return np.asarray(
            [
                _upper_float(abs(Fraction(float(realized)) - exact))
                for realized, exact in zip(
                    self.execution_coordinate, self.exact_coordinate, strict=True
                )
            ],
            dtype=np.float64,
        )


def empty_chart_restrictions(vertex_count: int, /) -> ChartRestrictionArrays:
    """Canonical empty restriction banks for ``vertex_count`` chart vertices."""
    if (
        isinstance(vertex_count, bool)
        or not isinstance(vertex_count, int)
        or vertex_count < 0
    ):
        raise ValueError("vertex_count must be a nonnegative integer.")
    return (
        np.zeros((vertex_count,), dtype=np.bool_),
        np.empty((0,), dtype=np.int64),
        np.empty((0, 2), dtype=np.int64),
        np.empty((0, 2, 2), dtype=np.float64),
        np.empty((0, 2), dtype=np.int64),
    )


def validate_chart_restrictions(
    charts: np.ndarray,
    required: np.ndarray,
    vertices: np.ndarray,
    edges: np.ndarray,
    endpoint_charts: np.ndarray,
    parameters: np.ndarray,
    source_revision: str,
    patch: int,
    /,
) -> tuple[np.ndarray, tuple[ExactRationalChartRestriction, ...], np.ndarray]:
    """Validate complete rational authority and return exact coordinates/errors.

    Rows are topologically ordered: every restriction references earlier chart
    vertices. Missing, duplicated, noncanonical, nonunique, or stale authority
    is refused. Ordinary vertices retain their exact binary64 values as dyadic
    rationals; restricted vertices recursively retain their source-edge roots.
    """
    chart_values = np.asarray(charts)
    required_values = np.asarray(required)
    vertex_values = np.asarray(vertices)
    edge_values = np.asarray(edges)
    endpoint_values = np.asarray(endpoint_charts)
    parameter_values = np.asarray(parameters)
    count = chart_values.shape[0] if chart_values.ndim == 2 else -1
    if chart_values.dtype != np.float64 or chart_values.shape != (count, 2):
        raise TypeError("Chart restriction coordinates must be a float64 (n, 2) array.")
    if not np.all(np.isfinite(chart_values)):
        raise ValueError("Chart restriction coordinates must be finite.")
    if required_values.dtype != np.bool_ or required_values.shape != (count,):
        raise TypeError("Chart restriction requirements must be a bool (n,) array.")
    for name, value, shape in (
        ("vertices", vertex_values, (vertex_values.size,)),
        ("edges", edge_values, (vertex_values.size, 2)),
        ("parameters", parameter_values, (vertex_values.size, 2)),
    ):
        if value.dtype != np.int64 or value.shape != shape:
            raise TypeError(f"Chart restriction {name} must be a canonical int64 array.")
    if endpoint_values.dtype != np.float64 or endpoint_values.shape != (
        vertex_values.size,
        2,
        2,
    ):
        raise TypeError(
            "Chart restriction endpoint coordinates must be a float64 (r, 2, 2) array."
        )
    if not np.all(np.isfinite(endpoint_values)):
        raise ValueError("Chart restriction endpoint coordinates must be finite.")
    expected = np.flatnonzero(required_values).astype(np.int64)
    if not np.array_equal(vertex_values, expected):
        raise ValueError(
            "Every rounded chart vertex must carry exactly one canonical rational authority."
        )
    exact = np.empty((count, 2), dtype=object)
    for row in range(count):
        exact[row, 0] = Fraction(float(chart_values[row, 0]))
        exact[row, 1] = Fraction(float(chart_values[row, 1]))
    restrictions: list[ExactRationalChartRestriction] = []
    errors = np.zeros((count, 2), dtype=np.float64)
    for record, vertex_ in enumerate(vertex_values.tolist()):
        vertex = int(vertex_)
        first, second = (int(value) for value in edge_values[record])
        if (
            first < 0
            or second < 0
            or first == second
            or max(first, second) >= vertex
            or vertex >= count
        ):
            raise ValueError(
                "A chart restriction must reference two distinct earlier chart vertices."
            )
        if not np.array_equal(endpoint_values[record], chart_values[[first, second]]):
            raise ValueError(
                "A chart restriction endpoint bank differs from its execution topology."
            )
        numerator, denominator = (int(value) for value in parameter_values[record])
        if (
            denominator <= 0
            or not 0 < numerator < denominator
            or math.gcd(numerator, denominator) != 1
        ):
            raise ValueError(
                "A chart restriction parameter must be a reduced open-unit rational."
            )
        parameter = Fraction(numerator, denominator)
        edge_coordinates = (
            (exact[first, 0], exact[first, 1]),
            (exact[second, 0], exact[second, 1]),
        )
        restriction = ExactRationalChartRestriction(
            str(source_revision),
            int(patch),
            vertex,
            (first, second),
            edge_coordinates,
            parameter,
            chart_values[vertex],
            root_coefficients=(-numerator, denominator),
            isolation=(Fraction(0), Fraction(1)),
        )
        exact[vertex] = restriction.exact_coordinate
        errors[vertex] = restriction.execution_error
        restrictions.append(restriction)
    return exact, tuple(restrictions), errors


__all__ = [
    "ChartRestrictionArrays",
    "ExactChartCoordinate",
    "ExactRationalChartRestriction",
    "empty_chart_restrictions",
    "validate_chart_restrictions",
]
