#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared Boolean policies, selection semantics, and rectilinear boundary assembly."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Literal, TypeAlias, TypedDict

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._exact_algebraic_geometry import (
    coefficient_payload,
    rational_coefficient,
)
from ...typing import checked, ConvertibleToArray, parse
from .._cad_revision import (
    AssociationGraph,
    CADOccurrence,
    CADRevision,
)
from .._interval_enclosure import interval_add
from .._rational import fraction_down, fraction_up
from .._source_identity import source_entity_id
from ._carrier_pieces import carrier_payload
from ._constructors import assemble_brep_model, BRepTessellationPolicy
from ._intersection import (
    BranchRootEndpoint,
    ConicParameterEndpoint,
    endpoint_root_payload,
    NativePeriodEndpoint,
    ParametricIntersectionPolicy,
    TrimIntersectionRoot,
    TrimRootEndpoint,
)
from ._intersection_curve import CurveTrimSegment, IntersectionPCurve
from ._model import BRepEntityId, BRepGeometry, BRepModel
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    LineCurve,
    PlanePatch,
)
from ._sewing import BRepSewingPolicy, sew_brep, shells
from ._trim_helpers import period_scalar_bounds, PeriodOffset


_BooleanRootEndpoint: TypeAlias = (
    TrimRootEndpoint | BranchRootEndpoint | NativePeriodEndpoint | ConicParameterEndpoint
)


def _require_boolean_root_endpoint(endpoint: object, /) -> _BooleanRootEndpoint:
    if isinstance(
        endpoint,
        (
            TrimRootEndpoint,
            BranchRootEndpoint,
            NativePeriodEndpoint,
            ConicParameterEndpoint,
        ),
    ):
        return endpoint
    raise TypeError("Boolean endpoint bindings require an exact native root endpoint.")


def root_parameter_in_source(
    carrier: AbstractCurve, endpoint: object, /
) -> tuple[Fraction, Fraction]:
    endpoint = _require_boolean_root_endpoint(endpoint)
    if isinstance(endpoint, TrimRootEndpoint):
        scale = Fraction(endpoint.affine_parameter_scale)
        offset = Fraction(endpoint.affine_parameter_offset)
    else:
        scale, offset = Fraction(1), Fraction(0)
    if isinstance(endpoint, (TrimRootEndpoint, BranchRootEndpoint)):
        for next_scale, next_offset in endpoint.affine_transforms:
            scale, offset = (
                Fraction(next_scale) * scale,
                Fraction(next_scale) * offset + Fraction(next_offset),
            )
    source, _, _ = _pcurve_affine_source(carrier)
    if isinstance(source, IntersectionPCurve) and source.reversed:
        return -scale, Fraction(source.first) + Fraction(source.last) - offset
    return scale, offset


def conic_rootline_endpoint_parameter(
    carrier: AbstractCurve, endpoint: object, /
) -> Fraction | None:
    """Return a source-bound rooted-line endpoint in its exact carrier chart."""
    if (
        not isinstance(endpoint, TrimRootEndpoint)
        or not isinstance(endpoint.root, TrimIntersectionRoot)
        or endpoint.root.endpoint_witness is None
        or endpoint.operand != "second"
        or endpoint.source_pcurve is None
        or endpoint.source_surface is None
    ):
        return None
    witness = endpoint.root.endpoint_witness
    current_source, _, _ = _pcurve_affine_source(carrier)
    witness_source, _, _ = _pcurve_affine_source(witness.source_pcurve)
    endpoint_source, _, _ = _pcurve_affine_source(endpoint.source_pcurve)
    root_source, _, _ = _pcurve_affine_source(endpoint.root.second)
    if (
        carrier_payload(current_source) != carrier_payload(witness_source)
        or carrier_payload(endpoint_source) != carrier_payload(witness_source)
        or carrier_payload(root_source) != carrier_payload(witness_source)
        or carrier_payload(endpoint.source_surface.patch)
        != carrier_payload(witness.source_region.patch)
    ):
        return None
    try:
        parameter = Fraction.from_float(
            endpoint.affine_parameter_scale
        ) * witness.parameter_exact + Fraction.from_float(
            endpoint.affine_parameter_offset
        )
        for scale, offset in endpoint.affine_transforms:
            parameter = Fraction.from_float(scale) * parameter + Fraction.from_float(
                offset
            )
    except (OverflowError, ValueError):
        return None
    return parameter


def root_endpoint_id(endpoint: object, /) -> str:
    """Canonical identity for an exact endpoint binding, including its source chart."""
    return canonical_fingerprint(
        endpoint_root_payload(_require_boolean_root_endpoint(endpoint))
    )


def same_root_endpoint(first: object, second: object, /) -> bool:
    """Compare canonical exact root bindings, including their source charts."""
    return endpoint_root_payload(
        _require_boolean_root_endpoint(first)
    ) == endpoint_root_payload(_require_boolean_root_endpoint(second))


def affine_root_endpoint(
    endpoint: object, scale: float, offset: float, /
) -> TrimRootEndpoint | BranchRootEndpoint:
    """Retain a root's exact source identity under an affine scalar transport."""
    endpoint = _require_boolean_root_endpoint(endpoint)
    if isinstance(endpoint, TrimRootEndpoint):
        return TrimRootEndpoint(
            endpoint.root,
            endpoint.operand,
            affine_parameter_offset=endpoint.affine_parameter_offset,
            affine_parameter_scale=endpoint.affine_parameter_scale,
            affine_transforms=(
                *endpoint.affine_transforms,
                (float(scale), float(offset)),
            ),
            source_pcurve=endpoint.source_pcurve,
            source_surface=endpoint.source_surface,
        )
    if isinstance(endpoint, BranchRootEndpoint):
        return BranchRootEndpoint(
            endpoint.root,
            endpoint.curve,
            endpoint.chart,
            source_pcurve=endpoint.source_pcurve,
            source_side=endpoint.source_side,
            source_first=endpoint.source_first,
            source_last=endpoint.source_last,
            periodic_shifts=endpoint.periodic_shifts,
            affine_transforms=(
                *endpoint.affine_transforms,
                (float(scale), float(offset)),
            ),
        )
    raise TypeError("Affine root transport requires a trim or branch endpoint binding.")


def _rational_numeric_shadow(values: object, /) -> np.ndarray | None:
    exact = np.asarray(values, dtype=object)
    try:
        result = np.asarray(
            tuple(float(value) for value in exact.reshape((-1,))),
            dtype=np.float64,
        ).reshape(exact.shape)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if np.all(np.isfinite(result)) else None


def _affine_execution_matrix_is_nonsingular(value: object, /) -> bool:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        return False
    a, b, c, d = (
        Fraction.from_float(float(matrix[0, 0])),
        Fraction.from_float(float(matrix[0, 1])),
        Fraction.from_float(float(matrix[1, 0])),
        Fraction.from_float(float(matrix[1, 1])),
    )
    return a * d - b * c != 0


class AffinePCurve(AbstractCurve):
    """Exact rational chart transport of an unchanged source p-curve."""

    curve: AbstractCurve
    matrix: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = eqx.field(
        static=True
    )
    offset: tuple[Fraction, Fraction] = eqx.field(static=True)
    matrix_value: Array
    offset_value: Array

    def __init__(
        self,
        curve: AbstractCurve,
        matrix: ConvertibleToArray,
        offset: ConvertibleToArray,
    ) -> None:
        matrix_host = np.asarray(matrix, dtype=object)
        offset_host = np.asarray(offset, dtype=object)
        if (
            curve.ambient_dimension != 2
            or matrix_host.shape != (2, 2)
            or offset_host.shape != (2,)
        ):
            raise ValueError(
                "AffinePCurve requires a planar source, 2x2 matrix, and 2-vector."
            )

        def rational(value: object) -> Fraction:
            if isinstance(value, Fraction):
                return value
            if isinstance(value, (int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            ):
                return Fraction(int(value))
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                return Fraction(float(value))
            raise ValueError("AffinePCurve coefficients must be finite rational values.")

        exact_matrix: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = (
            (
                rational(matrix_host[0, 0]),
                rational(matrix_host[0, 1]),
            ),
            (
                rational(matrix_host[1, 0]),
                rational(matrix_host[1, 1]),
            ),
        )
        exact_offset: tuple[Fraction, Fraction] = (
            rational(offset_host[0]),
            rational(offset_host[1]),
        )
        if (
            exact_matrix[0][0] * exact_matrix[1][1]
            == exact_matrix[0][1] * exact_matrix[1][0]
        ):
            raise ValueError(
                "AffinePCurve matrix must be nonsingular over rational coefficients."
            )
        matrix_value = _rational_numeric_shadow(exact_matrix)
        offset_value = _rational_numeric_shadow(exact_offset)
        if matrix_value is None or offset_value is None:
            raise ValueError(
                "AffinePCurve exact coefficients must have finite binary64 shadows."
            )
        if not _affine_execution_matrix_is_nonsingular(matrix_value):
            raise ValueError(
                "AffinePCurve matrix shadow must remain nonsingular in binary64."
            )
        self.curve = curve
        self.matrix, self.offset = exact_matrix, exact_offset
        self.matrix_value = jnp.asarray(matrix_value, dtype=jnp.float64)
        self.offset_value = jnp.asarray(offset_value, dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.curve.parameter_domain

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return (
            (-np.inf, np.inf) if self.parameter_domain is None else self.parameter_domain
        )

    @property
    def period(self) -> float | None:
        return self.curve.period

    def native_period_symbol(self) -> str | None:
        return self.curve.native_period_symbol()

    def is_c1_on(self, first: float, last: float, /) -> bool:
        return self.curve.is_c1_on(first, last)

    def evaluate(self, parameters: Array, /) -> Array:
        return (
            jnp.asarray(self.curve.evaluate(parameters)) @ self.matrix_value.T
            + self.offset_value
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        source = _curve_box(self.curve, first, last)
        lower, upper = np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
        for row in range(2):
            lower_exact = self.offset[row]
            upper_exact = self.offset[row]
            for column in range(2):
                source_lower = Fraction.from_float(float(source[0, column]))
                source_upper = Fraction.from_float(float(source[1, column]))
                coefficient = self.matrix[row][column]
                products = (coefficient * source_lower, coefficient * source_upper)
                lower_exact += min(products)
                upper_exact += max(products)
            lower[row] = fraction_down(lower_exact)
            upper[row] = fraction_up(upper_exact)
        return np.stack((lower, upper))

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        # The exact rational linear part maps every source derivative enclosure.
        source_lower, source_upper = self.curve.derivative_bounds(
            first, last, order=order
        )
        if not (np.all(np.isfinite(source_lower)) and np.all(np.isfinite(source_upper))):
            return np.full(2, -np.inf), np.full(2, np.inf)
        lower, upper = np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
        for row in range(2):
            lower_exact, upper_exact = Fraction(0), Fraction(0)
            for column in range(2):
                coefficient = self.matrix[row][column]
                products = (
                    coefficient * Fraction.from_float(float(source_lower[column])),
                    coefficient * Fraction.from_float(float(source_upper[column])),
                )
                lower_exact += min(products)
                upper_exact += max(products)
            lower[row] = fraction_down(lower_exact)
            upper[row] = fraction_up(upper_exact)
        return lower, upper

    def identity_payload(self, /) -> dict[str, object]:
        return {
            "kind": "boolean-affine-pcurve",
            "curve": carrier_payload(self.curve),
            "matrix": tuple(
                tuple(coefficient_payload(value) for value in row) for row in self.matrix
            ),
            "offset": tuple(coefficient_payload(value) for value in self.offset),
        }


def affine_transport_source(curve: AbstractCurve, /) -> AbstractCurve:
    """Return the source p-curve whose parameter and roots a transport keeps."""
    while isinstance(curve, AffinePCurve):
        curve = curve.curve
    return curve


class PeriodicPCurve(AbstractCurve):
    """Retain an exact integer native-period gauge on a source p-curve."""

    source_curve: AbstractCurve
    patch: AbstractSurfacePatch
    period_shifts: tuple[int, int] = eqx.field(static=True)
    offset_value: Array

    def __init__(
        self,
        source_curve: AbstractCurve,
        patch: AbstractSurfacePatch,
        period_shifts: tuple[int, int] = (0, 0),
    ) -> None:
        shifts = period_shifts
        if (
            source_curve.ambient_dimension != 2
            or len(shifts) != 2
            or any(type(value) is not int for value in shifts)
        ):
            raise ValueError(
                "PeriodicPCurve requires a planar source and two integer shifts."
            )
        symbols = patch.native_period_symbols()
        if any(
            shift and symbol != "two_pi"
            for shift, symbol in zip(shifts, symbols, strict=True)
        ):
            raise ValueError(
                "PeriodicPCurve shifts require declared mathematical two-pi periods."
            )
        self.source_curve, self.patch = source_curve, patch
        self.period_shifts = shifts
        try:
            offset = np.asarray(
                [shift * (2.0 * np.pi) for shift in shifts], dtype=np.float64
            )
        except (OverflowError, ValueError) as error:
            raise ValueError(
                "PeriodicPCurve integer gauge must have a finite numeric shadow."
            ) from error
        if not np.all(np.isfinite(offset)):
            raise ValueError(
                "PeriodicPCurve integer gauge must have a finite numeric shadow."
            )
        self.offset_value = jnp.asarray(offset, dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.source_curve.parameter_domain

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return (
            (-np.inf, np.inf) if self.parameter_domain is None else self.parameter_domain
        )

    @property
    def period(self) -> float | None:
        return self.source_curve.period

    def native_period_symbol(self) -> str | None:
        return self.source_curve.native_period_symbol()

    def is_c1_on(self, first: float, last: float, /) -> bool:
        return self.source_curve.is_c1_on(first, last)

    def evaluate(self, parameters: Array, /) -> Array:
        return jnp.asarray(self.source_curve.evaluate(parameters)) + self.offset_value

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        # A constant integer-period translation leaves every parameter derivative unchanged.
        return self.source_curve.derivative_bounds(first, last, order=order)

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        source = _curve_box(self.source_curve, first, last)
        lower, upper = source[0].copy(), source[1].copy()
        for axis, shift in enumerate(self.period_shifts):
            if shift:
                period = period_scalar_bounds(PeriodOffset(Fraction(0), Fraction(shift)))
                lower[axis], upper[axis] = interval_add(
                    (lower[axis], upper[axis]), period
                )
        return np.stack((lower, upper))

    def identity_payload(self) -> dict[str, object]:
        return {
            "kind": "boolean-periodic-pcurve",
            "source": carrier_payload(self.source_curve),
            "patch": carrier_payload(self.patch),
            "shifts": self.period_shifts,
        }


def _curve_box(curve: AbstractCurve, first: float, last: float, /) -> np.ndarray:
    return np.asarray(curve.bounding_box(first, last), dtype=np.float64)


def _same_pcurve_parameter(first: AbstractCurve, second: AbstractCurve, /) -> bool:
    if first is second:
        return True
    if isinstance(first, IntersectionPCurve) and isinstance(second, IntersectionPCurve):
        return (
            first.curve.branch_id == second.curve.branch_id
            and first.side == second.side
            and first.reversed == second.reversed
            and (
                not first.reversed
                or Fraction(first.first) + Fraction(first.last)
                == Fraction(second.first) + Fraction(second.last)
            )
        )
    return canonical_fingerprint(carrier_payload(first)) == canonical_fingerprint(
        carrier_payload(second)
    )


def _pcurve_affine_source(
    curve: AbstractCurve,
    /,
) -> tuple[
    AbstractCurve,
    tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    tuple[ChartScalar, ChartScalar],
]:
    identity: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    zero: tuple[ChartScalar, ChartScalar] = (Fraction(0), Fraction(0))
    if isinstance(curve, PeriodicPCurve):
        source, matrix, offset = _pcurve_affine_source(curve.source_curve)
        return (
            source,
            matrix,
            (
                chart_add(
                    offset[0], PeriodOffset(Fraction(0), Fraction(curve.period_shifts[0]))
                ),
                chart_add(
                    offset[1], PeriodOffset(Fraction(0), Fraction(curve.period_shifts[1]))
                ),
            ),
        )
    if isinstance(curve, AffinePCurve):
        source, inner, inner_offset = _pcurve_affine_source(curve.curve)
        matrix: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = (
            (
                curve.matrix[0][0] * inner[0][0] + curve.matrix[0][1] * inner[1][0],
                curve.matrix[0][0] * inner[0][1] + curve.matrix[0][1] * inner[1][1],
            ),
            (
                curve.matrix[1][0] * inner[0][0] + curve.matrix[1][1] * inner[1][0],
                curve.matrix[1][0] * inner[0][1] + curve.matrix[1][1] * inner[1][1],
            ),
        )
        offset: tuple[ChartScalar, ChartScalar] = (
            chart_add(
                curve.offset[0],
                chart_add(
                    chart_scale(inner_offset[0], curve.matrix[0][0]),
                    chart_scale(inner_offset[1], curve.matrix[0][1]),
                ),
            ),
            chart_add(
                curve.offset[1],
                chart_add(
                    chart_scale(inner_offset[0], curve.matrix[1][0]),
                    chart_scale(inner_offset[1], curve.matrix[1][1]),
                ),
            ),
        )
        return source, matrix, offset
    return curve, identity, zero


def pcurve_affine_correspondence(
    first: AbstractCurve,
    second: AbstractCurve,
    /,
) -> (
    tuple[
        tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
        tuple[ChartScalar, ChartScalar],
    ]
    | None
):
    """Return exact chart map only when both p-curves share one source parameter atom."""
    if not _pcurve_execution_shadows_match(first) or not _pcurve_execution_shadows_match(
        second
    ):
        return None
    source_a, matrix_a, offset_a = _pcurve_affine_source(first)
    source_b, matrix_b, offset_b = _pcurve_affine_source(second)
    if not _same_pcurve_parameter(source_a, source_b):
        return None
    a, b = matrix_a[0]
    c, d = matrix_a[1]
    determinant = a * d - b * c
    if determinant == 0:
        return None
    inverse = ((d / determinant, -b / determinant), (-c / determinant, a / determinant))
    matrix: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = (
        (
            matrix_b[0][0] * inverse[0][0] + matrix_b[0][1] * inverse[1][0],
            matrix_b[0][0] * inverse[0][1] + matrix_b[0][1] * inverse[1][1],
        ),
        (
            matrix_b[1][0] * inverse[0][0] + matrix_b[1][1] * inverse[1][0],
            matrix_b[1][0] * inverse[0][1] + matrix_b[1][1] * inverse[1][1],
        ),
    )
    offset: tuple[ChartScalar, ChartScalar] = (
        chart_add(
            offset_b[0],
            -chart_add(
                chart_scale(offset_a[0], matrix[0][0]),
                chart_scale(offset_a[1], matrix[0][1]),
            ),
        ),
        chart_add(
            offset_b[1],
            -chart_add(
                chart_scale(offset_a[0], matrix[1][0]),
                chart_scale(offset_a[1], matrix[1][1]),
            ),
        ),
    )
    return matrix, offset


def _pcurve_execution_shadows_match(curve: AbstractCurve, /) -> bool:
    if isinstance(curve, AffinePCurve):
        expected_matrix = _rational_numeric_shadow(curve.matrix)
        expected_offset = _rational_numeric_shadow(curve.offset)
        actual_matrix = np.asarray(curve.matrix_value, dtype=np.float64)
        actual_offset = np.asarray(curve.offset_value, dtype=np.float64)
        return (
            expected_matrix is not None
            and expected_offset is not None
            and actual_matrix.shape == expected_matrix.shape
            and actual_offset.shape == expected_offset.shape
            and np.array_equal(actual_matrix, expected_matrix)
            and np.array_equal(actual_offset, expected_offset)
            and _affine_execution_matrix_is_nonsingular(actual_matrix)
            and _pcurve_execution_shadows_match(curve.curve)
        )
    if isinstance(curve, PeriodicPCurve):
        try:
            expected = np.asarray(
                [shift * (2.0 * np.pi) for shift in curve.period_shifts],
                dtype=np.float64,
            )
        except (OverflowError, ValueError):
            return False
        if not np.all(np.isfinite(expected)):
            return False
        actual = np.asarray(curve.offset_value, dtype=np.float64)
        return (
            actual.shape == expected.shape
            and np.array_equal(actual, expected)
            and _pcurve_execution_shadows_match(curve.source_curve)
        )
    return True


def pcurve_periodic_source(
    curve: AbstractCurve,
    patch: AbstractSurfacePatch,
    /,
) -> tuple[AbstractCurve, tuple[int, int]] | None:
    if not _pcurve_execution_shadows_match(curve):
        return None
    shifts = [0, 0]
    while isinstance(curve, PeriodicPCurve):
        if canonical_fingerprint(carrier_payload(curve.patch)) != canonical_fingerprint(
            carrier_payload(patch)
        ):
            return None
        shifts = [a + b for a, b in zip(shifts, curve.period_shifts, strict=True)]
        curve = curve.source_curve
    if isinstance(curve, AffinePCurve):
        inner_result = pcurve_periodic_source(curve.curve, patch)
        if inner_result is None:
            return None
        source, inner = inner_result
        if any(inner):
            return None
        return source, (0, 0)
    if any(
        shift and symbol != "two_pi"
        for shift, symbol in zip(shifts, patch.native_period_symbols(), strict=True)
    ):
        return None
    return curve, (shifts[0], shifts[1])


def pcurve_surface_period_shifts(
    first: AbstractCurve,
    second: AbstractCurve,
    expected_matrix: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    expected_offset: tuple[Fraction, Fraction],
    patch: AbstractSurfacePatch,
    /,
) -> tuple[int, int] | None:
    if (
        not _pcurve_execution_shadows_match(first)
        or pcurve_periodic_source(second, patch) is None
    ):
        return None
    correspondence = pcurve_affine_correspondence(first, second)
    if correspondence is None or correspondence[0] != expected_matrix:
        return None
    shifts = []
    for actual, expected, symbol in zip(
        correspondence[1], expected_offset, patch.native_period_symbols(), strict=True
    ):
        delta = chart_add(actual, -expected)
        if isinstance(delta, PeriodOffset):
            if delta.rational or delta.turns.denominator != 1 or symbol != "two_pi":
                return None
            shifts.append(int(delta.turns))
        elif delta:
            return None
        else:
            shifts.append(0)
    return shifts[0], shifts[1]


def trim_source_join(first: AbstractCurve, second: AbstractCurve, /) -> bool:
    """Join exact source endpoints only when curve atoms and endpoint roots agree."""

    def oriented_end(
        curve: AbstractCurve, start: bool
    ) -> tuple[AbstractCurve, _BooleanRootEndpoint | None, Fraction] | None:
        # Transports keep their exact chart map over the unwrapped trim source,
        # whose parameter is the one ``nested`` reports.
        if isinstance(curve, AffinePCurve):
            nested = oriented_end(curve.curve, start)
            if nested is None:
                return None
            source = (
                curve
                if nested[0] is curve.curve
                else AffinePCurve(
                    nested[0],
                    np.asarray(curve.matrix, dtype=np.object_),
                    np.asarray(curve.offset, dtype=np.object_),
                )
            )
            return source, nested[1], nested[2]
        if isinstance(curve, PeriodicPCurve):
            nested = oriented_end(curve.source_curve, start)
            if nested is None:
                return None
            source = (
                curve
                if nested[0] is curve.source_curve
                else PeriodicPCurve(nested[0], curve.patch, curve.period_shifts)
            )
            return source, nested[1], nested[2]
        if not isinstance(curve, CurveTrimSegment):
            return None
        lower_root, upper_root = curve.first_root, curve.last_root
        start_root = upper_root if curve.reversed else lower_root
        end_root = lower_root if curve.reversed else upper_root
        parameter = (
            (curve.last if curve.reversed else curve.first)
            if start
            else (curve.first if curve.reversed else curve.last)
        )
        root = start_root if start else end_root
        return (
            curve.curve,
            None if root is None else _require_boolean_root_endpoint(root),
            Fraction(parameter),
        )

    left, right = oriented_end(first, False), oriented_end(second, True)
    if left is None or right is None:
        return False
    correspondence = pcurve_affine_correspondence(left[0], right[0])
    if correspondence is None:
        return False
    left_root, right_root = left[1], right[1]
    if left_root is None or right_root is None:
        return left_root is right_root and left[2] == right[2]
    if isinstance(left_root, NativePeriodEndpoint) and isinstance(
        right_root, NativePeriodEndpoint
    ):
        return left_root.exact_parameter == right_root.exact_parameter
    return same_root_endpoint(left_root, right_root) and root_parameter_in_source(
        left[0], left_root
    ) == root_parameter_in_source(right[0], right_root)


BRepBooleanOperation: TypeAlias = Literal["union", "intersection", "difference"]


type Cell = tuple[int, int, int]


def resolve_node(node: str, aliases: Mapping[str, str], /) -> str:
    """Return the representative of an arrangement node alias chain."""
    seen: set[str] = set()
    while node in aliases:
        if node in seen:
            raise BRepBooleanFailure(
                "BRepBoolean.node_aliases must be acyclic; got a cyclic alias graph"
            )
        seen.add(node)
        node = aliases[node]
    return node


def merge_node_aliases(aliases: dict[str, str], first: str, second: str, /) -> None:
    """Merge alias classes using their lexicographically smallest representative."""
    first = resolve_node(first, aliases)
    second = resolve_node(second, aliases)
    if first != second:
        aliases[max(first, second)] = min(first, second)


class BRepBooleanPolicy(StrictModule):
    """Scientific Boolean controls; tessellation is an explicit optional realization."""

    sewing: BRepSewingPolicy
    tessellation: BRepTessellationPolicy | None
    intersection: ParametricIntersectionPolicy
    construction_tolerance: float = eqx.field(static=True)
    trim_resolution: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        sewing: BRepSewingPolicy | None = None,
        tessellation: BRepTessellationPolicy | None = None,
        intersection: ParametricIntersectionPolicy | None = None,
        construction_tolerance: float = 1.0e-9,
        trim_resolution: float = 0.025,
    ) -> None:
        sewing_ = BRepSewingPolicy() if sewing is None else sewing
        intersection_ = (
            ParametricIntersectionPolicy() if intersection is None else intersection
        )
        for name, value in (
            ("BRepBooleanPolicy.construction_tolerance", construction_tolerance),
            ("BRepBooleanPolicy.trim_resolution", trim_resolution),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value!r}")
        self.sewing = sewing_
        self.tessellation = tessellation
        self.intersection = intersection_
        self.construction_tolerance = float(construction_tolerance)
        self.trim_resolution = float(trim_resolution)


class BRepBooleanFailure(RuntimeError):
    """Report a scientific failure with the input model revisions."""

    def __init__(self, reason: str, source_entities: tuple[str, ...] = ()) -> None:
        self.reason = reason
        self.source_entities = source_entities
        super().__init__(
            f"Native CAD Boolean unresolved: {reason}; sources={source_entities}."
        )


class BRepBooleanResult(StrictModule):
    """Return the Boolean model and its exhaustive ancestry, including empty sets."""

    model: BRepModel
    association_graph: AssociationGraph = eqx.field(static=True)
    operation: BRepBooleanOperation = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    deleted_source_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: BRepModel,
        graph: AssociationGraph,
        operation: BRepBooleanOperation,
        certificate_id: str,
        deleted_source_ids: tuple[str, ...],
    ) -> None:
        self.model = model
        self.association_graph = graph
        self.operation = parse(operation, BRepBooleanOperation, "operation")
        self.certificate_id = certificate_id
        self.deleted_source_ids = deleted_source_ids

    @property
    def empty(self) -> bool:
        return self.model.topology.num_solids == 0


type ChartScalar = Fraction | PeriodOffset


def chart_add(first: ChartScalar, second: ChartScalar) -> ChartScalar:
    left = first if isinstance(first, PeriodOffset) else PeriodOffset(first, Fraction(0))
    right = (
        second if isinstance(second, PeriodOffset) else PeriodOffset(second, Fraction(0))
    )
    value = left + right
    if not isinstance(value, (Fraction, PeriodOffset)):
        raise TypeError(
            f"face_cells.chart_sum must be a Fraction or PeriodOffset; got {type(value).__name__}"
        )
    return value


def chart_scale(value: ChartScalar, factor: Fraction) -> ChartScalar:
    source = (
        value if isinstance(value, PeriodOffset) else PeriodOffset(value, Fraction(0))
    )
    result = source * factor
    if not isinstance(result, (Fraction, PeriodOffset)):
        raise TypeError(
            f"face_cells.chart_product must be a Fraction or PeriodOffset; got {type(result).__name__}"
        )
    return result


def chart_parts(value: ChartScalar) -> tuple[Fraction, Fraction]:
    return (
        (value, Fraction(0))
        if isinstance(value, Fraction)
        else (value.rational, value.turns)
    )


def line_chart_coefficients(
    curve: AbstractCurve,
) -> tuple[tuple[ChartScalar, ...], tuple[Fraction, ...]] | None:
    """Exact ``origin + t * direction`` UV coefficients in Q + Q*(2*pi), or None."""
    from ._source_correspondence import _affine_curve_coefficients

    coefficients = _affine_curve_coefficients(curve, (Fraction(0), Fraction(0)))
    if coefficients is not None:
        try:
            origin, direction = (
                tuple(
                    rational_coefficient(value, role="line chart coefficient")
                    for value in row
                )
                for row in coefficients
            )
        except ValueError:
            return None
        return origin, direction
    if isinstance(curve, PeriodicPCurve):
        child = line_chart_coefficients(curve.source_curve)
        if child is None:
            return None
        origin, direction = child
        return (
            tuple(
                chart_add(value, PeriodOffset(Fraction(0), Fraction(shift)))
                for value, shift in zip(origin, curve.period_shifts, strict=True)
            ),
            direction,
        )
    if isinstance(curve, AffinePCurve):
        child = line_chart_coefficients(curve.curve)
        if child is None:
            return None
        origin, direction = child
        transformed = []
        for row in range(2):
            value: ChartScalar = curve.offset[row]
            for column in range(2):
                value = chart_add(
                    value, chart_scale(origin[column], curve.matrix[row][column])
                )
            transformed.append(value)
        return (
            tuple(transformed),
            tuple(
                sum((curve.matrix[i][j] * direction[j] for j in range(2)), Fraction(0))
                for i in range(2)
            ),
        )
    return None


def _chart_sign(value: ChartScalar, /) -> int | None:
    """Exact sign of ``q + t * 2 pi``; None only when directed bounds straddle zero."""
    rational, turns = chart_parts(value)
    if not turns:
        return (rational > 0) - (rational < 0)
    # A nonzero turn coefficient never cancels a rational: pi is irrational.
    lower, upper = period_scalar_bounds(value)
    if lower > 0.0:
        return 1
    if upper < 0.0:
        return -1
    return None


def _coedge_trim(geometry: BRepGeometry, coedge: int, /) -> CurveTrimSegment:
    edge = geometry.coedge_edges[coedge]
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
    first_root, last_root = geometry.coedge_endpoint_roots[coedge]
    return CurveTrimSegment(
        geometry.pcurves[coedge],
        first,
        last,
        reversed=geometry.coedge_senses[coedge] < 0,
        first_root=first_root,
        last_root=last_root,
    )


def _coedge_axis_values(
    geometry: BRepGeometry, coedge: int, axis: int, /
) -> tuple[ChartScalar, ChartScalar] | None:
    """Exact chart coordinates of an affine p-curve at its authored endpoints."""
    coefficients = line_chart_coefficients(geometry.pcurves[coedge])
    if coefficients is None:
        return None
    origin, direction = coefficients
    edge = geometry.coedge_edges[coedge]
    values: list[ChartScalar] = []
    for value, root in zip(
        np.asarray(geometry.edge_ranges)[edge],
        geometry.coedge_endpoint_roots[coedge],
        strict=True,
    ):
        match root:
            case None:
                parameter: ChartScalar = Fraction(float(value))
            case NativePeriodEndpoint():
                parameter = root.exact_parameter
            case _:
                return None
        values.append(chart_add(origin[axis], chart_scale(parameter, direction[axis])))
    return values[0], values[1]


def _full_period_seam(
    geometry: BRepGeometry,
    face: int,
    axis: int,
    /,
) -> Fraction | None:
    """Prove the face trim spans exactly one native turn from an authored seam.

    The premise is the face's own topology: one edge used by two opposite
    coedges whose exact affine p-curves are one isoline in two native period
    gauges differing by one turn, and every trim coedge on the closed strip
    between those gauges. Endpoint coordinates use authored native-period
    roots; other carriers need a certified enclosure strictly inside the strip.
    Box widths, near-period extents and numerical tolerances never prove it.
    """
    coedges = tuple(coedge for loop in geometry.face_loops[face] for coedge in loop)
    uses: dict[int, list[int]] = {}
    for coedge in coedges:
        uses.setdefault(geometry.coedge_edges[coedge], []).append(coedge)
    turn = PeriodOffset(Fraction(0), Fraction(1))
    seams: set[ChartScalar] = set()
    other = 1 - axis
    for pair in uses.values():
        if (
            len(pair) != 2
            or geometry.coedge_senses[pair[0]] == geometry.coedge_senses[pair[1]]
        ):
            continue
        first_line = line_chart_coefficients(geometry.pcurves[pair[0]])
        second_line = line_chart_coefficients(geometry.pcurves[pair[1]])
        if first_line is None or second_line is None:
            continue
        (first_origin, first_direction), (second_origin, second_direction) = (
            first_line,
            second_line,
        )
        if (
            first_direction != second_direction
            or first_direction[axis] != 0
            or first_direction[other] == 0
            or first_origin[other] != second_origin[other]
        ):
            continue
        difference = chart_add(
            second_origin[axis], chart_scale(first_origin[axis], Fraction(-1))
        )
        if difference == turn:
            seams.add(first_origin[axis])
        elif difference == -turn:
            seams.add(second_origin[axis])
    if len(seams) != 1:
        return None
    seam = next(iter(seams))
    if not isinstance(seam, Fraction) or Fraction(float(seam)) != seam:
        return None
    end = chart_add(seam, turn)
    lower_bounds, upper_bounds = period_scalar_bounds(seam), period_scalar_bounds(end)
    for coedge in coedges:
        values = _coedge_axis_values(geometry, coedge, axis)
        if values is not None:
            # An affine coordinate attains its range at the authored endpoints.
            if any(
                _chart_sign(chart_add(value, chart_scale(seam, Fraction(-1))))
                not in (0, 1)
                or _chart_sign(chart_add(end, chart_scale(value, Fraction(-1))))
                not in (0, 1)
                for value in values
            ):
                return None
            continue
        box = np.asarray(
            _coedge_trim(geometry, coedge).bounding_box(0.0, 1.0), dtype=np.float64
        )
        if not (box[0, axis] >= lower_bounds[1] and box[1, axis] <= upper_bounds[0]):
            return None
    return seam


@checked
def boolean_face_chart_box(
    model: BRepModel,
    face: int,
    /,
) -> np.ndarray:
    """Return the source face chart with only proved full-period seam adjustment."""
    box = np.array(np.asarray(model.parameter_bounds, dtype=np.float64)[face])
    geometry = model.geometry
    if geometry is None:
        return box
    patch = model.patches[face]
    symbols = patch.native_period_symbols()
    for axis, period in enumerate(patch.periods):
        if period is None or symbols[axis] != "two_pi":
            continue
        seam = _full_period_seam(geometry, face, axis)
        if seam is None:
            continue
        lower = float(seam)
        upper = lower + float(period)
        if box[0, axis] <= lower and upper <= box[1, axis]:
            box[0, axis], box[1, axis] = lower, upper
    return box


class _Rectangle(StrictModule):
    model: int = eqx.field(static=True)
    solid: int = eqx.field(static=True)
    face: int = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    coordinate: float = eqx.field(static=True)
    lower: tuple[float, float, float] = eqx.field(static=True)
    upper: tuple[float, float, float] = eqx.field(static=True)


def _rectilinear_coedge_vertex(
    model: BRepModel,
    geometry: BRepGeometry,
    points: np.ndarray,
    coedge: int,
) -> np.ndarray:
    edge = geometry.coedge_edges[coedge]
    curve_index = geometry.edge_curves[edge]
    if curve_index < 0:
        raise BRepBooleanFailure(
            "BRepBoolean.coedge.curve must have an explicit line carrier; got an implicit nonlinear trim",
            (f"{model.source_revision}:edge:{edge}",),
        )
    first, last = geometry.edge_vertices[edge]
    source_curve = geometry.curves[curve_index]
    curve = source_curve if isinstance(source_curve, LineCurve) else None
    if curve is None:
        raise BRepBooleanFailure(
            "BRepBoolean.coedge.curve must admit an affine line descriptor; got a nonlinear carrier"
        )
    direction = np.asarray(curve.direction)
    origin = np.asarray(curve.origin)
    ranges = np.asarray(geometry.edge_ranges)[edge]
    if np.count_nonzero(direction) != 1 or any(
        Fraction(float(origin[d]))
        + Fraction(float(parameter)) * Fraction(float(direction[d]))
        != Fraction(float(points[vertex, d]))
        for parameter, vertex in zip(ranges, (first, last), strict=True)
        for d in range(3)
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.coedge.endpoints must match rational line evaluation on one axis; got incompatible coordinates",
            (f"{model.source_revision}:edge:{edge}",),
        )
    return points[first if geometry.coedge_senses[coedge] > 0 else last]


def _prepare_rectilinear_face(
    model_index: int,
    model: BRepModel,
    geometry: BRepGeometry,
    points: np.ndarray,
    solid: int,
    face: int,
) -> _Rectangle:
    patch = model.patches[face]
    loops = geometry.face_loops[face]
    if not isinstance(patch, PlanePatch) or len(loops) != 1:
        raise BRepBooleanFailure(
            "BRepBoolean.rectilinear_face must use a planar carrier with one loop; got another face family",
            (f"{model.source_revision}:face:{face}",),
        )
    coedges = loops[0]
    if len(coedges) != 4:
        raise BRepBooleanFailure(
            "BRepBoolean.rectilinear_face.coedges must contain four lines; got another coedge count",
            (f"{model.source_revision}:face:{face}",),
        )
    vertices_ = np.stack(
        [
            _rectilinear_coedge_vertex(model, geometry, points, coedge)
            for coedge in coedges
        ]
    )
    lower, upper = vertices_.min(axis=0), vertices_.max(axis=0)
    fixed = np.flatnonzero(lower == upper)
    if fixed.size != 1:
        raise BRepBooleanFailure(
            "BRepBoolean.rectilinear_face.vertices must have one fixed coordinate; got another fixed-axis count",
            (f"{model.source_revision}:face:{face}",),
        )
    axis = int(fixed[0])
    if (
        float(patch.origin[axis]) != float(lower[axis])
        or float(patch.first_axis[axis]) != 0.0
        or float(patch.second_axis[axis]) != 0.0
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.rectilinear_face.patch must support its rectangle coordinates; got incompatible chart axes or origin",
            (f"{model.source_revision}:face:{face}",),
        )
    changing = tuple(index for index in range(3) if index != axis)
    expected = {
        tuple(point)
        for point in product(
            *((lower[d], upper[d]) if d in changing else (lower[d],) for d in range(3))
        )
    }
    differences = np.roll(vertices_, -1, axis=0) - vertices_
    if {tuple(point) for point in vertices_} != expected or np.any(
        np.count_nonzero(differences, axis=1) != 1
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.rectilinear_face.vertices must form an axis-aligned rectangle; got incompatible corners or edges",
            (f"{model.source_revision}:face:{face}",),
        )
    return _Rectangle(
        model_index,
        solid,
        face,
        axis,
        float(lower[axis]),
        (float(lower[0]), float(lower[1]), float(lower[2])),
        (float(upper[0]), float(upper[1]), float(upper[2])),
    )


def rectangles(
    models: tuple[BRepModel, ...],
    selected_solids: tuple[frozenset[int], ...] | None = None,
) -> tuple[_Rectangle, ...]:
    rectangles = []
    for model_index, model in enumerate(models):
        geometry = model.geometry
        if geometry is None or (not geometry.solid_shells and model.patches):
            raise BRepBooleanFailure(
                "BRepBoolean.model.geometry must define closed-solid shells; got missing geometry or shells",
                (model.model_id,),
            )
        points = np.asarray(geometry.vertex_points)
        for solid, faces in enumerate(model.topology.solid_faces):
            if selected_solids is not None and solid not in selected_solids[model_index]:
                continue
            signs = np.asarray(model.orientation).copy()
            for face, sense in zip(
                faces, model.topology.solid_face_orientations[solid], strict=True
            ):
                signs[face] *= sense
            shells(geometry, faces, signs)
            for face in faces:
                rectangles.append(
                    _prepare_rectilinear_face(
                        model_index, model, geometry, points, solid, face
                    )
                )
    return tuple(rectangles)


def membership(
    point: tuple[Fraction, Fraction, Fraction],
    rectangles: tuple[_Rectangle, ...],
    models: tuple[BRepModel, ...],
) -> tuple[tuple[int, int], ...]:
    crossings: dict[tuple[int, int], int] = {}
    for rectangle in rectangles:
        if rectangle.axis != 0 or Fraction(rectangle.coordinate) <= point[0]:
            continue
        if all(
            Fraction(rectangle.lower[d]) < point[d] < Fraction(rectangle.upper[d])
            for d in (1, 2)
        ):
            key = rectangle.model, rectangle.solid
            crossings[key] = crossings.get(key, 0) + 1
    return tuple(
        (model, solid)
        for model, carrier in enumerate(models)
        for solid in range(carrier.topology.num_solids)
        if crossings.get((model, solid), 0) % 2
    )


def selected(
    operation: BRepBooleanOperation, members: tuple[tuple[int, int], ...]
) -> bool:
    first = any(model == 0 for model, _ in members)
    second = any(model == 1 for model, _ in members)
    match operation:
        case "union":
            return first or second
        case "intersection":
            return first and second
        case "difference":
            return first and not second
        case _:
            raise ValueError(
                f"BRepBoolean.operation must be union, intersection, or difference; got {operation!r}"
            )


def precedence_owner(
    precedence: tuple[str, ...],
    voids: tuple[tuple[str, tuple[str, ...]], ...],
    present: frozenset[str],
    /,
) -> str | None:
    """Return the highest-precedence material owner among present operands.

    Ownership is decided first: the highest-precedence present region wins.
    A present void (``(void operand ID, targeted region IDs)``) that targets
    that winner empties the material; a lower present region never refills it.
    Region IDs are their region operand IDs. Planar and solid partitions share
    this rule.
    """
    owner = next((region for region in precedence if region in present), None)
    if owner is not None and any(
        void in present and owner in targets for void, targets in voids
    ):
        return None
    return owner


class RegionSelectionPayload(TypedDict):
    kind: Literal["precedence-region-partition"]
    operands: tuple[str, ...]
    precedence: tuple[str, ...]
    voids: tuple[tuple[str, tuple[str, ...]], ...]
    solids: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class RegionSelection:
    """Partition ownership of open solid material by declared region precedence.

    ``operand_ids`` names the arrangement operands by index; ``precedence`` and
    ``voids`` follow ``precedence_owner``. Only the declared
    ``(operand index, solid)`` selections participate.
    """

    operand_ids: tuple[str, ...]
    precedence: tuple[str, ...]
    voids: tuple[tuple[str, tuple[str, ...]], ...]
    solids: tuple[tuple[int, int], ...]

    @property
    def payload(self) -> RegionSelectionPayload:
        return {
            "kind": "precedence-region-partition",
            "operands": self.operand_ids,
            "precedence": self.precedence,
            "voids": self.voids,
            "solids": self.solids,
        }


type MaterialSelection = BRepBooleanOperation | RegionSelection


def material_owner(
    selection: MaterialSelection, members: tuple[tuple[int, int], ...]
) -> str | None:
    """Return the material owner whose open neighborhood lies in ``members``.

    A Boolean owns its selected material under the operation name; ``None`` is
    empty space. Two sides with different owners bound published material.
    """
    match selection:
        case RegionSelection():
            selected_bank = set(selection.solids)
            present = frozenset(
                selection.operand_ids[model]
                for model, solid in members
                if (model, solid) in selected_bank
            )
            return precedence_owner(selection.precedence, selection.voids, present)
        case str():
            return selection if selected(selection, members) else None
        case _:
            raise ValueError(
                f"BRepBoolean.material.selection must be a Boolean operation or RegionSelection; got {selection!r}"
            )


def selection_payload(
    selection: MaterialSelection,
) -> BRepBooleanOperation | RegionSelectionPayload:
    match selection:
        case RegionSelection():
            return selection.payload
        case str():
            return selection
        case _:
            raise ValueError(
                f"BRepBoolean.material.selection must be a Boolean operation or RegionSelection; got {selection!r}"
            )


def side_payload(
    selection: MaterialSelection,
    negative: str | None,
    positive: str | None,
) -> tuple[str | None, str | None] | tuple[bool, bool]:
    """Boolean sides are identified by selection alone; partition sides by owner."""
    match selection:
        case RegionSelection():
            return negative, positive
        case str():
            return negative is not None, positive is not None
        case _:
            raise ValueError(
                f"BRepBoolean.material.selection must be a Boolean operation or RegionSelection; got {selection!r}"
            )


def components(cells: set[Cell]) -> tuple[tuple[Cell, ...], ...]:
    remaining = set(cells)
    result = []
    while remaining:
        seed = min(remaining)
        queue = deque((seed,))
        remaining.remove(seed)
        component = []
        while queue:
            cell = queue.popleft()
            component.append(cell)
            for axis, step in product(range(3), (-1, 1)):
                candidate = list(cell)
                candidate[axis] += step
                neighbor = candidate[0], candidate[1], candidate[2]
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        result.append(tuple(sorted(component)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class BoundaryFace:
    solid: int
    axis: int
    side: int
    vertices: tuple[Cell, ...]
    sources: tuple[tuple[int, int, int], ...]


def boundary(
    components: tuple[tuple[Cell, ...], ...],
    coordinates: tuple[tuple[float, ...], ...],
    rectangles: tuple[_Rectangle, ...],
) -> tuple[BoundaryFace, ...]:
    faces = []
    for solid, component in enumerate(components):
        cells = set(component)
        for cell in component:
            for axis, side in product(range(3), (-1, 1)):
                neighbor = list(cell)
                neighbor[axis] += side
                if tuple(neighbor) in cells:
                    continue
                first, second = (axis + 1) % 3, (axis + 2) % 3
                plane = cell[axis] + (1 if side > 0 else 0)
                vertices = []
                for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    vertex = list(cell)
                    vertex[axis] = plane
                    vertex[first] += a
                    vertex[second] += b
                    vertices.append((vertex[0], vertex[1], vertex[2]))
                coordinate = coordinates[axis][plane]
                lower = tuple(coordinates[d][vertices[0][d]] for d in range(3))
                upper = tuple(coordinates[d][vertices[2][d]] for d in range(3))
                sources = tuple(
                    sorted(
                        {
                            (rectangle.model, rectangle.solid, rectangle.face)
                            for rectangle in rectangles
                            if rectangle.axis == axis
                            and rectangle.coordinate == coordinate
                            and all(
                                rectangle.lower[d] <= lower[d]
                                and upper[d] <= rectangle.upper[d]
                                for d in (first, second)
                            )
                        }
                    )
                )
                if not sources:
                    raise BRepBooleanFailure(
                        "BRepBoolean.boundary.fragment must retain a source face; got no source face"
                    )
                faces.append(BoundaryFace(solid, axis, side, tuple(vertices), sources))
    return tuple(faces)


def assemble(
    models: tuple[BRepModel, ...],
    coordinates: tuple[tuple[float, ...], ...],
    faces: tuple[BoundaryFace, ...],
    policy: BRepBooleanPolicy,
    certificate: str,
    *,
    shared_faces: bool = False,
) -> BRepModel:
    vertices: list[tuple[float, float, float]] = []
    vertex_ids: dict[tuple[int, Cell], int] = {}
    curves = []
    edge_vertices = []
    edge_ranges = []
    edge_ids: dict[tuple[int, int], int] = {}
    pcurves = []
    coedge_edges = []
    senses = []
    loops = []
    patches = []
    bounds = []
    orientation = []
    solid_count = max((face.solid for face in faces), default=-1) + 1
    solid_faces: list[list[int]] = [[] for _ in range(solid_count)]
    solid_signs: list[list[int]] = [[] for _ in range(solid_count)]
    face_ids: dict[tuple[Cell, ...], int] = {}
    for face in faces:
        face_key = tuple(sorted(face.vertices))
        if shared_faces and face_key in face_ids:
            face_index = face_ids[face_key]
            solid_faces[face.solid].append(face_index)
            solid_signs[face.solid].append(face.side)
            continue
        face_index = len(loops)
        face_ids[face_key] = face_index
        first, second = (face.axis + 1) % 3, (face.axis + 2) % 3
        indices = []
        for grid in face.vertices:
            key = (-1 if shared_faces else face.solid), grid
            if key not in vertex_ids:
                vertex_ids[key] = len(vertices)
                vertices.append(
                    (
                        coordinates[0][grid[0]],
                        coordinates[1][grid[1]],
                        coordinates[2][grid[2]],
                    )
                )
            indices.append(vertex_ids[key])
        loop = []
        for index, start in enumerate(indices):
            end = indices[(index + 1) % 4]
            p0, p1 = vertices[start], vertices[end]
            changing = next(d for d in range(3) if p0[d] != p1[d])
            lo, hi = (start, end) if p0[changing] < p1[changing] else (end, start)
            key = lo, hi
            if key not in edge_ids:
                origin = np.asarray(vertices[lo], dtype=np.float64).copy()
                origin[changing] = 0.0
                direction = np.zeros(3, dtype=np.float64)
                direction[changing] = 1.0
                edge_ids[key] = len(curves)
                curves.append(LineCurve(origin, direction))
                edge_vertices.append(key)
                edge_ranges.append((vertices[lo][changing], vertices[hi][changing]))
            p_origin = np.asarray(
                (vertices[lo][first], vertices[lo][second]), dtype=np.float64
            )
            p_direction = np.zeros(2, dtype=np.float64)
            parameter_axis = 0 if changing == first else 1
            p_origin[parameter_axis] = 0.0
            p_direction[parameter_axis] = 1.0
            pcurves.append(LineCurve(p_origin, p_direction))
            loop.append(len(coedge_edges))
            coedge_edges.append(edge_ids[key])
            senses.append(1 if start == lo else -1)
        loops.append((tuple(loop),))
        origin = np.zeros(3, dtype=np.float64)
        origin[face.axis] = vertices[indices[0]][face.axis]
        patches.append(PlanePatch(origin, np.eye(3)[first], np.eye(3)[second]))
        bounds.append(
            (
                (vertices[indices[0]][first], vertices[indices[0]][second]),
                (vertices[indices[2]][first], vertices[indices[2]][second]),
            )
        )
        orientation.append(face.side)
        solid_faces[face.solid].append(face_index)
        solid_signs[face.solid].append(face.side)
    geometry = BRepGeometry(
        vertex_points=np.asarray(vertices, dtype=np.float64).reshape((-1, 3)),
        vertex_exact_points=tuple(
            (Fraction(point[0]), Fraction(point[1]), Fraction(point[2]))
            for point in vertices
        ),
        vertex_roots=(None,) * len(vertices),
        edge_endpoint_roots=((None, None),) * len(edge_vertices),
        coedge_endpoint_roots=((None, None),) * len(coedge_edges),
        curves=tuple(curves),
        edge_curves=tuple(range(len(curves))),
        edge_ranges=np.asarray(edge_ranges, dtype=np.float64).reshape((-1, 2)),
        edge_vertices=tuple(edge_vertices),
        pcurves=tuple(pcurves),
        coedge_edges=tuple(coedge_edges),
        coedge_senses=tuple(senses),
        face_loops=tuple(loops),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    orientation_array = np.asarray(orientation, dtype=np.float64)
    sewed = sew_brep(
        geometry,
        orientation_array,
        tuple(tuple(group) for group in solid_faces),
        policy=policy.sewing,
        solid_orientations=tuple(tuple(group) for group in solid_signs),
    )
    return assemble_brep_model(
        sewed.geometry,
        tuple(patches),
        np.asarray(bounds, dtype=np.float64).reshape((-1, 2, 2)),
        orientation_array,
        tuple("plane" for _ in patches),
        coordinate_contract=models[0].coordinate_contract,
        source_id=f"native-boolean:{certificate}",
        source_format="native",
        source_digest=certificate,
        import_policy_id=certificate,
        tessellation=policy.tessellation,
    )


@dataclass(frozen=True, slots=True)
class BooleanSourceLineage:
    """Complete operand-scoped CAD inventory and canonical entity occurrence map."""

    revision: CADRevision
    entity_occurrences: Mapping[tuple[int, BRepEntityId], str]
    local_occurrence_ids: Mapping[tuple[int, str], str]

    def entities_for_index(
        self, operand: int, kind: str, index: int, /
    ) -> tuple[BRepEntityId, ...]:
        return tuple(
            sorted(
                entity
                for (entity_operand, entity) in self.entity_occurrences
                if entity_operand == operand
                and entity.kind == kind
                and entity.index == index
            )
        )

    def entities_for_solid(
        self, operand: int, kind: str, index: int, solid: int, /
    ) -> tuple[BRepEntityId, ...]:
        solid_paths = {
            entity.occurrence_path
            for entity in self.entities_for_index(operand, "solid", solid)
        }
        return tuple(
            entity
            for entity in self.entities_for_index(operand, kind, index)
            if entity.occurrence_path in solid_paths
        )

    def occurrence_id(self, operand: int, entity: BRepEntityId, /) -> str:
        try:
            return self.entity_occurrences[(operand, entity)]
        except KeyError as error:
            raise BRepBooleanFailure(
                "BRepBoolean.lineage.source_entity must bind to an exact operand occurrence",
                (
                    source_entity_id(
                        entity.source_id,
                        entity.source_revision,
                        entity.kind,
                        entity.index,
                        occurrence_path=entity.occurrence_path,
                    ),
                ),
            ) from error


def entity_occurrence_index(
    model: BRepModel, revision: CADRevision, /
) -> dict[tuple[str, int], tuple[tuple[BRepEntityId, str], ...]]:
    """Index actual CAD revision occurrences by source definition dimension/row."""
    local = {
        (occurrence.kind, occurrence.entity_id): occurrence.occurrence_id
        for occurrence in revision.occurrences
        if occurrence.kind != "assembly"
    }
    grouped: dict[tuple[str, int], list[tuple[BRepEntityId, str]]] = {}
    for record in model.geometry.qualified_entity_incidence(
        model.source_id, model.source_revision
    ):
        entity = record.member
        if record.container != entity:
            continue
        entity_key = source_entity_id(
            entity.source_id,
            entity.source_revision,
            entity.kind,
            entity.index,
            occurrence_path=entity.occurrence_path,
        )
        occurrence_id = local.get((entity.kind, entity_key))
        if occurrence_id is None:
            raise BRepBooleanFailure(
                "BRepBoolean.lineage.target_entity must exist in its canonical CAD revision",
                (entity_key,),
            )
        grouped.setdefault((entity.kind, entity.index), []).append(
            (entity, occurrence_id)
        )
    return {
        key: tuple(sorted(values, key=lambda item: item[0].occurrence_path))
        for key, values in grouped.items()
    }


def source_assembly_occurrence_id(
    lineage: BooleanSourceLineage,
    operand: int,
    model: BRepModel,
    path: tuple[str, ...],
    /,
) -> str:
    from ._partition import _assembly_namespace_occurrence_id

    local_id = _assembly_namespace_occurrence_id(
        model.source_id, model.source_revision, path
    )
    try:
        return lineage.local_occurrence_ids[(operand, local_id)]
    except KeyError as error:
        raise BRepBooleanFailure(
            "BRepBoolean.lineage.source_assembly must bind to an exact operand occurrence",
            (local_id,),
        ) from error


def target_assembly_occurrence_id(
    revision: CADRevision, model: BRepModel, path: tuple[str, ...], /
) -> str:
    from ._partition import _assembly_namespace_occurrence_id

    local_id = _assembly_namespace_occurrence_id(
        model.source_id, model.source_revision, path
    )
    matches = tuple(
        occurrence.occurrence_id
        for occurrence in revision.occurrences
        if occurrence.kind == "assembly" and occurrence.entity_id == local_id
    )
    if len(matches) != 1:
        raise BRepBooleanFailure(
            "BRepBoolean.lineage.target_assembly must resolve uniquely in its CAD revision",
            (local_id,),
        )
    return matches[0]


def _boolean_operand_occurrence_id(
    operand: int, model: BRepModel, occurrence: CADOccurrence, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "native-boolean-operand-occurrence",
            "operand_index": operand,
            "source_id": model.source_id,
            "source_revision": model.source_revision,
            "source_occurrence_id": occurrence.occurrence_id,
        }
    )


def source_lineage(
    models: tuple[BRepModel, ...], certificate: str, /
) -> BooleanSourceLineage:
    """Compose complete source CAD revisions without inventing entity labels."""
    from ._partition import cad_revision_from_brep_model

    occurrences: list[CADOccurrence] = []
    entity_occurrences: dict[tuple[int, BRepEntityId], str] = {}
    local_occurrence_ids: dict[tuple[int, str], str] = {}
    for operand, model in enumerate(models):
        local_revision = cad_revision_from_brep_model(model)
        occurrence_ids = {
            occurrence.occurrence_id: _boolean_operand_occurrence_id(
                operand, model, occurrence
            )
            for occurrence in local_revision.occurrences
        }
        local_occurrence_ids.update(
            ((operand, local_id), occurrence_id)
            for local_id, occurrence_id in occurrence_ids.items()
        )
        local_entities = {
            (occurrence.kind, occurrence.entity_id): occurrence
            for occurrence in local_revision.occurrences
            if occurrence.kind != "assembly"
        }
        occurrences.extend(
            CADOccurrence(
                certificate,
                occurrence_ids[occurrence.occurrence_id],
                occurrence.entity_id,
                occurrence.kind,
                tuple(occurrence_ids[part] for part in occurrence.path),
                (
                    None
                    if occurrence.parent_occurrence_id is None
                    else occurrence_ids[occurrence.parent_occurrence_id]
                ),
                orientation=occurrence.orientation,
            )
            for occurrence in local_revision.occurrences
        )
        for record in model.geometry.qualified_entity_incidence(
            model.source_id, model.source_revision
        ):
            entity = record.member
            if record.container != entity:
                continue
            entity_key = source_entity_id(
                entity.source_id,
                entity.source_revision,
                entity.kind,
                entity.index,
                occurrence_path=entity.occurrence_path,
            )
            local = local_entities.get((entity.kind, entity_key))
            if local is None:
                raise BRepBooleanFailure(
                    "BRepBoolean.lineage.source_entity must exist in its canonical source revision",
                    (entity_key,),
                )
            entity_occurrences[(operand, entity)] = occurrence_ids[local.occurrence_id]
    return BooleanSourceLineage(
        CADRevision(
            certificate,
            f"native-boolean-sources:{certificate}",
            tuple(occurrences),
            certificate,
        ),
        entity_occurrences,
        local_occurrence_ids,
    )


def rectilinear_family(models: tuple[BRepModel, ...]) -> bool:
    """Admit only solids whose source face carriers prove axis-aligned rectangles."""
    for model_index, model in enumerate(models):
        geometry = model.geometry
        if geometry is None or not model.topology.num_solids:
            return False
        points = np.asarray(geometry.vertex_points, dtype=np.float64)
        for solid, faces in enumerate(geometry.topology.solid_faces):
            for face in faces:
                if type(model.patches[face]) is not PlanePatch:
                    return False
                loops = geometry.face_loops[face]
                if len(loops) != 1 or len(loops[0]) != 4:
                    return False
                for coedge in loops[0]:
                    edge = geometry.coedge_edges[coedge]
                    curve_index = geometry.edge_curves[edge]
                    if (
                        curve_index < 0
                        or type(geometry.curves[curve_index]) is not LineCurve
                    ):
                        return False
                try:
                    _prepare_rectilinear_face(
                        model_index, model, geometry, points, solid, face
                    )
                except BRepBooleanFailure:
                    return False
    return True


__all__ = [
    "BRepBooleanFailure",
    "BRepBooleanOperation",
    "BRepBooleanPolicy",
    "BRepBooleanResult",
]
