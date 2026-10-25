#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Apply authored affine poses to CAD source definitions.

Orthogonality is admitted within the placement tolerance. Coefficients remain
unchanged; common-pose cancellation requires coefficient equality and
binary-rational invertibility.
"""

from __future__ import annotations

from typing import assert_never, final, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core

import phydrax.geometry.brep._intersection_curve as intersection_curve_owner
import phydrax.geometry.brep._patches as patch_owner

from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, ConvertibleToArray, Float64, parse
from .._interval_enclosure import interval_add, interval_multiply
from ._carrier_pieces import CurvePiece, SurfacePiece
from ._coordinate_request import coordinate_source_request
from ._exact import exact_determinant
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BernsteinRestrictionBank,
    MeridianOverlaps,
    surface_box_bank_bytes,
    SURFACE_BOX_FINITE_LIMIT,
    surface_parameter_boxes,
)
from ._request_preparations import TrimRequestPreparations


def validate_pose(
    rotation: ConvertibleToArray, translation: ConvertibleToArray, /
) -> tuple[Array, Array]:
    rotation_host = np.asarray(rotation, dtype=np.float64)
    translation_host = np.asarray(translation, dtype=np.float64)
    if rotation_host.shape != (3, 3) or translation_host.shape != (3,):
        raise ValueError(
            "placement.pose must have rotation shape (3, 3) and translation shape (3,); "
            f"got {rotation_host.shape}/{translation_host.shape}"
        )
    if not np.all(np.isfinite(rotation_host)) or not np.all(
        np.isfinite(translation_host)
    ):
        raise ValueError(
            "placement.pose must have finite rotation and translation coefficients; got nonfinite coefficients"
        )
    if (
        np.max(np.abs(rotation_host.T @ rotation_host - np.eye(3, dtype=np.float64)))
        > 1.0e-12
        or exact_determinant(rotation_host) == 0
    ):
        raise ValueError(
            "placement.rotation must be orthogonal within 1e-12 and invertible; got an inadmissible matrix"
        )
    return (
        parse(
            jnp.asarray(rotation_host, dtype=jnp.float64),
            Float64[Literal[3], Literal[3]],
            "rotation",
        ),
        parse(
            jnp.asarray(translation_host, dtype=jnp.float64),
            Float64[Literal[3]],
            "translation",
        ),
    )


def source_transform_bounds(
    rotation: np.ndarray, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Outward image of source-coordinate intervals, with coordinate axis first.

    The three multiply/add operations use the shared interval substrate. This
    also encloses unbounded source jets without a sampled or nominal inverse.
    """
    if rotation.shape != (3, 3) or lower.shape != upper.shape or lower.shape[0] != 3:
        raise ValueError(
            "source_transform.inputs must contain a 3x3 rotation and matching coordinate-first bounds; "
            f"got {rotation.shape}/{lower.shape}/{upper.shape}"
        )
    coefficient_shape = (3,) + (1,) * (lower.ndim - 1)
    coefficient = rotation[:, 0].reshape(coefficient_shape)
    result = interval_multiply((coefficient, coefficient), (lower[0], upper[0]))
    for axis in (1, 2):
        coefficient = rotation[:, axis].reshape(coefficient_shape)
        result = interval_add(
            result,
            interval_multiply(
                (coefficient, coefficient),
                (lower[axis], upper[axis]),
            ),
        )
    return (
        np.asarray(result[0], dtype=np.float64),
        np.asarray(result[1], dtype=np.float64),
    )


def _placed_box(
    rotation: Array, translation: Array, source_box: np.ndarray, /
) -> np.ndarray:
    bounds = source_transform_bounds(np.asarray(rotation), source_box[0], source_box[1])
    shift = np.asarray(translation)
    low, high = interval_add(bounds, (shift, shift))
    return np.stack((low, high))


# Smallest caller margin for which rounding below the binary64 normal range
# cannot matter in the definition stop derivation below.
_DEFINITION_STOP_MARGIN = 2.0**-900


def _definition_stop_magnitude(
    rotation: np.ndarray, translation: np.ndarray, stop_magnitude: float, /
) -> float:
    """Definition magnitude whose finite boxes place strictly within ``stop_magnitude``.

    With ``r = max|rotation|``, ``a = max|translation|`` and definition bounds at
    most ``(stop_magnitude - 2a) / (8r)``, every rounded-out product, sum and
    translation stays below ``(3 stop_magnitude / 8 + a / 4)(1 + 2**-47)``, which
    is less than ``stop_magnitude``. Otherwise no definition row is admitted
    before its own placement, so placed rows are enclosed one at a time.
    """
    margin = stop_magnitude - 2.0 * float(np.max(np.abs(translation)))
    if not (np.isfinite(margin) and margin >= _DEFINITION_STOP_MARGIN):
        return -1.0
    return margin / (8.0 * float(np.max(np.abs(rotation))))


def _placed_boxes_scratch_bytes(count: int, /) -> int:
    """Bound the coordinate-first interval temporaries of ``count`` placed boxes.

    Each of the at most 16 simultaneously live endpoint, product, minimum,
    maximum, outward-step, sum and mask arrays holds at most ``3 * count``
    binary64 values; the bank's ``(count, 2, 3)`` stop magnitudes occupy two.
    """
    return 4096 + 16 * (128 + 24 * count)


def _bind_pose(
    carrier: PlacedSurface | PlacedCurve,
    rotation: ConvertibleToArray,
    translation: ConvertibleToArray,
    /,
) -> None:
    carrier.rotation, carrier.translation = validate_pose(rotation, translation)


@final
class PlacedSurface(AbstractSurfacePatch):
    """Surface definition and immutable authored world-pose operation tree."""

    __strict_contract__ = True
    definition: AbstractSurfacePatch
    rotation: Float64[Literal[3], Literal[3]]
    translation: Float64[Literal[3]]

    @checked
    def __init__(
        self,
        definition: AbstractSurfacePatch,
        rotation: ConvertibleToArray,
        translation: ConvertibleToArray,
    ) -> None:
        _bind_pose(self, rotation, translation)
        self.definition = definition

    def evaluate(self, parameters: Array, /) -> Array:
        return self.definition.evaluate(parameters) @ self.rotation.T + self.translation

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.definition.periods

    @checked
    def source_pieces(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[SurfacePiece[AbstractSurfacePatch], ...]:
        """Restrict definition pieces and preserve their posed source identities."""
        return tuple(
            SurfacePiece(
                PlacedSurface(piece.evaluator, self.rotation, self.translation),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in self.definition.source_pieces(parameter_box, budget=budget)
        )

    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Return definition coefficient bounds without altering pose coefficients."""
        return patch_owner.coefficient_enclosures(self.definition)

    def native_period_symbols(self) -> tuple[str | None, str | None]:
        """Return the definition's symbolic periods on both parameter axes."""
        return self.definition.native_period_symbols()

    def degenerate_isolines(
        self, parameter_box: ConvertibleToArray, /
    ) -> tuple[tuple[int, float], ...]:
        return self.definition.degenerate_isolines(parameter_box)

    def validate_parameter_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.validate_parameter_box(parameter_box, budget=budget)
        return self.definition.validate_parameter_box(parameter_box, budget=budget)

    def is_c1_on(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.is_c1_on(
                    parameter_box, preparations=preparations, budget=budget
                )
        return self.definition.is_c1_on(
            parameter_box, preparations=preparations, budget=budget
        )

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        return _placed_box(
            self.rotation,
            self.translation,
            self.definition.bounding_box(
                parameter_box, preparations=preparations, budget=budget
            ),
        )

    def bounding_boxes(
        self,
        parameter_boxes: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        stop_magnitude: float = SURFACE_BOX_FINITE_LIMIT,
    ) -> np.ndarray:
        """Place definition boxes with one host pose and the scalar operation order.

        Definition rows stop at a magnitude whose finite boxes cannot reach
        ``stop_magnitude`` after placement, so only that stopped row can end
        this bank and no definition row after the first stopping placed row is
        evaluated. Nested placements recurse in their authored order.
        """
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_boxes,
                )
            ) as budget:
                return self.bounding_boxes(
                    parameter_boxes,
                    preparations=preparations,
                    budget=budget,
                    stop_magnitude=stop_magnitude,
                )
        boxes = surface_parameter_boxes(parameter_boxes)
        count = boxes.shape[0]
        budget.reserve(0, surface_box_bank_bytes(count))
        result = np.empty((count, 2, 3), dtype=np.float64)
        definition_boxes = type(self.definition).bounding_boxes
        with budget.temporary_scope():
            budget.reserve(0, 512)
            rotation = np.asarray(self.rotation)
            translation = np.asarray(self.translation)
            definition_stop = _definition_stop_magnitude(
                rotation, translation, stop_magnitude
            )
            shift = translation[:, None]
            start = 0
            while start < count:
                with budget.temporary_scope():
                    definition = definition_boxes(
                        self.definition, boxes[start:],
                        preparations=preparations,
                        budget=budget,
                        stop_magnitude=definition_stop,
                    )
                    stop = start + definition.shape[0]
                    budget.reserve(0, _placed_boxes_scratch_bytes(definition.shape[0]))
                    low, high = interval_add(
                        source_transform_bounds(
                            rotation, definition[:, 0].T, definition[:, 1].T
                        ),
                        (shift, shift),
                    )
                    del definition
                    result[start:stop, 0] = low.T
                    result[start:stop, 1] = high.T
                    del low, high
                    stopped = np.flatnonzero(
                        ~np.all(np.abs(result[start:stop]) <= stop_magnitude, axis=(1, 2))
                    )
                if stopped.size:
                    return result[: start + int(stopped[0]) + 1]
                start = stop
        return result

    def derivative_bounds_batch(
        self,
        parameter_boxes: ConvertibleToArray,
        /,
        *,
        order: int = 1,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_boxes,
                )
            ) as budget:
                return self.derivative_bounds_batch(
                    parameter_boxes,
                    order=order,
                    preparations=preparations,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                    budget=budget,
                )
        low, high = type(self.definition).derivative_bounds_batch(
            self.definition, parameter_boxes,
            order=order,
            preparations=preparations,
            budget=budget,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        low, high = source_transform_bounds(
            np.asarray(self.rotation), np.moveaxis(low, 1, 0), np.moveaxis(high, 1, 0)
        )
        return np.moveaxis(low, 0, 1), np.moveaxis(high, 0, 1)


@final
class PlacedCurve(AbstractCurve):
    """Three-dimensional curve definition and authored pose operation tree."""

    __strict_contract__ = True
    definition: AbstractCurve | intersection_curve_owner.IntersectionCurve
    rotation: Float64[Literal[3], Literal[3]]
    translation: Float64[Literal[3]]

    @checked
    def __init__(
        self,
        definition: AbstractCurve | intersection_curve_owner.IntersectionCurve,
        rotation: ConvertibleToArray,
        translation: ConvertibleToArray,
    ) -> None:
        match definition:
            case AbstractCurve() if definition.ambient_dimension != 3:
                raise ValueError(
                    f"placed_curve.definition must be three-dimensional; got {definition.ambient_dimension}"
                )
            case _:
                pass
        _bind_pose(self, rotation, translation)
        self.definition = definition

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def period(self) -> float | None:
        match self.definition:
            case AbstractCurve() as curve:
                return curve.period
            case intersection_curve_owner.IntersectionCurve() as curve:
                return float(curve.num_charts) if curve.closed else None
            case unreachable:
                assert_never(unreachable)

    @checked
    def source_pieces(
        self,
        first: float,
        last: float,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> tuple[CurvePiece[AbstractCurve], ...]:
        """Restrict definition pieces without changing interval or chart ordering."""
        return tuple(
            CurvePiece(
                PlacedCurve(piece.evaluator, self.rotation, self.translation),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in self.definition.source_pieces(
                first, last, coordinate_budget=coordinate_budget
            )
        )

    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Return coefficient bounds of the curve definition."""
        return patch_owner.coefficient_enclosures(self.definition)

    def native_period_symbol(self) -> str | None:
        """Return the definition's symbolic parameter period."""
        return self.definition.native_period_symbol()

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        match self.definition:
            case AbstractCurve() as curve:
                return curve.parameter_domain
            case intersection_curve_owner.IntersectionCurve() as curve:
                return 0.0, float(curve.num_charts)
            case unreachable:
                assert_never(unreachable)

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        return self.definition.validate_range(first, last)

    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.is_c1_on(
                    first, last, preparations=preparations, budget=budget
                )
        return self.definition.is_c1_on(
            first, last, preparations=preparations, budget=budget
        )

    def evaluate(self, parameters: Array, /) -> Array:
        match self.definition:
            case intersection_curve_owner.IntersectionCurve() as curve:
                if not curve.fully_certified:
                    raise ValueError(
                        "placed_curve.definition must have certified source charts; got uncertified charts"
                    )
                result = curve.evaluate(parameters)
                if not isinstance(result.parameter_bound, jax_core.Tracer) and not np.all(
                    np.isfinite(np.asarray(result.parameter_bound))
                ):
                    raise ValueError(
                        "placed_curve.query.parameter_bound must be finite; got unresolved source bounds"
                    )
                points = result.point
            case AbstractCurve() as curve:
                points = curve.evaluate(parameters)
            case unreachable:
                assert_never(unreachable)
        return points @ self.rotation.T + self.translation

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        return _placed_box(
            self.rotation,
            self.translation,
            self.definition.bounding_box(
                first, last, preparations=preparations, budget=budget
            ),
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        endpoint_roots: tuple[
            intersection_curve_owner.RootEndpoint | None,
            intersection_curve_owner.RootEndpoint | None,
        ]
        | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate definition jets; only a coupled branch takes extension roots."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.derivative_bounds(
                    first,
                    last,
                    order=order,
                    endpoint_roots=endpoint_roots,
                    preparations=preparations,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                    budget=budget,
                )
        match self.definition:
            case intersection_curve_owner.IntersectionCurve() as branch:
                low, high = branch.derivative_bounds(
                    first,
                    last,
                    order=order,
                    endpoint_roots=endpoint_roots,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
            case definition:
                if endpoint_roots is not None:
                    raise ValueError(
                        "PlacedCurve.endpoint_roots must bind a coupled intersection branch; "
                        f"got {type(definition).__name__}"
                    )
                low, high = definition.derivative_bounds(
                    first,
                    last,
                    order=order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
        return source_transform_bounds(np.asarray(self.rotation), low, high)


@checked
def same_source_pose(
    first: PlacedSurface | PlacedCurve, second: PlacedSurface | PlacedCurve, /
) -> bool:
    """Test equal pose coefficients and binary-rational invertibility.

    Definition identity is independent of pose equality.
    """
    rotation = np.asarray(first.rotation)
    translation = np.asarray(first.translation)
    return bool(
        np.array_equal(
            rotation.view(np.uint64), np.asarray(second.rotation).view(np.uint64)
        )
        and np.array_equal(
            translation.view(np.uint64), np.asarray(second.translation).view(np.uint64)
        )
        and np.all(np.isfinite(rotation))
        and np.all(np.isfinite(translation))
        and exact_determinant(rotation) != 0
    )
