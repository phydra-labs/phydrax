#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...metrix import CoordinateAtlas


class DivisorChart(StrictModule):
    defining_function: Callable[[Array], Array]
    chart_index: int = eqx.field(static=True)
    multiplicity: int = eqx.field(static=True)
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        chart_index: int,
        defining_function: Callable[[Array], Array],
        /,
        *,
        multiplicity: int = 1,
        component_id: str,
    ):
        if (
            int(chart_index) < 0
            or not callable(defining_function)
            or int(multiplicity) < 1
            or not component_id
        ):
            raise ValueError(
                "Divisor chart index/function/multiplicity/component_id are invalid."
            )
        self.chart_index = int(chart_index)
        self.defining_function = defining_function
        self.multiplicity = int(multiplicity)
        self.component_id = str(component_id)


class DivisorClearanceEvidence(StrictModule):
    lower_bounds: Array
    sampled: Array
    certified: Array
    clear: Array

    def __init__(
        self,
        lower_bounds: ArrayLike,
        /,
        *,
        sampled: ArrayLike,
        certified: ArrayLike,
        clear: ArrayLike,
    ):
        self.lower_bounds = jnp.asarray(lower_bounds)
        self.sampled = jnp.asarray(sampled, dtype=bool)
        self.certified = jnp.asarray(certified, dtype=bool)
        self.clear = jnp.asarray(clear, dtype=bool)


class DivisorIntersection(StrictModule):
    point: Array
    jacobian_rank: Array
    expected_rank: int = eqx.field(static=True)
    transverse: Array
    valid: Array

    def __init__(
        self,
        point: ArrayLike,
        jacobian_rank: ArrayLike,
        expected_rank: int,
        transverse: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        self.point = jnp.asarray(point)
        self.jacobian_rank = jnp.asarray(jacobian_rank, dtype=jnp.int32)
        self.expected_rank = int(expected_rank)
        self.transverse = jnp.asarray(transverse, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)


class CartierDivisor(StrictModule):
    """Finite-chart Cartier data with explicit nowhere-zero overlap units."""

    atlas: CoordinateAtlas
    charts: tuple[DivisorChart, ...]
    overlap_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    overlap_units: tuple[Callable[[Array], Array], ...]
    tolerance: float = eqx.field(static=True)
    divisor_id: str = eqx.field(static=True)

    def __init__(
        self,
        atlas: CoordinateAtlas,
        charts: Sequence[DivisorChart],
        overlap_units: Mapping[tuple[int, int], Callable[[Array], Array]],
        /,
        *,
        tolerance: float = 1e-8,
        divisor_id: str,
    ):
        charts_ = tuple(charts)
        if (
            not isinstance(atlas, CoordinateAtlas)
            or not charts_
            or any(not isinstance(value, DivisorChart) for value in charts_)
        ):
            raise TypeError(
                "CartierDivisor requires one atlas and declared divisor charts."
            )
        if len({value.chart_index for value in charts_}) != len(charts_) or any(
            value.chart_index >= len(atlas.charts) for value in charts_
        ):
            raise ValueError(
                "Divisor chart indices must be unique and owned by the atlas."
            )
        pairs = tuple(
            sorted((int(first), int(second)) for first, second in overlap_units)
        )
        units = tuple(overlap_units[pair] for pair in pairs)
        if any(first == second for first, second in pairs) or any(
            not callable(value) for value in units
        ):
            raise ValueError(
                "Cartier overlap units must be callable on distinct chart pairs."
            )
        if float(tolerance) <= 0.0 or not divisor_id:
            raise ValueError("Divisor tolerance/id are invalid.")
        self.atlas = atlas
        self.charts = charts_
        self.overlap_pairs = pairs
        self.overlap_units = units
        self.tolerance = float(tolerance)
        self.divisor_id = str(divisor_id)

    def _chart(self, index: int, /) -> DivisorChart:
        for chart in self.charts:
            if chart.chart_index == int(index):
                return chart
        raise ValueError("Divisor has no local equation on the requested chart.")

    def unit(self, source: int, target: int, coordinates: ArrayLike, /) -> Array:
        pair = (int(source), int(target))
        for declared, unit in zip(self.overlap_pairs, self.overlap_units, strict=True):
            if declared == pair:
                value = jnp.asarray(unit(jnp.asarray(coordinates)))
                return eqx.error_if(
                    value,
                    jnp.any(jnp.abs(value) <= self.tolerance)
                    | jnp.any(~jnp.isfinite(value)),
                    "Cartier overlap ratio is not a nowhere-zero finite unit.",
                )
        raise ValueError(
            "Cartier divisor has no declared unit for this directed overlap."
        )

    def overlap_residual(
        self, source: int, target: int, coordinates: ArrayLike, /
    ) -> Array:
        source_coordinates = jnp.asarray(coordinates)
        target_coordinates = self.atlas.transition(source, target)(source_coordinates)
        source_value = self._chart(source).defining_function(source_coordinates)
        target_value = self._chart(target).defining_function(target_coordinates)
        return jnp.max(
            jnp.abs(
                source_value
                - self.unit(source, target, source_coordinates) * target_value
            )
        )

    def cocycle_residual(
        self, first: int, second: int, third: int, coordinates: ArrayLike, /
    ) -> Array:
        first_coordinates = jnp.asarray(coordinates)
        second_coordinates = self.atlas.transition(first, second)(first_coordinates)
        direct = self.unit(first, third, first_coordinates)
        composed = self.unit(first, second, first_coordinates) * self.unit(
            second, third, second_coordinates
        )
        return jnp.max(jnp.abs(direct - composed))

    def clearance(
        self,
        chart_index: int,
        cells: ArrayLike,
        /,
        *,
        certified_lower_bounds: ArrayLike | None = None,
    ) -> DivisorClearanceEvidence:
        coordinates = jnp.asarray(cells)
        values = jnp.abs(
            jax.vmap(self._chart(chart_index).defining_function)(coordinates)
        )
        if certified_lower_bounds is None:
            lower = values
            certified = jnp.asarray(False)
            sampled = jnp.asarray(True)
        else:
            lower = jnp.asarray(certified_lower_bounds, dtype=values.dtype)
            if lower.shape != values.shape:
                raise ValueError(
                    "Certified divisor lower bounds must match declared cells."
                )
            certified = jnp.asarray(True)
            sampled = jnp.asarray(False)
        clear = jnp.all(lower > self.tolerance)
        return DivisorClearanceEvidence(
            lower, sampled=sampled, certified=certified & clear, clear=clear
        )

    def intersection(
        self, other: "CartierDivisor", chart_index: int, point: ArrayLike, /
    ) -> DivisorIntersection:
        if self.atlas is not other.atlas:
            raise ValueError("Divisor intersections require the same exact atlas.")
        point_ = jnp.asarray(point)
        functions = (
            self._chart(chart_index).defining_function,
            other._chart(chart_index).defining_function,
        )
        jacobian = jnp.stack(
            tuple(
                jax.jacfwd(function, holomorphic=True)(point_) for function in functions
            )
        )
        singular = jnp.linalg.svd(jacobian.reshape((2, -1)), compute_uv=False)
        rank = jnp.sum(singular > self.tolerance, dtype=jnp.int32)
        residual = jnp.max(
            jnp.abs(jnp.stack(tuple(function(point_) for function in functions)))
        )
        transverse = rank == 2
        return DivisorIntersection(
            point_, rank, 2, transverse, (residual <= self.tolerance) & transverse
        )


class MeromorphicSection(StrictModule):
    divisor: CartierDivisor
    numerators: tuple[Callable[[Array], Array], ...]
    denominators: tuple[Callable[[Array], Array], ...]

    def __init__(
        self,
        divisor: CartierDivisor,
        numerators: Sequence[Callable[[Array], Array]],
        denominators: Sequence[Callable[[Array], Array]],
        /,
    ):
        numerator_ = tuple(numerators)
        denominator_ = tuple(denominators)
        if (
            len(numerator_) != len(divisor.charts)
            or len(denominator_) != len(divisor.charts)
            or any(not callable(value) for value in numerator_ + denominator_)
        ):
            raise ValueError(
                "Meromorphic section requires numerator/denominator on every divisor chart."
            )
        self.divisor = divisor
        self.numerators = numerator_
        self.denominators = denominator_

    def evaluate(self, chart_index: int, coordinates: ArrayLike, /) -> Array:
        local = tuple(value.chart_index for value in self.divisor.charts).index(
            int(chart_index)
        )
        denominator = jnp.asarray(self.denominators[local](jnp.asarray(coordinates)))
        return eqx.error_if(
            self.numerators[local](jnp.asarray(coordinates)) / denominator,
            jnp.any(jnp.abs(denominator) <= self.divisor.tolerance),
            "Meromorphic section evaluation meets its declared divisor.",
        )


__all__ = [
    "CartierDivisor",
    "DivisorChart",
    "DivisorClearanceEvidence",
    "DivisorIntersection",
    "MeromorphicSection",
]
