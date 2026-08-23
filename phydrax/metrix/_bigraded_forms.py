#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from math import comb

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._complex import ComplexCoordinateConvention, wirtinger_derivatives
from ._utils import _pointwise_array


def _indices(dimension: int, degree: int, /) -> tuple[tuple[int, ...], ...]:
    return tuple(combinations(range(dimension), degree))


def _wedge_sign(left: tuple[int, ...], right: tuple[int, ...], /) -> int:
    if set(left).intersection(right):
        return 0
    inversions = sum(left_axis > right_axis for left_axis in left for right_axis in right)
    return -1 if inversions % 2 else 1


class BigradedForm(StrictModule):
    """A local complex differential form of declared bidegree ``(p, q)``."""

    coefficient_function: Callable[[Array], Array]
    convention: ComplexCoordinateConvention
    p: int
    q: int
    holomorphic_indices: tuple[tuple[int, ...], ...]
    antiholomorphic_indices: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        coefficients: Callable[[Array], Array],
        /,
        *,
        convention: ComplexCoordinateConvention,
        bidegree: tuple[int, int],
    ):
        if not callable(coefficients):
            raise TypeError("Bigraded coefficients must be callable.")
        if not isinstance(convention, ComplexCoordinateConvention):
            raise TypeError("convention must be a ComplexCoordinateConvention.")
        p, q = (int(bidegree[0]), int(bidegree[1]))
        dimension = convention.complex_dimension
        if p < 0 or q < 0 or p > dimension or q > dimension:
            raise ValueError("Bigraded form degrees must lie in [0, complex_dimension].")
        self.coefficient_function = coefficients
        self.convention = convention
        self.p = p
        self.q = q
        self.holomorphic_indices = _indices(dimension, p)
        self.antiholomorphic_indices = _indices(dimension, q)

    @property
    def bidegree(self) -> tuple[int, int]:
        return self.p, self.q

    @property
    def coefficient_count(self) -> int:
        dimension = self.convention.complex_dimension
        return comb(dimension, self.p) * comb(dimension, self.q)

    @property
    def index_pairs(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        return tuple(
            (holomorphic, antiholomorphic)
            for holomorphic in self.holomorphic_indices
            for antiholomorphic in self.antiholomorphic_indices
        )

    def _coefficients_point(self, coordinates: Array, /) -> Array:
        values = jnp.asarray(self.coefficient_function(coordinates))
        if self.coefficient_count == 1 and values.shape == ():
            values = values[None]
        expected = (self.coefficient_count,)
        if values.shape != expected:
            raise ValueError(
                f"Bigraded coefficients must have shape {expected}; got {values.shape}."
            )
        return values

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            self._coefficients_point,
            coordinates,
            self.convention.chart.dimension,
        )


class _DolbeaultCoefficients(StrictModule):
    form: BigradedForm
    antiholomorphic: bool
    output_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    input_lookup: dict[tuple[tuple[int, ...], tuple[int, ...]], int]

    def __init__(self, form: BigradedForm, *, antiholomorphic: bool):
        self.form = form
        self.antiholomorphic = bool(antiholomorphic)
        output_p = form.p if antiholomorphic else form.p + 1
        output_q = form.q + 1 if antiholomorphic else form.q
        self.output_pairs = tuple(
            (holomorphic, antiholomorphic_index)
            for holomorphic in _indices(form.convention.complex_dimension, output_p)
            for antiholomorphic_index in _indices(
                form.convention.complex_dimension, output_q
            )
        )
        self.input_lookup = {
            pair: position for position, pair in enumerate(form.index_pairs)
        }

    def __call__(self, coordinates: Array, /) -> Array:
        partial, partial_bar = wirtinger_derivatives(
            self.form._coefficients_point,
            self.form.convention,
            coordinates,
        )
        derivative = partial_bar if self.antiholomorphic else partial
        output = []
        for holomorphic, antiholomorphic in self.output_pairs:
            value = jnp.asarray(0.0, dtype=derivative.dtype)
            axes = antiholomorphic if self.antiholomorphic else holomorphic
            for position, axis in enumerate(axes):
                if self.antiholomorphic:
                    source_pair = (
                        holomorphic,
                        antiholomorphic[:position] + antiholomorphic[position + 1 :],
                    )
                    sign = -1 if (self.form.p + position) % 2 else 1
                else:
                    source_pair = (
                        holomorphic[:position] + holomorphic[position + 1 :],
                        antiholomorphic,
                    )
                    sign = -1 if position % 2 else 1
                value = value + sign * derivative[self.input_lookup[source_pair], axis]
            output.append(value)
        return jnp.stack(output)


class _BigradedWedgeCoefficients(StrictModule):
    left: BigradedForm
    right: BigradedForm
    output_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]

    def __init__(self, left: BigradedForm, right: BigradedForm, /):
        self.left = left
        self.right = right
        self.output_pairs = tuple(
            (holomorphic, antiholomorphic)
            for holomorphic in _indices(
                left.convention.complex_dimension, left.p + right.p
            )
            for antiholomorphic in _indices(
                left.convention.complex_dimension, left.q + right.q
            )
        )

    def __call__(self, coordinates: Array, /) -> Array:
        left_values = self.left._coefficients_point(coordinates)
        right_values = self.right._coefficients_point(coordinates)
        output_lookup = {pair: index for index, pair in enumerate(self.output_pairs)}
        result = jnp.zeros(
            (len(self.output_pairs),), dtype=jnp.result_type(left_values, right_values)
        )
        crossing = -1 if (self.left.q * self.right.p) % 2 else 1
        for left_position, (left_h, left_a) in enumerate(self.left.index_pairs):
            for right_position, (right_h, right_a) in enumerate(self.right.index_pairs):
                holomorphic_sign = _wedge_sign(left_h, right_h)
                antiholomorphic_sign = _wedge_sign(left_a, right_a)
                sign = crossing * holomorphic_sign * antiholomorphic_sign
                if sign == 0:
                    continue
                pair = (tuple(sorted(left_h + right_h)), tuple(sorted(left_a + right_a)))
                result = result.at[output_lookup[pair]].add(
                    sign * left_values[left_position] * right_values[right_position]
                )
        return result


def partial(form: BigradedForm, /) -> BigradedForm:
    if not isinstance(form, BigradedForm):
        raise TypeError("partial requires a BigradedForm.")
    if form.p >= form.convention.complex_dimension:
        raise ValueError("partial of maximal holomorphic degree is zero.")
    return BigradedForm(
        _DolbeaultCoefficients(form, antiholomorphic=False),
        convention=form.convention,
        bidegree=(form.p + 1, form.q),
    )


def partial_bar(form: BigradedForm, /) -> BigradedForm:
    if not isinstance(form, BigradedForm):
        raise TypeError("partial_bar requires a BigradedForm.")
    if form.q >= form.convention.complex_dimension:
        raise ValueError("partial_bar of maximal antiholomorphic degree is zero.")
    return BigradedForm(
        _DolbeaultCoefficients(form, antiholomorphic=True),
        convention=form.convention,
        bidegree=(form.p, form.q + 1),
    )


def bigraded_wedge(left: BigradedForm, right: BigradedForm, /) -> BigradedForm:
    if not isinstance(left, BigradedForm) or not isinstance(right, BigradedForm):
        raise TypeError("bigraded_wedge requires two BigradedForm objects.")
    if not left.convention.chart.compatible_with(right.convention.chart):
        raise ValueError("Bigraded form charts must match.")
    dimension = left.convention.complex_dimension
    if left.p + right.p > dimension or left.q + right.q > dimension:
        raise ValueError("Bigraded wedge exceeds complex dimension.")
    return BigradedForm(
        _BigradedWedgeCoefficients(left, right),
        convention=left.convention,
        bidegree=(left.p + right.p, left.q + right.q),
    )


__all__ = ["BigradedForm", "bigraded_wedge", "partial", "partial_bar"]
