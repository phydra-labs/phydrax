#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Interval-enclosed band Cholesky and extremal spectral certificates."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import final

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..typing import as_array, Bool, Dim, Float, Inexact, Int32, Scalar


class SpectrumBandDim(Dim):
    """Stored scalar diagonals of a banded matrix."""


class SpectrumColumnDim(Dim):
    """Columns of a banded matrix."""


class BandedSpectrumStatus(IntEnum):
    """Certified outcome; an inconclusive test never claims indefiniteness."""

    CERTIFIED = 0
    BRACKET_NOT_CONVERGED = 1
    INDEFINITE_AT_SHIFT = 2


@final
class BandedDefinitenessCertificate(StrictModule):
    """Positive definiteness, or a certified negative Schur pivot at a shift."""

    __strict_contract__ = True
    certified: Bool[Scalar]
    status: Int32[Scalar]
    minimal_pivot: Float[Scalar]
    rounding_margin: Float[Scalar]


@final
class BandedEigenvalueBrackets(StrictModule):
    """Outward enclosures remain valid even when the step limit is reached."""

    __strict_contract__ = True
    minimum_lower: Float[Scalar]
    minimum_upper: Float[Scalar]
    maximum_lower: Float[Scalar]
    maximum_upper: Float[Scalar]
    status: Int32[Scalar]
    steps: Int32[Scalar]
    minimum_width: Float[Scalar]
    maximum_width: Float[Scalar]
    minimal_pivot: Float[Scalar]
    minimum_rounding_floor: Float[Scalar]
    maximum_rounding_floor: Float[Scalar]


@final
class BandedSingularValueBrackets(StrictModule):
    """Enclosures for column-map singular values and a certified condition cap."""

    __strict_contract__ = True
    minimum_lower: Float[Scalar]
    minimum_upper: Float[Scalar]
    maximum_lower: Float[Scalar]
    maximum_upper: Float[Scalar]
    condition_upper_bound: Float[Scalar]
    status: Int32[Scalar]
    steps: Int32[Scalar]
    minimum_width: Float[Scalar]
    maximum_width: Float[Scalar]
    minimal_pivot: Float[Scalar]
    minimum_rounding_floor: Float[Scalar]
    maximum_rounding_floor: Float[Scalar]


def _bands(value: ArrayLike, lower: int, upper: int) -> Array:
    bands = as_array(value, Inexact[SpectrumBandDim, SpectrumColumnDim], "bands")
    if lower < 0 or upper < 0 or bands.shape[0] != lower + upper + 1:
        raise ValueError(
            "bands must have shape (lower_bandwidth + upper_bandwidth + 1, columns)."
        )
    if bands.shape[1] < 1 or bands.real.dtype.itemsize < 4:
        raise ValueError(
            "bands require nonempty columns and float32 or wider components."
        )
    return bands


def _roundoff(magnitude: Array, operations: int = 1) -> Array:
    # Covers real/complex arithmetic, reduction reassociation, and subnormal error.
    info = jnp.finfo(magnitude.dtype)
    return (16 * operations * info.eps) * magnitude + operations * info.tiny


def _hermitian_valid(bands: Array, bandwidth: int) -> Array:
    n = bands.shape[1]
    valid = jnp.all(jnp.isfinite(bands)) & jnp.all(jnp.imag(bands[bandwidth]) == 0)
    for offset in range(1, min(bandwidth + 1, n)):
        valid = valid & jnp.all(
            bands[bandwidth + offset, : n - offset]
            == jnp.conj(bands[bandwidth - offset, offset:])
        )
    return valid


def _definiteness(
    bands: Array, radii: Array, bandwidth: int, shift: Array, sign: int
) -> BandedDefinitenessCertificate:
    """Disk arithmetic encloses every Cholesky entry and real Schur pivot.

    Positive lower pivot endpoints prove SPD. A negative upper endpoint, after
    exclusively positive previous pivots, proves a negative quadratic direction.
    An interval crossing zero is inconclusive, not an indefiniteness claim.
    """
    n = bands.shape[1]
    width = bandwidth + 1
    factor = jnp.zeros((n, width), dtype=bands.dtype)
    radius = jnp.zeros((n, width), dtype=bands.real.dtype)
    initial = (
        factor,
        radius,
        jnp.asarray(True),
        jnp.asarray(False),
        jnp.asarray(jnp.inf, dtype=bands.real.dtype),
        jnp.asarray(0, dtype=bands.real.dtype),
    )

    def row_step(carry: tuple[Array, ...], row: Array) -> tuple[tuple[Array, ...], None]:
        def entry_step(index: int, state: tuple[Array, ...]) -> tuple[Array, ...]:
            f, r, active, negative, minimum, margin = state
            column = row - bandwidth + index

            def evaluate(current: tuple[Array, ...]) -> tuple[Array, ...]:
                f, r, active, negative, minimum, margin = current
                k = row - bandwidth + jnp.arange(width)
                left_distance = row - k
                right_distance = column - k
                mask = (k >= 0) & (k < column) & (right_distance <= bandwidth)
                left = jnp.where(mask, f[row, jnp.clip(left_distance, 0, bandwidth)], 0)
                right = jnp.where(
                    mask,
                    f[jnp.clip(column, 0, n - 1), jnp.clip(right_distance, 0, bandwidth)],
                    0,
                )
                lr = jnp.where(mask, r[row, jnp.clip(left_distance, 0, bandwidth)], 0)
                rr = jnp.where(
                    mask,
                    r[jnp.clip(column, 0, n - 1), jnp.clip(right_distance, 0, bandwidth)],
                    0,
                )
                products = left * jnp.conj(right)
                dot = jnp.sum(products)
                error = jnp.sum(jnp.abs(left) * rr + jnp.abs(right) * lr + lr * rr)
                error = error + _roundoff(jnp.sum(jnp.abs(left) * jnp.abs(right)), width)
                original = sign * bands[bandwidth + row - column, column]
                diagonal_shift = jnp.where(row == column, sign * shift, 0)
                center = original - diagonal_shift - dot
                error = error + radii[bandwidth + row - column, column]
                error = error + _roundoff(
                    jnp.abs(original) + jnp.abs(diagonal_shift) + jnp.abs(dot), 3
                )
                # The radius calculations and interval endpoints themselves
                # round too, including when uncertainty dominates the center.
                error = error + _roundoff(error + jnp.abs(center), 4 * width)

                def diagonal(_: None) -> tuple[Array, Array, Array, Array, Array]:
                    pivot = jnp.real(center)
                    lo, hi = pivot - error, pivot + error
                    positive = (lo > 0) & jnp.isfinite(hi)
                    safe = jnp.maximum(pivot, jnp.finfo(pivot.dtype).tiny)
                    value = jnp.sqrt(safe)
                    rad = jnp.maximum(
                        value - jnp.sqrt(jnp.maximum(lo, 0)),
                        jnp.sqrt(jnp.maximum(hi, 0)) - value,
                    )
                    rad = rad + _roundoff(value, 2)
                    return (
                        jnp.where(positive, value, 1).astype(bands.dtype),
                        jnp.where(positive, rad, 0),
                        positive,
                        hi < 0,
                        lo,
                    )

                def off_diagonal(_: None) -> tuple[Array, Array, Array, Array, Array]:
                    denominator = jnp.real(f[column, 0])
                    denominator_radius = r[column, 0]
                    lower = denominator - denominator_radius
                    lower = lower - _roundoff(jnp.abs(denominator) + denominator_radius)
                    value = center / denominator
                    rad = (error + jnp.abs(value) * denominator_radius) / lower
                    rad = rad + _roundoff(jnp.abs(value) + rad, 4)
                    return (
                        value,
                        rad,
                        jnp.isfinite(rad) & (lower > 0),
                        jnp.asarray(False),
                        minimum,
                    )

                value, rad, valid, neg, pivot = jax.lax.cond(
                    row == column, diagonal, off_diagonal, None
                )
                distance = row - column
                f = f.at[row, distance].set(value)
                r = r.at[row, distance].set(rad)
                return (
                    f,
                    r,
                    active & valid,
                    negative | neg,
                    jnp.minimum(minimum, pivot),
                    jnp.maximum(margin, error),
                )

            return jax.lax.cond(active & (column >= 0), evaluate, lambda x: x, state)

        updated = jax.lax.fori_loop(0, width, entry_step, carry)
        return updated, None

    final_state, _ = jax.lax.scan(row_step, initial, jnp.arange(n, dtype=jnp.int32))
    _, _, positive, negative, minimum, margin = final_state
    valid = (
        _hermitian_valid(bands, bandwidth)
        & jnp.all(jnp.isfinite(radii))
        & jnp.isfinite(shift)
    )
    positive = positive & valid
    negative = negative & valid
    status = jnp.where(
        positive,
        int(BandedSpectrumStatus.CERTIFIED),
        jnp.where(
            negative,
            int(BandedSpectrumStatus.INDEFINITE_AT_SHIFT),
            int(BandedSpectrumStatus.BRACKET_NOT_CONVERGED),
        ),
    ).astype(jnp.int32)
    return BandedDefinitenessCertificate(positive, status, minimum, margin)


def banded_definiteness_certificate(
    bands: ArrayLike, /, *, bandwidth: int, shift: ArrayLike = 0.0, upper: bool = False
) -> BandedDefinitenessCertificate:
    """Certify ``H-shift*I > 0`` (or ``shift*I-H > 0`` if ``upper``).

    Storage is ``bands[bandwidth + row-column, column]``. Both triangles
    must be supplied and exactly Hermitian; unused exterior slots are ignored.
    """
    value = _bands(bands, bandwidth, bandwidth)
    tau = jnp.asarray(shift, dtype=value.real.dtype)
    if tau.ndim != 0:
        raise ValueError("shift must be scalar.")
    return _definiteness(
        value, jnp.zeros_like(value.real), bandwidth, tau, -1 if upper else 1
    )


def _gershgorin(bands: Array, radii: Array, bandwidth: int) -> tuple[Array, Array]:
    n = bands.shape[1]
    diagonal = jnp.real(bands[bandwidth])
    radius = radii[bandwidth]
    for offset in range(1, min(bandwidth + 1, n)):
        magnitude = (
            jnp.abs(bands[bandwidth + offset, : n - offset])
            + radii[bandwidth + offset, : n - offset]
        )
        radius = radius.at[offset:].add(magnitude)
        radius = radius.at[: n - offset].add(magnitude)
    padding = _roundoff(jnp.abs(diagonal) + radius, 2 * bandwidth + 3)
    return jnp.min(diagonal - radius - padding), jnp.max(diagonal + radius + padding)


def _eigen_brackets(
    bands: Array, radii: Array, bandwidth: int, relative_tolerance: float, max_steps: int
) -> BandedEigenvalueBrackets:
    if not isfinite(relative_tolerance) or relative_tolerance <= 0 or max_steps < 0:
        raise ValueError("relative_tolerance must be positive and max_steps nonnegative.")
    lower, upper = _gershgorin(bands, radii, bandwidth)
    initial_floor = _roundoff(
        jnp.maximum(jnp.abs(lower), jnp.abs(upper)), 2 * bandwidth + 3
    )
    initial = (
        lower,
        upper,
        lower,
        upper,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=bands.real.dtype),
        jnp.asarray(False),
        jnp.asarray(False),
        initial_floor,
        initial_floor,
    )

    def target(lo: Array, hi: Array, floor: Array) -> Array:
        return jnp.maximum(
            floor,
            relative_tolerance
            * jnp.maximum(
                jnp.minimum(jnp.abs(lo), jnp.abs(hi)), jnp.finfo(lo.dtype).tiny
            ),
        )

    def unfinished(state: tuple[Array, ...]) -> Array:
        lo, hi, ml, mh, step, _, _, _, min_floor, max_floor = state
        return (step < max_steps) & (
            ((hi - lo) > target(lo, hi, min_floor))
            | ((mh - ml) > target(ml, mh, max_floor))
        )

    def advance(state: tuple[Array, ...]) -> tuple[Array, ...]:
        (
            lo,
            hi,
            ml,
            mh,
            step,
            pivot,
            min_uncertain,
            max_uncertain,
            min_floor,
            max_floor,
        ) = state
        # Bisect normally. Only a zero-crossing interval uses an alternate
        # interior probe, so an exact midpoint eigenvalue cannot stall.
        alternate = jnp.where(step % 2 == 0, 0.25, 0.75)
        mid = lo + (hi - lo) * jnp.where(min_uncertain, alternate, 0.5)
        maximum_mid = ml + (mh - ml) * jnp.where(max_uncertain, alternate, 0.5)
        minimum_test = _definiteness(bands, radii, bandwidth, mid, 1)
        maximum_test = _definiteness(bands, radii, bandwidth, maximum_mid, -1)
        lo = jnp.where(minimum_test.certified, mid, lo)
        hi = jnp.where(
            minimum_test.status == int(BandedSpectrumStatus.INDEFINITE_AT_SHIFT), mid, hi
        )
        mh = jnp.where(maximum_test.certified, maximum_mid, mh)
        ml = jnp.where(
            maximum_test.status == int(BandedSpectrumStatus.INDEFINITE_AT_SHIFT),
            maximum_mid,
            ml,
        )
        pivot = jnp.minimum(
            pivot, jnp.minimum(minimum_test.minimal_pivot, maximum_test.minimal_pivot)
        )
        min_uncertain = minimum_test.status == int(
            BandedSpectrumStatus.BRACKET_NOT_CONVERGED
        )
        max_uncertain = maximum_test.status == int(
            BandedSpectrumStatus.BRACKET_NOT_CONVERGED
        )
        # Only an inconclusive pivot establishes an attained arithmetic floor.
        # This stops refinement, never widens or weakens the enclosing bounds.
        min_floor = jnp.maximum(
            min_floor,
            jnp.where(
                min_uncertain & jnp.isfinite(minimum_test.rounding_margin),
                2 * minimum_test.rounding_margin,
                0,
            ),
        )
        max_floor = jnp.maximum(
            max_floor,
            jnp.where(
                max_uncertain & jnp.isfinite(maximum_test.rounding_margin),
                2 * maximum_test.rounding_margin,
                0,
            ),
        )
        return (
            lo,
            hi,
            ml,
            mh,
            step + 1,
            pivot,
            min_uncertain,
            max_uncertain,
            min_floor,
            max_floor,
        )

    lo, hi, ml, mh, steps, pivot, _, _, min_floor, max_floor = jax.lax.while_loop(
        unfinished, advance, initial
    )
    valid = _hermitian_valid(bands, bandwidth)
    lo, ml = jnp.where(valid, lo, -jnp.inf), jnp.where(valid, ml, -jnp.inf)
    hi, mh = jnp.where(valid, hi, jnp.inf), jnp.where(valid, mh, jnp.inf)
    converged = (
        valid
        & ((hi - lo) <= target(lo, hi, min_floor))
        & ((mh - ml) <= target(ml, mh, max_floor))
    )
    status = jnp.where(
        converged,
        int(BandedSpectrumStatus.CERTIFIED),
        int(BandedSpectrumStatus.BRACKET_NOT_CONVERGED),
    ).astype(jnp.int32)
    return BandedEigenvalueBrackets(
        lo, hi, ml, mh, status, steps, hi - lo, mh - ml, pivot, min_floor, max_floor
    )


def banded_eigenvalue_brackets(
    bands: ArrayLike,
    /,
    *,
    bandwidth: int,
    relative_tolerance: float = 1e-8,
    max_steps: int = 80,
) -> BandedEigenvalueBrackets:
    """Bound both Hermitian spectral endpoints without any dense matrix."""
    value = _bands(bands, bandwidth, bandwidth)
    return _eigen_brackets(
        value, jnp.zeros_like(value.real), bandwidth, relative_tolerance, max_steps
    )


def _gram(bands: Array, rows: int, lower: int, upper: int) -> tuple[Array, Array, int]:
    if rows < 1:
        raise ValueError("rows must be positive.")
    n = bands.shape[1]
    bandwidth = min(lower + upper, n - 1)
    gram = jnp.zeros((2 * bandwidth + 1, n), dtype=bands.dtype)
    radius = jnp.zeros_like(gram.real)
    columns = jnp.arange(n, dtype=jnp.int32)
    offsets = jnp.arange(-upper, lower + 1, dtype=jnp.int32)

    def diagonal(offset: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        gram, radius = state
        # Every shared row contributes, including off-diagonal Gram bands.
        row = columns[:, None] + offsets[None, :]
        other = columns + offset
        other_offset = offsets[None, :] - offset
        mask = (
            (row >= 0)
            & (row < rows)
            & (other[:, None] < n)
            & (other_offset >= -upper)
            & (other_offset <= lower)
        )
        first = bands[upper + offsets[None, :], columns[:, None]]
        second = bands[
            jnp.clip(upper + other_offset, 0, lower + upper),
            jnp.clip(other[:, None], 0, n - 1),
        ]
        products = jnp.where(mask, jnp.conj(first) * second, 0)
        value = jnp.sum(products, axis=1)
        # The mathematical Gram diagonal is real even if complex multiplication
        # leaves a rounded imaginary component; its disk already encloses it.
        value = jnp.where(offset == 0, jnp.real(value).astype(bands.dtype), value)
        magnitude = jnp.sum(jnp.where(mask, jnp.abs(first) * jnp.abs(second), 0), axis=1)
        error = _roundoff(magnitude, 2 * (lower + upper + 1))
        # lower Gram entry is the conjugate of <column, other>.
        gram = gram.at[bandwidth + offset, columns].set(jnp.conj(value))
        radius = radius.at[bandwidth + offset, columns].set(error)
        gram = gram.at[bandwidth - offset, other].set(value, mode="drop")
        radius = radius.at[bandwidth - offset, other].set(error, mode="drop")
        return gram, radius

    gram, radius = jax.lax.fori_loop(0, bandwidth + 1, diagonal, (gram, radius))
    return gram, radius, bandwidth


def banded_singular_value_brackets(
    bands: ArrayLike,
    /,
    *,
    rows: int,
    lower_bandwidth: int,
    upper_bandwidth: int,
    relative_tolerance: float = 1e-8,
    max_steps: int = 80,
) -> BandedSingularValueBrackets:
    """Certify column-map singular values from interval-enclosed ``AᴴA`` bands.

    Rectangular/block-banded matrices use scalar bands and an explicit row count.
    Wide matrices have zero minimum column-map singular value by definition.
    """
    value = _bands(bands, lower_bandwidth, upper_bandwidth)
    gram, radii, width = _gram(value, rows, lower_bandwidth, upper_bandwidth)
    result = _eigen_brackets(gram, radii, width, relative_tolerance, max_steps)
    lo = jnp.sqrt(jnp.maximum(result.minimum_lower, 0))
    hi = jnp.sqrt(jnp.maximum(result.minimum_upper, 0))
    ml = jnp.sqrt(jnp.maximum(result.maximum_lower, 0))
    mh = jnp.sqrt(jnp.maximum(result.maximum_upper, 0))
    lo = jnp.maximum(0, lo - _roundoff(lo))
    ml = jnp.maximum(0, ml - _roundoff(ml))
    hi, mh = hi + _roundoff(hi), mh + _roundoff(mh)
    if rows < value.shape[1]:
        lo, hi = jnp.zeros_like(lo), jnp.zeros_like(hi)
    condition = jnp.where(lo > 0, mh / lo + _roundoff(mh / lo), jnp.inf)
    min_base = jnp.maximum(result.minimum_lower, 0)
    max_base = jnp.maximum(result.maximum_lower, 0)
    min_floor = result.minimum_rounding_floor / (
        jnp.sqrt(min_base + result.minimum_rounding_floor) + jnp.sqrt(min_base)
    )
    max_floor = result.maximum_rounding_floor / (
        jnp.sqrt(max_base + result.maximum_rounding_floor) + jnp.sqrt(max_base)
    )
    min_floor = min_floor + _roundoff(jnp.abs(min_floor) + hi)
    max_floor = max_floor + _roundoff(jnp.abs(max_floor) + mh)
    return BandedSingularValueBrackets(
        lo,
        hi,
        ml,
        mh,
        condition,
        result.status,
        result.steps,
        hi - lo,
        mh - ml,
        result.minimal_pivot,
        min_floor,
        max_floor,
    )


def banded_singular_value_certificate(
    bands: ArrayLike,
    /,
    *,
    rows: int,
    lower_bandwidth: int,
    upper_bandwidth: int,
    threshold: ArrayLike,
) -> BandedDefinitenessCertificate:
    """One definiteness test certifies ``sigma_min >= threshold``; never densifies.

    A false certificate means unproven, not necessarily rank deficient. Threshold
    must be nonnegative; invalid dynamic values produce inconclusive evidence.
    """
    value = _bands(bands, lower_bandwidth, upper_bandwidth)
    tau = jnp.asarray(threshold, dtype=value.real.dtype)
    if tau.ndim != 0:
        raise ValueError("threshold must be scalar.")
    gram, radii, width = _gram(value, rows, lower_bandwidth, upper_bandwidth)
    square = tau * tau
    radii = radii.at[width].add(_roundoff(jnp.abs(square)))
    result = _definiteness(gram, radii, width, square, 1)
    valid = (tau >= 0) & (rows >= value.shape[1])
    return BandedDefinitenessCertificate(
        result.certified & valid,
        jnp.where(
            valid, result.status, int(BandedSpectrumStatus.BRACKET_NOT_CONVERGED)
        ).astype(jnp.int32),
        result.minimal_pivot,
        result.rounding_margin,
    )
