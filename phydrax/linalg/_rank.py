#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, NamedTuple, overload

import jax.numpy as jnp
import numpy as np
from jax import Array

from ._policies import RankPolicy


class NumericalRankData(NamedTuple):
    retained: Array
    rank: Array
    cutoff: Array
    condition_estimate: Array
    full_rank: Array
    finite: Array


class HostNumericalRankData(NamedTuple):
    """Canonical rank data retained entirely as concrete NumPy arrays."""

    retained: np.ndarray
    rank: np.ndarray
    cutoff: np.ndarray
    condition_estimate: np.ndarray
    full_rank: np.ndarray
    finite: np.ndarray


def numerical_rank_data(
    singular_values: Array,
    rows: int,
    columns: int,
    policy: RankPolicy,
    /,
) -> NumericalRankData:
    """Resolve one canonical numerical-rank decision over trailing modes."""
    return _numerical_rank_data(singular_values, rows, columns, policy, host=False)


def host_numerical_rank_data(
    singular_values: np.ndarray,
    rows: int,
    columns: int,
    policy: RankPolicy,
    /,
) -> HostNumericalRankData:
    """The same rank policy for immutable host preparation, without JAX staging."""
    return _numerical_rank_data(singular_values, rows, columns, policy, host=True)


@overload
def _numerical_rank_data(
    singular_values: Array,
    rows: int,
    columns: int,
    policy: RankPolicy,
    /,
    *,
    host: Literal[False],
) -> NumericalRankData: ...


@overload
def _numerical_rank_data(
    singular_values: np.ndarray,
    rows: int,
    columns: int,
    policy: RankPolicy,
    /,
    *,
    host: Literal[True],
) -> HostNumericalRankData: ...


def _numerical_rank_data(
    singular_values: Array | np.ndarray,
    rows: int,
    columns: int,
    policy: RankPolicy,
    /,
    *,
    host: bool,
) -> NumericalRankData | HostNumericalRankData:
    # Only this numerical kernel is shared; runtime callers retain JAX arrays.
    xp = np if host else jnp
    values = xp.asarray(singular_values)
    if values.ndim < 1:
        raise ValueError("singular_values must have a trailing mode axis.")
    if rows < 1 or columns < 1:
        raise ValueError("Matrix dimensions must be positive.")
    if not xp.issubdtype(values.dtype, xp.floating):
        raise TypeError("singular_values must have a real floating dtype.")

    relative = (
        float(max(rows, columns)) * float(xp.finfo(values.dtype).eps)
        if policy.relative_cutoff is None
        else policy.relative_cutoff
    )
    absolute = 0.0 if policy.absolute_cutoff is None else policy.absolute_cutoff
    largest = xp.max(values, axis=-1)
    cutoff = (
        xp.asarray(absolute, dtype=values.dtype)
        + xp.asarray(
            relative,
            dtype=values.dtype,
        )
        * largest
    )
    finite = xp.all(xp.isfinite(values), axis=-1) & xp.isfinite(cutoff)
    retained = (values > cutoff[..., None]) & xp.isfinite(values)
    rank = xp.sum(retained, axis=-1, dtype=xp.int32)
    smallest_retained = xp.min(
        xp.where(retained, values, xp.asarray(xp.inf, dtype=values.dtype)),
        axis=-1,
    )
    condition = xp.where(
        rank > 0,
        largest / smallest_retained,
        xp.asarray(xp.inf, dtype=values.dtype),
    )
    full_rank = rank == min(rows, columns)
    if host:
        return HostNumericalRankData(
            retained=np.asarray(retained),
            rank=np.asarray(rank),
            cutoff=np.asarray(cutoff),
            condition_estimate=np.asarray(condition),
            full_rank=np.asarray(full_rank),
            finite=np.asarray(finite),
        )
    return NumericalRankData(
        retained=jnp.asarray(retained),
        rank=jnp.asarray(rank),
        cutoff=jnp.asarray(cutoff),
        condition_estimate=jnp.asarray(condition),
        full_rank=jnp.asarray(full_rank),
        finite=jnp.asarray(finite),
    )


def singular_value_backward_error(largest: Array, rows: int, columns: int, /) -> Array:
    """Backward-error bound ``max(rows, columns) eps sigma_max`` of a stable SVD."""
    value = jnp.asarray(largest)
    return max(rows, columns) * jnp.finfo(value.dtype).eps * value


def rank_gap_margin(
    rank: Array,
    maximum_rank: int,
    smallest_retained: Array,
    largest_discarded: Array,
    threshold_lower: Array,
    threshold_upper: Array,
    /,
) -> Array:
    """Distance of a rank decision from its cutoff interval on either side.

    By Weyl's inequality the decision is invariant under every perturbation of
    spectral norm below this margin (less the spectrum's own error bound).
    """
    infinity = jnp.asarray(jnp.inf, dtype=jnp.result_type(smallest_retained))
    lower = jnp.where(rank > 0, smallest_retained - threshold_upper, infinity)
    upper = jnp.where(rank < maximum_rank, threshold_lower - largest_discarded, infinity)
    return jnp.minimum(lower, upper)


def exact_spectrum_fixed_rank(
    singular_values: Array,
    rows: int,
    columns: int,
    policy: RankPolicy,
    /,
) -> Array:
    """Whether a complete computed spectrum certifies a locally constant rank.

    Every singular value must lie on its side of the cutoff by more than the
    backward-error bound of the factorization that computed it.
    """
    values = jnp.asarray(singular_values)
    if values.shape[-1] != min(rows, columns):
        raise ValueError("A fixed-rank decision requires the complete spectrum.")
    data = numerical_rank_data(values, rows, columns, policy)
    infinity = jnp.asarray(jnp.inf, dtype=values.dtype)
    smallest_retained = jnp.min(jnp.where(data.retained, values, infinity), axis=-1)
    largest_discarded = jnp.max(
        jnp.where(~data.retained & jnp.isfinite(values), values, 0.0), axis=-1
    )
    margin = rank_gap_margin(
        data.rank,
        min(rows, columns),
        smallest_retained,
        largest_discarded,
        data.cutoff,
        data.cutoff,
    )
    allowance = singular_value_backward_error(jnp.max(values, axis=-1), rows, columns)
    return data.finite & ~jnp.isnan(margin) & (margin > allowance)
