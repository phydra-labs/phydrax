#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from ._policies import RankPolicy


class NumericalRankData(NamedTuple):
    retained: Array
    rank: Array
    cutoff: Array
    condition_estimate: Array
    full_rank: Array
    finite: Array


def numerical_rank_data(
    singular_values: Array,
    rows: int,
    columns: int,
    policy: RankPolicy,
    /,
) -> NumericalRankData:
    """Resolve one canonical numerical-rank decision over trailing modes."""
    values = jnp.asarray(singular_values)
    if values.ndim < 1:
        raise ValueError("singular_values must have a trailing mode axis.")
    if rows < 1 or columns < 1:
        raise ValueError("Matrix dimensions must be positive.")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("singular_values must have a real floating dtype.")

    relative = (
        float(max(rows, columns)) * float(jnp.finfo(values.dtype).eps)
        if policy.relative_cutoff is None
        else policy.relative_cutoff
    )
    absolute = 0.0 if policy.absolute_cutoff is None else policy.absolute_cutoff
    largest = jnp.max(values, axis=-1)
    cutoff = (
        jnp.asarray(absolute, dtype=values.dtype)
        + jnp.asarray(
            relative,
            dtype=values.dtype,
        )
        * largest
    )
    finite = jnp.all(jnp.isfinite(values), axis=-1) & jnp.isfinite(cutoff)
    retained = (values > cutoff[..., None]) & jnp.isfinite(values)
    rank = jnp.sum(retained, axis=-1, dtype=jnp.int32)
    smallest_retained = jnp.min(
        jnp.where(retained, values, jnp.asarray(jnp.inf, dtype=values.dtype)),
        axis=-1,
    )
    condition = jnp.where(
        rank > 0,
        largest / smallest_retained,
        jnp.asarray(jnp.inf, dtype=values.dtype),
    )
    full_rank = rank == min(rows, columns)
    return NumericalRankData(
        retained=retained,
        rank=rank,
        cutoff=cutoff,
        condition_estimate=condition,
        full_rank=full_rank,
        finite=finite,
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
