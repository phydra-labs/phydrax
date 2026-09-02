#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array

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
