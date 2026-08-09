#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._contracts import ML_INFEASIBLE, ML_INSUFFICIENT_DATA, ML_NONFINITE, ML_SUCCESS


class SpectralFitResult(StrictModule):
    """Fixed-rank weighted principal-subspace fit and diagnostics."""

    offset: Array
    components: Array
    singular_values: Array
    explained_energy: Array
    retained_energy: Array
    residual_energy: Array
    numerical_rank: Array
    orthogonality_error: Array
    minimum_retained_gap: Array
    valid: Array
    status: Array
    centered: bool = eqx.field(static=True)
    method: str = eqx.field(static=True)


def _canonicalize_rows(rows: Array, /) -> Array:
    pivot_indices = jnp.argmax(jnp.abs(rows), axis=-1)
    pivots = jnp.take_along_axis(rows, pivot_indices[..., None], axis=-1)[..., 0]
    magnitudes = jnp.abs(pivots)
    phases = jnp.where(magnitudes > 0.0, pivots / magnitudes, jnp.ones_like(pivots))
    return rows * jnp.conj(phases)[..., None]


def _fit_one(
    values: Array,
    weights: Array,
    rank: int,
    centered: bool,
    rcond: float,
) -> tuple[Array, ...]:
    valid_weight = jnp.isfinite(weights) & (weights >= 0.0)
    active = valid_weight & (weights > 0.0)
    safe_weights = jnp.where(valid_weight, weights, 0.0)
    safe_values = jnp.where(active[:, None], values, 0)
    total_weight = jnp.sum(safe_weights)
    offset = jnp.where(
        total_weight > 0.0,
        jnp.sum(safe_weights[:, None] * safe_values, axis=0)
        / jnp.maximum(total_weight, jnp.finfo(float).tiny),
        jnp.zeros(values.shape[-1], dtype=values.dtype),
    )
    if not centered:
        offset = jnp.zeros_like(offset)
    centered_values = jnp.where(active[:, None], safe_values - offset, 0)
    weighted = (
        jnp.sqrt(safe_weights / jnp.maximum(total_weight, jnp.finfo(float).tiny))[:, None]
        * centered_values
    )
    _u, singular_values, vh = jnp.linalg.svd(weighted, full_matrices=False)
    components = _canonicalize_rows(vh[:rank])
    energy = singular_values * singular_values
    total_energy = jnp.sum(energy)
    explained = jnp.where(total_energy > 0.0, energy / total_energy, 0.0)
    retained = jnp.sum(explained[:rank])
    residual = jnp.maximum(total_energy - jnp.sum(energy[:rank]), 0.0)
    largest = jnp.max(singular_values, initial=0.0)
    numerical_rank = jnp.sum(singular_values > largest * float(rcond), dtype=jnp.int32)
    gram = components @ jnp.conj(components).T
    orthogonality = jnp.max(jnp.abs(gram - jnp.eye(rank, dtype=gram.dtype)), initial=0.0)
    if rank < singular_values.shape[0]:
        gaps = singular_values[:rank] - singular_values[1 : rank + 1]
        minimum_gap = jnp.min(gaps, initial=jnp.inf)
    elif rank > 1:
        gaps = singular_values[: rank - 1] - singular_values[1:rank]
        minimum_gap = jnp.min(gaps, initial=jnp.inf)
    else:
        minimum_gap = jnp.asarray(jnp.inf, dtype=singular_values.dtype)
    finite_inputs = jnp.all(jnp.isfinite(weights))
    feasible = jnp.all(weights >= 0.0)
    finite_solution = (
        jnp.all(jnp.isfinite(jnp.real(components)))
        & jnp.all(jnp.isfinite(jnp.imag(components)))
        & jnp.all(jnp.isfinite(singular_values))
    )
    finite = finite_inputs & finite_solution
    enough = (total_weight > 0.0) & (numerical_rank >= rank)
    valid = finite & feasible & enough
    status = jnp.where(
        ~finite,
        ML_NONFINITE,
        jnp.where(
            ~feasible,
            ML_INFEASIBLE,
            jnp.where(enough, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        ),
    ).astype(jnp.int32)
    return (
        offset,
        components,
        singular_values[:rank],
        explained[:rank],
        retained,
        residual,
        numerical_rank,
        orthogonality,
        minimum_gap,
        valid,
        status,
    )


def fit_weighted_subspace(
    values: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    rank: int,
    centered: bool = True,
    rcond: float | None = None,
) -> SpectralFitResult:
    """Fit a fixed-rank weighted subspace over the penultimate sample axis."""
    x = jnp.asarray(values)
    w = jnp.asarray(weights, dtype=float)
    if x.ndim < 2 or w.shape != x.shape[:-1]:
        raise ValueError("values and weights must end in (sample, feature) and sample.")
    rank_ = int(rank)
    available = min(int(x.shape[-2]), int(x.shape[-1]))
    if rank_ <= 0 or rank_ > available:
        raise ValueError(f"rank must lie in [1, {available}].")
    case_shape = tuple(int(size) for size in x.shape[:-2])
    cases = 1
    for size in case_shape:
        cases *= size
    x_cases = x.reshape((cases, x.shape[-2], x.shape[-1]))
    w_cases = w.reshape((cases, w.shape[-1]))
    cutoff = (
        max(x.shape[-2], x.shape[-1]) * jnp.finfo(x.real.dtype).eps
        if rcond is None
        else float(rcond)
    )
    outputs = jax.vmap(
        lambda values_, weights_: _fit_one(
            values_, weights_, rank_, bool(centered), float(cutoff)
        )
    )(x_cases, w_cases)
    (
        offset,
        components,
        singular_values,
        explained,
        retained,
        residual,
        numerical_rank,
        orthogonality,
        minimum_gap,
        valid,
        status,
    ) = outputs
    return SpectralFitResult(
        offset=offset.reshape(case_shape + (x.shape[-1],)),
        components=components.reshape(case_shape + (rank_, x.shape[-1])),
        singular_values=singular_values.reshape(case_shape + (rank_,)),
        explained_energy=explained.reshape(case_shape + (rank_,)),
        retained_energy=retained.reshape(case_shape),
        residual_energy=residual.reshape(case_shape),
        numerical_rank=numerical_rank.reshape(case_shape),
        orthogonality_error=orthogonality.reshape(case_shape),
        minimum_retained_gap=minimum_gap.reshape(case_shape),
        valid=valid.reshape(case_shape),
        status=status.reshape(case_shape),
        centered=bool(centered),
        method="weighted-svd",
    )


__all__ = ["SpectralFitResult", "fit_weighted_subspace"]
