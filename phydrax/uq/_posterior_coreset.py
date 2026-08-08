#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from ._diagnostics import MCMCDiagnostics
from ._mcmc import MCMCResult
from ._posterior import PosteriorProblem
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField


class SteinThinning(StrictModule):
    """Chain-preserving greedy thinning under an inverse-multiquadric Stein kernel."""

    num_points: int = eqx.field(static=True)
    beta: Array
    offset: Array
    length_scale: Array | None

    def __init__(
        self,
        num_points: int,
        /,
        *,
        beta: float = -0.5,
        offset: float = 1.0,
        length_scale: ArrayLike | None = None,
    ):
        count = int(num_points)
        if count <= 0:
            raise ValueError("num_points must be positive.")
        if not -1.0 < float(beta) < 0.0:
            raise ValueError("beta must lie strictly between -1 and 0.")
        if not float(offset) > 0.0:
            raise ValueError("offset must be strictly positive.")
        scale = None if length_scale is None else jnp.asarray(length_scale, dtype=float)
        if scale is not None and bool(jnp.any((~jnp.isfinite(scale)) | (scale <= 0.0))):
            raise ValueError("length_scale must be finite and strictly positive.")
        self.num_points = count
        self.beta = jnp.asarray(beta, dtype=float)
        self.offset = jnp.asarray(offset, dtype=float)
        self.length_scale = scale


class PosteriorCoreset(StrictModule):
    """Chain-preserving constrained posterior draws selected from an MCMC result."""

    problem: PosteriorProblem
    samples: Any
    unconstrained_samples: Any
    log_density: Array
    chain_indices: Array
    draw_indices: Array
    kernel_stein_discrepancy: Array
    source_diagnostics: MCMCDiagnostics
    source_algorithm: str = eqx.field(static=True)
    source_chain_method: str = eqx.field(static=True)
    source_num_draws: int = eqx.field(static=True)

    def predict(
        self,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        chain_dim: str = "__phydra_uq_chain",
        draw_dim: str = "__phydra_uq_draw",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Evaluate selected draws without merging chain and draw dimensions."""
        return predict_from_position_samples(
            self.problem,
            self.unconstrained_samples,
            *args,
            sample_dims=(chain_dim, draw_dim),
            sample_sources=("epistemic", "epistemic"),
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )

    def predict_observations(
        self,
        key: Array,
        /,
        *args: Any,
        num_observation_samples: int,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        chain_dim: str = "__phydra_uq_chain",
        draw_dim: str = "__phydra_uq_draw",
        observation_dim: str = "__phydra_uq_observation",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Draw observations without merging retained posterior chains."""
        return sample_observations_from_position_samples(
            self.problem,
            key,
            self.unconstrained_samples,
            *args,
            num_observation_samples=num_observation_samples,
            sample_dims=(chain_dim, draw_dim),
            sample_sources=("epistemic", "epistemic"),
            observation_dim=observation_dim,
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )


def thin_posterior(
    result: MCMCResult,
    method: SteinThinning,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> PosteriorCoreset:
    """Compress each MCMC chain without changing the source result's semantics."""
    if not isinstance(result, MCMCResult):
        raise TypeError("result must be an MCMCResult.")
    if not isinstance(method, SteinThinning):
        raise TypeError("method must be a SteinThinning.")
    log_density = jnp.asarray(result.log_density, dtype=float)
    if log_density.ndim != 2:
        raise ValueError("MCMC log density must have shape (chains, draws).")
    if bool(jnp.any(~jnp.isfinite(log_density))):
        raise ValueError("MCMC log density must be finite before posterior thinning.")
    num_chains, num_draws = map(int, log_density.shape)
    if method.num_points > num_draws:
        raise ValueError("Cannot retain more posterior draws than each chain contains.")
    flat_samples, unravel = _flatten_position_samples(
        result.unconstrained_samples,
        num_chains,
        num_draws,
    )
    scores = jax.vmap(jax.vmap(jax.grad(lambda value: result.problem.log_density(unravel(value)))))(
        flat_samples
    )
    if bool(jnp.any(~jnp.isfinite(scores))):
        raise ValueError("Posterior score evaluation produced non-finite values.")
    scale = _resolve_length_scale(flat_samples, method.length_scale)
    precision = 1.0 / jnp.square(scale)
    keys = jr.split(key, num_chains)
    draw_indices, discrepancy = jax.vmap(
        lambda points, point_scores, chain_key: _thin_chain(
            points,
            point_scores,
            method,
            precision,
            chain_key,
        )
    )(flat_samples, scores, keys)
    chain_indices = jnp.broadcast_to(
        jnp.arange(num_chains, dtype=jnp.int32)[:, None],
        draw_indices.shape,
    )
    samples = jax.tree_util.tree_map(
        lambda leaf: _gather_draws(leaf, draw_indices),
        result.samples,
    )
    unconstrained = jax.tree_util.tree_map(
        lambda leaf: _gather_draws(leaf, draw_indices),
        result.unconstrained_samples,
    )
    selected_log_density = jnp.take_along_axis(log_density, draw_indices, axis=1)
    return PosteriorCoreset(
        problem=result.problem,
        samples=samples,
        unconstrained_samples=unconstrained,
        log_density=selected_log_density,
        chain_indices=chain_indices,
        draw_indices=draw_indices,
        kernel_stein_discrepancy=discrepancy,
        source_diagnostics=result.diagnostics,
        source_algorithm=result.algorithm,
        source_chain_method=result.chain_method,
        source_num_draws=num_draws,
    )


def _flatten_position_samples(
    samples: Any,
    num_chains: int,
    num_draws: int,
    /,
) -> tuple[Array, Any]:
    example = jax.tree_util.tree_map(lambda leaf: leaf[0, 0], samples)
    _, unravel = ravel_pytree(example)
    matrices = []
    for leaf in jax.tree_util.tree_leaves(samples):
        values = jnp.asarray(leaf, dtype=float)
        if values.shape[:2] != (num_chains, num_draws):
            raise ValueError("MCMC sample leaves must share leading chain and draw axes.")
        matrices.append(values.reshape((num_chains, num_draws, -1)))
    if not matrices:
        raise ValueError("MCMC samples must contain at least one parameter leaf.")
    return jnp.concatenate(tuple(matrices), axis=-1), unravel


def _resolve_length_scale(samples: Array, supplied: Array | None, /) -> Array:
    dimension = int(samples.shape[-1])
    if supplied is not None:
        scale = jnp.broadcast_to(jnp.asarray(supplied, dtype=float), (dimension,))
        if bool(jnp.any((~jnp.isfinite(scale)) | (scale <= 0.0))):
            raise ValueError("length_scale must broadcast to finite positive coordinates.")
        return scale
    scale = jnp.std(samples.reshape((-1, dimension)), axis=0)
    floor = jnp.sqrt(jnp.finfo(samples.dtype).eps)
    return jnp.where(scale > floor, scale, 1.0)


def _stein_kernel(
    left: Array,
    left_score: Array,
    right: Array,
    right_score: Array,
    precision: Array,
    beta: Array,
    offset: Array,
    /,
) -> Array:
    difference = left - right
    scaled_difference = difference * precision
    radius = offset * offset + jnp.sum(difference * scaled_difference, axis=-1)
    base = jnp.power(radius, beta)
    first = 2.0 * beta * jnp.power(radius, beta - 1.0)
    score_term = base * jnp.sum(left_score * right_score, axis=-1)
    gradient_term = first * jnp.sum(
        (right_score - left_score) * scaled_difference,
        axis=-1,
    )
    mixed_trace = (
        -4.0
        * beta
        * (beta - 1.0)
        * jnp.power(radius, beta - 2.0)
        * jnp.sum(scaled_difference * scaled_difference, axis=-1)
        - 2.0 * beta * jnp.power(radius, beta - 1.0) * jnp.sum(precision)
    )
    return score_term + gradient_term + mixed_trace


def _thin_chain(
    points: Array,
    scores: Array,
    method: SteinThinning,
    precision: Array,
    key: Key[Array, ""],
    /,
) -> tuple[Array, Array]:
    num_draws = int(points.shape[0])
    diagonal = _stein_kernel(
        points,
        scores,
        points,
        scores,
        precision,
        method.beta,
        method.offset,
    )
    selected = jnp.zeros((num_draws,), dtype=bool)
    indices = jnp.zeros((method.num_points,), dtype=jnp.int32)
    penalty = jnp.zeros((num_draws,), dtype=points.dtype)

    def body(iteration, state):
        chosen, used, accumulated = state
        objective = diagonal + 2.0 * accumulated
        minimum = jnp.min(jnp.where(~used, objective, jnp.inf))
        tied = (~used) & jnp.isclose(objective, minimum, rtol=1e-12, atol=1e-14)
        random_values = jr.uniform(jr.fold_in(key, iteration), (num_draws,))
        pivot = jnp.asarray(
            jnp.argmax(jnp.where(tied, random_values, -1.0)),
            dtype=jnp.int32,
        )
        chosen = chosen.at[iteration].set(pivot)
        used = used.at[pivot].set(True)
        kernel_column = _stein_kernel(
            points,
            scores,
            points[pivot],
            scores[pivot],
            precision,
            method.beta,
            method.offset,
        )
        return chosen, used, accumulated + kernel_column

    indices, _, _ = jax.lax.fori_loop(
        0,
        method.num_points,
        body,
        (indices, selected, penalty),
    )
    selected_points = points[indices]
    selected_scores = scores[indices]
    gram = _stein_kernel(
        selected_points[:, None, :],
        selected_scores[:, None, :],
        selected_points[None, :, :],
        selected_scores[None, :, :],
        precision,
        method.beta,
        method.offset,
    )
    discrepancy = jnp.sqrt(jnp.maximum(jnp.mean(gram), 0.0))
    return indices, discrepancy


def _gather_draws(leaf: Array, indices: Array, /) -> Array:
    values = jnp.asarray(leaf)
    expanded = indices.reshape(indices.shape + (1,) * (values.ndim - 2))
    gather_indices = jnp.broadcast_to(expanded, indices.shape + values.shape[2:])
    return jnp.take_along_axis(values, gather_indices, axis=1)


__all__ = ["PosteriorCoreset", "SteinThinning", "thin_posterior"]
