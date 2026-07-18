#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ._action import _paths_array, potential_action
from ._diffusion import DiffusionLike, DriftLike, sample_diffusion_paths
from ._discretization import PathDiscretization
from ._estimate import _estimate_positive_log_weights, PathIntegralEstimate
from ._potential import _as_point_time_callable, PotentialLike


def _terminal_values(
    terminal: PotentialLike,
    paths: Array,
    /,
    *,
    slicing: PathDiscretization,
    position_var: str,
    time_var: str,
    key: Key[Array, ""],
) -> Array:
    endpoint = paths[..., -1, :]
    state_dim = int(endpoint.shape[-1])
    flat_endpoint = jnp.reshape(endpoint, (-1, state_dim))
    terminal_fn = _as_point_time_callable(
        terminal,
        position_var=position_var,
        time_var=time_var,
        key=key,
        role="Terminal function",
    )
    values = jnp.asarray(
        jax.vmap(terminal_fn, in_axes=(0, None))(flat_endpoint, slicing.t1)
    )
    expected_shape = (flat_endpoint.shape[0],)
    if values.shape != expected_shape:
        raise ValueError(
            "terminal must return one real scalar per endpoint; "
            f"got {values.shape}, expected {expected_shape}."
        )
    if jnp.iscomplexobj(values):
        raise TypeError("Feynman-Kac terminal values must be real.")
    values = eqx.error_if(
        values,
        ~jnp.all(jnp.isfinite(values)),
        "Feynman-Kac terminal values must be finite.",
    )
    return jnp.reshape(values, endpoint.shape[:-1])


def feynman_kac_from_paths(
    terminal: PotentialLike,
    paths: ArrayLike,
    /,
    *,
    slicing: PathDiscretization,
    killing: PotentialLike | None = None,
    position_var: str = "x",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> PathIntegralEstimate:
    r"""Estimate a terminal Feynman--Kac expectation from supplied paths.

    The estimator is
    $\mathbb E[g(X_{t_1},t_1)\exp(-\int_{t_0}^{t_1}c(X_s,s)\,ds)]$.
    Midpoint quadrature is used for the optional killing rate ``c``.
    """
    q = _paths_array(paths, slicing)
    count = int(q.shape[-3])
    if count < 1:
        raise ValueError("paths must contain at least one path.")
    terminal_key, killing_key = jr.split(key)
    terminal_value = _terminal_values(
        terminal,
        q,
        slicing=slicing,
        position_var=position_var,
        time_var=time_var,
        key=terminal_key,
    )
    if killing is None:
        log_weights = jnp.zeros(q.shape[:-2], dtype=q.dtype)
    else:
        log_weights = -potential_action(
            q,
            killing,
            slicing=slicing,
            position_var=position_var,
            time_var=time_var,
            key=killing_key,
        )

    max_log_weight = jnp.max(log_weights, axis=-1, keepdims=True)
    scaled_weight = jnp.exp(log_weights - max_log_weight)
    scaled_samples = terminal_value * scaled_weight
    scale = jnp.exp(jnp.squeeze(max_log_weight, axis=-1))
    value = scale * jnp.mean(scaled_samples, axis=-1)
    if count == 1:
        standard_error = jnp.full_like(value, jnp.nan)
    else:
        centered = scaled_samples - jnp.mean(
            scaled_samples,
            axis=-1,
            keepdims=True,
        )
        sample_variance = jnp.sum(centered * centered, axis=-1) / float(count - 1)
        standard_error = scale * jnp.sqrt(sample_variance / float(count))

    weight_estimate = _estimate_positive_log_weights(
        log_weights,
        scale=jnp.ones_like(value),
    )
    return PathIntegralEstimate(
        value=value,
        standard_error=standard_error,
        effective_sample_size=weight_estimate.effective_sample_size,
        log_mean_weight=weight_estimate.log_mean_weight,
        num_paths=count,
    )


def feynman_kac_expectation(
    terminal: PotentialLike,
    drift: DriftLike,
    diffusion: DiffusionLike,
    x0: ArrayLike,
    /,
    *,
    slicing: PathDiscretization,
    num_paths: int,
    killing: PotentialLike | None = None,
    position_var: str = "x",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> PathIntegralEstimate:
    """Simulate diffusion paths and estimate a terminal Feynman--Kac value."""
    path_key, estimate_key = jr.split(key)
    paths = sample_diffusion_paths(
        drift,
        diffusion,
        x0,
        slicing=slicing,
        num_paths=num_paths,
        position_var=position_var,
        time_var=time_var,
        key=path_key,
    )
    return feynman_kac_from_paths(
        terminal,
        paths,
        slicing=slicing,
        killing=killing,
        position_var=position_var,
        time_var=time_var,
        key=estimate_key,
    )


__all__ = [
    "feynman_kac_expectation",
    "feynman_kac_from_paths",
]
