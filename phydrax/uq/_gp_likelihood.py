#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.kernels import AbstractPositiveDefiniteKernel, Matern32Kernel

from .._strict import StrictModule


class GaussianProcessLikelihoodState(StrictModule):
    """Resolved covariance kernel and observation-noise state for GP inference."""

    kernel: AbstractPositiveDefiniteKernel
    noise_scale: Array
    jitter: Array

    def __init__(
        self,
        *,
        kernel: AbstractPositiveDefiniteKernel | None = None,
        noise_scale: ArrayLike,
        jitter: ArrayLike = 1e-8,
    ):
        if kernel is not None and not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be an AbstractPositiveDefiniteKernel or None.")
        noise = jnp.asarray(noise_scale, dtype=float)
        if noise.ndim > 1 or (noise.ndim == 1 and noise.shape[0] == 0):
            raise ValueError("noise_scale must be scalar or a nonempty vector.")
        jitter_array = jnp.asarray(jitter, dtype=float)
        if jitter_array.ndim != 0:
            raise ValueError("jitter must be scalar.")
        self.kernel = Matern32Kernel() if kernel is None else kernel
        self.noise_scale = eqx.error_if(
            noise,
            jnp.any(~jnp.isfinite(noise)) | jnp.any(noise < 0.0),
            "noise_scale must be finite and nonnegative.",
        )
        self.jitter = eqx.error_if(
            jitter_array,
            ~jnp.isfinite(jitter_array) | (jitter_array <= 0.0),
            "jitter must be finite and strictly positive.",
        )

    @property
    def kernel_id(self) -> str:
        return self.kernel.kernel_id


__all__ = ["GaussianProcessLikelihoodState"]
