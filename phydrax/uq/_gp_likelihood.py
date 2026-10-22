#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from phydrax.kernels import AbstractPositiveDefiniteKernel, Matern32Kernel
from phydrax.linalg import ConditionTargetPolicy

from .._strict import StrictModule
from ._gp_exact import ExactGaussianProcessExecutionPolicy


class GaussianProcessLikelihoodState(StrictModule):
    """Resolved covariance kernel and observation-noise state for GP inference."""

    kernel: AbstractPositiveDefiniteKernel
    noise_scale: Array
    jitter: Array
    execution: ExactGaussianProcessExecutionPolicy
    conditioning: ConditionTargetPolicy | None

    def __init__(
        self,
        *,
        kernel: AbstractPositiveDefiniteKernel | None = None,
        noise_scale: ArrayLike,
        jitter: ArrayLike = 1e-8,
        execution: ExactGaussianProcessExecutionPolicy | None = None,
        conditioning: ConditionTargetPolicy | None = None,
    ) -> None:
        if kernel is not None and not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be an AbstractPositiveDefiniteKernel or None.")
        if execution is not None and not isinstance(
            execution, ExactGaussianProcessExecutionPolicy
        ):
            raise TypeError(
                "execution must be an ExactGaussianProcessExecutionPolicy or None."
            )
        if conditioning is not None and not isinstance(
            conditioning, ConditionTargetPolicy
        ):
            raise TypeError("conditioning must be a ConditionTargetPolicy or None.")
        noise = jnp.asarray(noise_scale, dtype=jnp.float64)
        if noise.ndim > 1 or (noise.ndim == 1 and noise.shape[0] == 0):
            raise ValueError("noise_scale must be scalar or a nonempty vector.")
        jitter_array = jnp.asarray(jitter, dtype=jnp.float64)
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
        self.execution = (
            ExactGaussianProcessExecutionPolicy() if execution is None else execution
        )
        self.conditioning = conditioning

    @property
    def kernel_id(self) -> str:
        return self.kernel.kernel_id


def _require_unconditioned_state(state: GaussianProcessLikelihoodState, /) -> None:
    """Refuse opt-in covariance changes outside their exact scalar owner."""
    if not isinstance(state, GaussianProcessLikelihoodState):
        raise TypeError("state must be a GaussianProcessLikelihoodState.")
    if state.conditioning is not None:
        raise ValueError(
            "Condition-target covariance is supported only by exact scalar GP "
            "execution, not finite-feature, FITC, action-projected, classification, "
            "or Bayesian-optimization execution."
        )


__all__ = ["GaussianProcessLikelihoodState"]
