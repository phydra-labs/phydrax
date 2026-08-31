#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._linear_gaussian import LinearGaussianDynamics
from ._process import GaussianProcessDistribution


def _vector(value: ArrayLike, dimension: int, /, *, owner: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != (dimension,):
        raise ValueError(f"{owner} must have shape ({dimension},); got {array.shape}.")
    if jnp.iscomplexobj(array):
        raise TypeError(f"{owner} must be real-valued.")
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


class AbstractItoScoreDiffusion(StrictModule):
    """Euclidean Itô process exposing factor, covariance, and reverse-drift operations."""

    state_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    terminal_time: float = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    @abstractmethod
    def drift(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def diffusion_factor(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        raise NotImplementedError

    def covariance(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        factor = self.diffusion_factor(time, state)
        return factor @ jnp.swapaxes(factor, -1, -2)

    def covariance_action(
        self,
        time: ArrayLike,
        state: ArrayLike,
        covector: ArrayLike,
        /,
    ) -> Array:
        factor = self.diffusion_factor(time, state)
        vector = _vector(covector, self.dimension, owner="score covector")
        return oe.contract("ik,jk,j->i", factor, factor, vector)

    def covariance_divergence(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        value = _vector(state, self.dimension, owner="state")

        def covariance_at(current):
            return self.covariance(time, current)

        derivative = jax.jacfwd(covariance_at)(value)
        return oe.contract("ijj->i", derivative)

    def reverse_drift(
        self,
        reverse_time: ArrayLike,
        state: ArrayLike,
        score: ArrayLike,
        /,
    ) -> Array:
        time = self.terminal_time - jnp.asarray(reverse_time)
        return (
            -self.drift(time, state)
            + self.covariance_divergence(time, state)
            + self.covariance_action(time, state, score)
        )

    def probability_flow_drift(
        self,
        reverse_time: ArrayLike,
        state: ArrayLike,
        score: ArrayLike,
        /,
    ) -> Array:
        time = self.terminal_time - jnp.asarray(reverse_time)
        return (
            -self.drift(time, state)
            + 0.5 * self.covariance_divergence(time, state)
            + 0.5 * self.covariance_action(time, state, score)
        )


class MatrixGaussianDiffusion(AbstractItoScoreDiffusion):
    """Constant-coefficient affine Gaussian diffusion with exact LTI transitions."""

    dynamics: LinearGaussianDynamics

    def __init__(
        self,
        drift_matrix: ArrayLike,
        dispersion: ArrayLike,
        /,
        *,
        offset: ArrayLike = 0.0,
        terminal_time: float = 1.0,
        process_id: str | None = None,
    ):
        matrix = jnp.asarray(drift_matrix)
        factor = jnp.asarray(dispersion)
        if jnp.iscomplexobj(matrix) or jnp.iscomplexobj(factor):
            raise TypeError("Matrix Gaussian diffusion coefficients must be real-valued.")
        if jnp.iscomplexobj(jnp.asarray(offset)):
            raise TypeError("Matrix Gaussian diffusion offset must be real-valued.")
        if not jnp.issubdtype(matrix.dtype, jnp.inexact):
            matrix = matrix.astype(float)
        if not jnp.issubdtype(factor.dtype, jnp.inexact):
            factor = factor.astype(float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("drift_matrix must be square.")
        dimension = int(matrix.shape[0])
        if factor.ndim != 2 or factor.shape[0] != dimension or factor.shape[1] <= 0:
            raise ValueError("dispersion must have shape (dimension, noise_dimension).")
        horizon = float(terminal_time)
        if not isfinite(horizon) or horizon <= 0.0:
            raise ValueError("terminal_time must be finite and positive.")
        resolved = process_id or canonical_fingerprint(
            {
                "kind": "matrix-gaussian-diffusion",
                "dimension": dimension,
                "noise_dimension": int(factor.shape[1]),
                "terminal_time": horizon,
            }
        )
        dynamics = LinearGaussianDynamics(
            matrix,
            factor,
            state_shape=(dimension,),
            offset=offset,
            dynamics_id=f"matrix-gaussian:{resolved}",
            process_id=resolved,
            approximation_id="exact-lti",
        )
        self.state_shape = (dimension,)
        self.dimension = dimension
        self.terminal_time = horizon
        self.process_id = resolved
        self.dynamics = dynamics

    def drift(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        return self.dynamics(time, state, None)

    def diffusion_factor(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        del time
        _vector(state, self.dimension, owner="state")
        return self.dynamics.dispersion


    def marginal_transition(
        self,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
    ) -> GaussianProcessDistribution:
        value = _vector(state, self.dimension, owner="state")
        parameters = self.dynamics.parameters(t0, t1, None)
        mean = parameters.transition @ value + parameters.offset
        return GaussianProcessDistribution(
            mean,
            parameters.covariance,
            event_shape=self.state_shape,
        )

    def conditional_score(
        self,
        value: ArrayLike,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
    ) -> Array:
        distribution = self.marginal_transition(state, t0=t0, t1=t1)
        observation = _vector(value, self.dimension, owner="value")
        residual = observation - distribution.mean
        solved = jsp.linalg.solve_triangular(
            distribution.scale_tril,
            residual,
            lower=True,
        )
        return -jsp.linalg.solve_triangular(
            jnp.swapaxes(distribution.scale_tril, -1, -2),
            solved,
            lower=False,
        )


class StateDependentItoDiffusion(AbstractItoScoreDiffusion):
    """Callable state-dependent Itô diffusion with exact covariance divergence."""

    drift_function: Any
    factor_function: Any
    noise_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        drift,
        diffusion_factor,
        /,
        *,
        dimension: int,
        noise_dimension: int,
        terminal_time: float = 1.0,
        process_id: str,
    ):
        if not callable(drift) or not callable(diffusion_factor):
            raise TypeError("drift and diffusion_factor must be callable.")
        size = int(dimension)
        noise = int(noise_dimension)
        horizon = float(terminal_time)
        if size <= 0 or noise <= 0:
            raise ValueError("State and noise dimensions must be positive.")
        if not isfinite(horizon) or horizon <= 0.0:
            raise ValueError("terminal_time must be finite and positive.")
        if not isinstance(process_id, str) or not process_id:
            raise ValueError("process_id must be non-empty.")
        probe = jnp.zeros((size,))
        drift_value = jnp.asarray(drift(jnp.asarray(0.0), probe))
        factor_value = jnp.asarray(diffusion_factor(jnp.asarray(0.0), probe))
        if jnp.iscomplexobj(drift_value) or jnp.iscomplexobj(factor_value):
            raise TypeError("State-dependent Itô coefficients must be real-valued.")
        if drift_value.shape != (size,):
            raise ValueError("drift must return one state vector.")
        if factor_value.shape != (size, noise):
            raise ValueError("diffusion_factor returned an incompatible shape.")
        self.state_shape = (size,)
        self.dimension = size
        self.terminal_time = horizon
        self.process_id = process_id
        self.drift_function = drift
        self.factor_function = diffusion_factor
        self.noise_dimension = noise

    def drift(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        value = _vector(state, self.dimension, owner="state")
        return _vector(
            self.drift_function(jnp.asarray(time), value),
            self.dimension,
            owner="state-dependent drift",
        )

    def diffusion_factor(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        value = _vector(state, self.dimension, owner="state")
        result = jnp.asarray(self.factor_function(jnp.asarray(time), value))
        if jnp.iscomplexobj(result):
            raise TypeError("State-dependent diffusion factor must be real-valued.")
        if not jnp.issubdtype(result.dtype, jnp.inexact):
            result = result.astype(float)
        if result.shape != (self.dimension, self.noise_dimension):
            raise ValueError("State-dependent diffusion factor changed shape.")
        return result


__all__ = [
    "AbstractItoScoreDiffusion",
    "MatrixGaussianDiffusion",
    "StateDependentItoDiffusion",
]
