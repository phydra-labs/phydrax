#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


class LinearGaussianParameters(NamedTuple):
    """Resolved parameters of one affine Gaussian transition interval."""

    transition: Array
    offset: Array
    covariance: Array


ParameterValue = Array | Callable[[Array, Array], ArrayLike]


def _shape(value: Sequence[int], /) -> tuple[int, ...]:
    resolved = tuple(int(size) for size in value)
    if any(size <= 0 for size in resolved):
        raise ValueError("state_shape dimensions must be positive.")
    return resolved


def _name(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _parameter(value: ParameterValue, start: Array, end: Array, /) -> Array:
    if callable(value):
        function = cast(Callable[[Array, Array], ArrayLike], value)
        return jnp.asarray(function(start, end))
    return jnp.asarray(value)


class LinearGaussianParameterization(StrictModule):
    """One interval-parameter source shared by transition sampling and inference."""

    transition: ParameterValue
    offset: ParameterValue
    covariance: ParameterValue
    state_shape: tuple[int, ...] = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        transition: ArrayLike | Callable[[Array, Array], ArrayLike],
        covariance: ArrayLike | Callable[[Array, Array], ArrayLike],
        /,
        *,
        state_shape: Sequence[int],
        offset: ArrayLike | Callable[[Array, Array], ArrayLike] = 0.0,
        parameterization_id: str = "linear-gaussian-parameters",
        resolved_method: str = "provided",
    ):
        for owner, value in (
            ("transition", transition),
            ("covariance", covariance),
            ("offset", offset),
        ):
            if not callable(value) and bool(jnp.any(~jnp.isfinite(jnp.asarray(value)))):
                raise ValueError(f"{owner} must be finite.")
        self.transition = (
            cast(Callable[[Array, Array], ArrayLike], transition)
            if callable(transition)
            else jnp.asarray(transition)
        )
        self.offset = (
            cast(Callable[[Array, Array], ArrayLike], offset)
            if callable(offset)
            else jnp.asarray(offset)
        )
        self.covariance = (
            cast(Callable[[Array, Array], ArrayLike], covariance)
            if callable(covariance)
            else jnp.asarray(covariance)
        )
        self.state_shape = _shape(state_shape)
        self.parameterization_id = _name(
            parameterization_id, owner="parameterization_id"
        )
        self.resolved_method = _name(resolved_method, owner="resolved_method")

    def parameters(self, t0: ArrayLike, t1: ArrayLike, /) -> LinearGaussianParameters:
        start, end = jnp.asarray(t0), jnp.asarray(t1)
        size = prod(self.state_shape) if self.state_shape else 1
        transition = _parameter(self.transition, start, end)
        covariance = _parameter(self.covariance, start, end)
        offset = _parameter(self.offset, start, end)
        if transition.shape[-2:] != (size, size):
            raise ValueError("transition must end in state_size by state_size.")
        if covariance.shape[-2:] != (size, size):
            raise ValueError("covariance must end in state_size by state_size.")
        offset = jnp.broadcast_to(offset, transition.shape[:-2] + (size,))
        covariance = jnp.broadcast_to(covariance, transition.shape)
        return LinearGaussianParameters(transition, offset, covariance)


class LinearGaussianDynamics(StrictModule):
    r"""Constant-coefficient affine Itô dynamics with exact LTI discretization.

    The continuous model is
    :math:`dX_t = (A X_t + b)dt + LdW_t`. ``dispersion`` is the factor ``L``,
    not an already-squared covariance. Calling the object follows the standard
    dynamics contract ``(time, state, args) -> state-shaped array``; ``parameters``
    returns the exact affine Gaussian transition over an interval.
    """

    drift_matrix: Array
    offset: Array
    dispersion: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        drift_matrix: ArrayLike,
        dispersion: ArrayLike,
        /,
        *,
        state_shape: Sequence[int],
        offset: ArrayLike = 0.0,
        dynamics_id: str = "linear-gaussian-lti",
        process_id: str = "linear-gaussian",
        approximation_id: str = "exact-lti",
    ):
        shape = _shape(state_shape)
        size = prod(shape) if shape else 1
        matrix = jnp.asarray(drift_matrix)
        factor = jnp.asarray(dispersion)
        if matrix.shape != (size, size):
            raise ValueError("drift_matrix must have shape (state_size, state_size).")
        if factor.ndim != 2 or factor.shape[0] != size or factor.shape[1] <= 0:
            raise ValueError("dispersion must have shape (state_size, noise_size).")
        dtype = jnp.result_type(matrix, factor)
        if not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.asarray(0.0).dtype
        matrix = matrix.astype(dtype)
        factor = factor.astype(dtype)
        affine = jnp.broadcast_to(jnp.asarray(offset, dtype=dtype), (size,))
        if bool(
            jnp.any(~jnp.isfinite(matrix))
            | jnp.any(~jnp.isfinite(factor))
            | jnp.any(~jnp.isfinite(affine))
        ):
            raise ValueError("Linear Gaussian dynamics coefficients must be finite.")
        self.drift_matrix = matrix
        self.offset = affine
        self.dispersion = factor
        self.state_shape = shape
        self.dynamics_id = _name(dynamics_id, owner="dynamics_id")
        self.process_id = _name(process_id, owner="process_id")
        self.approximation_id = _name(approximation_id, owner="approximation_id")
        self.resolved_method = "matrix-exponential/augmented-exponential/van-loan"

    def __call__(self, time: ArrayLike, state: ArrayLike, args, /) -> Array:
        del time, args
        values = jnp.asarray(state)
        if (
            self.state_shape
            and tuple(values.shape[-len(self.state_shape) :]) != self.state_shape
        ):
            raise ValueError(
                f"state must end in shape {self.state_shape}; got {values.shape}."
            )
        batch_shape = (
            values.shape[: -len(self.state_shape)] if self.state_shape else values.shape
        )
        flat = values.reshape(batch_shape + (self.drift_matrix.shape[0],))
        drift = jnp.einsum("ij,...j->...i", self.drift_matrix, flat) + self.offset
        return drift.reshape(values.shape)

    def parameters(self, t0: ArrayLike, t1: ArrayLike, /) -> LinearGaussianParameters:
        start, end = jnp.asarray(t0), jnp.asarray(t1)
        duration = end - start
        dtype = jnp.result_type(self.drift_matrix, duration)
        matrix = self.drift_matrix.astype(dtype)
        offset = self.offset.astype(dtype)
        dispersion = self.dispersion.astype(dtype)
        size = matrix.shape[0]

        transition = jax.scipy.linalg.expm(matrix * duration)
        augmented = jnp.zeros((size + 1, size + 1), dtype=dtype)
        augmented = augmented.at[:size, :size].set(matrix)
        augmented = augmented.at[:size, size].set(offset)
        affine = jax.scipy.linalg.expm(augmented * duration)[:size, size]

        covariance_rate = dispersion @ dispersion.T
        van_loan = jnp.zeros((2 * size, 2 * size), dtype=dtype)
        van_loan = van_loan.at[:size, :size].set(matrix)
        van_loan = van_loan.at[:size, size:].set(covariance_rate)
        van_loan = van_loan.at[size:, size:].set(-matrix.T)
        van_loan_exponential = jax.scipy.linalg.expm(van_loan * duration)
        covariance = van_loan_exponential[:size, size:] @ transition.T
        covariance = 0.5 * (covariance + covariance.T)

        is_zero = duration == jnp.zeros((), dtype=dtype)
        transition = jnp.where(is_zero, jnp.eye(size, dtype=dtype), transition)
        affine = jnp.where(is_zero, jnp.zeros_like(affine), affine)
        covariance = jnp.where(is_zero, jnp.zeros_like(covariance), covariance)
        return LinearGaussianParameters(transition, affine, covariance)

    def discretize(self, t0: ArrayLike, t1: ArrayLike, /) -> LinearGaussianParameters:
        """Return exact interval parameters; equivalent to ``parameters``."""
        return self.parameters(t0, t1)


def degenerate_gaussian_log_prob(residual: Array, covariance: Array, /) -> Array:
    """Evaluate a Gaussian log density on its covariance support without jitter."""
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    scale = jnp.max(jnp.abs(eigenvalues), axis=-1, keepdims=True)
    tolerance = jnp.finfo(covariance.dtype).eps * covariance.shape[-1] * scale
    positive = eigenvalues > tolerance
    invalid = jnp.any(eigenvalues < -tolerance, axis=-1)
    coordinates = jnp.einsum("...ji,...j->...i", eigenvectors, residual)
    supported = jnp.all(
        jnp.where(positive, True, jnp.abs(coordinates) <= jnp.sqrt(tolerance)),
        axis=-1,
    )
    safe_values = jnp.where(positive, eigenvalues, 1.0)
    quadratic = jnp.sum(
        jnp.where(positive, coordinates**2 / safe_values, 0.0), axis=-1
    )
    logdet = jnp.sum(jnp.where(positive, jnp.log(safe_values), 0.0), axis=-1)
    rank = jnp.sum(positive, axis=-1)
    value = -0.5 * (quadratic + logdet + rank * jnp.log(2.0 * jnp.pi))
    return jnp.where(~invalid & supported, value, -jnp.inf)


__all__ = [
    "LinearGaussianDynamics",
    "LinearGaussianParameterization",
    "LinearGaussianParameters",
]
