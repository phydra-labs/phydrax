#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    matrix_exponential_action,
    matrix_phi1_action,
    MatrixFunctionPolicy,
    MatrixFunctionResult,
)


class AffineExponentialResult(StrictModule):
    """Value and numerical evidence for one frozen affine flow."""

    value: PyTree[Array]
    exponential_action: MatrixFunctionResult
    forcing_action: MatrixFunctionResult
    successful: Array
    residual_estimate: Array
    matvec_count: Array


def _duration_array(
    duration: ArrayLike,
    operator: AbstractLinearOperator,
    /,
) -> Array:
    value = jnp.asarray(duration)
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise TypeError("duration must be real-valued.")
    expected_shape = operator.batch_shape if operator.batch_shape else ()
    if value.shape != expected_shape:
        raise ValueError(
            "duration must be scalar for an unbatched operator or have shape "
            f"{operator.batch_shape} for a batched operator; got {value.shape}."
        )
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype(jnp.result_type(value, jnp.float64))
    value = eqx.error_if(value, ~jnp.all(jnp.isfinite(value)), "duration must be finite.")
    return eqx.error_if(value, jnp.any(value < 0), "duration must be non-negative.")


def _expand_batch_scalar(
    scalar: Array,
    leaf: Array,
    batch_ndim: int,
    /,
) -> Array:
    return jnp.reshape(scalar, scalar.shape + (1,) * (leaf.ndim - batch_ndim))


def _scale_tree(
    scalar: Array,
    tree: PyTree[Array],
    batch_ndim: int,
    /,
) -> PyTree[Array]:
    return jax.tree.map(
        lambda leaf: _expand_batch_scalar(scalar, leaf, batch_ndim) * leaf,
        tree,
    )


def _select_zero_duration(
    duration: Array,
    initial: PyTree[Array],
    candidate: PyTree[Array],
    batch_ndim: int,
    /,
) -> PyTree[Array]:
    return jax.tree.map(
        lambda base, value: jnp.where(
            _expand_batch_scalar(duration == 0, value, batch_ndim),
            base,
            value,
        ),
        initial,
        candidate,
    )


def _finite_by_batch(
    value: PyTree[Array],
    batch_shape: tuple[int, ...],
    /,
) -> Array:
    batch_ndim = len(batch_shape)
    finite = jnp.ones(batch_shape if batch_shape else (), dtype=bool)
    for leaf in jax.tree.leaves(value):
        axes = tuple(range(batch_ndim, leaf.ndim))
        leaf_finite = (
            jnp.all(jnp.isfinite(leaf), axis=axes) if axes else jnp.isfinite(leaf)
        )
        finite = finite & leaf_finite
    return finite


def affine_exponential_step(
    operator: AbstractLinearOperator,
    initial_state: PyTree[Any],
    forcing: PyTree[Any],
    duration: ArrayLike,
    /,
    *,
    policy: MatrixFunctionPolicy | None = None,
    spectral: Any | None = None,
    spectral_bounds: tuple[float, float] | None = None,
) -> AffineExponentialResult:
    """Advance ``x' = A x + b`` with frozen ``A`` and ``b``.

    The affine contribution is evaluated as ``h φ₁(hA)b``. No inverse of
    ``A`` is formed, so singular and zero operators retain their exact limits.
    """

    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not operator.source.compatible(operator.target):
        raise ValueError("Affine exponential flow requires one endomorphism.")
    duration_ = _duration_array(duration, operator)
    exponential = matrix_exponential_action(
        operator,
        initial_state,
        duration_,
        policy=policy,
        spectral=spectral,
        spectral_bounds=spectral_bounds,
    )
    forcing_action = matrix_phi1_action(
        operator,
        forcing,
        duration_,
        policy=policy,
        spectral=spectral,
        spectral_bounds=spectral_bounds,
    )
    batch_ndim = len(operator.batch_shape)
    affine = _scale_tree(duration_, forcing_action.value, batch_ndim)
    candidate = jax.tree.map(lambda left, right: left + right, exponential.value, affine)
    value = _select_zero_duration(
        duration_,
        initial_state,
        candidate,
        batch_ndim,
    )
    finite = _finite_by_batch(value, operator.batch_shape)
    successful = exponential.converged & forcing_action.converged & finite
    residual_estimate = exponential.error_estimate + jnp.abs(duration_) * (
        forcing_action.error_estimate
    )
    matvec_count = exponential.matvec_count + forcing_action.matvec_count
    return AffineExponentialResult(
        value,
        exponential,
        forcing_action,
        successful,
        residual_estimate,
        matvec_count,
    )


__all__ = ["AffineExponentialResult", "affine_exponential_step"]
