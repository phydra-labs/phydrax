#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


MatrixFunctionKind: TypeAlias = Literal["exp", "phi1"]
MatrixFunctionMethod: TypeAlias = Literal[
    "auto",
    "spectral",
    "chebyshev",
    "lanczos",
    "arnoldi",
]
MatrixFunctionDifferentiation: TypeAlias = Literal["forward", "reverse"]


class MatrixFunctionPolicy(StrictModule):
    """Public policy for matrix-free exponential and φ₁ actions."""

    method: MatrixFunctionMethod = eqx.field(static=True)
    num_matvecs: int = eqx.field(static=True)
    reorthogonalization: Literal["none", "full"] = eqx.field(static=True)
    differentiation: MatrixFunctionDifferentiation = eqx.field(static=True)

    def __init__(
        self,
        method: MatrixFunctionMethod = "auto",
        /,
        *,
        num_matvecs: int = 32,
        reorthogonalization: Literal["none", "full"] = "full",
        differentiation: MatrixFunctionDifferentiation = "reverse",
    ):
        if method not in ("auto", "spectral", "chebyshev", "lanczos", "arnoldi"):
            raise ValueError(f"Unknown matrix-function method {method!r}.")
        count = int(num_matvecs)
        if count <= 0:
            raise ValueError("num_matvecs must be positive.")
        if reorthogonalization not in ("none", "full"):
            raise ValueError("reorthogonalization must be 'none' or 'full'.")
        if differentiation not in ("forward", "reverse"):
            raise ValueError("differentiation must be 'forward' or 'reverse'.")
        self.method = method
        self.num_matvecs = count
        self.reorthogonalization = reorthogonalization
        self.differentiation = differentiation


class SpectralMatrixRepresentation(StrictModule):
    """Exact finite spectral representation in physical coordinates."""

    eigenvalues: Array
    analysis: Array
    synthesis: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    representation_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        eigenvalues: ArrayLike,
        analysis: ArrayLike,
        synthesis: ArrayLike,
        /,
        *,
        state_shape: Sequence[int],
        representation_id: str | None = None,
    ):
        shape = tuple(int(size) for size in state_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("state_shape must contain positive dimensions.")
        count = int(np.prod(shape))
        values = jnp.asarray(eigenvalues, dtype=float).reshape((-1,))
        analysis_array = jnp.asarray(analysis, dtype=float)
        synthesis_array = jnp.asarray(synthesis, dtype=float)
        rank = int(values.size)
        if rank <= 0:
            raise ValueError("A spectral representation requires at least one mode.")
        if analysis_array.shape != (rank, count):
            raise ValueError(
                f"analysis must have shape {(rank, count)}; got {analysis_array.shape}."
            )
        if synthesis_array.shape != (count, rank):
            raise ValueError(
                f"synthesis must have shape {(count, rank)}; got {synthesis_array.shape}."
            )
        if not bool(
            jnp.allclose(
                analysis_array @ synthesis_array,
                jnp.eye(rank),
                rtol=1e-5,
                atol=1e-6,
            )
        ):
            raise ValueError("analysis and synthesis must be biorthogonal.")
        if representation_id is not None and (
            not isinstance(representation_id, str) or not representation_id
        ):
            raise ValueError("representation_id must be non-empty or None.")
        self.eigenvalues = values
        self.analysis = analysis_array
        self.synthesis = synthesis_array
        self.state_shape = shape
        self.representation_id = representation_id

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.size)

    def apply(self, vector: ArrayLike, multiplier: ArrayLike, /) -> Array:
        value = jnp.asarray(vector)
        spatial_rank = len(self.state_shape)
        if tuple(value.shape[:spatial_rank]) != self.state_shape:
            raise ValueError(
                f"vector must begin with state_shape {self.state_shape}; got {value.shape}."
            )
        trailing = value.shape[spatial_rank:]
        flattened = value.reshape((int(np.prod(self.state_shape)),) + trailing)
        coefficients = jnp.tensordot(self.analysis, flattened, axes=((1,), (0,)))
        scale = jnp.asarray(multiplier).reshape((self.rank,) + (1,) * len(trailing))
        result = jnp.tensordot(
            self.synthesis,
            scale * coefficients,
            axes=((1,), (0,)),
        )
        return result.reshape(self.state_shape + trailing)


def _phi1(value: Array, /) -> Array:
    threshold = jnp.sqrt(jnp.finfo(value.dtype).eps)
    series = 1.0 + 0.5 * value + value**2 / 6.0 + value**3 / 24.0
    safe = jnp.where(jnp.abs(value) > threshold, value, 1.0)
    quotient = jnp.expm1(value) / safe
    return jnp.where(jnp.abs(value) > threshold, quotient, series)


def _scalar_function(value: Array, kind: MatrixFunctionKind, /) -> Array:
    if kind == "exp":
        return jnp.exp(value)
    if kind == "phi1":
        return _phi1(value)
    raise ValueError("kind must be 'exp' or 'phi1'.")


def _small_matrix_function(
    matrix: Array,
    step: Array,
    kind: MatrixFunctionKind,
    /,
) -> Array:
    scaled = step * matrix
    if kind == "exp":
        return jsp.linalg.expm(scaled)
    if kind != "phi1":
        raise ValueError("kind must be 'exp' or 'phi1'.")
    size = int(matrix.shape[0])
    zero = jnp.zeros_like(scaled)
    identity = jnp.eye(size, dtype=scaled.dtype)
    augmented = jnp.concatenate(
        (
            jnp.concatenate((scaled, identity), axis=1),
            jnp.concatenate((zero, zero), axis=1),
        ),
        axis=0,
    )
    return jsp.linalg.expm(augmented)[:size, size:]


def _arnoldi_action(
    operator: Callable[[Array], Array],
    vector: Array,
    step: Array,
    kind: MatrixFunctionKind,
    /,
    *,
    num_matvecs: int,
    full_reorthogonalization: bool,
) -> Array:
    size = int(vector.size)
    iterations = min(int(num_matvecs), size)
    norm = jnp.linalg.norm(vector)
    safe_norm = jnp.where(norm > 0.0, norm, 1.0)
    basis = jnp.zeros((size, iterations + 1), dtype=vector.dtype)
    basis = basis.at[:, 0].set(vector / safe_norm)
    hessenberg = jnp.zeros((iterations + 1, iterations), dtype=vector.dtype)

    def body(index, carry):
        vectors, matrix = carry
        image = operator(vectors[:, index])
        active = (jnp.arange(iterations + 1) <= index).astype(vector.dtype)
        coefficients = (vectors.T @ image) * active
        residual = image - vectors @ coefficients
        if full_reorthogonalization:
            correction = (vectors.T @ residual) * active
            coefficients = coefficients + correction
            residual = residual - vectors @ correction
        residual_norm = jnp.linalg.norm(residual)
        matrix = matrix.at[:, index].set(coefficients)
        matrix = matrix.at[index + 1, index].set(residual_norm)
        next_vector = jnp.where(
            residual_norm > jnp.finfo(vector.dtype).eps,
            residual / jnp.where(residual_norm > 0.0, residual_norm, 1.0),
            jnp.zeros_like(residual),
        )
        vectors = vectors.at[:, index + 1].set(next_vector)
        return vectors, matrix

    basis, hessenberg = jax.lax.fori_loop(
        0,
        iterations,
        body,
        (basis, hessenberg),
    )
    projected = hessenberg[:iterations, :]
    function = _small_matrix_function(projected, step, kind)
    coefficients = function[:, 0]
    result = safe_norm * (basis[:, :iterations] @ coefficients)
    return jnp.where(norm > 0.0, result, jnp.zeros_like(result))


def _lanczos_action(
    operator: Callable[[Array], Array],
    vector: Array,
    step: Array,
    kind: MatrixFunctionKind,
    /,
    *,
    num_matvecs: int,
    full_reorthogonalization: bool,
) -> Array:
    size = int(vector.size)
    iterations = min(int(num_matvecs), size)
    norm = jnp.linalg.norm(vector)
    safe_norm = jnp.where(norm > 0.0, norm, 1.0)
    basis = jnp.zeros((size, iterations), dtype=vector.dtype)
    basis = basis.at[:, 0].set(vector / safe_norm)
    diagonal = jnp.zeros((iterations,), dtype=vector.dtype)
    off_diagonal = jnp.zeros((iterations,), dtype=vector.dtype)

    def body(index, carry):
        vectors, alpha_values, beta_values = carry
        current = vectors[:, index]
        previous = jax.lax.cond(
            index > 0,
            lambda _: vectors[:, index - 1],
            lambda _: jnp.zeros_like(current),
            operand=None,
        )
        previous_beta = jax.lax.cond(
            index > 0,
            lambda _: beta_values[index - 1],
            lambda _: jnp.asarray(0.0, dtype=vector.dtype),
            operand=None,
        )
        residual = operator(current) - previous_beta * previous
        alpha = jnp.vdot(current, residual).real
        residual = residual - alpha * current
        if full_reorthogonalization:
            active = (jnp.arange(iterations) <= index).astype(vector.dtype)
            residual = residual - vectors @ ((vectors.T @ residual) * active)
        beta = jnp.linalg.norm(residual)
        alpha_values = alpha_values.at[index].set(alpha)

        def set_next(values):
            vectors_value, beta_value = values
            beta_values_value = beta_values.at[index].set(beta_value)
            next_vector = jnp.where(
                beta_value > jnp.finfo(vector.dtype).eps,
                residual / jnp.where(beta_value > 0.0, beta_value, 1.0),
                jnp.zeros_like(residual),
            )
            return vectors_value.at[:, index + 1].set(next_vector), beta_values_value

        vectors, beta_values = jax.lax.cond(
            index < iterations - 1,
            set_next,
            lambda values: (values[0], beta_values),
            (vectors, beta),
        )
        return vectors, alpha_values, beta_values

    basis, diagonal, off_diagonal = jax.lax.fori_loop(
        0,
        iterations,
        body,
        (basis, diagonal, off_diagonal),
    )
    projected = jnp.diag(diagonal)
    if iterations > 1:
        couplings = off_diagonal[: iterations - 1]
        projected = projected + jnp.diag(couplings, 1) + jnp.diag(couplings, -1)
    function = _small_matrix_function(projected, step, kind)
    result = safe_norm * (basis @ function[:, 0])
    return jnp.where(norm > 0.0, result, jnp.zeros_like(result))


def _chebyshev_action(
    operator: Callable[[Array], Array],
    vector: Array,
    step: Array,
    kind: MatrixFunctionKind,
    spectral_bounds: tuple[float, float],
    /,
    *,
    num_matvecs: int,
) -> Array:
    lower, upper = (float(value) for value in spectral_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
        raise ValueError("spectral_bounds must be finite and strictly increasing.")
    degree = int(num_matvecs)
    indices = jnp.arange(degree, dtype=vector.dtype)
    theta = jnp.pi * (indices + 0.5) / float(degree)
    nodes = jnp.cos(theta)
    center = 0.5 * (upper + lower)
    radius = 0.5 * (upper - lower)
    values = _scalar_function(step * (center + radius * nodes), kind)
    coefficients = (2.0 / float(degree)) * (
        jnp.cos(indices[:, None] * theta[None, :]) @ values
    )

    def normalized_operator(value):
        return (operator(value) - center * value) / radius

    first = vector
    out = 0.5 * coefficients[0] * first
    if degree == 1:
        return out
    second = normalized_operator(vector)
    out = out + coefficients[1] * second
    for index in range(2, degree):
        current = 2.0 * normalized_operator(second) - first
        out = out + coefficients[index] * current
        first, second = second, current
    return out


def matrix_function_action(
    operator: Callable[[Array], ArrayLike],
    vector: ArrayLike,
    step: ArrayLike,
    /,
    *,
    kind: MatrixFunctionKind,
    policy: MatrixFunctionPolicy | None = None,
    spectral: SpectralMatrixRepresentation | None = None,
    spectral_bounds: tuple[float, float] | None = None,
    self_adjoint: bool = False,
    mass_weights: ArrayLike | None = None,
) -> Array:
    """Apply exp(step A) or φ₁(step A) without materializing A."""
    if not callable(operator):
        raise TypeError("operator must be callable.")
    if kind not in ("exp", "phi1"):
        raise ValueError("kind must be 'exp' or 'phi1'.")
    selected_policy = MatrixFunctionPolicy() if policy is None else policy
    if not isinstance(selected_policy, MatrixFunctionPolicy):
        raise TypeError("policy must be a MatrixFunctionPolicy.")
    value = jnp.asarray(vector)
    step_value = jnp.asarray(step, dtype=value.dtype).reshape(())
    method = selected_policy.method
    if method == "auto":
        if spectral is not None:
            method = "spectral"
        elif self_adjoint and spectral_bounds is not None:
            method = "chebyshev"
        elif self_adjoint:
            method = "lanczos"
        else:
            method = "arnoldi"
    if method == "spectral":
        if spectral is None:
            raise ValueError("The spectral method requires a spectral representation.")
        return spectral.apply(
            value,
            _scalar_function(step_value * spectral.eigenvalues, kind),
        )
    if method == "lanczos" and not self_adjoint:
        raise ValueError("Lanczos requires a declared mass-self-adjoint operator.")

    if self_adjoint and mass_weights is not None:
        weights = jnp.asarray(mass_weights, dtype=value.dtype)
        if value.ndim < weights.ndim or value.shape[: weights.ndim] != weights.shape:
            raise ValueError("mass_weights must match the leading spatial state shape.")
        root = jnp.sqrt(weights).reshape(
            weights.shape + (1,) * (value.ndim - weights.ndim)
        )
    else:
        root = jnp.ones_like(value)
    transformed = (root * value).reshape((-1,))

    def transformed_operator(flattened):
        physical = flattened.reshape(value.shape) / root
        image = jnp.asarray(operator(physical))
        if image.shape != value.shape:
            raise ValueError(
                f"operator must preserve vector shape {value.shape}; got {image.shape}."
            )
        return (root * image).reshape((-1,))

    if method == "chebyshev":
        if spectral_bounds is None:
            raise ValueError("The Chebyshev method requires spectral_bounds.")
        result = _chebyshev_action(
            transformed_operator,
            transformed,
            step_value,
            kind,
            spectral_bounds,
            num_matvecs=selected_policy.num_matvecs,
        )
    elif method == "lanczos":
        result = _lanczos_action(
            transformed_operator,
            transformed,
            step_value,
            kind,
            num_matvecs=selected_policy.num_matvecs,
            full_reorthogonalization=(selected_policy.reorthogonalization == "full"),
        )
    elif method == "arnoldi":
        result = _arnoldi_action(
            transformed_operator,
            transformed,
            step_value,
            kind,
            num_matvecs=selected_policy.num_matvecs,
            full_reorthogonalization=(selected_policy.reorthogonalization == "full"),
        )
    else:
        raise ValueError(f"Unsupported matrix-function method {method!r}.")
    return result.reshape(value.shape) / root


def matrix_exponential_action(
    operator: Callable[[Array], ArrayLike],
    vector: ArrayLike,
    step: ArrayLike,
    /,
    **kwargs,
) -> Array:
    return matrix_function_action(operator, vector, step, kind="exp", **kwargs)


def matrix_phi1_action(
    operator: Callable[[Array], ArrayLike],
    vector: ArrayLike,
    step: ArrayLike,
    /,
    **kwargs,
) -> Array:
    return matrix_function_action(operator, vector, step, kind="phi1", **kwargs)


__all__ = [
    "MatrixFunctionDifferentiation",
    "MatrixFunctionKind",
    "MatrixFunctionMethod",
    "MatrixFunctionPolicy",
    "SpectralMatrixRepresentation",
    "matrix_exponential_action",
    "matrix_function_action",
    "matrix_phi1_action",
]
