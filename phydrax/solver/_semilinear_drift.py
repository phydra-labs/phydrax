#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._matrix_functions import SpectralMatrixRepresentation


class SemilinearDrift(StrictModule):
    """Time-independent linear operator plus explicit nonlinear drift.

    The contract preserves ``A u + F(t, u, args)`` after spatial discretization.
    Optional spectral data are exact finite-dimensional data, never a request to
    materialize a global operator matrix.
    """

    linear_operator: Callable[[Array], ArrayLike]
    nonlinear_drift: Callable[[Array, Array, Any], ArrayLike] | None
    mass_weights: Array | None
    spectral_representation: SpectralMatrixRepresentation | None
    compatible_noise_eigenvalues: Array | None
    state_shape: tuple[int, ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    mass_self_adjoint: bool = eqx.field(static=True)
    spectral_bounds: tuple[float, float] | None = eqx.field(static=True)
    compatible_noise_basis_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        linear_operator: Callable[[Array], ArrayLike],
        nonlinear_drift: Callable[[Array, Array, Any], ArrayLike] | None,
        /,
        *,
        state_shape: Sequence[int],
        operator_id: str,
        mass_self_adjoint: bool = False,
        mass_weights: ArrayLike | None = None,
        spectral_bounds: tuple[float, float] | None = None,
        spectral_representation: SpectralMatrixRepresentation | None = None,
        compatible_noise_eigenvalues: ArrayLike | None = None,
        compatible_noise_basis_id: str | None = None,
    ):
        if not callable(linear_operator):
            raise TypeError("linear_operator must be callable.")
        if nonlinear_drift is not None and not callable(nonlinear_drift):
            raise TypeError("nonlinear_drift must be callable or None.")
        shape = tuple(int(size) for size in state_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("state_shape must contain positive dimensions.")
        identifier = str(operator_id)
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        if mass_weights is None:
            weights = None
        else:
            weights = jnp.asarray(mass_weights, dtype=float)
            if value_shape := tuple(int(size) for size in weights.shape):
                if shape[: len(value_shape)] != value_shape:
                    raise ValueError(
                        "mass_weights shape must be a leading prefix of state_shape."
                    )
            else:
                raise ValueError("mass_weights must have at least one dimension.")
            if bool(jnp.any(~jnp.isfinite(weights))) or bool(jnp.any(weights <= 0.0)):
                raise ValueError("mass_weights must be finite and positive.")
        bounds = None
        if spectral_bounds is not None:
            lower, upper = (float(value) for value in spectral_bounds)
            if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
                raise ValueError("spectral_bounds must be finite and increasing.")
            bounds = (lower, upper)
        if spectral_representation is not None:
            if not isinstance(spectral_representation, SpectralMatrixRepresentation):
                raise TypeError(
                    "spectral_representation must be SpectralMatrixRepresentation."
                )
            if spectral_representation.state_shape != shape:
                raise ValueError("Spectral representation state_shape must match.")
        if compatible_noise_eigenvalues is None:
            noise_values = None
            if compatible_noise_basis_id is not None:
                raise ValueError(
                    "compatible_noise_basis_id requires compatible noise eigenvalues."
                )
        else:
            noise_values = jnp.asarray(compatible_noise_eigenvalues, dtype=float).reshape(
                (-1,)
            )
            if int(noise_values.size) <= 0 or bool(jnp.any(~jnp.isfinite(noise_values))):
                raise ValueError(
                    "compatible noise eigenvalues must be a non-empty finite vector."
                )
            if (
                not isinstance(compatible_noise_basis_id, str)
                or not compatible_noise_basis_id
            ):
                raise ValueError(
                    "compatible noise eigenvalues require a non-empty basis ID."
                )
        self.linear_operator = linear_operator
        self.nonlinear_drift = nonlinear_drift
        self.mass_weights = weights
        self.spectral_representation = spectral_representation
        self.compatible_noise_eigenvalues = noise_values
        self.state_shape = shape
        self.operator_id = identifier
        self.mass_self_adjoint = bool(mass_self_adjoint)
        self.spectral_bounds = bounds
        self.compatible_noise_basis_id = compatible_noise_basis_id

    def linear(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"Semilinear state must have shape {self.state_shape}; got {value.shape}."
            )
        result = jnp.asarray(self.linear_operator(value))
        if result.shape != self.state_shape:
            raise ValueError(
                "linear_operator must preserve the declared state shape; "
                f"got {result.shape}."
            )
        return result

    def nonlinear(self, time: Array, state: ArrayLike, args: Any, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"Semilinear state must have shape {self.state_shape}; got {value.shape}."
            )
        if self.nonlinear_drift is None:
            return jnp.zeros_like(value)
        result = jnp.asarray(self.nonlinear_drift(time, value, args))
        if result.shape != self.state_shape:
            raise ValueError(
                "nonlinear_drift must preserve the declared state shape; "
                f"got {result.shape}."
            )
        return result

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return self.linear(state) + self.nonlinear(time, state, args)


__all__ = ["SemilinearDrift"]
