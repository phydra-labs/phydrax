#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._model import AbstractArrayModel
from ._base import AbstractPositiveDefiniteKernel


class ExactQuantumStateFidelityKernel(AbstractPositiveDefiniteKernel):
    """Exact squared pure-state fidelity kernel over a pointwise state model."""

    state_model: AbstractArrayModel
    in_size: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    feature_map_id: str = eqx.field(static=True)
    normalization_tolerance: float = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_model: AbstractArrayModel,
        feature_map_id: str,
        /,
        *,
        normalization_tolerance: float = 1e-6,
    ):
        if not isinstance(state_model, AbstractArrayModel):
            raise TypeError("state_model must be an AbstractArrayModel.")
        if not isinstance(state_model.in_size, int) or not isinstance(
            state_model.out_size, int
        ):
            raise ValueError(
                "Fidelity state models require flat vector input and output."
            )
        identifier = str(feature_map_id)
        if not identifier:
            raise ValueError("feature_map_id must be non-empty.")
        tolerance = float(normalization_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("normalization_tolerance must be finite and nonnegative.")
        if state_model.input_binding().batch_mode != "pointwise":
            raise ValueError("Fidelity state models must use pointwise input binding.")
        self.state_model = state_model
        self.in_size = state_model.in_size
        self.state_size = state_model.out_size
        self.feature_map_id = identifier
        self.normalization_tolerance = tolerance
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "exact-quantum-state-fidelity-kernel",
                "feature_map": identifier,
                "input_size": state_model.in_size,
                "state_size": state_model.out_size,
                "normalization_tolerance": tolerance,
            }
        )

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    @property
    def max_derivative_order(self) -> int:
        return 0

    @property
    def is_unit_diagonal(self) -> bool:
        return True

    def _state(self, point: Any, /) -> Array:
        state = jnp.asarray(self.state_model(point))
        if state.shape != (self.state_size,):
            raise ValueError("state_model output shape changed during kernel evaluation.")
        if not jnp.issubdtype(state.dtype, jnp.complexfloating):
            raise TypeError("state_model must return complex floating coordinates.")
        finite = jnp.all(jnp.isfinite(state))
        norm_residual = jnp.abs(jnp.sum(jnp.abs(state) ** 2) - 1.0)
        return eqx.error_if(
            state,
            ~(finite & (norm_residual <= self.normalization_tolerance)),
            "Fidelity feature state must be finite and normalized.",
        )

    def pairwise(self, x: Any, y: Any, /) -> Array:
        left = self._state(x)
        right = self._state(y)
        overlap = ein.contract("i,i->", jnp.conj(left), right)
        return jnp.real(overlap * jnp.conj(overlap))

    def matrix(self, x: ArrayLike, y: ArrayLike, /) -> Array:
        left_points = jnp.asarray(x)
        right_points = jnp.asarray(y)
        if (
            left_points.ndim != 2
            or right_points.ndim != 2
            or left_points.shape[-1] != self.in_size
            or right_points.shape[-1] != self.in_size
        ):
            raise ValueError(
                "Kernel matrices require shapes (N, in_size) and (M, in_size)."
            )
        left_states = jax.vmap(self._state)(left_points)
        right_states = jax.vmap(self._state)(right_points)
        overlaps = ein.contract("id,jd->ij", jnp.conj(left_states), right_states)
        return jnp.real(overlaps * jnp.conj(overlaps))

    def diagonal(self, x: ArrayLike, /) -> Array:
        points = jnp.asarray(x)
        if points.ndim != 2 or points.shape[-1] != self.in_size:
            raise ValueError("Kernel diagonals require shape (N, in_size).")
        states = jax.vmap(self._state)(points)
        return jnp.ones((states.shape[0],), dtype=jnp.real(states).dtype)


__all__ = ["ExactQuantumStateFidelityKernel"]
