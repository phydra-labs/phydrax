#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..nn._keys import EvalKey
from ..nn.layers._recurrent import AbstractRecurrentOutputCell
from ._signature import (
    _chen_multiply_increment,
    _inexact_array,
    _signature_identity,
    _signature_shape,
)
from ._signature_features import flatten_signature


class SignatureRecurrentState(StrictModule):
    """Truncated signature carry for one packed recurrent stream."""

    signature: tuple[Array, ...]
    previous_point: Array
    has_previous: Array


class SignatureRecurrentCell(AbstractRecurrentOutputCell):
    """Exact online path-signature updates under canonical recurrent semantics.

    The first valid point establishes a basepoint and emits the identity
    signature. Each later valid point applies one Chen update. Reset handling,
    invalid padding, and streaming carries are delegated to ``run_recurrent``.
    """

    dimension: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    include_scalar: bool = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        depth: int,
        /,
        *,
        include_scalar: bool = False,
    ):
        resolved_dimension = int(dimension)
        resolved_depth = int(depth)
        if resolved_dimension <= 0:
            raise ValueError("dimension must be positive.")
        if resolved_depth <= 0:
            raise ValueError("depth must be positive.")
        self.dimension = resolved_dimension
        self.depth = resolved_depth
        self.include_scalar = bool(include_scalar)
        self.feature_id = (
            "SignatureRecurrentCell["
            f"dimension={resolved_dimension},depth={resolved_depth},"
            f"scalar={self.include_scalar}]"
        )

    @property
    def output_size(self) -> int:
        return int(self.include_scalar) + sum(
            self.dimension**degree for degree in range(1, self.depth + 1)
        )

    def initial_state(
        self, case_shape: tuple[int, ...], /, *, dtype: Any
    ) -> SignatureRecurrentState:
        state_dtype = jnp.result_type(dtype, jnp.asarray(0.0))
        return SignatureRecurrentState(
            signature=_signature_identity(
                case_shape,
                self.dimension,
                self.depth,
                state_dtype,
            ),
            previous_point=jnp.zeros(case_shape + (self.dimension,), dtype=state_dtype),
            has_previous=jnp.zeros(case_shape, dtype=bool),
        )

    def step(
        self,
        state: SignatureRecurrentState,
        inputs: ArrayLike,
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[SignatureRecurrentState, Array]:
        del key
        if not isinstance(state, SignatureRecurrentState):
            raise TypeError("state must be a SignatureRecurrentState.")
        point = _inexact_array(inputs)
        levels, dimension = _signature_shape(state.signature)
        case_shape = levels[0].shape[:-1]
        expected_point_shape = case_shape + (self.dimension,)
        if dimension != self.dimension or len(levels) != self.depth:
            raise ValueError("state signature does not match this cell.")
        if state.previous_point.shape != expected_point_shape:
            raise ValueError(
                f"state.previous_point must have shape {expected_point_shape}."
            )
        if state.has_previous.shape != case_shape:
            raise ValueError(f"state.has_previous must have shape {case_shape}.")
        if point.shape != expected_point_shape:
            raise ValueError(f"inputs must have shape {expected_point_shape}.")
        point = eqx.error_if(
            point,
            jnp.any(~jnp.isfinite(point)),
            "Signature recurrent points must be finite.",
        )

        increment = jnp.where(
            state.has_previous[..., None],
            point - state.previous_point,
            jnp.zeros_like(point),
        )
        next_state = SignatureRecurrentState(
            signature=_chen_multiply_increment(levels, increment),
            previous_point=point,
            has_previous=jnp.ones_like(state.has_previous),
        )
        return next_state, self.output_from_state(next_state)

    def output_from_state(self, state: SignatureRecurrentState, /) -> Array:
        if not isinstance(state, SignatureRecurrentState):
            raise TypeError("state must be a SignatureRecurrentState.")
        return flatten_signature(
            state.signature,
            include_scalar=self.include_scalar,
        )


__all__ = ["SignatureRecurrentCell", "SignatureRecurrentState"]
