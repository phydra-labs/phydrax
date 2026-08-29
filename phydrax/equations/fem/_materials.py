#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._finite_element_material import ConstitutiveResponse


class LocalImplicitDiagnostics(StrictModule):
    converged: Array
    iterations: Array
    residual_norm: Array
    finite: Array


class FiniteElementAuxiliaryEvaluation(StrictModule):
    """Material/contact/history candidate state returned with one residual."""

    trial_state: object
    successful: Array
    admissible: Array
    retry_requested: Array
    suggested_step: Array
    diagnostics: object

    def __init__(
        self,
        trial_state: object = None,
        /,
        *,
        successful: ArrayLike = True,
        admissible: ArrayLike = True,
        retry_requested: ArrayLike = False,
        suggested_step: ArrayLike = 0.0,
        diagnostics: object = None,
    ):
        successful_ = jnp.asarray(successful, dtype=bool)
        admissible_ = jnp.asarray(admissible, dtype=bool)
        retry_ = jnp.asarray(retry_requested, dtype=bool)
        suggested = jnp.asarray(suggested_step)
        if any(
            value.shape != () for value in (successful_, admissible_, retry_, suggested)
        ):
            raise ValueError("Auxiliary decision values must be scalars.")
        self.trial_state = trial_state
        self.successful = successful_
        self.admissible = admissible_
        self.retry_requested = retry_
        self.suggested_step = suggested
        self.diagnostics = diagnostics

    @property
    def valid(self) -> Array:
        return self.successful & self.admissible


class LocalImplicitMaterial(StrictModule, NonTrainableState):
    """Bounded local constitutive root with implicit-function derivatives."""

    residual: Callable
    response: Callable
    state_shape: tuple[int, ...] = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual: Callable,
        response: Callable,
        /,
        *,
        state_shape: tuple[int, ...],
        max_steps: int = 25,
        tolerance: float = 1.0e-10,
        model_id: str,
    ):
        if not callable(residual) or not callable(response):
            raise TypeError("Local material residual and response must be callable.")
        shape = tuple(int(size) for size in state_shape)
        steps = int(max_steps)
        tolerance_ = float(tolerance)
        identifier = str(model_id)
        if (
            not shape
            or any(size <= 0 for size in shape)
            or steps <= 0
            or tolerance_ <= 0.0
            or not identifier
        ):
            raise ValueError("Local implicit material configuration is invalid.")
        self.residual = residual
        self.response = response
        self.state_shape = shape
        self.max_steps = steps
        self.tolerance = tolerance_
        self.model_id = canonical_fingerprint(
            {
                "kind": "local-implicit-material",
                "declared_id": identifier,
                "state_shape": list(shape),
                "max_steps": steps,
                "tolerance": tolerance_,
            }
        )

    def _newton_solve(self, function: Callable, initial: Array, args: object) -> Array:
        def body(_, value):
            residual = function(value, args)
            jacobian = jax.jacfwd(lambda candidate: function(candidate, args))(value)
            flat_residual = residual.reshape((-1,))
            flat_jacobian = jacobian.reshape((flat_residual.size, flat_residual.size))
            update = jnp.linalg.solve(flat_jacobian, -flat_residual).reshape(value.shape)
            candidate = value + update
            norm = jnp.linalg.norm(flat_residual)
            return jnp.where(norm <= self.tolerance, value, candidate)

        return jax.lax.fori_loop(0, self.max_steps, body, initial)

    def solve(self, initial_state: ArrayLike, args: object, /) -> Array:
        initial = jnp.asarray(initial_state)
        if initial.shape[-len(self.state_shape) :] != self.state_shape:
            raise ValueError("Initial local state has the wrong trailing shape.")
        function = self.residual

        def solve_fn(residual_fn, guess):
            return self._newton_solve(
                lambda state, parameters: residual_fn(state),
                guess,
                None,
            )

        def tangent_solve(linearize, right_hand_side):
            zero = jnp.zeros_like(right_hand_side)
            matrix = jax.jacfwd(linearize)(zero)
            flat_rhs = right_hand_side.reshape((-1,))
            flat_matrix = matrix.reshape((flat_rhs.size, flat_rhs.size))
            return jnp.linalg.solve(flat_matrix, flat_rhs).reshape(right_hand_side.shape)

        return jax.lax.custom_root(
            lambda state: function(state, args),
            initial,
            solve_fn,
            tangent_solve,
        )

    def solve_with_diagnostics(
        self,
        initial_state: ArrayLike,
        args: object,
        /,
    ) -> tuple[Array, LocalImplicitDiagnostics]:
        root = self.solve(initial_state, args)
        residual = jnp.asarray(self.residual(root, args))
        norm = jnp.linalg.norm(residual.reshape((-1,)))
        finite = jnp.all(jnp.isfinite(root)) & jnp.all(jnp.isfinite(residual))
        return root, LocalImplicitDiagnostics(
            converged=finite & (norm <= self.tolerance),
            iterations=jnp.asarray(self.max_steps, dtype=jnp.int32),
            residual_norm=norm,
            finite=finite,
        )

    def evaluate(
        self,
        initial_state: ArrayLike,
        args: object,
        /,
    ) -> ConstitutiveResponse:
        root, diagnostics = self.solve_with_diagnostics(initial_state, args)
        result = self.response(root, args)
        if not isinstance(result, ConstitutiveResponse):
            raise TypeError("Local material response must return ConstitutiveResponse.")
        return ConstitutiveResponse(
            result.response,
            result.trial_state,
            diagnostics={
                **result.diagnostics,
                "converged": diagnostics.converged,
                "iterations": diagnostics.iterations,
                "residual_norm": diagnostics.residual_norm,
                "finite": diagnostics.finite,
            },
        )


__all__ = [
    "FiniteElementAuxiliaryEvaluation",
    "LocalImplicitDiagnostics",
    "LocalImplicitMaterial",
]
