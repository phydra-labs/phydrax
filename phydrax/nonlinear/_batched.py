#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve as solve_linear,
)
from ._precision import NonlinearPrecisionPolicy
from ._types import NonlinearStatus


def _axis_norm(
    value: Array,
    axis: int,
    precision: NonlinearPrecisionPolicy,
    /,
) -> Array:
    accumulated = precision.accumulation(value)
    return precision.decision(
        jnp.sqrt(
            jnp.sum(
                jnp.real(jnp.conj(accumulated) * accumulated),
                axis=axis,
            )
        )
    )


class BatchedRootResult(StrictModule):
    state: Array
    residual: Array
    status: Array
    iterations: Array
    residual_evaluations: Array
    jacobian_evaluations: Array
    accepted_steps: Array
    residual_norm: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)

    @property
    def successful(self):
        return self.status == int(NonlinearStatus.SUCCESS)


class SmallRootKernel(StrictModule):
    """Fused masked dense Newton kernel for batches of small systems."""

    residual: Callable[[Array, Any], Array]
    maximum_dimension: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    minimum_damping: float = eqx.field(static=True)
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        residual: Callable[[Array, Any], Array],
        /,
        *,
        maximum_dimension: int = 16,
        maximum_steps: int = 16,
        absolute_tolerance: float = 1e-8,
        relative_tolerance: float = 1e-8,
        minimum_damping: float = 1e-4,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        dimension = int(maximum_dimension)
        steps = int(maximum_steps)
        if dimension < 1 or steps < 1:
            raise ValueError("Small-root dimensions and steps must be positive.")
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        precision_.validate_tolerance(absolute_tolerance)
        self.residual = residual
        self.maximum_dimension = dimension
        self.maximum_steps = steps
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.minimum_damping = float(minimum_damping)
        self.precision = precision_

    def solve(self, initial_states: Any, args: Any, /) -> BatchedRootResult:
        states = self.precision.state(jnp.asarray(initial_states))
        if states.ndim != 2:
            raise ValueError("initial_states must have shape (batch, dimension).")
        if states.shape[1] > self.maximum_dimension:
            raise ValueError("Small-root dimension exceeds maximum_dimension.")
        residual_function = self.residual
        evaluate = jax.vmap(residual_function)
        jacobian = jax.vmap(jax.jacfwd(residual_function))
        residuals = self.precision.residual(evaluate(states, args))
        if residuals.shape != states.shape:
            raise ValueError("Small-root residual shape must match state shape.")
        initial_norms = _axis_norm(residuals, 1, self.precision)
        thresholds = self.precision.decision(
            self.absolute_tolerance + self.relative_tolerance * initial_norms
        )
        active = jnp.isfinite(initial_norms) & (initial_norms > thresholds)
        iterations = jnp.zeros((states.shape[0],), dtype=jnp.int32)
        evaluations = jnp.ones_like(iterations)
        jacobian_evaluations = jnp.zeros_like(iterations)
        accepted_steps = jnp.zeros_like(iterations)

        class _Run(StrictModule):
            states: Array
            residuals: Array
            active: Array
            iterations: Array
            evaluations: Array
            jacobian_evaluations: Array
            accepted_steps: Array

        run = _Run(
            states,
            residuals,
            active,
            iterations,
            evaluations,
            jacobian_evaluations,
            accepted_steps,
        )

        def body(_, current):
            matrices = jacobian(current.states, args)
            regularized = (
                matrices
                + 1e-12 * jnp.eye(states.shape[1], dtype=states.dtype)[None, :, :]
            )
            directions = self.precision.direction(
                solve_linear(
                    LinearSystem(DenseLinearOperator(regularized)),
                    -current.residuals,
                    policy=self.precision.bind_linear(LinearSolvePolicy(DenseLU())),
                ).value
            )
            rates = jnp.asarray(
                [1.0, 0.5, 0.25, 0.125, 0.0625],
                dtype=states.dtype,
            )
            trial_states = jnp.asarray(
                current.states[None, :, :]
                + rates[:, None, None] * directions[None, :, :],
                dtype=states.dtype,
            )
            trial_residuals = self.precision.residual(
                jax.vmap(lambda candidates: evaluate(candidates, args))(trial_states)
            )
            trial_norms = _axis_norm(trial_residuals, 2, self.precision)
            trial_norms = jnp.where(
                jnp.isfinite(trial_norms),
                trial_norms,
                jnp.inf,
            )
            current_norms = _axis_norm(current.residuals, 1, self.precision)
            best_index = jnp.argmin(trial_norms, axis=0)
            selected_states = jnp.take_along_axis(
                trial_states,
                best_index[None, :, None],
                axis=0,
            )[0]
            selected_residuals = jnp.take_along_axis(
                trial_residuals,
                best_index[None, :, None],
                axis=0,
            )[0]
            selected_norms = jnp.take_along_axis(
                trial_norms,
                best_index[None, :],
                axis=0,
            )[0]
            accepted = current.active & (selected_norms < current_norms)
            next_states = jnp.where(
                current.active[:, None], selected_states, current.states
            )
            next_residuals = jnp.where(
                current.active[:, None], selected_residuals, current.residuals
            )
            norms = _axis_norm(next_residuals, 1, self.precision)
            next_active = current.active & accepted & (norms > thresholds)
            return _Run(
                next_states,
                next_residuals,
                next_active,
                current.iterations + current.active.astype(jnp.int32),
                current.evaluations + 5 * current.active.astype(jnp.int32),
                current.jacobian_evaluations + current.active.astype(jnp.int32),
                current.accepted_steps + accepted.astype(jnp.int32),
            )

        run = jax.lax.fori_loop(0, self.maximum_steps, body, run)
        norms = _axis_norm(run.residuals, 1, self.precision)
        finite = jnp.all(jnp.isfinite(run.states), axis=1) & jnp.all(
            jnp.isfinite(run.residuals), axis=1
        )
        status = jnp.where(
            finite & (norms <= thresholds),
            int(NonlinearStatus.SUCCESS),
            jnp.where(
                ~finite,
                int(NonlinearStatus.NONFINITE_EVALUATION),
                jnp.where(
                    run.active,
                    int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
                    int(NonlinearStatus.RESIDUAL_STAGNATION),
                ),
            ),
        ).astype(jnp.int32)
        output_states = self.precision.output(run.states)
        return BatchedRootResult(
            state=output_states,
            residual=run.residuals,
            status=status,
            iterations=run.iterations,
            residual_evaluations=run.evaluations,
            jacobian_evaluations=run.jacobian_evaluations,
            accepted_steps=run.accepted_steps,
            residual_norm=norms,
            precision_evidence=self.precision.evidence_for(
                run.states,
                run.residuals,
                output_value=output_states,
            ),
            precision_policy_id=self.precision.policy_id,
        )


def batched_small_root(
    residual: Callable[[Array, Any], Array],
    initial_states: Any,
    args: Any,
    /,
    **kernel_options: Any,
) -> BatchedRootResult:
    return SmallRootKernel(residual, **kernel_options).solve(initial_states, args)


__all__ = ["BatchedRootResult", "SmallRootKernel", "batched_small_root"]
