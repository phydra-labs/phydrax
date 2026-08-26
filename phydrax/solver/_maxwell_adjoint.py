#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


CheckpointMode: TypeAlias = Literal["full", "recompute"]


class PyTreeAdjointResult(StrictModule):
    final_state: Any
    loss: Array
    initial_gradient: Any
    parameter_gradient: Any


class PyTreeCheckpointedAdjointPlan(StrictModule):
    """Exact reverse-mode adjoint for a fixed-step PyTree time map."""

    step: Callable[[Array, Any, Array, Any, Any], Any] = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    mode: CheckpointMode = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        step: Callable[[Array, Any, Array, Any, Any], Any],
        steps: int,
        /,
        *,
        mode: CheckpointMode = "recompute",
    ):
        if not callable(step):
            raise TypeError("step must be callable.")
        count = int(steps)
        if count <= 0 or mode not in ("full", "recompute"):
            raise ValueError("Adjoint steps/mode are invalid.")
        self.step = step
        self.steps = count
        self.mode = mode
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pytree-checkpointed-adjoint",
                "step": repr(step),
                "steps": count,
                "mode": mode,
            }
        )

    def evolve(
        self,
        initial_state: Any,
        parameters: Any,
        time: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> Any:
        t0 = jnp.asarray(time)
        dt = jnp.asarray(step_size)
        if t0.shape != () or dt.shape != ():
            raise ValueError("Adjoint time and step_size must be scalar.")

        def body(state, index):
            next_state = self.step(t0 + index * dt, state, dt, parameters, args)
            return next_state, None

        selected = jax.checkpoint(body) if self.mode == "recompute" else body
        final, _ = jax.lax.scan(selected, initial_state, jnp.arange(self.steps))
        return final

    def value_and_gradient(
        self,
        initial_state: Any,
        parameters: Any,
        time: ArrayLike,
        step_size: ArrayLike,
        loss: Callable[[Any, Any], ArrayLike],
        args: Any = None,
        /,
    ) -> PyTreeAdjointResult:
        if not callable(loss):
            raise TypeError("loss must be callable.")

        def objective(initial, parameter_values):
            final = self.evolve(
                initial,
                parameter_values,
                time,
                step_size,
                args,
            )
            return jnp.asarray(loss(final, parameter_values)), final

        (loss_value, final_state), gradients = jax.value_and_grad(
            objective,
            argnums=(0, 1),
            has_aux=True,
        )(initial_state, parameters)
        return PyTreeAdjointResult(
            final_state,
            loss_value,
            gradients[0],
            gradients[1],
        )


class DirectionalDerivativeReport(StrictModule):
    automatic: Array
    finite_difference: Array
    absolute_residual: Array
    relative_residual: Array
    step_size: Array
    passed: Array


def audit_directional_derivative(
    function: Callable[[Any], ArrayLike],
    point: Any,
    direction: Any,
    /,
    *,
    step_size: float = 1e-5,
    tolerance: float = 1e-5,
) -> DirectionalDerivativeReport:
    if not callable(function):
        raise TypeError("function must be callable.")
    step = float(step_size)
    tolerance_ = float(tolerance)
    if (
        not np.isfinite(step)
        or step <= 0.0
        or not np.isfinite(tolerance_)
        or tolerance_ <= 0.0
    ):
        raise ValueError("Directional derivative step/tolerance must be positive.")
    value, tangent = jax.jvp(function, (point,), (direction,))
    del value
    plus = jax.tree.map(lambda x, d: x + step * d, point, direction)
    minus = jax.tree.map(lambda x, d: x - step * d, point, direction)
    finite = (jnp.asarray(function(plus)) - jnp.asarray(function(minus))) / (2.0 * step)
    automatic = jnp.asarray(tangent)
    absolute = jnp.abs(automatic - finite)
    scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(automatic), jnp.abs(finite)))
    relative = absolute / scale
    return DirectionalDerivativeReport(
        automatic,
        finite,
        absolute,
        relative,
        jnp.asarray(step),
        relative <= tolerance_,
    )


class MaxwellDFTAdjointResult(StrictModule):
    objective: Array
    gradient: Any
    forward_fields: Any
    adjoint_fields: Any


class MaxwellDFTAdjointPlan(StrictModule):
    """Two-run time/DFT adjoint orchestration with explicit source construction."""

    forward_run: Callable[[Any], tuple[ArrayLike, Any]] = eqx.field(static=True)
    adjoint_source: Callable[[Any, Array], Any] = eqx.field(static=True)
    adjoint_run: Callable[[Any], Any] = eqx.field(static=True)
    contraction: Callable[[Any, Any, Any], Any] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward_run: Callable[[Any], tuple[ArrayLike, Any]],
        adjoint_source: Callable[[Any, Array], Any],
        adjoint_run: Callable[[Any], Any],
        contraction: Callable[[Any, Any, Any], Any],
        /,
    ):
        if not all(
            callable(value)
            for value in (forward_run, adjoint_source, adjoint_run, contraction)
        ):
            raise TypeError("DFT adjoint components must be callable.")
        self.forward_run = forward_run
        self.adjoint_source = adjoint_source
        self.adjoint_run = adjoint_run
        self.contraction = contraction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-dft-adjoint",
                "forward": repr(forward_run),
                "source": repr(adjoint_source),
                "adjoint": repr(adjoint_run),
                "contraction": repr(contraction),
            }
        )

    def evaluate(self, design: Any, /) -> MaxwellDFTAdjointResult:
        objective, forward_fields = self.forward_run(design)
        objective_ = jnp.asarray(objective)
        if objective_.shape != () or jnp.iscomplexobj(objective_):
            raise ValueError("DFT adjoint objective must be a real scalar.")
        _, pullback = jax.vjp(
            lambda fields: jnp.asarray(self.forward_run(fields)[0]), design
        )
        design_cotangent = pullback(jnp.asarray(1.0))[0]
        source = self.adjoint_source(forward_fields, objective_)
        adjoint_fields = self.adjoint_run(source)
        gradient = self.contraction(forward_fields, adjoint_fields, design_cotangent)
        return MaxwellDFTAdjointResult(
            objective_, gradient, forward_fields, adjoint_fields
        )


def bloch_adjoint_wavevector(wavevector: ArrayLike, /) -> Array:
    value = jnp.asarray(wavevector)
    if value.ndim != 1 or bool(jnp.any(~jnp.isfinite(value))):
        raise ValueError("Bloch wavevector must be a finite vector.")
    return -value


__all__ = [
    "CheckpointMode",
    "DirectionalDerivativeReport",
    "MaxwellDFTAdjointPlan",
    "MaxwellDFTAdjointResult",
    "PyTreeAdjointResult",
    "PyTreeCheckpointedAdjointPlan",
    "audit_directional_derivative",
    "bloch_adjoint_wavevector",
]
