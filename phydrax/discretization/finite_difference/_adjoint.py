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
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


FDCheckpointingMode: TypeAlias = Literal["full", "recompute"]


class FDAdjointIdentityReport(StrictModule, NonTrainableState):
    residual: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual: float,
        tolerance: float,
        subject_id: str,
        /,
    ):
        residual_ = float(residual)
        tolerance_ = float(tolerance)
        if not np.isfinite(residual_) or not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("FD adjoint residual/tolerance must be finite and positive.")
        self.residual = residual_
        self.tolerance = tolerance_
        self.passed = residual_ <= tolerance_
        self.report_id = canonical_fingerprint(
            {
                "kind": "fd-adjoint-identity-report",
                "subject": subject_id,
                "residual": residual_,
                "tolerance": tolerance_,
            }
        )


class FDActionAdjointPlan(StrictModule):
    """Exact JAX VJP of one fixed-topology FD, boundary, halo, or transfer action."""

    action: Callable[..., Array]
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: Callable[..., Array],
        /,
        *,
        action_id: str | None = None,
    ):
        if not callable(action):
            raise TypeError("FD adjoint action must be callable.")
        identifier = (
            canonical_fingerprint({"kind": "fd-action-adjoint", "action": repr(action)})
            if action_id is None
            else str(action_id)
        )
        if not identifier:
            raise ValueError("action_id must be non-empty.")
        self.action = action
        self.action_id = identifier

    def transpose(
        self,
        primals: tuple[Any, ...],
        cotangent: ArrayLike,
        /,
    ) -> tuple[Any, ...]:
        output, pullback = jax.vjp(self.action, *primals)
        cotangent_ = jnp.asarray(cotangent)
        if cotangent_.shape != output.shape:
            raise ValueError("FD adjoint cotangent must match action output shape.")
        return pullback(cotangent_)

    def identity_report(
        self,
        primals: tuple[Any, ...],
        tangent_index: int,
        tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> FDAdjointIdentityReport:
        index = int(tangent_index)
        if index < 0 or index >= len(primals):
            raise ValueError("tangent_index is outside the action primals.")
        tangent_ = jnp.asarray(tangent)
        tangents = tuple(
            tangent_ if position == index else jax.tree.map(jnp.zeros_like, primal)
            for position, primal in enumerate(primals)
        )
        output, forward = jax.jvp(self.action, primals, tangents)
        cotangent_ = jnp.asarray(cotangent)
        if cotangent_.shape != output.shape:
            raise ValueError("FD adjoint test cotangent must match action output shape.")
        pullback = self.transpose(primals, cotangent_)[index]
        left = jnp.vdot(cotangent_, forward)
        right = jnp.vdot(tangent_, pullback)
        scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(left), jnp.abs(right)))
        residual = float(np.asarray(jnp.abs(left - right) / scale))
        return FDAdjointIdentityReport(
            residual,
            tolerance,
            self.action_id,
        )


class FDTimeAdjointResult(StrictModule):
    final_state: Array
    loss: Array
    initial_gradient: Array
    parameter_gradient: PyTree[Array]


class CheckpointedFDAdjointPlan(StrictModule):
    """Exact time-discrete scan adjoint with full or recomputed stage state."""

    step: Callable[[Array, Array, Array, Any], Array]
    steps: int = eqx.field(static=True)
    checkpointing: FDCheckpointingMode = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        step: Callable[[Array, Array, Array, Any], Array],
        steps: int,
        /,
        *,
        checkpointing: FDCheckpointingMode = "recompute",
    ):
        count = int(steps)
        if (
            not callable(step)
            or count <= 0
            or checkpointing
            not in (
                "full",
                "recompute",
            )
        ):
            raise ValueError("FD adjoint step/count/checkpointing is invalid.")
        self.step = step
        self.steps = count
        self.checkpointing = checkpointing
        self.plan_id = canonical_fingerprint(
            {
                "kind": "checkpointed-fd-adjoint",
                "step": repr(step),
                "steps": count,
                "checkpointing": checkpointing,
            }
        )

    def evolve(
        self,
        initial_state: Array,
        parameters: Any,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> Array:
        initial = jnp.asarray(initial_state)
        time_ = jnp.asarray(time)
        dt = jnp.asarray(step_size)
        indices = jnp.arange(self.steps)

        def body(state: Array, index: Array):
            next_state = self.step(time_ + index * dt, state, dt, parameters)
            return next_state, None

        selected_body = (
            jax.checkpoint(body) if self.checkpointing == "recompute" else body
        )
        final, _ = jax.lax.scan(selected_body, initial, indices)
        return final

    def value_and_gradient(
        self,
        initial_state: Array,
        parameters: Any,
        time: ArrayLike,
        step_size: ArrayLike,
        loss: Callable[[Array, Any], Array],
        /,
    ) -> FDTimeAdjointResult:
        if not callable(loss):
            raise TypeError("FD adjoint loss must be callable.")

        def objective(initial, parameter_values):
            final = self.evolve(
                initial,
                parameter_values,
                time,
                step_size,
            )
            return jnp.asarray(loss(final, parameter_values)), final

        (loss_value, final_state), gradients = jax.value_and_grad(
            objective,
            argnums=(0, 1),
            has_aux=True,
        )(initial_state, parameters)
        return FDTimeAdjointResult(
            final_state=final_state,
            loss=loss_value,
            initial_gradient=gradients[0],
            parameter_gradient=gradients[1],
        )


__all__ = [
    "CheckpointedFDAdjointPlan",
    "FDActionAdjointPlan",
    "FDAdjointIdentityReport",
    "FDCheckpointingMode",
    "FDTimeAdjointResult",
]
