#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._numerics._ssp_runge_kutta import ssprk33_step
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import PreparedFiniteVolumeDynamics


SplittingKind: TypeAlias = Literal["godunov", "strang"]


class FiniteVolumeStepResult(StrictModule):
    state: Array
    time: Array
    step_size: Array
    temporal_method_id: str = eqx.field(static=True)


class UnsplitFiniteVolumeSSPRK3Plan(StrictModule, NonTrainableState):
    """Three-stage SSPRK update of the complete FV semidiscretization."""

    dynamics: PreparedFiniteVolumeDynamics
    temporal_method_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedFiniteVolumeDynamics, /):
        if not isinstance(dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError("SSPRK3 requires prepared finite-volume dynamics.")
        self.dynamics = dynamics
        self.temporal_method_id = "temporal:ssprk:3:3"
        self.plan_id = canonical_fingerprint(
            {"kind": "unsplit-fv-ssprk3", "dynamics": dynamics.dynamics_id}
        )

    def advance(
        self,
        time: ArrayLike,
        state: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> FiniteVolumeStepResult:
        time_ = jnp.asarray(time).reshape(())
        value = jnp.asarray(state)
        dt = jnp.asarray(step_size).reshape(())
        updated = ssprk33_step(self.dynamics, time_, value, dt, args)
        return FiniteVolumeStepResult(updated, time_ + dt, dt, self.temporal_method_id)


class DirectionalSplitFiniteVolumePlan(StrictModule, NonTrainableState):
    """Godunov or symmetric Strang composition of directional FV operators."""

    dynamics: PreparedFiniteVolumeDynamics
    splitting: SplittingKind = eqx.field(static=True)
    temporal_method_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedFiniteVolumeDynamics,
        /,
        *,
        splitting: SplittingKind = "strang",
    ):
        if not isinstance(dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError("Directional splitting requires finite-volume dynamics.")
        if splitting not in ("godunov", "strang"):
            raise ValueError("splitting must be 'godunov' or 'strang'.")
        self.dynamics = dynamics
        self.splitting = splitting
        self.temporal_method_id = f"temporal:split:{splitting}"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "directional-split-fv",
                "dynamics": dynamics.dynamics_id,
                "splitting": splitting,
            }
        )

    def _heun_axis(
        self,
        time: Array,
        state: Array,
        step_size: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        first_rate = self.dynamics.axis_residual(time, state, axis, args)
        predictor = state + step_size * first_rate
        second_rate = self.dynamics.axis_residual(time + step_size, predictor, axis, args)
        return state + 0.5 * step_size * (first_rate + second_rate)

    def _auxiliary_rhs(
        self,
        time: Array,
        state: Array,
        args: Any,
        /,
    ) -> Array:
        directional = sum(
            (
                self.dynamics.axis_residual(time, state, axis, args)
                for axis in range(len(self.dynamics.discretization.cell_shape))
            ),
            jnp.zeros_like(state),
        )
        return self.dynamics(time, state, args) - directional

    def advance(
        self,
        time: ArrayLike,
        state: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> FiniteVolumeStepResult:
        time_ = jnp.asarray(time).reshape(())
        value = jnp.asarray(state)
        dt = jnp.asarray(step_size).reshape(())
        dimension = len(self.dynamics.discretization.cell_shape)
        if self.splitting == "godunov":
            updated = value
            for axis in range(dimension):
                updated = self._heun_axis(time_, updated, dt, axis, args)
        elif dimension == 1:
            updated = self._heun_axis(time_, value, dt, 0, args)
        else:
            updated = value
            half = 0.5 * dt
            for axis in range(dimension):
                updated = self._heun_axis(time_, updated, half, axis, args)
            for axis in reversed(range(dimension)):
                updated = self._heun_axis(time_ + half, updated, half, axis, args)
        first_auxiliary = self._auxiliary_rhs(time_, updated, args)
        auxiliary_predictor = updated + dt * first_auxiliary
        second_auxiliary = self._auxiliary_rhs(time_ + dt, auxiliary_predictor, args)
        updated = updated + 0.5 * dt * (first_auxiliary + second_auxiliary)
        return FiniteVolumeStepResult(updated, time_ + dt, dt, self.temporal_method_id)


__all__ = [
    "DirectionalSplitFiniteVolumePlan",
    "FiniteVolumeStepResult",
    "SplittingKind",
    "UnsplitFiniteVolumeSSPRK3Plan",
]
