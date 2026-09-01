#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim import (
    AbstractMinimizationMethod,
    MinimizationResult,
    minimize,
    OptimizationTermination,
)


class VortexControlRollout(StrictModule):
    states: Array
    outputs: Array
    event_signature: Array
    finite: Array
    rollout_id: str = eqx.field(static=True)


class VortexControlResult(StrictModule):
    controls: Array
    rollout: VortexControlRollout
    minimization_result: MinimizationResult
    event_program_valid: Array
    successful: Array
    control_id: str = eqx.field(static=True)


class VortexTrajectoryControlPlan(StrictModule, NonTrainableState):
    transition: Callable[[Array, Array, Array, Any], tuple[Array, Array, Array]]
    running_cost: Callable[[Array, Array, Array, Any], Array]
    terminal_cost: Callable[[Array, Any], Array]
    method: AbstractMinimizationMethod
    termination: OptimizationTermination
    mode: str = eqx.field(static=True)
    continuity_weight: float = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition,
        running_cost,
        terminal_cost,
        method: AbstractMinimizationMethod,
        /,
        *,
        mode: str = "direct-shooting",
        continuity_weight: float = 1.0e3,
        termination: OptimizationTermination | None = None,
        control_id: str,
    ):
        if (
            not callable(transition)
            or not callable(running_cost)
            or not callable(terminal_cost)
            or not isinstance(method, AbstractMinimizationMethod)
            or mode not in ("direct-shooting", "multiple-shooting", "direct-collocation")
            or continuity_weight <= 0.0
            or not str(control_id)
        ):
            raise ValueError("Vortex control plan inputs are invalid.")
        self.transition, self.running_cost, self.terminal_cost, self.method = (
            transition,
            running_cost,
            terminal_cost,
            method,
        )
        self.termination = (
            OptimizationTermination() if termination is None else termination
        )
        self.mode, self.continuity_weight, self.control_id = (
            mode,
            float(continuity_weight),
            str(control_id),
        )

    def _rollout(
        self, initial_state: Array, controls: Array, time: Array, args: Any, /
    ) -> VortexControlRollout:
        states, outputs, events = [initial_state], [], []
        state = initial_state
        for step in range(int(controls.shape[0])):
            state, output, event = self.transition(
                time[step], state, controls[step], args
            )
            states.append(state)
            outputs.append(output)
            events.append(event)
        states_array = jnp.stack(tuple(states))
        outputs_array = jnp.stack(tuple(outputs))
        event_array = jnp.stack(tuple(events))
        finite = jnp.all(jnp.isfinite(states_array)) & jnp.all(
            jnp.isfinite(outputs_array)
        )
        return VortexControlRollout(
            states_array,
            outputs_array,
            event_array,
            finite,
            canonical_fingerprint(
                {
                    "kind": "vortex-control-rollout",
                    "control": self.control_id,
                    "step_count": int(controls.shape[0]),
                }
            ),
        )

    def solve(
        self,
        initial_state: ArrayLike,
        initial_controls: ArrayLike,
        time: ArrayLike,
        args: Any = None,
        /,
        *,
        fixed_event_signature: ArrayLike | None = None,
    ) -> VortexControlResult:
        state0, controls0, time_ = (
            jnp.asarray(initial_state),
            jnp.asarray(initial_controls),
            jnp.asarray(time),
        )
        if controls0.ndim < 2 or time_.shape != (controls0.shape[0] + 1,):
            raise ValueError("Control horizon/time shapes are incompatible.")
        expected_event = (
            None if fixed_event_signature is None else jnp.asarray(fixed_event_signature)
        )

        def objective(controls, objective_args):
            del objective_args
            rollout = self._rollout(state0, controls, time_, args)
            value = jnp.asarray(0.0, dtype=state0.dtype)
            for step in range(int(controls.shape[0])):
                value = value + self.running_cost(
                    time_[step], rollout.states[step], controls[step], args
                )
            value = value + self.terminal_cost(rollout.states[-1], args)
            if expected_event is not None:
                value = value + self.continuity_weight * jnp.sum(
                    (rollout.event_signature - expected_event) ** 2
                )
            return value

        minimization = minimize(
            objective, controls0, method=self.method, termination=self.termination
        )
        controls = jnp.asarray(minimization.parameters)
        rollout = self._rollout(state0, controls, time_, args)
        event_valid = (
            jnp.asarray(True)
            if expected_event is None
            else jnp.all(rollout.event_signature == expected_event)
        )
        successful = minimization.successful & rollout.finite & event_valid
        return VortexControlResult(
            controls, rollout, minimization, event_valid, successful, self.control_id
        )


class VortexMPCPlan(StrictModule, NonTrainableState):
    trajectory: VortexTrajectoryControlPlan
    apply_steps: int = eqx.field(static=True)
    mpc_id: str = eqx.field(static=True)

    def __init__(self, trajectory: VortexTrajectoryControlPlan, apply_steps: int = 1, /):
        if (
            not isinstance(trajectory, VortexTrajectoryControlPlan)
            or int(apply_steps) <= 0
        ):
            raise ValueError("MPC requires trajectory plan and positive apply_steps.")
        self.trajectory, self.apply_steps = trajectory, int(apply_steps)
        self.mpc_id = canonical_fingerprint(
            {
                "kind": "vortex-mpc-plan",
                "trajectory": trajectory.control_id,
                "apply_steps": self.apply_steps,
            }
        )

    def solve_window(
        self, state: ArrayLike, controls: ArrayLike, time: ArrayLike, args: Any = None, /
    ):
        result = self.trajectory.solve(state, controls, time, args)
        return result.controls[: self.apply_steps], result


__all__ = [
    "VortexControlResult",
    "VortexControlRollout",
    "VortexMPCPlan",
    "VortexTrajectoryControlPlan",
]
