#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._layout import InputLayout, StateLayout
from .._model_system import DiscreteModelTransition
from .._system import DiscreteStepContext, DiscreteSystem, DiscreteTransitionResult


class DiscreteModelRolloutTransitionResult(StrictModule):
    """Candidate/accepted state and separate training/physical validity."""

    candidate_state: Array
    accepted_state: Array
    training_usable: Array
    physically_converged: Array
    status: Array
    residual: Array
    iterations: Array
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate_state: Array,
        accepted_state: Array,
        /,
        *,
        training_usable: Any,
        physically_converged: Any,
        status: Any,
        residual: Any,
        iterations: Any,
        transition_id: str,
    ):
        self.candidate_state = jnp.asarray(candidate_state)
        self.accepted_state = jnp.asarray(accepted_state)
        self.training_usable = jnp.asarray(training_usable, dtype=bool)
        self.physically_converged = jnp.asarray(physically_converged, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.residual = jnp.asarray(residual)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.transition_id = str(transition_id)


class AbstractDiscreteModelRolloutTransition(StrictModule, NonTrainableState):
    """Interpret a learned model output inside one accepted discrete transition."""

    transition_id: str = eqx.field(static=True)
    state_layout: StateLayout
    input_layout: InputLayout | None
    step_size: float = eqx.field(static=True)
    step_rtol: float = eqx.field(static=True)
    step_atol: float = eqx.field(static=True)

    @abstractmethod
    def validate_model(self, model: AbstractArrayModel, /) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        model: AbstractArrayModel,
        context: DiscreteStepContext,
        state: Array,
        inputs: Array | None,
        /,
        *,
        key: Array | None,
        iteration: Array | None,
        control: Any = None,
    ) -> DiscreteModelRolloutTransitionResult:
        raise NotImplementedError

    @property
    def supports_linear_refinement(self) -> bool:
        return False

    def bind(self, model: AbstractArrayModel, /, *, system_id: str) -> DiscreteSystem:
        self.validate_model(model)
        return DiscreteSystem(
            _BoundDiscreteModelRolloutTransition(model, self),
            state_layout=self.state_layout,
            input_layout=self.input_layout,
            system_id=system_id,
            step_size=self.step_size,
            step_rtol=self.step_rtol,
            step_atol=self.step_atol,
        )


class DirectDiscreteModelRolloutTransition(AbstractDiscreteModelRolloutTransition):
    """Current direct next-state model semantics as an explicit rollout route."""

    def __init__(
        self,
        state_layout: StateLayout,
        /,
        *,
        input_layout: InputLayout | None = None,
        step_size: float,
        step_rtol: float = 1e-7,
        step_atol: float = 1e-12,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        step = float(step_size)
        rtol = float(step_rtol)
        atol = float(step_atol)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if not np.isfinite(rtol) or rtol < 0.0 or not np.isfinite(atol) or atol < 0.0:
            raise ValueError("Step tolerances must be finite and nonnegative.")
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.step_size = step
        self.step_rtol = rtol
        self.step_atol = atol
        self.transition_id = canonical_fingerprint(
            {
                "kind": "direct-discrete-model-rollout-transition",
                "state_layout": state_layout.layout_id,
                "input_layout": None if input_layout is None else input_layout.layout_id,
                "step_size": step,
                "step_rtol": rtol,
                "step_atol": atol,
            }
        )

    def validate_model(self, model: AbstractArrayModel, /) -> None:
        DiscreteModelTransition(
            model,
            state_layout=self.state_layout,
            input_layout=self.input_layout,
            step_size=self.step_size,
            step_rtol=self.step_rtol,
            step_atol=self.step_atol,
        )

    def evaluate(
        self,
        model: AbstractArrayModel,
        context: DiscreteStepContext,
        state: Array,
        inputs: Array | None,
        /,
        *,
        key: Array | None,
        iteration: Array | None,
        control: Any = None,
    ) -> DiscreteModelRolloutTransitionResult:
        del context
        if control is not None:
            raise ValueError("Direct model transitions do not accept numerical control.")
        binding = model.input_binding()
        point = (
            binding.pack_point((state,))
            if inputs is None
            else binding.pack_point((state, inputs))
        )
        candidate = jnp.asarray(
            binding.call(model, point, key=key, iter_=iteration, kwargs={}),
            dtype=state.dtype,
        )
        if candidate.shape != self.state_layout.shape:
            raise ValueError("Direct model output does not match the state layout shape.")
        finite = jnp.all(jnp.isfinite(candidate))
        member = self.state_layout.geometry.contains(candidate)
        valid = finite & member
        accepted = jnp.where(valid, candidate, jnp.zeros_like(candidate))
        return DiscreteModelRolloutTransitionResult(
            candidate,
            accepted,
            training_usable=valid,
            physically_converged=valid,
            status=jnp.where(valid, 0, 1),
            residual=jnp.asarray(0.0, dtype=jnp.real(candidate).dtype),
            iterations=jnp.asarray(1, dtype=jnp.int32),
            transition_id=self.transition_id,
        )


class _BoundDiscreteModelRolloutTransition(StrictModule):
    model: AbstractArrayModel
    transition: AbstractDiscreteModelRolloutTransition

    def __init__(
        self,
        model: AbstractArrayModel,
        transition: AbstractDiscreteModelRolloutTransition,
        /,
    ):
        self.model = model
        self.transition = transition

    def __call__(
        self,
        context: DiscreteStepContext,
        state: Array,
        *arguments: Any,
    ) -> DiscreteTransitionResult:
        if self.transition.input_layout is None:
            if len(arguments) != 1:
                raise TypeError("Autonomous learned transitions require args.")
            inputs = None
        else:
            if len(arguments) != 2:
                raise TypeError("Controlled learned transitions require inputs and args.")
            inputs = jnp.asarray(arguments[0])
        result = self.transition.evaluate(
            self.model,
            context,
            jnp.asarray(state),
            inputs,
            key=None,
            iteration=context.step_index,
        )
        physically_accepted = jnp.where(
            result.physically_converged,
            result.candidate_state,
            jnp.asarray(state),
        )
        return DiscreteTransitionResult(
            result.candidate_state,
            physically_accepted,
            result.physically_converged,
            result.status,
        )


__all__ = [
    "AbstractDiscreteModelRolloutTransition",
    "DirectDiscreteModelRolloutTransition",
    "DiscreteModelRolloutTransitionResult",
]
