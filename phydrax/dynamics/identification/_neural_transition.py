#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._admissibility import (
    AdmissibilityHeader,
    AdmissibilityReason,
    guard_derivative_validity,
    reason_bits_where,
)
from ..._differentiation import (
    branch_policy_contract,
    BranchDifferentiationPolicy,
    ComponentAuthority,
    DerivativeContract,
    DerivativeSurface,
)
from ..._fingerprint import canonical_fingerprint
from ..._model import (
    AbstractArrayModel,
    AbstractComponentSlot,
    ComponentBinding,
    ModelPorts,
    PortMapping,
)
from ..._model._component import bind_positional_component
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._layout import InputLayout, StateLayout
from .._model_system import DiscreteModelTransition
from .._system import DiscreteStepContext, DiscreteSystem, DiscreteTransitionResult


# The direct model output is accepted by a state-membership decision.
_DIRECT_DERIVATIVE_CONTRACT = branch_policy_contract(
    BranchDifferentiationPolicy.FROZEN_DECISION,
    surfaces=(DerivativeSurface.PRIMAL_STATE, DerivativeSurface.MODEL_PARAMETER),
)


class DiscreteModelRolloutTransitionResult(StrictModule):
    """Candidate/accepted state and separate training/physical validity.

    Beside the domain status, `header` is the transition's common
    admissibility evidence (eligible exactly when the step is physically
    converged; `model_id` is the learned component's identity, `evidence_id`
    the transition binding), `derivative_contract` the canonical derivative
    declaration, and `derivative_valid` whether derivatives of the candidate and
    accepted states are valid. Invalid derivatives are NaN; primals are kept.
    """

    candidate_state: Array
    accepted_state: Array
    training_usable: Array
    physically_converged: Array
    status: Array
    residual: Array
    iterations: Array
    header: AdmissibilityHeader
    derivative_contract: DerivativeContract
    derivative_valid: Array
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
        header: AdmissibilityHeader,
        derivative_contract: DerivativeContract,
        derivative_valid: Any,
        transition_id: str,
    ):
        if not isinstance(header, AdmissibilityHeader):
            raise TypeError("header must be an AdmissibilityHeader.")
        if not isinstance(derivative_contract, DerivativeContract):
            raise TypeError("derivative_contract must be a DerivativeContract.")
        self.candidate_state = jnp.asarray(candidate_state)
        self.accepted_state = jnp.asarray(accepted_state)
        self.training_usable = jnp.asarray(training_usable, dtype=jnp.bool_)
        self.physically_converged = jnp.asarray(physically_converged, dtype=jnp.bool_)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.residual = jnp.asarray(residual)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.header = header
        self.derivative_contract = derivative_contract
        self.derivative_valid = jnp.asarray(derivative_valid, dtype=jnp.bool_)
        self.transition_id = str(transition_id)


class AbstractDiscreteModelRolloutTransition(AbstractComponentSlot):
    """Interpret a learned model output inside one accepted discrete transition.

    The model bound to this slot has `MODEL` authority; the transition owns the
    physical interpretation and acceptance of its output. Built-in transitions
    are fixed analytic interpretations and hold no learned component: the model
    is supplied at evaluation.

    `owner_ports()` are the ports of the values the transition passes to and
    reads from the model, in the order it packs them, derived from its layouts.
    A model declaring ports binds to them only through the transition's explicit
    `port_mapping`, in exactly that order (`component_binding`); a model without
    ports takes no mapping. `validate_model` and every evaluation bind through
    `component_binding`.
    """

    component_authority: ClassVar[ComponentAuthority] = ComponentAuthority.MODEL
    slot_semantic_id: ClassVar[str] = "dynamics.discrete-model-rollout-transition"

    transition_id: str = eqx.field(static=True)
    state_layout: StateLayout
    input_layout: InputLayout | None
    port_mapping: eqx.AbstractVar[PortMapping | None]
    step_size: float = eqx.field(static=True)
    step_rtol: float = eqx.field(static=True)
    step_atol: float = eqx.field(static=True)

    @abstractmethod
    def owner_ports(self) -> ModelPorts:
        """Return the ordered ports of the model values this transition owns."""
        raise NotImplementedError

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

    def component_binding(self, model: AbstractArrayModel, /) -> ComponentBinding:
        """Bind `model` to this slot through `owner_ports()` and `port_mapping`."""
        return bind_positional_component(
            model,
            type(self),
            self.owner_ports(),
            self.port_mapping,
            site=type(self).__name__,
        )

    @property
    def supports_linear_refinement(self) -> bool:
        return False

    def _evidenced_result(
        self,
        model: AbstractArrayModel,
        candidate_state: Array,
        accepted_state: Array,
        /,
        *,
        training_usable: Any,
        physically_converged: Any,
        status: Any,
        residual: Any,
        iterations: Any,
        reason_bits: Array,
        derivative_contract: DerivativeContract,
        derivative_valid: Any,
        dependencies: Any,
    ) -> DiscreteModelRolloutTransitionResult:
        """Attach the common evidence and poison derivatives where invalid.

        The header's `model_id` is the model's evidence identity (its declared
        semantic provenance, else a marked contract fingerprint) and its
        `evidence_id` binds this transition to the model's contract bound to
        this slot. The model's parameters join `dependencies` so a rejected
        state that no longer depends on them is still poisoned.
        """
        component = self.component_binding(model).contract()
        model_id = component.model_contract.evidence_model_id
        converged = jnp.asarray(physically_converged, dtype=jnp.bool_)
        header = AdmissibilityHeader(
            jnp.where(converged, 1.0, -1.0),
            reason_bits,
            model_id,
            canonical_fingerprint(
                {
                    "kind": "discrete-model-rollout-transition-evidence",
                    "transition": self.transition_id,
                    "component": component.bound_semantic_id,
                    "derivative_contract": derivative_contract.contract_id,
                }
            ),
        )
        valid = jnp.asarray(derivative_valid, dtype=jnp.bool_)
        candidate_state, accepted_state = guard_derivative_validity(
            (candidate_state, accepted_state),
            valid,
            dependencies=(model, dependencies),
        )
        return DiscreteModelRolloutTransitionResult(
            candidate_state,
            accepted_state,
            training_usable=training_usable,
            physically_converged=converged,
            status=status,
            residual=residual,
            iterations=iterations,
            header=header,
            derivative_contract=derivative_contract,
            derivative_valid=valid,
            transition_id=self.transition_id,
        )

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


class DirectDiscreteModelRolloutTransition(
    AbstractDiscreteModelRolloutTransition, NonTrainableState
):
    """Current direct next-state model semantics as an explicit rollout route.

    The owner ports are those of `DiscreteModelTransition` with fixed steps:
    `state_layout.value_port(role="point")`, then `input_layout.value_port()`
    when an input layout is bound, as inputs, and the state point port as the
    output. A model declaring ports requires `port_mapping` binding its ordered
    ports to exactly that order.
    """

    port_mapping: PortMapping | None

    def __init__(
        self,
        state_layout: StateLayout,
        /,
        *,
        input_layout: InputLayout | None = None,
        step_size: float,
        step_rtol: float = 1e-7,
        step_atol: float = 1e-12,
        port_mapping: PortMapping | None = None,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        if port_mapping is not None and not isinstance(port_mapping, PortMapping):
            raise TypeError("port_mapping must be a PortMapping or None.")
        step = float(step_size)
        rtol = float(step_rtol)
        atol = float(step_atol)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if not np.isfinite(rtol) or rtol < 0.0 or not np.isfinite(atol) or atol < 0.0:
            raise ValueError("Step tolerances must be finite and nonnegative.")
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.port_mapping = port_mapping
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

    def owner_ports(self) -> ModelPorts:
        state = self.state_layout.value_port(role="point")
        inputs = (state,)
        if self.input_layout is not None:
            inputs = inputs + (self.input_layout.value_port(),)
        return ModelPorts(inputs=inputs, outputs=(state,))

    def validate_model(self, model: AbstractArrayModel, /) -> None:
        DiscreteModelTransition(
            model,
            state_layout=self.state_layout,
            input_layout=self.input_layout,
            step_size=self.step_size,
            step_rtol=self.step_rtol,
            step_atol=self.step_atol,
            port_mapping=self.port_mapping,
        )
        self.component_binding(model)

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
        return self._evidenced_result(
            model,
            candidate,
            accepted,
            training_usable=valid,
            physically_converged=valid,
            status=jnp.where(valid, 0, 1),
            residual=jnp.asarray(0.0, dtype=jnp.real(candidate).dtype),
            iterations=jnp.asarray(1, dtype=jnp.int32),
            reason_bits=reason_bits_where(finite, AdmissibilityReason.NONFINITE)
            | reason_bits_where(member, AdmissibilityReason.OUTSIDE_SUPPORT),
            derivative_contract=_DIRECT_DERIVATIVE_CONTRACT,
            derivative_valid=valid,
            dependencies=(state, inputs),
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
