#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...operators.quantum._observables import LocalObservable
from ...operators.quantum._parameterized import QuantumProgramTemplate
from ...solver._quantum_expectation import (
    DenseQuantumExpectationResult,
    DenseQuantumObservablePlan,
    DenseQuantumObservablePolicy,
    evaluate_dense_quantum_observables,
    plan_dense_quantum_observables,
)
from ...solver._quantum_gradients import (
    _parameter_shift_expectation_values,
    execute_dense_quantum_template,
    ParameterShiftPlan,
    plan_parameter_shift,
    prepare_dense_quantum_template,
    PreparedDenseQuantumTemplate,
)
from ...solver._quantum_program import (
    DenseQuantumProgramPolicy,
    DenseQuantumProgramResult,
)


CircuitGradientMethod: TypeAlias = Literal["autodiff", "parameter-shift"]


class _DenseCircuitExecution(StrictModule, NonTrainableState):
    prepared: PreparedDenseQuantumTemplate
    initial_state: Array
    execution_id: str = eqx.field(static=True)


class _DenseExpectationExecution(StrictModule, NonTrainableState):
    circuit: _DenseCircuitExecution
    observable_plan: DenseQuantumObservablePlan
    shift_plan: ParameterShiftPlan
    execution_id: str = eqx.field(static=True)


def _initial_state(
    template: QuantumProgramTemplate,
    initial_state: ArrayLike | None,
    /,
) -> Array:
    dimension = template.layout.dimension
    if initial_state is None:
        if template.state_kind == "state-vector":
            return jnp.zeros((dimension,), dtype=template.complex_dtype).at[0].set(1.0)
        return (
            jnp.zeros((dimension, dimension), dtype=template.complex_dtype)
            .at[0, 0]
            .set(1.0)
        )
    value = jnp.asarray(initial_state)
    expected_shape = (
        (dimension,) if template.state_kind == "state-vector" else (dimension, dimension)
    )
    if value.shape != expected_shape:
        raise ValueError("initial_state shape must match the template state kind.")
    if value.dtype != template.complex_dtype:
        raise TypeError("initial_state and template dtypes must match exactly.")
    return value


def _prepare_execution(
    template: QuantumProgramTemplate,
    initial_state: ArrayLike | None,
    policy: DenseQuantumProgramPolicy | None,
    /,
) -> _DenseCircuitExecution:
    prepared = prepare_dense_quantum_template(template, policy)
    state = _initial_state(template, initial_state)
    execution_id = canonical_fingerprint(
        {
            "kind": "dense-circuit-execution",
            "prepared_template": prepared.prepared_template_id,
            "initial_state_shape": state.shape,
            "initial_state_dtype": str(state.dtype),
        }
    )
    return _DenseCircuitExecution(prepared, state, execution_id)


def _validate_angle_model(
    angle_model: AbstractArrayModel,
    template: QuantumProgramTemplate,
    /,
) -> tuple[int, int]:
    if not isinstance(angle_model, AbstractArrayModel):
        raise TypeError("angle_model must be an AbstractArrayModel.")
    if not isinstance(angle_model.in_size, int) or not isinstance(
        angle_model.out_size, int
    ):
        raise ValueError("Circuit angle models require flat vector input and output.")
    if angle_model.out_size != template.angle_count:
        raise ValueError("angle_model.out_size must equal template.angle_count.")
    return angle_model.in_size, angle_model.out_size


class DenseCircuitStateModel(AbstractArrayModel):
    """Pointwise exact dense state feature map from one angle model and template."""

    angle_model: AbstractArrayModel
    execution: _DenseCircuitExecution
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        angle_model: AbstractArrayModel,
        template: QuantumProgramTemplate,
        /,
        *,
        initial_state: ArrayLike | None = None,
        policy: DenseQuantumProgramPolicy | None = None,
    ):
        if template.state_kind != "state-vector":
            raise ValueError("DenseCircuitStateModel requires a state-vector template.")
        in_size, _ = _validate_angle_model(angle_model, template)
        execution = _prepare_execution(template, initial_state, policy)
        self.angle_model = angle_model
        self.execution = execution
        self.in_size = in_size
        self.out_size = template.layout.dimension
        self.model_id = canonical_fingerprint(
            {
                "kind": "dense-circuit-state-model",
                "angle_model_type": (
                    f"{type(angle_model).__module__}.{type(angle_model).__qualname__}"
                ),
                "input_size": in_size,
                "execution": execution.execution_id,
            }
        )

    def evaluate(
        self,
        x: Any,
        /,
        *,
        key: Any = None,
    ) -> DenseQuantumProgramResult:
        angles = self.angle_model(x, key=key)
        return execute_dense_quantum_template(
            self.execution.prepared,
            angles,
            self.execution.initial_state,
        )

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        result = self.evaluate(x, key=key)
        return eqx.error_if(
            result.final_state,
            ~jnp.all(result.diagnostics.successful),
            "Dense quantum state feature execution was invalid.",
        )


class DenseCircuitExpectationModel(AbstractArrayModel):
    """Pointwise exact dense local-observable feature model."""

    angle_model: AbstractArrayModel
    execution: _DenseExpectationExecution
    gradient_method: CircuitGradientMethod = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        angle_model: AbstractArrayModel,
        template: QuantumProgramTemplate,
        observables: tuple[LocalObservable, ...] | list[LocalObservable],
        /,
        *,
        gradient_method: CircuitGradientMethod = "autodiff",
        initial_state: ArrayLike | None = None,
        program_policy: DenseQuantumProgramPolicy | None = None,
        observable_policy: DenseQuantumObservablePolicy | None = None,
    ):
        if gradient_method not in ("autodiff", "parameter-shift"):
            raise ValueError("Unknown circuit gradient method.")
        in_size, _ = _validate_angle_model(angle_model, template)
        circuit = _prepare_execution(template, initial_state, program_policy)
        observable_plan = plan_dense_quantum_observables(
            circuit.prepared.prepared_program,
            observables,
            observable_policy,
        )
        shift_plan = plan_parameter_shift(template)
        execution_id = canonical_fingerprint(
            {
                "kind": "dense-circuit-expectation-execution",
                "circuit": circuit.execution_id,
                "observables": observable_plan.plan_id,
                "shift": shift_plan.plan_id,
            }
        )
        self.angle_model = angle_model
        self.execution = _DenseExpectationExecution(
            circuit,
            observable_plan,
            shift_plan,
            execution_id,
        )
        self.gradient_method = gradient_method
        self.in_size = in_size
        self.out_size = observable_plan.cost.observable_count
        self.model_id = canonical_fingerprint(
            {
                "kind": "dense-circuit-expectation-model",
                "angle_model_type": (
                    f"{type(angle_model).__module__}.{type(angle_model).__qualname__}"
                ),
                "input_size": in_size,
                "execution": execution_id,
                "gradient_method": gradient_method,
            }
        )

    def evaluate(
        self,
        x: Any,
        /,
        *,
        key: Any = None,
    ) -> DenseQuantumExpectationResult:
        angles = self.angle_model(x, key=key)
        program_result = execute_dense_quantum_template(
            self.execution.circuit.prepared,
            angles,
            self.execution.circuit.initial_state,
        )
        return evaluate_dense_quantum_observables(
            self.execution.observable_plan,
            program_result,
        )

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        angles = self.angle_model(x, key=key)
        if self.gradient_method == "parameter-shift":
            return _parameter_shift_expectation_values(
                angles,
                self.execution.circuit.prepared,
                self.execution.observable_plan,
                self.execution.shift_plan,
                self.execution.circuit.initial_state,
            )
        program_result = execute_dense_quantum_template(
            self.execution.circuit.prepared,
            angles,
            self.execution.circuit.initial_state,
        )
        return evaluate_dense_quantum_observables(
            self.execution.observable_plan,
            program_result,
        ).real_values


class BinaryVariationalCircuitClassifier(AbstractArrayModel):
    """Binary probabilistic classifier over exact circuit expectation features."""

    feature_model: DenseCircuitExpectationModel
    weight: Array
    bias: Array
    negative_label: float = eqx.field(static=True)
    positive_label: float = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        feature_model: DenseCircuitExpectationModel,
        weight: ArrayLike,
        bias: ArrayLike,
        negative_label: float,
        positive_label: float,
        /,
    ):
        if not isinstance(feature_model, DenseCircuitExpectationModel):
            raise TypeError("feature_model must be DenseCircuitExpectationModel.")
        selected_weight = jnp.asarray(weight)
        selected_bias = jnp.asarray(bias)
        labels = (float(negative_label), float(positive_label))
        if selected_weight.shape != (feature_model.out_size,):
            raise ValueError("weight must have shape (feature_model.out_size,).")
        if selected_bias.shape != ():
            raise ValueError("bias must be scalar.")
        if not all(isfinite(label) for label in labels) or labels[0] == labels[1]:
            raise ValueError("Binary class labels must be distinct finite scalars.")
        if not jnp.issubdtype(selected_weight.dtype, jnp.floating):
            raise TypeError("Classifier weight must use real floating coordinates.")
        if selected_bias.dtype != selected_weight.dtype:
            raise TypeError("Classifier weight and bias dtypes must match exactly.")
        self.feature_model = feature_model
        self.weight = selected_weight
        self.bias = selected_bias
        self.negative_label = labels[0]
        self.positive_label = labels[1]
        self.in_size = feature_model.in_size
        self.out_size = "scalar"

    def _logit(self, x: Any, /, *, key: Any = None) -> Array:
        features = self.feature_model(x, key=key)
        return oe.contract("f,f->", self.weight, features) + self.bias

    def decision_function(self, x: Any, /) -> Array:
        return self._logit(x)

    def positive_probability(self, x: Any, /) -> Array:
        return jax.nn.sigmoid(self._logit(x))

    def predict_log_proba(self, x: Any, /) -> Array:
        logit = self._logit(x)
        return jnp.stack((jax.nn.log_sigmoid(-logit), jax.nn.log_sigmoid(logit)))

    def predict(self, x: Any, /) -> Array:
        return jnp.where(
            self._logit(x) >= 0.0,
            jnp.asarray(self.positive_label, dtype=self.weight.dtype),
            jnp.asarray(self.negative_label, dtype=self.weight.dtype),
        )

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        return jax.nn.sigmoid(self._logit(x, key=key))


__all__ = [
    "BinaryVariationalCircuitClassifier",
    "CircuitGradientMethod",
    "DenseCircuitExpectationModel",
    "DenseCircuitStateModel",
]
