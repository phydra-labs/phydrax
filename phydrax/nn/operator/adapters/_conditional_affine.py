#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._doc import DOC_KEY0
from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax.dynamics import (
    DiscreteStepContext,
    DiscreteSystem,
    InputLayout,
    StateLayout,
)
from phydrax.nn._keys import EvalKey
from phydrax.nn.operator.architectures import ChemicalConditionalAffineOperator
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch


class TrainedChemicalConditionalAffineTransition(StrictModule):
    trained_operator: Any
    state_layout: StateLayout
    input_layout: InputLayout
    template: OperatorBatch
    transition_id: str = eqx.field(static=True)
    minimum_duration: float | None = eqx.field(static=True)
    maximum_duration: float | None = eqx.field(static=True)

    def __init__(
        self,
        trained_operator: Any,
        /,
        *,
        state_layout: StateLayout,
        input_layout: InputLayout,
        minimum_duration: float | None = None,
        maximum_duration: float | None = None,
        transition_id: str | None = None,
    ):
        from phydrax.nn.operator.training._trained_operator import TrainedOperator

        if not isinstance(trained_operator, TrainedOperator):
            raise TypeError("trained_operator must be TrainedOperator.")
        model = trained_operator.execution_model
        if not isinstance(model, ChemicalConditionalAffineOperator):
            raise TypeError(
                "trained_operator must contain ChemicalConditionalAffineOperator."
            )
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be StateLayout.")
        if not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be InputLayout.")
        if state_layout.shape != (model.out_size,) or (
            state_layout.component_names != model.chemistry.mechanism.schema.species_names
        ):
            raise ValueError(
                "state_layout must exactly match the mechanism species ordering."
            )
        if input_layout.shape != (2,) or input_layout.component_names != (
            model.temperature_name,
            model.pressure_name,
        ):
            raise ValueError(
                "input_layout must contain temperature and pressure in model order."
            )
        if trained_operator.execution_plan.normalization is not None:
            raise ValueError(
                "Conditional-affine deployment requires no external operator normalization."
            )
        if trained_operator.execution_plan.output_pipeline is not None:
            raise ValueError(
                "Conditional-affine deployment requires no external output pipeline."
            )
        relevant_sources = {
            model.state_name,
            model.temperature_name,
            model.pressure_name,
        }
        for field in trained_operator.task.source_fields:
            assert field.source_name is not None
            if field.source_name in relevant_sources and (
                any(value != 1.0 for value in field.scale)
                or any(value != 0.0 for value in field.offset)
            ):
                raise ValueError(
                    "Conditional-affine task source fields require identity physical scaling."
                )
        for field in trained_operator.task.target_fields:
            if any(value != 1.0 for value in field.scale) or any(
                value != 0.0 for value in field.offset
            ):
                raise ValueError(
                    "Conditional-affine task target fields require identity physical scaling."
                )
        minimum = None if minimum_duration is None else float(minimum_duration)
        maximum = None if maximum_duration is None else float(maximum_duration)
        if minimum is not None and minimum <= 0.0:
            raise ValueError("minimum_duration must be positive or None.")
        if maximum is not None and maximum <= 0.0:
            raise ValueError("maximum_duration must be positive or None.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum_duration must not exceed maximum_duration.")
        dtype = model.scaling.state_scale.dtype
        template = OperatorBatch(
            inputs={
                model.state_name: FunctionSamples(
                    values=jnp.zeros((model.out_size,), dtype=dtype)
                ),
                model.temperature_name: FunctionSamples(
                    values=jnp.zeros((), dtype=dtype)
                ),
                model.pressure_name: FunctionSamples(values=jnp.ones((), dtype=dtype)),
            },
            queries={
                model.query_name: FunctionSamples(
                    values=None,
                    coordinates=jnp.zeros((1, 1), dtype=dtype),
                )
            },
        )
        generated = canonical_fingerprint(
            {
                "kind": "trained-chemical-conditional-affine-transition",
                "trained_operator": trained_operator.artifact_id,
                "certificate": model.chemistry.certificate.certificate_id,
                "state_layout": state_layout.layout_id,
                "input_layout": input_layout.layout_id,
                "minimum_duration": minimum,
                "maximum_duration": maximum,
            }
        )
        identifier = generated if transition_id is None else str(transition_id)
        if not identifier:
            raise ValueError("transition_id must be non-empty.")
        self.trained_operator = trained_operator
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.template = template
        self.transition_id = identifier
        self.minimum_duration = minimum
        self.maximum_duration = maximum

    @property
    def model(self) -> ChemicalConditionalAffineOperator:
        model = self.trained_operator.execution_model
        assert isinstance(model, ChemicalConditionalAffineOperator)
        return model

    def _batch(
        self,
        state: Array,
        duration: Array,
        inputs: Array,
        /,
    ) -> OperatorBatch:
        batch = eqx.tree_at(
            lambda value: value.inputs[self.model.state_name].values,
            self.template,
            state,
        )
        batch = eqx.tree_at(
            lambda value: value.inputs[self.model.temperature_name].values,
            batch,
            inputs[0],
        )
        batch = eqx.tree_at(
            lambda value: value.inputs[self.model.pressure_name].values,
            batch,
            inputs[1],
        )
        return eqx.tree_at(
            lambda value: value.queries[self.model.query_name].coordinates,
            batch,
            duration.reshape((1, 1)),
        )

    def evaluate_with_evidence(
        self,
        state: ArrayLike,
        duration: ArrayLike,
        inputs: ArrayLike,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ):
        state_ = jnp.asarray(state)
        duration_ = jnp.asarray(duration, dtype=state_.dtype)
        inputs_ = jnp.asarray(inputs, dtype=state_.dtype)
        if state_.shape != self.state_layout.shape:
            raise ValueError("state shape does not match state_layout.")
        if duration_.shape != ():
            raise ValueError("duration must be scalar.")
        if inputs_.shape != self.input_layout.shape:
            raise ValueError("inputs shape does not match input_layout.")
        batch = self._batch(state_, duration_, inputs_)
        prepared = self.trained_operator.execution_plan.prepare_prevalidated(batch)
        return self.model.evaluate_transition(prepared.execution_batch, key=key)

    def __call__(
        self,
        context: DiscreteStepContext,
        state: Array,
        inputs: Array,
        args: Any = None,
        /,
    ) -> Array:
        del args
        result = self.evaluate_with_evidence(
            state,
            context.duration,
            inputs,
            key=DOC_KEY0,
        )
        candidate = result.candidate_state[0]
        return eqx.error_if(
            candidate,
            ~result.successful[0],
            "Trained chemical conditional-affine transition failed.",
        )

    def discrete_system(
        self,
        /,
        *,
        step_size: float | None = None,
    ) -> DiscreteSystem:
        return DiscreteSystem(
            self,
            state_layout=self.state_layout,
            input_layout=self.input_layout,
            system_id=self.transition_id,
            step_size=step_size,
            minimum_step_size=self.minimum_duration,
            maximum_step_size=self.maximum_duration,
        )


__all__ = ["TrainedChemicalConditionalAffineTransition"]
